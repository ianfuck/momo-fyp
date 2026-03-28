import { useEffect, useState } from "react";
import { CameraPanel } from "../components/CameraPanel";
import { ConfigPanel } from "../components/ConfigPanel";
import { EventLog } from "../components/EventLog";
import { PipelineStage } from "../components/PipelineStage";
import { PromptPanel } from "../components/PromptPanel";
import { RuntimeOverview } from "../components/RuntimeOverview";
import { StatusTable } from "../components/StatusTable";
import { SystemStats } from "../components/SystemStats";
import { fetchAudioDevices, fetchCameras, fetchConfig, fetchOllamaModels, fetchStatus, simulatePipeline, updateConfig } from "../lib/api";
import type { AudioDevice, CameraInfo, ConfigApplyCheck, ConfigField, ConfigUpdateResponse, StatusSnapshot } from "../lib/types";

const initialStatus: StatusSnapshot = {
  ts: new Date().toISOString(),
  mode: "IDLE",
  locked_track_id: null,
  sentence_index: 0,
  active_sentence_index: 0,
  audience: {
    person_bbox: null,
    top_color: "unknown",
    bottom_color: "unknown",
    height_class: "unknown",
    build_class: "unknown",
    distance_class: "unknown",
    face_bbox: null,
    left_eye_bbox: null,
    right_eye_bbox: null,
    bbox_area_ratio: 0,
    eye_midpoint: null,
    eye_confidence: 0,
    pose_keypoints: {},
    left_wrist_point: null,
    right_wrist_point: null,
    pose_confidence: 0,
    actions: {
      wave: false,
      crouch: false,
      defocus: false,
      moving_away: false,
      approaching: false,
      returned_after_defocus: false,
    },
  },
  pipeline: { stage: "IDLE", elapsed_ms: 0, last_error: null },
  servo: { left_deg: 90, right_deg: 90, tracking_source: "none" },
  stats: { memory_rss_mb: 0, memory_vms_mb: 0, temp_file_count: 0, temp_file_size_mb: 0 },
  serial_connected: false,
  ollama_connected: false,
  tts_loaded: false,
  playback_progress: 0,
  current_prompt_system: null,
  current_prompt_user: null,
  last_llm_output: null,
  last_spoken_text: null,
  event_log: [],
};

export function Dashboard() {
  const [status, setStatus] = useState<StatusSnapshot>(initialStatus);
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [fields, setFields] = useState<ConfigField[]>([]);
  const [cameras, setCameras] = useState<CameraInfo[]>([]);
  const [audioDevices, setAudioDevices] = useState<AudioDevice[]>([]);
  const [models, setModels] = useState<string[]>([]);
  const [updateState, setUpdateState] = useState<{
    saving: boolean;
    message: string | null;
    kind: "idle" | "success" | "error";
    validationErrors: string[];
    applyChecks: ConfigApplyCheck[];
    effectiveChanges: string[];
  }>({
    saving: false,
    message: null,
    kind: "idle",
    validationErrors: [],
    applyChecks: [],
    effectiveChanges: [],
  });

  useEffect(() => {
    void (async () => {
      const [statusData, configData, cameraData, modelData, audioData] = await Promise.all([
        fetchStatus(),
        fetchConfig(),
        fetchCameras(),
        fetchOllamaModels(),
        fetchAudioDevices(),
      ]);
      const browserDevices = await getBrowserCameras();
      setStatus(statusData);
      setConfig(configData.config);
      setFields(configData.fields);
      setCameras(browserDevices.length > 0 ? browserDevices : cameraData);
      setAudioDevices(audioData);
      setModels(modelData);
    })();
    const interval = window.setInterval(() => {
      void Promise.all([fetchStatus(), fetchConfig()])
        .then(([statusData, configData]) => {
          setStatus(statusData);
          setConfig(configData.config);
          setFields(configData.fields);
        })
        .catch(() => undefined);
    }, 1000);
    return () => window.clearInterval(interval);
  }, []);

  return (
    <main className="dashboard">
      <header className="hero">
        <div>
          <p className="eyebrow">Momo MVP</p>
          <h1>Audience Tracking Console</h1>
        </div>
        <button className="primary" onClick={() => void simulatePipeline().then((data) => setStatus(data.snapshot))}>
          Simulate Pipeline
        </button>
      </header>
      <div className="grid">
        <CameraPanel
          status={status}
          cameraDeviceId={String(config.camera_device_id ?? "default")}
          width={Number(config.camera_width ?? 640)}
          height={Number(config.camera_height ?? 480)}
          fps={Number(config.camera_fps ?? 10)}
          mirror={Boolean(config.camera_mirror_preview ?? false)}
        />
        <RuntimeOverview status={status} config={config} applyChecks={updateState.applyChecks} />
        <PipelineStage active={status.pipeline.stage} elapsedMs={status.pipeline.elapsed_ms} />
        <StatusTable status={status} config={config} />
        <PromptPanel status={status} />
        <SystemStats status={status} />
        <EventLog items={status.event_log} />
        <ConfigPanel
          fields={fields}
          config={config}
          cameras={cameras}
          audioDevices={audioDevices}
          models={models}
          updateState={updateState}
          onSubmit={async (payload) => {
            setUpdateState({
              saving: true,
              message: "Applying runtime config...",
              kind: "idle",
              validationErrors: [],
              applyChecks: [],
              effectiveChanges: [],
            });
            let result: ConfigUpdateResponse;
            try {
              result = await updateConfig(payload);
            } catch (error) {
              const message = error instanceof Error ? error.message : "Failed to update config";
              setUpdateState({
                saving: false,
                message,
                kind: "error",
                validationErrors: [message],
                applyChecks: [],
                effectiveChanges: [],
              });
              throw error;
            }
            setConfig(result.applied_config);
            const refreshed = await fetchConfig();
            setFields(refreshed.fields);
            setUpdateState({
              saving: false,
              message:
                result.validation_errors.length > 0
                  ? "Config rejected."
                  : result.effective_changes.length > 0
                    ? "Config applied."
                    : "No runtime changes detected.",
              kind: result.validation_errors.length > 0 || result.apply_checks.some((item) => item.status === "error") ? "error" : "success",
              validationErrors: result.validation_errors,
              applyChecks: result.apply_checks,
              effectiveChanges: result.effective_changes,
            });
            return result;
          }}
        />
      </div>
    </main>
  );
}

async function getBrowserCameras(): Promise<CameraInfo[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((device) => device.kind === "videoinput")
      .map((device, index) => ({
      device_id: device.deviceId || `browser-${index}`,
      device_name: device.label || `Browser Camera ${index + 1}`,
      modes: [
        { width: 640, height: 480, fps: 10 },
        { width: 640, height: 480, fps: 15 },
        { width: 640, height: 480, fps: 30 },
        { width: 1280, height: 720, fps: 10 },
        { width: 640, height: 480, fps: 30 },
        { width: 1280, height: 720, fps: 15 },
        { width: 1280, height: 720, fps: 30 },
        { width: 1920, height: 1080, fps: 10 },
        { width: 1920, height: 1080, fps: 30 },
      ],
    }));
}
