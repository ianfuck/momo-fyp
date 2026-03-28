import type { ConfigApplyCheck, StatusSnapshot } from "../lib/types";

export function RuntimeOverview({
  status,
  config,
  applyChecks,
}: {
  status: StatusSnapshot;
  config: Record<string, unknown>;
  applyChecks: ConfigApplyCheck[];
}) {
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Runtime Overview</h3>
        <span>{status.ts}</span>
      </div>
      <div className="overview-grid">
        <RuntimeItem label="LLM Model" value={String(config.ollama_model ?? "-")} />
        <RuntimeItem label="LLM Prompt Mode" value={Boolean(config.llm_use_person_crop) ? "Person Crop + Image" : "Prompt Only"} />
        <RuntimeItem label="Ollama URL" value={String(config.ollama_base_url ?? "-")} />
        <RuntimeItem label="Camera Source" value={String(config.camera_source ?? "-")} />
        <RuntimeItem label="Camera Device" value={String(config.camera_device_id ?? "-")} />
        <RuntimeItem label="Applied Camera Mode" value={status.camera_mode ?? "-"} />
        <RuntimeItem label="Audio Output" value={String(config.audio_output_device ?? "-")} />
        <RuntimeItem label="TTS Timeout" value={`${String(config.tts_timeout_sec ?? "-")} s`} />
        <RuntimeItem label="Serial Port" value={String(config.serial_port ?? "-")} />
        <RuntimeItem label="Lock Threshold" value={String(config.lock_bbox_threshold_ratio ?? "-")} />
        <RuntimeItem label="Defocus Threshold" value={String(config.defocus_bbox_threshold_ratio ?? "-")} />
        <RuntimeItem label="Tracking Examples" value={String((config.tracking_examples_selected as unknown[] | undefined)?.length ?? 0)} />
        <RuntimeItem label="Idle Examples" value={String((config.idle_examples_selected as unknown[] | undefined)?.length ?? 0)} />
        <RuntimeItem label="TTS Loaded" value={String(status.tts_loaded)} />
      </div>
      <div className="health-strip">
        <HealthChip label="Vision" status={status.audience.track_id !== null ? "ok" : "idle"} />
        <HealthChip label="LLM" status={status.ollama_connected ? "ok" : "error"} />
        <HealthChip label="TTS" status={status.tts_loaded ? "ok" : "idle"} />
        <HealthChip label="Serial" status={status.serial_connected ? "ok" : "warning"} />
        <HealthChip label="Audio" status={status.pipeline.last_error ? "warning" : "ok"} />
      </div>
      {applyChecks.length > 0 ? (
        <div className="apply-checks compact">
          {applyChecks.map((check) => (
            <div key={`${check.component}-${check.message}`} className={`apply-check ${check.status}`}>
              <strong>{check.component}</strong>
              <span>{check.message}</span>
            </div>
          ))}
        </div>
      ) : null}
    </section>
  );
}

function RuntimeItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="runtime-item">
      <strong>{label}</strong>
      <span>{value}</span>
    </div>
  );
}

function HealthChip({ label, status }: { label: string; status: "ok" | "warning" | "error" | "idle" }) {
  return <span className={`health-chip ${status}`}>{label}: {status}</span>;
}
