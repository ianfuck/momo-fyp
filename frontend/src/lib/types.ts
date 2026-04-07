export type PipelineStage = "IDLE" | "LLM" | "TTS" | "PLAYBACK" | "ERROR";

export type ConfigField = {
  key: string;
  label: string;
  description: string;
  type: string;
  default: unknown;
  value: unknown;
  valid_range?: string | null;
  enum?: string[] | null;
  applies_to: string;
  requires_restart: boolean;
};

export type StatusSnapshot = {
  ts: string;
  mode: string;
  locked_track_id: number | null;
  sentence_index: number;
  active_sentence_index: number;
  audience: {
    track_id?: number | null;
    person_bbox?: [number, number, number, number] | null;
    top_color: string;
    bottom_color: string;
    height_class: string;
    build_class: string;
    distance_class: string;
    face_bbox?: [number, number, number, number] | null;
    left_eye_bbox?: [number, number, number, number] | null;
    right_eye_bbox?: [number, number, number, number] | null;
    bbox_area_ratio: number;
    eye_midpoint?: [number, number] | null;
    eye_confidence: number;
    pose_keypoints: Record<string, [number, number] | null>;
    left_wrist_point?: [number, number] | null;
    right_wrist_point?: [number, number] | null;
    pose_confidence: number;
    actions: {
      wave: boolean;
      crouch: boolean;
      defocus: boolean;
      moving_away: boolean;
      approaching: boolean;
      returned_after_defocus: boolean;
    };
  };
  pipeline: {
    stage: PipelineStage;
    elapsed_ms: number;
    last_error?: string | null;
  };
  servo: {
    left_deg: number;
    right_deg: number;
    tracking_source: string;
  };
  serial_monitor: {
    port?: string | null;
    baud_rate?: number | null;
    last_tx?: string | null;
    last_tx_at?: string | null;
    last_rx?: string | null;
    last_rx_at?: string | null;
    last_error?: string | null;
    last_error_at?: string | null;
    entries: Array<{
      ts: string;
      direction: string;
      message: string;
    }>;
  };
  stats: {
    memory_rss_mb: number;
    memory_vms_mb: number;
    gpu_memory_mb?: number | null;
    temp_file_count: number;
    temp_file_size_mb: number;
  };
  yolo_person_runtime: RuntimeComponentStats;
  yolo_pose_runtime: RuntimeComponentStats;
  tts_runtime: RuntimeComponentStats;
  ollama_runtime: RuntimeComponentStats;
  camera_device_id?: string | null;
  camera_mode?: string | null;
  serial_connected: boolean;
  ollama_connected: boolean;
  tts_loaded: boolean;
  llm_latency_ms?: number | null;
  tts_latency_ms?: number | null;
  playback_progress: number;
  tts_emotion_raw?: string | null;
  tts_emotion_applied?: string | null;
  tts_emotion_used: boolean;
  tts_input_text?: string | null;
  tts_reference_raw?: string | null;
  tts_reference_pair?: string | null;
  tts_reference_audio_path?: string | null;
  tts_reference_text_path?: string | null;
  current_prompt_system?: string | null;
  current_prompt_user?: string | null;
  last_llm_output?: string | null;
  last_spoken_text?: string | null;
  event_log: string[];
};

export type RuntimeComponentStats = {
  requested_mode?: string | null;
  effective_device?: string | null;
  backend?: string | null;
  selection_source?: string | null;
  semantic_dispatch_mode?: string | null;
  ram_mb?: number | null;
  vram_mb?: number | null;
};

export type CameraMode = { width: number; height: number; fps: number };
export type CameraInfo = { device_id: string; device_name: string; modes: CameraMode[] };
export type AudioDevice = { id: string; name: string };
export type ConfigApplyCheck = { component: string; status: "ok" | "warning" | "error"; message: string };
export type ConfigUpdateResponse = {
  applied_config: Record<string, unknown>;
  validation_errors: string[];
  effective_changes: string[];
  apply_checks: ConfigApplyCheck[];
  requires_pipeline_restart: boolean;
};
