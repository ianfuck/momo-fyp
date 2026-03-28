import type { StatusSnapshot } from "../lib/types";

export function StatusTable({
  status,
  config,
}: {
  status: StatusSnapshot;
  config: Record<string, unknown>;
}) {
  const actions = status.audience.actions;
  const activeEvents = Object.entries(actions).filter(([, v]) => v).map(([k]) => k).join(", ") || "none";
  return (
    <section className="panel">
      <div className="panel-header">
        <h3>Live Status</h3>
      </div>
      <table className="status-table">
        <tbody>
          <tr><th>Mode</th><td>{status.mode}</td></tr>
          <tr><th>Locked Track</th><td>{status.locked_track_id ?? "-"}</td></tr>
          <tr><th>Current Sentence</th><td>{status.active_sentence_index || status.sentence_index}</td></tr>
          <tr><th>Completed Sentence</th><td>{status.sentence_index}</td></tr>
          <tr><th>Audience</th><td>{status.audience.top_color} / {status.audience.height_class} / {status.audience.build_class}</td></tr>
          <tr><th>Distance</th><td>{status.audience.distance_class}</td></tr>
          <tr><th>Area Ratio</th><td>{status.audience.bbox_area_ratio.toFixed(3)}</td></tr>
          <tr><th>Eye Confidence</th><td>{status.audience.eye_confidence.toFixed(2)}</td></tr>
          <tr><th>Pose Confidence</th><td>{status.audience.pose_confidence.toFixed(2)}</td></tr>
          <tr><th>Events</th><td>{activeEvents}</td></tr>
          <tr><th>Servo</th><td>L {status.servo.left_deg} / R {status.servo.right_deg}</td></tr>
          <tr><th>Tracking Source</th><td>{status.servo.tracking_source}</td></tr>
          <tr><th>Camera Mode</th><td>{status.camera_mode ?? "-"}</td></tr>
          <tr><th>Playback</th><td>{Math.round(status.playback_progress * 100)}%</td></tr>
          <tr><th>Pipeline Elapsed</th><td>{status.pipeline.elapsed_ms} ms</td></tr>
          <tr><th>LLM Timeout</th><td>{Number(config.ollama_timeout_sec ?? 0) * 1000 || "-"} ms</td></tr>
          <tr><th>TTS Timeout</th><td>{Number(config.tts_timeout_sec ?? 0) * 1000 || "-"} ms</td></tr>
          <tr><th>LLM Latency</th><td>{status.llm_latency_ms ?? "-"} ms</td></tr>
          <tr><th>TTS Latency</th><td>{status.tts_latency_ms ?? "-"} ms</td></tr>
          <tr><th>LLM Output</th><td>{status.last_llm_output ?? "-"}</td></tr>
          <tr><th>Spoken Text</th><td>{status.last_spoken_text ?? "-"}</td></tr>
          <tr><th>Pipeline Error</th><td>{status.pipeline.last_error ?? "-"}</td></tr>
        </tbody>
      </table>
    </section>
  );
}
