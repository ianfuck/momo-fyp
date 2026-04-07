import type { RuntimeComponentStats, StatusSnapshot } from "../lib/types";

export function DeviceStatusBar({ status }: { status: StatusSnapshot }) {
  const yoloRam = sumMetric(status.yolo_person_runtime.ram_mb, status.yolo_pose_runtime.ram_mb);
  const yoloVram = sumMetric(status.yolo_person_runtime.vram_mb, status.yolo_pose_runtime.vram_mb);
  const semanticLabel = formatSemanticDispatch(status.tts_runtime);
  return (
    <section className="device-strip">
      <DeviceCard
        label="YOLO"
        value={`${status.yolo_person_runtime.backend ?? "-"} / ${status.yolo_pose_runtime.backend ?? "-"}`}
        detail={`person ${status.yolo_person_runtime.effective_device ?? "-"} | pose ${status.yolo_pose_runtime.effective_device ?? "-"}`}
        ramMb={yoloRam}
        vramMb={yoloVram}
      />
      <DeviceCard
        label="TTS"
        value={status.tts_runtime.backend ?? "-"}
        detail={[
          `main ${status.tts_runtime.effective_device ?? "-"}`,
          formatSelectionSource(status.tts_runtime.selection_source),
          semanticLabel,
        ].join(" | ")}
        ramMb={status.tts_runtime.ram_mb}
        vramMb={status.tts_runtime.vram_mb}
      />
      <DeviceCard
        label="Ollama"
        value={status.ollama_runtime.backend ?? "-"}
        detail={[
          status.ollama_runtime.effective_device ?? "-",
          formatSelectionSource(status.ollama_runtime.selection_source),
        ].join(" | ")}
        ramMb={status.ollama_runtime.ram_mb}
        vramMb={status.ollama_runtime.vram_mb}
      />
    </section>
  );
}

function DeviceCard({
  label,
  value,
  detail,
  ramMb,
  vramMb,
}: {
  label: string;
  value: string;
  detail: string;
  ramMb?: number | null;
  vramMb?: number | null;
}) {
  return (
    <div className="device-card">
      <div className="device-card-header">
        <strong>{label}</strong>
        <span>{value}</span>
      </div>
      <div className="device-card-detail">{detail}</div>
      <div className="device-card-metrics">
        <span>RAM {formatMb(ramMb)}</span>
        <span>VRAM {formatMb(vramMb)}</span>
      </div>
    </div>
  );
}

function formatMb(value?: number | null): string {
  if (value === null || value === undefined) return "-";
  return `${value.toFixed(2)} MB`;
}

function sumMetric(...values: Array<number | null | undefined>): number | null {
  const present = values.filter((value): value is number => value !== null && value !== undefined);
  if (present.length === 0) return null;
  return present.reduce((total, value) => total + value, 0);
}

function formatSelectionSource(source?: string | null): string {
  if (source === "benchmark") return "auto benchmark winner";
  if (source === "user") return "chosen in UI";
  if (source === "default") return "default setting";
  return "selection unknown";
}

function formatSemanticDispatch(runtime: RuntimeComponentStats): string {
  if (runtime.semantic_dispatch_mode === "auto") {
    return "semantic split across CPU + GPU/MPS";
  }
  if (runtime.semantic_dispatch_mode === "single") {
    return "semantic stays on one device";
  }
  return "semantic dispatch unknown";
}
