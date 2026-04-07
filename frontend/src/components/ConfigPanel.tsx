import { useEffect, useMemo, useState } from "react";
import type { AudioDevice, CameraInfo, ConfigApplyCheck, ConfigField, ConfigUpdateResponse } from "../lib/types";

type Props = {
  fields: ConfigField[];
  config: Record<string, unknown>;
  cameras: CameraInfo[];
  audioDevices: AudioDevice[];
  models: string[];
  onSubmit: (payload: Record<string, unknown>) => Promise<ConfigUpdateResponse>;
  updateState: {
    saving: boolean;
    message: string | null;
    kind: "idle" | "success" | "error";
    validationErrors: string[];
    applyChecks: ConfigApplyCheck[];
    effectiveChanges: string[];
  };
};

export function ConfigPanel({ fields, config, cameras, audioDevices, models, onSubmit, updateState }: Props) {
  const [draft, setDraft] = useState<Record<string, unknown>>(config);
  const [hasLocalEdits, setHasLocalEdits] = useState(false);
  const selectedCamera = cameras.find(
    (camera) => camera.device_id === String(draft.camera_device_id ?? config.camera_device_id ?? "default"),
  );
  const selectedWidth = Number(draft.camera_width ?? config.camera_width ?? 640);
  const selectedHeight = Number(draft.camera_height ?? config.camera_height ?? 480);
  const selectedFps = Number(draft.camera_fps ?? config.camera_fps ?? 10);
  const uniqueSizes = Array.from(
    new Map((selectedCamera?.modes ?? []).map((mode) => [`${mode.width}x${mode.height}`, { width: mode.width, height: mode.height }])).values(),
  );
  const fpsOptions = Array.from(
    new Set(
      (selectedCamera?.modes ?? [])
        .filter((mode) => mode.width === selectedWidth && mode.height === selectedHeight)
        .map((mode) => mode.fps),
    ),
  ).sort((a, b) => a - b);

  const grouped = useMemo(() => {
    return fields.reduce<Record<string, ConfigField[]>>((acc, field) => {
      acc[field.applies_to] ??= [];
      acc[field.applies_to].push(field);
      return acc;
    }, {});
  }, [fields]);

  useEffect(() => {
    if (!hasLocalEdits) {
      setDraft(config);
    }
  }, [config, hasLocalEdits]);

  const hasChanges = JSON.stringify(draft) !== JSON.stringify(config);

  return (
    <section className="panel config-panel">
      <div className="panel-header">
        <h3>Runtime Config</h3>
        {updateState.saving ? <span className="save-pill saving">Saving...</span> : null}
      </div>
      {updateState.message ? (
        <div className={`update-feedback ${updateState.kind}`}>
          <strong>{updateState.message}</strong>
          {updateState.effectiveChanges.length > 0 ? (
            <div>Applied: {updateState.effectiveChanges.join(", ")}</div>
          ) : null}
        </div>
      ) : null}
      {updateState.validationErrors.length > 0 ? (
        <div className="update-feedback error">
          {updateState.validationErrors.map((item) => (
            <div key={item}>{item}</div>
          ))}
        </div>
      ) : null}
      {updateState.applyChecks.length > 0 ? (
        <div className="apply-checks">
          {updateState.applyChecks.map((check) => (
            <div key={`${check.component}-${check.message}`} className={`apply-check ${check.status}`}>
              <strong>{check.component}</strong>
              <span>{check.message}</span>
            </div>
          ))}
        </div>
      ) : null}
      {Object.entries(grouped).map(([group, groupFields]) => (
        <div key={group} className="config-group">
          <h4>{group}</h4>
          {groupFields.map((field) => {
            const value = draft[field.key] ?? field.value;
            const appliedValue = config[field.key] ?? field.value;
            const isDirty = JSON.stringify(value) !== JSON.stringify(appliedValue);
            if (field.key === "camera_device_id") {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <select value={String(value)} onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.value)}>
                    {cameras.map((camera) => <option key={camera.device_id} value={camera.device_id}>{camera.device_name}</option>)}
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description}</small>
                </label>
              );
            }
            if (field.key === "camera_source") {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <select value={String(value)} onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.value)}>
                    <option value="browser">Browser</option>
                    <option value="backend">Backend OpenCV</option>
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description}</small>
                </label>
              );
            }
            if (field.key === "camera_width") {
              return (
                <label key={field.key} className="field">
                  <span>Resolution</span>
                  <select
                    value={`${selectedWidth}x${selectedHeight}`}
                    onChange={(e) => {
                      const size = e.target.value;
                      const [width, height] = size.split("x");
                      const matched = (selectedCamera?.modes ?? []).find(
                        (mode) => mode.width === Number.parseInt(width, 10) && mode.height === Number.parseInt(height, 10),
                      );
                      setHasLocalEdits(true);
                      setDraft({
                        ...draft,
                        camera_width: Number.parseInt(width, 10),
                        camera_height: Number.parseInt(height, 10),
                        camera_fps: matched?.fps ?? selectedFps,
                      });
                    }}
                  >
                    {uniqueSizes.map((mode) => {
                      const modeValue = `${mode.width}x${mode.height}`;
                      return (
                        <option key={modeValue} value={modeValue}>
                          {modeValue}
                        </option>
                      );
                    })}
                  </select>
                  <FieldMeta
                    appliedValue={`${config.camera_width ?? field.value}x${config.camera_height ?? 480}`}
                    pendingValue={`${selectedWidth}x${selectedHeight}`}
                    isDirty={
                      Number(config.camera_width ?? field.value) !== selectedWidth ||
                      Number(config.camera_height ?? 480) !== selectedHeight
                    }
                  />
                  <small>Choose camera resolution.</small>
                </label>
              );
            }
            if (field.key === "camera_height") {
              return null;
            }
            if (field.key === "camera_fps") {
              return (
                <label key={field.key} className="field">
                  <span>FPS</span>
                  <select
                    value={String(selectedFps)}
                    onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, "camera_fps", Number.parseInt(e.target.value, 10))}
                  >
                    {(fpsOptions.length > 0 ? fpsOptions : [5, 10, 15, 24, 30]).map((fps) => (
                      <option key={fps} value={fps}>
                        {fps}
                      </option>
                    ))}
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>Choose camera refresh rate. Lower values reduce CPU and UI lag.</small>
                </label>
              );
            }
            if (field.key === "ollama_model") {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <select value={String(value)} onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.value)}>
                    {models.map((model) => <option key={model} value={model}>{model}</option>)}
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description}</small>
                </label>
              );
            }
            if (field.key === "audio_output_device") {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <select value={String(value)} onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.value)}>
                    <option value="default">System Default</option>
                    {audioDevices.map((device) => (
                      <option key={device.id} value={device.id}>
                        {device.name}
                      </option>
                    ))}
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description}</small>
                </label>
              );
            }
            if (field.type === "boolean") {
              return (
                <label key={field.key} className="field checkbox-field">
                  <span>{field.label}</span>
                  <input
                    type="checkbox"
                    checked={Boolean(value)}
                    onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.checked)}
                  />
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description} Default: {String(field.default)}</small>
                </label>
              );
            }
            if (field.enum && field.enum.length > 0) {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <select value={String(value)} onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, e.target.value)}>
                    {field.enum.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description} Default: {String(field.default)}</small>
                </label>
              );
            }
            if (field.type === "string[]") {
              return (
                <label key={field.key} className="field">
                  <span>{field.label}</span>
                  <textarea
                    rows={3}
                    value={Array.isArray(value) ? value.join("\n") : ""}
                    onChange={(e) => {
                      setHasLocalEdits(true);
                      setDraft({
                        ...draft,
                        [field.key]: e.target.value
                          .split("\n")
                          .map((item) => item.trim())
                          .filter(Boolean),
                      });
                    }}
                  />
                  <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                  <small>{field.description} One path per line.</small>
                </label>
              );
            }
            return (
              <label key={field.key} className="field">
                <span>{field.label}</span>
                <input
                  value={String(value)}
                  onChange={(e) => updateDraft(setDraft, setHasLocalEdits, draft, field.key, parseInput(field.type, e.target.value))}
                />
                <FieldMeta appliedValue={appliedValue} pendingValue={value} isDirty={isDirty} />
                <small>{field.description} Default: {String(field.default)}{field.valid_range ? ` | Range: ${field.valid_range}` : ""}</small>
              </label>
            );
          })}
        </div>
      ))}
      <div className="camera-modes">
        <h4>Camera Modes</h4>
        {cameras.map((camera) => (
          <div key={camera.device_id}>
            <strong>{camera.device_name}</strong>
            <span>{camera.modes.map((mode) => `${mode.width}x${mode.height}@${mode.fps}`).join(", ")}</span>
          </div>
        ))}
      </div>
      <button
        className="primary"
        disabled={updateState.saving}
        onClick={() => {
          void onSubmit(draft).then((result) => {
            if (result.validation_errors.length === 0) {
              setHasLocalEdits(false);
            }
          });
        }}
      >
        {updateState.saving ? "Updating..." : hasChanges ? "Update" : "Re-apply / Verify"}
      </button>
    </section>
  );
}

function FieldMeta({
  appliedValue,
  pendingValue,
  isDirty,
}: {
  appliedValue: unknown;
  pendingValue: unknown;
  isDirty: boolean;
}) {
  return (
    <div className="field-meta">
      <span className="field-chip applied">Using: {formatValue(appliedValue)}</span>
      {isDirty ? <span className="field-chip pending">Pending: {formatValue(pendingValue)}</span> : null}
    </div>
  );
}

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.join(" | ");
  if (typeof value === "boolean") return value ? "true" : "false";
  if (value === null || value === undefined || value === "") return "(empty)";
  return String(value);
}

function updateDraft(
  setDraft: (value: Record<string, unknown>) => void,
  setHasLocalEdits: (value: boolean) => void,
  current: Record<string, unknown>,
  key: string,
  value: unknown,
) {
  setHasLocalEdits(true);
  setDraft({ ...current, [key]: value });
}

function parseInput(type: string, value: string): unknown {
  if (type === "int") return Number.parseInt(value, 10);
  if (type === "float") return Number.parseFloat(value);
  if (type === "boolean") return value === "true";
  return value;
}
