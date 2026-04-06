import { useEffect, useRef, useState } from "react";
import { uploadCameraFrame } from "../lib/api";
import type { StatusSnapshot } from "../lib/types";

type Props = {
  status: StatusSnapshot;
  cameraDeviceId?: string;
  width?: number;
  height?: number;
  fps?: number;
  mirror?: boolean;
};

export function CameraPanel({ status, cameraDeviceId, width = 1280, height = 720, fps = 30, mirror = false }: Props) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const uploadInFlightRef = useRef(false);
  const [permission, setPermission] = useState<"idle" | "granted" | "denied">("idle");
  const [streamMode, setStreamMode] = useState<string>("pending");
  const [lastUploadAt, setLastUploadAt] = useState<number | null>(null);

  useEffect(() => {
    let stream: MediaStream | null = null;
    let timer = 0;
    async function boot() {
      try {
        const browserDeviceId = await resolveBrowserCameraDeviceId(cameraDeviceId);
        stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            deviceId: browserDeviceId ? { exact: browserDeviceId } : undefined,
            width: { ideal: width },
            height: { ideal: height },
            frameRate: { ideal: fps },
          },
        });
        setPermission("granted");
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
          const track = stream.getVideoTracks()[0];
          const settings = track?.getSettings();
          if (settings) {
            setStreamMode(`${settings.width ?? "?"}x${settings.height ?? "?"}@${Math.round(settings.frameRate ?? 0) || "?"}`);
          }
        }
        timer = window.setInterval(() => {
          const video = videoRef.current;
          const canvas = canvasRef.current;
          if (!video || !canvas || video.readyState < 2 || uploadInFlightRef.current) return;
          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;
          if (mirror) {
            ctx.save();
            ctx.scale(-1, 1);
            ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
            ctx.restore();
          } else {
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
          }
          uploadInFlightRef.current = true;
          canvas.toBlob((blob) => {
            if (!blob) {
              uploadInFlightRef.current = false;
              return;
            }
            void uploadCameraFrame(blob)
              .catch(() => undefined)
              .finally(() => {
                setLastUploadAt(Date.now());
                uploadInFlightRef.current = false;
              });
          }, "image/jpeg", 0.55);
        }, Math.max(160, Math.round(1000 / Math.max(1, fps))));
      } catch {
        setPermission("denied");
      }
    }
    void boot();
    return () => {
      window.clearInterval(timer);
      stream?.getTracks().forEach((track) => track.stop());
    };
  }, [cameraDeviceId, fps, width, height, mirror]);

  return (
    <section className="panel camera-panel">
      <div className="panel-header">
        <h3>Camera Preview</h3>
        <span>{status.camera_mode ?? "n/a"}</span>
      </div>
      <div className="camera-stage">
        <video ref={videoRef} className="camera-video" playsInline muted />
        <canvas ref={canvasRef} className="camera-hidden" />
        <div className="camera-stream">
          <SkeletonOverlay points={status.audience.pose_keypoints} />
          {status.audience.person_bbox && <Box bbox={status.audience.person_bbox} width={width} height={height} label="Person" className="person-bbox" />}
          {status.audience.face_bbox && <Box bbox={status.audience.face_bbox} width={width} height={height} label="Face" className="face-bbox" />}
          {status.audience.left_eye_bbox && <Box bbox={status.audience.left_eye_bbox} width={width} height={height} label="L Eye" className="eye-bbox left-eye" />}
          {status.audience.right_eye_bbox && <Box bbox={status.audience.right_eye_bbox} width={width} height={height} label="R Eye" className="eye-bbox right-eye" />}
          {status.audience.left_wrist_point && <Point point={status.audience.left_wrist_point} label="L Wrist" className="wrist-point left-wrist" />}
          {status.audience.right_wrist_point && <Point point={status.audience.right_wrist_point} label="R Wrist" className="wrist-point right-wrist" />}
        </div>
      </div>
      <p className="hint">
        {permission === "granted"
          ? "相機權限已由瀏覽器取得，前端持續上傳 frame 給後端做偵測與標記。"
          : permission === "denied"
            ? "瀏覽器相機權限被拒絕，請在站點權限中允許 camera。"
            : "等待瀏覽器取得 camera 權限。"}
      </p>
      <div className="camera-meta">
        <span className="field-chip applied">Requested: {width}x{height}@{fps}</span>
        <span className="field-chip applied">Browser Stream: {streamMode}</span>
        <span className={`field-chip ${lastUploadAt ? "applied" : "pending"}`}>
          Upload: {lastUploadAt ? "active" : "waiting"}
        </span>
      </div>
    </section>
  );
}

async function resolveBrowserCameraDeviceId(cameraDeviceId?: string): Promise<string | undefined> {
  if (!cameraDeviceId || cameraDeviceId === "default" || !navigator.mediaDevices?.enumerateDevices) {
    return undefined;
  }
  const devices = await navigator.mediaDevices.enumerateDevices();
  const browserIds = new Set(
    devices
      .filter((device) => device.kind === "videoinput")
      .map((device) => device.deviceId)
      .filter(Boolean),
  );
  return browserIds.has(cameraDeviceId) ? cameraDeviceId : undefined;
}

function SkeletonOverlay({ points }: { points: Record<string, [number, number] | null> }) {
  const segments: Array<[string, string]> = [
    ["nose", "left_shoulder"],
    ["nose", "right_shoulder"],
    ["left_shoulder", "right_shoulder"],
    ["left_shoulder", "left_elbow"],
    ["left_elbow", "left_wrist"],
    ["right_shoulder", "right_elbow"],
    ["right_elbow", "right_wrist"],
    ["left_shoulder", "left_hip"],
    ["right_shoulder", "right_hip"],
    ["left_hip", "right_hip"],
  ];

  return (
    <>
      <svg className="skeleton-overlay" viewBox="0 0 100 100" preserveAspectRatio="none">
        {segments.map(([from, to]) => {
          const start = points[from];
          const end = points[to];
          if (!start || !end) return null;
          return (
            <line
              key={`${from}-${to}`}
              x1={start[0] * 100}
              y1={start[1] * 100}
              x2={end[0] * 100}
              y2={end[1] * 100}
              className="skeleton-line"
            />
          );
        })}
      </svg>
      {Object.entries(points).map(([name, point]) =>
        point ? <Point key={name} point={point} label={labelForKeypoint(name)} className={`point keypoint ${name}`} /> : null,
      )}
    </>
  );
}

function labelForKeypoint(name: string) {
  switch (name) {
    case "left_wrist":
      return "L Wrist";
    case "right_wrist":
      return "R Wrist";
    case "left_elbow":
      return "L Elbow";
    case "right_elbow":
      return "R Elbow";
    case "left_shoulder":
      return "L Shoulder";
    case "right_shoulder":
      return "R Shoulder";
    case "left_hip":
      return "L Hip";
    case "right_hip":
      return "R Hip";
    case "nose":
      return "Nose";
    default:
      return name;
  }
}

function Point({
  point,
  label,
  className,
}: {
  point: [number, number];
  label: string;
  className: string;
  compact?: boolean;
}) {
  const [x, y] = point;
  return (
    <div
      className={`point ${className}`}
      style={{
        left: `${x * 100}%`,
        top: `${y * 100}%`,
      }}
    >
      <span>{label}</span>
    </div>
  );
}

function Box({
  bbox,
  width,
  height,
  label,
  className,
}: {
  bbox: [number, number, number, number];
  width: number;
  height: number;
  label: string;
  className: string;
}) {
  const [x1, y1, x2, y2] = bbox;
  return (
    <div
      className={`bbox ${className}`}
      style={{
        left: `${(x1 / width) * 100}%`,
        top: `${(y1 / height) * 100}%`,
        width: `${((x2 - x1) / width) * 100}%`,
        height: `${((y2 - y1) / height) * 100}%`,
      }}
    >
      {label}
    </div>
  );
}
