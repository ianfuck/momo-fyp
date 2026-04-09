from __future__ import annotations

import json
import threading
import time
from collections import deque

import serial
from serial.tools import list_ports

from backend.types import SerialMonitorEntry, SerialMonitorSnapshot


class ESP32Link:
    def __init__(self, port: str, baud_rate: int) -> None:
        self.requested_port = port
        self.port = port
        self.baud_rate = baud_rate
        self.connected = False
        self.serial_port: serial.Serial | None = None
        self._lock = threading.RLock()
        self._entries: deque[SerialMonitorEntry] = deque(maxlen=50)
        self._last_tx: tuple[str, str] | None = None
        self._last_rx: tuple[str, str] | None = None
        self._last_error: tuple[str, str] | None = None
        self._reader_stop = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self.connect()

    @staticmethod
    def list_ports() -> list[dict[str, str]]:
        return [
            {"path": p.device, "description": p.description}
            for p in list_ports.comports()
        ] or [{"path": "auto", "description": "Auto detect"}]

    def connect(self) -> None:
        with self._lock:
            if self.serial_port and self.serial_port.is_open:
                self.connected = True
                return
            try:
                target = self._resolve_target_port()
                if target:
                    self.serial_port = serial.Serial(target, self.baud_rate, timeout=0.1)
                    self.port = target
                    self.connected = True
                    self._record("status", f"connected {self.port} @ {self.baud_rate}")
                    self._ensure_reader_thread()
                else:
                    self.connected = False
                    self._record_error("no serial port detected")
            except Exception as exc:
                self.connected = False
                self._record_error(str(exc))

    def refresh_connection(self) -> None:
        with self._lock:
            serial_port = self.serial_port
            current_port = self.port
            requested_port = self.requested_port
        preferred_target = self._resolve_target_port()
        if requested_port == "auto" and current_port not in {"", "auto", preferred_target} and preferred_target:
            self._drop_connection()
            self.connect()
            return
        if serial_port and serial_port.is_open:
            if current_port and not self._port_exists(current_port):
                self._drop_connection()
                self._record_error("serial device disconnected")
                return
            try:
                serial_port.in_waiting
                with self._lock:
                    self.connected = True
                return
            except Exception as exc:
                self._drop_connection()
                self._record_error(str(exc))
                return

        target = self._resolve_target_port()
        if target:
            self.connect()
        else:
            with self._lock:
                self.connected = False
                self.port = self.requested_port

    def close(self) -> None:
        self._reader_stop.set()
        reader_thread = self._reader_thread
        if reader_thread and reader_thread.is_alive():
            reader_thread.join(timeout=0.5)
        self._drop_connection()

    def snapshot(self) -> SerialMonitorSnapshot:
        with self._lock:
            return SerialMonitorSnapshot(
                port=self.port or None,
                baud_rate=self.baud_rate,
                last_tx=self._last_tx[1] if self._last_tx else None,
                last_tx_at=self._last_tx[0] if self._last_tx else None,
                last_rx=self._last_rx[1] if self._last_rx else None,
                last_rx_at=self._last_rx[0] if self._last_rx else None,
                last_error=self._last_error[1] if self._last_error else None,
                last_error_at=self._last_error[0] if self._last_error else None,
                entries=list(self._entries),
            )

    def build_servo_command(
        self,
        left_deg: float,
        right_deg: float,
        mode: str = "track",
        tracking_source: str = "eye_midpoint",
        led_left_pct: float = 50.0,
        led_right_pct: float = 50.0,
    ) -> str:
        return json.dumps(
            {
                "type": "servo",
                "mode": mode,
                "left_deg": round(left_deg, 2),
                "right_deg": round(right_deg, 2),
                "led_left_pct": round(led_left_pct, 2),
                "led_right_pct": round(led_right_pct, 2),
                "tracking_source": tracking_source,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def send_servo_command(
        self,
        left_deg: float,
        right_deg: float,
        mode: str = "track",
        tracking_source: str = "eye_midpoint",
        led_left_pct: float = 50.0,
        led_right_pct: float = 50.0,
    ) -> str:
        payload = self.build_servo_command(
            left_deg,
            right_deg,
            mode=mode,
            tracking_source=tracking_source,
            led_left_pct=led_left_pct,
            led_right_pct=led_right_pct,
        )
        with self._lock:
            if not self.connected:
                self.refresh_connection()
            if self.serial_port and self.serial_port.is_open:
                try:
                    self.serial_port.write((payload + "\n").encode("utf-8"))
                    self._record("tx", payload)
                except Exception as exc:
                    self._drop_connection()
                    self._record_error(str(exc))
            else:
                self._record_error("send skipped: serial port unavailable")
        return payload

    def _resolve_target_port(self) -> str:
        target = self.requested_port
        if target != "auto":
            return target
        candidates = list_ports.comports()
        for item in candidates:
            if self._looks_like_hardware_serial(item):
                return item.device
        return ""

    @staticmethod
    def _port_exists(path: str) -> bool:
        return any(item.device == path for item in list_ports.comports())

    @staticmethod
    def _looks_like_hardware_serial(item) -> bool:
        device = (getattr(item, "device", "") or "").lower()
        description = (getattr(item, "description", "") or "").lower()
        hwid = (getattr(item, "hwid", "") or "").lower()
        product = (getattr(item, "product", "") or "").lower()
        manufacturer = (getattr(item, "manufacturer", "") or "").lower()
        haystack = " ".join([device, description, hwid, product, manufacturer])
        return any(token in haystack for token in ("usb", "wch", "cp210", "ch340", "uart", "serial"))

    def _ensure_reader_thread(self) -> None:
        if self._reader_thread and self._reader_thread.is_alive():
            return
        self._reader_stop.clear()
        self._reader_thread = threading.Thread(target=self._reader_loop, name="esp32-serial-reader", daemon=True)
        self._reader_thread.start()

    def _reader_loop(self) -> None:
        while not self._reader_stop.is_set():
            with self._lock:
                serial_port = self.serial_port
            if serial_port is None or not serial_port.is_open:
                time.sleep(0.1)
                continue
            try:
                raw = serial_port.readline()
            except Exception as exc:
                self._drop_connection()
                self._record_error(str(exc))
                time.sleep(0.2)
                continue
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                self._record("rx", line)

    def _drop_connection(self) -> None:
        with self._lock:
            serial_port = self.serial_port
            self.serial_port = None
            self.connected = False
            self.port = self.requested_port
        if serial_port and serial_port.is_open:
            try:
                serial_port.close()
            except Exception:
                pass

    def _record(self, direction: str, message: str) -> None:
        timestamp = self._now()
        entry = SerialMonitorEntry(ts=timestamp, direction=direction, message=message)
        with self._lock:
            self._entries.appendleft(entry)
            if direction in {"status", "tx", "rx"}:
                self._last_error = None
            if direction == "tx":
                self._last_tx = (timestamp, message)
            elif direction == "rx":
                self._last_rx = (timestamp, message)

    def _record_error(self, message: str) -> None:
        timestamp = self._now()
        entry = SerialMonitorEntry(ts=timestamp, direction="error", message=message)
        with self._lock:
            self._entries.appendleft(entry)
            self._last_error = (timestamp, message)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
