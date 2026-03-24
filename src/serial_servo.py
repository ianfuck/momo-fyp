"""Serial protocol to Arduino SG90 firmware."""

from __future__ import annotations

import threading
import time
from typing import Literal

import serial

ServoMode = Literal["TRACK", "IDLE", "FRENZY", "DEAD"]


class SerialServo:
    def __init__(self) -> None:
        self._ser: serial.Serial | None = None
        self._lock = threading.Lock()
        self.mode: ServoMode = "IDLE"
        self.last_left = 90.0
        self.last_right = 90.0

    def connect(self, port: str, baud: int) -> None:
        with self._lock:
            if self._ser and self._ser.is_open:
                self._ser.close()
            self._ser = None
            if not port or not port.strip():
                return
            try:
                self._ser = serial.Serial(port.strip(), baud, timeout=0.2)
                time.sleep(0.1)
            except serial.SerialException:
                self._ser = None

    def send(
        self,
        mode: ServoMode,
        left_deg: float,
        right_deg: float,
    ) -> None:
        self.mode = mode
        self.last_left = left_deg
        self.last_right = right_deg
        with self._lock:
            if not self._ser or not self._ser.is_open:
                return
            line = f"{mode} {left_deg:.1f} {right_deg:.1f}\n"
            try:
                self._ser.write(line.encode("utf-8"))
            except Exception:
                pass
