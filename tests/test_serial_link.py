from __future__ import annotations

from types import SimpleNamespace

from backend.serial.esp32_link import ESP32Link


class FakeSerialPort:
    def __init__(self, port: str, baud_rate: int, timeout: float) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.is_open = True
        self.buffer: list[bytes] = []

    @property
    def in_waiting(self) -> int:
        return 0

    def write(self, data: bytes) -> int:
        self.buffer.append(data)
        return len(data)

    def readline(self) -> bytes:
        return b""

    def close(self) -> None:
        self.is_open = False


def test_refresh_connection_marks_disconnected_when_port_removed(monkeypatch):
    active_ports = [SimpleNamespace(device="/dev/cu.usbmodem1101", description="CP210 USB")]

    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: list(active_ports))
    monkeypatch.setattr("backend.serial.esp32_link.serial.Serial", FakeSerialPort)

    link = ESP32Link("auto", 115200)
    try:
        assert link.connected is True
        assert link.port == "/dev/cu.usbmodem1101"

        active_ports.clear()
        link.refresh_connection()

        assert link.connected is False
        assert link.port == "auto"
    finally:
        link.close()


def test_refresh_connection_reconnects_after_auto_port_reappears(monkeypatch):
    active_ports = [SimpleNamespace(device="/dev/cu.usbmodem1101", description="CP210 USB")]
    opened_ports: list[str] = []

    def fake_serial(port: str, baud_rate: int, timeout: float) -> FakeSerialPort:
        opened_ports.append(port)
        return FakeSerialPort(port, baud_rate, timeout)

    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: list(active_ports))
    monkeypatch.setattr("backend.serial.esp32_link.serial.Serial", fake_serial)

    link = ESP32Link("auto", 115200)
    try:
        active_ports.clear()
        link.refresh_connection()
        assert link.connected is False

        active_ports.append(SimpleNamespace(device="/dev/cu.usbmodem2202", description="CP210 USB"))
        link.refresh_connection()
        payload = link.send_servo_command(100.0, 80.0)

        assert link.connected is True
        assert link.port == "/dev/cu.usbmodem2202"
        assert opened_ports == ["/dev/cu.usbmodem1101", "/dev/cu.usbmodem2202"]
        assert payload.startswith("{")
        assert link.snapshot().last_tx == payload
    finally:
        link.close()


def test_auto_does_not_latch_to_debug_console(monkeypatch):
    ports = [
        SimpleNamespace(device="/dev/cu.debug-console", description="n/a", hwid="n/a", manufacturer=None, product=None),
        SimpleNamespace(device="/dev/cu.Bluetooth-Incoming-Port", description="n/a", hwid="n/a", manufacturer=None, product=None),
    ]

    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: list(ports))

    link = ESP32Link("auto", 115200)
    try:
        assert link.connected is False
        assert link.port == "auto"
    finally:
        link.close()


def test_refresh_connection_switches_from_debug_console_to_usb_serial(monkeypatch):
    ports = [
        SimpleNamespace(device="/dev/cu.debug-console", description="n/a", hwid="n/a", manufacturer=None, product=None),
        SimpleNamespace(device="/dev/cu.usbserial-10", description="USB Serial", hwid="USB VID:PID=1A86:7523", manufacturer=None, product="USB Serial"),
    ]
    opened_ports: list[str] = []

    def fake_serial(port: str, baud_rate: int, timeout: float) -> FakeSerialPort:
        opened_ports.append(port)
        return FakeSerialPort(port, baud_rate, timeout)

    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: list(ports))
    monkeypatch.setattr("backend.serial.esp32_link.serial.Serial", fake_serial)

    link = ESP32Link("auto", 115200)
    try:
        assert link.port == "/dev/cu.usbserial-10"
        assert opened_ports == ["/dev/cu.usbserial-10"]
        link.port = "/dev/cu.debug-console"
        link.refresh_connection()
        assert link.port == "/dev/cu.usbserial-10"
    finally:
        link.close()


def test_build_servo_command_is_compact_for_esp32_parser(monkeypatch):
    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: [])
    link = ESP32Link("auto", 115200)
    try:
        payload = link.build_servo_command(100.24, 80.1)
        assert payload == '{"type":"servo","mode":"track","left_deg":100.24,"right_deg":80.1,"led_left_pct":50.0,"led_right_pct":50.0,"tracking_source":"eye_midpoint"}'
    finally:
        link.close()


def test_build_servo_command_includes_led_brightness(monkeypatch):
    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: [])
    link = ESP32Link("auto", 115200)
    try:
        payload = link.build_servo_command(100.24, 80.1, led_left_pct=72.5, led_right_pct=27.5)
        assert payload == '{"type":"servo","mode":"track","left_deg":100.24,"right_deg":80.1,"led_left_pct":72.5,"led_right_pct":27.5,"tracking_source":"eye_midpoint"}'
    finally:
        link.close()


def test_successful_serial_activity_clears_last_error(monkeypatch):
    monkeypatch.setattr("backend.serial.esp32_link.list_ports.comports", lambda: [])
    link = ESP32Link("auto", 115200)
    try:
        link._record_error("send skipped: serial port unavailable")
        assert link.snapshot().last_error == "send skipped: serial port unavailable"
        link._record("tx", '{"type":"servo"}')
        assert link.snapshot().last_error is None
    finally:
        link.close()
