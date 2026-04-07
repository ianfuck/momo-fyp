from __future__ import annotations

import contextlib
import signal
import threading
from collections.abc import Callable

_shutdown_requested = threading.Event()


def request_shutdown() -> None:
    _shutdown_requested.set()


def clear_shutdown_request() -> None:
    _shutdown_requested.clear()


def shutdown_requested() -> bool:
    return _shutdown_requested.is_set()


def install_shutdown_signal_bridge() -> Callable[[], None]:
    previous_handlers: dict[int, object] = {}

    def _make_handler(previous: object):
        def _handler(signum, frame) -> None:
            request_shutdown()
            if callable(previous):
                previous(signum, frame)

        return _handler

    for signum_name in ("SIGINT", "SIGTERM"):
        signum = getattr(signal, signum_name, None)
        if signum is None:
            continue
        previous = signal.getsignal(signum)
        previous_handlers[signum] = previous
        signal.signal(signum, _make_handler(previous))

    def _restore() -> None:
        for signum, previous in previous_handlers.items():
            with contextlib.suppress(Exception):
                signal.signal(signum, previous)

    return _restore
