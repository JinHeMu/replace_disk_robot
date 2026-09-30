"""SDK session lifetime and monotonic scheduling shared by JAKA applications."""
from __future__ import annotations
import argparse
import time
from contextlib import contextmanager
from typing import Iterator
import numpy as np
from .interfaces import EdgState, JakaEdgClient

def parse_bool(value: str) -> bool:
    """Parse explicit CLI switches without treating the string 'false' as true."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
    raise argparse.ArgumentTypeError("expected true or false")


def add_network_args(parser: argparse.ArgumentParser) -> None:
    """Add the IP arguments shared by every example script."""

    parser.add_argument("--robot-ip", default="10.5.5.100")
    parser.add_argument("--local-ip", default="10.5.5.127")


def make_client(args: argparse.Namespace) -> JakaEdgClient:
    return JakaEdgClient(args.robot_ip, args.local_ip)


def wait_for_edg_state(
    client: JakaEdgClient,
    *,
    retries: int = 20,
    delay_s: float = 0.1,
) -> EdgState:
    """Wait until the first valid EDG state is available."""

    if retries <= 0:
        raise ValueError("retries must be positive")
    last_error: Exception | None = None
    for index in range(retries):
        try:
            return client.read_edg_state()
        except Exception as exc:  # noqa: BLE001 - SDK may raise several types
            last_error = exc
            if index + 1 < retries:
                time.sleep(delay_s)
    raise RuntimeError(f"no EDG state after {retries} attempts: {last_error}")


@contextmanager
def edg_session(
    client: JakaEdgClient,
    *,
    torque_sensor_mode: int | None = None,
) -> Iterator[JakaEdgClient]:
    """Login, optionally configure the torque sensor, enable EDG, then clean up."""

    client.login()
    try:
        if torque_sensor_mode is not None and torque_sensor_mode >= 0:
            client.set_torque_sensor_mode(torque_sensor_mode)
        client.edg_enable(True)
        wait_for_edg_state(client)
        yield client
    finally:
        try:
            client.edg_disable(ignore_errors=True)
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass
        try:
            client.logout()
        except Exception:  # noqa: BLE001 - best-effort teardown
            pass


class RateLoop:
    """Monotonic, periodic iterator for 125 Hz-style control loops.

    Yields ``(elapsed_s, now_perf_counter)``.  The caller breaks when the
    elapsed time reaches the desired duration.
    """

    def __init__(self, rate_hz: float) -> None:
        if not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("rate_hz must be finite and positive")
        self.rate_hz = float(rate_hz)
        self.period_s = 1.0 / self.rate_hz
        self.overruns = 0

    def __iter__(self) -> Iterator[tuple[float, float]]:
        start = time.perf_counter()
        next_tick = start
        while True:
            now = time.perf_counter()
            if now < next_tick:
                time.sleep(min(next_tick - now, self.period_s))
                continue
            yield now - start, now
            next_tick += self.period_s
            if now - next_tick > self.period_s:
                self.overruns += 1
                next_tick = now
