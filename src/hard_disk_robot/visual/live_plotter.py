"""Reusable Matplotlib time-series display; GUI work runs on the caller thread."""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np

from .type_samples import PlotSample, to_plot_sample


class LiveTypePlotter:
    """Display one stream of a supported core type in a rolling time window.

    A plotter binds to the structure of its first sample. Create separate
    instances when displaying different types or JointState name sets.
    Matplotlib is imported lazily so importing ``hard_disk_robot.visual`` does
    not select a GUI backend.
    """

    def __init__(
        self,
        *,
        window_s: float = 10.0,
        refresh_hz: float = 20.0,
        show: bool = True,
    ) -> None:
        if not np.isfinite(window_s) or window_s <= 0:
            raise ValueError("window_s must be finite and positive")
        if not np.isfinite(refresh_hz) or refresh_hz <= 0:
            raise ValueError("refresh_hz must be finite and positive")
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "LiveTypePlotter requires Matplotlib; install the project's visual extra"
            ) from exc
        self._plt = plt
        self.window_s = float(window_s)
        self.refresh_period_s = 1.0 / float(refresh_hz)
        self.show = show
        self._times: deque[float] = deque()
        self._values: deque[np.ndarray] = deque()
        self._signature = None
        self._figure = None
        self._axes = ()
        self._lines = ()
        self._minimum_half_ranges = ()
        self._start_s = None
        self._last_draw_s = -np.inf

    def _sample_signature(self, sample: PlotSample):
        return tuple(
            (group.title, group.ylabel, group.labels, group.minimum_half_range)
            for group in sample.groups
        )

    def _initialize(self, sample: PlotSample) -> None:
        self._signature = self._sample_signature(sample)
        figure, axes = self._plt.subplots(
            len(sample.groups), 1, squeeze=False, sharex=True,
            figsize=(9, 3.2 * len(sample.groups)),
        )
        self._figure = figure
        self._axes = tuple(axes[:, 0])
        line_groups = []
        for axis, group in zip(self._axes, sample.groups):
            axis.set_title(group.title)
            axis.set_ylabel(group.ylabel)
            axis.grid(True, alpha=0.3)
            line_groups.append(tuple(axis.plot([], [], label=label)[0] for label in group.labels))
            axis.legend(loc="upper left", ncols=min(3, len(group.labels)))
        self._axes[-1].set_xlabel("time [s]")
        figure.suptitle(sample.title)
        figure.tight_layout()
        self._lines = tuple(line_groups)
        self._minimum_half_ranges = tuple(group.minimum_half_range for group in sample.groups)
        if self.show:
            figure.show()

    def update(self, value, time_s: float | None = None) -> None:
        """Append one value and service the GUI; layout/drawing may take time."""
        sample = to_plot_sample(value)
        if self._figure is None:
            self._initialize(sample)
        elif self._sample_signature(sample) != self._signature:
            raise ValueError("sample structure changed; use a separate LiveTypePlotter")

        timestamp = time.monotonic() if time_s is None else float(time_s)
        if not np.isfinite(timestamp):
            raise ValueError("time_s must be finite")
        if self._times and timestamp < self._times[-1]:
            raise ValueError("time_s must be monotonically non-decreasing")
        if self._start_s is None:
            self._start_s = timestamp
        relative_s = timestamp - self._start_s
        self._times.append(relative_s)
        self._values.append(np.concatenate([group.values for group in sample.groups]))
        while self._times and relative_s - self._times[0] > self.window_s:
            self._times.popleft()
            self._values.popleft()

        wall_s = time.monotonic()
        if wall_s - self._last_draw_s >= self.refresh_period_s:
            self._draw()
            self._last_draw_s = wall_s

    def _draw(self) -> None:
        if not self._times or self._figure is None:
            return
        times = np.asarray(self._times)
        values = np.vstack(self._values)
        column = 0
        for axis, line_group, minimum_half_range in zip(
            self._axes, self._lines, self._minimum_half_ranges
        ):
            group_values = []
            for line in line_group:
                line.set_data(times, values[:, column])
                group_values.append(values[:, column])
                column += 1
            half_range = max(
                minimum_half_range,
                float(np.max(np.abs(np.column_stack(group_values)))),
            )
            axis.set_ylim(-1.1 * half_range, 1.1 * half_range)
            axis.set_xlim(max(0.0, times[-1] - self.window_s), max(self.window_s, times[-1]))
        self._figure.canvas.draw_idle()
        self._figure.canvas.flush_events()

    @property
    def is_open(self) -> bool:
        return self._figure is not None and self._plt.fignum_exists(self._figure.number)

    def pump_events(self) -> None:
        """Keep the plot responsive even while its producer is not sending samples."""
        if self.is_open:
            self._figure.canvas.flush_events()

    def save(self, path: str | Path) -> None:
        """Save the current plot, including samples not yet drawn by throttling."""
        if self._figure is None:
            raise RuntimeError("cannot save before the first update")
        self._draw()
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        self._figure.savefig(output)

    def close(self) -> None:
        if self._figure is not None:
            self._plt.close(self._figure)
            self._figure = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
