"""Bounded, non-blocking transport to a plotter in its own GUI process."""
from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full
import time
import traceback


def _plot_worker(samples, events, stop, options):
    # Spawn keeps the Matplotlib GUI and its event loop out of the servo process.
    plotter = None
    try:
        import matplotlib
        if not options.get('show', True):
            matplotlib.use('Agg')
        from .live_plotter import LiveTypePlotter
        # Where supported, showing a new plot must not steal keyboard focus.
        with matplotlib.rc_context({'figure.raise_window': False}):
            plotter = LiveTypePlotter(**options)
            ready = False
            while not stop.is_set():
                if ready and not plotter.is_open:
                    events.put(('closed', 'Plot window closed'))
                    return
                try:
                    value, timestamp = samples.get(timeout=.05)
                except Empty:
                    if ready:
                        plotter.pump_events()
                    continue
                plotter.update(value, timestamp)
                if not ready:
                    events.put(('ready', ''))
                    ready = True
    except Exception:
        events.put(('error', traceback.format_exc()))
    finally:
        if plotter is not None:
            plotter.close()


class ProcessTypePlotter:
    """LiveTypePlotter-compatible producer that never draws or waits in update.

    A bounded queue drops incoming samples when full rather than delaying the
    control loop. Closing/failing the optional plot does not stop the robot GUI.
    """
    def __init__(self, *, window_s=10., refresh_hz=10., show=True, queue_size=256):
        if queue_size <= 0:
            raise ValueError('queue_size must be positive')
        if not (0 < window_s < float('inf') and 0 < refresh_hz < float('inf')):
            raise ValueError('Plot window and refresh rate must be finite and positive')
        context = mp.get_context('spawn')
        self._samples = context.Queue(maxsize=queue_size)
        self._events = context.Queue()
        self._stop = context.Event()
        self._process = context.Process(target=_plot_worker,
            args=(self._samples, self._events, self._stop,
                  dict(window_s=window_s, refresh_hz=refresh_hz, show=show)), daemon=True)
        self.status = 'starting'
        self.error = None
        self.dropped_samples = 0
        self._closed = False
        self._process.start()

    def poll(self):
        if self._closed:
            return self.status
        while True:
            try:
                self.status, detail = self._events.get_nowait()
            except Empty:
                break
            if self.status == 'error':
                self.error = detail
        if self._process.exitcode is not None and self.status not in ('closed', 'error'):
            self.status = 'error'
            self.error = f'Plot process exited with code {self._process.exitcode}'
        return self.status

    def update(self, value, time_s=None):
        if self.poll() in ('closed', 'error') or self._closed:
            return False
        timestamp = time.monotonic() if time_s is None else float(time_s)
        if not -float('inf') < timestamp < float('inf'):
            raise ValueError('time_s must be finite')
        try:
            self._samples.put_nowait((value, timestamp))
            return True
        except Full:
            self.dropped_samples += 1
            return False

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        self._process.join(timeout=.5)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=.5)
        for queue in (self._samples, self._events):
            queue.cancel_join_thread()
            queue.close()
        self.status = 'closed'
