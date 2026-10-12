"""Live controller polling with optional MuJoCo viewer and force plot."""
from contextlib import nullcontext
from dataclasses import replace
import time

from replace_disk_robot.adapters.jaka.session import RateLoop


def run(app, source):
    viewer_context = nullcontext(None)
    if app.args.backend == "sim" and not app.args.headless:
        import mujoco.viewer
        viewer_context = mujoco.viewer.launch_passive(app.client.model, app.client.data)
    plotter = None
    try:
        if app.args.plot_wrench_enable:
            from replace_disk_robot.visual import ProcessTypePlotter
            plotter = ProcessTypePlotter(window_s=10., refresh_hz=10.)
        with viewer_context as viewer:
            if viewer is not None:
                viewer.cam.lookat[:] = app.client.data.body("jaka_base_link").xpos + [0, 0, .35]
                viewer.cam.distance = 1.8
                viewer.cam.azimuth, viewer.cam.elevation = 135, -25
            start = time.monotonic()
            previous, next_render = None, 0.
            for _, now in RateLoop(app.args.rate_hz):
                # Source timestamp and control clock share one monotonic origin.
                sample = source.poll()
                elapsed = time.monotonic() - start
                sample = replace(sample, timestamp_s=sample.timestamp_s - start)
                dt = 1. / app.args.rate_hz if previous is None else elapsed - previous
                previous = elapsed
                app.apply_sample(sample, elapsed)
                if app.exit_requested or (viewer is not None and not viewer.is_running()):
                    break
                app.tick(elapsed, dt)
                if app.args.backend == "sim":
                    app.client.step()
                if plotter is not None and app.last_wrench is not None:
                    plotter.update(app.last_wrench, elapsed)
                if viewer is not None and elapsed >= next_render:
                    viewer.sync()
                    next_render = elapsed + 1. / 30.
                if app.args.seconds > 0 and elapsed >= app.args.seconds:
                    break
    finally:
        if plotter is not None:
            plotter.close()
