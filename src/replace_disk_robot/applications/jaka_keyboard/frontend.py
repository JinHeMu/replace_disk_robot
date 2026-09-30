"""GLFW input/window frontend; hardware commands remain in the node."""
from __future__ import annotations
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import glfw
import numpy as np
from replace_disk_robot.adapters.jaka.session import RateLoop
if TYPE_CHECKING:
    from .node import JakaKeyboardServo

_KEY_TO_NAME = {
    glfw.KEY_W: "w",
    glfw.KEY_S: "s",
    glfw.KEY_A: "a",
    glfw.KEY_D: "d",
    glfw.KEY_R: "r",
    glfw.KEY_F: "f",
    glfw.KEY_Q: "q",
    glfw.KEY_E: "e",
    glfw.KEY_UP: "up",
    glfw.KEY_DOWN: "down",
    glfw.KEY_LEFT: "left",
    glfw.KEY_RIGHT: "right",
}

def _on_key(window, key, scancode, action, mods) -> None:  # noqa: ARG001
    app = glfw.get_window_user_pointer(window)
    if app is None:
        return
    app.record_input_event(key, action)
    if key == glfw.KEY_ESCAPE and action == glfw.PRESS:
        glfw.set_window_should_close(window, True)
        return
    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        app.stop("stopped")
        return
    if key == glfw.KEY_ENTER and action == glfw.PRESS:
        app.resume()
        return
    name = _KEY_TO_NAME.get(key)
    if name is None:
        return
    if action in (glfw.PRESS, glfw.REPEAT):
        app.keys.press(name)
    elif action == glfw.RELEASE:
        app.keys.release(name)


def _on_focus(window, focused: bool) -> None:
    app = glfw.get_window_user_pointer(window)
    if app is not None:
        app.handle_focus(bool(focused))


def _window_title(app: JakaKeyboardServo) -> str:
    force = 0.0
    torque = 0.0
    if app.last_wrench is not None:
        force = float(np.linalg.norm(app.last_wrench.force_n))
        torque = float(np.linalg.norm(app.last_wrench.torque_nm))
    admittance = app.admittance_text()
    return (
        f"JAKA keyboard servo | {app.command_frame} | "
        f"F={force:.2f} N | T={torque:.2f} Nm | "
        + (f"{admittance} | " if admittance else "")
        + f"{app._status_text()}"
    )


def _run_window(app: JakaKeyboardServo) -> None:
    if not glfw.init():
        raise RuntimeError("cannot initialize GLFW display; use --headless")

    window = None
    clear_window = None
    plotter = None
    plot_error_reported = False
    try:
        # Use a normal OpenGL-capable window and clear it every frame.  A
        # GLFW_NO_API window can be invisible on some Wayland compositors
        # because no buffer is ever attached, which makes keyboard focus hard.
        try:
            from OpenGL.GL import GL_COLOR_BUFFER_BIT, glClear, glClearColor
            clear_window = (glClear, glClearColor, GL_COLOR_BUFFER_BIT)
        except Exception:
            clear_window = None

        window = glfw.create_window(620, 220, "JAKA keyboard servo", None, None)
        if window is None:
            raise RuntimeError("cannot create GLFW window; use --headless")

        glfw.make_context_current(window)
        glfw.swap_interval(0)
        glfw.set_window_user_pointer(window, app)
        glfw.set_key_callback(window, _on_key)
        glfw.set_window_focus_callback(window, _on_focus)

        # Bring the control window to the front.  Some compositors deny
        # focus stealing, so the operator may still need to click it.
        glfw.show_window(window)
        glfw.focus_window(window)
        glfw.set_window_title(window, _window_title(app))

        if app.args.plot_wrench_enable:
            import os
            import tempfile

            os.environ.setdefault(
                "MPLCONFIGDIR",
                str(Path(tempfile.gettempdir()) / "replace_disk_robot_matplotlib"),
            )
            from replace_disk_robot.visual import ProcessTypePlotter
            plotter = ProcessTypePlotter(window_s=10.0, refresh_hz=10.0)

        print(
            "[keyboard] a small window named 'JAKA keyboard servo' has opened.\n"
            "[keyboard] Click that window, then hold keys there. "
            "Space stop, Enter resume, Esc exit."
            + (
                "\n[keyboard] admittance is on: keys move the nominal pose, the "
                "gravity-compensated external wrench moves the compliant offset."
                if app.admittance_enabled else ""
            )
        )
        loop = RateLoop(app.args.rate_hz)
        last_now = None
        for elapsed_s, now in loop:
            glfw.poll_events()
            if glfw.window_should_close(window):
                break
            dt_s = 1.0 / app.args.rate_hz if last_now is None else now - last_now
            last_now = now
            app.tick(elapsed_s, dt_s)
            if plotter is not None and app.last_wrench is not None:
                plotter.update(app.last_wrench, elapsed_s)
                if plotter.error and not plot_error_reported:
                    print(
                        f"[plot] plot disabled; keyboard control remains active:\n"
                        f"{plotter.error}",
                        file=sys.stderr,
                        flush=True,
                    )
                    plot_error_reported = True
            glfw.set_window_title(window, _window_title(app))
            if clear_window is not None:
                glClear, glClearColor, color_bit = clear_window
                glClearColor(0.72, 0.80, 0.90, 1.0)
                glClear(color_bit)
                glfw.swap_buffers(window)
            if app.args.seconds > 0 and elapsed_s >= app.args.seconds:
                break
    finally:
        if plotter is not None:
            plotter.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()
