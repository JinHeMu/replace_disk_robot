"""Polling SDL's named controller axes avoids driver-specific joystick indices."""
from __future__ import annotations

import os
import time
from ..control.joy_control import JoySample


class XboxController:
    def __init__(self, index=0):
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", "1")
        try:
            import pygame
            from pygame._sdl2 import controller
        except ImportError as exc:
            raise RuntimeError("Install controller support with: pip install -e '.[joy]'") from exc
        self.pygame, self.api, self.device = pygame, controller, None
        try:
            pygame.display.init()
            controller.init()
            if index < 0 or index >= controller.get_count() or not controller.is_controller(index):
                raise RuntimeError(f"SDL controller {index} unavailable; connect an Xbox controller")
            self.device = controller.Controller(index)
            self.name = controller.name_forindex(index)
            self.mapping = self.device.get_mapping()
            self.guid = self.device.as_joystick().get_guid()
        except BaseException:
            self.close()
            raise

    def poll(self):
        try:
            return self._poll()
        except self.pygame.error:
            return JoySample(time.monotonic(), connected=False)

    def _poll(self):
        p = self.pygame
        # Drain events each tick, including removal; do not accumulate an event backlog.
        events = p.event.get()
        now = time.monotonic()
        if any(e.type == p.QUIT for e in events):
            return JoySample(now, buttons=frozenset({"back"}))
        if self.device is not None and not self.device.attached():
            self.device.quit()
            self.device = None
            # Report at least one disconnected sample, even for a quick replug.
            return JoySample(now, connected=False)
        if self.device is None:
            for index in range(self.api.get_count()):
                if not self.api.is_controller(index):
                    continue
                candidate = self.api.Controller(index)
                if candidate.as_joystick().get_guid() == self.guid:
                    self.device = candidate
                    break
                candidate.quit()
            if self.device is None:
                return JoySample(now, connected=False)
        try:
            axes = []
            for name in ("LEFTX", "LEFTY", "RIGHTX", "RIGHTY", "TRIGGERLEFT", "TRIGGERRIGHT"):
                raw = self.device.get_axis(getattr(p, "CONTROLLER_AXIS_" + name))
                denominator = 32768. if raw < 0 or name.startswith("TRIGGER") else 32767.
                axes.append(raw / denominator)
            names = {"a": "A", "b": "B", "lb": "LEFTSHOULDER", "rb": "RIGHTSHOULDER",
                     "back": "BACK", "dpleft": "DPAD_LEFT", "dpright": "DPAD_RIGHT"}
            buttons = frozenset(name for name, constant in names.items()
                                if self.device.get_button(getattr(p, "CONTROLLER_BUTTON_" + constant)))
            return JoySample(now, tuple(axes), buttons)
        except p.error:
            return JoySample(now, connected=False)

    def close(self):
        if self.device is not None:
            self.device.quit()
            self.device = None
        self.api.quit()
        self.pygame.display.quit()
