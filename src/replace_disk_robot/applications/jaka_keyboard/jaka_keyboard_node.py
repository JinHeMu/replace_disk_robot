#!/usr/bin/env python3
"""Keyboard entrypoint with cycle logging.

The keyboard frontend and its existing configuration remain independent.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from replace_disk_robot.applications.jaka_keyboard.node import main


if __name__ == "__main__":
    main()
