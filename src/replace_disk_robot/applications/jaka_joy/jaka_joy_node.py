#!/usr/bin/env python3
"""Xbox Cartesian teleoperation; defaults to MuJoCo, --backend real selects JAKA."""
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[3]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from replace_disk_robot.applications.jaka_joy.joy_node import main

if __name__ == "__main__":
    main()
