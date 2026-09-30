#!/usr/bin/env python3
"""Analyze a recorded JAKA session without importing or contacting the SDK."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from replace_disk_robot.applications.jaka_keyboard.analysis import main

if __name__ == "__main__":
    main()
