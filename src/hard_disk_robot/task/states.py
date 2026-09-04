"""Stable names for the replacement task and reusable manipulation primitives."""

from enum import Enum


class TaskPrimitive(str, Enum):
    REACH = "reach"
    PRESS_LATCH = "press_latch"
    GRASP = "grasp"
    PULL = "pull"
    PLACE = "place"
    PICK = "pick"
    ALIGN = "align"
    INSERT = "insert"
    VERIFY = "verify"


class TaskState(str, Enum):
    IDLE = "idle"
    PRECHECK = "precheck"
    LOCALIZE = "localize"
    APPROACH_LATCH = "approach_latch"
    PRESS_LATCH = "press_latch"
    EJECT = "eject"
    GRASP_OLD = "grasp_old"
    EXTRACT_OLD = "extract_old"
    PLACE_OLD = "place_old"
    GRASP_NEW = "grasp_new"
    ALIGN_NEW = "align_new"
    INSERT_NEW = "insert_new"
    VERIFY = "verify"
    DONE = "done"
    FAILED = "failed"
