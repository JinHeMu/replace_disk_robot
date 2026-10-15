#!/usr/bin/env bash
# Source this file in each Bash terminal used for the project.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo '请使用 source scripts/activate_env.sh' >&2
    exit 1
fi

_rdr_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -f "$_rdr_root/.venv/bin/activate" ]]; then
    echo '请先运行 bash scripts/setup_env.sh' >&2
    unset _rdr_root
    return 1
fi
source "$_rdr_root/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export PYGAME_HIDE_SUPPORT_PROMPT=1
# This standalone SDK stack must not load ROS Humble's native Python modules.
unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH
unset CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
unset _rdr_root
