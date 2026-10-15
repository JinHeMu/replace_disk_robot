#!/usr/bin/env bash
# Ubuntu 22.04/24.04, x86_64. All Python tools live inside this checkout.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
skip_system=0
case "${1:-}" in
    '') ;;
    --skip-system) skip_system=1 ;;
    -h|--help)
        echo '用法: bash scripts/setup_env.sh [--skip-system]'
        echo '默认安装项目环境、系统图形库和 D435 USB 权限，并执行软件自检。'
        echo '--skip-system 仅安装项目内软件；跳过 apt 和需要 sudo 的 USB 权限。'
        exit 0 ;;
    *) echo "未知参数: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then
    echo '参数过多，请运行 bash scripts/setup_env.sh --help' >&2
    exit 2
fi
if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo '当前配置支持 Linux x86_64（仓库内 JAKA SDK 的目标平台）。' >&2
    exit 1
fi

unset PYTHONPATH PYTHONHOME LD_LIBRARY_PATH VIRTUAL_ENV
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH CMAKE_PREFIX_PATH PKG_CONFIG_PATH
unset CPATH CPLUS_INCLUDE_PATH LIBRARY_PATH ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION
export PYTHONNOUSERSITE=1
export UV_PYTHON_INSTALL_DIR="$repo_root/.tools/python"
export UV_PYTHON_BIN_DIR="$repo_root/.tools/bin"
export UV_CACHE_DIR="$repo_root/.tools/cache"
export UV_PROJECT_ENVIRONMENT="$repo_root/.venv"

as_root=()
if (( EUID != 0 )); then as_root=(sudo); fi
if (( ! skip_system )); then
    missing=()
    for package in curl ca-certificates libgl1 libegl1 libopengl0 libglfw3 libusb-1.0-0; do
        if [[ "$(dpkg-query -W -f='${db:Status-Status}' "$package" 2>/dev/null || true)" != installed ]]; then
            missing+=("$package")
        fi
    done
    if (( ${#missing[@]} )); then
        echo "安装系统依赖（需要 sudo）: ${missing[*]}"
        "${as_root[@]}" apt-get update
        "${as_root[@]}" apt-get install -y --no-install-recommends "${missing[@]}"
    fi
fi

uv_version=0.13.1
uv_bin="$repo_root/.tools/uv"
if [[ ! -x "$uv_bin" ]] || [[ "$("$uv_bin" --version)" != "uv $uv_version "* ]]; then
    mkdir -p "$repo_root/.tools"
    installer="$(mktemp)"
    trap 'rm -f -- "$installer"' EXIT
    curl -fLsS --retry 3 "https://astral.sh/uv/$uv_version/install.sh" -o "$installer"
    UV_UNMANAGED_INSTALL="$repo_root/.tools" sh "$installer"
    rm -f -- "$installer"
    trap - EXIT
fi

echo '同步 Python、JAKA/MuJoCo、D435 和 RL 依赖（首次需要下载 PyTorch CUDA 包）。'
"$uv_bin" sync --locked --managed-python
"$uv_bin" pip check --python "$repo_root/.venv/bin/python"
source "$repo_root/scripts/activate_env.sh"
python "$repo_root/examples/check_environment.py"

if (( ! skip_system )); then
    rules="$repo_root/.tools/99-realsense-libusb.rules"
    rules_sha=34c287000862710b60732aea942bda6f51faf7436b5b53f66fbfd021bc8c22dd
    if [[ ! -f "$rules" ]] || ! echo "$rules_sha  $rules" | sha256sum --check --status; then
        curl -fLsS --retry 3 \
            https://raw.githubusercontent.com/realsenseai/librealsense/v2.59.1/config/99-realsense-libusb.rules \
            -o "$rules"
    fi
    echo "$rules_sha  $rules" | sha256sum --check --status
    target=/etc/udev/rules.d/99-realsense-libusb.rules
    if ! cmp -s "$rules" "$target"; then
        echo '安装 RealSense 官方 USB 权限规则（需要 sudo；无需安装 CUDA Toolkit 或 DKMS）。'
        if [[ -e "$target" ]]; then
            "${as_root[@]}" cp -a "$target" "$target.bak.$(date -u +%Y%m%dT%H%M%SZ)"
        fi
        "${as_root[@]}" install -D -m 644 "$rules" "$target"
    fi
    "${as_root[@]}" udevadm control --reload-rules
    echo 'D435 USB 规则已安装；请重新插拔相机，使权限生效。'
else
    echo '已跳过系统依赖与 D435 USB 权限；需要时重新运行不带参数的安装命令。'
fi

echo '项目软件环境已就绪。新终端执行: source scripts/activate_env.sh'
echo 'D435 接上 USB 3 后验证取流: python examples/check_environment.py --camera'
