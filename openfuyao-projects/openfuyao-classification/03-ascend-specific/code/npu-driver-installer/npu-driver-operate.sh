#!/bin/bash
###############################################################
# Copyright (c) 2024 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

EXEC_DIR_NAME=""
DRIVER_VERSION=""
MODE="auto"
HOST_OFFLINE_PATH=""
INSTALL_WORK_DIR=""
BASE_DIR=""

function _log() {
  local prefix="$1"
  shift
  echo "$(date +"[%Y-%m-%d %H:%M:%S,%N]") [${prefix}] $*"
}

function info_log () {
    _log "INFO" "$*"
}

function warning_log () {
    _log "WARNING" "$*"
}

function error_log () {
    _log "ERROR" "$*"
}

function fatal_log () {
    _log "FATAL" "$*"
    exit 1
}

function parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --work-dir)     EXEC_DIR_NAME="$2"; shift 2 ;;
            --version)      DRIVER_VERSION="$2"; shift 2 ;;
            --mode)         MODE="$2"; shift 2 ;;
            --offline-path) HOST_OFFLINE_PATH="$2"; shift 2 ;;
            *)              shift ;;
        esac
    done

    BASE_DIR="/opt/${EXEC_DIR_NAME}"
    INSTALL_WORK_DIR="${BASE_DIR}/npu-driver"
}

function detect_chip() {
    if lspci -n | grep -i "19e5:d802" >/dev/null; then
            echo "910b"
    elif lspci -n | grep -i "19e5:d801" >/dev/null; then echo "910";
    elif lspci -n | grep -i "19e5:d500" >/dev/null; then echo "310p";
    elif lspci -n | grep -i "19e5:d803" >/dev/null; then echo "A3";
    else
        if lspci | grep -i "d802" >/dev/null; then echo "910b";
        elif lspci | grep -i "d803" >/dev/null; then echo "A3";
        elif lspci | grep -i "d500" >/dev/null; then echo "310p";
        elif lspci | grep -i "d801" >/dev/null; then echo "910";
        else echo "unknown"; fi
    fi
}

function check_existing_driver() {
    local SMI="/usr/local/sbin/npu-smi"
    if [ -f "$SMI" ] && "$SMI" info >/dev/null 2>&1; then
        info_log "NPU driver detected and functional. Skipping installation."
        exit 0
    fi
    info_log "NPU driver not found or malfunction. Proceeding to install."
}

function install_os_dependencies() {
    info_log "Checking and installing OS dependencies..."

    local PACKAGE_MANAGER=""
    local PKG_LIST=""

    if command -v yum >/dev/null 2>&1; then
        info_log "Using yum for package installation"
        yum clean all >/dev/null 2>&1
        yum makecache >/dev/null 2>&1
        PACKAGE_MANAGER="yum"
        PKG_LIST="jq wget unzip which net-tools pciutils gcc make kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms"

    elif command -v dnf >/dev/null 2>&1; then
        info_log "Using dnf for package installation"
        dnf makecache >/dev/null 2>&1
        PACKAGE_MANAGER="dnf"
        PKG_LIST="jq wget unzip which net-tools pciutils gcc make kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms"

    elif command -v apt-get >/dev/null 2>&1; then
        info_log "Using apt-get for package installation"
        apt-get update >/dev/null 2>&1
        PACKAGE_MANAGER="apt-get"
        PKG_LIST="jq wget unzip debianutils net-tools pciutils gcc make dkms linux-headers-$(uname -r)"

    else
        fatal_log "No compatible package manager found (yum, apt-get, dnf)."
    fi

    for pkg in $PKG_LIST; do
        case "$PACKAGE_MANAGER" in
            yum|dnf)
                if ! rpm -q "$pkg" >/dev/null 2>&1; then
                    info_log "Installing missing package: $pkg ..."
                    if ! $PACKAGE_MANAGER -y install "$pkg"; then
                        fatal_log "Failed to install $pkg using $PACKAGE_MANAGER"
                    fi
                else
                    info_log "Package $pkg is already installed. Skipping."
                fi
                ;;
            apt-get)
                if ! dpkg -l | grep -q "^ii\s\+$pkg"; then
                    info_log "Installing missing package: $pkg ..."
                    if ! $PACKAGE_MANAGER -y install "$pkg"; then
                        fatal_log "Failed to install $pkg using $PACKAGE_MANAGER"
                    fi
                else
                    info_log "Package $pkg is already installed. Skipping."
                fi
                ;;
        esac
    done

    if [ "$MODE" == "online" ] && ! command -v jq >/dev/null; then
        fatal_log "Critical dependency 'jq' missing after installation attempt. Cannot proceed with online mode."
    fi

    info_log "All dependency packages verified successfully."
}

function find_offline_package() {
    local chip=$1
    local raw_arch=$(uname -m)
    local arch_pattern=""

    if [ "$raw_arch" == "x86_64" ]; then
        arch_pattern="x86?64"
    else
        arch_pattern="$raw_arch"
    fi

    if [ -z "$HOST_OFFLINE_PATH" ] || [ ! -d "$HOST_OFFLINE_PATH" ]; then
        return 1
    fi

    info_log "Searching offline packages in: $HOST_OFFLINE_PATH (Arch Pattern: $arch_pattern)"

    local zip=$(find "$HOST_OFFLINE_PATH" -name "*${chip}*${arch_pattern}*.zip" -o -name "*${arch_pattern}*${chip}*.zip" | head -n 1)

    if [ -z "$zip" ]; then
        zip=$(find "$HOST_OFFLINE_PATH" -name "*${arch_pattern}*.zip" | head -n 1)
    fi

    if [ -n "$zip" ]; then
        info_log "Found offline package: $zip"
        unzip -o -q "$zip" -d "$INSTALL_WORK_DIR"
        return 0
    fi

    return 1
}

function download_online_package() {
    local chip=$1
    local arch=$2
    local config_path="${BASE_DIR}/NPU/${DRIVER_VERSION}/config.json"

    info_log "Attempting online download (Version: $DRIVER_VERSION)..."

    if [ ! -f "$config_path" ]; then
        fatal_log "Config file missing: $config_path"
    fi

    local key="${chip}_${arch}"
    local url=$(jq -r --arg k "$key" '.other[] | select(has($k)) | .[$k].url' "$config_path")

    if [ -z "$url" ] || [ "$url" == "null" ]; then
        fatal_log "Download URL not found for key: $key"
    fi

    local filename=$(basename "$url")
    wget --no-check-certificate --referer="https://www.hiascend.com/" \
         -O "${INSTALL_WORK_DIR}/${filename}" "$url" || fatal_log "Download failed."

    unzip -o -q "${INSTALL_WORK_DIR}/${filename}" -d "$INSTALL_WORK_DIR"
}

function fetch_package() {
    local chip=$(detect_chip)
    local arch=$(uname -m)

    if [ "$chip" == "unknown" ]; then fatal_log "Unsupported NPU chip."; fi

    mkdir -p "$INSTALL_WORK_DIR"

    if ! command -v unzip &> /dev/null; then
        fatal_log "unzip command not found. Please install unzip manually or check the system repository."
    fi

    if [ "$MODE" != "online" ]; then
        if find_offline_package "$chip"; then
            return
        fi

        if [ "$MODE" == "local" ]; then
            fatal_log "Mode is 'local' but no package found in $HOST_OFFLINE_PATH"
        fi
    fi

    download_online_package "$chip" "$arch"
}

function run_installer_script() {
    cd "$INSTALL_WORK_DIR" || fatal_log "Cannot verify work dir."

    local nested=$(find . -maxdepth 1 -name "*.zip" | head -n 1)
    if [ -n "$nested" ]; then
        info_log "Extracting nested zip: $nested"
        unzip -o -q "$nested"
    fi

    local script=$(find . -name "install.sh" | head -n 1)
    if [ -z "$script" ]; then
        fatal_log "install.sh not found in package."
    fi

    chmod +x "$script"
    cd "$(dirname "$script")"

    info_log "Executing installation..."
    bash install.sh install all

    if [ $? -eq 0 ]; then
        info_log "Installation successful."
    else
        fatal_log "Installation script returned error."
    fi
}

function main() {
    parse_args "$@"
    check_existing_driver
    install_os_dependencies
    fetch_package
    run_installer_script
}

main "$@"
