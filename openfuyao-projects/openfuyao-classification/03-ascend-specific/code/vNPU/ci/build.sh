#!/bin/bash

# /*
#  * Copyright (c) 2026 Huawei Technologies Co., Ltd.
#  * openFuyao is licensed under Mulan PSL v2.
#  * You can use this software according to the terms and conditions of the Mulan PSL v2.
#  * You may obtain a copy of Mulan PSL v2 at:
#  *          http://license.coscl.org.cn/MulanPSL2
#  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  * See the Mulan PSL v2 for more details.
#  */

# 本地构建脚本 —— 在宿主机上编译二进制并打包为 Docker 镜像 tar 与 helm chart 包。
#
# 前置条件: Go >= 1.24.5, CANN, HDK 驱动, gcc, cmake, make, patch, objcopy, docker, helm
# 环境变量: source <cann-path>/set_env.sh (CANN), ENPU_ASCEND_DRIVER_PATH (HDK, 默认 /usr/local/Ascend)
#
# 用法:
#   ./ci/build.sh                                 # 全部组件
#   ./ci/build.sh npu-device-plugin               # 指定组件
#   ./ci/build.sh npu-device-plugin xpu-exporter  # 多个组件
#   ./ci/build.sh volcano                         # volcano 三件套（一次构建 vc-scheduler/vc-controller-manager/vc-webhook-manager 三个镜像）
#   ./ci/build.sh chart                           # 单独打chart包（仅需helm，不清理已有构建产物）
# 可用组件: acl-client  npu-device-plugin  xpu-exporter  volcano  chart
# 镜像: LOCAL_ARTIFACTS=true 复用 ./ci/output/ 产物，镜像 tar 输出到 ci/output/images/
# chart: helm package 打包 charts/vnpu，输出到 ci/output/chart/
#

set -e

# ==================== 路径初始化 ====================

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
TOP_DIR=$(dirname "${SCRIPT_DIR}")
SCRIPT_PATH="${SCRIPT_DIR}/pipeline/script"
OUTPUT_DIR="${SCRIPT_DIR}/output"
IMAGE_OUTPUT_DIR="${OUTPUT_DIR}/images"

# ==================== 架构识别 ====================

PLATFORM=$(uname -m)
case "${PLATFORM}" in
    x86_64)  PLATFORM="x86_64"  ;;
    aarch64) PLATFORM="aarch64" ;;
    *) echo "[ERROR] 不支持的架构: ${PLATFORM}"; exit 1 ;;
esac

# ==================== 环境检测 ====================

CANN_ENV_SCRIPT=""

# 查找 CANN set_env.sh：优先 ASCEND_HOME_PATH，回退标准安装路径
function find_cann_env_script() {
    if [ -n "${ASCEND_HOME_PATH:-}" ] && [ -f "${ASCEND_HOME_PATH}/set_env.sh" ]; then
        CANN_ENV_SCRIPT="${ASCEND_HOME_PATH}/set_env.sh"
        return
    fi

    if [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
        CANN_ENV_SCRIPT="/usr/local/Ascend/ascend-toolkit/set_env.sh"
        return
    fi
}

function check_command() {
    if ! command -v "$1" >/dev/null 2>&1; then
        echo "[ERROR] 未找到命令: $1，请先安装。"
        exit 1
    fi
}

# 查找 HDK 驱动：优先 ENPU_ASCEND_DRIVER_PATH（与 vcann-rt/make_build.sh 约定一致），回退 /usr/local/Ascend
function find_hdk_env() {
    HDK_DRIVER_PATH="${ENPU_ASCEND_DRIVER_PATH:-/usr/local/Ascend}"
    [ -d "${HDK_DRIVER_PATH}/driver" ] && HDK_FOUND=true || HDK_FOUND=false
}

function check_hdk_env() {
    find_hdk_env
    if [ "${HDK_FOUND}" = true ]; then
        echo "  HDK:      ${HDK_DRIVER_PATH}"
    else
        echo "  HDK:     未检测到 (${HDK_DRIVER_PATH}/driver 不存在)"
        echo "  [WARNING] 请安装 HDK 驱动或设置 ENPU_ASCEND_DRIVER_PATH=<HDK安装路径>，否则 acl-client 编译失败。"
    fi
}

function check_env() {
    echo "========== 环境检查 =========="

    check_command go
    check_command gcc
    check_command cmake
    check_command make
    check_command patch
    check_command objcopy
    check_command docker
    check_command helm

    echo "  Go:       $(go version 2>/dev/null | awk '{print $3}')"
    echo "  Docker:   $(docker --version)"
    echo "  Helm:     $(helm version --short 2>/dev/null)"
    echo "  PLATFORM: ${PLATFORM}"

    find_cann_env_script
    if [ -n "${CANN_ENV_SCRIPT}" ]; then
        echo "  CANN:     ${CANN_ENV_SCRIPT}"
    else
        echo "  CANN:     未检测到 set_env.sh"
        echo "  [WARNING] 请设置 ASCEND_HOME_PATH 或安装 CANN，否则 acl-client 编译失败。"
    fi

    check_hdk_env

    echo "  OUTPUT:   ${OUTPUT_DIR}"
    echo "  IMAGES:   ${IMAGE_OUTPUT_DIR}"
    echo "=============================="
}

# ==================== CANN 环境加载 ====================

function load_cann_env() {
    find_cann_env_script
    if [ -n "${CANN_ENV_SCRIPT}" ]; then
        source "${CANN_ENV_SCRIPT}"
    else
        echo "[WARNING] 未找到 CANN set_env.sh，跳过环境加载"
    fi
}

# ==================== 单组件二进制构建 ====================

function build_acl_client() {
    echo ">>> 构建 acl-client ..."
    load_cann_env
    bash "${SCRIPT_PATH}/build_acl_client.sh"
    echo ">>> acl-client 构建完成"
}

function build_npu_device_plugin() {
    echo ">>> 构建 npu-device-plugin ..."
    bash "${SCRIPT_PATH}/build_npu_device_plugin.sh"
    echo ">>> npu-device-plugin 构建完成"
}

function build_xpu_exporter() {
    echo ">>> 构建 xpu-exporter ..."
    bash "${SCRIPT_PATH}/build_xpu_exporter.sh"
    echo ">>> xpu-exporter 构建完成"
}

function build_volcano_all() {
    echo ">>> 构建 volcano 组件 (scheduler / controller-manager / webhook-manager) ..."

    # 三个 volcano 脚本共用一次完整编译，用 vc-scheduler 脚本编译后统一拷贝产物
    local work_dir="${SCRIPT_PATH}"
    local ci_dir
    ci_dir=$(cd "$(dirname "$(dirname "${work_dir}")")" && pwd)
    local volcano_top_dir="${ci_dir}/volcano_build/volcano"
    local volcano_base_path="${volcano_top_dir}/pkg/scheduler/plugins/volcano-xpu-plugin/"

    bash "${SCRIPT_PATH}/build_vc_scheduler.sh"

    if [ -f "${volcano_base_path}/output/vc-controller-manager" ]; then
        cp -P --remove-destination -r "${volcano_base_path}/output/vc-controller-manager" "${OUTPUT_DIR}/"
    fi

    if [ -f "${volcano_base_path}/output/vc-webhook-manager" ]; then
        cp -P --remove-destination -r "${volcano_base_path}/output/vc-webhook-manager" "${OUTPUT_DIR}/"
    fi

    if [ -f "${volcano_top_dir}/installer/dockerfile/webhook-manager/gen-admission-secret.sh" ]; then
        cp -P --remove-destination -r "${volcano_top_dir}/installer/dockerfile/webhook-manager/gen-admission-secret.sh" "${OUTPUT_DIR}/"
    fi

    # 下载 kubectl
    local os_type="arm64"
    if [[ $(uname -m) == "x86_64" ]]; then
        os_type="amd64"
    fi
    if [ ! -f "${OUTPUT_DIR}/kubectl" ]; then
        wget --no-check-certificate -P "${OUTPUT_DIR}" -q \
            "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/of-v1.33.1/bin/linux/${os_type}/kubectl"
    fi

    echo ">>> volcano 组件构建完成"
}

# 可用组件列表（volcano 为别名，一次构建以下三个 volcano 组件）
ALL_COMPONENTS=("acl-client" "npu-device-plugin" "xpu-exporter" \
                "vc-scheduler" "vc-controller-manager" "vc-webhook-manager")

# ==================== 镜像构建 ====================

# chart 组件镜像的仓库地址，需与 charts/vnpu/values.yaml 的 basic.swr_addr 保持一致
CHART_IMAGE_REGISTRY="cr.openfuyao.cn/openfuyao/vnpu"

# 组件名 -> "Dockerfile名 镜像tar基础名 镜像名 版本"
declare -A IMAGE_MAP=(
    ["acl-client"]="acl-client-update.Dockerfile acl-client-update ${CHART_IMAGE_REGISTRY}/acl-client-update 1.0.0"
    ["npu-device-plugin"]="npu-device-plugin.Dockerfile npu-device-plugin ${CHART_IMAGE_REGISTRY}/npu-device-plugin 1.0.0"
    ["xpu-exporter"]="xpu-exporter.Dockerfile xpu-exporter ${CHART_IMAGE_REGISTRY}/xpu-exporter 1.0.0"
    ["vc-controller-manager"]="vc-controller-manager.Dockerfile vc-controller-manager vc-controller-manager 1.15.0"
    ["vc-scheduler"]="vc-scheduler.Dockerfile vc-scheduler vc-scheduler 1.15.0"
    ["vc-webhook-manager"]="vc-webhook-manager.Dockerfile vc-webhook-manager vc-webhook-manager 1.15.0"
)

# ==================== chart 打包 ====================

# 打包 charts/vnpu 为 tgz（版本取自 Chart.yaml），输出到 ci/output/chart/
function build_chart_package() {
    echo ">>> 打包 helm chart ..."
    mkdir -p "${OUTPUT_DIR}/chart"
    # 删除旧包，避免 helm package 因目标文件已存在而失败，保证重复执行幂等
    rm -f "${OUTPUT_DIR}/chart"/vnpu-*.tgz
    helm package "${TOP_DIR}/charts/vnpu" -d "${OUTPUT_DIR}/chart"
    echo ">>> chart 包已输出到 ${OUTPUT_DIR}/chart/"
}

function build_image() {
    local component=$1
    local mapping="${IMAGE_MAP[$component]:-}"
    if [ -z "${mapping}" ]; then
        echo "[ERROR] 组件 '${component}' 没有对应的镜像定义"
        return 1
    fi

    local dockerfile tar_basename image_name version
    # shellcheck disable=SC2086
    read -r dockerfile tar_basename image_name version <<< "${mapping}"

    local tag="${image_name}:${version}"
    local tar_path="${IMAGE_OUTPUT_DIR}/${tar_basename}-${version}-${PLATFORM}.tar"

    echo ">>> 构建镜像 ${tag} ..."

    docker build \
        --build-arg LOCAL_ARTIFACTS=true \
        -f "${SCRIPT_DIR}/pipeline/Dockerfile/${dockerfile}" \
        -t "${tag}" \
        "${TOP_DIR}"

    mkdir -p "${IMAGE_OUTPUT_DIR}"
    docker save -o "${tar_path}" "${tag}"
    echo ">>> 镜像 ${tag} 已保存到 ${tar_path}"
}

function build_images() {
    local targets=("$@")
    echo "===== 镜像构建 ====="
    for c in "${targets[@]}"; do
        build_image "${c}"
    done
    echo ""
    echo "镜像 tar 目录: ${IMAGE_OUTPUT_DIR}"
    ls -la "${IMAGE_OUTPUT_DIR}/"
}

# ==================== 单组件二进制构建调度 ====================

function build_component_single() {
    local name=$1
    case "${name}" in
        acl-client)         build_acl_client ;;
        npu-device-plugin)  build_npu_device_plugin ;;
        xpu-exporter)       build_xpu_exporter ;;
    esac
}

# ==================== 主流程 ====================

function main() {
    # 单独打chart包：仅需要helm，不清理已有构建产物
    if [ "$#" -eq 1 ] && [ "$1" = "chart" ]; then
        check_command helm
        echo "===== chart 包构建 ====="
        build_chart_package
        return
    fi

    check_env

    # 清理旧产物
    rm -rf "${OUTPUT_DIR}"
    mkdir -p "${OUTPUT_DIR}"

    if [ $# -eq 0 ]; then
        # 无参数：构建全部
        echo "===== 本地全量构建 ====="
        build_acl_client
        build_npu_device_plugin
        build_xpu_exporter
        build_volcano_all
        build_chart_package

        echo ""
        echo "二进制构建完成。产物目录: ${OUTPUT_DIR}"
        ls -la "${OUTPUT_DIR}/"

        echo ""
        build_images "${ALL_COMPONENTS[@]}"
        return
    fi

    local image_targets=()
    for arg in "$@"; do
        case "${arg}" in
            chart)
                build_chart_package
                ;;
            acl-client|npu-device-plugin|xpu-exporter)
                build_component_single "${arg}"
                image_targets+=("${arg}")
                ;;
            volcano)
                build_volcano_all
                image_targets+=("vc-scheduler" "vc-controller-manager" "vc-webhook-manager")
                ;;
            *)
                echo "[ERROR] 未知组件: ${arg}"
                echo "可用组件: acl-client npu-device-plugin xpu-exporter volcano chart"
                exit 1
                ;;
        esac
    done

    echo ""
    echo "二进制构建完成。产物目录: ${OUTPUT_DIR}"
    ls -la "${OUTPUT_DIR}/"

    echo ""
    build_images "${image_targets[@]}"
}

main "$@"
