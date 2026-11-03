#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

FUYAO_REPO="https://helm.openfuyao.cn/_core"
FUYAO_ADDON_REPO="https://helm.openfuyao.cn"
OPENFUYAO_CHART_VERSION="0.0.0-latest"
OPENFUYAO_ADDON_CHART_VERSION="0.0.0-latest"
LOCAL_HARBOR_CHART_VERSION="1.11.4"

# oauth-webhook chart 版本
OAUTH_WEBHOOK_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# oauth-server chart 版本
OAUTH_SERVER_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# console-website chart 版本
CONSOLE_WEBSITE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# monitoring-service chart 版本
MONITORING_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# console-service chart 版本
CONSOLE_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# marketplace_service chart 版本
MARKETPLACE_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# application-management-service chart 版本
APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# plugin-management-service chart 版本
PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# user-management-operator chart 版本
USER_MANAGEMENT_OPERATOR_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# local-harbor chart 版本
LOCAL_HARBOR_CHART_VERSION="$LOCAL_HARBOR_CHART_VERSION"
# web-terminal-service chart 版本
WEB_TERMINAL_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# installer-service chart 版本
INSTALLER_SERVICE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"
# installer-website chart 版本
INSTALLER_WEBSITE_CHART_VERSION="$OPENFUYAO_CHART_VERSION"

# volcano
VOLCANO_CONFIG_SERVICE_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# ray
RAY_PACKAGE_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# colocation
COLOCATION_PACKAGE_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# npu-operator
NPU_OPERATOR_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# logging
LOGGING_PACKAGE_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# multi-cluster
MULTI_CLUSTER_SERVICE_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"
# monitoring-dashboard一个大包
MONITORING_DASHBOARD_CHART_VERSION="$OPENFUYAO_ADDON_CHART_VERSION"

# oauth-webhook chart name
OAUTH_WEBHOOK_CHART_NAME="oauth-webhook"
# oauth-server chart name
OAUTH_SERVER_CHART_NAME="oauth-server"
# console-website chart name
CONSOLE_WEBSITE_CHART_NAME="console-website"
# monitoring-service chart name
MONITORING_SERVICE_CHART_NAME="monitoring-service"
# console-service chart name
CONSOLE_SERVICE_CHART_NAME="console-service"
# marketplace-service chart name
MARKETPLACE_SERVICE_CHART_NAME="marketplace-service"
# application-management-service chart name
APPLICATION_MANAGEMENT_SERVICE_CHART_NAME="application-management-service"
# plugin-management-service chart name
PLUGIN_MANAGEMENT_SERVICE_CHART_NAME="plugin-management-service"
# user-management-operator chart name
USER_MANAGEMENT_OPERATOR_CHART_NAME="user-management-operator"
# harbor chart name
HARBOR_CHART_NAME="harbor"
# web-terminal-service chart name
WEB_TERMINAL_SERVICE_CHART_NAME="web-terminal-service"
# installer-service chart name
INSTALLER_SERVICE_CHART_NAME="installer-service"
# installer-website chart name
INSTALLER_WEBSITE_CHART_NAME="installer-website"

# volcano
VOLCANO_CONFIG_SERVICE_CHART_NAME="volcano-config-service"
# ray
RAY_PACKAGE_CHART_NAME="ray-package"
# colocation
COLOCATION_PACKAGE_CHART_NAME="colocation-package"
# npu-operator
NPU_OPERATOR_CHART_NAME="npu-operator"
# logging
LOGGING_PACKAGE_CHART_NAME="logging-package"
# multi-cluster
MULTI_CLUSTER_SERVICE_CHART_NAME="multi-cluster-service"
# monitoring-dashboard一个大包
MONITORING_DASHBOARD_CHART_NAME="monitoring-dashboard"

# 使用关联数组，key为chart名称，value为chart版本
declare -A chart_map
chart_map["${OAUTH_WEBHOOK_CHART_NAME}"]="${OAUTH_WEBHOOK_CHART_VERSION}"
chart_map["${OAUTH_SERVER_CHART_NAME}"]="${OAUTH_SERVER_CHART_VERSION}"
chart_map["${HARBOR_CHART_NAME}"]="${LOCAL_HARBOR_CHART_VERSION}"
chart_map["${CONSOLE_WEBSITE_CHART_NAME}"]="${CONSOLE_WEBSITE_CHART_VERSION}"
chart_map["${MONITORING_SERVICE_CHART_NAME}"]="${MONITORING_SERVICE_CHART_VERSION}"
chart_map["${CONSOLE_SERVICE_CHART_NAME}"]="${CONSOLE_SERVICE_CHART_VERSION}"
chart_map["${MARKETPLACE_SERVICE_CHART_NAME}"]="${MARKETPLACE_SERVICE_CHART_VERSION}"
chart_map["${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}"]="${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}"
chart_map["${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}"]="${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}"
chart_map["${USER_MANAGEMENT_OPERATOR_CHART_NAME}"]="${USER_MANAGEMENT_OPERATOR_CHART_VERSION}"
chart_map["${WEB_TERMINAL_SERVICE_CHART_NAME}"]="${WEB_TERMINAL_SERVICE_CHART_VERSION}"
chart_map["${INSTALLER_SERVICE_CHART_NAME}"]="${INSTALLER_SERVICE_CHART_VERSION}"
chart_map["${INSTALLER_WEBSITE_CHART_NAME}"]="${INSTALLER_WEBSITE_CHART_VERSION}"

declare -A addon_chart_map
addon_chart_map["${VOLCANO_CONFIG_SERVICE_CHART_NAME}"]="${VOLCANO_CONFIG_SERVICE_CHART_VERSION}"
addon_chart_map["${RAY_PACKAGE_CHART_NAME}"]="${RAY_PACKAGE_CHART_VERSION}"
addon_chart_map["${COLOCATION_PACKAGE_CHART_NAME}"]="${COLOCATION_PACKAGE_CHART_VERSION}"
addon_chart_map["${NPU_OPERATOR_CHART_NAME}"]="${NPU_OPERATOR_CHART_VERSION}"
addon_chart_map["${LOGGING_PACKAGE_CHART_NAME}"]="${LOGGING_PACKAGE_CHART_VERSION}"
addon_chart_map["${MULTI_CLUSTER_SERVICE_CHART_NAME}"]="${MULTI_CLUSTER_SERVICE_CHART_VERSION}"
addon_chart_map["${MONITORING_DASHBOARD_CHART_NAME}"]="${MONITORING_DASHBOARD_CHART_VERSION}"

for chart_name in "${!chart_map[@]}"; do
    echo "Start downloading chart ${chart_name}:${chart_map[$chart_name]}"
    attempts=0
    while [ $attempts -lt 3 ]; do
        helm fetch ${chart_name} --repo ${FUYAO_REPO} --version "${chart_map[$chart_name]}"
        if [ $? -eq 0 ]; then
            break
        else
            ((attempts++))
            sleep 2
        fi
    done

    if [ $attempts -eq 3 ]; then
        echo "Failed to download $chart_name after 3 attempts."
        exit 1
    fi

    # 检查保存到本地的chart是否存在
    chart_file="${chart_name}-${chart_map[$chart_name]}.tgz"
    if [ -f "$chart_file" ]; then
        echo "Chart $chart_file saved to $chart_file successfully."
    else
        echo "Failed to save chart $chart_file to $chart_file."
        exit 1
    fi
done

# 下载扩展组件
for chart_name in "${!addon_chart_map[@]}"; do
    echo "Start downloading chart ${chart_name}:${addon_chart_map[$chart_name]}"
    attempts=0
    while [ $attempts -lt 3 ]; do
        helm fetch ${chart_name} --repo ${FUYAO_ADDON_REPO} --version "${addon_chart_map[$chart_name]}"
        if [ $? -eq 0 ]; then
            break
        else
            ((attempts++))
            sleep 2
        fi
    done

    if [ $attempts -eq 3 ]; then
        echo "Failed to download $addon_chart_name after 3 attempts."
        exit 1
    fi

    # 检查保存到本地的chart是否存在
    chart_file="${chart_name}-${addon_chart_map[$chart_name]}.tgz"
    if [ -f "${chart_file}" ]; then
        temp_dir=$(mktemp -d)
        echo "temp ${temp_dir}"
        tar -xzf "${chart_file}" -C "${temp_dir}" || {
            rm -rf "${temp_dir}"
            exit 1
        }
        rm -f "${chart_file}"

        find "${temp_dir}" -iname "values.yaml" | xargs sed -i 's/cr.openfuyao.cn\/openfuyao/deploy.bocloud.k8s:40443\/kubernetes/g'

        find "${temp_dir}" -iname "values.yaml" | xargs sed -i 's/cr.openfuyao.cn\/docker.io/deploy.bocloud.k8s:40443\/kubernetes/g'

        pkg="${temp_dir}"/"${chart_name}"
        helm package "${pkg}"

        rm -rf ${temp_dir}
        echo "Chart $chart_file saved to $chart_file successfully."
    else
        echo "Failed to save chart $chart_file to $chart_file."
        exit 1
    fi
done
