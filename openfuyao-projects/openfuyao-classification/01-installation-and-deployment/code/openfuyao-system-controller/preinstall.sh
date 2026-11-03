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

source ./log.sh
source ./consts.sh

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${CONSOLE_SERVICE}"; then
    info_log "${CONSOLE_SERVICE} has been installed"
    CONSOLE_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${CONSOLE_WEBSITE}"; then
    info_log "${CONSOLE_WEBSITE} has been installed"
    CONSOLE_WEBSITE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${MONITORING_SERVICE}"; then
    info_log "${MONITORING_SERVICE} has been installed"
    MONITORING_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${LOCAL_HARBOR}"; then
    info_log "${LOCAL_HARBOR} has been installed"
    LOCAL_HARBOR_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${OAUTH_SERVER}"; then
    info_log "${OAUTH_SERVER} has been installed"
    OAUTH_SERVER_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${OAUTH_WEBHOOK}"; then
    info_log "${OAUTH_WEBHOOK} has been installed"
    OAUTH_WEBHOOK_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${MARKETPLACE_SERVICE}"; then
    info_log "${MARKETPLACE_SERVICE} has been installed"
    MARKETPLACE_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${APPLICATION_MANAGEMENT_SERVICE}"; then
    info_log "${APPLICATION_MANAGEMENT_SERVICE} has been installed"
    APPLICATION_MANAGEMENT_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${PLUGIN_MANAGEMENT_SERVICE}"; then
    info_log "${PLUGIN_MANAGEMENT_SERVICE} has been installed"
    PLUGIN_MANAGEMENT_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${USER_MANAGEMENT_OPERATOR}"; then
    info_log "${USER_MANAGEMENT_OPERATOR} has been installed"
    USER_MANAGEMENT_OPERATOR_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${WEB_TERMINAL_SERVICE}"; then
    info_log "${WEB_TERMINAL_SERVICE} has been installed"
    WEB_TERMINAL_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${INSTALLER_SERVICE}"; then
    info_log "${INSTALLER_SERVICE} has been installed"
    INSTALLER_SERVICE_INSTALLED="true"
fi

if kubectl get pod -n "${OPENFUYAO_SYSTEM_NAMESPACE}" | grep -q "${INSTALLER_WEBSITE}"; then
    info_log "${INSTALLER_WEBSITE} has been installed"
    INSTALLER_WEBSITE_INSTALLED="true"
fi

if kubectl get pod -n "${MONITOR_NAMESPACE}" | grep -q "${PROMETHEUS}"; then
    info_log "${PROMETHEUS} has been installed"
    KUBE_PROMETHEUS_INSTALLED="true"
fi

if kubectl get pod -n kube-system | grep -q "${METRICS_SERVER}"; then
    info_log "${PROMETHEUS} has been installed"
    METRICS_SERVER_INSTALLED="true"
fi

if kubectl get pod -n "${INGRESS_NGINX_NAMESPACE}" | grep -q "${INGRESS_NGINX_CONTROLLER}"; then
    info_log "${INGRESS_NGINX_CONTROLLER} has been installed"
    INGRESS_NGINX_CONTROLLER_INSTALLED="true"
fi
