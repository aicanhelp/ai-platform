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
source ../../utils/consts.sh

# openfuyao管理面命名空间
OPENFUYAO_SYSTEM_NAMESPACE="openfuyao-system"
SESSION_SECRET_NAMESPACE="session-secret"
INGRESS_NGINX_NAMESPACE="ingress-nginx"

MONITOR_NAMESPACE="monitoring"

# image tag
CONSOLE_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
OAUTH_SERVER_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
OAUTH_WEBHOOK_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
MONITORING_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
CONSOLE_WEBSITE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
MARKETPLACE_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
APPLICATION_MANAGEMENT_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
PLUGIN_MANAGEMENT_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
USER_MANAGEMENT_OPERATOR_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
HARBOR_IMAGE_TAG="$LOCAL_HARBOR_IMAGE_TAG"
OAUTH_PROXY_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
WEB_TERMINAL_SERVICE_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"
KUBECTL_OPENFUYAO_IMAGE_TAG="$OPENFUYAO_IMAGE_TAG"

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

# oauth-webhook release name
OAUTH_WEBHOOK_RELEASE_NAME="oauth-webhook"
# oauth-server release name
OAUTH_SERVER_RELEASE_NAME="oauth-server"
# console-website release name
CONSOLE_WEBSITE_RELEASE_NAME="console-website"
# monitoring-service release name
MONITORING_SERVICE_RELEASE_NAME="monitoring-service"
# console-service release name
CONSOLE_SERVICE_RELEASE_NAME="console-service"
# marketplace_service release name
MARKETPLACE_SERVICE_RELEASE_NAME="marketplace-service"
# application-management-service release name
APPLICATION_MANAGEMENT_SERVICE_RELEASE_NAME="application-management-service"
# plugin-management-service release name
PLUGIN_MANAGEMENT_SERVICE_RELEASE_NAME="plugin-management-service"
# user-management-operator release name
USER_MANAGEMENT_OPERATOR_RELEASE_NAME="user-management-operator"
# local-harbor release name
HARBOR_RELEASE_NAME="local-harbor"
# web-terminal-service release name
WEB_TERMINAL_SERVICE_RELEASE_NAME="web-terminal-service"

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

CONSOLE_SERVICE="console-service"
CONSOLE_WEBSITE="console-website"
MARKETPLACE_SERVICE="marketplace-service"
APPLICATION_MANAGEMENT_SERVICE="application-management-service"
PLUGIN_MANAGEMENT_SERVICE="plugin-management-service"
LOCAL_HARBOR="local-harbor"
OAUTH_SERVER="oauth-server"
OAUTH_WEBHOOK="oauth-webhook"
MONITORING_SERVICE="monitoring-service"
INGRESS_NGINX_CONTROLLER="ingress-nginx-controller"
PROMETHEUS="prometheus"
METRICS_SERVER="metrics-server"
LABEL_KEY="fuyao-harbor-local-install-node"
INGRESS_NGINX_TLS_SECRET="ingress-nginx-tls"
INGRESS_NGINX_FRONT_TLS_SECRET="ingress-nginx-front-tls"

FUYAO_WEBHOOK_PATH="/etc/kubernetes/webhook"
FUYAO_CERTS_PATH="pki"
DNS_3_SUFFIX="svc"
DNS_4_SUFFIX="svc.cluster.local"
OPENFUYAO_SYSTEM_ROOT_CA_SECRET="openfuyao-system-root-ca"

LOCAL_HARBOR_HOST="https://local-harbor.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local"
OAUTH_SERVER_HOST="https://oauth-server.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:9096"
CONSOLE_SERVICE_HOST="https://console-service.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:443"
MONITORING_HOST="https://monitoring-service.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:443"
MONITORING_HOST_HTTP="http://monitoring-service.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:80"
CONSOLE_WEBSITE_HOST="https://console-website.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:80"
CONSOLE_WEBSITE_HOST_HTTP="http://console-website.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local:80"
