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

source ./log.sh
source ./consts.sh
source ./utils.sh

echo "online and offline Repository is : $REPO"
REGISTRY="${REPO%/}"
echo "registry is : ${REGISTRY}"
echo "openfuyao version is : ${OPENFUYAO_VERSION}"
#REGISTRY="${FUYAO_RGISTRY}"
#REGISTRY="${REGISTRY%/}"

BUSY_BOX_REPOSITORY="${REGISTRY}/busybox"
OAUTH_CERTS_EXPIRATION_TIME="1752000h"
ENABLE_HTTPS="true"
OPENFUYAO_REPO="${FUYAO_REPO}"

echo "init node is : $HOST_IP"
# 检查是否是离线安装,修改chart包拉取的地址
if [ "$OFFLINE_INSTALL" = "true" ]; then
    echo "检测到离线安装模式，设置 OPENFUYAO_REPO 为 http://引导节点：38080"
    OPENFUYAO_REPO="http://${HOST_IP}:38080"
else
    echo "在线安装模式，使用原有的 OPENFUYAO_REPO 值: ${OPENFUYAO_REPO}"
fi

OS=$(echo `uname`|tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac


function download_charts_with_retry() {
    local chart_name=$1
    local chart_version=$2

    if [ ! -d "${CHART_PATH}" ]; then
        mkdir -p "${CHART_PATH}"
    fi

    local cur_path=$(pwd)
    cd "${CHART_PATH}"  || fatal_log "Failed to change directory to ${CHART_PATH}"
    if [ -f "${chart_name}-${chart_version}.tgz" ]; then
        info_log "${chart_name}-${chart_version}.tgz already exists, skip downloading"
        cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
        return
    fi
    info_log "Downloading ${chart_name} chart"

    local attempts=0
    while [ $attempts -lt 3 ]; do
        if [ "$OFFLINE_INSTALL" = "true" ]; then
            helm fetch "${chart_name}" --repo "${OPENFUYAO_REPO}" --version "${chart_version}"
        else
            helm fetch "${FUYAO_REPO}/${chart_name}" --version "${chart_version}"
        fi
        if [ $? -eq 0 ]; then
            info_log "Successfully downloaded ${chart_name} chart"
            break
        else
            ((attempts++))
            sleep 2
        fi
    done

    if [ $attempts -eq 3 ]; then
        fatal_log "Failed to download $chart_name after 3 attempts."
    fi

    cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
    return
}

function create_service_cert() {
    local service_name=$1
    local dns_1=$2
    local dns_2=$3
    local dns_3=$4
    local dns_4=$5

    mkdir -p "${FUYAO_CERTS_PATH}/${service_name}"
    cd "${FUYAO_CERTS_PATH}/${service_name}" || fatal_log "failed to change directory to ${FUYAO_CERTS_PATH}/${service_name}"

    # 生成业务Pod私钥
    openssl genrsa -out "${service_name}".key 4096

    # 创建CSR配置文件 mypod-csr.conf
    cat > "${service_name}"-csr.conf <<EOF
[ req ]
default_bits       = 4096
prompt             = no
default_md         = sha256
distinguished_name = dn

[ dn ]
CN = ${dns_4}

[ v3_req ]
keyUsage = critical, keyEncipherment, dataEncipherment, digitalSignature, keyAgreement
extendedKeyUsage = serverAuth, clientAuth
subjectAltName = @alt_names

[ alt_names ]
DNS.1 = ${dns_1}
DNS.2 = ${dns_2}
DNS.3 = ${dns_3}
DNS.4 = ${dns_4}
EOF

    # 使用配置文件生成CSR
    openssl req -new -key ${service_name}.key -out ${service_name}.csr -config ${service_name}-csr.conf
    # 使用CA签署业务Pod证书
    openssl x509 -req -in ${service_name}.csr -CA ../ca.crt -CAkey ../ca.key -CAcreateserial -out ${service_name}.crt -days 1095 -sha256 -extensions v3_req -extfile ${service_name}-csr.conf
    cd ../..
}

function install_oauth_server() {
    info_log "install oauth_server"
    local signing_key="$1"
    local encryption_key="$2"
    local jwt_private_key="$3"
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_RELEASE_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"

    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${OAUTH_SERVER}" "${OAUTH_SERVER}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_oauth_server_enable_https "$signing_key" "$encryption_key" "$jwt_private_key"
    else
        install_oauth_server_disable_https "$signing_key" "$encryption_key" "$jwt_private_key"
    fi
}

function install_oauth_server_disable_https() {
    local signing_key="$1"
    local encryption_key="$2"
    local jwt_private_key="$3"
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_RELEASE_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"

    info_log "start installing oauth_server disable https"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_server_chart_path}" ]; then
        fatal_log "oauth_server chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.httpServerConfig.enableHttps=false
        --set config.loginSessionConfig.signingKey="${signing_key}"
        --set config.loginSessionConfig.encryptionKey="${encryption_key}"
        --set config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}"
        --set config.loginSessionConfig.name="idpLogin_bke"
        --set config.loginSessionConfig.csrfCookieName="csrf_bke"
        --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-server"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        )
    fi

    helm install "${OAUTH_SERVER_RELEASE_NAME}" \
        "${oauth_server_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --kubeconfig /root/.kube/config \
        "${SET_ARGS[@]}"
}

function install_oauth_server_enable_https() {
    local signing_key="$1"
    local encryption_key="$2"
    local jwt_private_key="$3"
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_RELEASE_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"

    info_log "start installing oauth_server enable https"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_server_chart_path}" ]; then
        fatal_log "oauth_server chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.httpServerConfig.enableHttps=true
        --set config.loginSessionConfig.signingKey="${signing_key}"
        --set config.loginSessionConfig.encryptionKey="${encryption_key}"
        --set config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}"
        --set config.loginSessionConfig.name="idpLogin_bke"
        --set config.loginSessionConfig.csrfCookieName="csrf_bke"
        --set-file config.httpServerConfig.tlsCert="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.crt"
        --set-file config.httpServerConfig.tlsKey="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.key"
        --set-file config.httpServerConfig.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
        --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-server"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        )
    fi

    helm install "${OAUTH_SERVER_RELEASE_NAME}" \
        "${oauth_server_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --kubeconfig /root/.kube/config \
        "${SET_ARGS[@]}"
}

function wait_oauth_webhook_ready() {
    info_log "Waiting for oauth-webhook to be ready"

    local timeout="300s"
    if kubectl wait --for=condition=ready pod \
        -l "app.kubernetes.io/instance=${OAUTH_WEBHOOK_RELEASE_NAME}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --timeout="${timeout}"; then
        info_log "oauth-webhook is ready"
        return 0
    fi

    info_log "oauth-webhook was not found by app.kubernetes.io/instance label, trying app label"
    if kubectl wait --for=condition=ready pod \
        -l "app=${OAUTH_WEBHOOK_RELEASE_NAME}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --timeout="${timeout}"; then
        info_log "oauth-webhook is ready"
        return 0
    fi

    error_log "oauth-webhook did not become ready within ${timeout}"
    return 1
}

function install_oauth_webhook() {
    local jwt_private_key="$1"
    local oauth_webhook_chart_path="./${CHART_PATH}/${OAUTH_WEBHOOK_CHART_NAME}-${OAUTH_WEBHOOK_CHART_VERSION}.tgz"

    # 检查 chart 是否存在
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_webhook_chart_path}" ]; then
        fatal_log "oauth_webhook chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.JWTPrivateKey="${jwt_private_key}"
        --set images.core.tag="${OAUTH_WEBHOOK_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-webhook"
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        )
    fi

    helm install "${OAUTH_WEBHOOK_RELEASE_NAME}" \
        "${oauth_webhook_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --kubeconfig /root/.kube/config \
        "${SET_ARGS[@]}"
}

function is_crd_status_ready() {
    # 设置要检查的 CRD 名称
    CRD_NAME="$1"

    # 最大等待时间（秒）
    MAX_WAIT_TIME=10800
    # 每次检查的间隔（秒）
    CHECK_INTERVAL=10

    # 初始化已等待时间
    elapsed_time=0

    # 循环检查 CRD 状态
    while true; do
        # 获取 CRD 的状态信息
        CRD_STATUS=$(kubectl get crd $CRD_NAME -o jsonpath='{.status.conditions[?(@.type=="Established")].status}')

        # 检查 CRD 是否就绪
        if [ "$CRD_STATUS" == "True" ]; then
            info_log "CRD $CRD_NAME is ready."
            break
        else
            info_log "CRD $CRD_NAME is not ready. Waiting for $CHECK_INTERVAL seconds..."
        fi

        # 增加已等待时间
        elapsed_time=$((elapsed_time + CHECK_INTERVAL))

        # 检查是否超过最大等待时间
        if [ $elapsed_time -ge $MAX_WAIT_TIME ]; then
            error_log "Exceeded maximum wait time of $MAX_WAIT_TIME seconds. Exiting."
            exit 1
        fi

        sleep $CHECK_INTERVAL
    done
}

function install_user_management_operator() {
    # 检查是否已安装
    if [ -n "${USER_MANAGEMENT_OPERATOR_INSTALLED}" ]; then
        info_log "user_management_operator has been installed, skip installation"
        return
    fi

    info_log "Start installing user_management_operator"

    local user_management_operator_chart_path="./${CHART_PATH}/${USER_MANAGEMENT_OPERATOR_CHART_NAME}-${USER_MANAGEMENT_OPERATOR_CHART_VERSION}.tgz"

    # 检查chart文件是否存在
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${user_management_operator_chart_path}" ]; then
        fatal_log "user_management_operator chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set images.core.tag="${USER_MANAGEMENT_OPERATOR_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/user-management-operator"
        )
    fi

    helm install "${USER_MANAGEMENT_OPERATOR_RELEASE_NAME}" \
        "${user_management_operator_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function create_default_user() {
    if kubectl get users admin >/dev/null 2>&1; then
        info_log "default user admin already exists"
        return
    fi

    info_log "create default user admin"

    local crd_name="users.users.openfuyao.com"
    local checks=150
    local sleep_duration=2
    for ((i=1; i<=checks; i++)); do
        if kubectl get crd "$crd_name" &> /dev/null; then
            break
        else
            info_log "CRD '$crd_name' 不存在，等待 $sleep_duration 秒后重试..."
            sleep $sleep_duration
        fi
    done

    is_crd_status_ready "${crd_name}"
    kubectl apply -f ./resource/user-manager/default-user.yaml
    info_log "default user admin created"
}

function ensure_oauth_keys() {
    if [ ! -f signing_key.txt ]; then
        signing_key=$(openssl rand -base64 32)
        echo "signing_key: ${signing_key}"
        echo "signing_key: ${signing_key}" > signing_key.txt
    else
        signing_key=$(cut -d ':' -f2- signing_key.txt | sed 's/^ //')
    fi

    if [ ! -f encryption_key.txt ]; then
        encryption_key=$(openssl rand -base64 32)
        echo "encryption_key: ${encryption_key}"
        echo "encryption_key: ${encryption_key}" > encryption_key.txt
    else
        encryption_key=$(cut -d ':' -f2- encryption_key.txt | sed 's/^ //')
    fi

    if [ ! -f jwt_private_key.txt ]; then
        jwt_private_key=$(openssl rand -base64 64 | tr -d '\n')
        echo "jwt_private_key: ${jwt_private_key}"
        echo "jwt_private_key: ${jwt_private_key}" > jwt_private_key.txt
    else
        jwt_private_key=$(cut -d ':' -f2- jwt_private_key.txt | sed 's/^ //')
    fi
}

function prepare_oauth_charts() {
    download_charts_with_retry "${OAUTH_SERVER_RELEASE_NAME}" "${OAUTH_SERVER_CHART_VERSION}"
    download_charts_with_retry "${OAUTH_WEBHOOK_CHART_NAME}" "${OAUTH_WEBHOOK_CHART_VERSION}"
    download_charts_with_retry "${USER_MANAGEMENT_OPERATOR_CHART_NAME}" "${USER_MANAGEMENT_OPERATOR_CHART_VERSION}"
}

INSTALL_STAGE="${1:-all}"
#
ENABLE_HTTPS="true"
#
generate_var
#
ensure_oauth_keys
prepare_oauth_charts

case "${INSTALL_STAGE}" in
    webhook)
        install_oauth_webhook "${jwt_private_key}"
        wait_oauth_webhook_ready
        info_log "Completing the installation of oauth-webhook"
        ;;
    server)
        install_oauth_server "${signing_key}" "${encryption_key}" "${jwt_private_key}"
        info_log "Completing the installation of oauth-server"
        install_user_management_operator
        info_log "Completing the installation of user-management-operator"
        create_default_user
        info_log "Completing the creation of default-user"
        ;;
    all)
        install_oauth_webhook "${jwt_private_key}"
        wait_oauth_webhook_ready
        info_log "Completing the installation of oauth-webhook"
        install_oauth_server "${signing_key}" "${encryption_key}" "${jwt_private_key}"
        info_log "Completing the installation of oauth-server"
        install_user_management_operator
        info_log "Completing the installation of user-management-operator"
        create_default_user
        info_log "Completing the creation of default-user"
        ;;
    *)
        fatal_log "unknown install stage: ${INSTALL_STAGE}"
        ;;
esac
