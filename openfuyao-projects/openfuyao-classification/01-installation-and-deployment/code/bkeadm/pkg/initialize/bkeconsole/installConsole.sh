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
#REGISTRY="${REGISTRY%/}"   #移除最后的/. 如果有的话
BUSY_BOX_REPOSITORY="${REGISTRY}/busybox"
ENABLE_HTTPS="true"

# chart包拉取地址
OPENFUYAO_REPO="${FUYAO_REPO}"

echo "init node is : $HOST_IP"
# 检查是否是离线安装,修改chart包拉取的地址
if [ "$OFFLINE_INSTALL" = "true" ]; then
    echo "检测到离线安装模式，设置 OPENFUYAO_REPO 为 http://引导节点：38080"
    OPENFUYAO_REPO="http://${HOST_IP}:38080"
else
    echo "在线安装模式，使用原有的 OPENFUYAO_REPO 值: ${OPENFUYAO_REPO}"
fi

OAUTH_CERTS_EXPIRATION_TIME="1752000h"


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

function install_yq() {
    if command -v yq >/dev/null 2>&1; then
        info_log "yq already installed"
        return
    fi

    info_log "Start installing yq"
    sudo mv -f ${BKE_FILE_PATH}/yq_linux_${ARCH} /usr/local/bin/yq
    sudo chmod +x /usr/local/bin/yq
    info_log "success install yq"
}

function install_jq() {
    if command -v jq >/dev/null 2>&1; then
        info_log "jq already installed"
        return
    fi

    info_log "Start installing jq"
    sudo mv -f ${BKE_FILE_PATH}/jq-linux-${ARCH} /usr/local/bin/jq
    sudo chmod +x /usr/local/bin/jq
    info_log "success install jq"
}

function install_helm() {
    if command -v helm >/dev/null 2>&1; then
        info_log "helm already installed"
        return
    fi

    info_log "Start installing helm"
    sudo tar -zxvf ${BKE_FILE_PATH}/helm-v3.14.2-"${OS}"-${ARCH}.tar.gz
    sudo mv "${OS}"-$ARCH/helm /usr/local/bin/helm
    info_log "success install helm"
}

function exec_install_tools() {
    local name=$1

    if command -v "$name" >/dev/null 2>&1; then
        info_log "$name already installed"
        return
    fi
    info_log "Start installing $name"
    sudo mv -f ${BKE_FILE_PATH}/"${name}_1.6.4_linux_${ARCH}" /usr/local/bin/"$name"
    sudo chmod +x /usr/local/bin/"$name"
    info_log "success install $name"
}

function install_cfssl() {
    exec_install_tools cfssl
    exec_install_tools cfssl-certinfo
    exec_install_tools cfssljson
}

function create_root_ca() {
    # 创建根证书目录
    if [ ! -d "${FUYAO_CERTS_PATH}" ]; then
        mkdir -p "${FUYAO_CERTS_PATH}"
    fi

    cd "${FUYAO_CERTS_PATH}"  || fatal_log "failed to change directory to ${FUYAO_CERTS_PATH}"

    # todo 这里先判断是否有这个secret，如果有从里面把证书读取出来，写到指定路径
    if [  -f "ca.key" ] && [ -f "ca.crt" ]; then
        info_log "fuyao ca.key already exists"
        create_root_ca_secret
        cd ..
        return
    fi

    # 生成CA私钥
    openssl genrsa -out ca.key 4096

    # 生成自签名的CA证书，有效期10年
    openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt -subj "/C=US/ST=California/L=San Francisco/O=MyCompany/OU=MyOrg/CN=MyRootCA"
    create_root_ca_secret
    cd ..
}

function create_root_ca_secret() {
    if [ "${ENABLE_HTTPS}" != "true" ]; then
        info_log "disable https"
        return
    fi

    if kubectl get secret -n "$OPENFUYAO_SYSTEM_NAMESPACE" | grep "$OPENFUYAO_SYSTEM_ROOT_CA_SECRET"; then
        info_log "openfuyao-system ca secret already exists"
        return
    fi

    info_log "create openfuyao-system ca secret"
    cat > openfuyao-system-ca-secret.yaml <<EOF
        apiVersion: v1
        data:
          ca.crt: |
            $(cat ca.crt | base64 | tr -d '\n')
          ca.key: |
            $(cat ca.key | base64 | tr -d '\n')
        kind: Secret
        metadata:
          name: $OPENFUYAO_SYSTEM_ROOT_CA_SECRET
          namespace: $OPENFUYAO_SYSTEM_NAMESPACE
EOF
    sudo kubectl apply -f openfuyao-system-ca-secret.yaml
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

function create_ingress_nginx_tls_secret() {
    if [ "${ENABLE_HTTPS}" != "true" ]; then
        info_log "disable https"
        return
    fi

    info_log "create ingress-nginx-tls secret"
    if kubectl get secret -n "${INGRESS_NGINX_NAMESPACE}" | grep "${INGRESS_NGINX_TLS_SECRET}"; then
        info_log "ingress-nginx-tls secret already exists"
        return
    fi

    create_service_cert "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_3_SUFFIX}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_4_SUFFIX}"
    kubectl create secret generic "${INGRESS_NGINX_TLS_SECRET}" --namespace "${INGRESS_NGINX_NAMESPACE}" \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    kubectl create secret generic "${INGRESS_NGINX_FRONT_TLS_SECRET}" --namespace "${INGRESS_NGINX_NAMESPACE}" \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    info_log "ingress-nginx tls secret created"
}

function is_ingress_nginx_running() {
    if ! kubectl get ns | grep "${INGRESS_NGINX_NAMESPACE}"; then
        info_log "not deploy ingress-nginx"
        return 1
    fi

    local count=1
    local is_running=1

    while [ $count -le 60 ]
    do
        info_log "waiting times $count"
        status_list=$(kubectl get pod -n "${INGRESS_NGINX_NAMESPACE}" | awk 'NR > 1 {print $3}')
        is_ok=0

        while read -r status; do
            if [ "${status}" == "Running" ] || [ "${status}" == "Completed" ]; then
                info_log "${status}"
            else
                info_log "ingress nginx pod status is abnormal"
                sleep 10
                is_ok=1
                break
            fi
        done <<< "$status_list"

        if [ $is_ok -eq 0 ]; then
            is_running=0
            kubectl delete -A ValidatingWebhookConfiguration ingress-nginx-admission
            break
        fi

        count=$((count + 1))
    done

    return $is_running
}

function install_console_website() {
    if [ -n "${CONSOLE_WEBSITE_INSTALLED}" ]; then
        info_log "bke_console_website has been installed, skip installation"
        return
    fi
    info_log "installing bke_console_website"

    download_charts_with_retry "${BKE_CONSOLE_WEBSITE_CHART_NAME}" "${BKE_CONSOLE_WEBSITE_CHART_VERSION}"

     if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${CONSOLE_WEBSITE}" "${CONSOLE_WEBSITE}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_console_website_enable_https
    else
        install_console_website_disable_https
    fi
    info_log "Completing the installation of bke_console_website"
}

function install_console_website_disable_https() {
    info_log "Start installing console_website without https"
    local bke_console_website_chart_path="./${CHART_PATH}/${BKE_CONSOLE_WEBSITE_CHART_NAME}-${BKE_CONSOLE_WEBSITE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${bke_console_website_chart_path}" ]; then
        fatal_log "console_website chart not found"
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableTLS=false
        --set images.core.tag="${BKE_CONSOLE_WEBSITE_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/bke-console-website"
        )
    fi

    helm install "${BKE_CONSOLE_WEBSITE_RELEASE_NAME}" \
        "${bke_console_website_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_website_enable_https() {
    info_log "Start installing bke_console_website with https"
    local bke_console_website_chart_path="./${CHART_PATH}/${BKE_CONSOLE_WEBSITE_CHART_NAME}-${BKE_CONSOLE_WEBSITE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${bke_console_website_chart_path}" ]; then
        fatal_log "bke_console_website chart not found"
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableTLS=true
        --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.crt"
        --set-file config.tlsKey="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.key"
        --set-file config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
        --set images.core.tag="${BKE_CONSOLE_WEBSITE_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/bke-console-website"
        )
    fi

    helm install "${BKE_CONSOLE_WEBSITE_RELEASE_NAME}" \
        "${bke_console_website_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_service() {
    if [ -n "${CONSOLE_SERVICE_INSTALLED}" ]; then
        info_log "bke_console_service has been installed, skip installation"
        return
    fi

    info_log "Start installing bke_console_service"
    create_namespace "${SESSION_SECRET_NAMESPACE}"
    is_ingress_nginx_running
    if [ $? -eq 0 ]; then
        info_log "ingress nginx pod status is normal, go on."
    else
        fatal_log "ingress nginx pod status is abnormal"
    fi
    info_log "waiting dns pod running..."
    kubectl wait -n kube-system --for=condition=ready pod -l k8s-app=kube-dns

    download_charts_with_retry "${BKE_CONSOLE_SERVICE_CHART_NAME}" "${BKE_CONSOLE_SERVICE_CHART_VERSION}"

    # 创建证书
    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${CONSOLE_SERVICE}" "${CONSOLE_SERVICE}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_console_service_enable_https
    else
        install_console_service_disable_https
    fi
    info_log "Completing the installation of bke-console-service"
}

function install_console_service_disable_https() {
    info_log "Start installing bke_console_service without https"
    local bke_console_service_chart_path="./${CHART_PATH}/${BKE_CONSOLE_SERVICE_CHART_NAME}-${BKE_CONSOLE_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${bke_console_service_chart_path}" ]; then
        fatal_log "bke_console_service chart not found"
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set serverHost.monitoring="${MONITORING_HOST_HTTP}"
        --set serverHost.consoleWebsite="${CONSOLE_WEBSITE_HOST_HTTP}"
        --set symmetricKey.tokenKey="$(openssl rand -base64 32)"
        --set symmetricKey.secretKey="$(openssl rand -base64 32)"
        --set config.enableHttps=false
        --set images.core.tag="${BKE_CONSOLE_SERVICE_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/bke-console-service"
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${BKE_CONSOLE_SERVICE_RELEASE_NAME}" \
        "${bke_console_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_service_enable_https() {
    info_log "Start installing console_service with https"
    local bke_console_service_chart_path="./${CHART_PATH}/${BKE_CONSOLE_SERVICE_CHART_NAME}-${BKE_CONSOLE_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${bke_console_service_chart_path}" ]; then
        fatal_log "bke_console_service chart not found"
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set ingress.secretName="${INGRESS_NGINX_NAMESPACE}/${INGRESS_NGINX_TLS_SECRET}"
        --set symmetricKey.tokenKey="$(openssl rand -base64 32)"
        --set symmetricKey.secretKey="$(openssl rand -base64 32)"
        --set config.enableHttps=true
        --set serverHost.localHarbor="${LOCAL_HARBOR_HOST}"
        --set serverHost.oauthServer="${OAUTH_SERVER_HOST}"
        --set serverHost.consoleService="${CONSOLE_SERVICE_HOST}"
        --set serverHost.consoleWebsite="${CONSOLE_WEBSITE_HOST}"
        --set serverHost.monitoring="${MONITORING_HOST_HTTP}"
        --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${CONSOLE_SERVICE}/${CONSOLE_SERVICE}.crt"
        --set-file config.tlsKey="./${FUYAO_CERTS_PATH}/${CONSOLE_SERVICE}/${CONSOLE_SERVICE}.key"
        --set-file config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
        --set images.core.tag="${BKE_CONSOLE_SERVICE_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/bke-console-service"
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${BKE_CONSOLE_SERVICE_RELEASE_NAME}" \
        "${bke_console_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_ingress_nginx() {
    if [ -n "${INGRESS_NGINX_CONTROLLER_INSTALLED}" ]; then
        info_log "ingress_nginx has been installed, skip installation"
        return
    fi

    info_log "Start installing ingress-nginx"
    create_namespace "${INGRESS_NGINX_NAMESPACE}"
    create_ingress_nginx_tls_secret

    YAML_FILE="./resource/ingress-nginx/ingress-nginx.yaml"
    update_image_tag_from_cm "controller" "$YAML_FILE" "cm.${OPENFUYAO_VERSION}" "openfuyao-patch" "${OPENFUYAO_VERSION}"
    update_image_tag_from_cm "kube-webhook-certgen" "$YAML_FILE" "cm.${OPENFUYAO_VERSION}" "openfuyao-patch" "${OPENFUYAO_VERSION}"

    if [ -n "$REPO" ]; then
      sudo sed -i "s|${FUYAO_THIRD_REGISTRY}|${REGISTRY}|g" ./resource/ingress-nginx/ingress-nginx.yaml
    fi

    kubectl create -f ./resource/ingress-nginx/ingress-nginx.yaml
    info_log "success install ingress-nginx"
}

function create_ingress_nginx_tls_secret() {
    if [ "${ENABLE_HTTPS}" != "true" ]; then
        info_log "disable https"
        return
    fi

    info_log "create ingress-nginx-tls secret"
    if kubectl get secret -n "${INGRESS_NGINX_NAMESPACE}" | grep "${INGRESS_NGINX_TLS_SECRET}"; then
        info_log "ingress-nginx-tls secret already exists"
        return
    fi

    create_service_cert "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_3_SUFFIX}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_4_SUFFIX}"
    kubectl create secret generic "${INGRESS_NGINX_TLS_SECRET}" --namespace "${INGRESS_NGINX_NAMESPACE}" \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    kubectl create secret generic "${INGRESS_NGINX_FRONT_TLS_SECRET}" --namespace "${INGRESS_NGINX_NAMESPACE}" \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    info_log "ingress-nginx tls secret created"
}

function create_namespace() {
    local namespace=$1
    if kubectl get namespace "$namespace" >/dev/null 2>&1; then
        info_log "namespace $namespace already exists"
    else
        kubectl create namespace "$namespace"
        info_log "create namespace $namespace success"
    fi
}

function install_plugin_management_service() {
    if [ -n "${PLUGIN_MANAGEMENT_SERVICE_INSTALLED}" ]; then
        info_log "plugin_management_service has been installed, skip installation"
        return
    fi
    info_log "Start installing plugin_management_service"

    download_charts_with_retry "${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}" "${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}"
    local plugin_management_service_chart_path="./${CHART_PATH}/${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}-${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${plugin_management_service_chart_path}" ]; then
        fatal_log "plugin_management_service chart not found"
    fi

    if [ "${ENABLE_HTTPS}" == "true" ]; then
        install_plugin_management_service_disable_https
    else
        install_plugin_management_service_disable_https
    fi
    info_log "Completing the installation of plugin_management_service"
}

function install_plugin_management_service_disable_https() {
    info_log "disable https for plugin_management_service"

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableHttps=false
        --set enableOAuth=true
        --set images.core.tag="${PLUGIN_MANAGEMENT_SERVICE_IMAGE_TAG}"
        --set images.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/plugin-management-service"
            --set images.busyBox.repository="${BUSY_BOX_REPOSITORY}"
            --set images.oauthProxy.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    helm install "${PLUGIN_MANAGEMENT_SERVICE_RELEASE_NAME}" \
        "${plugin_management_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_installer_website() {
    # 检查是否存在指定的 Pod,两个都有，则继续安装
    if kubectl get pods -n cluster-system | grep -q "bke-controller-manager" && \
       kubectl get pods -n cluster-system | grep -q "capi-controller-manager"; then
        info_log "Pod bke-controller-manager and capi-controller-manager exist, go on..."

    else
        info_log "Pod bke-controller-manager 或 capi-controller-manager do not exist，exit..."
        return
    fi

    if [ -n "${INSTALLER_WEBSITE_INSTALLED}" ]; then
        info_log "installer_website has been installed, skip installation"
        return
    fi
    info_log "installing installer_website"


    download_charts_with_retry "${INSTALLER_WEBSITE_CHART_NAME}" "${INSTALLER_WEBSITE_CHART_VERSION}"

    install_installer_website_disable_https
    info_log "Completing the installation of installer_website"
}

function install_installer_website_disable_https() {
    info_log "Start installing installer_website without https"

    local installer_website_chart_path="./${CHART_PATH}/${INSTALLER_WEBSITE_CHART_NAME}-${INSTALLER_WEBSITE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${installer_website_chart_path}" ]; then
        fatal_log "installer_website chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableTLS=false
        --set images.core.tag="${INSTALLER_WEBSITE_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/installer-website"
        )
    fi

    helm install "${INSTALLER_WEBSITE_RELEASE_NAME}" \
        "${installer_website_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_installer_service() {

    if kubectl get pods -n cluster-system | grep -q "bke-controller-manager" && \
       kubectl get pods -n cluster-system | grep -q "capi-controller-manager"; then
        info_log "Pod bke-controller-manager and capi-controller-manager exist, go on..."

    else
        info_log "Pod bke-controller-manager 或 capi-controller-manager do not exist，exit..."
        return
    fi

    if [ -n "${INSTALLER_SERVICE_INSTALLED}" ]; then
        info_log "installer_service has been installed, skip installation"
        return
    fi
    info_log "Start installing installer_service"
    download_charts_with_retry "${INSTALLER_SERVICE_CHART_NAME}" "${INSTALLER_SERVICE_CHART_VERSION}"
    install_installer_service_disable_https
    info_log "Completing the installation of installer-service"
}

function install_installer_service_disable_https() {
    info_log "Start installing installer_service without https"

    # 定义chart包路径
    local installer_service_chart_path="./${CHART_PATH}/${INSTALLER_SERVICE_CHART_NAME}-${INSTALLER_SERVICE_CHART_VERSION}.tgz"

    # 确认chart包路径存在
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${installer_service_chart_path}" ]; then
        fatal_log "installer_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set images.core.tag="${INSTALLER_SERVICE_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/installer-service"
        )
    fi

    helm install "${INSTALLER_SERVICE_RELEASE_NAME}" \
        "${installer_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

install_yq
install_jq
install_helm
install_cfssl
generate_var

create_root_ca
create_namespace "${OPENFUYAO_SYSTEM_NAMESPACE}"
install_ingress_nginx
install_console_website
install_console_service
install_plugin_management_service
install_installer_website
install_installer_service
