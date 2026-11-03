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
source ./preinstall.sh
source ./utils.sh

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
        if [ "$IS_ONLINE" = "false" ]; then
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
    local installer_service_chart_path="./${CHART_PATH}/${INSTALLER_SERVICE_CHART_NAME}-${INSTALLER_SERVICE_CHART_VERSION}.tgz"
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

function install_console_website() {
    if [ -n "${CONSOLE_WEBSITE_INSTALLED}" ]; then
        info_log "console_website has been installed, skip installation"
        return
    fi
    info_log "installing console_website"

    download_charts_with_retry "${CONSOLE_WEBSITE_CHART_NAME}" "${CONSOLE_WEBSITE_CHART_VERSION}"

     if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${CONSOLE_WEBSITE}" "${CONSOLE_WEBSITE}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${CONSOLE_WEBSITE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_console_website_enable_https
    else
        install_console_website_disable_https
    fi
    info_log "Completing the installation of console_website"
}

function install_console_website_disable_https() {
    info_log "Start installing console_website without https"
    local console_website_chart_path="./${CHART_PATH}/${CONSOLE_WEBSITE_CHART_NAME}-${CONSOLE_WEBSITE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${console_website_chart_path}" ]; then
        fatal_log "console_website chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableTLS=false
        --set images.core.tag="${CONSOLE_WEBSITE_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/console-website"
        )
    fi

    helm install "${CONSOLE_WEBSITE_RELEASE_NAME}" \
        "${console_website_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_website_enable_https() {
    info_log "Start installing console_website with https"
    local console_website_chart_path="./${CHART_PATH}/${CONSOLE_WEBSITE_CHART_NAME}-${CONSOLE_WEBSITE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${console_website_chart_path}" ]; then
        fatal_log "console_website chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableTLS=true
        --set images.core.tag="${CONSOLE_WEBSITE_IMAGE_TAG}"
        --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.crt"
        --set-file config.tlsKey="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.key"
        --set-file config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/console-website"
        )
    fi

    helm install "${CONSOLE_WEBSITE_RELEASE_NAME}" \
        "${console_website_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_service() {
    if [ -n "${CONSOLE_SERVICE_INSTALLED}" ]; then
        info_log "console_service has been installed, skip installation"
        return
    fi

    info_log "Start installing console_service"
    create_namespace "${SESSION_SECRET_NAMESPACE}"
    is_ingress_nginx_running
    if [ $? -eq 0 ]; then
        info_log "ingress nginx pod status is normal, go on."
    else
        fatal_log "ingress nginx pod status is abnormal"
    fi
    info_log "waiting dns pod running..."
    kubectl wait -n kube-system --for=condition=ready pod -l k8s-app=kube-dns

    download_charts_with_retry "${CONSOLE_SERVICE_CHART_NAME}" "${CONSOLE_SERVICE_CHART_VERSION}"

    # 创建证书
    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${CONSOLE_SERVICE}" "${CONSOLE_SERVICE}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${CONSOLE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_console_service_enable_https
    else
        install_console_service_disable_https
    fi
    info_log "Completing the installation of console-service"
}

function install_console_service_disable_https() {
    info_log "Start installing console_service without https"
    local console_service_chart_path="./${CHART_PATH}/${CONSOLE_SERVICE_CHART_NAME}-${CONSOLE_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${console_service_chart_path}" ]; then
        fatal_log "console_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set images.core.tag="${CONSOLE_SERVICE_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        --set serverHost.monitoring="${MONITORING_HOST_HTTP}"
        --set serverHost.consoleWebsite="${CONSOLE_WEBSITE_HOST_HTTP}"
        --set symmetricKey.tokenKey="$(openssl rand -base64 32)"
        --set symmetricKey.secretKey="$(openssl rand -base64 32)"
        --set config.enableHttps=false
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/console-service"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${CONSOLE_SERVICE_RELEASE_NAME}" \
        "${console_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_console_service_enable_https() {
    info_log "Start installing console_service with https"
    local console_service_chart_path="./${CHART_PATH}/${CONSOLE_SERVICE_CHART_NAME}-${CONSOLE_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${console_service_chart_path}" ]; then
        fatal_log "console_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set ingress.secretName="${INGRESS_NGINX_NAMESPACE}/${INGRESS_NGINX_TLS_SECRET}"
        --set images.core.tag="${CONSOLE_SERVICE_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
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
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/console-service"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${CONSOLE_SERVICE_RELEASE_NAME}" \
        "${console_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_metrics_server() {
    if [ -n "${METRICS_SERVER_INSTALLED}" ]; then
        info_log "metrics-server has been installed, skip installation"
        return
    fi
    info_log "Start installing metrics-server"

    YAML_FILE="./resource/metrics-server/metrics-server.yaml"
    update_image_tag_from_cm "metrics-server" "$YAML_FILE"

    if [ -n "$REPO" ]; then
      sudo sed -i "s|${FUYAO_THIRD_REGISTRY}|${REGISTRY}|g" ./resource/metrics-server/metrics-server.yaml
    fi

    if [ -f "/etc/kubernetes/pki/front-proxy-ca.crt" ] && ! kubectl get secret front-proxy-ca-cert -n kube-system 2>/dev/null 1>/dev/null; then
        kubectl create secret generic front-proxy-ca-cert \
              --from-file=front-proxy-ca.crt=/etc/kubernetes/pki/front-proxy-ca.crt \
              --namespace="kube-system"
    fi

    kubectl apply -f ./resource/metrics-server/metrics-server.yaml
    info_log "Completing the installation of metrics-server"
}

function install_kube_prometheus() {
    if [ -n "${KUBE_PROMETHEUS_INSTALLED}" ]; then
        info_log "kube_prometheus has been installed, skip installation"
        return
    fi
    info_log "Start installing kube_prometheus"
    create_namespace "${MONITOR_NAMESPACE}"

    control_plane_nodes=$(kubectl get nodes -l node-role.kubernetes.io/control-plane -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    master_nodes=$(kubectl get nodes -l node-role.kubernetes.io/master -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    merged_nodes=$(echo -e "$control_plane_nodes\n$master_nodes" | sed '/^\s*$/d')
    if [ -z "$merged_nodes" ]; then
        fatal_log "not found control plane nodes"
    else
        unique_nodes=$(echo "$merged_nodes" | sort -u)
        info_log "control plane nodes: $unique_nodes"
    fi

    # 取首个节点名
    node_name=$(printf '%s\n' "$unique_nodes" | head -n1)
    info_log "get secret on node $node_name"
    kubectl label nodes "$node_name" openfuyao.io/"${node_name}"="${node_name}"

    pki_path=$(kubectl describe cm kubeadm-config -n  kube-system | grep certificatesDir | awk '{print $2}')
    if [ -z "$pki_path" ]; then
        info_log "pki_path is empty, use default value"
        pki_path="/etc/kubernetes/pki"
    fi

    if [ -d "${pki_path}/etcd" ]; then
        kubectl create secret generic etcd-certs \
              --from-file=ca.crt=${pki_path}/etcd/ca.crt \
              --from-file=etcd.crt=${pki_path}/etcd/peer.crt \
              --from-file=etcd.key=${pki_path}/etcd/peer.key \
              --namespace="${MONITOR_NAMESPACE}"
    else
      cat <<EOF > etce-certs-secret-${node_name}.yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: create-etcd-certs-secret-${node_name}
  namespace: ${MONITOR_NAMESPACE}
spec:
  template:
    spec:
      nodeSelector:
        openfuyao.io/${node_name}: ${node_name}
      containers:
      - name: create-etcd-certs-secret-${node_name}
        image: ${OPENFUYAO_SYSTEM_CONTROLLER_REPOSITORY}:${OPENFUYAO_IMAGE_TAG}
        imagePullPolicy: IfNotPresent
        command:
          - /bin/sh
          - -c
          - |
            set -e
            if [ ! -d "${pki_path}/etcd" ]; then
                echo "pki_path is not exist, skip creating etcd-certs secret"
                exit 1
            fi
            kubectl create secret generic etcd-certs \
                          --from-file=ca.crt=${pki_path}/etcd/ca.crt \
                          --from-file=etcd.crt=${pki_path}/etcd/peer.crt \
                          --from-file=etcd.key=${pki_path}/etcd/peer.key \
                          --namespace="${MONITOR_NAMESPACE}"
            echo "etcd-certs secret created successfully"
        volumeMounts:
        - name: etc
          mountPath: /etc
        - name: usr
          mountPath: /usr
        - name: root
          mountPath: /root
      restartPolicy: OnFailure
      volumes:
      - name: etc
        hostPath:
          path: /etc
      - name: usr
        hostPath:
          path: /usr
      - name: root
        hostPath:
          path: /root
EOF
      kubectl apply -f ./etce-certs-secret-${node_name}.yaml
      kubectl wait --for=condition=complete job/create-etcd-certs-secret-${node_name} -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" --timeout=1800s
    fi

    kubectl apply -f ./resource/kube-prometheus/kubernetes-components-service/
    info_log "Completing the installation of kube-prometheus service"

    # 处理  ./resource/kube-prometheus/nodeExporter-daemonset.yaml
    YAML_FILE="./resource/kube-prometheus/nodeExporter-daemonset.yaml"
    update_image_tag_from_cm "node-exporter" "$YAML_FILE"
    update_image_tag_from_cm "kube-rbac-proxy" "$YAML_FILE"

    # 处理  ./resource/kube-prometheus/alertmanager-alertmanager.yaml
    YAML_FILE="./resource/kube-prometheus/alertmanager-alertmanager.yaml"
    update_image_tag_from_cm "alertmanager" "$YAML_FILE"

    # 处理  ./resource/kube-prometheus/prometheus-prometheus.yaml
    YAML_FILE="./resource/kube-prometheus/prometheus-prometheus.yaml"
    update_image_tag_from_cm "prometheus" "$YAML_FILE"

    # 处理  ./resource/kube-prometheus/prometheusOperator-deployment.yaml
    YAML_FILE="./resource/kube-prometheus/prometheusOperator-deployment.yaml"
    update_image_tag_from_cm "prometheus-operator" "$YAML_FILE"
    update_image_tag_from_cm "kube-rbac-proxy" "$YAML_FILE"

    if PROMETHEUS_CONFIG_RELOADER_TAG=$(get_cm_image_tag "patch-config" "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" "prometheus-config-reloader"); then
        info_log "Print prometheus-config-reloader tag: $PROMETHEUS_OPERATOR_TAG"
        # use input prometheus-config-reloader image tag
        YAML_FILE="./resource/kube-prometheus/prometheusOperator-deployment.yaml"
        sed -i "s|\(--prometheus-config-reloader=[^:]*\):[^[:space:]]*|\1:$PROMETHEUS_CONFIG_RELOADER_TAG|" "$YAML_FILE"
    else
        info_log "No update needed for prometheus-config-reloader tag as it was not found."
    fi

    # 处理  ./resource/kube-prometheus/blackboxExporter-deployment.yaml
    YAML_FILE="./resource/kube-prometheus/blackboxExporter-deployment.yaml"
    update_image_tag_from_cm "blackbox-exporter" "$YAML_FILE"
    update_image_tag_from_cm "configmap-reload" "$YAML_FILE"
    update_image_tag_from_cm "kube-rbac-proxy" "$YAML_FILE"

    # 处理  ./resource/kube-prometheus/kubeStateMetrics-deployment.yaml
    YAML_FILE="./resource/kube-prometheus/kubeStateMetrics-deployment.yaml"
    update_image_tag_from_cm "kube-state-metrics" "$YAML_FILE"
    update_image_tag_from_cm "kube-rbac-proxy" "$YAML_FILE"


    if [ -n "$REPO" ]; then
      # 自定义仓库需要进行repo替换
      local yamlFiles=(
          "./resource/kube-prometheus/nodeExporter-daemonset.yaml"
          "./resource/kube-prometheus/alertmanager-alertmanager.yaml"
          "./resource/kube-prometheus/prometheus-prometheus.yaml"
          "./resource/kube-prometheus/prometheusOperator-deployment.yaml"
          "./resource/kube-prometheus/blackboxExporter-deployment.yaml"
          "./resource/kube-prometheus/kubeStateMetrics-deployment.yaml"
      )
      for yamlFile in "${yamlFiles[@]}" ; do
          sed -i  "s|${FUYAO_THIRD_REGISTRY}|${REGISTRY}|g" "${yamlFile}"
      done
    fi

    kubectl apply --server-side -f ./resource/kube-prometheus/setup
    kubectl wait \
    	--for condition=Established \
    	--all CustomResourceDefinition \
    	--namespace="${MONITOR_NAMESPACE}"
    kubectl apply -f ./resource/kube-prometheus/
    info_log "Completing the installation of kube-prometheus"
}

function install_monitoring_service() {
    if [ -n "${MONITORING_SERVICE_INSTALLED}" ]; then
        info_log "monitoring_service has been installed, skip installation"
        return
    fi
    info_log "installing monitoring_service"
    download_charts_with_retry "${MONITORING_SERVICE_CHART_NAME}" "${MONITORING_SERVICE_CHART_VERSION}"

    # 创建证书
    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${MONITORING_SERVICE}" "${MONITORING_SERVICE}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_monitoring_service_disable_https
    else
        install_monitoring_service_disable_https
    fi
    info_log "Completing the installation of monitoring-service"
}

function install_monitoring_service_enable_https() {
    local monitoring_service_chart_path="./${CHART_PATH}/${MONITORING_SERVICE_CHART_NAME}-${MONITORING_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${monitoring_service_chart_path}" ]; then
        fatal_log "monitoring_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set enableOAuth=true
        --set config.httpServerConfig.enableHttps=true
        --set images.busybox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.busybox.repository="${BUSY_BOX_REPOSITORY}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/monitoring-service"
            --set images.oauth.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    SET_ARGS+=(
        --set-file config.httpServerConfig.tlsCert="./${FUYAO_CERTS_PATH}/${MONITORING_SERVICE}/${MONITORING_SERVICE}.crt"
        --set-file config.httpServerConfig.tlsKey="./${FUYAO_CERTS_PATH}/${MONITORING_SERVICE}/${MONITORING_SERVICE}.key"
        --set-file config.httpServerConfig.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
    )

    helm install "${MONITORING_SERVICE_RELEASE_NAME}" \
        "${monitoring_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_monitoring_service_disable_https() {
    info_log "installing monitoring_service without https"
    local monitoring_service_chart_path="./${CHART_PATH}/${MONITORING_SERVICE_CHART_NAME}-${MONITORING_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${monitoring_service_chart_path}" ]; then
        fatal_log "monitoring_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set enableOAuth=true
        --set config.httpServerConfig.enableHttps=false
        --set images.busybox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.busybox.repository="${BUSY_BOX_REPOSITORY}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/monitoring-service"
            --set images.oauth.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    helm install "${MONITORING_SERVICE_RELEASE_NAME}" \
        "${monitoring_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_helm_chart_repository() {
    if [ -n "${LOCAL_HARBOR_INSTALLED}" ]; then
        info_log "local_harbor has been installed, skip installation"
        return
    fi
    info_log "Start installing helm_chart_repository"

    # 获取 master 节点，local-harbor 仅允许部署在 master 节点
    control_plane_nodes=$(kubectl get nodes -l node-role.kubernetes.io/control-plane -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    master_nodes=$(kubectl get nodes -l node-role.kubernetes.io/master -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    merged_nodes=$(echo -e "$control_plane_nodes\n$master_nodes" | sed '/^\s*$/d')
    if [ -z "$merged_nodes" ]; then
        fatal_log "not found control plane nodes"
    else
        unique_nodes=$(echo "$merged_nodes" | sort -u)
        info_log "control plane nodes: $unique_nodes"
    fi

    # 取首个 master 节点作为 local-harbor 部署节点
    harbor_node=$(printf '%s\n' "$unique_nodes" | head -n1)
    info_log "local-harbor will be installed on master node $harbor_node"
    kubectl label nodes "$harbor_node" openfuyao.io/"${harbor_node}"="${harbor_node}"

    # create initContainer job yaml
    cat <<EOF > harbor-init.yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: init-harbor
      namespace: ${OPENFUYAO_SYSTEM_NAMESPACE}
    spec:
      template:
        spec:
          nodeSelector:
            openfuyao.io/${harbor_node}: ${harbor_node}
          tolerations:
            - operator: "Exists"
          containers:
          - name: init-harbor-container
            image: ${BUSY_BOX_REPOSITORY}:${BUSY_BOX_IMAGE_TAG}
            command:
              - sh
              - -c
              - |
                set -e
                if [ ! -d /data/harbor/registry ]; then mkdir -p /data/harbor/registry; fi
                chown -R 10000:10000 /data/harbor/registry && chmod -R 700 /data/harbor/registry

                if [ ! -d /data/harbor/chartmuseum ]; then mkdir -p /data/harbor/chartmuseum; fi
                chown -R 10000:10000 /data/harbor/chartmuseum && chmod -R 700 /data/harbor/chartmuseum

                if [ ! -d /data/harbor/redis ]; then mkdir -p /data/harbor/redis; fi
                chown -R 999:999 /data/harbor/redis && chmod -R 700 /data/harbor/redis

                if [ ! -d /data/harbor/jobservice ]; then mkdir -p /data/harbor/jobservice; fi
                chown -R 10000:10000 /data/harbor/jobservice && chmod -R 700 /data/harbor/jobservice

                if [ ! -d /data/harbor/database ]; then mkdir -p /data/harbor/database; fi
                chown -R 999:999 /data/harbor/database && chmod -R 700 /data/harbor/database

                echo "Initialization complete"
            volumeMounts:
            - name: data
              mountPath: /data
          restartPolicy: OnFailure
          volumes:
          - name: data
            hostPath:
              path: /data
EOF

    info_log "making Harbor persistence directory..."
    # apply initContainer job yaml
    kubectl apply -f harbor-init.yaml

    # wait job to be finished
    info_log "waiting job finished..."
    kubectl wait --for=condition=complete --timeout=1800s job/init-harbor -n "${OPENFUYAO_SYSTEM_NAMESPACE}"

    # check whether initContainer job succeed
    job_status=$(kubectl get job init-harbor -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -o jsonpath='{.status.succeeded}')
    if [ "$job_status" != "1" ]; then
      error_log "initContainer job failed，please check logs"
      kubectl logs -n "${OPENFUYAO_SYSTEM_NAMESPACE}" job/init-harbor
      exit 1
    fi

    # get initContainer job node name
    node_name=$(kubectl get pods --selector=job-name=init-harbor -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -o jsonpath='{.items[0].spec.nodeName}')
    # check whether initContainer job node name is valid
    if [ -z "$node_name" ]; then
        fatal_log "unable to retrieve initContainer job node name，built-in Harbor installation failed"
    fi

    info_log "install in node '$node_name'"
    kubectl label nodes "$node_name" $LABEL_KEY="$node_name"

    # delete initContainer job
    kubectl delete -f harbor-init.yaml

    # substitute harbor-local-values-x86.yaml's node selector for Node_Name
    sed -i "s/{NODE_NAME}/$node_name/g" ./resource/helm-chart-repository/harbor-local-values.yaml

    info_log "modify harbor pv pvc config"
    local harbor_pv_path="./resource/helm-chart-repository/harbor-local-pv.yaml"
    sudo yq e "select(di == 0).spec.capacity.storage = \"${HARBOR_REGISTRY_PV_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 1).spec.resources.requests.storage = \"${HARBOR_REGISTRY_PVC_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 2).spec.capacity.storage = \"${HARBOR_CHARTMUSEUM_PV_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 3).spec.resources.requests.storage = \"${HARBOR_CHARTMUSEUM_PVC_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 4).spec.capacity.storage = \"${HARBOR_REDIS_PV_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 5).spec.resources.requests.storage = \"${HARBOR_REDIS_PVC_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 6).spec.capacity.storage = \"${HARBOR_JOBSERVICE_PV_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 7).spec.resources.requests.storage = \"${HARBOR_JOBSERVICE_PVC_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 8).spec.capacity.storage = \"${HARBOR_DATABASE_PV_SIZE}\"" -i ${harbor_pv_path}
    sudo yq e "select(di == 9).spec.resources.requests.storage = \"${HARBOR_DATABASE_PVC_SIZE}\"" -i ${harbor_pv_path}

    # create corresponding pv & pvc
    info_log "creating pv & pvc..."
    kubectl apply -f ./resource/helm-chart-repository/harbor-local-pv.yaml

    sudo yq e '.expose.tls.enabled = false' -i ./resource/helm-chart-repository/harbor-local-values.yaml

    # install Harbor Helm chart
    info_log "install Harbor Helm chart ${HARBOR_CHART_VERSION}..."

    # use input harbor image tag
    YAML_FILE="./resource/helm-chart-repository/harbor-local-values.yaml"
    yq eval "(.. | select(has(\"tag\")).tag) |= \"$HARBOR_IMAGE_TAG\"" "$YAML_FILE" > "${YAML_FILE}.tmp" && mv -f "${YAML_FILE}.tmp" "$YAML_FILE"

    if [ -n "${REPO}" ]; then
        sudo sed -i "s|${FUYAO_RGISTRY}|${REGISTRY}|g" ./resource/helm-chart-repository/harbor-local-values.yaml
    fi
    sudo yq e ".harborAdminPassword = \"${HARBOR_ADMIN_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml
    sudo yq e ".registry.credentials.password = \"${HARBOR_REGISTRY_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml
    sudo yq e ".database.internal.password = \"${HARBOR_DATABASE_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml

    info_log "modify redis version..."
    # 提取 osImage
    OS_IMAGE=$(kubectl get node ${node_name} -o jsonpath='{.status.nodeInfo.osImage}')
    # 提取 architecture
    ARCH=$(kubectl get node ${node_name} -o jsonpath='{.status.nodeInfo.architecture}')
    KERNEL_VERSION=$(kubectl get node ${node_name} -o jsonpath='{.status.nodeInfo.kernelVersion}')
    info_log "OS Image: $OS_IMAGE and Architecture: $ARCH and Kernel Version: $KERNEL_VERSION"

    if [[ "${KERNEL_VERSION}" == 4.19* && "${ARCH}" == "arm64" ]]; then
        USED_REPO=${FUYAO_RGISTRY}
        if [ -n "${REPO}" ]; then
            USED_REPO=${REPO}
        fi
        USED_REPO=${USED_REPO%/}
        sudo yq e ".redis.internal.image.repository = \"${USED_REPO}/harbor/redis\" " -i ./resource/helm-chart-repository/harbor-local-values.yaml
        sudo yq e '.redis.internal.image.tag =  "v7.4.2"' -i ./resource/helm-chart-repository/harbor-local-values.yaml
        echo "yes"
    fi

    download_charts_with_retry "${HARBOR_CHART_NAME}" "${HARBOR_CHART_VERSION}"

    local harbor_chart_path="./${CHART_PATH}/${HARBOR_CHART_NAME}-${HARBOR_CHART_VERSION}.tgz"
    # 本地有chart文件则进行离线安装
    if [ -d "${CHART_PATH}" ] && [ -f "$harbor_chart_path" ]; then
        helm install "${HARBOR_RELEASE_NAME}" "$harbor_chart_path" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -f ./resource/helm-chart-repository/harbor-local-values.yaml
    else
        fatal_log "Harbor chart not found"
    fi

    if [[ "${KERNEL_VERSION}" == 4.19* && "${ARCH}" == "arm64" ]]; then
      # 等待 local-harbor-redis StatefulSet 对象被创建
      info_log "开始等待 local-harbor-redis StatefulSet 对象创建..."
      max_retries=24               # 最多重试次数
      retry_interval=5             # 重试间隔（秒）

      for ((i=1; i<=max_retries; i++)); do
          if kubectl get statefulset local-harbor-redis -n openfuyao-system >/dev/null 2>&1; then
              info_log "检测到 StatefulSet local-harbor-redis (尝试 $i/$max_retries)"
              break
          fi
          info_log "未检测到 StatefulSet，${retry_interval}s 后重试 (尝试 $i/$max_retries)…"
          sleep $retry_interval
      done

      # 超时判定
      if (( i > max_retries )); then
          error_log "ERROR: 等待 StatefulSet local-harbor-redis 创建超时"
      fi

      # Patch：给 Redis 容器追加启动参数
      info_log "开始对 local-harbor-redis 添加 Redis 启动参数..."
      kubectl patch statefulset -n openfuyao-system local-harbor-redis --type='json' -p='[
        {
          "op": "add",
          "path": "/spec/template/spec/containers/0/args",
          "value": [
            "--ignore-warnings",
            "ARM64-COW-BUG"
          ]
        }
      ]'
      if [[ $? -ne 0 ]]; then
          error_log "ERROR: patch 操作失败，请检查 StatefulSet 配置或参数路径"
      fi

      kubectl get statefulset -n openfuyao-system local-harbor-redis -o jsonpath='{.spec.template.spec.containers[0].args}'
      info_log "成功追加 Redis 启动参数：--ignore-warnings, ARM64-COW-BUG"

      NAMESPACE="openfuyao-system"

      echo "===== Harbor 组件重启 ====="
      echo "注意: 将按顺序重建Redis、Jobservice和Core组件"

      # 1. 重建Redis
      echo ""
      echo "步骤1: 重建Redis (StatefulSet)"
      kubectl delete pod -n $NAMESPACE local-harbor-redis-0 --force --grace-period=0
      echo "等待Redis启动..."
      sleep 10  # 初始等待

      # 等待Redis就绪
      while true; do
        STATUS=$(kubectl get pod -n $NAMESPACE local-harbor-redis-0 -o jsonpath='{.status.phase}' 2>/dev/null)
        if [ "$STATUS" = "Running" ]; then
          echo "Redis已运行"
          break
        else
          echo "等待Redis启动...当前状态: ${STATUS:-未知}"
          sleep 10
        fi
      done

      # 2. 重启Jobservice
      echo ""
      echo "步骤2: 重启Jobservice (Deployment)"
      kubectl rollout restart deployment -n $NAMESPACE local-harbor-jobservice
      echo "等待Jobservice重建..."
      sleep 10  # 初始等待

      # 等待Jobservice就绪
      while true; do
        READY=$(kubectl get pods -n $NAMESPACE -l component=jobservice -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
        if [ "$READY" = "True" ]; then
          echo "Jobservice已就绪"
          break
        else
          echo "等待Jobservice启动..."
          sleep 10
        fi
      done

      # 3. 重启Core
      echo ""
      echo "步骤3: 重启Core (Deployment)"
      kubectl rollout restart deployment -n $NAMESPACE local-harbor-core
      echo "等待Core重建..."
      sleep 10  # 初始等待

      # 等待Core就绪
      while true; do
        READY=$(kubectl get pods -n $NAMESPACE -l component=core -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
        if [ "$READY" = "True" ]; then
          echo "Core已就绪"
          break
        else
          echo "等待Core启动..."
          sleep 10
        fi
      done

      echo "Finish modify local-harbor!"

    fi

    info_log "built-in Harbor installed"
}

function generate_harbor_local_tls() {
    info_log "create local-harbor ca"
    create_service_cert "${LOCAL_HARBOR}" "${LOCAL_HARBOR}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
    kubectl create secret generic harbor-local-tls -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --from-file=tls.key=./"${FUYAO_CERTS_PATH}"/"${LOCAL_HARBOR}"/"${LOCAL_HARBOR}".key \
        --from-file=tls.crt=./"${FUYAO_CERTS_PATH}"/"${LOCAL_HARBOR}"/"${LOCAL_HARBOR}".crt \
    info_log "created local-harbor tls"
}

function install_oauth_webhook_and_oauth_server() {
    if [ -n "${OAUTH_WEBHOOK_INSTALLED}" ] && [ -n "${OAUTH_SERVER_INSTALLED}" ]; then
        info_log "oauth_webhook and oauth_server has been installed, skip installation"
        return
    fi
    info_log "Start installing oauth_webhook and oauth_server"

    generate_oauth_webhook_tls_cert

    is_ingress_nginx_running
    if [ $? -eq 0 ]; then
        info_log "ingress nginx pod status is normal, go on."
        sleep 10
    else
        fatal_log "ingress nginx pod status is abnormal"
    fi
    download_charts_with_retry "${OAUTH_SERVER_RELEASE_NAME}" "${OAUTH_SERVER_CHART_VERSION}"
    download_charts_with_retry "${OAUTH_WEBHOOK_CHART_NAME}" "${OAUTH_WEBHOOK_CHART_VERSION}"

    local signing_key=$(openssl rand -base64 32)
    local encryption_key=$(openssl rand -base64 32)
    local jwt_private_key=$(openssl rand -base64 64 | tr -d '\n')
    info_log "generate signing encryption jwt_private"
    if [ -z "${OAUTH_WEBHOOK_INSTALLED}" ] || [ -z "${OAUTH_SERVER_INSTALLED}" ]; then
        # 为保证oauth-webhook与oauth-server jwt_private_key 值一样，如果一个组件已经安装，那么先卸载再全部重新安装
        if [ -n "${OAUTH_WEBHOOK_INSTALLED}" ]; then
            info_log "uninstall oauth-webhook"
            helm uninstall "${OAUTH_WEBHOOK_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
        fi
        if [ -n "${OAUTH_SERVER_INSTALLED}" ]; then
            info_log "uninstall oauth-server"
            helm uninstall "${OAUTH_SERVER_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
        fi

        install_oauth_webhook "${jwt_private_key}"
        info_log "Completing the installation of oauth-webhook"
        wait_oauth_webhook_available
        modify_kubernetes_manifests
        if ! grep -q "authentication-token-webhook-config-file" /etc/kubernetes/manifests/kube-apiserver.yaml; then
            info_log "start modify kubernetes manifests"
        fi
        install_oauth_server "${signing_key}" "${encryption_key}" "${jwt_private_key}"
        info_log "Completing the installation of oauth-server"
    fi
    info_log "Completing the installation of oauth-server and oauth-webhook"
}

function modify_kubernetes_manifests() {
    info_log "modify kubernetes manifests"

    control_plane_nodes=$(kubectl get nodes -l node-role.kubernetes.io/control-plane -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    master_nodes=$(kubectl get nodes -l node-role.kubernetes.io/master -o custom-columns='NAME:.metadata.name' | awk 'NR > 1')
    merged_nodes=$(echo -e "$control_plane_nodes\n$master_nodes" | sed '/^\s*$/d')
    if [ -z "$merged_nodes" ]; then
        fatal_log "not found control plane nodes"
    else
        unique_nodes=$(echo "$merged_nodes" | sort -u)
        info_log "control plane nodes: $unique_nodes"
    fi

    node_count=$(echo "$unique_nodes" | wc -l)
    info_log "control plane node count: $node_count"
    # 如果有多个master节点，调度job去改
    save_webhook_config_yaml_to_cm
    modify_kubernetes_manifests_multi_node "${unique_nodes[@]}"
}

# 记录webhook-config.yaml到configmap中，方便调度job去获取
function save_webhook_config_yaml_to_cm() {
    local yaml_file_path="./resource/oauth-webhook/webhook-config.yaml"
    kubectl create configmap "${OAUTH_WEBHOOK_CONFIG_YAML_CM}" --from-file=$yaml_file_path -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}"
}

function modify_kubernetes_manifests_multi_node() {
    info_log "modify kubernetes manifests multi node"

    local nodes_name=$1
    local kube_apiserver_manifest_path="/etc/kubernetes/manifests/kube-apiserver.yaml"

    for node_name in $nodes_name; do
        info_log "modify kubernetes manifests on node $node_name"
        kubectl label nodes "$node_name" openfuyao.io/"${node_name}"="${node_name}"

        # create initContainer job yaml
        cat <<EOF > modify-${node_name}.yaml
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: modify-manifests-${node_name}
          namespace: ${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}
        spec:
          template:
            spec:
              nodeSelector:
                openfuyao.io/${node_name}: ${node_name}
              containers:
              - name: modify-manifests-${node_name}
                image: ${OPENFUYAO_SYSTEM_CONTROLLER_REPOSITORY}:${OPENFUYAO_IMAGE_TAG}
                imagePullPolicy: IfNotPresent
                command:
                  - /bin/sh
                  - -c
                  - |
                    if [ ! -f "${kube_apiserver_manifest_path}" ]; then
                        echo "kube-apiserver.yaml not found in ${kube_apiserver_manifest_path}"
                        exit 1
                    fi

                    if [ -f "/etc/kubernetes/pki/front-proxy-ca.crt" ] && ! kubectl get secret front-proxy-ca-cert -n kube-system 2>/dev/null 1>/dev/null; then
                        kubectl create secret generic front-proxy-ca-cert \
                              --from-file=front-proxy-ca.crt=/etc/kubernetes/pki/front-proxy-ca.crt \
                              --namespace="kube-system"
                    fi

                    echo "Start installing yq"
                    chmod +x /home/openfuyao-system/amd64-bin/yq_linux_amd64
                    chmod +x /home/openfuyao-system/arm64-bin/yq_linux_arm64
                    /home/openfuyao-system/arm64-bin/yq_linux_arm64 -h > /etc/kubernetes/yq_linux_arm64.log 2>&1
                    if grep -q "syntax error" /etc/kubernetes/yq_linux_arm64.log; then
                        echo "use yq_linux_amd64"
                        mv -f /home/openfuyao-system/amd64-bin/yq_linux_amd64 /usr/local/bin/yq
                    else
                        echo "use yq_linux_arm64"
                        mv -f /home/openfuyao-system/arm64-bin/yq_linux_arm64 /usr/local/bin/yq
                    fi
                    echo "yq installed"

                    echo "write cert to webhook directory"
                    mkdir -p "${FUYAO_WEBHOOK_PATH}"
                    kubectl get configmap ${OAUTH_WEBHOOK_CONFIG_YAML_CM} -n ${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE} -o jsonpath='{.data.webhook-config\.yaml}' > ${FUYAO_WEBHOOK_PATH}/webhook-config.yaml
                    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE} -o yaml | yq eval '.data."ca.crt"' | base64 -d > "${FUYAO_WEBHOOK_PATH}/ca.pem"
                    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE} -o yaml | yq eval '.data."tls.crt"' | base64 -d > "${FUYAO_WEBHOOK_PATH}/server.crt"
                    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE} -o yaml | yq eval '.data."tls.key"' | base64 -d > "${FUYAO_WEBHOOK_PATH}/server.key"
                    chmod 400 "${FUYAO_WEBHOOK_PATH}/ca.pem"
                    chmod 400 "${FUYAO_WEBHOOK_PATH}/server.crt"
                    chmod 400 "${FUYAO_WEBHOOK_PATH}/server.key"

                    echo "modify kube-apiserver.yaml"
                    if ! grep -q "authentication-token-webhook-config-file" ${kube_apiserver_manifest_path}; then
                        yq -i '.spec.containers[0].command += ["--authentication-token-webhook-config-file=/etc/webhook/webhook-config.yaml"]' ${kube_apiserver_manifest_path}
                        yq -i '.spec.containers[0].command += ["--authentication-token-webhook-cache-ttl=5m"]' ${kube_apiserver_manifest_path}
                        yq -i '.spec.containers[0].volumeMounts += [{"mountPath": "/etc/webhook", "name": "webhook-config", "readOnly": true}]' ${kube_apiserver_manifest_path}
                        yq -i '.spec.volumes += [{"hostPath": {"path": "/etc/kubernetes/webhook", "type": "DirectoryOrCreate"}, "name": "webhook-config"}]' ${kube_apiserver_manifest_path}
                    fi
                    yq -i '.spec += {"dnsPolicy": "ClusterFirstWithHostNet"}' ${kube_apiserver_manifest_path}
                    echo "Initialization complete"
                volumeMounts:
                - name: etc
                  mountPath: /etc
                - name: usr
                  mountPath: /usr
                - name: root
                  mountPath: /root
              restartPolicy: OnFailure
              volumes:
              - name: etc
                hostPath:
                  path: /etc
              - name: usr
                hostPath:
                  path: /usr
              - name: root
                hostPath:
                  path: /root
EOF

        # apply job yaml
        kubectl apply -f modify-${node_name}.yaml
        kubectl wait --for=condition=complete --timeout=1800s job/modify-manifests-${node_name} -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}"

        MAX_RETRIES=60
        RETRY_DELAY=5
        ATTEMPTS=0
        # 修改apiserver yaml后要等几分钟，apiserver才能启动，默认最多等待5分钟
        while [ $ATTEMPTS -lt $MAX_RETRIES ]; do
            kubectl get node
            if [ $? -eq 0 ]; then
                info_log "api-server has been restarted."
                break
            else
                info_log "api-server status is abnormal. Retrying in $RETRY_DELAY seconds..."
                sleep $RETRY_DELAY
                ((attempts++))
            fi
        done

        if [ $ATTEMPTS -eq $MAX_RETRIES ]; then
            error_log "Maximum retries reached. Command execution failed after $MAX_RETRIES attempts."
            exit 1
        fi

        # check whether job succeed
        job_status=$(kubectl get job modify-manifests-${node_name} -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" -o jsonpath='{.status.succeeded}')
        if [ "$job_status" != "1" ]; then
          error_log "failed to modify kubernetes manifests on node $node_name，please check logs"
          kubectl logs -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" job/modify-manifests-"${node_name}"
          exit 1
        fi
    done
}

function modify_kubernetes_manifests_single_node() {
    info_log "modify kubernetes manifests single node"

    local kube_apiserver_manifest_path="/etc/kubernetes/manifests/kube-apiserver.yaml"
    if [ ! -f "${kube_apiserver_manifest_path}" ]; then
        fatal_log "kube-apiserver.yaml not found in ${kube_apiserver_manifest_path}"
    fi

    # 拷贝证书与配置文件到webhook目录
    mkdir -p "${FUYAO_WEBHOOK_PATH}"
    cp -f ./resource/oauth-webhook/webhook-config.yaml "${FUYAO_WEBHOOK_PATH}"/webhook-config.yaml
    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" -o yaml | yq eval '.data."ca.crt"' | base64 --decode > "${FUYAO_WEBHOOK_PATH}/ca.pem"
    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" -o yaml | yq eval '.data."tls.crt"' | base64 --decode > "${FUYAO_WEBHOOK_PATH}/server.crt"
    kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n "${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" -o yaml | yq eval '.data."tls.key"' | base64 --decode > "${FUYAO_WEBHOOK_PATH}/server.key"
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/ca.pem"
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.crt"
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.key"

    info_log "modify kube-apiserver.yaml"
    sudo yq -i '.spec.containers[0].command += ["--authentication-token-webhook-config-file=/etc/webhook/webhook-config.yaml"]' ${kube_apiserver_manifest_path}
    sudo yq -i '.spec.containers[0].command += ["--authentication-token-webhook-cache-ttl=5m"]' ${kube_apiserver_manifest_path}
    sudo yq -i '.spec.containers[0].volumeMounts += [{"mountPath": "/etc/webhook", "name": "webhook-config", "readOnly": true}]' ${kube_apiserver_manifest_path}
    sudo yq -i '.spec += {"dnsPolicy": "ClusterFirstWithHostNet"}' ${kube_apiserver_manifest_path}
    sudo yq -i '.spec.volumes += [{"hostPath": {"path": "/etc/kubernetes/webhook", "type": "DirectoryOrCreate"}, "name": "webhook-config"}]' ${kube_apiserver_manifest_path}

    # 修改apiserver yaml后要等几分钟，apiserver才能启动，默认最多等待5分钟
    MAX_RETRIES=60
    RETRY_DELAY=5
    ATTEMPTS=0
    while [ $ATTEMPTS -lt $MAX_RETRIES ]; do
        kubectl get node
        if [ $? -eq 0 ]; then
            info_log "api-server has been restarted."
            break
        else
            info_log "api-server status is abnormal. Retrying in $RETRY_DELAY seconds..."
            sleep $RETRY_DELAY
            ((attempts++))
        fi
    done

    if [ $ATTEMPTS -eq $MAX_RETRIES ]; then
        fatal_log "Maximum retries reached. Command execution failed after $MAX_RETRIES attempts."
    fi
}

function install_oauth_webhook() {
    local jwt_private_key="$1"
    local oauth_webhook_chart_path="./${CHART_PATH}/${OAUTH_WEBHOOK_CHART_NAME}-${OAUTH_WEBHOOK_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_webhook_chart_path}" ]; then
        fatal_log "oauth_webhook chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.JWTPrivateKey="${jwt_private_key}"
        --set images.core.tag="${OAUTH_WEBHOOK_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-webhook"
        )
    fi

    helm install "${OAUTH_WEBHOOK_RELEASE_NAME}" \
        "${oauth_webhook_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function wait_oauth_webhook_available() {
    info_log "wait oauth-webhook service endpoints available"

    local max_retries=60
    local retry_delay=5
    local attempts=0

    while [ $attempts -lt $max_retries ]; do
        local endpoint_addresses
        endpoint_addresses=$(kubectl get endpoints "${OAUTH_WEBHOOK_RELEASE_NAME}" \
            -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            -o jsonpath='{.subsets[*].addresses[*].ip}' 2>/dev/null)
        if [ -n "${endpoint_addresses}" ]; then
            info_log "oauth-webhook service endpoints are available: ${endpoint_addresses}"
            return
        fi

        info_log "oauth-webhook service endpoints are not ready. Retrying in ${retry_delay} seconds..."
        sleep ${retry_delay}
        ((attempts++))
    done

    fatal_log "oauth-webhook service endpoints are not available after $((max_retries * retry_delay)) seconds"
}

function install_oauth_server() {
    info_log "install oauth_server"
    local signing_key=$1
    local encryption_key=$2
    local jwt_private_key=$3
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_RELEASE_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"

    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${OAUTH_SERVER}" "${OAUTH_SERVER}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${OAUTH_SERVER}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        install_oauth_server_enable_https
    else
        install_oauth_server_disable_https
    fi
}

function install_oauth_server_disable_https() {
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_CHART_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"
    info_log "start installing oauth_server disable https"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_server_chart_path}" ]; then
        fatal_log "oauth_server chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.httpServerConfig.enableHttps=false
        --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set config.loginSessionConfig.signingKey="${signing_key}"
        --set config.loginSessionConfig.encryptionKey="${encryption_key}"
        --set config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-server"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${OAUTH_SERVER_RELEASE_NAME}" \
        "${oauth_server_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_oauth_server_enable_https() {
    local oauth_server_chart_path="./${CHART_PATH}/${OAUTH_SERVER_CHART_NAME}-${OAUTH_SERVER_CHART_VERSION}.tgz"
    info_log "start installing oauth_server enable https"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${oauth_server_chart_path}" ]; then
        fatal_log "oauth_server chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.httpServerConfig.enableHttps=true
        --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}"
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set config.loginSessionConfig.signingKey="${signing_key}"
        --set config.loginSessionConfig.encryptionKey="${encryption_key}"
        --set config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}"
        --set-file config.httpServerConfig.tlsCert="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.crt"
        --set-file config.httpServerConfig.tlsKey="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.key"
        --set-file config.httpServerConfig.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/oauth-server"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${OAUTH_SERVER_RELEASE_NAME}" \
        "${oauth_server_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function generate_oauth_webhook_tls_cert() {
    if kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" >/dev/null 2>&1; then
        info_log "oauth-webhook tls secret already exists, skip generation"
        return
    fi

    info_log "generate oauth-webhook cert"
    local cur_path=$(pwd)
    mkdir -p "${OAUTH_WEBHOOK_CHART_PATH}"
    cd "${OAUTH_WEBHOOK_CHART_PATH}"  || fatal_log "Failed to change directory to ${OAUTH_WEBHOOK_CHART_PATH}"
    sudo jq ".signing.default.expiry = \"$OAUTH_CERTS_EXPIRATION_TIME\"" ../resource/oauth-webhook/server-signing-config.json > oauthtmpfile.json && mv oauthtmpfile.json ../resource/oauth-webhook/server-signing-config.json -f

    # 生成 oauth-webhook 的私钥
    cat <<EOF | sudo cfssl genkey - | sudo cfssljson -bare server
    {
      "hosts": [
        "oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}.svc.cluster.local",
        "oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}.pod.cluster.local"
      ],
      "CN": "oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}.pod.cluster.local",
      "key": {
        "algo": "rsa",
        "size": 4096
      }
    }
EOF

    # 创建证书签名请求（CSR）并发送到 Kubernetes API
    cat <<EOF | kubectl apply -f -
    apiVersion: certificates.k8s.io/v1
    kind: CertificateSigningRequest
    metadata:
      name: oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} # my-svc.my-namespace
    spec:
      request: $(cat server.csr | base64 | tr -d '\n')
      signerName: openfuyao.io/oauth-signer # kubernetes.io/kube-apiserver-client
      usages:
      - digital signature
      - key encipherment
      - server auth
      - client auth
EOF

    kubectl certificate approve oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} # my-svc.my-namespace

    # 作为签名者签署证书，并将颁发的证书上传到API服务器
    cat <<EOF | sudo cfssl gencert -initca - | sudo cfssljson -bare ca
    {
      "CN": "openfuyao.io/oauth-signer",
      "key": {
        "algo": "rsa",
        "size": 4096
      }
    }
EOF

    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath='{.spec.request}' | \
      base64 --decode | \
      sudo cfssl sign -ca ca.pem -ca-key ca-key.pem -config ../resource/oauth-webhook/server-signing-config.json - | \
      sudo cfssljson -bare ca-signed-server

    # 在 API 对象的状态中填充签名证书
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o json | \
      sudo jq '.status.certificate = "'$(base64 ca-signed-server.pem | tr -d '\n')'"' | \
      kubectl replace --raw /apis/certificates.k8s.io/v1/certificatesigningrequests/oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}/status -f -

    # 下载颁发的证书并将其保存到 server.crt 文件中
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath='{.status.certificate}' \
      | base64 --decode > server.crt

    kubectl create secret generic "${OAUTH_WEBHOOK_TLS}" --namespace="${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --from-file=ca.crt=ca.pem \
        --from-file=tls.crt=server.crt \
        --from-file=tls.key=server-key.pem

    kubectl create secret generic "${OAUTH_WEBHOOK_TLS}" --namespace="${OPENFUYAO_SYSTEM_CONTROLLER_NAMESPACE}" \
        --from-file=ca.crt=ca.pem \
        --from-file=tls.crt=server.crt \
        --from-file=tls.key=server-key.pem

    cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
    info_log "Successfully generated oauth-webhook cert"
}

function install_marketplace_service() {
    if [ -n "${MARKETPLACE_SERVICE_INSTALLED}" ]; then
        info_log "marketplace_service has been installed, skip installation"
        return
    fi
    info_log "Start installing marketplace_service"

    download_charts_with_retry "${MARKETPLACE_SERVICE_CHART_NAME}" "${MARKETPLACE_SERVICE_CHART_VERSION}"
    local marketplace_service_chart_path="./${CHART_PATH}/${MARKETPLACE_SERVICE_CHART_NAME}-${MARKETPLACE_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${marketplace_service_chart_path}" ]; then
        fatal_log "marketplace_service chart not found"
    fi

    if [ "${ENABLE_HTTPS}" == "true" ]; then
        install_marketplace_service_disable_https
    else
        install_marketplace_service_disable_https
    fi
}

function install_marketplace_service_enable_https() {
    local marketplace_service_chart_path="./${CHART_PATH}/${MARKETPLACE_SERVICE_CHART_NAME}-${MARKETPLACE_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${marketplace_service_chart_path}" ]; then
        fatal_log "marketplace_service chart not found"
        return 1
    fi

    create_service_cert "${MARKETPLACE_SERVICE}" \
        "${MARKETPLACE_SERVICE}" \
        "${MARKETPLACE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${MARKETPLACE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" \
        "${MARKETPLACE_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableHttps=true
        --set config.insecureSkipVerify=true
        --set enableOAuth=true
        --set symmetricKey.secretKey="$(openssl rand -base64 32)"
        --set images.core.tag="${MARKETPLACE_SERVICE_IMAGE_TAG}"
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/marketplace-service"
            --set images.oauthProxy.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    SET_ARGS+=(
        --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${MARKETPLACE_SERVICE}/${MARKETPLACE_SERVICE}.crt"
        --set-file config.tlsKey="./${FUYAO_CERTS_PATH}/${MARKETPLACE_SERVICE}/${MARKETPLACE_SERVICE}.key"
        --set-file config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt"
    )

    helm install "${MARKETPLACE_SERVICE_RELEASE_NAME}" \
        "${marketplace_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_marketplace_service_disable_https() {
    info_log "disable https for marketplace_service"
    local marketplace_service_chart_path="./${CHART_PATH}/${MARKETPLACE_SERVICE_CHART_NAME}-${MARKETPLACE_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${marketplace_service_chart_path}" ]; then
        fatal_log "marketplace_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableHttps=false
        --set config.insecureSkipVerify=true
        --set enableOAuth=true
        --set symmetricKey.secretKey="$(openssl rand -base64 32)"
        --set images.core.tag="${MARKETPLACE_SERVICE_IMAGE_TAG}"
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/marketplace-service"
            --set images.oauthProxy.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    helm install "${MARKETPLACE_SERVICE_RELEASE_NAME}" \
        "${marketplace_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
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
    local plugin_management_service_chart_path="./${CHART_PATH}/${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}-${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${plugin_management_service_chart_path}" ]; then
        fatal_log "plugin_management_service chart not found"
        return 1
    fi
    info_log "disable https for plugin_management_service"

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set config.enableHttps=false
        --set enableOAuth=true
        --set images.core.tag="${PLUGIN_MANAGEMENT_SERVICE_IMAGE_TAG}"
        --set images.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set images.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/plugin-management-service"
            --set images.oauthProxy.repository="${REGISTRY}/oauth-proxy"
        )
    fi

    helm install "${PLUGIN_MANAGEMENT_SERVICE_RELEASE_NAME}" \
        "${plugin_management_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_application_management_service() {
    if [ -n "${APPLICATION_MANAGEMENT_SERVICE_INSTALLED}" ]; then
      info_log "application_management_service has been installed, skip installation"
      return
    fi
    info_log "Start installing application_management_service"

    download_charts_with_retry "${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}" "${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}"
    local application_management_service_chart_path="./${CHART_PATH}/${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}-${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${application_management_service_chart_path}" ]; then
        fatal_log "application_management_service chart not found"
    fi

    install_application_management_service_disable_https
}

function install_application_management_service_disable_https() {
    local application_management_service_chart_path="./${CHART_PATH}/${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}-${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    info_log "disable https for application_management_service"

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set enableOAuth=true
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
        --set images.core.tag="${APPLICATION_MANAGEMENT_SERVICE_IMAGE_TAG}"
        --set images.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set images.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.oauthProxy.repository="${REGISTRY}/oauth-proxy"
            --set images.core.repository="${REGISTRY}/application-management-service"
        )
    fi

    helm install "${APPLICATION_MANAGEMENT_SERVICE_RELEASE_NAME}" \
        "${application_management_service_chart_path}" \
        -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        "${SET_ARGS[@]}"
}

function install_user_management_operator() {
    if [ -n "${USER_MANAGEMENT_OPERATOR_INSTALLED}" ]; then
        info_log "user_management_operator has been installed, skip installation"
        return
    fi

    info_log "Start installing user_management_operator"

    download_charts_with_retry "${USER_MANAGEMENT_OPERATOR_CHART_NAME}" "${USER_MANAGEMENT_OPERATOR_CHART_VERSION}"
    local user_management_operator_chart_path="./${CHART_PATH}/${USER_MANAGEMENT_OPERATOR_CHART_NAME}-${USER_MANAGEMENT_OPERATOR_CHART_VERSION}.tgz"

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

function install_web_terminal_service() {
    if [ -n "${WEB_TERMINAL_SERVICE_INSTALLED}" ]; then
        info_log "web_terminal_service has been installed, skip installation"
        return
    fi
    info_log "Start installing web_terminal_service"

    download_charts_with_retry "${WEB_TERMINAL_SERVICE_CHART_NAME}" "${WEB_TERMINAL_SERVICE_CHART_VERSION}"
    local web_terminal_service_chart_path="./${CHART_PATH}/${WEB_TERMINAL_SERVICE_CHART_NAME}-${WEB_TERMINAL_SERVICE_CHART_VERSION}.tgz"

    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${web_terminal_service_chart_path}" ]; then
        fatal_log "web_terminal_service chart not found"
        return 1
    fi

    local -a SET_ARGS=()
    SET_ARGS+=(
        --set images.core.tag="${WEB_TERMINAL_SERVICE_IMAGE_TAG}"
        --set images.kubectl.tag="${WEB_TERMINAL_SERVICE_IMAGE_TAG}"
        --set thirdPartyImage.busyBox.repository="${BUSY_BOX_REPOSITORY}"
        --set thirdPartyImage.busyBox.tag="${BUSY_BOX_IMAGE_TAG}"
    )

    if [ -n "$REPO" ]; then
        SET_ARGS+=(
            --set images.core.repository="${REGISTRY}/web-terminal-service"
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao"
        )
    fi

    helm install "${WEB_TERMINAL_SERVICE_RELEASE_NAME}" \
        "${web_terminal_service_chart_path}" \
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
    update_image_tag_from_cm "controller" "$YAML_FILE"
    update_image_tag_from_cm "kube-webhook-certgen" "$YAML_FILE"

    if [ -n "$REPO" ]; then
      sudo sed -i "s|${FUYAO_THIRD_REGISTRY}|${REGISTRY}|g" ./resource/ingress-nginx/ingress-nginx.yaml
    fi

    kubectl create -f ./resource/ingress-nginx/ingress-nginx.yaml
    info_log "success install ingress-nginx"
}

function install_yq() {
    if command -v yq >/dev/null 2>&1; then
        info_log "yq already installed"
        return
    fi

    info_log "Start installing yq"
    sudo mv -f ./${ARCH}-bin/yq_linux_${ARCH} /usr/local/bin/yq
    sudo chmod +x /usr/local/bin/yq
    info_log "success install yq"
}

function install_jq() {
    if command -v jq >/dev/null 2>&1; then
        info_log "jq already installed"
        return
    fi
    
    info_log "Start installing jq"
    sudo mv -f ./${ARCH}-bin/jq-linux-${ARCH} /usr/local/bin/jq
    sudo chmod +x /usr/local/bin/jq
    info_log "success install jq"
}

function exec_install_tools() {
    local name=$1

    if command -v "$name" >/dev/null 2>&1; then
        info_log "$name already installed"
        return
    fi
    info_log "Start installing $name"
    sudo mv -f ./${ARCH}-bin/"${name}_linux_${ARCH}" /usr/local/bin/"$name"
    sudo chmod +x /usr/local/bin/"$name"
    info_log "success install $name"
}

function install_cfssl() {
    exec_install_tools cfssl
    exec_install_tools cfssl-certinfo
    exec_install_tools cfssljson
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

function create_namespace() {
    local namespace=$1
    if kubectl get namespace "$namespace" >/dev/null 2>&1; then
        info_log "namespace $namespace already exists"
    else
        kubectl create namespace "$namespace"
        info_log "create namespace $namespace success"
    fi
}

function create_root_ca() {
    # 创建根证书目录
    if [ ! -d "${FUYAO_CERTS_PATH}" ]; then
        mkdir -p "${FUYAO_CERTS_PATH}"
    fi

    cd "${FUYAO_CERTS_PATH}"  || fatal_log "failed to change directory to ${FUYAO_CERTS_PATH}"

    # 先检查 openfuyao-system-ca-secret 是否存在，如果存在则从中获取证书
    if kubectl get secret "${OPENFUYAO_SYSTEM_ROOT_CA_SECRET}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" >/dev/null 2>&1; then
        info_log "found ${OPENFUYAO_SYSTEM_ROOT_CA_SECRET} secret, extracting ca.key and ca.crt from secret"
        kubectl get secret "${OPENFUYAO_SYSTEM_ROOT_CA_SECRET}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -o jsonpath='{.data.ca\.key}' | base64 -d > ca.key
        kubectl get secret "${OPENFUYAO_SYSTEM_ROOT_CA_SECRET}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -o jsonpath='{.data.ca\.crt}' | base64 -d > ca.crt
        info_log "successfully extracted ca.key and ca.crt from secret"
        cd ..
        return
    fi

    # 如果本地文件已存在，直接使用
    if [ -f "ca.key" ] && [ -f "ca.crt" ]; then
        info_log "fuyao ca.key and ca.crt already exist locally"
        create_root_ca_secret
        cd ..
        return
    fi

    info_log "no existing ca secret or local files found, generating new CA certificates"

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

function add_helm_repo() {
    curl "${OPENFUYAO_REPO}" -k --connect-timeout 20
    if [ $? -ne 0 ]; then
        info_log "offline mode, skip add helm repo"
        return
    fi
    helm repo remove "${OPENFUYAO}" >/dev/null 2>&1
    helm repo add "${OPENFUYAO}" "${OPENFUYAO_REPO}"
    helm repo update
    info_log "add helm repo ${OPENFUYAO_REPO} success"
}

function set_kubeconfig() {
    if [ -f "/etc/kubernetes/admin.conf" ]; then
        export KUBECONFIG="/etc/kubernetes/admin.conf"
        return
    fi
    # todo 添加从secret中获取kubeconfig
    fatal_log "failed to set kubeconfig"
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

function create_default_user() {
    local user_resource="users.users.openfuyao.com"

    if kubectl get "${user_resource}" admin >/dev/null 2>&1; then
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

function install_helm() {
    if command -v helm >/dev/null 2>&1; then
        info_log "helm already installed"
        return
    fi

    info_log "Start installing helm"
    sudo tar -zxvf ./${ARCH}-bin/helm-v3.14.2-"${OS}"-${ARCH}.tar.gz
    sudo mv "${OS}"-$ARCH/helm /usr/local/bin/helm
    info_log "success install helm"
}

# 定义函数：判断IP是否属于子网
function is_ip_in_subnet() {
    local ip="$1"
    local subnet="$2"

    local network_addr="${subnet%/*}"
    local prefix_length="${subnet#*/}"

    # 将IP地址转换为整数
    ip_to_int() {
        local ip="$1"
        local a b c d
        IFS=. read -r a b c d <<< "$ip"
        echo $((a * 256 ** 3 + b * 256 ** 2 + c * 256 + d))
    }

    local ip_int=$(ip_to_int "$ip")
    local network_int=$(ip_to_int "$network_addr")
    local mask_int=$((0xffffffff << (32 - prefix_length) & 0xffffffff))

    if (( (ip_int & mask_int) == network_int )); then
        return 0  # 属于子网
    else
        return 1  # 不属于子网
    fi
}

function reboot_pods() {
     # 存在pod ip 分配错误问题，重启pod后可解决问题
     # 如何确认正常的pod ip，在下面范围中的ip可认为是正常的ip
     ## 1、kubectl get node -A -owide 回显结果中的INTERNAL-IP字段
     ## 2、kubectl get ippool -n default-ipv4-ippool -oyaml 回显结果中spec/cidr的网络段

     # 获取所有 Node 的 InternalIP
     NODE_IPS=$(kubectl get nodes -o json | jq -r '.items[].status.addresses[] | select(.type=="InternalIP") | .address')

     # 获取 Calico IPPool 的 CIDR 网段
     CALICO_CIDR=$(kubectl get ippools.crd.projectcalico.org default-ipv4-ippool -o json | jq -r '.spec.cidr')

     # 检查是否获取到 IPPool
     if [ -z "$CALICO_CIDR" ]; then
         info_log "Calico IPPool not found!"
         return
     fi

     # 遍历所有 Pod，检查 IP 是否合法
     kubectl get pods -A -o json | jq -r '.items[] | select(.status.podIP != null) | "\(.metadata.namespace) \(.metadata.name) \(.status.podIP)"' | while read -r ns pod ip; do
         # 检查 Pod IP 是否在 Node IP 列表里
         if grep -q "$ip" <<< "$NODE_IPS"; then
             info_log "Pod $ns/$pod IP $ip is a Node IP, skipping..."
             continue
         fi

         # 检查 Pod IP 是否在 Calico IPPool 网段
         is_ip_in_subnet "$ip" "$CALICO_CIDR"
         if [ $? -ne 0 ]; then
             info_log "Pod $ns/$pod IP $ip is outside Calico IPPool $CALICO_CIDR, restarting..."
             kubectl delete pod -n "$ns" "$pod" --force --grace-period=0
         else
             info_log "Pod $ns/$pod IP $ip is valid."
         fi
     done
}

function usage() {
    echo "see source file"
}

function download_addon_charts_with_retry() {
    local chart_name=$1
    local chart_version=$2

    if [ ! -d "${ADDON_CHART_PATH}" ]; then
        mkdir -p "${ADDON_CHART_PATH}"
    fi

    local cur_path=$(pwd)
    cd "${ADDON_CHART_PATH}"  || fatal_log "Failed to change directory to ${ADDON_CHART_PATH}"
    if [ -f "${chart_name}-${chart_version}.tgz" ]; then
        info_log "${chart_name}-${chart_version}.tgz already exists, skip downloading"
        cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
        return
    fi
    info_log "Downloading ${chart_name} ${chart_version} chart"

    local attempts=0
    while [ $attempts -lt 3 ]; do
        if [ "$IS_ONLINE" = "false" ]; then
            helm fetch "${chart_name}" --repo "${OPENFUYAO_REPO}" --version "${chart_version}"
        else
            helm fetch "${FUYAO_REPO}/${chart_name}" --version "${chart_version}"
        fi
        if [ $? -eq 0 ]; then
            info_log "Successfully downloaded ${chart_name} ${chart_version} chart"
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

# get-opt 参数
main(){
    while true
    do
        case "$1" in
        -r|--registry)  # 镜像仓库地址
            REPO="$2"
            info_log "registry is ${REPO}"
            shift
            ;;
        --enableHttps)
            ENABLE_HTTPS="$2"
            info_log "enable https is ${ENABLE_HTTPS}"
            shift
            ;;
        --harborAdminPassword)
            HARBOR_ADMIN_PASSWORD="$2"
            info_log "harborAdminPassword is ${HARBOR_ADMIN_PASSWORD}"
            shift
            ;;
        --harborRegistryPassword)
            HARBOR_REGISTRY_PASSWORD="$2"
            info_log "harborRegistryPassword is ${HARBOR_REGISTRY_PASSWORD}"
            shift
            ;;
        --harborDatabasePassword)
           HARBOR_DATABASE_PASSWORD="$2"
           info_log "harborDatabasePassword is ${HARBOR_DATABASE_PASSWORD}"
            shift
            ;;
        --harborRegistryPvSize)
            HARBOR_REGISTRY_PV_SIZE="$2"
            info_log "harborRegistryPvSize is ${HARBOR_REGISTRY_PV_SIZE}"
            shift
            ;;
        --harborJobservicePvSize)
            HARBOR_JOBSERVICE_PV_SIZE="$2"
            info_log "harborJobservicePvSize is ${HARBOR_JOBSERVICE_PV_SIZE}"
            shift
            ;;
        --harborJobservicePvcSize)
            HARBOR_JOBSERVICE_PVC_SIZE="$2"
            info_log "harborJobservicePvcSize is ${HARBOR_JOBSERVICE_PVC_SIZE}"
            shift
            ;;
        --harborDatabasePvSize)
            HARBOR_DATABASE_PV_SIZE="$2"
            info_log "harborDatabasePvSize is ${HARBOR_DATABASE_PV_SIZE}"
            shift
            ;;
        --harborDatabasePvcSize)
            HARBOR_DATABASE_PVC_SIZE="$2"
            info_log "harborDatabasePvcSize is ${HARBOR_DATABASE_PVC_SIZE}"
            shift
            ;;
        --harborRegistryPvcSize)
            HARBOR_REGISTRY_PVC_SIZE="$2"
            info_log "harborRegistryPvcSize is ${HARBOR_REGISTRY_PVC_SIZE}"
            shift
            ;;
        --harborChartmuseumPvSize)
            HARBOR_CHARTMUSEUM_PV_SIZE="$2"
            info_log "harborChartmuseumPvSize is ${HARBOR_CHARTMUSEUM_PV_SIZE}"
            shift
            ;;
        --harborChartmuseumPvcSize)
            HARBOR_CHARTMUSEUM_PVC_SIZE="$2"
            info_log "harborChartmuseumPvcSize is ${HARBOR_CHARTMUSEUM_PVC_SIZE}"
            shift
            ;;
        --harborRedisPvSize)
            HARBOR_REDIS_PV_SIZE="$2"
            info_log "harborRedisPvSize is ${HARBOR_REDIS_PV_SIZE}"
            shift
            ;;
        --harborRedisPvcSize)
            HARBOR_REDIS_PVC_SIZE="$2"
            info_log "harborRedisPvcSize is ${HARBOR_REDIS_PVC_SIZE}"
            shift
            ;;
        --oauthCertsExpirationTime)
            OAUTH_CERTS_EXPIRATION_TIME="$2"
            info_log "oauthCertsExpirationTime is ${OAUTH_CERTS_EXPIRATION_TIME}"
            shift
            ;;
        --repo) # chart仓库地址
            OPENFUYAO_REPO="$2"
            info_log "repo is ${OPENFUYAO_REPO}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
        shift
        break
        ;;
        *)
        echo "$1 is not option, please use -h to view help"
        shift
        break
        ;;
        esac
        shift
    done

    # 判断是否为离线安装，使用 deploy.bocloud.k8s 的就默认为离线安装，其他情况就是在线安装
    substring="deploy.bocloud.k8s"
    if [[ "${REPO}" == *"${substring}"* ]]; then
        info_log "offline deploy"
        IS_ONLINE="false"
    else
        info_log "online deploy"
        IS_ONLINE="true"
    fi

    REGISTRY="${REPO%/}"
    if [ -n "$REPO" ]; then
      BUSY_BOX_REPOSITORY="${REGISTRY}/busybox"
    else
      BUSY_BOX_REPOSITORY="${FUYAO_THIRD_REGISTRY}/busybox"
    fi
    if [ -n "$REPO" ]; then
      OPENFUYAO_SYSTEM_CONTROLLER_REPOSITORY="${REGISTRY}/openfuyao-system-controller"
    else
      OPENFUYAO_SYSTEM_CONTROLLER_REPOSITORY="${FUYAO_RGISTRY}/openfuyao-system-controller"
    fi
    info_log "registry is : ${REGISTRY} busybox repository : ${BUSY_BOX_REPOSITORY} openfuyao system controller : ${OPENFUYAO_SYSTEM_CONTROLLER_REPOSITORY}"

    if [ -z "${ENABLE_HTTPS}" ];then
        ENABLE_HTTPS="true"
    fi

    if [ -z "${OPENFUYAO_REPO}" ];then
        OPENFUYAO_REPO="${FUYAO_REPO}"
    fi

    if [ -z "${OAUTH_CERTS_EXPIRATION_TIME}" ];then
        OAUTH_CERTS_EXPIRATION_TIME="1752000h"
    fi

    set_kubeconfig
    kubectl create ns "${OPENFUYAO_SYSTEM_NAMESPACE}"

    install_yq
    install_jq
    install_helm
    install_cfssl
    generate_var
    create_root_ca
    add_helm_repo
    reboot_pods
    install_ingress_nginx

    install_console_website
    install_helm_chart_repository
    install_kube_prometheus
    install_monitoring_service
    install_console_service
    install_marketplace_service
    install_application_management_service
    install_oauth_webhook_and_oauth_server
    install_plugin_management_service
    install_user_management_operator
    install_web_terminal_service
    install_installer_website
    install_installer_service
    install_metrics_server

    create_default_user

    ./postinstall.sh

    unset HARBOR_ADMIN_PASSWORD
    unset HARBOR_REGISTRY_PASSWORD
    unset HARBOR_DATABASE_PASSWORD
    info_log "reset is over"
}

INSTALL_SHELL=$(getopt -n install.sh -o r: --long registry:,repo:,oauthCertsExpirationTime:,harborAdminPassword:,harborRegistryPassword:,harborDatabasePassword:,enableHttps:,harborRegistryPvSize:,harborJobservicePvSize:,harborJobservicePvcSize:,harborDatabasePvSize:,harborDatabasePvcSize:,harborRegistryPvcSize:,harborChartmuseumPvSize:,harborChartmuseumPvcSize:,harborRedisPvSize:,harborRedisPvcSize:,help -- "$@")
[ $? -ne 0 ] && fatal_log "failed to parse command line options"
eval set -- "$INSTALL_SHELL"
main "$@"
