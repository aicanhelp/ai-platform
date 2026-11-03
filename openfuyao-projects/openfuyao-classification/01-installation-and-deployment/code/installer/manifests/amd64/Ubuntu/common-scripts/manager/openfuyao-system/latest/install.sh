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

source ../../utils/log.sh "openfuyao-system"
source ./consts.sh
source ../../utils/consts.sh
source ./preinstall.sh

REGISTRY="{{ .registry }}"
ENABLE_HTTPS="{{ .enableHttps }}"
BUSY_BOX_IMAGE="{{ .registry }}/busybox/busybox:1.36.1"
BUSY_BOX_REPOSITORY="${REGISTRY}/busybox/busybox"

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
        helm fetch "${chart_name}" --repo "${FUYAO_REPO}" --version "${chart_version}"
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

function install_console_website() {
    if [ -n "${CONSOLE_WEBSITE_INSTALLED}" ]; then
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
    if [ -d "${CHART_PATH}" ] && [ -f "${console_website_chart_path}" ]; then
        helm install "${CONSOLE_WEBSITE_RELEASE_NAME}" "${console_website_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set config.enableTLS=false \
            --set images.core.repository="${REGISTRY}"/console-website,images.core.tag="${CONSOLE_WEBSITE_IMAGE_TAG}" >> "${OUTPUT}"
    else
        fatal_log "console_website chart not found"
    fi
}

function install_console_website_enable_https() {
    info_log "Start installing console_website with https"
    local console_website_chart_path="./${CHART_PATH}/${CONSOLE_WEBSITE_CHART_NAME}-${CONSOLE_WEBSITE_CHART_VERSION}.tgz"
    if [ -d "${CHART_PATH}" ] && [ -f "${console_website_chart_path}" ]; then
        helm install "${CONSOLE_WEBSITE_RELEASE_NAME}" "${console_website_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set config.enableTLS=true \
            --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.crt",config.tlsKey="./${FUYAO_CERTS_PATH}/${CONSOLE_WEBSITE}/${CONSOLE_WEBSITE}.key",config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt" \
            --set images.core.repository="${REGISTRY}"/console-website,images.core.tag="${CONSOLE_WEBSITE_IMAGE_TAG}" >> "${OUTPUT}"
    else
        fatal_log "console_website chart not found"
    fi
}

function install_console_service() {
    if [ -n "${CONSOLE_SERVICE_INSTALLED}" ]; then
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
    kubectl wait -n kube-system --for=condition=ready pod -l k8s-app=kube-dns >> "${OUTPUT}"

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
    if [ -d "${CHART_PATH}" ] && [ -f "${console_service_chart_path}" ]; then
        helm install "${CONSOLE_SERVICE_RELEASE_NAME}" "${console_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set images.core.repository="${REGISTRY}"/console-service \
            --set images.core.tag="${CONSOLE_SERVICE_IMAGE_TAG}" \
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
            --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao" \
            --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}" \
            --set serverHost.monitoring="${MONITORING_HOST_HTTP}" \
            --set serverHost.consoleWebsite="${CONSOLE_WEBSITE_HOST_HTTP}" \
            --set symmetricKey.tokenKey="$(openssl rand -base64 32)",symmetricKey.secretKey="$(openssl rand -base64 32)" \
            --set config.enableHttps=false >> "${OUTPUT}"
    else
        fatal_log "console_service chart not found"
    fi
}

function install_console_service_enable_https() {
    info_log "Start installing console_service with https"
    local console_service_chart_path="./${CHART_PATH}/${CONSOLE_SERVICE_CHART_NAME}-${CONSOLE_SERVICE_CHART_VERSION}.tgz"
    if [ -d "${CHART_PATH}" ] && [ -f "${console_service_chart_path}" ]; then
        helm install "${CONSOLE_SERVICE_RELEASE_NAME}" "${console_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set ingress.secretName="${INGRESS_NGINX_NAMESPACE}/${INGRESS_NGINX_TLS_SECRET}" \
            --set images.core.repository="${REGISTRY}"/console-service \
            --set images.core.tag="${CONSOLE_SERVICE_IMAGE_TAG}" \
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
            --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao" \
            --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}" \
            --set symmetricKey.tokenKey="$(openssl rand -base64 32)",symmetricKey.secretKey="$(openssl rand -base64 32)",config.enableHttps=true \
            --set serverHost.localHarbor="${LOCAL_HARBOR_HOST}" \
            --set serverHost.oauthServer="${OAUTH_SERVER_HOST}" \
            --set serverHost.consoleService="${CONSOLE_SERVICE_HOST}" \
            --set serverHost.consoleWebsite="${CONSOLE_WEBSITE_HOST}" \
            --set serverHost.monitoring="${MONITORING_HOST_HTTP}" \
            --set-file config.tlsCert="./${FUYAO_CERTS_PATH}/${CONSOLE_SERVICE}/${CONSOLE_SERVICE}.crt",config.tlsKey="./${FUYAO_CERTS_PATH}/${CONSOLE_SERVICE}/${CONSOLE_SERVICE}.key",config.rootCA="./${FUYAO_CERTS_PATH}/ca.crt" >> "${OUTPUT}"
    else
        fatal_log "console_service chart not found"
    fi
}

function install_metrics_server() {
    if [ -n "${METRICS_SERVER_INSTALLED}" ]; then
        return
    fi
    info_log "Start installing metrics-server"
    sudo tar -xvf ./yaml/metrics-server-yaml.tar.gz -C ./yaml
    sudo sed -i "s|registry.k8s.io|${REGISTRY}|g" ./yaml/metrics-server-yaml/metrics-server.yaml
    kubectl apply -f ./yaml/metrics-server-yaml/metrics-server.yaml
    info_log "Completing the installation of metrics-server"
}

function install_kube_prometheus() {
    if [ -n "${KUBE_PROMETHEUS_INSTALLED}" ]; then
        return
    fi
    info_log "Start installing kube_prometheus"
    sudo tar -xvf ./yaml/kube-prometheus-yaml.tar.gz -C ./yaml
    create_namespace "${MONITOR_NAMESPACE}"

    pki_path=$(kubectl describe cm kubeadm-config -n  kube-system | grep certificatesDir | awk '{print $2}')
    if [ -z "$pki_path" ]; then
        info_log "pki_path is empty, use default value"
        pki_path="/etc/kubernetes/pki"
    fi

    kubectl create secret generic etcd-certs \
      --from-file=ca.crt=${pki_path}/etcd/ca.crt \
      --from-file=etcd.crt=${pki_path}/etcd/peer.crt \
      --from-file=etcd.key=${pki_path}/etcd/peer.key \
      --namespace="${MONITOR_NAMESPACE}"
    kubectl apply -f ./yaml/kube-prometheus/kubernetes-components-service/
    info_log "Completing the installation of kube-prometheus service"

    # 离线部署时替换镜像仓地址
    local yamlFiles=(
        "./yaml/kube-prometheus/nodeExporter-daemonset.yaml"
        "./yaml/kube-prometheus/alertmanager-alertmanager.yaml"
        "./yaml/kube-prometheus/prometheus-prometheus.yaml"
        "./yaml/kube-prometheus/prometheusOperator-deployment.yaml"
        "./yaml/kube-prometheus/blackboxExporter-deployment.yaml"
        "./yaml/kube-prometheus/kubeStateMetrics-deployment.yaml"
    )
    for yamlFile in "${yamlFiles[@]}" ; do
        sed -i  "s|${FUYAO_RGISTRY}|${REGISTRY}|g" "${yamlFile}"
    done

    kubectl apply --server-side -f ./yaml/kube-prometheus/setup
    kubectl wait \
    	--for condition=Established \
    	--all CustomResourceDefinition \
    	--namespace="${MONITOR_NAMESPACE}"
    kubectl apply -f ./yaml/kube-prometheus/
    info_log "Completing the installation of kube-prometheus"
}

function install_monitoring_service() {
    if [ -n "${MONITORING_SERVICE_INSTALLED}" ]; then
        return
    fi
    info_log "installing monitoring_service"
    download_charts_with_retry "${MONITORING_SERVICE_CHART_NAME}" "${MONITORING_SERVICE_CHART_VERSION}"

    # 创建证书
    if [ "${ENABLE_HTTPS}" == "true" ]; then
        create_service_cert "${MONITORING_SERVICE}" "${MONITORING_SERVICE}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${MONITORING_SERVICE}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
        # install_monitoring_service_enable_https
        install_monitoring_service_disable_https
    else
        install_monitoring_service_disable_https
    fi
    info_log "Completing the installation of monitoring-service"
}

function install_monitoring_service_enable_https() {
    local monitoring_service_chart_path="./${CHART_PATH}/${MONITORING_SERVICE_CHART_NAME}-${MONITORING_SERVICE_CHART_VERSION}.tgz"
    if [ -d "${CHART_PATH}" ] && [ -f "${monitoring_service_chart_path}" ]; then
        helm install "${MONITORING_SERVICE_RELEASE_NAME}" "${monitoring_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set images.core.repository="${REGISTRY}"/monitoring-service \
            --set images.busybox.repository="${BUSY_BOX_REPOSITORY}" \
            --set images.busybox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set enableOAuth=true \
            --set images.oauth.repository="${REGISTRY}"/oauth-proxy \
            --set config.httpServerConfig.enableHttps=true \
            --set-file config.httpServerConfig.tlsCert="./${FUYAO_CERTS_PATH}/${MONITORING_SERVICE}/${MONITORING_SERVICE}.crt",config.httpServerConfig.tlsKey="./${FUYAO_CERTS_PATH}/${MONITORING_SERVICE}/${MONITORING_SERVICE}.key",config.httpServerConfig.rootCA="./${FUYAO_CERTS_PATH}/ca.crt" >> "${OUTPUT}"
    else
        fatal_log "monitoring_service chart not found"
    fi
}

function install_monitoring_service_disable_https() {
    info_log "installing monitoring_service without https"
    local monitoring_service_chart_path="./${CHART_PATH}/${MONITORING_SERVICE_CHART_NAME}-${MONITORING_SERVICE_CHART_VERSION}.tgz"
    if [ -d "${CHART_PATH}" ] && [ -f "${monitoring_service_chart_path}" ]; then
        helm install "${MONITORING_SERVICE_RELEASE_NAME}" "${monitoring_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set images.core.repository="${REGISTRY}"/monitoring-service \
            --set images.busybox.repository="${BUSY_BOX_REPOSITORY}" \
            --set images.busybox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set enableOAuth=true \
            --set images.oauth.repository="${REGISTRY}"/oauth-proxy \
            --set config.httpServerConfig.enableHttps=false \
            --set image.repository="${REGISTRY}"/monitoring-service,image.tag="${MONITORING_SERVICE_IMAGE_TAG}" >> "${OUTPUT}"
    else
        fatal_log "monitoring_service chart not found"
    fi
}

function install_helm_chart_repository() {
    if [ -n "${LOCAL_HARBOR_INSTALLED}" ]; then
        return
    fi
    info_log "Start installing helm_chart_repository"

    # create initContainer job yaml
    cat <<EOF > harbor-init.yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: init-harbor
      namespace: $OPENFUYAO_SYSTEM_NAMESPACE
    spec:
      template:
        spec:
          containers:
          - name: init-harbor-container
            image: {{ .registry }}/busybox/busybox:1.36.1
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
      kubectl logs -n "${OPENFUYAO_SYSTEM_NAMESPACE}" job/init-harbor >> ${OUTPUT}
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
    sudo yq -i 'select(di == 0).spec.capacity.storage = "{{ .harborRegistryPvSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 1).spec.resources.requests.storage = "{{ .harborRegistryPvcSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 2).spec.capacity.storage = "{{ .harborChartmuseumPvSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 3).spec.resources.requests.storage = "{{ .harborChartmuseumPvcSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 4).spec.capacity.storage = "{{ .harborRedisPvSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 5).spec.resources.requests.storage = "{{ .harborRedisPvcSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 6).spec.capacity.storage = "{{ .harborJobservicePvSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 7).spec.resources.requests.storage = "{{ .harborJobservicePvcSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 8).spec.capacity.storage = "{{ .harborDatabasePvSize }}"' ${harbor_pv_path}
    sudo yq -i 'select(di == 9).spec.resources.requests.storage = "{{ .harborDatabasePvcSize }}"' ${harbor_pv_path}

    # create corresponding pv & pvc
    info_log "creating pv & pvc..."
    kubectl apply -f ./resource/helm-chart-repository/harbor-local-pv.yaml

    sudo yq e '.expose.tls.enabled = false' -i ./resource/helm-chart-repository/harbor-local-values.yaml

    # install Harbor Helm chart
    info_log "install Harbor Helm chart ${LOCAL_HARBOR_CHART_VERSION}..."

    sudo sed -i "s|${FUYAO_RGISTRY}|${REGISTRY}|g" ./resource/helm-chart-repository/harbor-local-values.yaml
    
    kubectl apply -f ./resource/helm-chart-repository/harbor-local-secret.yaml
    HARBOR_ADMIN_PASSWORD=$(kubectl get secret harbor-credentials-param -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath="{.data.HARBOR_ADMIN_PASSWORD}" | base64 -d)
    HARBOR_REGISTRY_PASSWORD=$(kubectl get secret harbor-credentials-param -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath="{.data.HARBOR_REGISTRY_PASSWORD}" | base64 -d)
    HARBOR_DATABASE_PASSWORD=$(kubectl get secret harbor-credentials-param -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath="{.data.HARBOR_DATABASE_PASSWORD}" | base64 -d)

    sudo yq e ".harborAdminPassword = \"${HARBOR_ADMIN_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml
    sudo yq e ".registry.credentials.password = \"${HARBOR_REGISTRY_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml
    sudo yq e ".database.internal.password = \"${HARBOR_DATABASE_PASSWORD}\"" -i ./resource/helm-chart-repository/harbor-local-values.yaml

    download_charts_with_retry "${HARBOR_CHART_NAME}" "${LOCAL_HARBOR_CHART_VERSION}"

    local harbor_chart_path="./${CHART_PATH}/${HARBOR_CHART_NAME}-${LOCAL_HARBOR_CHART_VERSION}.tgz"
    # 本地有chart文件则进行离线安装
    if [ -d "${CHART_PATH}" ] && [ -f "$harbor_chart_path" ]; then
        helm install "${HARBOR_RELEASE_NAME}" "$harbor_chart_path" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -f ./resource/helm-chart-repository/harbor-local-values.yaml >> "${OUTPUT}"
    else
        fatal_log "Harbor chart not found"
    fi
    info_log "built-in Harbor installed"
}

function generate_harbor_local_tls() {
    info_log "create local-harbor ca"
    create_service_cert "${LOCAL_HARBOR}" "${LOCAL_HARBOR}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_3_SUFFIX}" "${LOCAL_HARBOR}.${OPENFUYAO_SYSTEM_NAMESPACE}.${DNS_4_SUFFIX}"
    sudo yq e ".data.\"tls.crt\" = \"$(cat ./${FUYAO_CERTS_PATH}/${LOCAL_HARBOR}/${LOCAL_HARBOR}.crt | base64)\"" -i ./resource/helm-chart-repository/harbor-local-tls.yaml
    sudo yq e ".data.\"tls.key\" = \"$(cat ./${FUYAO_CERTS_PATH}/${LOCAL_HARBOR}/${LOCAL_HARBOR}.key | base64)\"" -i ./resource/helm-chart-repository/harbor-local-tls.yaml
    kubectl apply -f ./resource/helm-chart-repository/harbor-local-tls.yaml
    info_log "created local-harbor tls"
}

function install_oauth_webhook_and_oauth_server() {
    info_log "Start installing oauth_webhook and oauth_server"

    if [ ! -d "${FUYAO_WEBHOOK_PATH}" ]; then
        info_log "start generate tls cert"
        ./generate_tls_cert.sh "{{ .oauthCertsExpirationTime }}"
    fi

    if ! grep -q "authentication-token-webhook-config-file" /etc/kubernetes/manifests/kube-apiserver.yaml; then
        info_log "start modify kubernetes manifests"
        ../../utils/set-k8s-manifests.sh
    fi
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
        install_oauth_server "${signing_key}" "${encryption_key}" "${jwt_private_key}"
        info_log "Completing the installation of oauth-server"
    fi
    info_log "Completing the installation of oauth-server and oauth-webhook"
}

function install_oauth_webhook() {
    local jwt_private_key=$1
    local oauth_webhook_chart_path="./${CHART_PATH}/${OAUTH_WEBHOOK_CHART_NAME}-${OAUTH_WEBHOOK_CHART_VERSION}.tgz"
    if [ -d "${CHART_PATH}" ] && [ -f "${oauth_webhook_chart_path}" ]; then
        helm install "${OAUTH_WEBHOOK_RELEASE_NAME}" "${oauth_webhook_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
            --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set images.core.repository="${REGISTRY}"/oauth-webhook \
            --set images.core.tag="${OAUTH_WEBHOOK_IMAGE_TAG}" \
            --set config.JWTPrivateKey="${jwt_private_key}" >> "${OUTPUT}"
    else
        fatal_log "oauth_webhook chart not found"
    fi
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
    info_log "start installing oauth_server disable https"
    if [ -d "${CHART_PATH}" ] && [ -f "${oauth_server_chart_path}" ]; then
        helm install "${OAUTH_SERVER_RELEASE_NAME}" "${oauth_server_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
            --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set images.core.repository="${REGISTRY}"/oauth-server \
            --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}" \
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao" \
            --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}" \
            --set config.httpServerConfig.enableHttps=false \
            --set config.loginSessionConfig.signingKey="${signing_key}",config.loginSessionConfig.encryptionKey="${encryption_key}",config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}" \
            --set config.httpServerConfig.enableHttps=falsee >> "${OUTPUT}"
    else
        fatal_log "oauth_server chart not found"
    fi
}

function install_oauth_server_enable_https() {
    info_log "start installing oauth_server enable https"
    if [ -d "${CHART_PATH}" ] && [ -f "${oauth_server_chart_path}" ]; then
        helm install "${OAUTH_SERVER_RELEASE_NAME}" "${oauth_server_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
            --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
            --set images.core.repository="${REGISTRY}"/oauth-server \
            --set images.core.tag="${OAUTH_SERVER_IMAGE_TAG}" \
            --set images.kubectl.repository="${REGISTRY}/kubectl-openfuyao" \
            --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}" \
            --set config.loginSessionConfig.signingKey="${signing_key}",config.loginSessionConfig.encryptionKey="${encryption_key}",config.oauthServerConfig.JWTPrivateKey="${jwt_private_key}" \
            --set config.httpServerConfig.enableHttps=true \
            --set-file config.httpServerConfig.tlsCert="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.crt",config.httpServerConfig.tlsKey="./${FUYAO_CERTS_PATH}/${OAUTH_SERVER}/${OAUTH_SERVER}.key",config.httpServerConfig.rootCA="./${FUYAO_CERTS_PATH}/ca.crt" >> "${OUTPUT}"
    else
        fatal_log "oauth_server chart not found"
    fi
}

function install_marketplace_service() {
    info_log "Start installing marketplace_service"

    download_charts_with_retry "${MARKETPLACE_SERVICE_CHART_NAME}" "${MARKETPLACE_SERVICE_CHART_VERSION}"
    local marketplace_service_chart_path="./${CHART_PATH}/${MARKETPLACE_SERVICE_CHART_NAME}-${MARKETPLACE_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${marketplace_service_chart_path}" ]; then
        fatal_log "marketplace_service chart not found"
    fi

    install_marketplace_service_disable_https
}

function install_marketplace_service_disable_https() {
    info_log "disable https for marketplace_service"
    helm install "${MARKETPLACE_SERVICE_RELEASE_NAME}" "${marketplace_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --set images.core.repository="${REGISTRY}"/marketplace-service \
        --set images.core.tag="${MARKETPLACE_SERVICE_IMAGE_TAG}" \
        --set config.enableHttps=false,config.insecureSkipVerify=true \
        --set enableOAuth=true \
        --set symmetricKey.secretKey="$(openssl rand -base64 32)" \
        --set images.oauthProxy.repository="${REGISTRY}"/oauth-proxy \
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}" \
        --set thirdPartyImages.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
        --set thirdPartyImages.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" >> "${OUTPUT}"
}

function install_plugin_management_service() {
    info_log "Start installing plugin_management_service"

    download_charts_with_retry "${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}" "${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}"
    local plugin_management_service_chart_path="./${CHART_PATH}/${PLUGIN_MANAGEMENT_SERVICE_CHART_NAME}-${PLUGIN_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${plugin_management_service_chart_path}" ]; then
        fatal_log "plugin_management_service chart not found"
    fi

    install_plugin_management_service_disable_https
    info_log "Completing the installation of plugin_management_service"
}

function install_plugin_management_service_disable_https() {
    info_log "disable https for plugin_management_service"
    helm install "${PLUGIN_MANAGEMENT_SERVICE_RELEASE_NAME}" "${plugin_management_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --set config.enableHttps=false \
        --set enableOAuth=true \
        --set images.core.repository="${REGISTRY}"/plugin-management-service \
        --set images.core.tag="${PLUGIN_MANAGEMENT_SERVICE_IMAGE_TAG}" \
        --set images.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
        --set images.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" \
        --set images.oauthProxy.repository="${REGISTRY}"/oauth-proxy \
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}"
}

function install_application_management_service() {
    info_log "Start installing application_management_service"

    download_charts_with_retry "${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}" "${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}"
    local application_management_service_chart_path="./${CHART_PATH}/${APPLICATION_MANAGEMENT_SERVICE_CHART_NAME}-${APPLICATION_MANAGEMENT_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${application_management_service_chart_path}" ]; then
        fatal_log "application_management_service chart not found"
    fi

    install_application_management_service_disable_https
}

function install_application_management_service_disable_https() {
    info_log "disable https for application_management_service"
    helm install "${APPLICATION_MANAGEMENT_SERVICE_RELEASE_NAME}" "${application_management_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --set enableOAuth=true \
        --set images.oauthProxy.repository="${REGISTRY}"/oauth-proxy \
        --set images.oauthProxy.tag="${OAUTH_PROXY_IMAGE_TAG}" \
        --set images.core.repository="${REGISTRY}"/application-management-service \
        --set images.core.tag="${APPLICATION_MANAGEMENT_SERVICE_IMAGE_TAG}" \
        --set images.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
        --set images.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" >> "${OUTPUT}"
}

function install_user_management_operator() {
    info_log "Start installing user_management_operator"
    download_charts_with_retry "${USER_MANAGEMENT_OPERATOR_CHART_NAME}" "${USER_MANAGEMENT_OPERATOR_CHART_VERSION}"
    local user_management_operator_chart_path="./${CHART_PATH}/${USER_MANAGEMENT_OPERATOR_CHART_NAME}-${USER_MANAGEMENT_OPERATOR_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${user_management_operator_chart_path}" ]; then
        fatal_log "user_management_operator chart not found"
    fi

    helm install "${USER_MANAGEMENT_OPERATOR_RELEASE_NAME}" "${user_management_operator_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
            --set images.core.repository="${REGISTRY}"/user-management-operator \
            --set images.core.tag="${USER_MANAGEMENT_OPERATOR_IMAGE_TAG}" >> "${OUTPUT}"
}

function install_web_terminal_service() {
    info_log "Start installing web_terminal_service"
    download_charts_with_retry "${WEB_TERMINAL_SERVICE_CHART_NAME}" "${WEB_TERMINAL_SERVICE_CHART_VERSION}"
    local web_terminal_service_chart_path="./${CHART_PATH}/${WEB_TERMINAL_SERVICE_CHART_NAME}-${WEB_TERMINAL_SERVICE_CHART_VERSION}.tgz"
    if [ ! -d "${CHART_PATH}" ] || [ ! -f "${web_terminal_service_chart_path}" ]; then
        fatal_log "web_terminal_service chart not found"
    fi

    helm install "${WEB_TERMINAL_SERVICE_RELEASE_NAME}" "${web_terminal_service_chart_path}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --set images.core.repository="${REGISTRY}"/web-terminal-service \
        --set images.core.tag="${WEB_TERMINAL_SERVICE_IMAGE_TAG}" \
        --set images.kubectl.repository="${REGISTRY}"/kubectl-openfuyao \
        --set images.kubectl.tag="${KUBECTL_OPENFUYAO_IMAGE_TAG}" \
        --set thirdPartyImage.busyBox.repository="${BUSY_BOX_REPOSITORY}" \
        --set thirdPartyImage.busyBox.tag="${BUSY_BOX_IMAGE_TAG}" >> "${OUTPUT}"
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
    if ! kubectl get namespace | grep -q "${namespace}"; then
        kubectl create namespace "${namespace}"
    fi
}

function create_root_ca() {
    # 创建根证书目录
    if [ ! -d "${FUYAO_CERTS_PATH}" ]; then
        mkdir -p "${FUYAO_CERTS_PATH}"
    fi

    cd "${FUYAO_CERTS_PATH}"  || fatal_log "failed to change directory to ${FUYAO_CERTS_PATH}"

    if [  -f "ca.key" ] && [ -f "ca.crt" ]; then
        info_log "fuyao ca.key already exists"
        if ! kubectl get secret -n $OPENFUYAO_SYSTEM_NAMESPACE | grep $OPENFUYAO_SYSTEM_ROOT_CA_SECRET; then
          create_root_ca_secret
        fi
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
    openssl genrsa -out ${service_name}.key 4096

    # 创建CSR配置文件 mypod-csr.conf
    cat > ${service_name}-csr.conf <<EOF
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

    info_log "create ingress-nginx tls secret"
    create_service_cert "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_3_SUFFIX}" "${INGRESS_NGINX_CONTROLLER}.${INGRESS_NGINX_NAMESPACE}.${DNS_4_SUFFIX}"

    if kubectl get secret -n ${INGRESS_NGINX_NAMESPACE} | grep ${INGRESS_NGINX_TLS_SECRET}; then
        info_log "ingress-nginx tls secret already exists"
        return
    fi

    kubectl create secret generic ${INGRESS_NGINX_TLS_SECRET} --namespace ${INGRESS_NGINX_NAMESPACE} \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    kubectl create secret generic ${INGRESS_NGINX_FRONT_TLS_SECRET} --namespace ${INGRESS_NGINX_NAMESPACE} \
      --from-file=tls.key="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.key" \
      --from-file=tls.crt="./${FUYAO_CERTS_PATH}/${INGRESS_NGINX_CONTROLLER}/${INGRESS_NGINX_CONTROLLER}.crt" \
      --from-file=ca.crt="./${FUYAO_CERTS_PATH}/ca.crt"
    info_log "ingress-nginx tls secret created"
}

function add_helm_repo() {
    curl "${FUYAO_REPO}" -k --connect-timeout 20 >> "${OUTPUT}"
    if [ $? -ne 0 ]; then
        info_log "offline mode, skip add helm repo"
        return
    fi
    helm repo add "${OPENFUYAO}" "${FUYAO_REPO}"
    helm repo update
}

create_namespace "${OPENFUYAO_SYSTEM_NAMESPACE}"
create_root_ca
create_ingress_nginx_tls_secret
add_helm_repo
install_console_website
install_helm_chart_repository
install_metrics_server
install_kube_prometheus
install_monitoring_service
install_console_service
install_marketplace_service
install_application_management_service
install_plugin_management_service
install_oauth_webhook_and_oauth_server
install_user_management_operator
install_web_terminal_service

./postinstall.sh
