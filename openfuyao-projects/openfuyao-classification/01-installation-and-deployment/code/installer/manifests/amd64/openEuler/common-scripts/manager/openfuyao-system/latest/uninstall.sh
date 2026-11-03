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

function uninstall_console_website() {
    info_log "Start uninstalling console_website"
    helm uninstall "${CONSOLE_WEBSITE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "console_website uninstalled"
}

function uninstall_console_service() {
    info_log "Start uninstalling console_service"
    helm uninstall "${CONSOLE_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "console_service uninstalled"
}

function uninstall_metrics_server() {
    info_log "Start uninstalling metrics-server"
    sudo tar -xvf ./yaml/metrics-server-yaml.tar.gz -C ./yaml
    kubectl delete -f ./yaml/metrics-server-yaml/metrics-server.yaml
}

function uninstall_kube_prometheus() {
    info_log "Start uninstalling kube_prometheus"
    sudo tar -xvf ./yaml/kube-prometheus-yaml.tar.gz -C ./yaml
    kubectl delete --ignore-not-found=true -f ./yaml/kube-prometheus/ \
      ./yaml/kube-prometheus/setup

    kubectl delete --ignore-not-found=true -f ./yaml/kube-prometheus/kubernetes-components-service/
    kubectl delete secret etcd-certs --namespace="${MONITOR_NAMESPACE}"
    kubectl delete namespace "${MONITOR_NAMESPACE}"
    info_log "kube_prometheus uninstalled"
}

function uninstall_monitoring_service() {
    info_log "Start uninstalling monitoring_service"
    helm uninstall "${MONITORING_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "monitoring_service uninstalled"
}

function uninstall_helm_chart_repository() {
    info_log "Start uninstalling helm_chart_repository"
    # built-in harbor helm release name

    #get Helm release pods installed node
    node_name=$(kubectl get nodes -l "$LABEL_KEY" --output=jsonpath='{.items[0].metadata.name}')

    # Check if node_name is empty
    if [ -z "$node_name" ]; then
      echo "No node found with the $LABEL_KEY label."
      exit 1
    fi

    # remove release
    info_log "uninstalling built-in harbor helm release..."
    helm uninstall "${HARBOR_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"

    # create destroyContainer job yaml
    cat <<EOF > harbor-destroy.yaml
    apiVersion: batch/v1
    kind: Job
    metadata:
      name: destroy-harbor
      namespace: $OPENFUYAO_SYSTEM_NAMESPACE
    spec:
      template:
        spec:
          nodeName: $node_name
          containers:
          - name: destroy-harbor-container
            image: {{ .registry }}/busybox/busybox:1.36.1
            command:
              - sh
              - -c
              - |
                set -e
                rm -rf /data/harbor

                echo "remove complete"
            volumeMounts:
            - name: data
              mountPath: /data
          restartPolicy: OnFailure
          volumes:
          - name: data
            hostPath:
              path: /data
EOF

    # remove directory
    info_log "removing /data/harbor/ directory..."
    kubectl apply -f harbor-destroy.yaml

    # wait job to be finished
    info_log "waiting job finished..."
    kubectl wait --for=condition=complete --timeout=1800s job/destroy-harbor -n "${OPENFUYAO_SYSTEM_NAMESPACE}"

    # check whether destroyContainer job succeed
    job_status=$(kubectl get job destroy-harbor -n "${OPENFUYAO_SYSTEM_NAMESPACE}" -o jsonpath='{.status.succeeded}')
    if [ "$job_status" != "1" ]; then
      error_log "destroyContainer job failed，please check logs"
      kubectl logs -n "${OPENFUYAO_SYSTEM_NAMESPACE}" job/destroy-harbor >> ${OUTPUT}
      exit 1
    fi

    # remove the node label with the label key
    info_log "removing label in node $node_name..."
    kubectl label nodes "$node_name" "$LABEL_KEY"-

    # remove pv & pvc
    info_log "removing pv & pvc..."
    kubectl delete -f ./resource/helm-chart-repository/harbor-local-pv.yaml

    info_log "built-in Harbor uninstalled"
}

function uninstall_oauth_webhook_and_oauth_server() {
    info_log "Start uninstalling oauth_webhook_and_oauth_server"
    local namespace="${OPENFUYAO_SYSTEM_NAMESPACE}"

    sudo rm -f /usr/local/bin/cfssl
    sudo rm -f /usr/local/bin/cfssljson
    sudo rm -f /usr/local/bin/cfssl-certinfo
    sudo rm -f /usr/local/bin/jq

    helm uninstall "${OAUTH_SERVER_RELEASE_NAME}" -n ${namespace}
    helm uninstall "${OAUTH_WEBHOOK_RELEASE_NAME}" -n ${namespace}
    kubectl delete namespace ${namespace}
}

function uninstall_marketplace_service() {
    info_log "Start uninstalling marketplace_service"
    helm uninstall "${MARKETPLACE_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "marketplace_service uninstalled"
}

function uninstall_application_management_service() {
    info_log "Start uninstalling application_management_service"
    helm uninstall "${APPLICATION_MANAGEMENT_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "application_management_service uninstalled"
}

function uninstall_plugin_management_service() {
    info_log "Start uninstalling plugin_management_service"
    helm uninstall "${PLUGIN_MANAGEMENT_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "plugin_management_service uninstalled"
}

function uninstall_user_management_operator() {
    info_log "Start uninstalling user_management_operator"
    helm uninstall "${USER_MANAGEMENT_OPERATOR_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "user_management_operator uninstalled"
}

function uninstall_web_terminal_service() {
    info_log "Start uninstalling web_terminal_service"
    helm uninstall "${WEB_TERMINAL_SERVICE_RELEASE_NAME}" -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
    info_log "web_terminal_service uninstalled"
}

uninstall_console_website
uninstall_console_service
uninstall_metrics_server
uninstall_kube_prometheus
uninstall_monitoring_service
uninstall_helm_chart_repository
uninstall_marketplace_service
uninstall_application_management_service
uninstall_plugin_management_service
uninstall_oauth_webhook_and_oauth_server
uninstall_user_management_operator
uninstall_web_terminal_service

kubectl delete namespace "${OPENFUYAO_SYSTEM_NAMESPACE}"
kubectl delete namespace "${SESSION_SECRET_NAMESPACE}"
