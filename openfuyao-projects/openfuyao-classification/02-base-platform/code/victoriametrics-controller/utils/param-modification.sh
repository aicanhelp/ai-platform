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

# vmsingle模式下修改values.yaml文件
function modify_vmsingle_values_yaml() {
    info_log "Use victoria metrics single mode"

    yq eval '.vmsingle.enabled = true |
      .vmcluster.enabled = false |
      .vmsingle.spec.storage.resources.requests.storage = "'"$vmSingleStorageSize"'"
      ' -i ${VM_VALUES_YAML}

    info_log "Successfully modify victoria metrics single mode values.yaml"
}

# vmcluster模式下修改values.yaml文件
function modify_vmcluster_values_yaml() {
    info_log "Use victoria metrics cluster mode"
    export VMClusterReplicationFactor=\"${vmClusterReplicationFactor}\"
    export VMStorageReplicaCount=\"${vmStorageReplicaCount}\"
    export VMSelectReplicaCount=\"${vmSelectReplicaCount}\"
    export VMInsertReplicaCount=\"${vmInsertReplicaCount}\"
    VMClusterReplicationFactor=${VMClusterReplicationFactor//\"/} 
    VMStorageReplicaCount=${VMStorageReplicaCount//\"/} 
    VMSelectReplicaCount=${VMSelectReplicaCount//\"/}
    VMInsertReplicaCount=${VMInsertReplicaCount//\"/}

    yq eval '.vmsingle.enabled = false |
      .vmcluster.enabled = true |
      .vmcluster.spec.retentionPeriod = "'"$vmClusterRetentionPeriod"'" |
      .vmcluster.spec.replicationFactor = env(VMClusterReplicationFactor) |
      .vmcluster.spec.vmstorage.replicaCount = env(VMStorageReplicaCount) |
      .vmcluster.spec.vmstorage.storage.volumeClaimTemplate.spec.resources.requests.storage = "'"$vmStorageStorageSize"'" |
      .vmcluster.spec.vmstorage.resources.requests.memory = "'"$vmStorageMemorySize"'" |
      .vmcluster.spec.vmstorage.resources.requests.cpu = "'"$vmStorageCPUCount"'" |
      .vmcluster.spec.vmstorage.resources.limits.memory = "'"$vmStorageMemorySize"'" |
      .vmcluster.spec.vmstorage.resources.limits.cpu = "'"$vmStorageCPUCount"'" |
      .vmcluster.spec.vmstorage.image.repository = "victoriametrics/vmstorage" |
      .vmcluster.spec.vmstorage.image.pullPolicy = "IfNotPresent" |
      .vmcluster.spec.vmstorage.extraArgs = {"dedup.minScrapeInterval": "20s", "inmemoryDataFlushInterval" = "10s"} |
      .vmcluster.spec.vmselect.replicaCount = env(VMSelectReplicaCount) |
      .vmcluster.spec.vmselect.resources.requests.memory = "'"$vmSelectMemorySize"'" |
      .vmcluster.spec.vmselect.resources.requests.cpu = "'"$vmSelectCpuCount"'" |
      .vmcluster.spec.vmselect.resources.limits.memory = "'"$vmSelectMemorySize"'" |
      .vmcluster.spec.vmselect.resources.limits.cpu = "'"$vmSelectCpuCount"'" |
      .vmcluster.spec.vmselect.storage.volumeClaimTemplate.spec.resources.requests.storage = "'"$vmSelectStorageSize"'" |
      .vmcluster.spec.vmselect.extraArgs = {"search.maxQueryDuration": "120s"} |
      .vmcluster.spec.vmselect.image.repository = "victoriametrics/vmselect" |
      .vmcluster.spec.vmselect.image.pullPolicy = "IfNotPresent" |
      .vmcluster.spec.vminsert.replicaCount = env(VMInsertReplicaCount) |
      .vmcluster.spec.vminsert.resources.requests.memory = "'"$vmInsertMemorySize"'" |
      .vmcluster.spec.vminsert.resources.requests.cpu = "'"$vmInsertCpuCount"'" |
      .vmcluster.spec.vminsert.resources.limits.memory = "'"$vmInsertMemorySize"'" |
      .vmcluster.spec.vminsert.resources.limits.cpu = "'"$vmInsertCpuCount"'" |
      .vmcluster.spec.vminsert.image.repository = "victoriametrics/vminsert" |
      .vmcluster.spec.vminsert.image.pullPolicy = "IfNotPresent" |
      .vmcluster.spec.vminsert.extraArgs = {"insert.maxQueueDuration": "2m0s"}
      ' -i ${VM_VALUES_YAML}

    yq -e ".vmcluster.spec.vmstorage.image.tag = \"${VM_CLUSTER_IMAGE_TAG}\"" -i ${VM_VALUES_YAML}
    yq -e ".vmcluster.spec.vmselect.image.tag = \"${VM_CLUSTER_IMAGE_TAG}\"" -i ${VM_VALUES_YAML}
    yq -e ".vmcluster.spec.vminsert.image.tag = \"${VM_CLUSTER_IMAGE_TAG}\"" -i ${VM_VALUES_YAML}

    info_log "Successfully modify victoria metrics cluster mode values.yaml"
}

# 修改alertmanager的values.yaml文件
function modify_alertmanager_values_yaml() {
    export VMAlertManagerReplicaCount=\"${vmAlertManagerReplicaCount}\"
    VMAlertManagerReplicaCount=${VMAlertManagerReplicaCount//\"/} 

    yq eval '
      .alertmanager.labels.dedicated = "dedicated" |
      .alertmanager.spec.replicaCount = env(VMAlertManagerReplicaCount) |
      .alertmanager.spec.image.repository = "prom/alertmanager" |
      .alertmanager.spec.image.pullPolicy = "IfNotPresent" |
      .alertmanager.spec.resources.requests.memory = "'"$vmAlertManagerMemorySize"'" |
      .alertmanager.spec.resources.requests.cpu = "'"$vmAlertManagerCpuCount"'" |
      .alertmanager.spec.resources.limits.memory = "'"$vmAlertManagerMemorySize"'" |
      .alertmanager.spec.resources.limits.cpu = "'"$vmAlertManagerCpuCount"'" 
    ' -i ${VM_VALUES_YAML}

    info_log "Successfully modify alertmanager values.yaml"
}

# 修改vmalert的values.yaml文件
function modify_vmalert_values_yaml() {
    export VMAlertReplicaCount=\"${vmAlertReplicaCount}\"
    VMAlertReplicaCount=${VMAlertReplicaCount//\"/} 

    yq eval '.vmalert.spec.replicaCount = env(VMAlertReplicaCount) |
      .vmalert.spec.notifiers[0].selector.labelSelector.matchLabels.usage = "dedicated" |
      .vmalert.spec.image.repository = "victoriametrics/vmalert" |
      .vmalert.spec.image.pullPolicy = "IfNotPresent" |
      .vmalert.spec.image.tag = "v1.122.0" |
      .vmalert.spec.resources.requests.memory = "'"$vmAlertMemorySize"'" |
      .vmalert.spec.resources.requests.cpu = "'"$vmAlertCpuCount"'" |
      .vmalert.spec.resources.limits.memory = "'"$vmAlertMemorySize"'" |
      .vmalert.spec.resources.limits.cpu = "'"$vmAlertCpuCount"'"
    ' -i ${VM_VALUES_YAML}

    yq -e ".vmalert.spec.notifiers[0].selector.namespaceSelector.matchNames[0] = \"${VM_NAMESPACE}\"" -i ${VM_VALUES_YAML}

    info_log "Successfully modify vmalert values.yaml"
}

# 修改vmagent的values.yaml文件
function modify_vmagent_values_yaml() {
    yq eval '
      .vmagent.spec.image.repository = "victoriametrics/vmagent" |
      .vmagent.spec.image.pullPolicy = "IfNotPresent" |
      .vmagent.spec.image.tag = "v1.122.0" 
    ' -i ${VM_VALUES_YAML}

    export VMAgentReplicaCount=\"${vmAgentReplicaCount}\"
    export VMAgentShareCount=\"${vmAgentShareCount}\"
    VMAgentReplicaCount=${VMAgentReplicaCount//\"/} 
    VMAgentShareCount=${VMAgentShareCount//\"/}

    yq eval '
      .vmagent.extraSecretMounts[0].name = "vmagent-tls-secrets" |
      .vmagent.extraSecretMounts[0].mountPath = "/etc/vmagent/tls" |
      .vmagent.extraSecretMounts[0].readOnly = true |
      .vmagent.extraSecretMounts[0].secretName = "vmagent-tls-secrets" |
      .vmagent.spec.additionalScrapeConfigs.name = "etcd-scrape-config" |
      .vmagent.spec.additionalScrapeConfigs.key = "etcd-scrape-config.yaml" |
      .vmagent.spec.volumeMounts[0].mountPath = "/etc/vmagent/tls" |
      .vmagent.spec.volumeMounts[0].name = "vmagent-tls-certs" |
      .vmagent.spec.volumes[0].name = "vmagent-tls-certs" |
      .vmagent.spec.volumes[0].secret.secretName = "vmagent-tls-secrets" |
      .vmagent.spec.nodeSelector = {"monitoring.victoria.com/vmagent": "vmagent"} |
      .vmagent.spec.resources.requests.memory = "'"$vmAgentMemorySize"'" |
      .vmagent.spec.resources.requests.cpu = "'"$vmAgentCpuCount"'" |
      .vmagent.spec.resources.limits.memory = "'"$vmAgentMemorySize"'" |
      .vmagent.spec.resources.limits.cpu = "'"$vmAgentCpuCount"'" |   
      .vmagent.spec.replicaCount = env(VMAgentReplicaCount) |
      .vmagent.spec.shardCount = env(VMAgentShareCount)|
      .vmagent.spec.scrapeInterval = "'"$vmAgentScrapeInterval"'" |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].weight = 100 |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.labelSelector.matchLabels = {"app.kubernetes.io/name": "vmagent", "shard-num": "0"  } |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[0].podAffinityTerm.topologyKey = "kubernetes.io/hostname" |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[1].weight = 100 |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[1].podAffinityTerm.labelSelector.matchLabels = {"app.kubernetes.io/name": "vmagent", "shard-num": "1"  } |
      .vmagent.spec.affinity.podAntiAffinity.preferredDuringSchedulingIgnoredDuringExecution[1].podAffinityTerm.topologyKey = "kubernetes.io/hostname"
    ' -i ${VM_VALUES_YAML}

    if ${vmAgentAllowStatefulSet} == "true"; then
        yq eval '.vmagent.spec.statefulMode = true' -i ${VM_VALUES_YAML}
        yq eval '.vmagent.spec.statefulStorage.volumeClaimTemplate.spec.resources.requests.storage = "'"$vmAgentStorageSize"'"' -i ${VM_VALUES_YAML}

    else
        yq eval '.vmagent.spec.statefulMode = false' -i ${VM_VALUES_YAML}
    fi

    info_log "Successfully modify vmagent values.yaml"
}

# 修改kube-state-metrics的values.yaml文件
function modify_kube_state_metrics_values_yaml() {

    yq eval '.global.imageRegistry = "'"$imageRepo"'" |
      .autosharding.enabled = "'"$kubeStateMetricsAutoSharding"'" |
      .replicas = "'"$kubeStateMetricsReplicaCount"'" |
      .extraArgs += "--pod=\"$(POD_NAME)\"" |
      .extraArgs += "--pod-namespace=\"$(POD_NAMESPACE)\"" |
      .extraArgs += "--use-apiserver-cache=true" |
      .nodeSelector = {"monitoring.victoria.com/metrics": "kube-state-metrics"} |
      .affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].key = "app.kubernetes.io/name" |
      .affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].operator = "In" |
      .affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].labelSelector.matchExpressions[0].values[0] = "kube-state-metrics" |
      .affinity.podAntiAffinity.requiredDuringSchedulingIgnoredDuringExecution[0].topologyKey = "kubernetes.io/hostname" |
      .resources.requests.memory = "'"$kubeStateMetricsMemorySize"'" |
      .resources.requests.cpu = "'"$kubeStateMetricsCpuCount"'" |
      .resources.limits.memory = "'"$kubeStateMetricsMemorySize"'" |
      .resources.limits.cpu = "'"$kubeStateMetricsCpuCount"'"
      ' -i ${CHART_PATH}/${VMS_CHART_NAME}/charts/kube-state-metrics/values.yaml

    info_log "Successfully modify kube-state-metrics values.yaml"
}

# 修改node-exporter的values.yaml文件
function modify_node_exporter_values_yaml() {
    yq eval '.global.imageRegistry = "'"$imageRepo"'"
        ' -i ${CHART_PATH}/${VMS_CHART_NAME}/charts/victoria-metrics-operator/values.yaml
    info_log "Successfully modify node-exporter values.yaml"
}

# 修改grafana的values.yaml文件
function modify_grafana_values_yaml() {
    export GrafanaNodePort=\"${grafanaNodePort}\"
    GrafanaNodePort=${GrafanaNodePort//\"/} 
    yq eval '.global.imageRegistry = "'"$imageRepo"'" |
      .nodeSelector = {"monitoring.victoria.com/vm-grafana": "vm-grafana"} |
      .service.type = "NodePort" |
      .service.nodePort = env(GrafanaNodePort) |
      .defaultDashboards.dashboards.kubernetes-views-nodes-static.enabled = true |
      .defaultDashboards.dashboards.kubernetes-views-nodes.enabled = false
      ' -i ${CHART_PATH}/${VMS_CHART_NAME}/charts/grafana/values.yaml

    info_log "Successfully modify grafana values.yaml"
}

# 修改vm-operator的values.yaml文件
function modify_vm_operator_values_yaml() {
    yq eval '.victoria-metrics-operator.crds.cleanup.image.repository = "'"$imageRepo/rancher/kubectl"'"
      .victoria-metrics-operator.crds.cleanup.image.tag = "v1.34.3" 
      ' -i ${CHART_PATH}/${VMS_CHART_NAME}/values.yaml

    yq eval '.global.image.registry = "'"$imageRepo"'" |
      .crds.cleanup.enabled = true |
      .admissionWebhooks.enabled = false |
      .crds.cleanup.image.registry = "'"$imageRepo"'" 
      ' -i ${CHART_PATH}/${VMS_CHART_NAME}/charts/victoria-metrics-operator/values.yaml

    info_log "Successfully modify vm-operator values.yaml"
}

# 修改k8s-components的values.yaml文件
function modify_k8s_components_values_yaml() {
    yq eval '.kubeProxy.enabled = true |
      .kubeProxy.vmScrape.spec.endpoints[0].scheme = "http" |
      .kubeEtcd.enabled = true
      ' -i ${VM_VALUES_YAML}

    info_log "Successfully modify k8s-components values.yaml"
}

# 修改values.yaml文件
function modify_values_yaml() {
    info_log "Modifying values.yaml"

    # 能在values.yaml中配置nodeSelector组件的nodeSelector都在values.yaml中直接写了
    if ${useVMSingle} == "true"; then
        modify_vmsingle_values_yaml
    else
        modify_vmcluster_values_yaml
    fi

    modify_alertmanager_values_yaml
    modify_vmalert_values_yaml
    modify_vmagent_values_yaml
    modify_kube_state_metrics_values_yaml
    modify_node_exporter_values_yaml
    modify_grafana_values_yaml
    modify_vm_operator_values_yaml
    modify_k8s_components_values_yaml
}