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

# vmagent 所有相关配置在values文件中已经完成
# etcd 相关配置
function configure_for_etcd() {
    # 1. 等待 Deployment 滚动完成
    kubectl -n ${VM_NAMESPACE} rollout status deployment/vm-victoria-metrics-operator --timeout=300s

    crd_name="vm-victoria-metrics-k8s-stack-kube-etcd"

    # 2. 使用 yq 修改配置
    kubectl get vmservicescrapes $crd_name -n ${VM_NAMESPACE} -o yaml | yq -e '
    .spec.endpoints[0].tlsConfig.caFile = "/etc/vmagent/tls/etcd-ca-vm.crt" |
    .spec.endpoints[0].tlsConfig.certFile = "/etc/vmagent/tls/etcd-client-vm.crt" |
    .spec.endpoints[0].tlsConfig.keyFile = "/etc/vmagent/tls/etcd-client-vm.key"
    ' | kubectl apply -f -

    etcd=${etcdNodeIP}
    etcdEventIP=${etcdDataNodeIP}
    etcdDataIP=${etcdEventsNodeIP}

    if [[  "$etcd" != ""  ]]; then
        IFS=',' read -r -a nodes <<<"$etcd"
        targets=""
        for ip in "${nodes[@]}"; do
            targets+="      - ${ip}:2379"$'\n'
        done
    elif [[ $etcdEventIP != "" && $etcdDataIP != "" ]]; then
        IFS=',' read -r -a Eventnodes <<<"$etcdEventIP"
        IFS=',' read -r -a Datanodes <<<"$etcdDataIP"
        targets=""
        for ip in "${Eventnodes[@]}"; do
            targets+="      - ${ip}:2379"$'\n'
        done
        targets=""
        for ip in "${Datanodes[@]}"; do
            targets+="      - ${ip}:2379"$'\n'
        done
    fi

    # 创建 YAML 配置文件
    cat > etcd-scrape-config.yaml << EOF
- job_name: 'etcd-external'
  kubernetes_sd_configs: []
  static_configs:
    - targets:
${targets}  scheme: https
  tls_config:
    ca_file: /etc/vmagent/tls/etcd-ca-vm.crt  
    cert_file: /etc/vmagent/tls/etcd-client-vm.crt  
    key_file: /etc/vmagent/tls/etcd-client-vm.key  
EOF

    # 创建对应的 Secret
    kubectl create secret generic etcd-scrape-config \
    --namespace=${VM_NAMESPACE} \
    --from-file=etcd-scrape-config.yaml \
    --dry-run=client -o yaml | kubectl apply -f -

    # 延迟1秒，防止执行下一操作安装失败
    sleep 1

}

# Controller-manager 相关配置
function configure_for_controller_manager() {
    # 生成私钥controller-manager.key
    openssl genrsa -out /etc/kubernetes/pki/kube-controller-manager.key 2048

    # 使用 yq 修改配置crd
    crd_name="vm-victoria-metrics-k8s-stack-kube-controller-manager"
    kubectl get vmservicescrapes $crd_name -n ${VM_NAMESPACE} -o yaml | yq -e '
    .spec.endpoints[0].tlsConfig.caFile = "/etc/vmagent/tls/k8s-ca-vm.crt" |
    del(.spec.endpoints[0].tlsConfig.serverName) 
    ' | kubectl apply -f -
    # 创建 CSR 配置文件
    ControllerIPs=${controllerManagerIP}
    grep -v "# IP_PLACEHOLDER" "./resources/default-controller-manager-csr.conf" > "controller-manager-csr.conf"
    # 添加多个Controller-manager节点 IP
    IFS=',' read -r -a IPs <<< "$ControllerIPs"
    for i in "${!IPs[@]}"; do
        echo "IP.$((i+2)) = ${ControllerIPs[i]}" >> "controller-manager-csr.conf"
    done


    # 生成 CSR 文件
    openssl req -new -key /etc/kubernetes/pki/kube-controller-manager.key \
    -out /etc/kubernetes/pki/kube-controller-manager.csr \
    -config controller-manager-csr.conf
    # 使用 CA 签发证书
    openssl x509 -req -in /etc/kubernetes/pki/kube-controller-manager.csr \
    -CA /etc/kubernetes/pki/ca.crt -CAkey /etc/kubernetes/pki/ca.key \
    -CAcreateserial -out /etc/kubernetes/pki/kube-controller-manager.crt \
    -days 3650 -extensions v3_req -extfile controller-manager-csr.conf
    # 修改controller-manager启动参数的部分在外部执行。进入每一个Controller-manager节点执行
}


# kube-scheduler相关配置
function configure_for_kube_scheduler() {
    # 使用 yq 修改配置crd
    crd_name="vm-victoria-metrics-k8s-stack-kube-scheduler"
    kubectl get vmservicescrapes $crd_name -n ${VM_NAMESPACE} -o yaml | yq -e '
    .spec.endpoints[0].tlsConfig.caFile = "/etc/vmagent/tls/k8s-ca-vm.crt" 
    ' | kubectl apply -f -

    # 生成私钥scheduler.key
    openssl genrsa -out /etc/kubernetes/pki/kube-scheduler.key 2048

    # 创建 CSR 配置文件
    SchedulerIPs=${kubeSchedulerIP}
    grep -v "# IP_PLACEHOLDER" "./resources/default-scheduler-csr.conf" > "scheduler-csr.conf"
    # 添加多个kube-scheduler节点 IP
    IFS=',' read -r -a IPs <<< "$SchedulerIPs"
    for i in "${!IPs[@]}"; do
        echo "IP.$((i+2)) = ${SchedulerIPs[i]}" >> "scheduler-csr.conf"
    done

    # 生成 CSR 文件
    openssl req -new -key /etc/kubernetes/pki/kube-scheduler.key \
    -out /etc/kubernetes/pki/kube-scheduler.csr \
    -config scheduler-csr.conf
    # 使用 CA 签发证书
    openssl x509 -req -in /etc/kubernetes/pki/kube-scheduler.csr \
    -CA /etc/kubernetes/pki/ca.crt -CAkey /etc/kubernetes/pki/ca.key \
    -CAcreateserial -out /etc/kubernetes/pki/kube-scheduler.crt \
    -days 3650 -extensions v3_req -extfile scheduler-csr.conf
    # 修改kube-scheduler启动参数的部分在外部执行。进入每一个kube-scheduler节点执行
}

function configure_certificates_secrets() {
    # 获取 etcd 证书
    get_certificates_secrets
    # 创建 vmagent-tls-secrets Secret，用于存储 etcd 和 k8s 的证书
    kubectl -n ${VM_NAMESPACE} create secret generic vmagent-tls-secrets \
    --from-file=etcd-ca-vm.crt=/etc/kubernetes/pki/etcd/ca.crt \
    --from-file=etcd-client-vm.crt=/etc/kubernetes/pki/etcd/healthcheck-client.crt \
    --from-file=etcd-client-vm.key=/etc/kubernetes/pki/etcd/healthcheck-client.key \
    --from-file=k8s-ca-vm.crt=/etc/kubernetes/pki/ca.crt

    # 延迟1秒，防止执行下一操作安装失败
    info_log "successfully create secret generic vmagent-tls-secrets"
    sleep 1
    configure_for_etcd
    configure_for_controller_manager
    configure_for_kube_scheduler
}

function get_certificates_secrets() {
    # 获取 etcd 节点
    etcd_node=$(kubectl get pods -n kube-system -l component=etcd -o jsonpath='{.items[0].spec.nodeName}')
    echo "Etcd 节点: $etcd_node"
    # 创建 Job 提取证书
    cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: etcd-cert-extract
  namespace: kube-system
spec:
  template:
    spec:
      nodeName: $etcd_node
      containers:
      - name: extract
        image: busybox
        command:
        - sh
        - -c
        - |
          # 创建输出目录
          mkdir -p /output/etcd-certs
          
          # 复制 etcd 证书（从 hostPath 挂载的主机文件系统）
          cp /host/etc/kubernetes/pki/etcd/ca.crt /output/etcd-certs/
          cp /host/etc/kubernetes/pki/etcd/healthcheck-client.crt /output/etcd-certs/
          cp /host/etc/kubernetes/pki/etcd/healthcheck-client.key /output/etcd-certs/

          # 列出文件确认
          echo "已复制文件:"
          ls -la /output/etcd-certs/
          
          # 保持运行
          sleep 3600
        volumeMounts:
        - name: host
          mountPath: /host
        - name: output
          mountPath: /output
      volumes:
      - name: host
        hostPath:
          path: /
      - name: output
        emptyDir: {}
      restartPolicy: Never
EOF

# 等待 Job 完成
echo "等待 Job 启动..."
sleep 3
# 获取 Pod 名称
job_pod=$(kubectl get pods -n kube-system -l job-name=etcd-cert-extract -o jsonpath='{.items[0].metadata.name}')
echo "Job Pod: $job_pod"

# 等待 Pod Ready
kubectl wait --for=condition=ready pod/$job_pod -n kube-system --timeout=60s

# 复制证书到本地
echo "复制证书到本地..."
mkdir  -p /etc/kubernetes/pki/etcd
kubectl cp kube-system/$job_pod:/output/etcd-certs/ca.crt /etc/kubernetes/pki/etcd/ca.crt
kubectl cp kube-system/$job_pod:/output/etcd-certs/healthcheck-client.crt /etc/kubernetes/pki/etcd/healthcheck-client.crt
kubectl cp kube-system/$job_pod:/output/etcd-certs/healthcheck-client.key /etc/kubernetes/pki/etcd/healthcheck-client.key

echo "证书已提取到: /etc/kubernetes/pki/etcd/"
ls -la /etc/kubernetes/pki/etcd/

# 清理 Job
kubectl delete job etcd-cert-extract -n kube-system

}