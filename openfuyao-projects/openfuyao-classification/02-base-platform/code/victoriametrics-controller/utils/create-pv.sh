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

function create_pv_for_vmagent() {
    vmagent_node_names=$1

    vmagent_mode=${vmAgentAllowStatefulSet}
    if [ "$vmagent_mode" != "true" ]; then
        info_log "vmagent is not allow statefulset, skip creating pv"
        return
    fi

    IFS=',' read -r -a node_names <<< "$vmagent_node_names"

    # 将字符串转换为整数
    VMAgentReplicaCount=$(echo \"${vmAgentReplicaCount}\" | tr -d '"' | awk '{print int($1)}')
    VMAgentShareCount=$(echo \"${vmAgentShareCount}\" | tr -d '"' | awk '{print int($1)}')
  
    # 参数验证
    if [ -z "$VMAgentReplicaCount" ] || [ "$VMAgentReplicaCount" -lt 1 ]; then
        echo "Error: VMAgentReplicaCount must be at least 1"
        exit 1
    fi

    if [ -z "$VMAgentShareCount" ] || [ "$VMAgentShareCount" -lt 1 ]; then
        echo "Error: VMAgentShareCount must be at least 1"
        exit 1
    fi

    TotalPvCount=$((VMAgentReplicaCount * VMAgentShareCount))

    if (( ${#node_names[@]} < TotalPvCount )); then
        echo "Error: No enough nodes for vmagent"
        exit 1
    fi
    
    # 创建pv
    echo "Starting vmagent PV creation:"
    echo "  - Shares: $vmagent_share_count"
    echo "  - Replicas per share: $vmagent_replicas"
    echo "  - Total PVs needed: $TotalPvCount"
    echo "  - Available nodes: ${#node_names[@]}"

    for ((share=0; share < VMAgentShareCount; share++)); do
        for ((replica=0; replica < VMAgentReplicaCount; replica++)); do
            # 计算节点索引
            node_index=$((share * VMAgentReplicaCount + replica))
            node_name="${node_names[$node_index]}"
            
            # 生成唯一的 PV 名称
            pv_name="pv-vmagent-${node_name}"
            if [ -z "$VMAgentReplicaCount" ] || [ "$VMAgentReplicaCount" -eq 1 ]; then
                pvc_name="persistent-queue-data-vmagent-vm-victoria-metrics-k8s-stack-${share}"
            else
                pvc_name="persistent-queue-data-vmagent-vm-victoria-metrics-k8s-stack-${share}-${replica}"
            fi
            echo "Creating PV $pv_name on node $node_name for PVC $pvc_name"
            
            # 应用 YAML 修改
            yq -e ".metadata.name = \"${pv_name}\"" -i ./resources/vmagent-pv.yaml
            yq -e ".spec.capacity.storage = \"${vmAgentStorageSize}\"" -i ./resources/vmagent-pv.yaml
            yq -e ".spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].values[0] = \"${node_name}\"" -i ./resources/vmagent-pv.yaml
            yq -e ".spec.claimRef.namespace = \"${VM_NAMESPACE}\"" -i ./resources/vmagent-pv.yaml
            yq -e ".spec.claimRef.name = \"${pvc_name}\"" -i ./resources/vmagent-pv.yaml

            kubectl apply -f ./resources/vmagent-pv.yaml
        done
    done
    info_log "Successfully create pv for vmagent"
}



function create_pv_for_vmslelct() {
    vmselect_node_names=$1

    IFS=',' read -r -a node_names <<< "$vmselect_node_names"
    replicas=0
    for node_name in "${node_names[@]}"; do
        yq -e ".metadata.name = \"pv-vmselect-${node_name}\"" -i ./resources/vmselect-pv.yaml
        yq -e ".spec.capacity.storage = \"${vmSelectStorageSize}\"" -i ./resources/vmselect-pv.yaml
        yq -e ".spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].values[0] = \"${node_name}\"" -i ./resources/vmselect-pv.yaml
        yq -e ".spec.claimRef.namespace = \"${VM_NAMESPACE}\"" -i ./resources/vmselect-pv.yaml
        yq -e ".spec.claimRef.name = \"vmselect-cachedir-vmselect-vm-victoria-metrics-k8s-stack-${replicas}\"" -i ./resources/vmselect-pv.yaml

        kubectl apply -f ./resources/vmselect-pv.yaml

        replicas=$((replicas+1))
    done

    info_log "Successfully create pv for vmselect"
}


function create_pv_for_vmstorage() {
     vmstorage_node_names=$1

     IFS=',' read -r -a node_names <<< "$vmstorage_node_names"
     replicas=0
     for node_name in "${node_names[@]}"; do
         yq -e ".metadata.name = \"pv-vmstorage-${node_name}\"" -i ./resources/vmstorage-pv.yaml
         yq -e ".spec.capacity.storage = \"${vmStorageStorageSize}\"" -i ./resources/vmstorage-pv.yaml
         yq -e ".spec.nodeAffinity.required.nodeSelectorTerms[0].matchExpressions[0].values[0] = \"${node_name}\"" -i ./resources/vmstorage-pv.yaml
         yq -e ".spec.claimRef.namespace = \"${VM_NAMESPACE}\"" -i ./resources/vmstorage-pv.yaml
         yq -e ".spec.claimRef.name = \"vmstorage-db-vmstorage-vm-victoria-metrics-k8s-stack-${replicas}\"" -i ./resources/vmstorage-pv.yaml
         kubectl apply -f ./resources/vmstorage-pv.yaml

         replicas=$((replicas+1))
     done

     info_log "Successfully create pv for vmstorage"
}


function create_pv_for_vmsingle() {
    vmsingle_node_names=${DeployVMSingleNodeNames}
    
    IFS=',' read -r -a node_names <<< "$vmsingle_node_names"
    for node_name in "${node_names[@]}"; do
       yq -e ".metadata.name = \"pv-vmsingle-${node_name}\"" -i ./resources/vmsingle-pv.yaml
       yq -e ".spec.capacity.storage = \"${vmSingleStorageSize}\"" -i ./resources/vmsingle-pv.yaml
       yq -e ".spec.claimRef.namespace = \"${VM_NAMESPACE}\"" -i ./resources/vmstorage-pv.yaml

        kubectl apply -f ./resources/vmsingle-pv.yaml
    done

    kubectl apply -f ./resources/vmsingle-pv.yaml
    info_log "Successfully create pv for vmsingle"
}

function create_path_for_pv() {
    all_nodes=$(kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{","}{end}' | sed 's/,$//')
    IFS=',' read -r -a node_array <<< "$all_nodes"

    for node in "${node_array[@]}"; do
        echo "=== 在 $node 上创建目录 ==="
        
        # 判断该节点需要哪些目录
        need_vmagent=false
        need_vmselect=false
        need_vmstorage=false
        
        [[ ",${vmagent_node_names}," == *",$node,"* && "${vmAgentAllowStatefulSet}" == "true" ]] && need_vmagent=true
        [[ ",${vmselect_node_names}," == *",$node,"* ]] && need_vmselect=true
        [[ ",${vmstorage_node_names}," == *",$node,"* ]] && need_vmstorage=true
        
        # 如果这个节点不需要任何目录，跳过
        if [[ "$need_vmagent" != "true" && "$need_vmselect" != "true" && "$need_vmstorage" != "true" ]]; then
            echo "节点 $node 不需要创建任何目录，跳过"
            echo ""
            continue
        fi
        
        # 构建创建目录的命令 - 使用数组避免空命令问题
        mkdir_cmd="mkdir -p"
        [[ "$need_vmagent" == "true" ]] && mkdir_cmd="$mkdir_cmd /host/mnt/data/vmagent"
        [[ "$need_vmselect" == "true" ]] && mkdir_cmd="$mkdir_cmd /host/mnt/data/vmselect"
        [[ "$need_vmstorage" == "true" ]] && mkdir_cmd="$mkdir_cmd /host/mnt/data/vmstorage"
        
        echo "需要在 $node 上创建的目录:"
        [[ "$need_vmagent" == "true" ]] && echo "  - /mnt/data/vmagent"
        [[ "$need_vmselect" == "true" ]] && echo "  - /mnt/data/vmselect"
        [[ "$need_vmstorage" == "true" ]] && echo "  - /mnt/data/vmstorage"
        
        # 创建 Job
        cat <<EOF | kubectl apply -f -
apiVersion: batch/v1
kind: Job
metadata:
  name: create-dirs-$node
  namespace: kube-system
spec:
  ttlSecondsAfterFinished: 60
  template:
    spec:
      nodeName: $node
      containers:
      - name: create
        image: busybox
        command:
        - sh
        - -c
        - |
          echo "在节点 $node 上创建目录..."
          $mkdir_cmd
          echo "创建完成，验证目录:"
          for dir in /host/mnt/data/vmagent /host/mnt/data/vmselect /host/mnt/data/vmstorage; do
            if [ -d "\$dir" ]; then
              echo "✓ \$dir 已存在"
              ls -la "\$dir"
            fi
          done
        volumeMounts:
        - name: host
          mountPath: /host
      volumes:
      - name: host
        hostPath:
          path: /
      restartPolicy: Never
EOF

        # 等待完成
        if kubectl wait --for=condition=complete job/create-dirs-$node -n kube-system --timeout=60s 2>/dev/null; then
            # 查看日志
            job_pod=$(kubectl get pods -n kube-system -l job-name=create-dirs-$node -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
            if [ -n "$job_pod" ]; then
                echo "Job 执行日志:"
                kubectl logs $job_pod -n kube-system 2>/dev/null
            fi
            echo "✓ 节点 $node 目录创建成功"
        else
            echo "✗ 节点 $node 目录创建超时或失败"
        fi
        
        echo "------------------------"
        echo ""
    done

    echo "所有需要创建目录的节点处理完成！"
}


function create_pv() {
    # 单节点判断
    if ${useVMSingle} == "true"; then
        create_pv_for_vmsingle
        return
    fi

    vmagent_node_names=$(kubectl get nodes -l monitoring.victoria.com/vmagent=vmagent -o jsonpath='{.items[*].metadata.name}' | tr ' ' ',')
    vmselset_node_names=$(kubectl get nodes -l monitoring.victoria.com/vmselect=vmselect -o jsonpath='{.items[*].metadata.name}' | tr ' ' ',')
    vmstorage_node_names=$(kubectl get nodes -l monitoring.victoria.com/vmstorage=vmstorage -o jsonpath='{.items[*].metadata.name}' | tr ' ' ',')

    create_pv_for_vmagent "$vmagent_node_names"
    create_pv_for_vmslelct "$vmselset_node_names"
    create_pv_for_vmstorage "$vmstorage_node_names"
    create_path_for_pv
}