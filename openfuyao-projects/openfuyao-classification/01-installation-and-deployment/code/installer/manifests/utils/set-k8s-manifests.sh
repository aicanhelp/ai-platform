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

# 对apiserver做动态修改
APISERVER_YAML_PATH="/etc/kubernetes/manifests/kube-apiserver.yaml"
echo "edit ${APISERVER_YAML_PATH}"
sudo yq -i '.spec.containers[0].command += ["--authentication-token-webhook-config-file=/etc/webhook/webhook-config.yaml"]' ${APISERVER_YAML_PATH}
sudo yq -i '.spec.containers[0].command += ["--authentication-token-webhook-cache-ttl=5m"]' ${APISERVER_YAML_PATH}

sudo yq -i '.spec.containers[0].volumeMounts += [{"mountPath": "/etc/webhook", "name": "webhook-config", "readOnly": true}]' ${APISERVER_YAML_PATH}

sudo yq -i '.spec += {"dnsPolicy": "ClusterFirstWithHostNet"}' ${APISERVER_YAML_PATH}

sudo yq -i '.spec.volumes += [{"hostPath": {"path": "/etc/kubernetes/webhook", "type": "DirectoryOrCreate"}, "name": "webhook-config"}]' ${APISERVER_YAML_PATH}

CONTROLLER_MANAGER_YAML_PATH="/etc/kubernetes/manifests/kube-controller-manager.yaml"
echo "edit CONTROLLER_MANAGER_YAML_PATH"
sudo yq -i '.spec.containers[0].command[] |= select(. == "--bind-address=127.0.0.1") |= "--bind-address=0.0.0.0"' ${CONTROLLER_MANAGER_YAML_PATH}

SCHEDULER_YAML_PATH="/etc/kubernetes/manifests/kube-scheduler.yaml"
sudo yq -i '.spec.containers[0].command[] |= select(. == "--bind-address=127.0.0.1") |= "--bind-address=0.0.0.0"' ${SCHEDULER_YAML_PATH}

echo "finished edit yaml"

# 修改apiserver yaml后要等几分钟，apiserver才能启动，默认最多等待5分钟
MAX_RETRIES=60
RETRY_DELAY=5
ATTEMPTS=0
while [ $ATTEMPTS -lt $MAX_RETRIES ]; do
    kubectl get node
    if [ $? -eq 0 ]; then
        echo "api-server has been restarted."
        break
    else
        echo "api-server status is abnormal. Retrying in $RETRY_DELAY seconds..."
        sleep $RETRY_DELAY
        ((attempts++))
    fi
done

if [ $ATTEMPTS -eq $MAX_RETRIES ]; then
    echo "Maximum retries reached. Command execution failed after $MAX_RETRIES attempts."
    exit 1
fi
