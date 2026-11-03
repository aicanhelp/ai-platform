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

REGISTRY="${FUYAO_RGISTRY}"
REGISTRY="${REGISTRY%/}"
BUSY_BOX_REPOSITORY="${REGISTRY}/busybox/busybox"
ENABLE_HTTPS="true"
OPENFUYAO_REPO="${FUYAO_REPO}"
OAUTH_CERTS_EXPIRATION_TIME="1752000h"

OS=$(echo `uname`|tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

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

    echo "111111111111111111111111111111111"
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
    echo "22222222222222222222222222222"
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
    echo "33333333333333333333333333333333333"
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
    echo "44444444444444444444444444444444444444"
    # 在 API 对象的状态中填充签名证书
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o json | \
      sudo jq '.status.certificate = "'$(base64 ca-signed-server.pem | tr -d '\n')'"' | \
      kubectl replace --raw /apis/certificates.k8s.io/v1/certificatesigningrequests/oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}/status -f -
    echo "55555555555555555555555555555555"
    # 下载颁发的证书并将其保存到 server.crt 文件中
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath='{.status.certificate}' \
      | base64 --decode > server.crt
    echo "66666666666666666666666666"
    kubectl create secret generic "${OAUTH_WEBHOOK_TLS}" --namespace="${OPENFUYAO_SYSTEM_NAMESPACE}" \
        --from-file=ca.crt=ca.pem \
        --from-file=tls.crt=server.crt \
        --from-file=tls.key=server-key.pem


    cd "${cur_path}" || fatal_log "Failed to change directory to ${cur_path}"
    info_log "Successfully generated oauth-webhook cert"
}

function save_webhook_config_yaml_to_cm() {
    local yaml_file_path="./resource/oauth-webhook/webhook-config.yaml"
    kubectl create configmap "${OAUTH_WEBHOOK_CONFIG_YAML_CM}" --from-file=$yaml_file_path -n "${OPENFUYAO_SYSTEM_NAMESPACE}"
}

# 生成证书
generate_oauth_webhook_tls_cert

# 拷贝证书与配置文件到宿主机的 /var/lib/rancher/k3s/webhook
save_webhook_config_yaml_to_cm
mkdir -p "${K3S_WEBHOOK_PATH}"
kubectl get configmap ${OAUTH_WEBHOOK_CONFIG_YAML_CM} -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath='{.data.webhook-config\.yaml}' > ${K3S_WEBHOOK_PATH}/webhook-config.yaml
kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o yaml | yq eval '.data."ca.crt"' | base64 -d > "${K3S_WEBHOOK_PATH}/ca.pem"
kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o yaml | yq eval '.data."tls.crt"' | base64 -d > "${K3S_WEBHOOK_PATH}/server.crt"
kubectl get secret "${OAUTH_WEBHOOK_TLS}" -n ${OPENFUYAO_SYSTEM_NAMESPACE} -o yaml | yq eval '.data."tls.key"' | base64 -d > "${K3S_WEBHOOK_PATH}/server.key"
chmod 400 "${K3S_WEBHOOK_PATH}/ca.pem"
chmod 400 "${K3S_WEBHOOK_PATH}/server.crt"
chmod 400 "${K3S_WEBHOOK_PATH}/server.key"





