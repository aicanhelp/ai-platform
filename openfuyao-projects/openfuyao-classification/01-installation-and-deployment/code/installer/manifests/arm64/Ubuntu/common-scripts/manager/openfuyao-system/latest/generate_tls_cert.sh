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

CERTSEXPRATIONTIME=$1

source ./consts.sh
source ../../utils/log.sh "openfuyao-system"

ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

function install_cfssl() {
    info_log "Start installing cfssl and jq"
    sudo cp -f ./${ARCH}-bin/cfssl /usr/local/bin/
    sudo cp -f ./${ARCH}-bin/cfssljson /usr/local/bin/
    sudo cp -f ./${ARCH}-bin/cfssl-certinfo /usr/local/bin/
    sudo cp -f ./${ARCH}-bin/jq /usr/local/bin/

    sudo chmod +x /usr/local/bin/cfssl
    sudo chmod +x /usr/local/bin/cfssljson
    sudo chmod +x /usr/local/bin/cfssl-certinfo
    sudo chmod +x /usr/local/bin/jq
    info_log "Successfully installed cfssl and jq"
}

function generate_oauth_webhook_tls_cert() {
    info_log "generate oauth-webhook cert"
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
      sudo cfssl sign -ca ca.pem -ca-key ca-key.pem -config ./resource/oauth-webhook/server-signing-config.json - | \
      sudo cfssljson -bare ca-signed-server

    # 在 API 对象的状态中填充签名证书
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o json | \
      sudo jq '.status.certificate = "'$(base64 ca-signed-server.pem | tr -d '\n')'"' | \
      kubectl replace --raw /apis/certificates.k8s.io/v1/certificatesigningrequests/oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE}/status -f -

    # 下载颁发的证书并将其保存到 server.crt 文件中
    kubectl get csr oauth-webhook.${OPENFUYAO_SYSTEM_NAMESPACE} -o jsonpath='{.status.certificate}' \
      | base64 --decode > server.crt

    kubectl create secret generic oauth-webhook-tls --namespace=${OPENFUYAO_SYSTEM_NAMESPACE} \
        --from-file=ca.crt=ca.pem \
        --from-file=tls.crt=server.crt \
        --from-file=tls.key=server-key.pem
    info_log "Successfully generated oauth-webhook cert"
}

function copy_cert_and_config_file() {
    info_log "copy cert to webhook path"
    sudo mkdir -p "${FUYAO_WEBHOOK_PATH}"

    sudo cp -f ./resource/oauth-webhook/webhook-config.yaml ${FUYAO_WEBHOOK_PATH}/webhook-config.yaml

    sudo cp -f ca.pem  "${FUYAO_WEBHOOK_PATH}"
    sudo cp -f server.crt  "${FUYAO_WEBHOOK_PATH}"
    sudo cp -f server-key.pem  ${FUYAO_WEBHOOK_PATH}/server.key
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/ca.pem"
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.crt"
    sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.key"
    info_log "finished cert copy"
}

install_cfssl
sudo jq ".signing.default.expiry = \"$CERTSEXPRATIONTIME\"" ./resource/oauth-webhook/server-signing-config.json > oauthtmpfile.json && mv oauthtmpfile.json ./resource/oauth-webhook/server-signing-config.json -f
generate_oauth_webhook_tls_cert
copy_cert_and_config_file
