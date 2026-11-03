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

mkdir -p /opt/openFuyao/webhooks/pki && cd /opt/openFuyao/webhooks/pki

# 生成 oauth-webhook 的私钥
cat <<EOF | sudo cfssl genkey - | sudo cfssljson -bare server
{
  "hosts": [
    "oauth-webhook.openfuyao-system.svc.cluster.local",
    "oauth-webhook.openfuyao-system.pod.cluster.local"
  ],
  "CN": "oauth-webhook.openfuyao-system.pod.cluster.local",
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
  name: oauth-webhook.openfuyao-system # my-svc.my-namespace
spec:
  signerName: openfuyao.io/oauth-signer # kubernetes.io/kube-apiserver-client
  request: $(cat server.csr | base64 | tr -d '\n')
  usages:
  - key encipherment
  - digital signature
  - server auth
  - client auth
EOF

kubectl certificate approve oauth-webhook.openfuyao-system # my-svc.my-namespace

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

cat <<EOF > server-signing-config.json
{
  "signing": {
    "default": {
      "usages": [
        "digital signature",
        "key encipherment",
        "server auth",
        "client auth"
      ],
      "expiry": "36000h",
      "ca_constraint": {
        "is_ca": false
      }
    }
  }
}
EOF

kubectl get csr oauth-webhook.openfuyao-system -o jsonpath='{.spec.request}' | \
  base64 --decode | \
  sudo cfssl sign -ca ca.pem -ca-key ca-key.pem -config server-signing-config.json - | \
  sudo cfssljson -bare ca-signed-server

# 在 API 对象的状态中填充签名证书
kubectl get csr oauth-webhook.openfuyao-system -o json | \
  sudo jq '.status.certificate = "'$(base64 ca-signed-server.pem | tr -d '\n')'"' | \
  kubectl replace --raw /apis/certificates.k8s.io/v1/certificatesigningrequests/oauth-webhook.openfuyao-system/status -f -

# 下载颁发的证书并将其保存到 server.crt 文件中
kubectl get csr oauth-webhook.openfuyao-system -o jsonpath='{.status.certificate}' \
  | base64 --decode > server.crt

kubectl create secret generic oauth-webhook-tls -n openfuyao-system --from-file ca.crt=ca.pem --from-file tls.crt=server.crt --from-file tls.key=server-key.pem
