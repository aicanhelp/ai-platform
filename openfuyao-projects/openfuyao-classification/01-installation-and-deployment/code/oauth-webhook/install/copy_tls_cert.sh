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

cat <<EOF > webhook-config.yaml
apiVersion: v1
kind: Config
clusters:
  - name: oauth-webhook
    cluster:
      certificate-authority: /etc/webhook/ca.pem    # 用来验证远程服务的CA
      server: https://oauth-webhook.openfuyao-system.svc.cluster.local:9095/oauth/tokenauth/fuyao         # 远程服务URL
# users指代APIServer的Webhook配置
users:
  - name: kubernetes
    user:
      client-certificate: /etc/webhook/server.crt     # Webhook插件使用的证书
      client-key: /etc/webhook/server.key         # 与证书匹配的密钥
current-context: webhook
contexts:
  - context:
      cluster: oauth-webhook
      user: kubernetes
    name: webhook
EOF

FUYAO_WEBHOOK_PATH="/etc/kubernetes/webhook"
mkdir -p ${FUYAO_WEBHOOK_PATH}
sudo cp -f webhook-config.yaml ${FUYAO_WEBHOOK_PATH}/webhook-config.yaml

sudo cp -f /opt/openFuyao/webhooks/pki/ca.pem  "${FUYAO_WEBHOOK_PATH}"
sudo cp -f /opt/openFuyao/webhooks/pki/server.crt  "${FUYAO_WEBHOOK_PATH}"
sudo cp -f /opt/openFuyao/webhooks/pki/server-key.pem  ${FUYAO_WEBHOOK_PATH}/server.key
sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/ca.pem"
sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.crt"
sudo chmod 400 "${FUYAO_WEBHOOK_PATH}/server.key"