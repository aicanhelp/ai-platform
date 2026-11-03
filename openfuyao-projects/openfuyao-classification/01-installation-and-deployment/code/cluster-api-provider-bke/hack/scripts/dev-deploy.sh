#!/bin/bash
# ******************************************************************
# Copyright (c) 2025 Bocloud Technologies Co., Ltd.
# cluster-api-provider-bke is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# ******************************************************************

echo "stop bke"
#kubectl scale deployment -n cluster-system bke-controller-manager --replicas=0
kubectl delete deployment -n cluster-system bke-controller-manager
kubectl delete secret -n cluster-system bke-webhook-secret
kubectl delete ValidatingWebhookConfiguration bke-validating-webhook-configuration
kubectl delete MutatingWebhookConfiguration bke-mutating-webhook-configuration

# create local webhook cert
sudo ./bin/certgen create \
  --kubeconfig /etc/rancher/k3s/k3s.yaml \
  --secret-name bke-webhook-secret \
  --cert-name tls.crt \
  --key-name tls.key \
  --host 172.28.8.66 \
  --namespace cluster-system





echo "step 1 apply yaml"
kubectl apply -f hack/deploy/dev.yaml
echo "step 2 wait bke start"
for i in {1..60}; do
  if kubectl get deployment -n cluster-system | grep bke | grep 1/1; then
    echo "bke deployment is ready"
    break
  fi
  echo "$(date --rfc-3339=seconds) bke deployment is not ready"
  sleep 5
done
echo "step 3 copy webhook cert"
kubectl get secret -n cluster-system bke-webhook-secret -o=jsonpath='{.data.tls\.crt}' | base64 -d >hack/dev-hook-certs/tls.crt
kubectl get secret -n cluster-system bke-webhook-secret -o=jsonpath='{.data.tls\.key}' | base64 -d >hack/dev-hook-certs/tls.key

caBundle=$(kubectl get secret -n cluster-system bke-webhook-secret -o jsonpath='{.data.ca\.crt}' | tr -d '\n')

kubectl patch MutatingWebhookConfiguration bke-mutating-webhook-configuration --type='json' -p='[{"op": "replace", "path": "/webhooks/0/clientConfig/caBundle", "value": "'"$caBundle"'"}]'
kubectl patch ValidatingWebhookConfiguration bke-validating-webhook-configuration --type='json' -p='[{"op": "replace", "path": "/webhooks/0/clientConfig/caBundle", "value": "'"$caBundle"'"}]'
kubectl patch ValidatingWebhookConfiguration bke-validating-webhook-configuration --type='json' -p='[{"op": "replace", "path": "/webhooks/1/clientConfig/caBundle", "value": "'"$caBundle"'"}]'


echo "stop bke"
kubectl scale deployment -n cluster-system bke-controller-manager --replicas=0
