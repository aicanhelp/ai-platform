#!/bin/bash
################################################################################################
 # Copyright (c) 2024 Huawei Technologies Co., Ltd.
 # openFuyao is licensed under Mulan PSL v2.
 # You can use this software according to the terms and conditions of the Mulan PSL v2.
 # You may obtain a copy of Mulan PSL v2 at:
 #          http://license.coscl.org.cn/MulanPSL2
 # THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 # EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 # MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 # See the Mulan PSL v2 for more details.
################################################################################################

set -o errexit
set -o nounset
set -o pipefail

# 删除deployment
if [ -n $(kubectl get pod -n openfuyao-colocation -oname | grep colocation-controller | awk 'BEGIN{FS="/"}{print $2}') ]; then
  # kubectl get deployment -n openfuyao-colocation
  kubectl delete deployment colocation-controller -n openfuyao-colocation
fi

# 安装admission webhook
kubectl apply -k ./config/controller/manifest/overlays/dev

sleep 30
kubectl logs -f $(kubectl get pod -n openfuyao-colocation -oname | grep colocation-controller | awk 'BEGIN{FS="/"}{print $2}') -n openfuyao-colocation 