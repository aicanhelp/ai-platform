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

# 删除所有的CR和CRDs
apiresources=$(kubectl api-resources --api-group=colocation.openfuyao.com -o name)

for apiresource in ${apiresources}; do
  kubectl delete ${apiresource} --all
  kubectl delete crd/${apiresource}
done