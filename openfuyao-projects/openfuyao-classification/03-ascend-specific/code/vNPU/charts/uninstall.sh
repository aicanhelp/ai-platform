#!/bin/bash

#
# /*
#  * Copyright (c) 2025 Huawei Technologies Co., Ltd.
#  * openFuyao is licensed under Mulan PSL v2.
#  * You can use this software according to the terms and conditions of the Mulan PSL v2.
#  * You may obtain a copy of Mulan PSL v2 at:
#  *          http://license.coscl.org.cn/MulanPSL2
#  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
#  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
#  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#  * See the Mulan PSL v2 for more details.
#  */
#

set -e

work_path="/opt/xpu"
var_log_xpu_path="/var/log/xpu"
var_lib_xpu_path="/var/lib/xpu"
etc_xpu_path="/etc/xpu"
work_lib_path=${work_path}/lib
lib_path="/usr/lib64"
namespace="xpu"
cuda_path=${lib_path}/libcuda.so
cuda_original_path=${work_lib_path}/libcuda-original.so
deployment_name="npupool"
RED='\033[0;31m'
NC='\033[0m'

pods=$(kubectl get pod -n "${namespace}" | tail -n +2 | awk '{print $1}')

if helm list --deployed --short | grep -q ${deployment_name}; then
    helm uninstall npupool
else
    echo "The helm of npupool has been uninstalled. Next, we will clean up the remaining residues"
fi

pod_delete_flag="true"
echo "Wait for Pods to be deleted."
for pod in ${pods}; do
    kubectl wait --for=delete pod/${pod} -n "${namespace}" --timeout=120s 2>/dev/null || status=$? || true
    if [[ ${status} -ne 0 ]]; then
       echo "'${pod}' deletion timeout"
       pod_delete_flag="false"
    fi
done

echo "The following XPU related folders will be retained."
echo ${work_path}
echo ${var_log_xpu_path}
echo ${var_lib_xpu_path}
echo ${etc_xpu_path}

if [ ${pod_delete_flag} = "false" ]; then
    echo -e "${RED}Some pods have not been deleted.${NC}"
    echo -e "${RED}You are advised to manually delete the pod and run the script again to clear residual data.${NC}"
    exit 1
fi

echo "Uninstall npupool completed!"
