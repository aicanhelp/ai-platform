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

output=$(nsenter -i/proc/1/ns/ipc -m/proc/1/ns/mnt -n/proc/1/ns/net kubectl get npuclusterpolicies.npu.openfuyao.com -o jsonpath='{.items[0].spec.mindioacp.version}')

# 检查输出是否为空
if [ -z "$output" ]; then
  echo "Error: Unable to retrieve the tag value."
  exit 1
fi

echo "$output"
