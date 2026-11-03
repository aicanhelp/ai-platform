#!/bin/bash

#
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# openFuyao is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

OUTPUT_BIN_DIR="_output/bin"

cd ${ROOT_DIR} || exit 1
echo "begin build binary of ub-network-device-plugin"
GOFLAGS="-buildvcs=false -buildmode=pie -trimpath -tags=osusergo,netgo -mod=readonly -modcacherw"
GOLDFLAGS="all=-w -s -buildid= -linkmode=external"
export CGO_CFLAGS="-Wall -fstack-protector-strong"
export CGO_LDFLAGS="-Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none"
CGO_ENABLED=1 GOOS=linux go build ${GOFLAGS} -ldflags="${GOLDFLAGS}" -o ${OUTPUT_BIN_DIR}/ub-network-device-plugin cmd/ub-network-device-plugin/main.go
build_result=$?
if [[ ${build_result} != 0 ]]; then
  echo "build ub-network-device-plugin failed, please check..."
  exit ${build_result}
fi
chmod 550 ${OUTPUT_BIN_DIR}/ub-network-device-plugin
echo "build ub-network-device-plugin succeed in directory ${ROOT_DIR}/${OUTPUT_BIN_DIR}"
