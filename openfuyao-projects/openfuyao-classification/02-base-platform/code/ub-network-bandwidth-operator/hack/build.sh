#!/bin/bash

#
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
mkdir -p "${OUTPUT_BIN_DIR}"
echo "begin build binary of ub-network-bandwidth-operator"
GOFLAGS="-buildvcs=false -buildmode=pie -trimpath -tags=osusergo,netgo -mod=readonly -modcacherw"
GOLDFLAGS="-w -s -buildid="
TARGET_GOOS="${GOOS:-linux}"
TARGET_GOARCH="${GOARCH:-$(go env GOARCH)}"
TARGET_CGO_ENABLED="${CGO_ENABLED:-0}"

if [[ "${TARGET_CGO_ENABLED}" == "1" ]]; then
  GOLDFLAGS="all=-w -s -buildid= -linkmode=external"
  export CGO_CFLAGS="-Wall -fstack-protector-strong"
  export CGO_LDFLAGS="-Wl,-z,now -Wl,-z,noexecstack -Wl,--build-id=none"
fi

CGO_ENABLED="${TARGET_CGO_ENABLED}" GOOS="${TARGET_GOOS}" GOARCH="${TARGET_GOARCH}" \
  go build ${GOFLAGS} -ldflags="${GOLDFLAGS}" -o ${OUTPUT_BIN_DIR}/ub-network-bandwidth cmd/ub-network-bandwidth/main.go
build_result=$?
if [[ ${build_result} != 0 ]]; then
  echo "build ub-network-bandwidth-operator failed, please check..."
  exit ${build_result}
fi
chmod 550 ${OUTPUT_BIN_DIR}/ub-network-bandwidth
echo "build ub-network-bandwidth-operator succeed in directory ${ROOT_DIR}/${OUTPUT_BIN_DIR}"
