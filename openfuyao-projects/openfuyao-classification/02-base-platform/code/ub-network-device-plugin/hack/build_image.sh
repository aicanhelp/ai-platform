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

ARCH=`uname -m`
DEFAULT_TAG="v1.0.1"
OUTPUT_BASE_DIR="${ROOT_DIR}/_output"
BINARY_DIR="${OUTPUT_BASE_DIR}/bin"
IMAGE_DIR="${OUTPUT_BASE_DIR}/image"

TAG=${TAG:-$DEFAULT_TAG}
FULL_IMAGE_NAME="ub-network-device-plugin-${ARCH}:${TAG}"

cd ${ROOT_DIR}/ || exit 1
mkdir -p "${IMAGE_DIR}"

# check binary
if [ ! -f "${BINARY_DIR}/ub-network-device-plugin" ]; then
    echo "binary file not exist in ${BINARY_DIR}, please build binary first"
    exit 1
fi

# build image
echo "begin build image of ub-network-device-plugin"
cp hack/Dockerfile ${BINARY_DIR}/
cd ${BINARY_DIR} || exit 1
docker build -t "${FULL_IMAGE_NAME}" .
build_result=$?
if [[ ${build_result} != 0 ]]; then
  echo "build image of ub-network-device-plugin failed, please check..."
  exit ${build_result}
fi

# save image
EXPORT_FILE="${IMAGE_DIR}/ub-network-device-plugin-${TAG}-${ARCH}.tar"
docker save -o "${EXPORT_FILE}" "${FULL_IMAGE_NAME}"
save_result=$?
if [[ ${save_result} != 0 ]]; then
  echo "save image of ${FULL_IMAGE_NAME} failed, please check..."
  exit ${save_result}
fi

echo "build image ub-network-device-plugin-${TAG}-${ARCH}.tar succeed in directory ${IMAGE_DIR}"