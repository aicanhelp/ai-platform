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

ARCH=`uname -m`
if [[ "${ARCH}" == "arm64" ]]; then
  ARCH="aarch64"
fi
DEFAULT_TAG="latest"
DEFAULT_IMAGE_REPOSITORY="cr.openfuyao.cn/openfuyao/ub-network-bandwidth-operator"
OUTPUT_BASE_DIR="${ROOT_DIR}/_output"
IMAGE_DIR="${OUTPUT_BASE_DIR}/image"

TAG=${TAG:-$DEFAULT_TAG}
IMAGE_REPOSITORY=${IMAGE_REPOSITORY:-$DEFAULT_IMAGE_REPOSITORY}
FULL_IMAGE_NAME="${IMAGE_REPOSITORY}:${TAG}"
DOCKER_BUILD_ARGS=()
if [[ "${NO_CACHE:-1}" == "1" ]]; then
  DOCKER_BUILD_ARGS+=(--no-cache --pull)
fi

cd ${ROOT_DIR}/ || exit 1
mkdir -p "${IMAGE_DIR}"

# build image
echo "begin build image of ub-network-bandwidth"
docker build "${DOCKER_BUILD_ARGS[@]}" -f cmd/ub-network-bandwidth-operator/Dockerfile -t "${FULL_IMAGE_NAME}" .
build_result=$?
if [[ ${build_result} != 0 ]]; then
  echo "build image of ub-network-bandwidth failed, please check..."
  exit ${build_result}
fi

# save image
EXPORT_FILE="${IMAGE_DIR}/ub-network-bandwidth-${TAG}-${ARCH}.tar"
docker save -o "${EXPORT_FILE}" "${FULL_IMAGE_NAME}"
save_result=$?
if [[ ${save_result} != 0 ]]; then
  echo "save image of ${FULL_IMAGE_NAME} failed, please check..."
  exit ${save_result}
fi

echo "build image ub-network-bandwidth-${TAG}-${ARCH}.tar succeed in directory ${IMAGE_DIR}"
