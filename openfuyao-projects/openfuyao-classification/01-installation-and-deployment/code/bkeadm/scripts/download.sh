#!/bin/bash
###############################################################
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# installer is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
###############################################################

if [ "x$(uname)" != "xLinux" ]; then
  echo ""
  echo 'Warning: Non-Linux operating systems are not supported!'
  exit 1
fi

if [ -z "${ARCH}" ]; then
  case "$(uname -m)" in
  x86_64)
    ARCH=amd64
    ;;
  aarch64*)
    ARCH=arm64
    ;;
  *)
    echo "${ARCH}, isn't supported"
    exit 1
    ;;
  esac
fi

if [ -z "${VERSION}" ]; then
  VERSION="latest"
fi

if [[ ${VERSION} =~ ^[0-9a-zA-Z\.-]+$ ]]; then
  echo ""
else
  echo "Version information incorrect. Set VERSION env var and re-run. For example: export VERSION=1.0.0 or export VERSION=v25.12"
  exit 1
fi

if [ -f "bkeadm_linux_${ARCH}" ]; then
  echo ""
  echo "bkeadm ${VERSION} already exists in current directory!"
  mv bkeadm_linux_${ARCH} bkeadm_linux_${ARCH}_bk
  echo "the original bkeadm_linux_${ARCH} has been renamed to bkeadm_linux_${ARCH}_bk"
  echo ""
fi

DOWNLOAD_URL="https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/${VERSION}/bkeadm_linux_${ARCH}"
SHA256SUM_URL="https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/${VERSION}/bkeadm_linux_${ARCH}.sha256"

echo "Downloading bkeadm ${VERSION} from ${DOWNLOAD_URL} ..."
echo ""

curl -fsLO "$DOWNLOAD_URL"
if [ $? -ne 0 ]; then
  echo ""
  echo "Failed to download bkeadm ${VERSION} !"
  echo ""
  echo "Please verify the version you are trying to download."
  echo ""
  exit 1
fi

echo ""
echo "bkeadm ${VERSION} Download Complete!"
echo ""

echo "Downloading bkeadm sha256sum ${VERSION} from ${SHA256SUM_URL} ..."
echo ""

curl -fsLO "$SHA256SUM_URL"
if [ $? -ne 0 ]; then
  echo ""
  echo "Failed to download bkeadm sha256sum ${VERSION} !"
  echo ""
  echo "Please verify the version you are trying to download."
  echo ""
  exit 1
fi

echo ""
echo "bkeadm sha256sum ${VERSION} Download Complete!"
echo ""

sha256sum -c <(cat bkeadm_linux_${ARCH}.sha256) < bkeadm_linux_${ARCH}
if [ $? -ne 0 ]; then
  echo ""
  echo "The ${VERSION} SHA256 checksum verification failed."
  echo ""
  echo "Please contact openFuyao maintainer."
  echo ""
  exit 1
fi

echo ""
echo "The ${VERSION} SHA256 checksum verification success."
echo ""

if [ -f "/usr/local/bin/bke" ]; then
  echo ""
  echo "bke command already exists in /usr/local/bin!"
  mv /usr/local/bin/bke /usr/local/bin/bke_bk -f
  echo "the original bke has been renamed to bke_bk"
  echo ""
fi

chmod +x bkeadm_linux_${ARCH}
mv bkeadm_linux_${ARCH} /usr/local/bin/bke -f
hash -r >> /dev/null

rm -f bkeadm_linux_${ARCH}.sha256

echo ""
echo "bke has been installed to /usr/local/bin"
echo ""

