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
  echo "Version information incorrect. Set VERSION env var and re-run. For example: export VERSION=1.0.0"
  exit 1
fi

if [ "${VERSION}" != "latest" ]; then
  VERSION="v${VERSION}"
fi

if [ -z "${OS_NAME}" ]; then
  if [ "$(cat /etc/os-release | grep openEuler)" ]; then
    OS_NAME="openEuler"
  elif [ "$(cat /etc/os-release | grep Ubuntu)" ]; then
    OS_NAME="Ubuntu"
  else
    echo ""
    echo "This operating system is not supported."
    exit 1
  fi
fi

if [ -z "${OS_VERSION}" ];  then
  OS_VERSION=$(cat /etc/os-release | grep VERSION_ID | cut -d '=' -f 2 | sed 's/"//g')
fi

DOWNLOAD_URL="https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/${VERSION}/fuyao-${VERSION}-${OS_NAME}-${OS_VERSION}-${ARCH}.tar.gz"

echo "Downloading installer ${VERSION} from ${DOWNLOAD_URL} ..."
echo ""

curl -fsLO "$DOWNLOAD_URL"
if [ $? -ne 0 ]; then
  echo ""
  echo "Failed to download installer ${OS_NAME} ${VERSION} !"
  echo ""
  echo "Please verify the version you are trying to download."
  echo ""
  exit 1
fi

PACKAGE_NAME="fuyao-${VERSION}-${OS_NAME}-${OS_VERSION}-${ARCH}.tar.gz"
if command -v tar >/dev/null 2>&1; then
  tar -xzf "${PACKAGE_NAME}"
else
  if [ "`cat /etc/os-release | grep openEuler`" ]; then
    sudo yum install -y tar >/dev/null
  elif [ -f /etc/debian_version ]; then
    sudo apt-get update >/dev/null
    sudo apt-get install -y tar >/dev/null
  elif [ -f /etc/redhat-release ]; then
    sudo yum install -y tar >/dev/null
  fi
  tar -xzf "${PACKAGE_NAME}"
fi

echo ""
echo "installer ${VERSION} Download Complete!"
echo ""
