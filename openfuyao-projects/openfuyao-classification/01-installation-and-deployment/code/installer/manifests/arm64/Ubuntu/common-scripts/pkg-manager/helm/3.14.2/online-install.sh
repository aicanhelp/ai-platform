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

source ../../utils/log.sh "helm"
source ../../utils/consts.sh
source ./consts.sh

OS=$(echo `uname`|tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

DOWNLOAD_URL="${FUYAO_OBS}/helm/releases/download/v${VERSION}/helm-v${VERSION}-${OS}-${ARCH}.tar.gz"
sudo -E curl -sOL "${DOWNLOAD_URL}"

sudo tar -zxvf helm-v${VERSION}-${OS}-${ARCH}.tar.gz
sudo mv ${OS}-$ARCH/helm /usr/local/bin/helm
