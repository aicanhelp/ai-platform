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

source ../../utils/log.sh "harbor"
source ./preinstall.sh
source ../../utils/consts.sh

ARCH=$(uname -m)
case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
esac

# 加载镜像
IMAGES=(
    ${FUYAO_RGISTRY}/harbor/harbor-exporter:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/redis-photon:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/trivy-adapter-photon:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-registryctl:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/registry-photon:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/nginx-photon:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-log:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-jobservice:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-core:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-portal:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/harbor-db:v${HARBOR_VERSION}
    ${FUYAO_RGISTRY}/harbor/prepare:v${HARBOR_VERSION}
)

for image_name in ${IMAGES[@]} ; do
    NAME_VERSION=$(echo $image_name | awk -F '/' '{print $4}')
    NAME=$(echo ${NAME_VERSION} | awk -F ':' '{print $1}')
    sudo nerdctl load -i ./image/${NAME}.tar --platform=linux/arm64,linux/amd64

    NEW_TAG="goharbor/${NAME_VERSION}"
    sudo nerdctl tag ${image_name} ${NEW_TAG}
done

info_log "Finished load images"

# 到制定路径生成证书
ABS_PATH=$(pwd)
sudo mkdir -p ${HARBOR_CERT_PATH} && cd ${HARBOR_CERT_PATH}
sudo openssl genrsa -out ${HARBOR_CA_KEY} 4096
sudo openssl req -newkey rsa:4096 -nodes -keyout ${HARBOR_CA_KEY} -x509 -days {{ .certExpiryDays }} -out ${HARBOR_CA_CRT} -subj "/CN={{ .registryDomain }}" -addext "subjectAltName = DNS:{{ .registryDomain }}, IP:{{ .ip }}"
sudo chmod 600 ${HARBOR_CA_KEY} ${HARBOR_CA_CRT}
sudo cp ${HARBOR_CA_CRT} /etc/ssl/certs/
# 返回harbor安装路径
cd ${ABS_PATH}

info_log "Finished generate cert"

mkdir -p "${HARBOR_INSTALL_PATH}"
sudo tar -xvf ./${ARCH}-bin/harbor-online-installer-v${HARBOR_VERSION}.tgz -C "${HARBOR_INSTALL_PATH}"

cd ${HARBOR_INSTALL_PATH}/harbor
sudo mv harbor.yml.tmpl harbor.yml
sudo sed -i 's/reg.mydomain.com/{{ .registryDomain }}/' harbor.yml

# 修改证书
sudo sed -i "s|/your/certificate/path|${HARBOR_CERT_PATH}/${HARBOR_CA_CRT}|" harbor.yml
sudo sed -i "s|/your/private/key/path|${HARBOR_CERT_PATH}/${HARBOR_CA_KEY}|" harbor.yml

# 修改密码
sudo sed -i "s|harbor_admin_password: .*|harbor_admin_password: {{ .registryPassword }}|" harbor.yml
sudo sed -i 's/^  password: .*/  password: {{ .registryDBPassword }}/' harbor.yml
# 修改数据卷
mkdir -p "${HARBOR_DATA_VOLUME}"
sudo sed -i "s|data_volume: /data|data_volume: ${HARBOR_DATA_VOLUME}|" harbor.yml

# 不再检查docker与docker-compose是否存在
sudo sed -i 's/^check_docker$/# check_docker/' install.sh
sudo sed -i 's/^check_dockercompose$/# check_dockercompose/' install.sh
sudo sed -i 's/^DOCKER_COMPOSE=docker-compose$/DOCKER_COMPOSE="nerdctl compose"/' install.sh
sudo sed -i 's/docker run --rm -v/nerdctl run --rm -v/' prepare

# 修改端口号
sudo sed -i "s/port: 80/port: {{ .HttpPort }}/" harbor.yml
sudo sed -i "s/port: 443/port: {{ .HttpsPort }}/" harbor.yml

info_log "start installing harbor"

mv install.sh install-harbor.sh
./install-harbor.sh
sudo chmod 400 harbor.yml

cd ${ABS_PATH}

source ./postinstall.sh
