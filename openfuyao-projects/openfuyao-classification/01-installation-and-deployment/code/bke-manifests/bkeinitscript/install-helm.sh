#!/bin/bash

url=$1


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"

log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-helm.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-helm.sh     $msg"
}

# 1.检查是否是Linux系统
if [ -f /etc/os-release ]; then
    source /etc/os-release
    os_type=$ID
    os_version=$VERSION_ID
# 若不是Linux系统，则显示不支持
else
    log "不支持的操作系统"
    exit 1
fi


arch=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')

if [ -f "/usr/bin/helm" ]  || command -v helm > /dev/null 2>&1; then
    log "helm 已安装"
    exit 0
fi

delete_path="false"
helm_path="/opt/helm"
if [ -d "$helm_path" ]; then
    log "helm path already exists"
else
    mkdir -p "$helm_path"
    delete_path="true"
    log "创建 helm 目录"
fi

/etc/openFuyao/bkeagent/scripts/file-downloader.sh $url/helm-v3.14.2-linux-$arch.tar.gz $helm_path/helm-v3.14.2-linux-$arch.tar.gz 777

tar -zxvf $helm_path/helm-v3.14.2-linux-$arch.tar.gz -C $helm_path
OS=$(echo `uname`|tr '[:upper:]' '[:lower:]')
mv -f $helm_path/${OS}-$arch/helm /usr/bin/helm

if [ "$delete_path" = "true" ]; then
    rm -rf "$helm_path"
    log "删除 helm 目录"
fi

log "helm 安装完成"