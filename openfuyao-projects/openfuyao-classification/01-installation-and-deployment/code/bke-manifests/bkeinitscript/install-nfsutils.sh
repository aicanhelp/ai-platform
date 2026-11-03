#!/bin/bash

log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"
log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-nfsutils.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-nfsutils.sh     $msg"
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

# 2.下载 nfs-utils
if [ "$os_type" == "ubuntu" ]; then
    /etc/openFuyao/bkeagent/scripts/package-downloader.sh nfs-common
# centos 7
elif [ "$os_type" == "centos" ] || [ "$os_type" == "kylin" ]; then
    /etc/openFuyao/bkeagent/scripts/package-downloader.sh nfs-utils
else
    log "不支持的操作系统 $os_type $os_version"
    exit 1
fi

log "nfs-utils 安装完成"