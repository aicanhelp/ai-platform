#!/bin/bash

url=$1


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"

log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-etcdctl.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-etcdctl.sh     $msg"
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


/etc/openFuyao/bkeagent/scripts/file-downloader.sh $url/etcdctl-v3.5.6-linux-$arch /usr/bin/etcdctl 755

log "etcdctl 安装完成"