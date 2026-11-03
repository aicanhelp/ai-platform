#!/bin/bash

url=$1


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"
log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    package-downloader.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    package-downloader.sh     $msg"
}

# 检查 runc 版本
runc_version=$(runc -v | grep -oP 'runc version \K[0-9]+\.[0-9]+\.[0-9]+')

# 检查版本是否小于 1.1.12
if [[ "$(echo -e "$runc_version\n1.1.12" | sort -V | head -n1)" == "1.1.12" ]]; then
    log "runc 版本 $runc_version 已经是最新版本，无需更新。"
else
    log "runc 版本 $runc_version 需要更新。"


    arch=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')


    /etc/openFuyao/bkeagent/scripts/file-downloader.sh $url/runc-$arch /usr/bin/runc 755

    systemctl daemon-reload
    systemctl restart docker

    log "runc 更新完成"
fi

