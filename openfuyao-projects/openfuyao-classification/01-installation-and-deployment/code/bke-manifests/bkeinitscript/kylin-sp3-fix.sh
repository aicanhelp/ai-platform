#!/bin/bash

log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"
log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    kylin-sp3-fix.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    kylin-sp3-fix.sh     $msg"
}


#判断是否是kylin
if [ ! -f /etc/kylin-release ]; then
    log "不是kylin系统"
    exit 0
fi


# remove docker-runc and podman

log "删除 docker-runc"
yum autoremove -y docker-runc


log "删除 podman"
yum autoremove -y podman

