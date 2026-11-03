#!/bin/bash

url=$1

log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"

log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-calicoctl.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-calicoctl.sh     $msg"
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

#创建目录/etc/calico
if [ ! -d "/etc/calico" ]; then
     mkdir -p /etc/calico
fi

# 写入配置
echo -e "apiVersion: projectcalico.org/v3\nkind: CalicoAPIConfig\nmetadata:\nspec:\n  datastoreType: 'kubernetes'\n  kubeconfig: '/etc/kubernetes/admin.conf'" > /etc/calico/calicoctl.cfg

arch=$(uname -m | sed 's/x86_64/amd64/;s/aarch64/arm64/;s/^unknown$/amd64/')

/etc/openFuyao/bkeagent/scripts/file-downloader.sh $url/calicoctl-v3.25.0-linux-$arch /usr/bin/calicoctl 755

log "calicoctl 安装完成"