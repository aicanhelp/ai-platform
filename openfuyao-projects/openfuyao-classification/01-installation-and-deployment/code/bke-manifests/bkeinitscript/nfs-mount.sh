#!/bin/bash

nfs_server=$1
src_path=$2
dst_path=$3


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"
log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    nfs-mount.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    nfs-mount.sh     $msg"
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

# 创建目录
mkdir -p $dst_path
# 挂载
mount -t nfs $nfs_server:$src_path $dst_path

# 判断是否已经写入到/etc/fstab
if grep -q "$nfs_server:$src_path" /etc/fstab; then
    log "已经写入 $nfs_server:$src_path 到 $dst_path"
    exit 0
else
    log "写入 $nfs_server:$src_path 到 $dst_path"
    echo "$nfs_server:$src_path $dst_path nfs _netdev 0 0" >> /etc/fstab
fi
log "挂载 $nfs_server:$src_path 到 $dst_path 完成"

