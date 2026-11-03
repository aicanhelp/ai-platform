#!/bin/bash


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"

log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-lxcfs.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    install-lxcfs.sh     $msg"
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

# 2.创建目录/var/lib/lxc/lxcfs 如果不存在
if [ ! -d "/var/lib/lxc/lxcfs" ]; then
    log "创建目录 /var/lib/lxc/lxcfs"
    mkdir -p /var/lib/lxc/lxcfs
fi

# 3.下载 lxcfs
if [ "$os_type" == "ubuntu" ]; then
    /etc/openFuyao/bkeagent/package-downloader.sh lxcfs
# centos 7
elif [ "$os_type" == "centos" ] && [ "$os_version" == "7" ]; then
    /etc/openFuyao/bkeagent/package-downloader.sh lxcfs
    /etc/openFuyao/bkeagent/package-downloader.sh fuse-libs
# centos 8
elif [ "$os_type" == "centos" ] && [ "$os_version" == "8" ]; then
    /etc/openFuyao/bkeagent/package-downloader.sh lxcfs
# kylin
elif [ "$os_type" == "kylin" ]; then
    /etc/openFuyao/bkeagent/package-downloader.sh lxcfs
    /etc/openFuyao/bkeagent/package-downloader.sh lxcfs-tools
elif [ "$os_type" == "openEuler" ]; then
    log "openEuler not install lxcfs"
else
    log "不支持的操作系统 $os_type $os_version"
    exit 1
fi

# 4.modeprobe fuse
modprobe fuse &> /dev/null


# 5.修改service文件
if [ -f /usr/lib/systemd/system/lxcfs.service ]; then
   # 替换 /var/lib/lxcfs 为 /var/lib/lxc/lxcfs
  sed -i 's/\/var\/lib\/lxcfs/\/var\/lib\/lxc\/lxcfs/g' /usr/lib/systemd/system/lxcfs.service
  # 重启lxcfs
  systemctl daemon-reload
  systemctl restart lxcfs
  systemctl enable lxcfs

  log "lxcfs 安装成功"
fi



