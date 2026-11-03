#!/bin/bash

# 通用下载程序
# 用法: ./download.sh <软件包名称>

package_name="$1"  # 要下载的软件包名称

if [ -z "$package_name" ]; then
    echo "用法: $0 <软件包名称>"
    exit 1
fi


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"
log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    package-downloader.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    package-downloader.sh     $msg"
}

# 检查是否存在 apt 命令
if command -v apt > /dev/null; then
    log "正在使用 apt 安装 $package_name ..."
    sudo apt install -y "$package_name"
# 检查是否存在 yum 命令
elif command -v yum > /dev/null; then
    log "正在使用 yum 安装 $package_name ..."
    sudo yum install -y "$package_name"
else
    log "无法找到适用的包管理器，无法安装软件包"
    exit 1
fi

log "$package_name 安装完成"
