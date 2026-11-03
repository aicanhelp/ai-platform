#!/bin/bash

# 通用下载程序
# 用法: ./download.sh <下载地址> <保存文件名> <文件权限>

url="$1"           # 下载地址
target_file="$2"   # 保存文件名
file_permission="$3"  # 文件权限

if [ -z "$url" ] || [ -z "$target_file" ] || [ -z "$file_permission" ]; then
    echo "用法: $0 <下载地址> <保存文件名> <文件权限>"
    exit 1
fi


log_file="/var/log/openFuyao/bkeagent.log"
[ -f "$log_file" ] || touch "$log_file"

log() {
  local msg=$1
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    file-downloader.sh     $msg" >> "$log_file"
  echo "$(date +'%Y-%m-%d %H:%M:%S')     INFO    file-downloader.sh     $msg"
}

# 检查是否存在 curl 命令
if command -v curl > /dev/null; then
    log "正在使用 curl 下载 $url 到 $target_file ..."
    curl -sSf -o "$target_file" "$url"
elif command -v wget > /dev/null; then
    log "正在使用 wget 下载 $url 到 $target_file ..."
    wget -q -O "$target_file" "$url"
else
    log "无法找到 curl 或 wget 命令，无法下载文件"
    exit 1
fi

# 检查下载工具的退出状态
if [ $? -eq 0 ]; then
    if [ -z "$file_permission" ]; then
        log "文件权限未设置"
        log "下载完成"
        exit 0
    fi
    chmod "$file_permission" "$target_file"  # 设置文件权限
    log "下载 $url 完成，文件权限已设置为 $file_permission"
else
    log "下载 $url 失败"
fi
