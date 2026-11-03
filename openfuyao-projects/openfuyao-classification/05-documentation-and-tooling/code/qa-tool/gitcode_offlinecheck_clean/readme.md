# 离线校验和镜像残留及文件清理工具
## 功能
- 离线校验：使用ping命令进行校验，ping不通给定的所有目标则为离线，支持自定义目标
- 残留镜像检测：检测是否有containerd和cri镜像残留
- 残留文件检测：支持指定需要检测的文件
- 资源清理：清理所有containerd和cri资源和指定文件
- 查看现有资源：列出现有containerd和cri资源
## 用法
- 显示帮助信息
```bash
offlinecheck.bash -h
```
- 检测是否符合测试环境（离线、无资源残留）
```bash
offlinecheck.bash
```
- 清理资源
```bash
offlinecheck.bash -c
offlinecheck.bash -f # 强制清理
```
- 列出现有资源
```bash
offlinecheck.bash -l
```
## 配置文件
变量名
- NETWORK_TARGETS：检测是否离线的ping目标 （数组）
- CONTAINERD_NAMESPACES：需要检测的containerd资源的命名空间 （数组）
- CHECK_DIRS：检测是否有残留的文件路径 （数组）
- SIZE_THRESHOLD：可以指定文件大小阈值（MB），大于阈值认为是残留文件
- CLEANUP_DIRS：需要清理的文件路径 （数组）

配置文件路径默认为当前文件夹下 `checker_config.conf` 文件
