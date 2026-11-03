# 路径挂载进行集群安装

本文档介绍使用路径挂载来进行集群的安装部署。

## 背景信息

因集群安装的数据默认都是存储于系统盘，如遇机器的系统盘大小受限，集群部署也会相应受到阻碍。

## 前提条件

使用openFuyao工具安装的集群。

## 使用限制

机器上存在系统盘和数据盘，下面假设系统盘挂载在`/`目录，数据盘挂载在`/home`目录。

## 使用场景

- 在线场景下，K3s引导集群安装。
- 离线场景下，安装K3s引导集群。
- 管理和业务K8s集群安装。

## K3s引导集群安装

openFuyao使用K3s集群作为引导集群，是一个简易版的K8s集群，提供临时性的集群管理功能。引导节点初始化提供的可挂载目录如下。

- 初始化的全局工作目录，这个目录会提供`OS`相关的`RPM`文件存储、集群安装需要的二进制文件（比如containerd、kubelet、kubectl等）和离线的镜像包。
- 集群运行时的数据存储目录`/var/lib`，这个目录支持进行独立磁盘挂载。

### 在线安装K3s引导集群

可以通过文件`/opt/BKE_WORKSPACE`或者环境变量`BKE_WORKSPACE`指定`bke init`的全局工作目录，环境变量的优先级更高，如果都没有指定的话，就是默认的`/bke`目录。

本章节以文件的方式进行自定义初始化目录的引导节点初始化，按照如下步骤。

1. 设置文件内容。

   ```bash
   echo "/home" > /opt/BKE_WORKSPACE
   ```

   

2. 挂载`/var/lib`运行时目录。

   下面介绍从`/dev/vdb`创建新分区，然后挂载到`/var/lib`目录。

   ```bash
   # 创建分区
   fdisk /dev/vdb
   # 进入到交互界面
   # 输入 n 新建分区
   # 选择分区号（默认即可，如 2）
   # 起始扇区（默认回车，使用剩余空间起始位置）
   # 结束扇区：+20G
   # 输入 w 保存退出
   
   # 让系统识别到新分区并格式化新分区
   partprobe /dev/vdb && mkfs.ext4 /dev/vdb2
   
   # 挂载目录
   mv /var/lib /var/lib.old
   mkdir /var/lib
   mount /dev/vdb2 /var/lib
   cp -a /var/lib.old/* /var/lib/
   
   # 设置开机自动挂载
   echo "UUID=$(blkid -s UUID -o value /dev/vdb2) /var/lib ext4 defaults 0 0" >> /etc/fstab
   ```

   

3. 下载`bkeadm`二进制安装工具。

   ```bash
   # 下载 download.sh 脚本
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
   chmod +x download.sh && ./download.sh
   ```

   

4. 引导节点初始化。

   ```bash
   bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest
   ```

   

### 离线安装K3s引导集群

可以通过文件`/opt/BKE_WORKSPACE`或者环境变量`BKE_WORKSPACE`指定`bke init`的全局工作目录，环境变量的优先级更高，如果都没有指定的话，就是默认的`/bke`目录。

离线安装包的构建参考[离线安装引导集群](../../zh/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)。

本章节以环境变量的方式进行自定义初始化目录的引导节点初始化，按照如下步骤。

1. 设置环境变量。

   ```bash
   # 设置环境变量，仅当前窗口生效，/home是举例参考，实际环境需要替换
   export BKE_WORKSPACE="/home"
   
   # 设置环境变量，所有窗口永久生效
   echo 'export BKE_WORKSPACE="/home"' >> ~/.bashrc
   source ~/.bashrc
   ```

   

2. `/var/lib`目录挂载，操作与[在线安装](#在线安装k3s引导集群)步骤2类似。

3. 解压缩离线安装包。

   ```bash
   # bke.tzr.gz是离线压缩包名称，实际环境上替换为实际设置的目录
   tar -xzvf bke.tar.gz -C $BKE_WORKSPACE
   ```

   

4. 移动`bkeadm`二进制安装工具到系统环境目录下。

   ```bash
   ARCH=$(uname -m)
   case $ARCH in
   x86_64) ARCH="amd64";;
   aarch64) ARCH="arm64";;
   esac
   mv $BKE_WORKSPACE/usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
   ```

   

5. 初始化引导节点。

   ```bash
   bke init
   ```

   

## 管理和业务K8s集群安装

在每个节点上进行`/var/lib`目录的挂载，参考[K3s引导集群在线安装](#在线安装k3s引导集群)，然后使用引导集群部署管理或者业务集群即可。

