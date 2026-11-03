# v1.28.8版本K8s集群离线安装指导

## 背景信息

openFuyao社区使用的K8s版本是不断演进的，比如已发布的v25.12版本使用的K8s版本为v1.34.3，但社区仍然存在使用K8s低版本的集群，本文档旨在提供安装低版本K8s集群的解决方案。

## 前提条件

无。

## 使用限制

- 离线安装。
- 使用的K8s的镜像和二进制都是社区自构建的。

## 使用场景

- 引导节点已初始化，安装v1.28.8版本K8s业务集群。
- 引导节点未初始化，安装v1.28.8版本K8s业务集群。

## 操作步骤

对于上面的两种使用场景，下面分开介绍。

### 引导节点已初始化，安装v1.28.8版本K8s业务集群

按照下面步骤进行操作。

1. 构建增量离线镜像包。

   下面是配置文件，以component.yaml命名，便于后面操作说明。

   ```yaml
   registry:
     imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
     architecture:
       - amd64
       - arm64
   repos:
     - architecture:
         - amd64
         - arm64
       needDownload: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               tag:
                 - 1.28.8-of.1
             - name: kube-controller-manager
               tag:
                 - 1.28.8-of.1
             - name: kube-scheduler
               tag:
                 - 1.28.8-of.1
             - name: kube-proxy
               tag:
                 - 1.28.8-of.1
             - name: etcd
               tag:
                 - 3.5.12-of.1
   files:
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/arm64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.28.8-of.1-arm64
         - fileName: kubelet
           fileAlias: kubelet-v1.28.8-of.1-arm64
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/amd64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.28.8-of.1-amd64
         - fileName: kubelet
           fileAlias: kubelet-v1.28.8-of.1-amd64
   ```

   执行如下命令，构建增量离线包。

   ```bash
   # 下载 download.sh 脚本
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
   chmod +x download.sh && ./download.sh
   
   # 构建增量包
   bke build patch -f component.yaml -t component.tar.gz --strategy=oci
   ```

   

2. 同步镜像和二进制到引导节点。

   将步骤1中制备的离线包复制到引导节点上，执行如下命令进行同步。

   ```bash
   # 解压离线包
   tar -xzvf component.tar.gz
   
   # 同步镜像和复制二进制文件
   bke registry patch --source ./component --target 127.0.0.1:40443
   cp ./component/volumes/kubectl-v1.28.8-of.1-arm64 /bke/mount/source_registry/files
   cp ./component/volumes/kubelet-v1.28.8-of.1-arm64 /bke/mount/source_registry/files
   cp ./component/volumes/kubectl-v1.28.8-of.1-amd64 /bke/mount/source_registry/files
   cp ./component/volumes/kubelet-v1.28.8-of.1-amd64 /bke/mount/source_registry/files
   ```

   

3. 上传新的版本文件。

   执行如下命令下载索引文件index.yaml。

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/index.yaml
   ```

   index.yaml文件的内容如下。

   ```yaml
   - openFuyaoVersion: latest
     filePath: ./VersionConfig-latest.yaml
   - openFuyaoVersion: v25.12-rc.2
     filePath: ./Core-VersionConfig-v25.12-rc.2.yaml
   - openFuyaoVersion: v25.12-rc.3
     filePath: ./Core-VersionConfig-v25.12-rc.3.yaml
   - openFuyaoVersion: v25.12
     filePath: ./Core-VersionConfig-v25.12.yaml
   - openFuyaoVersion: v25.12-rc.5
     filePath: ./Core-VersionConfig-v25.12-rc.5.yaml
   - openFuyaoVersion: v26.03-rc.1
     filePath: ./Core-VersionConfig-v26.03-rc.1.yaml
   ```

   下载对应的版本文件，若要安装v26.03-rc.1版本的openFuyao集群，请执行如下命令。

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-v26.03-rc.1.yaml
   ```

   修改版本文件中的K8s版本相关字段和镜像tag等，具体修改点如下。

   ```yaml
   kubernetesVersion: v1.34.3-of.1 # 修改为 v1.28.8-of.1
   etcdVersion: v3.6.7-of.1 # 修改为 v3.5.12-of.1
   
   repos:
     - architecture: # k8s 组件, need updated by Orchestration-engine SIG
         - amd64
         - arm64
       needDownload: true
       isKubernetes: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao/
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               usedPodInfo:
                 - podPrefix: kube-apiserver
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # APIServer镜像tag，修改为 1.28.8-of.1
             - name: kube-controller-manager
               usedPodInfo:
                 - podPrefix: kube-controller-manager
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # controller-manager镜像tag，修改为 1.28.8-of.1
             - name: kube-scheduler
               usedPodInfo:
                 - podPrefix: kube-scheduler
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # scheduler镜像tag，修改为 1.28.8-of.1
             - name: kube-proxy
               usedPodInfo:
                 - podPrefix: kube-proxy
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # kube-proxy镜像tag，修改为 1.28.8-of.1
             - name: etcd
               usedPodInfo:
                 - podPrefix: etcd
                   namespace: kube-system
               tag:
                 - 3.6.7-of.1 # etcd镜像tag，修改为 3.5.12-of.1
   addons:
     - name: kubeproxy
       version: v1.34.3-of.1 # kube-proxy版本，修改为 v1.28.8-of.1
   ```

   登录引导节点的openFuyao管理面，左侧导航栏选择“集群生命周期管理>补丁上传”，选择步骤3中已修改完成的版本文件并上传。
   > ![输入图片说明](figures/icon-notice.gif) **注意：**<br />
   > 待安装的业务集群的所有镜像需要存在于本地镜像源服务。

4. 业务集群安装。

    openFuyao管理面选择对应的版本安装业务集群，参考[管理面安装业务集群](../../zh/cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md)。


### 引导节点未初始化，安装v1.28.8版本K8s业务集群

按照下面步骤进行操作。

1. 准备版本文件。
   执行如下命令，下载索引文件。

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/index.yaml
   ```

   索引文件的内容如下。

   ```yaml
   - openFuyaoVersion: latest
     filePath: ./VersionConfig-latest.yaml
   - openFuyaoVersion: v25.12-rc.2
     filePath: ./Core-VersionConfig-v25.12-rc.2.yaml
   - openFuyaoVersion: v25.12-rc.3
     filePath: ./Core-VersionConfig-v25.12-rc.3.yaml
   - openFuyaoVersion: v25.12
     filePath: ./Core-VersionConfig-v25.12.yaml
   - openFuyaoVersion: v25.12-rc.5
     filePath: ./Core-VersionConfig-v25.12-rc.5.yaml
   - openFuyaoVersion: v26.03-rc.1
     filePath: ./Core-VersionConfig-v26.03-rc.1.yaml
   ```

   下载对应的版本文件，若要安装v26.03-rc.1版本的openFuyao集群，请执行如下命令。

   ```bash
   curl -O https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/Core-VersionConfig-v26.03-rc.1.yaml
   ```

   修改版本文件中的K8s版本相关字段和镜像tag等，具体修改点如下。

   ```yaml
   kubernetesVersion: v1.34.3-of.1 # 修改为 v1.28.8-of.1
   etcdVersion: v3.6.7-of.1 # 修改为 v3.5.12-of.1
   
   repos:
     - architecture: # k8s 组件, need updated by Orchestration-engine SIG
         - amd64
         - arm64
       needDownload: true
       isKubernetes: true
       subImages:
         - sourceRepo: cr.openfuyao.cn/openfuyao/
           targetRepo: kubernetes
           images:
             - name: kube-apiserver
               usedPodInfo:
                 - podPrefix: kube-apiserver
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # APIServer镜像tag，修改为 1.28.8-of.1
             - name: kube-controller-manager
               usedPodInfo:
                 - podPrefix: kube-controller-manager
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # controller-manager镜像tag，修改为 1.28.8-of.1
             - name: kube-scheduler
               usedPodInfo:
                 - podPrefix: kube-scheduler
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # scheduler镜像tag，修改为 1.28.8-of.1
             - name: kube-proxy
               usedPodInfo:
                 - podPrefix: kube-proxy
                   namespace: kube-system
               tag:
                 - 1.34.3-of.1 # kube-proxy镜像tag，修改为 1.28.8-of.1
             - name: etcd
               usedPodInfo:
                 - podPrefix: etcd
                   namespace: kube-system
               tag:
                 - 3.6.7-of.1 # etcd镜像tag，修改为 3.5.12-of.1
   files:
     # 下载链接修改如下
     # address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/arm64/
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/arm64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.34.3-of.1-arm64 # kubectl别名，修改为 kubectl-v1.28.8-of.1-arm64
         - fileName: kubelet
           fileAlias: kubelet-v1.34.3-of.1-arm64 # kubelet别名，修改为 kubelet-v1.28.8-of.1-arm64
     # 下载链接修改如下
     # address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.28.8-of.1/bin/linux/amd64/
     - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/amd64/
       files:
         - fileName: kubectl
           fileAlias: kubectl-v1.34.3-of.1-amd64 # kubectl别名，修改为 kubectl-v1.28.8-of.1-amd64
         - fileName: kubelet
           fileAlias: kubelet-v1.34.3-of.1-amd64 # kubelet别名，修改为 kubelet-v1.28.8-of.1-amd64
   addons:
     - name: kubeproxy
       version: v1.34.3-of.1 # kube-proxy版本，修改为 v1.28.8-of.1
   ```

      

2. 构建离线安装包。

   ```bash
   # 下载 download.sh 脚本
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   # 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   # 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
   chmod +x download.sh && ./download.sh
   
   # 构建离线安装包
   bke build -f Core-VersionConfig-v26.03-rc.1.yamls -t bke-v26.03-rc.1.tar.gz
   ```

   离线安装包的制备参考[离线安装引导集群](../../zh/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)。

3. 引导节点初始化。

    引导节点初始化，参考[离线安装引导集群](../../zh/cluster_installation_guide/bootstrap_cluster_installation/offline_bootstrap_cluster_installation.md)。

4. 上传版本文件。

   登录引导节点的openFuyao管理面，左侧导航栏选择“集群生命周期管理>补丁上传”，选择步骤3中已修改完成的版本文件并上传。

5. 业务集群安装。

   openFuyao管理面选择对应的版本安装业务集群，参考[管理面安装业务集群](../../zh/cluster_installation_guide/service_cluster_installation/service_cluster_installation_on_the_management_plane.md)。

