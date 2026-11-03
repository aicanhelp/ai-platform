# 离线openFuyao-system-controller手动部署

## 背景信息

本文档介绍在离线场景下，先制备并部署 `K8s-only` 离线包，创建不包含 `openFuyao-system-controller` 的业务集群，再通过增量包补充 `openFuyao-system-controller` 及其依赖镜像和 Chart，最后在业务集群中手动部署 `openFuyao-system-controller` 组件，拉起 OpenFuyao 管理面。

<!-- 整体流程分为五部分：

1. K8s 镜像包构建
2. 在引导节点上安装仅含 K8s 镜像的引导集群
3. 由引导集群拉起只有 K8s 的业务集群
4. 新增 openFuyao-system-controller 镜像包制备与上传
5. 在业务集群上部署 openFuyao-system-controller 组件 -->

## 前提条件

已准备以下机器：

| 角色 | 示例 IP | 用途 |
| --- | --- | --- |
| 构建机 | `192.168.200.210` | 构建 `K8s-only` 离线包和增量包 |
| 引导节点 | `192.168.200.217` | 部署引导节点，提供镜像仓库和 Chart 仓库 |
| 业务集群节点 | `192.168.200.216` | 创建只包含 K8s 的业务集群，并增量安装 controller |

机器硬件配置可参考[快速入门](https://docs.openfuyao.cn/zh/docs/v26.03/quick_start.html#快速安装)中的配置要求；构建机主要用于制备离线包和增量包，建议磁盘不小于 100GB，并预留不少于 50GB 可用空间用于保存下载内容和构建产物。

已准备以下文件：

| 文件 | 用途 | 准备方式 |
| --- | --- | --- |
| `K8s-only.yaml` | 构建第一阶段 `K8s-only` 离线包 | 参考本文[附录](#附录)中的 `K8s-only` 下载配置，按实际版本调整 |
| `offline-incremental.yaml` | 构建第二阶段增量包 | 参考本文[附录](#附录)中的增量组件下载配置，按实际版本调整 |
| `openFuyao-system-controller.yaml` | 部署业务集群 controller 组件 | 后续在 [5.2 章节](#52-准备-openfuyao-system-controller-yaml) 从引导节点拷贝并渲染生成 |
| `VersionConfig-latest.yaml` | `latest` 版本离线安装使用的组件版本配置 | 随离线包下载到引导节点；其他版本需替换为对应版本的文件|

本文附录中的 YAML 示例适用于 `latest` 版本离线安装场景。可查看离线包中的版本配置文件确认当前版本：`VersionConfig-latest.yaml` 且 `openFuyaoVersion: latest` 表示 `latest` 版本；其他版本请以对应版本配置文件中的镜像 tag、Chart 文件名和下载地址为准。

## 使用限制

需要离线安装。

## 使用教程

### 1. K8s 镜像包构建

#### 1.1 K8s-only.yaml 需要保留的镜像

纯 K8s 集群实际使用镜像：

```text
deploy.bocloud.k8s:40443/kubernetes/bkeagent-deployer:latest
deploy.bocloud.k8s:40443/kubernetes/calico/cni:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/calico/kube-controllers:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/calico/node:v3.31.3
deploy.bocloud.k8s:40443/kubernetes/coredns:1.12.2-of.1
deploy.bocloud.k8s:40443/kubernetes/etcd:3.6.7-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-apiserver:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-controller-manager:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-proxy:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/kube-scheduler:1.34.3-of.1
deploy.bocloud.k8s:40443/kubernetes/pause:3.9
```

引导节点实际使用镜像：

```text
deploy.bocloud.k8s:443/kubernetes/bke-console-service:latest
deploy.bocloud.k8s:443/kubernetes/bke-console-website:latest
deploy.bocloud.k8s:443/kubernetes/bke-manifests:latest
deploy.bocloud.k8s:443/kubernetes/busybox:1.36.1
deploy.bocloud.k8s:443/kubernetes/cluster-api/cluster-api-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api/kubeadm-bootstrap-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api/kubeadm-control-plane-controller:v1.4.3
deploy.bocloud.k8s:443/kubernetes/cluster-api-provider-bke:latest
deploy.bocloud.k8s:443/kubernetes/coredns:1.12.2-of.1
deploy.bocloud.k8s:443/kubernetes/ingress-nginx/controller:v1.9.4
deploy.bocloud.k8s:443/kubernetes/ingress-nginx/kube-webhook-certgen:v20231011-8b53cabe0
deploy.bocloud.k8s:443/kubernetes/installer-service:latest
deploy.bocloud.k8s:443/kubernetes/installer-website:latest
deploy.bocloud.k8s:443/kubernetes/oauth-proxy:latest
deploy.bocloud.k8s:443/kubernetes/oauth-server:latest
deploy.bocloud.k8s:443/kubernetes/oauth-webhook:latest
deploy.bocloud.k8s:443/kubernetes/plugin-management-service:latest
deploy.bocloud.k8s:443/kubernetes/user-management-operator:latest
```

`K8s-only` 离线包需要同时保留纯 K8s 集群镜像，以及引导节点自身运行所需镜像和 Chart。此阶段不包含 `openFuyao-system-controller` 及其后续管理面业务组件。

YAML 文件格式参考本文档[附录](#附录)章节。

#### 1.2 构建 K8s-only 离线包

执行构建前，请确认构建机已安装 `bke` 命令。若未安装，可执行如下命令下载并准备：

```bash
# 下载 download.sh 脚本
curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
# 下载 download.sh 的校验文件并进行校验；校验成功会输出 -: OK，校验失败请联系 openFuyao sig-installation邮箱installation@list.openfuyao.cn或openFuyao 社区邮箱 contact@openfuyao.cn
curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
sha256sum -c <(cat download.sh.sha256) < download.sh
# 运行 download.sh 下载 BKE 安装工具，执行过程中会校验安装工具的 sha256sum
chmod +x download.sh && ./download.sh

ARCH=$(uname -m)
case $ARCH in
x86_64) ARCH="amd64";;
aarch64) ARCH="arm64";;
esac

mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
chmod +x /usr/local/bin/bke
bke version
```

> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />
> 本文以 `latest` 版本为例。若使用指定 openFuyao 版本，请将下载地址中的 `latest` 替换为对应版本，并确保 `K8s-only.yaml` 中的镜像 tag、Chart 文件名和版本配置文件同步替换。

在构建机执行：

```bash
rm -rf /bke
bke build -f K8s-only.yaml -t K8s-only.tar.gz
```

将离线包传到引导节点：

```bash
scp /root/K8s-only.tar.gz root@192.168.200.217:/root/
```

### 2. 在引导节点上安装仅含 K8s 镜像的引导集群

#### 2.1 清理引导节点旧环境

在引导节点执行：

```bash
bke reset --all --mount
rm -rf /bke
rm -f /usr/local/bin/bke
```

#### 2.2 解压 K8s-only 离线包

```bash
tar -zxf /root/K8s-only.tar.gz -C /
ls -lh /bke/volumes/image.tar.gz /bke/volumes/source.tar.gz
```

#### 2.3 准备 bke 命令

```bash
ARCH=$(uname -m)
case $ARCH in
x86_64) ARCH="amd64";;
aarch64) ARCH="arm64";;
esac
mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
chmod +x /usr/local/bin/bke
bke version
```

如果当前离线包中没有对应架构的 `bkeadm_linux_$ARCH`，需要额外准备同架构的 `bke` 二进制文件。

#### 2.4 断网

```bash
sed -i 's/^\s*nameserver/# nameserver/' /etc/resolv.conf
```

#### 2.5 初始化引导节点

```bash
bke init
```

如需显式指定端口，可使用：

```bash
bke init --kubernetesPort 6443 --imageRepoPort 40443
```

#### 2.6 检查引导节点

```bash
kubectl get pods -A
kubectl get pods -A | grep openfuyao-system-controller
```

第二条命令应无输出，表示引导节点未部署 `openFuyao-system-controller`。

### 3. 由引导集群拉起只有 K8s 的业务集群

#### 3.1 创建业务集群

登录引导节点页面（访问地址为 `https://<引导节点IP>:30010`，本文示例为 `https://192.168.200.217:30010`），创建业务集群。

创建业务集群时，不添加 `openFuyao-system-controller` 组件。

#### 3.2 检查业务集群

在业务集群节点执行：

```bash
kubectl get nodes
kubectl get pods -A
crictl images
```

确认业务集群中只有纯 K8s 相关组件，没有 `openFuyao-system-controller` 和 OpenFuyao 管理面业务组件。

### 4. 新增 openFuyao-system-controller 镜像包制备与上传

#### 4.1 构建增量包

在构建机准备 `offline-incremental.yaml` 后执行：
YAML 文件格式参考本文档[附录](#附录)章节。

```bash
bke build patch -f offline-incremental.yaml -t patch-log.tar.gz --strategy oci
```
> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />本文以 OCI 格式构建增量包（`--strategy oci`），可不依赖 Docker。

patch-log.tar.gz解压后应包含：

```text
patch-log/
├── manifests.yaml
└── volumes/
    ├── oci-layout/
    └── charts/
        ├── application-management-service-0.0.0-latest.tgz
        ├── console-service-0.0.0-latest.tgz
        └── ...
```

> ![输入图片说明](../figures/icon-notice.gif) **注意：**<br />`bke build patch` 只负责把二进制、Chart 等收集进增量包，不会把它们导入引导节点已有的系统包/二进制仓库（HTTP 仓）或 Chart Registry；导入镜像需另行执行 `bke registry patch`，二进制与 Chart 也需按对应流程手动导入。

#### 4.2 传输增量包到引导节点

```bash
scp /root/patch-log.tar.gz root@192.168.200.217:/root/
```

#### 4.3 在引导节点导入增量镜像

在引导节点执行：

```bash
cd /root
rm -rf patch-log
tar -xzvf patch-log.tar.gz

bke registry patch --source ./patch-log --target 0.0.0.0:40443
```

> ![输入图片说明](../figures/icon-notice.gif) **注意：**<br />`bke registry patch` 只导入增量镜像到 `40443` 镜像仓库，不会导入 `volumes/charts/*.tgz`，也不会导入二进制/系统包到 HTTP 仓。

检查镜像：

```bash
curl -k https://127.0.0.1:40443/v2/kubernetes/openfuyao-system-controller/tags/list
curl -k https://127.0.0.1:40443/v2/kubernetes/application-management-service/tags/list
```

#### 4.4 在引导节点同步增量 Chart

查看增量包中的 Chart：

```bash
find /root/patch-log -name "*.tgz"
```

上传 Chart 到引导节点 ChartMuseum：

```bash
find /root/patch-log/volumes/charts -name "*.tgz" -print0 | while IFS= read -r -d '' f; do
  echo "upload $f"
  curl --data-binary "@$f" http://127.0.0.1:38080/api/charts
done
```

如果返回 `permission denied`，需要先处理引导节点 ChartMuseum 挂载目录权限，再重新上传。

检查 Chart：

```bash
curl http://127.0.0.1:38080/index.yaml | grep -E "application-management-service|console-service|console-website|marketplace-service|harbor|monitoring-service|web-terminal-service"
```

### 5. 在业务集群上部署 openFuyao-system-controller 组件

#### 5.1 在业务集群创建 patch-config
> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />
> `VersionConfig-latest.yaml` 仅适用于 `latest` 版本离线安装场景。
> 用户可在引导节点执行以下命令确认离线包实际携带的版本配置：
> ```bash
> ls /bke/mount/source_registry/files/patches/
> find /bke/mount/source_registry/files/patches -maxdepth 1 -type f \
>   \( -name "VersionConfig-*.yaml" -o -name "Core-VersionConfig-*.yaml" \) \
>   -exec grep -n "^openFuyaoVersion:" {} \;
> ```
> 如果版本配置文件为 `VersionConfig-latest.yaml`，且 `openFuyaoVersion` 为 `latest`，则使用下方示例命令；如果是 `v26.06-rc.1` 等指定版本，请将下方命令中的 `VersionConfig-latest.yaml`  替换为实际版本配置文件名。

在业务集群节点从引导节点拷贝版本配置：

```bash
scp root@192.168.200.217:/bke/mount/source_registry/files/patches/VersionConfig-latest.yaml /tmp/VersionConfig-latest.yaml
```

创建 ConfigMap：

```bash
kubectl create ns openfuyao-system-controller --dry-run=client -o yaml | kubectl apply -f -

kubectl create cm patch-config \
  -n openfuyao-system-controller \
  --from-file=patch-data=/tmp/VersionConfig-latest.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

kubectl get cm patch-config -n openfuyao-system-controller
```

#### 5.2 准备 openFuyao-system-controller YAML

在引导节点执行：

```bash
cp /etc/openFuyao/addons/manifests/kubernetes/openfuyao-system-controller/latest/openfuyao-system-controller.yaml /root/openfuyao-system-controller.yaml
```

渲染模板：

```bash
sed -i 's#{{ if .repo }}{{ .repo }}{{ else }}cr.openfuyao.cn/openfuyao/{{ end }}#deploy.bocloud.k8s:40443/kubernetes/#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .tagVersion }}#latest#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .repo }}#deploy.bocloud.k8s:40443/kubernetes#g' /root/openfuyao-system-controller.yaml
sed -i 's#{{ .helmRepo }}#http://192.168.200.217:38080#g' /root/openfuyao-system-controller.yaml
```

检查模板变量是否已清理：

```bash
grep -n "{{" /root/openfuyao-system-controller.yaml
```

检查关键字段：

```bash
grep -nE "image:|OPENFUYAO_REGISTRY|HELM_REPORITORY_URL|entrypoint.sh" /root/openfuyao-system-controller.yaml
grep -nA2 "OPENFUYAO_REGISTRY\|HELM_REPORITORY_URL" /root/openfuyao-system-controller.yaml
```

将文件拷贝到业务集群节点：

```bash
scp /root/openfuyao-system-controller.yaml root@192.168.200.216:/tmp/openfuyao-system-controller.yaml
```

#### 5.3 部署 openFuyao-system-controller

在业务集群节点执行：

```bash
kubectl apply -f /tmp/openfuyao-system-controller.yaml
kubectl get pods -n openfuyao-system-controller -w
```

#### 5.4 检查安装结果

查看全量组件：

```bash
kubectl get pods -A
```

查看异常 Pod：

```bash
kubectl get pods -A | grep -Ev "Running|Completed"
```

如果无异常输出，表示 `openFuyao-system-controller` 及其拉起的管理面组件已部署完成。

### 附录
> ![输入图片说明](../figures/icon-note.gif) **说明：**<br />下面的yaml文件仅适用于 `latest` 版本离线安装场景。

下面是K8s-only只含K8s的下载配置yaml文件。
```yaml
registry:
  imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
repos:
  - architecture: # k3s集群镜像
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        imageTrack: ""
        images:
          - name: registry
            tag:
              - 2.8.1
          - name: nginx
            tag:
              - 1.23.0-alpine
          - name: helm/chartmuseum
            tag:
              - v0.16.2
          - name: openebs/nfs-server-alpine
            tag:
              - 0.9.0
          - name: rancher/k3s
            tag:
              - v1.25.16-k3s4
          - name: rancher/mirrored-pause
            tag:
              - 3.6
  - architecture: # K8s 组件
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: coredns
            tag:
              - 1.12.2-of.1
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: kube-apiserver
            tag:
              - 1.34.3-of.1
          - name: kube-controller-manager
            tag:
              - 1.34.3-of.1
          - name: kube-scheduler
            tag:
              - 1.34.3-of.1
          - name: kube-proxy
            tag:
              - 1.34.3-of.1
          - name: etcd
            tag:
              - 3.6.7-of.1
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        images:
          - name: pause
            tag:
              - 3.9
      - sourceRepo: cr.openfuyao.cn/openfuyao/keepalived # keepalived 组件
        targetRepo: kubernetes/keepalived
        images:
          - name: keepalived
            tag:
              - 1.3.5
      - sourceRepo: hub.oepkgs.net/openfuyao # haproxy 组件
        targetRepo: kubernetes
        images:
          - name: haproxy
            tag:
              - 2.1.4
      - sourceRepo: hub.oepkgs.net/openfuyao/calico # calico 组件
        targetRepo: kubernetes/calico
        images:
          - name: cni
            tag:
              - v3.31.3
          - name: node
            tag:
              - v3.31.3
          - name: kube-controllers
            tag:
              - v3.31.3
  - architecture: # openFuyao组件
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao # 核心组件
        targetRepo: kubernetes
        images:
          - name: bke-console-website
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.3
          - name: bke-console-service
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.2
          - name: installer-website
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.2.2
          - name: installer-service
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.2.3
          - name: upgrade-path
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准
          - name: cluster-api-provider-bke  # kubernetes
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.3.0
          - name: bke-manifests  #   # kubernetes
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.2.3
          - name: bkeagent-deployer
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.3.0
          - name: oauth-proxy
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.2
          - name: oauth-server
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.3
          - name: oauth-webhook
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.3
          - name: plugin-management-service
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.2
          - name: user-management-operator
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.2
          - name: kubectl-openfuyao
            tag:
              - latest # 非 latest 版本请以对应 Core-VersionConfig 中该组件的实际 tag 为准，例如 v26.06-rc.1 为 1.0.2
      - sourceRepo: hub.oepkgs.net/openfuyao # busybox 组件
        targetRepo: kubernetes
        images:
          - name: busybox
            tag:
              - 1.36.1
      - sourceRepo: hub.oepkgs.net/openfuyao/cluster-api # cluster-api 组件
        targetRepo: kubernetes/cluster-api
        images:
          - name: cluster-api-controller
            tag:
              - v1.4.3
          - name: kubeadm-bootstrap-controller
            tag:
              - v1.4.3
          - name: kubeadm-control-plane-controller
            tag:
              - v1.4.3
      - sourceRepo: hub.oepkgs.net/openfuyao/ingress-nginx # ingress-nginx 组件
        targetRepo: kubernetes/ingress-nginx
        images:
          - name: controller
            tag:
              - v1.9.4
          - name: kube-webhook-certgen
            tag:
              - v20231011-8b53cabe0

rpms:
debs: []
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/arm64/
    files:
      - fileName: kubectl
        fileAlias: kubectl-v1.34.3-of.1-arm64
      - fileName: kubelet
        fileAlias: kubelet-v1.34.3-of.1-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/kubernetes/kubernetes/releases/download/1.34.3-of.1/bin/linux/amd64/
    files:
      - fileName: kubectl
        fileAlias: kubectl-v1.34.3-of.1-amd64
      - fileName: kubelet
        fileAlias: kubelet-v1.34.3-of.1-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - fileName: containerd-v2.1.1-linux-amd64.tar.gz
      - fileName: containerd-v2.1.1-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containernetworking/plugins/releases/download/v1.4.1/
    files:
      - fileName: cni-plugins-linux-amd64-v1.4.1.tgz
      - fileName: cni-plugins-linux-arm64-v1.4.1.tgz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/helm/releases/download/v3.14.2/
    files:
      - fileName: helm-v3.14.2-linux-amd64.tar.gz
      - fileName: helm-v3.14.2-linux-arm64.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/mikefarah/yq/releases/download/v4.43.1
    files:
      - fileName: yq_linux_arm64
      - fileName: yq_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/jqlang/jq/releases/download/v1.7.1
    files:
      - fileName: jq-linux-arm64
      - fileName: jq-linux-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/cloudflare/cfssl/releases/download/v1.6.4
    files:
      - fileName: cfssl-certinfo_1.6.4_linux_arm64
      - fileName: cfssl-certinfo_1.6.4_linux_amd64
      - fileName: cfssl_1.6.4_linux_arm64
      - fileName: cfssl_1.6.4_linux_amd64
      - fileName: cfssljson_1.6.4_linux_arm64
      - fileName: cfssljson_1.6.4_linux_amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/opencontainers/runc/releases/download/v1.1.12/
    files:
      - fileName: runc-arm64
      - fileName: runc-amd64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/etcd-io/etcd/releases/download/v3.5.6/
    files:
      - fileName: etcdctl-v3.5.6-linux-amd64
      - fileName: etcdctl-v3.5.6-linux-arm64
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/rpm/releases/download/v0.0.1/
    files:
      - fileName: rpm.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/nfs/
    files:
      - fileName: nfsshare.tar.gz
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/ #非 latest 版本请以对应 Core-VersionConfig 中 files.address 为准，例如 v26.06-rc.1 为 https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/1.2.3/
    files:
      - fileName: bkeadm_linux_amd64
      - fileName: bkeadm_linux_arm64
patches:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/
    files:
      - fileName: VersionConfig-latest.yaml #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为Core-VersionConfig-v26.06-rc.1.yaml
charts:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/ #非 latest 版本请以对应 Core-VersionConfig 中的 charts.address 为准，例如 v26.06-rc.1 为 https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/
    files:
      - fileName: oauth-webhook-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 oauth-webhook-1.0.3.tgz
      - fileName: oauth-server-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 oauth-server-1.0.3.tgz
      - fileName: plugin-management-service-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 plugin-management-service-1.0.2.tgz
      - fileName: user-management-operator-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 user-management-operator-1.0.2.tgz
      - fileName: installer-service-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 installer-service-1.2.3.tgz
      - fileName: installer-website-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 installer-website-1.2.2.tgz
      - fileName: bke-console-website-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 bke-console-website-1.0.3.tgz
      - fileName: bke-console-service-0.0.0-latest.tgz # 非 latest 版本请以对应 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 bke-console-service-1.0.2.tgz
```

下面是openFuyao-system-controller增量组件的镜像下载配置yaml文件。
```yaml
registry:
  imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
repos:
  - architecture: # openFuyao组件
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao # 核心组件
        targetRepo: kubernetes
        images:
          # 非 latest 版本请以对应 Core-VersionConfig 文件中该组件的实际 tag 为准
          - name: application-management-service
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.0.2
          - name: bkeagent-launcher
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.3.0
          - name: console-service
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.1.0
          - name: console-website
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.1.0
          - name: marketplace-service
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.0.3
          - name: monitoring-service
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.0.4
          - name: openfuyao-system-controller
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.1.3
          - name: web-terminal-service
            tag:
              - latest #若为其他版本修改为其他版本，e.g.:若为v26.06-rc.1版本则修改为1.0.3
      - sourceRepo: hub.oepkgs.net/openfuyao/brancz # brancz 组件
        targetRepo: kubernetes/brancz
        images:
          - name: kube-rbac-proxy
            tag:
              - v0.18.0
      - sourceRepo: cr.openfuyao.cn/openfuyao/harbor # harbor 组件
        targetRepo: kubernetes/harbor
        images:
          - name: chartmuseum-photon
            tag:
              - v2.7.0
          - name: harbor-core
            tag:
              - v2.7.0
          - name: harbor-portal
            tag:
              - v2.7.0
          - name: harbor-registryctl
            tag:
              - v2.7.0
          - name: harbor-db
            tag:
              - v2.7.0
          - name: harbor-jobservice
            tag:
              - v2.7.0
          - name: nginx-photon
            tag:
              - v2.7.0
          - name: redis-photon
            tag:
              - v2.7.0
          - name: registry-photon
            tag:
              - v2.7.0
          - name: redis
            tag:
              - v7.4.2
      - sourceRepo: hub.oepkgs.net/openfuyao/jimmidyson # jimmidyson 组件
        targetRepo: kubernetes/jimmidyson
        images:
          - name: configmap-reload
            tag:
              - v0.13.0
      - sourceRepo: hub.oepkgs.net/openfuyao/kube-state-metrics # kube-state-metrics 组件
        targetRepo: kubernetes/kube-state-metrics
        images:
          - name: kube-state-metrics
            tag:
              - v2.12.0
      - sourceRepo: hub.oepkgs.net/openfuyao/metrics-server # metrics-server 组件
        targetRepo: kubernetes/metrics-server
        images:
          - name: metrics-server
            tag:
              - v0.7.1
      - sourceRepo: hub.oepkgs.net/openfuyao/prometheus-operator # prometheus-operator 组件
        targetRepo: kubernetes/prometheus-operator
        images:
          - name: prometheus-config-reloader
            tag:
              - v0.74.0
          - name: prometheus-operator
            tag:
              - v0.74.0
      - sourceRepo: hub.oepkgs.net/openfuyao/prometheus  # prometheus 组件
        targetRepo: kubernetes/prometheus
        images:
          - name: alertmanager
            tag:
              - v0.27.0
          - name: blackbox-exporter
            tag:
              - v0.25.0
          - name: node-exporter
            tag:
              - v1.8.1
          - name: prometheus
            tag:
              - v2.52.0
rpms:
debs: []
files: []
patches: []
charts:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/charts/releases/download/latest/ #非 latest 版本请以对应 Core-VersionConfig 文件中的 charts.address 为准
    files:
      - fileName: console-website-0.0.0-latest.tgz #非 latest 版本请以对应 Core-VersionConfig 文件中的 charts.files 为准，例如 v26.06-rc.1 为 console-website-1.1.0.tgz
      - fileName: monitoring-service-0.0.0-latest.tgz # 非 latest 版本请以 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 monitoring-service-1.0.4.tgz
      - fileName: console-service-0.0.0-latest.tgz # 非 latest 版本请以 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 console-service-1.1.0.tgz
      - fileName: marketplace-service-0.0.0-latest.tgz # 非 latest 版本请以 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 marketplace-service-1.0.3.tgz
      - fileName: application-management-service-0.0.0-latest.tgz # 非 latest 版本请以 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 application-management-service-1.0.2.tgz
      - fileName: web-terminal-service-0.0.0-latest.tgz # 非 latest 版本请以 Core-VersionConfig 中的实际 chart 文件名为准，例如 v26.06-rc.1 为 web-terminal-service-1.0.3.tgz
      - fileName: harbor-1.11.4.tgz

```
