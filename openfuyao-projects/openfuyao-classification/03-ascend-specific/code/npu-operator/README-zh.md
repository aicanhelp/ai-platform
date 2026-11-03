# NPU Operator

## 特性介绍
Kubernetes通过设备插件提供对特殊硬件资源（如昇腾NPU）的访问。但是配置和管理具有这些硬件资源的节点需要配置多个软件组件（如驱动程序、容器运行时或其他库），这些组件安装复杂困难且容易出错。NPU Operator使用Kubernetes中的Operator Framework来自动管理配置昇腾设备所需的所有软件组件。这些组件包括昇腾驱动程序和固件，使能构建集群全流程运行，支持集群作业调度、运维监控、故障恢复等功能的MindCluster设备插件。通过安装相应的组件，可以实现NPU资源管理、工作负载的优化调度、训推任务的容器化支持，使能AI作业以容器形式部署并运行在NPU设备上。

## 文档

- [架构总览](docs/design/architecture.md)
- [CRD 字段参考](docs/design/crd-reference.md)
- [Reconcile 逻辑](docs/design/reconcile.md)
- [文档索引与维护规范](docs/README.md)
- [变更日志](CHANGELOG.md)
- [贡献指引](CONTRIBUTING.md)

## 安装部署

### 在线安装

#### 前提条件

- 当前计算机上有kubectl和Helm CLI，或集群中有可配置的应用商店或仓库。

- 请确认环境中含有bash工具，否则可能导致驱动固件安装脚本解析失败。

- Kubernetes集群中运行NPU工作负载的所有工作节点或节点组运行的操作系统版本都需满足openEuler 22.03 LTS或者Ubuntu 22.04（ARM架构）。

- 对于仅运行CPU工作负载的工作节点或节点组，节点可以运行任何操作系统，因为NPU Operator不会对非NPU工作负载的节点执行任何配置或管理。

- 当前NPU Operator所安装的组件需要运行环境满足的NPU芯片型号为910B和310P。具体OS和硬件配套关系请参考[MindCluster文档](https://www.hiascend.com/document/detail/zh/mindcluster/2600/clustersched/dlug/docs/zh/scheduling/introduction/03_supported_product_models_and_os.md)。

#### 二进制安装

1.拉取项目安装包。

```bash
helm pull oci://cr.openfuyao.cn/charts/npu-operator xxx
```

其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。拉取到的安装包为压缩包形式。

2.安装部署。

以命名空间`default`、release名称`npu-operator`为例，在安装包所在的目录下，可直接通过以下命令进行安装。

```bash
helm install npu-operator npu-operator-xxx.tgz -n default
```

#### 源码安装

1.从仓库拉取项目。

```bash
git clone -b xxx https://gitcode.com/openfuyao/npu-operator.git
```

其中`xxx`指的是代码的分支。

2.安装部署。

以命名空间`default`、release名称`npu-operator`为例，在`npu-operator`同级目录下执行如下命令。

```bash
cd npu-operator/charts/npu-operator
helm install -n default npu-operator .
```

#### 参数配置

详细的参数配置说明请参考npu-operator用户手册的[常见定制选项](https://gitcode.com/openFuyao/docs/blob/master/docs/zh/user_guide/npu_operator.md#%E5%B8%B8%E8%A7%81%E5%AE%9A%E5%88%B6%E9%80%89%E9%A1%B9)章节。

### 离线安装

#### 前提条件

- 当前计算机上有kubectl和Helm CLI，或集群中有可配置的应用商店或仓库。

- 请确认环境中含有bash工具，否则可能导致驱动固件安装脚本解析失败。

- Kubernetes集群中运行NPU工作负载的所有工作节点或节点组运行的操作系统版本都需满足openEuler 22.03 LTS或者Ubuntu 22.04（ARM架构）。

- 对于仅运行CPU工作负载的工作节点或节点组，节点可以运行任何操作系统，因为NPU Operator不会对非NPU工作负载的节点执行任何配置或管理。

- 当前NPU Operator所安装的组件需要运行环境满足的NPU芯片型号为910B和310P。具体OS和硬件配套关系请参考[MindCluster文档](https://www.hiascend.com/document/detail/zh/mindcluster/2600/clustersched/dlug/docs/zh/scheduling/introduction/03_supported_product_models_and_os.md)。

- 下载离线镜像，将需要安装的组件所用到的全部镜像下载到本地。详情可见表1。

- 准备驱动固件zip包和MindIO组件的zip包：
  - **下载驱动固件zip包**：可前往npu-driver-installer的[代码仓](https://gitcode.com/openFuyao/npu-driver-installer/blob/master/downloader/NPU/25.3.RC1/config.json)，找到对应驱动固件版本号的config.json文件，按照对应节点的NPU型号、OS架构单击相应链接来下载对应的驱动固件压缩包；910C型号NPU支持离线驱动固件安装，需要从[昇腾社区-固件与驱动下载](https://www.hiascend.com/hardware/firmware-drivers/community?product=1&model=30&cann=8.5.0.alpha002&driver=Ascend+HDK+25.3.RC1)自行下载，其余型号的驱动固件zip包也可前往上述链接进行下载。
  - **下载MindIO组件zip包**：可前往npu-node-provision的[代码仓](https://gitcode.com/openFuyao/npu-node-provision/blob/master/downloader/software/6.0.0/config.json)，找到组件对应版本号的config.json文件，按照对应节点的NPU型号、OS架构，单击相应链接来下载对应的SDK压缩包。

- 将驱动固件包的zip文件放到需要进行离线安装的节点路径下：`/tmp/driver_pkg/`，该路径可在values.yaml的`driver.env`字段中自定义修改。

- 将MindIO组件的zip包放到需要进行离线安装的节点路径下：`/opt/openFuyao/mindio/`

- 检查待安装节点中是否含有以下工具：
  - 如果使用的包管理工具为yum，则需要安装的pkg为：**"jq wget unzip which initscripts coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms"**<br/>
    可使用如下命令进行检测。
    ```bash
    pkgs=(jq wget unzip which initscripts coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms); rpm -q "${pkgs[@]}" >/dev/null || rpm -q "${pkgs[@]}" | grep "is not installed"
    ```
  - 如果使用的包管理工具为apt-get，需要安装的pkg为：**"jq wget unzip debianutils coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch dkms linux-headers-$(uname -r)"**<br/>
    可使用如下命令进行检测。
    ```bash
    pkgs=(jq wget unzip debianutils coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch dkms linux-headers-$(uname -r)); dpkg-query -W -f='${Package}\t${Status}\n' "${pkgs[@]}" 2>&1 | grep -Ev "install ok installed"
    ```

  - 如果使用的包管理工具为dnf，则需要安装的pkg为：**"jq wget unzip which initscripts coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms"**<br/>
    可使用如下命令进行检测。
    ```bash
    pkgs=(jq wget unzip which initscripts coreutils findutils gawk e2fsprogs util-linux net-tools pciutils gcc make automake autoconf libtool git patch kernel-devel-$(uname -r) kernel-headers-$(uname -r) dkms); rpm -q "${pkgs[@]}" >/dev/null || rpm -q "${pkgs[@]}" | grep "is not installed"
    ```

**表1** 组件涉及到的镜像列表

| 组件           | 镜像                  |
| -------------- | -------------------- |
| 昇腾驱动和固件        | cr.openfuyao.cn/openfuyao/npu-driver-installer:latest        |
| ascend-docker-runtime        | cr.openfuyao.cn/openfuyao/ascend-docker-runtime:v7.3.0<br/>cr.openfuyao.cn/openfuyao/npu-container-toolkit:latest      |
| clusterd        | hub.oepkgs.net/openfuyao/ascendhub/clusterd:v7.3.0        |
| device-plugin        | hub.oepkgs.net/openfuyao/ascendhub/ascend-k8sdeviceplugin:v7.3.0<br/>hub.oepkgs.net/openfuyao/busybox:1.36.1        |
| mindio        | cr.openfuyao.cn/openfuyao/npu-node-provision:latest        |
| noded        | hub.oepkgs.net/openfuyao/ascendhub/noded:v7.3.0        |
| npu-exporter        | hub.oepkgs.net/openfuyao/ascendhub/npu-exporter:v7.3.0        |
| resilience-controller        | hub.oepkgs.net/openfuyao/ascendhub/resilience-controller:v7.1.RC1        |
| ascend-operator        | hub.oepkgs.net/openfuyao/ascendhub/ascend-operator:v7.3.0<br/>hub.oepkgs.net/openfuyao/busybox:1.36.1        |
| volcano        | hub.oepkgs.net/openfuyao/ascendhub/vc-controller-manager:v1.9.0-v7.3.0<br/>hub.oepkgs.net/openfuyao/ascendhub/vc-scheduler:v1.9.0-v7.3.0<br/>hub.oepkgs.net/openfuyao/busybox:1.36.1        |
| npu-operator        | cr.openfuyao.cn/openfuyao/npu-operator:latest        |
| node-feature-discovery       | hub.oepkgs.net/openfuyao/nfd/node-feature-discovery:v0.16.4        |
| npu-feature-discovery       | cr.openfuyao.cn/openfuyao/npu-feature-discovery:latest        |

其中镜像`tag`可根据组件不同版本自定义修改。

#### 二进制安装

1.拉取项目安装包。

```bash
helm pull oci://cr.openfuyao.cn/charts/npu-operator xxx
```

其中`xxx`需替换为具体项目安装包版本，如`0.0.0-latest`。拉取到的安装包为压缩包形式。

2.安装部署。

以命名空间`default`、release名称`npu-operator`为例，在安装包所在的目录下，可直接通过以下命令进行安装。

```bash
helm install npu-operator npu-operator-xxx.tgz -n default
```

#### 源码安装

1.从仓库拉取项目。

```bash
git clone -b xxx https://gitcode.com/openfuyao/npu-operator.git
```

其中`xxx`指的是gitcode仓库源代码的分支。

2.安装部署。

以命名空间`default`、release名称`npu-operator`为例，在`npu-operator`同级目录下执行如下命令。

```bash
cd npu-operator/charts/npu-operator
helm install -n default npu-operator .
```

根据集群中提前准备的离线镜像对应的`tag`，同步在npu-operator/charts/npu-operator/values.yaml中，对组件镜像进行修改，以确保组件的镜像与已导入集群的离线镜像相同。

## 本地构建

### 镜像构建

#### 构建参数

- `GOPRIVATE`：配置Go语言私有仓库，相当于`GOPRIVATE`环境变量。
- `COMMIT`：当前git commit的哈希值。
- `VERSION`：组件版本。
- `SOURCE_DATE_EPOCH`：镜像rootfs的时间戳。

#### 构建命令

- 构建并推送到指定OCI仓库。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

- 构建并导出OCI Layout到本地tarball。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

- 构建并导出镜像rootfs到本地目录。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径，`path/to/output`为本地目录路径。

### Helm Chart构建
- 打包Helm Chart。

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  其中，`<path/to/chart>`为Chart文件夹路径。

- 推送Chart包到指定OCI仓库。

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  其中，`<path/to/chart.tgz>`为Chart包路径，`<oci/repository>`为Chart包推送地址，`<tag>`为Chart包tag。

### NPU Exporter 的 ServiceMonitor

普通 exporter 的 ServiceMonitor 默认关闭，独立于 exporter 工作负载。需要 Prometheus Operator 采集时，在 Helm values 中配置：

```yaml
exporter:
  enabled: true
  serviceMonitor:
    enabled: true
```

直接修改 CR 时对应 `spec.exporter.managed: true` 和 `spec.exporter.serviceMonitor.enabled: true`。开启前需安装 `monitoring.coreos.com/v1` ServiceMonitor CRD，并创建 `monitoring` 命名空间；Prometheus 还需选择该 ServiceMonitor。

关闭开关或停管 exporter 会清理属于当前 CR 的 ServiceMonitor，保留其他部署拥有的资源。升级前已使用该采集配置的环境需显式开启；新增字段需同步更新 NPUClusterPolicy CRD，不能只更换 Operator 镜像。
