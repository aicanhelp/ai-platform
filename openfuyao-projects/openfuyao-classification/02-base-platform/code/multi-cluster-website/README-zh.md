# multi-cluster-website

## 特性介绍

openFuyao多集群管理作为扩展组件旨在提供高效、灵活的Kubernetes集群跨环境管理，可以通过应用市场安装到openFuyao平台。可支持：

集群列表，提供集群列表界面查看和管理所有集群的状态和基本信息。
集群生命周期管理，简化集群的扩展和销毁过程，支持用户纳管新集群，调整现有集群的规模，更新集群配置，或安全的取消纳管不再需要的集群。
查看集群凭证：支持用户使用集群凭证进行集群的快速访问。
集群标签管理，支持用户快速编辑集群标签等操作。
服务跨集群访问，提供跨集群访问等功能，支持用户在一个主集群入口访问纳管集群上的服务。

本项目是多集群组件的前端实现。

所属sig: [sig-container-platform](https://gitcode.com/openFuyao/sig-container-platform)

## 本地构建

### 镜像构建

#### 构建参数

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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

- 构建并导出OCI Layout到本地tarball。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

- 构建并导出镜像rootfs到本地目录。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`path/to/output`为本地目录路径。

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

## 安装说明

### 前提条件

多集群管理组件作为扩展组件安装至openFuyao平台，对硬件、软件以及网络要求与openFuyao平台整体安装要求一致。

### 开始安装

1. 登录openFuyao平台管理面。
2. 在openFuyao平台左侧导航栏选择“应用市场 > 应用列表”，进入“应用市场”界面。
3. 勾选左侧应用类型“扩展组件”，查看所有扩展组件。或在“搜索框”中输入“multi-cluster-service”。
4. 单击“multi-cluster-service”卡片，进入扩展组件“详情”界面。
5. 单击“部署”按钮进入“部署”界面。
6. 输入“应用名称”、选择“安装版本”和“命名空间”。
7. 在参数配置的Values.yaml中输入要部署的values信息。
8. 单击“部署”按钮完成部署。
9. 在左侧导航栏单击“扩展组件管理”，管理多集群扩展组件。
> **注意：**
> - 在集群节点重启后，可能导致karmada服务不可用，请谨慎操作节点重启。
> - 若节点重启导致karmada服务不可用，请卸载多集群组件重新安装即可。
> - 当前多集群管理组件只能在openFuyao容器平台上作为扩展组件安装，暂不支持独立helm-chart部署。

## 使用说明

详细使用说明请参考[用户指南](https://docs.openfuyao.cn/zh/docs/v26.03/user_guide/multi_cluster_management.html)