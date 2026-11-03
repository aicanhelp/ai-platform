# marketplace-service

## 特性介绍

openFuyao应用市场主要支持展示应用并管理应用仓库，为平台提供扩展能力和组件的便捷部署服务。

### 应用场景

openFuyao应用市场广泛用于用户和管理人员的仓库管理和应用部署操作，提供多种实际应用场景：

- 平台扩展能力：通过部署openFuyao扩展组件，显著增强平台的整体功能。
- 快速部署：用户可以直接将Helm包上传至openFuyao平台，实现快速部署和测试操作。

### 能力范围

- 应用部署：用户可以在应用市场中选择所需应用进行部署，并在后续管理中查看部署状态。
- 扩展能力部署：通过应用市场选择扩展组件，帮助平台增强功能和性能。
- 仓库管理：支持添加、删除和管理应用仓库，方便快速接入新的应用仓库，灵活选择可部署的应用。
- 本地仓库：支持多应用、多版本上传至本地，而无需将包上传至仓库中。

## 使用

openFuyao应用市场具体使用包括使用应用列表和使用包管理两大块，前者支持仓库查看、仓库同步、仓库添加等功能，后者支持管理包版本和添加包等功能。

### 前提条件

- 集群已安装Kubernetes 1.28以及Helm 3.14.2。
- 集群中marketplace-service运行正常。

### 查看仓库

仓库用于存储和管理应用及扩展组件，提供集中化的资源管理入口，通过“仓库详情”界面查看基本信息及仓库内包含的资源。

### 同步仓库

用于更新仓库中的应用或扩展组件信息，确保资源信息是最新的。

### 仓库添加

添加仓库功能用于将新的应用或扩展组件仓库接入openFuyao平台，以便openFuyao统一管理，同时也可以接入多种类型的资源。

### 管理包版本

包管理用于在openFuyao平台内对包进行版本管理，包括添加新版本或删除老版本。

### 添加包

用于在平台中上传新的应用包或组件包，为后续版本管理和资源部署提供支持。

## 本地构建

### 镜像构建

#### 构建命令

- 构建并推送到指定OCI仓库。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
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

## 安全能力说明

openFuyao v26.06 及往期版本仅提供基础安全功能：服务间双向认证（mTLS）与组件级用户认证鉴权（RBAC）的需使用者/集成伙伴根据自身部署环境自行适配与加固（如引入 cert-manager、做统一证书、部署鉴权中间件、配置 NetworkPolicy 收缩可达范围等）。当前聚焦基础功能交付，安全防护需用户自行加强，后续版本安全能力会逐步规划落地。