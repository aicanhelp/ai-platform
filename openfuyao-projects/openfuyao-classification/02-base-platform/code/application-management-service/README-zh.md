# application-management-service

## 特性介绍

应用管理是openFuyao平台提供的支持应用管理工具，提供了完整的部署、更新和卸载程序功能。当前版本支持的应用安装方式是通过Chart包使用Helm进行安装，在应用管理中可呈现整个应用列表并进行后续的运维管理操作。

### 应用场景

应用管理当前支持对于Helm格式的release应用进行完整的运维操作。

- 应用信息获取：以列表形式获取应用名称、状态、更新时间信息，并支持筛选与排序，同时详情页提供yaml、资源列表、事件、监控等更多信息。
- 应用生命周期管理：对集群中安装的各应用进行安装、卸载、升级与回退的操作。
- 支持各种形式的Helm Chart包上传后的安装，以及对于应用的相关设置参数进行修改。

### 能力范围

- 基础信息：支持展示应用名称、状态、创建时间、应用模板、模板版本和更新时间。应用列表页面支持按照名称和更新时间排序，支持根据安装状态进行筛选。
- 运维管理：支持应用的升级、回退和卸载。升级功能仅支持通过openFuyao应用市场安装的应用。支持应用的资源展示（如日志、事件等），配置修改（YAML形式）。

## 使用

在openFuyao平台的左侧导航栏单击“应用管理”，进入“应用管理”界面，可以查看已部署的应用实例，包含应用的名称、状态、更新时间。单击“应用名称”，进入“应用详情”界面，查看应用的详细信息，包括详情、YAML、资源、日志、事件、监控。

### 应用操作

#### 前提条件

- 集群已安装Kubernetes 1.28以上版本。
- 集群已安装Helm 3.14.2以上版本。
- 集群中marketplace-service运行正常。

#### 使用限制

待安装软件格式为Helm Chart，当前仅支持上传到仓库软件的安装。

#### 应用升级

1. 在“应用详情”界面单击“操作”，选择“升级”，进入“应用升级”界面。
2. 单击“版本信息”下拉框，选择要升级的指定版本。
3. 单击“确定”完成升级。

#### 应用卸载和回退

1. 应用卸载：支持卸载已安装的应用。
2. 应用回退：支持回退到历史安装的版本，以版本号和序列号作为标识。

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