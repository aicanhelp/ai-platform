# openfuyao-rayservice

Ray 是支持异构算力函数级按需调度的分布式计算框架，可在通算、智算以及通智融合场景提供高效、易用的多语言 API，提升算力占空比，优化算力综合利用率。Ray 支持以 RayCluster、RayJob、RayService 等多种作业形态，集群中往往会部署多个 Ray 集群分别处理不同租户、应用的计算作业。本特性提供 Ray 集群及作业的管理能力，减少运维成本，同时提供集群可观测、故障定位、最佳实践等增强能力，实现云原生解决算力使能解决方案。

特性按照扩展组件形态设计，支持 openFuyao 平台集成部署，同时支持其他兼容平台的独立部署。

## 快速入门

### 先决条件

- Go 语言版本 v1.21.0+
- Docker 版本 17.03+
- kubectl 版本 v1.11.3+
- 能够访问 Kubernetes v1.11.3+ 集群

### 镜像构建

### 构建参数

- `GOPRIVATE`：配置Go语言私有仓库，相当于`GOPRIVATE`环境变量
- `COMMIT`：当前git commit的哈希值
- `VERSION`：组件版本
- `SOURCE_DATE_EPOCH`：镜像rootfs的时间戳

### 构建命令

- 构建并推送到指定OCI仓库

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

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag

- 构建并导出OCI Layout到本地tarball

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

  其中，`<path/to/dockerfile>`为Dockerfile路径，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径

- 构建并导出镜像rootfs到本地目录

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

  其中，`<path/to/dockerfile>`为Dockerfile路径，`path/to/output`为本地目录路径

### Helm Chart构建

- 打包Helm Chart

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  其中，`<path/to/chart>`为Chart文件夹路径

- 推送Chart包到指定OCI仓库

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  其中，`<path/to/chart.tgz>`为Chart包路径，`<oci/repository>`为Chart包推送地址，`<tag>`为Chart包tag

### 在集群上部署

**构建镜像并将其推送到 `IMG` 指定的位置：**

```sh
make docker-build docker-push IMG=<某个镜像仓库>/openfuyao-rayservice:标签
```

**注意：** 此镜像需要发布到您指定的个人镜像仓库中。工作环境必须能够拉取该镜像。如果上述命令不工作，请确保您拥有对该仓库的适当权限。

**将 CRD (自定义资源定义) 安装到集群：**

```sh
make install
```

**使用 `IMG` 指定的镜像将控制器部署到集群：**

```sh
make deploy IMG=<某个镜像仓库>/openfuyao-rayservice:标签
```

> **注意：** 如果您遇到 RBAC (基于角色的访问控制) 错误，您可能需要授予自己 `cluster-admin` 权限或使用管理员身份登录。

**创建解决方案的实例**
您可以从 `config/sample/` 目录应用示例配置：

```sh
kubectl apply -k config/samples/
```

>**注意：** 请确保示例配置包含用于测试的默认值。

### 卸载

**从集群中删除实例 (CRs)：**

```sh
kubectl delete -k config/samples/
```

**从集群中删除 API (CRDs)：**

```sh
make uninstall
```

**从集群中卸载控制器：**

```sh
make undeploy
```

## 项目分发

以下是构建安装程序并将此项目分发给用户的步骤：

1. **构建安装程序：** 为已发布到镜像仓库的镜像构建安装程序

```sh
make build-installer IMG=<某个镜像仓库>/openfuyao-rayservice:标签
```

**注意：** 上述 Makefile 目标会在 `dist` 目录下生成 `install.yaml` 文件。该文件包含使用 Kustomize 构建的所有必要资源，用于安装此项目（不包含依赖项）。

2. **使用安装程序：** 用户只需运行以下命令即可安装项目

```sh
kubectl apply -f https://raw.githubusercontent.com/<组织>/openfuyao-rayservice/<标签或分支>/dist/install.yaml
```

**注意：** 运行 `make help` 可查看所有可能的 `make` 目标的更多信息。

更多信息请参考 [Kubebuilder 文档](https://book.kubebuilder.io/introduction.html)

## 许可证

版权所有 © 2025。

根据 Apache 许可证 2.0 版（"许可证"）授权；
除非遵守许可证，否则您不得使用此文件。
您可以在以下网址获取许可证副本：

    http://www.apache.org/licenses/LICENSE-2.0

除非适用法律要求或书面同意，根据许可证分发的软件是基于"按原样"基础分发的，
没有任何明示或暗示的担保或条件。
请参阅许可证以了解特定语言下的权限和限制。