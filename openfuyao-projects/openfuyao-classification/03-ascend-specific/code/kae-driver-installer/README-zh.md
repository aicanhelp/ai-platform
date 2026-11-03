# KAE Driver Installer

## 功能

KAE Driver Installer是专为鲲鹏加速引擎（KAE）设计的容器化驱动安装组件，负责在Kubernetes节点上自动安装、升级和维护KAE加速卡所需的驱动程序，确保硬件与操作系统内核兼容。具体包括：
- 检测节点内核版本和硬件型号，匹配对应的驱动版本。
- 处理驱动依赖（如内核模块、固件文件）。
- 驱动安装后的验证（如通过modprobe确认模块加载成功）。
- 支持驱动回滚（当新版本安装失败时自动恢复到上一版本）。

## 特性介绍

+ 属于KAE资源的底层驱动管理层，直接与宿主机交互。
+ 采用Job模式运行，确保一次性任务的可靠执行
+ 支持多种鲲鹏服务器型号的驱动适配

## 开始使用

### 前提条件

- 节点需搭载鲲鹏处理器（aarch64架构）并配备KAE加速器硬件。
- 节点已加入Kubernetes集群，且可被调度（无影响Pod运行的污点）。
- 节点内核版本需与目标 KAE 驱动版本兼容（可参考官方兼容性矩阵）。
- 集群已部署KAE Operator（推荐）或具备Job资源创建权限。

### 通过KAE Operator自动调用

1. 在KAE Operator的CR配置中指定驱动版本，Operator会自动触发kae-driver-installer：

    ```bash
    # kae-cluster.yaml
    apiVersion: kae.example.com/v1alpha1
    kind: KAECluster
    metadata:
    name: kae-cluster
    spec:
    driver:
        version: v1.3.0  # 指定驱动版本
        installTimeout: 300s
    nodeSelector:
        kae: enabled  # 仅在匹配标签的节点安装
    ```
2. 应用配置后，查看安装进度：

    ```bash
    # 查看驱动安装 Job
    kubectl get jobs -n <namesapce> -l app=kae-driver-installer

    # 查看安装日志
    kubectl logs <pod-name> -n <namesapce> -f
    ```

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

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

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

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

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
