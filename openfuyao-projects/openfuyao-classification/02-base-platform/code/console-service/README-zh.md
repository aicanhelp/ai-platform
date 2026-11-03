## 介绍

console-service是一个基于kubernetes，用于集群管理的服务。

## 服务依赖

 | 服务 | 版本   | 说明 |
|---  |------| --- |
| kubernetes | 1.28.8 | 集群管理 |
 | go | 1.21.8 | 编程语言|

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