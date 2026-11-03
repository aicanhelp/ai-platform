# 介绍

bkeadm是一款基于Kubernetes的容器编排工具，旨在为用户提供便捷、高效的容器编排和应用部署能力。

## 镜像构建

### 构建参数

- `GOPRIVATE`：配置Go语言私有仓库，相当于`GOPRIVATE`环境变量。
- `COMMIT`：当前git commit的哈希值。
- `VERSION`：组件版本。
- `SOURCE_DATE_EPOCH`：镜像rootfs的时间戳。

### 构建命令

构建或发布前必须先执行 `make test-preflight`。该门禁校验内置 `preflight.yaml`
策略可加载、可校验并能编译出运行时 Bundle；校验失败时不得继续执行下列
`docker buildx`/`nerdctl build` 命令。

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

## 安装集群

### 在线安装

1. 下载并自动安装bkeadm。

    ```shell
    # 方式1：快捷下载
    curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh | bash
    ```
    ```shell
   # 方式2：校验下载文件的完整下载
   ## 下载download.sh脚本文件
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh
   ## 下载download.sh文件的校验文件并进行校验（可选），校验成功会输出-: OK，校验失败就需要联系openFuyao社区维护人员定位原因
   curl -LO https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/latest/download.sh.sha256
   sha256sum -c <(cat download.sh.sha256) < download.sh
   ## 运行download.sh文件下载bke安装工具，执行过程中会校验安装工具的sha256sum
   chmod +x download.sh && ./download.sh
    ```

2. 初始化引导节点。

    - 该功能会在引导节点上部署一个轻量级的K3s集群，集群会部署cluster-api、provider-bke以及openFuyao安装部署相关的Pod。

    ```shell
    bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest
    ```

### 离线安装

1. 构建离线安装部署包，参考构建部署包。

2. 将部署包复制到离线环境引导节点。

3. 解压部署包到根目录。

    ```shell
    rm -rf /bke && tar zxvf <部署包名字 eg: bke.tar.gz> -C /
    ```

    > 要求解压后根目录存储空间大于29GB，否则不能初始化成功。

4. 修改bke安装工具名字并初始化引导节点。

    - 要求引导节点干净，未提前安装docker、containerd等组件。

    ```shell
    # 修改安装工具名字
    ARCH=$(uname -m)
    case $ARCH in
    x86_64) ARCH="amd64";;
    aarch64) ARCH="arm64";;
    esac
    mv /usr/local/bin/bkeadm_linux_$ARCH /usr/local/bin/bke
   
    # 初始化引导节点
    bke init 
    ```

## 构建部署包

- 要求构建机器已安装tar、pigz工具与bkeadm。
- 要求构建在线部署依赖环境提前安装好docker与buildx，可参考[docker官方文档](https://docs.docker.com/engine/install/)安装。  
- 要求构建离线部署制品环境提前安装好docker。 
- 完成docker安装后需要在docker配置文件中增加如下配置：
    - 编辑`/etc/docker/daemon.json`，在docker配置文件中增加以下配置。

    ```shell
    "insecure-registries": [
    "deploy.bocloud.k8s:40443",
    "0.0.0.0/0"
    ],
    ```

    - 修改后重启docker。

    ```shell
    systemctl restart docker
    ```

### 构建离线部署制品

- 收集二进制文件、rpm包、chart包等最后生成一个压缩包，offline-artifacts.yaml请使用[assets文件夹下offline-artifacts.yaml文件](./assets/offline-artifacts.yaml)。

    ```shell
    rm -rf /bke && bke build -f offline-artifacts.yaml -t bke.tar.gz
    ```

## 其他指令

- 镜像同步命令参考如下：

```shell
bke registry sync --dest-tls-verify --src-tls-verify  --multi-arch --source  deploy.bocloud.k8s:40443/openfuyao/cluster-api-provider-bke:latest    --target deploy.bocloud.k8s:40443/kubernetes/cluster-api-provider-bke:latest
```

## 历史版本声明

- v25.06版本将k3s集群使用的证书优化为部署过程中自动创建，对于v25.06之前的版本，需要进行证书和密钥的轮换，防止外部攻击。替换证书请参考[k3s certificate](https://docs.k3s.io/zh/cli/certificate)。
