# 介绍

installer是一个轻量级部署工具，用于快速部署Kubernetes集群、openFuyao管理面。

## 镜像构建

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

## 快速安装

### 前提条件

openFuyao容器平台解决方案支持在如下操作系统与架构上进行安装与使用：

| 操作系统     | 版本     | 架构        |
|--------------|--------|-----------|
| openEuler    | 22.03  | arm64、x86 |
| Ubuntu       | 22.04  | arm64、x86 |

> **说明：**<br />其他版本操作系统暂未测试，可能会出现不在预期内的问题。

对节点的资源配置要求如下：

| 配置规格                  | vCPU | 内存（GB） |
|-----------------------|------|----|
| 建议配置                  | 4    | 16 |
| 最小配置（不安装fuyao-system）| 2    | 4  |

> **说明：**<br />
> - 当未启用并发多线程（SMT）或超线程时，一个vCPU相当于一个物理内核。启用后，使用以下公式换算：vCPU=(每个内核的线程数*内核数)*插槽数。

### 节点要求

- 节点可以联通外部网络。
- 节点可以使用root用户登录。

> **须知：**  
> 建议您的节点环境足够干净，未安装任何Kubernetes组件，否则可能会发生版本冲突导致安装失败。

### 安装带有openFuyao的集群

1. 使用root用户登录待安装节点。

2. 配置使用密钥登录。

>  **说明：**<br />
> 若已提前配置使用密钥登录，则跳过此步骤。

```shell
if [ ! -f "$HOME/.ssh/id_rsa" ] || [ ! -f "$HOME/.ssh/id_rsa.pub" ]; then
    ssh-keygen -t rsa-sha2-512 -P "" -f $HOME/.ssh/id_rsa && ls $HOME/.ssh;
fi

echo "" >> $HOME/.ssh/authorized_keys && \
           echo "$(cat $HOME/.ssh/id_rsa.pub)" >> $HOME/.ssh/authorized_keys && \
           awk ' !x[$0]++{print > "'$HOME'/.ssh/authorized_keys"}' $HOME/.ssh/authorized_keys
```

3. 下载安装包与安装脚本。

>  **说明：**<br />
> 若已提前下载安装包，将已下载的安装包上传至本机即可。

```
curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/latest/download.sh | bash -
```

执行上述命令默认下载最新版本的安装包，若想下载其他版本，请使用如下命令指定版本下载。

```
export VERSION=x.y.z

curl -sfL https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/installer/releases/download/latest/download.sh | bash -
```

4. 在安装包目录内执行安装。

```
./fuyao create cluster -A
```

5. 安装结果确认。

上一步自动化安装过程执行成功后，请记录openFuyao管理面的登录用户名和密码。<br />
此外可以查看机器上所有pod信息，当所有pod状态都为running后表示环境安装完毕。

```
kubectl get pod -A
```
>  **说明：**<br />
> 单节点的安装需要约20分钟，所有pod状态都为running状态则安装完毕。具体时间与实际网络情况有关，网速较慢时等待时间会比较长。

## 登录openFuyao管理面

浏览器中输入“https:// _管理面的登录IP地址_ : _openFuyao的web服务端口_ ”，输入用户名和密码即可登录openFuyao管理面。<br />
>  **说明：**<br />首次登录需要修改密码。<br />openFuyao的web服务端口安装完成后默认为31616。<br />


### 卸载

1、以root身份登录安装执行机  
2、在安装包目录内执行卸载

步骤一：进入安装包目录

```shell
cd <package-name, eg: fuyao-latest-openEuler-22.03-amd64>
```

步骤二：执行卸载

- 全量组件卸载

```shell
./fuyao destroy cluster -A
```

## 参与贡献

1.  Fork 本仓库
2.  新建 Feat_xxx 分支
3.  提交代码
4.  新建 Pull Request
