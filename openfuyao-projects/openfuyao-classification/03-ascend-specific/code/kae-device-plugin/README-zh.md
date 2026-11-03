# 鲲鹏加速引擎（KAE）设备插件（Device Plugin）使用指南

## KAE 是什么

KAE是鲲鹏加速引擎（Kunpeng Accelerator Engine），是鲲鹏920、920+系列处理器中携带的能力。用户需要通过加载License来启用KAE
证书加载方式详见[安装前准备中证书载入](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/kae/usermanual/kunpengaccel_06_0008.html)。
KAE主要用于加解密加速和压缩解压缩加速，其提供了一套与openSSL完全兼容的接口，使得使用openSSL的应用能通过openssl引擎机制加载kae动态引擎从而将加解密操作卸载到硬件进行加速。

KAE架构说明请参见[KAE架构说明](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/kae/usermanual/kunpengaccel_16_0002.html)。

## KAE 设备插件（Device Plugin） 描述

KAE设备插件用于将KAE虚拟化设备暴露到集群层中，使得设备资源可以被k8s、openFuyao集群识别和管理；KAE设备插件基于k8s原生Device Plugin框架，实现gRPC Server用于设备插件与节点Kubelet的通信，从而实现设备的发现、注册和资源的分配调度。

KAE设备插件架构请见[KAE设备插件架构](https://gitee.com/openfuyao/enhancements/tree/master/design/feature/KAE%E9%B2%B2%E9%B9%8F%E5%8A%A0%E9%80%9F%E7%89%B9%E6%80%A7%E8%AF%B4%E6%98%8E%E4%B9%A6)

## 开始使用

在开始安装KAE设备插件前，请确认您的集群中**至少有一个**节点满足以下要求：

1. 物理机节点服务器携带鲲鹏920/920+系列芯片，且KAE许可证已加载；或虚机节点宿主机服务器携带鲲鹏920/920+系列芯片，且许可证已加载；在虚机节点的使用场景下，已完成设备虚拟化和绑定。

2. 节点已安装KAE驱动软件，安装指导请见[KAE驱动安装指导](https://www.hikunpeng.com/document/detail/zh/kunpengaccel/kae/usermanual/kunpengaccel_06_0012.html )

确认您的节点环境满足要求后，您即可通过应用市场部署此插件。

KAE设备插件以DaemonSet守护进程运行（节点标签：accelerator/kae.hpre），安装成功后，所有具有节点标签（accelerator/kae）的节点均会运行该守护进程的实例，从而进行设备资源发现、注册和调度。

### 在集群中使用KAE加速

当前KAE HPRE加解密支持OpenSSL 1.1.1.1x版本非对称加解密。针对集群中使用OpenSSL 1.1.1.1x版本加解密的服务可以做如下配置：

1. 编辑集群中需要使用KAE加速需要在工作负载如`nginx_deployment.yaml`：

   ```shell
   kubectl edit deployments nginx_deployment -n <namespace>
   ```

2. 添加`resources`字段，添加对KAE的资源请求：

   ```yaml
   resources:
     requests:
       openfuyao.com/kae.hpre: 1
   ```

3. 挂载openssl配置文件到容器镜像：

   ```yaml
     volumeMounts:
       - name: openssl-conf
         mountPath: /<container_path>/openssl.cnf
   volumes:
     - name: openssl-conf
       hostPath:
       path: /<host_path>/openssl.cnf
   ```

   `/<host_path>`为配置文件宿主机路径，`/<container_path>`为容器内路径。

   `openssl.cnf`文件内容为：

   ```text
   openssl_conf=openssl_def
   [openssl_def]
   engines=engine_section
   [engine_section]
   kae=kae_section
   [kae_section]
   engine_id=kae
   dynamic_path=/usr/local/lib/engines-1.1/kae.so
   KAE_CMD_ENABLE_ASYNC=1
   KAE_CMD_ENABLE_SM3=1
   KAE_CMD_ENABLE_SM4=1
   default_algorithms=ALL
   init=1
   ```

   亦可通过configMap方式将上述文件挂载到容器中。

4. 设置环境变量：

   ```yaml
   env:
     - name: OPENSSL_CONF
       value: /<container_path>/openssl.cnf
   ```

5. 保存编辑内容，待Pod重新调度后生效。

### 卸载、回退、升级KAE 设备插件

组件的卸载、升级、回退均可通过应用市场，或通过Helm命令行进行卸载、回退和升级。

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

## 许可证 License

[MulanPSL-2.0](http://license.coscl.org.cn/MulanPSL2)
