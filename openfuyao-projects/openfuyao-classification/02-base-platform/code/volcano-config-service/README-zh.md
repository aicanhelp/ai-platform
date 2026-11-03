# volcano-config-service

## 特性简介

在现代高性能计算和大规模分布式系统中，非统一内存访问（NUMA）架构越来越普遍。NUMA架构通过将内存划分到不同的节点（NUMA节点），每个节点有其自己的局部内存和CPU，旨在减少内存访问延迟，提高系统性能。然而，NUMA架构的复杂性增加了系统资源管理的难度，特别是在多任务、多线程环境中。为了充分利用NUMA架构的优势，需要对系统资源进行精细化管理和监控。NUMA资源监控可视化旨在通过直观的图形界面，实时展示系统中NUMA资源的分配和使用情况，帮助用户更好地理解和管理NUMA资源，从而提升系统性能和资源利用率。在容器集群中，丰富多样的调度器提供了资源调度能力。本特性将分为三个角度，提供针对优先级高、中、低三种不同类型的任务，不同种类的优化方式，在集群调度以及节点内NUMA调度的层面上，保证Pod达到最优的分配效果，在性能提升上达到了显著效果。详情请参见[《NUMA亲和调度》](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/zh/user_guide/numa-aware_scheduling_user_guide.md#特性介绍)。

### 能力范围

本特性分为volcano-config-service和volcano-config-website两个代码仓，分别包含后端以及前端功能。

- NUMA拓扑图：直观展示系统中各NUMA节点的拓扑结构，包括每个节点的CPU和内存分布。显示各节点之间的连接关系和访问延迟。
- 实时资源监控：实时显示各NUMA节点的CPU使用情况，包括使用率和空闲状态。实时显示各NUMA节点的内存使用情况，包括总内存、已用内存和空闲内存。
- 详细资源信息：显示每个NUMA节点的详细资源信息，如CPU列表、内存大小。支持查看每个容器在NUMA节点上的CPU和内存分配情况。

### 约束与限制

- 使用该模块时，需要修改Kubelet的Config文件，具体配置步骤如下所示，此步骤会在开启NUMA亲和策略或最优NUMA Distance策略时自动完成。

  1. 打开Kubelet配置文件。

      ```shell
      vi /var/lib/kubelet/config.yaml
      ```

  2. 新增或修改配置项。

      ```shell
      cpuManagerPolicy:static
      systemReserved:
          cpu: "0.5"
      kubeReserved:
          cpu: "0.5"
      topologyManagerPolicy:xxx
      ```

  3. 修改应用。

      ```shell
      rm -rf /var/lib/kubelet/cpu_manager_state
      systemctl daemon-reload
      systemctl restart kubelet
      ```

  4. 查看Kubelet运行状态。

      ```shell
      systemctl status kubelet
      ```

  Kubelet运行状态为running，则表示成功。修改节点的拓扑策略时会重启Kubelet，可能会导致部分Pod重新调度，现网环境请谨慎使用。

- 使用运行时Pod亲和优化功能时，需要提前修改系统配置，具体步骤如下。

  1. 修改操作系统cmdline参数。

     ```shell
     vim /etc/grub2-efi.cfg
     ```

     在当前操作系统镜像对应的linux参数后添加容器运行时：

     ```shell
     mem_sampling_on numa_icon=enable
     ```

     例如：

     ```shell
     linux   /vmlinuz-5.10.0-216.0.0.115.oe2203sp4.aarch64 root=/dev/mapper/openeuler-root ro rd.lvm.lv=openeuler/root rd.lvm.lv=openeuler/swap video=VGA-1:640x480-32@60me cgroup_disable=files apparmor=0 crashkernel=1024M,high smmu.bypassdev=0x1000:0x17 smmu.bypassdev=0x1000:0x15 arm64.nopauth console=tty0 kpti=off mem_sampling_on numa_icon=enable
     ```

  2. 修改containerd的配置文件。

     ```shell
     vi /etc/containerd/config.toml
     ```

     将其中的内容修改：

     ```shell
     [plugins."io.containerd.nri.v1.nri"]
     disable = false
     ```

     重启containerd以应用更新：

     ```shell
     systemctl restart containerd
     ```

说明：运行时Pod亲和优化功能目前只支持ARM环境+openEuler 22.03 LTS SP4环境使用。且优先级低于绑核，在集群中有绑核的Pod的情况下，自动关闭特性。

- 使用集群NUMA监控功能时，需要提前配置普罗米修斯。
  1. 配置Prometheus Operator接入NUMA Exporter参考任务场景2。
  2. 配置Prometheu接入NUMA Exporter参考任务场景3。

## 安全能力说明

openFuyao v26.06 及往期版本仅提供基础安全功能：服务间双向认证（mTLS）与组件级用户认证鉴权（RBAC）需使用者/集成伙伴根据自身部署环境自行适配与加固（如引入 cert-manager、做统一证书、部署鉴权中间件、配置 NetworkPolicy 收缩可达范围等）。当前聚焦基础功能交付，安全防护需用户自行加强，后续版本安全能力会逐步规划落地。

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

  其中，`<path/to/dockerfile>`为Dockerfile路径（build/Dockerfile），`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

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

  其中，`<path/to/dockerfile>`为Dockerfile路径（build/Dockerfile），`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

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

  其中，`<path/to/dockerfile>`为Dockerfile路径（build/Dockerfile），`path/to/output`为本地目录路径。

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


## 安装

### 前提条件

- 已部署Kubernetes v1.21及以上。
- 已部署Prometheus。
- 已部署Containerd且开启NRI功能。

### 开始安装

#### openFuyao平台

1. 在openFuyao平台的左侧导航栏选择“应用市场 > 应用列表”，进入“应用列表”界面。
2. 勾选左侧类型“扩展组件”，查看所有扩展组件。或在搜索框中输入“volcano”。
3. 单击“volcano-config-service”卡片，进入调度扩展组件“详情”界面。
4. 单击“部署”进入“部署”界面。
5. 输入应用名称、选择安装版本和命名空间。
6. 在参数配置的“Values.yaml”中输入要部署的values信息。
7. 单击“部署”完成部署。
8. 在左侧导航栏单击“扩展组件管理”，管理调度组件。

#### 独立部署

相比于应用市场安装部署，本组件提供了独立部署功能，步骤如下：

1. 拉取镜像。

   ```shell
   helm pull oci://cr.openfuyao.cn/charts/volcano-config-service --version xxx
   ```

   将其中的xxx替换成需要拉取的helm镜像版本，例如：0.0.0-latest。

2. 解压安装包。

   ```shell
   tar -zxvf volcano-config-service-xxx.tgz
   ```

3. 关闭openFuyao及Oauth开关。

   ```shell
   vim volcano-config-service/charts/volcano-config-website/values.yaml
   ```

   将enableOAuth以及openFuyao选项改为false。

4. 独立安装。

   ```shell
   helm install volcano-config-service ./
   ```

5. 对接普罗。

   参考NUMA亲和调度开发指南[任务场景2](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/zh/developer_guide/numa-aware_scheduling_development_guide.md#任务场景2配置prometheus-operator接入numa-exporter)及[任务场景3](https://gitcode.com/openFuyao/sig-numa-affinity/blob/master/docs/zh/developer_guide/numa-aware_scheduling_development_guide.md#任务场景3配置prometheu接入numa-exporter)。

6. 访问独立前端。

   可通过http://*管理面的客户端登录IP地址*:30881 访问独立前端。
