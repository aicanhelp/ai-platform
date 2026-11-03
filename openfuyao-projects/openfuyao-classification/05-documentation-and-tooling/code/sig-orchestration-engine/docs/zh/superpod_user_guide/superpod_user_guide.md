# SuperPod

## 特性介绍

SuperPod（超节点）是基于UBS（Unified Bus System）拓扑发现机制构建的Kubernetes集群拓扑抽象实体，将物理节点按`superPodId`聚合成逻辑拓扑单元，并以自定义资源（`SuperPod` CR）的形式纳管到集群中。控制器周期性采集节点拓扑与内存信息，自动维护`SuperPod` CR的生命周期，并可选联动生成Volcano `HyperNode` CR，使能Volcano调度器的网络拓扑感知调度能力。同时提供独立`superpod-exporter`组件以Prometheus格式暴露SuperPod成员关系、内存借用、共享内存、URMA设备等指标，支撑监控平台对SuperPod资源的可视化与运维。

当前版本仅支持**FM（Full-Mesh）组网形态**：单个SuperPod内的物理节点间全互联（两两直连，跳数均为1），SuperPod间不互联（仅通过Eth互联）。

当用户在大规模集群中需要按物理拓扑边界（UB互联域）对节点进行逻辑分组、并希望Volcano调度器据此执行拓扑感知调度（如将通信密集型负载调度到同一SuperPod内）时，可使用SuperPod特性。

### 应用场景

- **大规模集群分层调度**：集群节点数量多、跨多个SuperPod分布时，借助SuperPod拓扑抽象支撑Volcano的网络拓扑感知调度，将通信密集型负载（如分布式训练）调度到同一SuperPod内，提升通信局部性。
- **UB节点拓扑感知**：UB互联场景下，将同一`superPodId`的节点归并，Volcano基于`HyperNode`选择通信最优的节点组合部署分布式训练或推理服务，避免跨SuperPod调度导致通信性能劣化。
- **内存池化拓扑纳管**：与容器内存借用特性配合，为远端内存借用提供拓扑边界，借用优先发生在同一SuperPod内。
- **运维侧拓扑覆盖**：运维人员通过为节点打`unifiedbus.com/superpod`标签即可覆盖节点归属的SuperPod，无需依赖底层metric上报。
- **SuperPod资源监控与运维**：集群管理员部署`superpod-exporter`后，可在监控平台查看各SuperPod信息、各SuperPod中NUMA内存借用信息、共享内存分布信息与URMA设备健康信息，无需登录节点执行CLI，便于及时发现资源倾斜与设备故障。

### 能力范围

- **架构支持**：支持操作系统openEuler 24.03 LTS SP3及以上版本，Kubernetes v1.31.1及以上版本，架构ARM64。
- **组网形态**：当前版本仅支持FM组网（节点全互联，顶层Tier=1，单Tier-1子组）。
- **拓扑抽象**：支持按`superPodId`聚合节点，自动生成`SuperPod`CR（cluster scope，group `matrix.openfuyao.cn`，version `v1`）。
- **superPodId双来源解析**：
  - 优先来源1：节点label `unifiedbus.com/superpod`（运维侧覆盖）。
  - 备选来源2：MatrixMetric拓扑metric中上报的`superPodId`字段。
- **内存纳管**：基于NUMA信息采集节点`total`/`used`内存，写入`SuperPod`的`spec.groups[].nodes[].memory`字段。
- **HyperNode可选联动**：通过环境变量`HYPERNODE_ENABLED`开关，可选生成Volcano `HyperNode` CR（`topology.volcano.sh/v1alpha1`），供Volcano调度器执行网络拓扑感知调度。
- **调谐机制**：MatrixMetric CR变更触发去抖调谐（5s去抖），同时周期全量resync（默认5min）；拓扑数据12h刷新一次，NUMA数据30s刷新。
- **指标采集与上报**：独立`superpod-exporter` DaemonSet组件以Prometheus格式暴露SuperPod成员关系、NUMA内存借用、共享内存提供、URMA设备信息与健康等指标，默认经`:9102/metrics`端点暴露，由监控平台抓取汇聚。
- **规格限制**：
  - 同一`superPodId`下所有节点被装入同一个Tier-1 group（`group-0`）。
  - `HyperNode`联动需集群已安装Volcano并注册`hypernodes` CRD。

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> kube-matrix-agent单实例可管理最多150个Pod、300个容器、300个进程，此为matrix-agent组件的通用规格限制，非SuperPod特性独有约束。

### 亮点特征

- **双来源superPodId**：节点label优先于metric，保证运维侧通过打标签即可覆盖拓扑归属，无需依赖底层metric上报，也无需重启组件。
- **声明式调谐**：MatrixMetric CR变更自动触发调谐，最终一致，无需手动干预。
- **可插拔HyperNode**：默认不产生`HyperNode` CR，仅当显式开启`HYPERNODE_ENABLED=true`且`HyperNode` CRD已注册时联动，避免对未安装Volcano的集群造成负担。
- **CRD就绪感知**：`SuperPod`/`HyperNode` CRD未就绪时自动退避重试，不崩溃、不影响既有控制器。
- **独立指标导出器**：`superpod-exporter`独立DaemonSet部署，单类指标采集失败不阻塞其他类别，`/metrics`始终可访问；SuperPod维度`superpod_name`label源自节点标签，当前即可承载SuperPod级汇聚。

### 基本概念

- **SuperPod**：超节点，由同一`superPodId`的物理节点聚合而成的逻辑拓扑单元，对应`SuperPod` CR（`matrix.openfuyao.cn/v1`，集群级）。
- **superPodId**：SuperPod标识，直接决定SuperPod命名（`superpod-<superPodId>`）；可通过节点label `unifiedbus.com/superpod`覆盖。
- **FM组网**：SuperPod内物理节点全互联（两两直连，跳数均为1）、SuperPod间不互联的组网形态，顶层Tier=1。
- **HyperNode**：Volcano定义的拓扑层级CR（`topology.volcano.sh/v1alpha1`），由本特性控制器在`HYPERNODE_ENABLED=true`时联动产出，供Volcano调度器执行网络拓扑感知调度。
- **Tier-1子组**：SuperPod内部按1跳连通分量划分的拓扑子组，FM组网下每个SuperPod仅含一个Tier-1子组（`group-0`），包含该SuperPod下全部节点。
- **superpod-exporter**：独立SuperPod指标导出组件，以DaemonSet形态部署于SuperPod每个物理节点，经Prometheus文本格式在`:9102/metrics`端点暴露指标，由监控平台抓取汇聚。

### 实现原理

#### 总体方案

总体思路：matrixagent采集单物理节点拓扑（含`superPodId`）通过MatrixMetric CR上报 → matrixcontroller按`superPodId`汇总组装`SuperPod`资源 → Volcano拓扑感知调度。

- 采集与上报复用既有matrixagent DaemonSet框架与MatrixMetric CR，新增`node_network_topology_info`指标项。
- 组装控制器内嵌既有matrixcontroller进程，与既有容器逃生告警控制器并行、互不干扰。
- `SuperPod`为本仓新增CRD（`matrix.openfuyao.cn/v1`，集群级）。
- 是否组装Volcano `HyperNode`资源由环境变量`HYPERNODE_ENABLED`控制，**默认`false`（不启用）**。关闭时Controller仅产出`SuperPod`（`hyperNodeRef`字段留空），不依赖Volcano与HyperNode CRD；启用时额外产出`HyperNode`并填充`SuperPod`中的`hyperNodeRef`引用。

#### 工作流程

SuperPod由matrixcontroller中的控制器协程负责装配和维护，整体工作流程如下：

1. **拓扑上报**：matrixagent采集本节点`superPodId`与邻居链路信息，将拓扑信息与NUMA内存信息写入MatrixMetric CR。
2. **事件触发**：MatrixMetric CR的增、改、删事件触发去抖调谐（5s去抖窗口）；同时每5min执行一次周期全量resync。
3. **调谐流程**：控制器执行一次完整reconcile：
   1. 校验`SuperPod` CRD已注册，未就绪则退避重试。
   2. 若开启`HYPERNODE_ENABLED=true`，校验`HyperNode` CRD已注册；未就绪则跳过HyperNode装配，仅产出`SuperPod`。
   3. 读取全部MatrixMetric CR与Node标签，解析每个节点的`superPodId`与内存信息。
   4. **superPodId解析**：节点label优先，metric字段次之；二者皆空则跳过该节点。
   5. 按`superPodId`分组所有节点，FM组网下同一`superPodId`的所有节点装入单个Tier-1 group（`group-0`）。
   6. 若开启HyperNode，每个`superPodId`额外生成一个Tier-1 `HyperNode` CR。
   7. 创建或更新各SuperPod/HyperNode CR，并清理已不存在的stale CR。

#### 指标采集机制

`superpod-exporter`以独立DaemonSet部署于SuperPod中（通过节点亲和性筛选标签包含`unifiedbus.com/superpod`的节点），经HTTP`/metrics`（默认`:9102/metrics`）暴露Prometheus指标，由集群监控平台抓取。指标为**节点级**：所有指标均携带`node`/`slot_id`label；URMA虽为SuperPod粒度数据，但每节点全量上报后通过PromQL `min by`聚合为SuperPod级视图。采集周期由Prometheus `scrape_interval`驱动（建议30s），导出器对低频数据（拓扑）做缓存（12h），高频数据（借用/设备）实时采集。

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> - `superpod_name`label取自本节点K8s Node标签`unifiedbus.com/superpod`，缓存12h。标签缺失时为`"unknown"`。
> - URMA设备为SuperPod粒度数据，采用每节点全量上报 + PromQL `min by (superpod_name, device_name)`聚合去重模式（故障优先，任一节点观测到故障即判故障）。

### 与相关特性的关系

- **依赖UBS Engine**：matrixagent上报的拓扑信息依赖底层ubs-engine及其拓扑发现组件，需预先安装，UBS Engine SDK socket（`/run/ubse`）需可用。
- **与容器内存借用特性共用组件**：SuperPod与容器内存借用特性共用matrixagent、matrixcontroller组件，部署流程一致（参见[安装](#安装)章节）。
- **可选联动Volcano HyperNode**：需集群已安装Volcano并注册`hypernodes` CRD（`topology.volcano.sh/v1alpha1`）。Volcano Scheduler需开启network-topology特性以消费`HyperNode`执行网络拓扑感知调度。未安装Volcano时保持`HYPERNODE_ENABLED=false`（默认值）即可，仅产生`SuperPod` CR。
- **与Prometheus/Grafana监控平台关系**：`superpod-exporter`以Prometheus文本格式暴露`ubs_*`指标，需集群已部署Prometheus（抓取方）与Grafana（可视化，可选）方可汇聚查看。不部署监控平台时导出器仍运行，但指标无人消费。`superpod-exporter`不创建任何CR，不影响调度与既有资源。

### 相关实例

业务Pod使用样例请参见本文档[配置样例](#配置样例)小节，包括SuperPod CR、HyperNode CR以及基于Volcano gang调度将业务Pod部署到同一SuperPod的完整示例。`superpod-exporter`指标输出样例与Grafana PromQL汇聚示例亦参见[配置样例](#配置样例)小节。

## 安装

### 前提条件

* **操作系统：** openEuler 24.03 LTS SP3或更高版本
* **CPU架构：** ARM64
* **内存：** 大于等于64GB
* **磁盘：** SSD，IOPS 500MB/s
* **芯片互联：** UB
* **用户权限：** 安装与管理需root权限
* **软件要求：**
    1. Kubernetes v1.31.1及以上版本。
    2. 参考[ubs-engine](https://gitcode.com/openeuler/ubs-engine)安装ubs-engine及其依赖组件，确保UBS Engine SDK socket（`/run/ubse`）可用。节点拓扑上报功能需ubs-engine版本不低于v0.1.7。
    3. 参考[Helm安装文档](https://helm.sh/docs/intro/install/)安装Helm。
    4. （可选）如需使用`superpod-exporter`指标采集与上报能力，需集群已部署Prometheus（抓取方）与Grafana（可视化，可选），且目标节点已配置`unifiedbus.com/superpod`标签。

### 开始安装

1. 构建指导。

    1.1 拉取源码。

    ```shell
    git clone -b master https://gitcode.com/openFuyao/ubs-k8s-enable.git
    ```

    1.2 安装依赖。

    构建前请确保宿主机已安装以下工具（版本要求如下）：

    ```shell
    docker  # 版本要求 > 20.10
    helm    # 版本要求 v3 及以上
    ```
    Dockerfile使用了BuildKit特性，执行`docker build`前请确保已启用BuildKit。

    1.3 执行构建镜像。

    ```shell
    # 版本号示例，可按实际发布版本调整
    export VERSION=1.0.0
    export DOCKER_BUILDKIT=1

    # 构建 matrixagent 镜像
    # 如需使用自定义镜像仓，请将 cr.openfuyao.cn 替换为实际镜像仓库地址
    docker build -f build/matrixagent.dockerfile -t cr.openfuyao.cn/openfuyao/matrixagent:${VERSION} .

    # 构建 matrixcontroller 镜像
    docker build -f build/matrixcontroller.dockerfile -t cr.openfuyao.cn/openfuyao/matrixcontroller:${VERSION} .
    ```

    1.4 导出镜像包。

    ```shell
    mkdir -p output

    docker save cr.openfuyao.cn/openfuyao/matrixagent:${VERSION} | gzip -c > output/ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
    docker save cr.openfuyao.cn/openfuyao/matrixcontroller:${VERSION} | gzip -c > output/ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
    ```

    1.5 打包Helm Chart。

    ```shell
    helm package charts/matrixagent --destination output
    helm package charts/matrixcontroller --destination output
    mv output/matrixagent-*.tgz output/ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
    mv output/matrixcontroller-*.tgz output/ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
    ```

    构建产物如下：

    ```
        └── output
        ├── ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
    ```

    1.6 （可选）构建superpod-exporter镜像与Chart。
    如需使用指标采集与上报能力，需额外构建`superpod-exporter`镜像与Chart。

    ```shell
    export VERSION=1.0.0
    export DOCKER_BUILDKIT=1

    # 构建 superpod-exporter 镜像
    # 如需使用自定义镜像仓，请将 cr.openfuyao.cn 替换为实际镜像仓库地址
    docker build -f build/superpodexporter.dockerfile -t cr.openfuyao.cn/openfuyao/superpod-exporter:${VERSION} .

    # 导出镜像包
    docker save cr.openfuyao.cn/openfuyao/superpod-exporter:${VERSION} | gzip -c > output/ubs-k8s.superpodexporter.image.${VERSION}.aarch64.tgz

    # 打包Helm Chart
    helm package charts/superpodexporter --destination output
    mv output/superpod-exporter-*.tgz output/ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz
    ```

    构建产物更新如下：
    ```
        └── output
        ├── ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz
        ├── ubs-k8s.superpodexporter.image.${VERSION}.aarch64.tgz
        ├── ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz
    ```

2. 部署步骤。
    执行如下命令，设置版本变量：

    ```bash
    export VERSION=1.0.0
    export OCI_VERSION=0.0.0-latest
    ```

    > ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
    > `VERSION`用于离线方式（方式一）匹配本地构建产物版本号；`OCI_VERSION`用于在线方式（方式二）从OCI仓拉取Chart的版本号，两者相互独立，按实际场景设置其一即可。

    2.1 获取部署文件。
    可根据实际场景选择以下任一种方式获取部署所需镜像和Helm Chart。

    - 方式一：使用离线发布件。

    准备以下文件：

    * `ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz`
    * `ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz`

    - 方式二：从镜像仓和OCI仓获取。

    拉取镜像：

    ```bash
    docker pull cr.openfuyao.cn/openfuyao/matrixcontroller:latest
    docker pull cr.openfuyao.cn/openfuyao/matrixagent:latest
    ```

    拉取Helm Chart：

    ```bash
    helm pull oci://cr.openfuyao.cn/charts/matrixagent --version ${OCI_VERSION}
    helm pull oci://cr.openfuyao.cn/charts/matrixcontroller --version ${OCI_VERSION}
    ```

   2.2 导入离线镜像（仅离线方式）。

    ```bash
    gunzip -c ubs-k8s.matrixagent.image.${VERSION}.aarch64.tgz | ctr -n k8s.io images import -
    gunzip -c ubs-k8s.matrixcontroller.image.${VERSION}.aarch64.tgz | ctr -n k8s.io images import -
    ```

    > ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
    > 步骤1.4使用`docker save`导出的镜像包为docker tar格式，`ctr images import`兼容该格式可直接导入，无需额外转换。
    > 如果使用“方式二”直接从镜像仓拉取镜像，可跳过此步骤。

    2.3 部署服务。
    可根据实际场景选择以下任一种方式部署服务。
    - 使用离线Chart部署。

    ```bash
    helm install matrixagent ubs-k8s.matrixagent.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set images.matrixagent.tag=${VERSION}
    helm install matrixcontroller ubs-k8s.matrixcontroller.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set images.matrixcontroller.tag=${VERSION}
    ```

    - 使用OCI Chart部署。

    ```bash
    helm install matrixagent oci://cr.openfuyao.cn/charts/matrixagent --version ${OCI_VERSION} -n kube-system \
      --set images.matrixagent.tag=latest
    helm install matrixcontroller oci://cr.openfuyao.cn/charts/matrixcontroller --version ${OCI_VERSION} -n kube-system \
      --set images.matrixcontroller.tag=latest
    ```

    2.4 验证结果。
    执行以下命令，查看Pod状态。

    ```bash
    kubectl get pods -A
    ```

    预期结果如下：
    * 每个节点应有对应的`matrixagent`相关Pod，且状态为`Running`。
    * 集群中应有`matrixcontroller`相关Pod，且状态为`Running`。

    2.5（可选）部署superpod-exporter。
    如需使用指标采集与上报能力，部署`superpod-exporter` DaemonSet。

    - 方式一：使用离线Chart部署。

    ```bash
    helm install superpod-exporter ubs-k8s.superpodexporter.chart.${VERSION}.aarch64.tgz -n kube-system \
      --set image.tag=${VERSION}
    ```

    - 方式二：使用OCI Chart部署。

    ```bash
    helm install superpod-exporter oci://cr.openfuyao.cn/charts/superpod-exporter --version ${OCI_VERSION} -n kube-system \
      --set image.tag=latest
    ```

    > ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
    > superpod-exporter DaemonSet默认配置节点亲和性筛选`unifiedbus.com/superpod`标签节点、暴露9102端口、关联ServiceAccount。如需对接Prometheus Operator自动发现，可在部署时设置`serviceMonitor.enabled=true`。部署前请确认目标节点`unifiedbus.com/superpod`标签已配置且UBS socket可访问。

    验证superpod-exporter部署结果：

    ```bash
    kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
    ```

    预期结果：每个配置了`unifiedbus.com/superpod`标签的节点应有对应的`superpod-exporter`Pod，且状态为`Running`。

## 使用SuperPod

### 前提条件

- 已按[安装](#安装)章节完成matrixagent、matrixcontroller组件部署，且各组件Pod状态为`Running`。
- UBS Engine SDK socket（`/run/ubse`）可用。
- （可选）若需使能Volcano HyperNode联动，需已安装Volcano并注册`hypernodes` CRD（`topology.volcano.sh/v1alpha1`），Volcano Scheduler需开启network-topology特性。
- （可选）若需查看SuperPod指标，需已按[安装](#安装)章节2.5部署`superpod-exporter`，且已部署Prometheus抓取与Grafana可视化。

### 背景信息

在大规模集群或UB互联场景中，单节点视角的调度无法感知物理拓扑边界，容易导致跨SuperPod的负载分散，通信效率下降。通过部署UBS K8S Enable相关组件中的SuperPod控制器，可以将物理节点按`superPodId`聚合成逻辑拓扑单元，并以`SuperPod`CR的形式暴露给上层调度器。开启`HYPERNODE_ENABLED=true`后，控制器额外产出Volcano`HyperNode`CR，使Volcano调度器可基于拓扑层级执行网络拓扑感知调度，将通信密集型负载（如分布式训练）调度到同一SuperPod内，提升通信局部性与故障隔离能力。SuperPod控制器以声明式方式自动维护`SuperPod`/`HyperNode`CR的生命周期，无需人工干预。

当前版本不支持Clos组网。

`superpod-exporter`组件以Prometheus格式暴露SuperPod成员关系、NUMA内存借用、共享内存、URMA设备等指标，管理员可在监控平台查看各SuperPod资源使用与健康度，无需登录节点执行CLI，便于及时发现资源倾斜与设备故障（指标定义详见[配置说明-指标定义](#指标定义)，使用样例详见[配置样例](#配置样例)）。

### 使用限制

- **组网限制**：不支持Clos组网。
- **架构限制**：仅支持ARM64架构。
- **superPodId来源要求**：节点必须具备以下任一superPodId来源，否则该节点不会被纳入任何`SuperPod`：
  - 节点label `unifiedbus.com/superpod`；
  - matrixagent上报的`node_network_topology_info`中包含非空`superPodId`字段。
- **层级限制**：FM组网下顶层Tier恒为1，同一`superPodId`下所有节点被装入同一个Tier-1 group（`group-0`）。Volcano拓扑约束`highest-tier`只能取`"1"`。
- **HyperNode联动前置**：开启`HYPERNODE_ENABLED=true`时，集群必须已安装Volcano并注册`hypernodes` CRD；否则控制器会跳过HyperNode装配，仅产生`SuperPod` CR。
- **覆盖语义**：节点label是`superPodId`的优先来源。若节点同时存在label和metric中的`superPodId`，以label为准；label为空时才回退使用metric值。
- **指标导出器限制**：
  - 节点标签`unifiedbus.com/superpod`缺失或RBAC权限不足时，`superpod_name`回退为`"unknown"`，SuperPod级汇聚不可用，节点级指标正常。
  - URMA服务不支持时`ubs_urma_*`指标缺失，其他类别指标正常。

> ![image](./figures/icon-notice.gif) **注意：**
>
> - `SuperPod` CR为cluster scope资源，`metadata.name`由控制器按`superpod-<superPodId>`规则自动生成，请勿手工创建或重命名，否则会被控制器视为stale资源删除。
> - 修改节点label后，控制器会在下一次去抖或周期resync时（最长5min）生效，无需重启matrixcontroller。

### 配置说明

`SuperPod`CRD注册于`matrix.openfuyao.cn`组、`v1`版本、cluster scope，资源名`superpods`。其字段说明如下。

**表3** SuperPod CR字段说明

| 字段路径 | 类型 | 说明 |
| :--- | :--- | :--- |
| `spec.superPodId` | string | 必选。SuperPod标识，直接决定SuperPod命名（`superpod-<superPodId>`）。 |
| `spec.tier` | integer | 必选。SuperPod顶层层级。FM组网下固定为1。 |
| `spec.hyperNodeRef` | string | 可选。引用顶层Volcano HyperNode名称，仅当`HYPERNODE_ENABLED=true`时填充，格式为`hn-t1-<superPodId>`。 |
| `spec.groups[]` | array | 必选。Tier-1拓扑子分组列表。FM组网下每个SuperPod仅含一个group（`group-0`），包含该SuperPod下全部节点。 |
| `spec.groups[].name` | string | group名称，格式为`group-<ordinal>`（FM下为`group-0`）。 |
| `spec.groups[].tier` | integer | group层级，FM组网下固定为1。 |
| `spec.groups[].hyperNodeRef` | string | 可选。引用该group对应的Tier-1 HyperNode，仅当`HYPERNODE_ENABLED=true`时填充。 |
| `spec.groups[].nodes[]` | array | group下的节点资源信息列表。 |
| `spec.groups[].nodes[].name` | string | 节点名。 |
| `spec.groups[].nodes[].ip` | string | 节点内网IP地址。 |
| `spec.groups[].nodes[].memory.total` | string | 节点物理内存总量（BinarySI，如`256Gi`）。 |
| `spec.groups[].nodes[].memory.used` | string | 节点已用物理内存（BinarySI）。 |
| `status.nodeCount` | integer | SuperPod成员节点数。 |
| `status.conditions[]` | array | 状态条件列表，遵循Kubernetes Condition规范。 |
| `metadata.annotations["superpod.matrix.huawei.com/node-hash"]` | string | 成员节点名排序后SHA256取前8字符，作为拓扑指纹，用于成员变更校验。 |

#### 指标定义

`superpod-exporter`产出的Prometheus指标统一`ubs_`前缀，单位字节。指标为节点级，所有指标携带`node`/`slot_id`label，SuperPod维度以`superpod_name`label承载。

**表5** superpod-exporter公共Label语义

| Label | 含义 |
| :--- | :--- |
| `superpod_name` | SuperPod名称，取自节点标签`unifiedbus.com/superpod` |
| `node` | K8s节点名 |
| `slot_id` | UBS物理节点唯一标识 |
| `export_node` | 借出/提供方节点的K8s节点名 |
| `export_slot_id` | 借出/提供方节点的UBS slot_id |
| `name` | 借用/共享资源的名称 |
| `numa_id` | 借用形成的远端NUMA id |
| `device_name` | URMA设备名称 |
| `hw_res_id` | URMA硬件资源ID |

**表6** superpod-exporter指标定义

| 指标名 | 描述 | 数据类型 | 指标值 | 指标label |
| :--- | :--- | :--- | :--- | :--- |
| `ubs_exporter_up` | superpod-exporter就绪状态 | Gauge | 1=就绪，0=不可用 | 无 |
| `ubs_superpod_info` | 节点与SuperPod归属信息 | Gauge | 恒1 | `superpod_name`, `node`, `slot_id` |
| `ubs_mem_numa_borrow_bytes` | 本节点借入的NUMA远端内存大小 | Gauge | NUMA借用大小（字节） | `superpod_name`, `node`, `slot_id`, `export_node`, `export_slot_id`, `numa_id`, `name` |
| `ubs_mem_numa_borrow_count` | 本节点NUMA借用关系数 | Gauge | NUMA借用关系总数 | `superpod_name`, `node`, `slot_id` |
| `ubs_mem_shm_provide_bytes` | 本节点提供的共享内存大小 | Gauge | 共享内存大小（字节） | `superpod_name`, `node`, `slot_id`, `name` |
| `ubs_mem_shm_provide_count` | 本节点提供的共享内存数 | Gauge | 共享内存数 | `superpod_name`, `node`, `slot_id` |
| `ubs_urma_device_info` | URMA设备信息（用于发现/关联） | Gauge | 恒1 | `superpod_name`, `node`, `slot_id`, `device_name`, `hw_res_id` |
| `ubs_urma_device_healthy` | URMA设备健康状态 | Gauge | 1=健康，0=故障 | `superpod_name`, `node`, `slot_id`, `device_name` |

### 配置样例

**表2** FM组网组装结果

以含2个SuperPod（`superPodId=0`、`superPodId=1`，共16节点）的集群为例，组装产出（按`superPodId`分组为2个SuperPod，FM全互联下每SuperPod内仅单Tier-1子组，topTier=1）：

| 层级 | HyperNode | 成员类型 | 成员 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| Tier 1 | `hn-t1-0` | Node | node1~node8 | SuperPod A内8节点1跳连通分量。 |
| Tier 1 | `hn-t1-1` | Node | node9~node16 | SuperPod B内8节点1跳连通分量。 |

**样例1**：SuperPod CR（FM组网，8节点全互联，`HYPERNODE_ENABLED=true`）。

```yaml
apiVersion: matrix.openfuyao.cn/v1
kind: SuperPod
metadata:
  name: superpod-0               # 命名直接来自 superPodId
  annotations:
    superpod.matrix.huawei.com/node-hash: a1b2c3d4   # 成员节点名排序 hash 前 8 位
spec:
  superPodId: "0"                 # SuperPod 标识
  tier: 1                         # FM 全互联，仅 Tier-1
  hyperNodeRef: hn-t1-0           # 引用顶层 HyperNode（HYPERNODE_ENABLED=true 时填充）
  groups:                         # 单组（全体8节点1跳连通）
    - name: group-0
      tier: 1
      hyperNodeRef: hn-t1-0       # 引用同层 Tier-1 HyperNode
      nodes:
        - name: node1
          ip: "10.8.0.1"
          memory:
            total: "256Gi"
            used: "128Gi"
        # ... node2 ~ node7
        - name: node8
          ip: "10.8.0.8"
          memory:
            total: "256Gi"
            used: "120Gi"
status:
  nodeCount: 8
```

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> 当`HYPERNODE_ENABLED=false`（默认）时，`hyperNodeRef`字段留空，SuperPod仍完整承载`superPodId`、拓扑层级与节点资源，仅不引用HyperNode。上方示例为启用时的形态，未启用时将`hyperNodeRef`行去掉即可。

**样例2**：HyperNode CR（FM组网，Tier-1，8节点全互联，`HYPERNODE_ENABLED=true`时由控制器自动产出）。

```yaml
apiVersion: topology.volcano.sh/v1alpha1
kind: HyperNode
metadata:
  name: hn-t1-0
spec:
  tier: 1
  tierName: "superpod"
  members:
    - type: Node
      selector:
        exactMatch:
          name: node1
    # ... node2 ~ node7
    - type: Node
      selector:
        exactMatch:
          name: node8
status:
  nodeCount: 8
```

**样例3**：三Pod Gang部署到同一SuperPod（FM组网，`highest-tier=1`）。

场景：3个业务Pod（如分布式训练Worker）需部署在同一个SuperPod内以使用高速互联与池化内存通信，且要求gang调度（3个全部调度成功，否则全部等待）。

1. PodGroup：声明gang策略 + 拓扑硬约束。

```yaml
apiVersion: scheduling.volcano.sh/v1beta1
kind: PodGroup
metadata:
  name: gang-in-superpod
  annotations:
    volcano.sh/network-topology-mode: "hard"       # 硬约束：同组 Pod 必须落在同一 HyperNode
    volcano.sh/network-topology-highest-tier: "1" # FM 组网仅 Tier-1，限制在同一 SuperPod 内
spec:
  minMember: 3                    # gang 策略：3 个 Pod 必须全部调度成功，否则全部 pending
  queue: default
  priorityClassName: high
```

2. 业务Pod：关联PodGroup，由Volcano调度。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: worker-0
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod   # 关联 PodGroup
spec:
  schedulerName: volcano                               # 使用 Volcano 调度器
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
---
apiVersion: v1
kind: Pod
metadata:
  name: worker-1
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod
spec:
  schedulerName: volcano
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
---
apiVersion: v1
kind: Pod
metadata:
  name: worker-2
  annotations:
    scheduling.k8s.io/group-name: gang-in-superpod
spec:
  schedulerName: volcano
  containers:
    - name: worker
      image: registry.example.com/app/worker:1.0
      resources:
        requests: { cpu: "8", memory: "16Gi" }
        limits:   { cpu: "8", memory: "16Gi" }
```

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> - **gang + hard topology组合**：Volcano先做gang检查（`minMember`），再做拓扑约束校验；`hard`模式下若没有任何Tier-1 HyperNode能同时容纳3个Pod，则全部pending，不会部分调度。
> - **soft模式**（可选）：将`mode`改为`soft`，则拓扑为打分偏好而非硬约束，优先调度到同一SuperPod但允许降级到其他SuperPod。
> - **FM组网`highest-tier`**：FM下SuperPod间不互联，`highest-tier`只能取`"1"`。

**样例4**：superpod-exporter指标输出样例（`curl <node>:9102/metrics`，节选）。

```
# HELP ubs_exporter_up superpod-exporter is up and UBSE SDK is initialized (1=up, 0=SDK unavailable).
# TYPE ubs_exporter_up gauge
ubs_exporter_up 1
# HELP ubs_superpod_info SuperPod membership info: which SuperPods exist and which physical nodes belong to each.
# TYPE ubs_superpod_info gauge
ubs_superpod_info{superpod_name="0",node="node1",slot_id="1"} 1
ubs_superpod_info{superpod_name="0",node="node2",slot_id="2"} 1
# HELP ubs_mem_numa_borrow_bytes Bytes of numa-form remote memory borrowed by this node from export_node.
# TYPE ubs_mem_numa_borrow_bytes gauge
ubs_mem_numa_borrow_bytes{superpod_name="0",node="node1",slot_id="1",export_node="node2",export_slot_id="2",numa_id="4",name="numa-remote-0"} 2.147483648e+09
# HELP ubs_mem_numa_borrow_count Number of numa-form memory borrow relationships on this node.
# TYPE ubs_mem_numa_borrow_count gauge
ubs_mem_numa_borrow_count{superpod_name="0",node="node1",slot_id="1"} 1
# HELP ubs_mem_shm_provide_bytes Bytes of shared memory provided by this node (export_node == local).
# TYPE ubs_mem_shm_provide_bytes gauge
ubs_mem_shm_provide_bytes{superpod_name="0",node="node1",slot_id="1",name="shm-provide-0"} 1.073741824e+09
# HELP ubs_mem_shm_provide_count Number of shared memory regions provided by this node.
# TYPE ubs_mem_shm_provide_count gauge
ubs_mem_shm_provide_count{superpod_name="0",node="node1",slot_id="1"} 1
# HELP ubs_urma_device_info URMA device info (always 1, used for discovery/association).
# TYPE ubs_urma_device_info gauge
ubs_urma_device_info{superpod_name="0",node="node1",slot_id="1",device_name="urma-0",hw_res_id="100"} 1
# HELP ubs_urma_device_healthy URMA device health status: 1=healthy, 0=fault.
# TYPE ubs_urma_device_healthy gauge
ubs_urma_device_healthy{superpod_name="0",node="node1",slot_id="1",device_name="urma-0"} 1
```

**样例5**：Grafana PromQL汇聚示例。

| 视图 | PromQL |
| :--- | :--- |
| SuperPod数量 | `count(count by (superpod_name)(ubs_superpod_info))` |
| 各SuperPod物理节点数 | `count by (superpod_name)(ubs_superpod_info)` |
| 指定SuperPod的成员节点列表 | `ubs_superpod_info{superpod_name="$superpod"}` |
| SuperPod NUMA借入总量 | `sum by (superpod_name)(ubs_mem_numa_borrow_bytes)` |
| 节点NUMA借入总量 | `sum by (node)(ubs_mem_numa_borrow_bytes)` |
| 节点借出总量 | `sum by (export_node)(ubs_mem_numa_borrow_bytes)` |
| SuperPod共享内存提供总量 | `sum by (superpod_name)(ubs_mem_shm_provide_bytes)` |
| SuperPod URMA健康设备数 | `sum by (superpod_name)(min by (superpod_name, device_name)(ubs_urma_device_healthy))` |
| URMA故障设备 | `min by (superpod_name, device_name)(ubs_urma_device_healthy) == 0` |

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> - URMA指标为SuperPod粒度数据，每节点全量上报会产生N份重复series，SuperPod级查询必须带`min by (superpod_name, device_name)`前缀聚合去重（故障优先，任一节点观测到故障即判故障）。
> - 节点内存借还仅有一种运行态（借入或借出的关系实例），借还总量可直接由本节点已上报的借用关系series汇总得出：节点借入总量 = `sum by (node)(ubs_mem_numa_borrow_bytes)`，节点借出总量 = `sum by (export_node)(ubs_mem_numa_borrow_bytes)`。

### 操作步骤

1. 使能SuperPod拓扑纳管。
    **前置条件**
    完成matrixagent和matrixcontroller的安装。UBS Engine SDK socket（`/run/ubse`）可用。
    1.1 配置节点superPodId标签。
    在K8s的master节点通过命令行配置worker节点的标签，标识节点所属的SuperPod。节点label是superPodId的优先来源，未打标签的节点会回退使用matrixagent上报的`superPodId`字段；二者皆空则该节点不会被纳入任何SuperPod。

    ```shell
    kubectl label nodes <node-name> unifiedbus.com/superpod=<superPodId>
    # <superPodId> 替换为该节点所属的 SuperPod 标识（如 0、1、2）
    # <node-name> 替换为需要纳管的节点名
    # 示例：将 node1~node8 归入 SuperPod 0（FM 组网）
    #   kubectl label nodes node1 unifiedbus.com/superpod=0
    #   kubectl label nodes node2 unifiedbus.com/superpod=0
    #   ... 至 node8
    ```

    > ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
    > `unifiedbus.com/superpod`的值为字符串形式的superPodId，控制器据此将节点归入`superpod-<superPodId>`。FM组网下，建议同一UB互联域（同一SuperPod）内的所有物理节点使用相同的superPodId。

    1.2 （可选）使能Volcano HyperNode联动。
    若需将SuperPod拓扑联动到Volcano的`HyperNode` CR（供Volcano调度器执行网络拓扑感知调度），需将matrixcontroller的环境变量`HYPERNODE_ENABLED`设置为`true`。默认值为`false`，即仅产生`SuperPod` CR，不联动HyperNode。

    前置条件：
    - 集群已安装Volcano并注册`hypernodes` CRD（`topology.volcano.sh/v1alpha1`）。可通过以下命令确认：

    ```bash
    kubectl get crd hypernodes.topology.volcano.sh
    ```

    - Volcano Scheduler已开启network-topology特性。

    - 方式一：部署后通过`kubectl set env`修改（无需重新部署Chart）。

    ```bash
    kubectl set env deployment/kube-matrix-controller -n kube-system HYPERNODE_ENABLED=true
    kubectl rollout restart deployment/kube-matrix-controller -n kube-system
    ```

    - 方式二：部署前修改`charts/matrixcontroller/templates/deploy.yaml`，将`HYPERNODE_ENABLED`的`value`改为`"true"`，再按[开始安装](#开始安装)中部署服务的步骤部署matrixcontroller。

    > ![输入图片说明](./figures/icon-notice.gif) **注意：**<br />
    > 仅当`HyperNode` CRD已注册时控制器才会装配HyperNode；若CRD未就绪，控制器会跳过HyperNode装配并打印告警日志，但`SuperPod` CR的产生不受影响。

    1.3 触发拓扑调谐。
    完成节点label配置后，matrixcontroller会自动侦听MatrixMetric CR的变化并触发调谐：
    - MatrixMetric CR增、改、删事件触发去抖调谐（5s去抖窗口）。
    - 每5min执行一次周期全量resync。
    - 拓扑数据12h刷新一次。

    如需立即触发调谐，可等待matrixagent下一次上报，或手动触发MatrixMetric CR变更（如`kubectl annotate`任一MatrixMetric CR触发UpdateEvent）。

    1.4 验证SuperPod CR。
    执行以下命令，查看集群中的SuperPod CR。

    ```bash
    kubectl get superpods.matrix.openfuyao.cn
    ```

    预期结果：每个有节点归属的`superPodId`对应一个`superpod-<id>`资源，例如：

    ```
    NAME          SUPERPODID   TIER   NODECOUNT
    superpod-0    0            1      8
    superpod-1    1            1      8
    ```

    查看SuperPod详细信息（含group、节点、内存）：

    ```bash
    kubectl get superpod superpod-0 -o yaml
    ```

    预期输出（FM组网，`HYPERNODE_ENABLED=true`，节选）：

    ```yaml
    apiVersion: matrix.openfuyao.cn/v1
    kind: SuperPod
    metadata:
      name: superpod-0
      annotations:
        superpod.matrix.huawei.com/node-hash: a1b2c3d4
    spec:
      superPodId: "0"
      tier: 1
      hyperNodeRef: hn-t1-0
      groups:
        - name: group-0
          tier: 1
          hyperNodeRef: hn-t1-0
          nodes:
            - name: node1
              ip: 10.8.0.1
              memory:
                total: 256Gi
                used: 128Gi
            # ... node2 ~ node7
            - name: node8
              ip: 10.8.0.8
              memory:
                total: 256Gi
                used: 120Gi
    status:
      nodeCount: 8
    ```

    1.5 （可选）验证HyperNode CR。
    若已开启`HYPERNODE_ENABLED=true`，执行以下命令查看联动产生的Volcano HyperNode CR。

    ```bash
    kubectl get hypernodes.topology.volcano.sh
    ```

    预期结果：每个`superPodId`对应一个`hn-t1-<id>`资源，其`spec.members`包含该SuperPod下所有节点。

    1.6 查看节点哈希指纹。
    SuperPod的annotation `superpod.matrix.huawei.com/node-hash`为成员节点名排序后的SHA256取前8字符，可用于快速判断SuperPod成员拓扑是否变化。

    ```bash
    kubectl get superpod <superpod-name> \
      -o jsonpath='{.metadata.annotations.superpod\.matrix\.huawei\.com/node-hash}'
    ```

    - 观察拓扑变更结果。
    当节点label变更或MatrixMetric CR更新导致SuperPod成员变化时，控制器会在下一次调谐中更新`SuperPod` CR的`spec.groups[].nodes[]`与`status.nodeCount`，并刷新`node-hash`。可重复执行验证SuperPod CR小节的命令观察变化。
2. （可选）业务Pod调度到同一SuperPod。
    **前置条件**
    完成HyperNode联动使能（[操作步骤](#操作步骤)的“使能SuperPod拓扑纳管”小节），Volcano Scheduler已开启network-topology特性。
    2.1 创建PodGroup。
    参考[配置样例-样例3](#配置样例)，创建声明gang策略与拓扑硬约束的PodGroup。FM组网下`highest-tier`只能取`"1"`。

    ```bash
    kubectl apply -f podgroup-gang-in-superpod.yaml
    ```

    2.2 创建业务Pod。
    创建关联PodGroup的业务Pod，由Volcano调度器调度。

    ```bash
    kubectl apply -f workers.yaml
    ```

    2.3 验证调度结果。
    执行以下命令，查看Pod调度状态。

    ```bash
    kubectl get pod -o wide
    ```

    预期结果：3个worker Pod全部调度成功且位于同一SuperPod内（同一`superPodId`的节点上）。若没有任何Tier-1 HyperNode能同时容纳3个Pod，则全部pending。
3. （可选）验证superpod-exporter指标。
    **前置条件**
    已按[安装](#安装)章节2.5部署`superpod-exporter`，且Pod状态为`Running`。
    3.1 验证导出器就绪。
    执行以下命令，查看superpod-exporter Pod状态与就绪指标。

    ```bash
    kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
    ```

    预期结果：每个配置了`unifiedbus.com/superpod`标签的节点应有对应的`superpod-exporter`Pod，且状态为`Running`。

    3.2 查询指标端点。
    通过`kubectl exec`进入任一superpod-exporter Pod，或直接`curl`节点IP查询指标端点。

    ```bash
    # 方式一：kubectl exec 查询
    POD=$(kubectl -n kube-system get pods -l app.kubernetes.io/name=superpod-exporter -o jsonpath='{.items[0].metadata.name}')
    kubectl -n kube-system exec $POD -- curl -s localhost:9102/metrics | grep ubs_exporter_up

    # 方式二：直接 curl 节点IP（需节点端口可达）
    curl -s <node-ip>:9102/metrics | grep ubs_superpod_info
    ```

    预期结果：`ubs_exporter_up`值为1；`ubs_superpod_info`含本节点`node`/`slot_id`/`superpod_name`，`superpod_name`值与节点标签`unifiedbus.com/superpod`一致。

    3.3 验证SuperPod维度汇聚。
    在Grafana或Prometheus中执行[配置样例-样例5](#配置样例)中的PromQL，验证SuperPod级汇聚视图。

    ```bash
    # SuperPod数量
    count(count by (superpod_name)(ubs_superpod_info))
    # 各SuperPod物理节点数
    count by (superpod_name)(ubs_superpod_info)
    # SuperPod NUMA借入总量
    sum by (superpod_name)(ubs_mem_numa_borrow_bytes)
    # URMA故障设备
    min by (superpod_name, device_name)(ubs_urma_device_healthy) == 0
    ```

    预期结果：SuperPod数量、成员数、借入总量、URMA健康状态等查询返回正确结果。

### 后续操作

- **验证组件状态**：通过`kubectl get pods -A`查看matrixagent、matrixcontroller与`superpod-exporter`运行状态，确认均为`Running`。
- **检查CRD注册**：通过`kubectl get crd superpods.matrix.openfuyao.cn`确认SuperPod CRD已注册；若开启HyperNode联动，通过`kubectl get crd hypernodes.topology.volcano.sh`确认HyperNode CRD已注册。
- **验证指标导出器**：通过`curl <node-ip>:9102/metrics`确认`ubs_exporter_up=1`且各`ubs_*`指标series数与节点实际借用关系一致；`superpod_name`应与节点标签一致，非`unknown`。
- **调整拓扑归属**：如需调整某节点的SuperPod归属，重新执行[操作步骤](#操作步骤)中配置节点superPodId标签的步骤即可，无需重启matrixcontroller；控制器会在下一次去抖或周期resync时（最长5min）生效。`superpod-exporter`的`superpod_name`label源自节点标签，标签变更后exporter会在12h缓存到期后刷新，或重启exporter立即生效。
- **切换HyperNode联动**：如需开启/关闭HyperNode联动，通过`kubectl set env deployment/kube-matrix-controller -n kube-system HYPERNODE_ENABLED=<true|false>`并`kubectl rollout restart`即可，全量resync会幂等覆盖既有资源。
- **故障排查**：
  1. `kubectl get superpod`检查数量与成员，`nodeCount`应与该SuperPod内实际节点数一致。
  2. 查matrixcontroller日志：`kubectl logs -n kube-system -l app=kube-matrix-controller`。
  3. 查matrixagent日志确认采集正常：`kubectl logs -n kube-system -l app=kube-matrix-agent`。
  4. 检查UBS Engine SDK socket：节点上`/run/ubse`是否存在。
  5. 指标缺失排查：`kubectl logs -n kube-system -l app.kubernetes.io/name=superpod-exporter`查exporter日志；`superpod_name`为`unknown`时检查节点标签`unifiedbus.com/superpod`与RBAC权限。

### 相关操作

- **查看SuperPod**：

  ```bash
  kubectl get superpods.matrix.openfuyao.cn
  kubectl get superpod <superpod-name> -o yaml
  kubectl get superpod <superpod-name> \
    -o jsonpath='{.metadata.annotations.superpod\.matrix\.huawei\.com/node-hash}'
  ```

- **查看HyperNode**（仅`HYPERNODE_ENABLED=true`时存在）：

  ```bash
  kubectl get hypernodes.topology.volcano.sh
  kubectl get hypernode <hn-name> -o yaml
  ```

- **删除SuperPod / HyperNode**：

  > ![image](./figures/icon-notice.gif) **注意：**
  >
  > 控制器会自动维护`SuperPod`/`HyperNode` CR的生命周期，正常情况下无需手工删除。仅在停用特性、清理残留资源或排查异常时手工删除。建议优先按名称删除特定资源，仅在下线或重置场景使用批量删除。

  ```bash
  kubectl delete superpod <superpod-name>
  kubectl delete hypernode <hn-name>
  # 批量清理全部资源（仅下线/重置场景使用）
  kubectl delete superpods.matrix.openfuyao.cn --all
  kubectl delete hypernodes.topology.volcano.sh --all
  ```

- **查看superpod-exporter指标**：

  ```bash
  kubectl get pods -n kube-system -l app.kubernetes.io/name=superpod-exporter -o wide
  curl -s <node-ip>:9102/metrics | grep ubs_
  curl -s <node-ip>:9102/healthz
  ```

- **停用superpod-exporter**：卸载`superpod-exporter` DaemonSet即停用指标采集，无残留K8s资源，不影响既有matrixagent、matrixcontroller与`SuperPod`/`HyperNode`资源。

  ```bash
  helm uninstall superpod-exporter -n kube-system
  ```
