# vNPU

## 特性介绍

vNPU是构建在容器平台之上的昇腾NPU算力切分和动态调度能力组件。它支持多个容器共享同一NPU设备的同时，具备显存和AI Core资源的隔离能力，从而显著提升NPU资源利用率。通过提供多样化调度策略、多样化算力切分模式、可观测等能力，打造一体化极简易用的昇腾NPU算力接入和纳管方案。

当用户在Kubernetes集群中运行AI训练/推理负载、希望以更细粒度共享昂贵的昇腾NPU资源、并要求按容器隔离显存与算力配额时，可使用vNPU特性。

### 应用场景

- **多租户推理服务**：同一张昇腾NPU卡上并发承载多个推理服务实例，每个实例按需申请显存和AI Core配额，降低单实例成本。
- **AI开发环境共享**：多名算法工程师在同一物理卡上各占一份vNPU作为开发调试环境，互不影响显存使用。
- **训练与推理混部**：同一节点上同时承载训练和推理任务，通过调度策略（binpack/spread）控制Pod在节点和设备上的分布，平衡资源利用率与故障隔离。
- **资源精细化运营**：通过xpu-exporter对接Prometheus采集vNPU使用率、Pod数量等指标，支撑容量规划和成本核算。

### 能力范围

- **算力切分**：支持软切分（CANN Runtime API劫持的用户态虚拟化）和硬切分（基于昇腾HDK的虚拟化）两种模式，可按节点粒度配置。
  - 软切分：单张物理NPU最多切分为20个vNPU实例，AI Core切分粒度最小1%，显存切分粒度最小1Gi；AI Core与显存隔离（显存严格隔离，AI Core为时分复用，波动小于整卡的10%）；支持只限制显存不限制AI Core；支持固定配额、弹性、争抢三种AI Core调度策略；性能损耗小于5%。
  - 硬切分：基于昇腾HDK将NPU切分为多份vNPU挂载到容器中使用。
- **集群调度**：基于Volcano扩展NPU资源调度插件，支持资源共享调度、binpack/spread调度、整卡调度，以及[基于DRA的调度](https://gitcode.com/openFuyao/sig-orchestration-engine/blob/master/docs/zh/npu_dra_plugin_user_guide/npu_dra_plugin.md)。允许同一节点上部分容器使用整卡、部分容器共享一张卡。
- **可观测**：xpu-exporter采集vNPU切分模式、调度策略、显存使用率、vNPU数量、Pod数量等指标，对接Prometheus。

### 亮点特征

- **软切分细粒度**：AI Core切分粒度最小1%、显存切分粒度最小1Gi，业界同等方案中切分粒度更细。
- **软切分性能损耗低**：性能损耗小于5%，接近裸卡算力。
- **AI Core调度策略丰富**：固定配额、弹性、争抢三种策略，弹性策略通过时间片借用机制提升整卡利用率。
- **调度策略灵活**：节点级与设备级分别支持binpack/spread，适配不同负载特征。
- **整卡与切分混部**：同一节点同时支持整卡容器与共享容器，无需独立集群。

### 基本概念

- **软切分（Soft）**：基于CANN Runtime API劫持的用户态虚拟化方案。AI Core为时分复用，非严格隔离；显存严格隔离。
- **硬切分（Hard）**：基于昇腾HDK的虚拟化方案，将物理NPU切分为多份vNPU挂载到容器中使用。
- **vNPU**：虚拟NPU，由一张物理NPU切分得到的逻辑单元，业务Pod通过`huawei.com/vnpu-number`等资源声明申请使用。
- **dieID**：NPU设备的唯一标识，用于精确指定Pod落位到某张物理卡。
- **AI Core调度策略**：软切分场景下AI Core时间片的分配方式，包括固定配额（`fixed-share`）、弹性（`elastic`）、争抢（`best-effort`）三种。
- **节点级调度策略**（`huawei.com/vnpu-pod-node-scheduler-policy`）：控制Pod在节点间的分布，`binpack`紧凑调度，`spread`打散调度。
- **设备级调度策略**（`huawei.com/vnpu-pod-device-scheduler-policy`）：控制vNPU在单节点NPU设备间的分布，`binpack`紧凑调度，`spread`打散调度。

### 实现原理

vNPU由四个核心组件协作完成NPU资源的发现、调度、挂载与监控，整体工作流程如下：

1. **xpu-device-plugin**：通过Kubernetes Device Plugin机制向kubelet上报节点上的NPU设备及可切分资源（注册到`huawei.com/vnpu-*`资源），并向节点注解`huawei.com/node-vnpu-register`写入设备详情（dieID、芯片类型等字段）。
2. **volcano-xpu-plugin**：扩展Volcano调度器，在Pod调度阶段根据Pod声明的`huawei.com/vnpu-*`资源量、切分模式、调度策略，从候选节点的可用vNPU中筛选合适的节点和设备，完成过滤和打分。
3. **client_update**：在节点侧部署软切分劫持库，业务Pod启动时由device plugin挂载劫持库到容器内，实现AI Core和显存的隔离限制。
4. **xpu-exporter**：持续采集vNPU运行时指标（切分模式、显存利用率、vNPU数量等），通过Prometheus暴露供监控。

Pod调度关键时机：

- Pod声明`spec.schedulerName: volcano`后，调度由Volcano接管。
- Volcano调用vxpu plugin执行NPU资源过滤与打分，选定节点与设备。
- Pod绑定到节点后，device plugin根据调度结果通过`Allocate`回调向容器挂载设备文件和劫持库。
- Pod启动业务进程时，劫持库接管CANN Runtime调用，按声明配额执行AI Core时间片调度和显存隔离。

### 与相关特性的关系

- **Volcano**：vNPU调度依赖Volcano 1.15.0作为调度器底座，通过Volcano的扩展插件机制接入NPU资源调度。业务Pod必须显式声明`spec.schedulerName: volcano`。
- **Ascend Docker Runtime**：用于容器内挂载NPU设备，vNPU要求节点已部署Ascend Docker Runtime（建议7.2.RC1+版本）。
- **昇腾HDK驱动**：NPU硬件驱动，软切分和硬切分均依赖。硬切分依赖HDK提供的设备切分能力；支持的芯片型号及HDK版本配套关系详见[附录-支持的NPU型号](#支持的npu型号)。
- **昇腾MindCluster NPU Exporter**：用于采集NPU整卡指标，vNPU的xpu-exporter仅采集vNPU切分维度指标，二者互补。
- **npu-dra-plugin**：vNPU提供[基于DRA的调度](https://gitcode.com/openFuyao/sig-orchestration-engine/blob/master/docs/zh/npu_dra_plugin_user_guide/npu_dra_plugin.md)作为Volcano调度路径的可选替代。

### 相关实例

业务Pod使用样例请参见本文档[使用vNPU-配置样例](#配置样例)小节。更多示例与Demo工程请参考[vNPU仓库](https://gitcode.com/openFuyao/vNPU)。

## 安装

### 前提条件

- 已准备Kubernetes集群。
- 已以root用户在节点上安装昇腾HDK驱动（支持的芯片型号及HDK版本配套关系详见[附录-支持的NPU型号](#支持的npu型号)）。
- 已部署Ascend Docker Runtime，建议使用7.2.RC1+版本。以7.2.RC1版本为例，详细部署方式请参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)。
- 编译机已安装以下工具：
  - git
  - docker > 20.10
  - helm

### 开始安装

#### 编译构建

1. 下载源码并初始化子模块。

   ```shell
   git clone https://gitcode.com/openFuyao/vNPU.git
   cd vNPU
   git submodule update --init --recursive
   ```

2. 进入`ci`目录执行构建脚本。

   ```shell
   cd vNPU/ci
   # 使用默认参数构建，镜像版本号默认为1.0.0
   sh build.sh
   # 指定镜像版本号为1.1.1
   sh build.sh 1.1.1
   ```

   构建产物输出在`vNPU/output`目录下。

#### 在线部署

1. 为需要部署vNPU组件的节点增加label。

   ```bash
   kubectl label node {nodename} huawei.com/vnpu=ready
   ```

2. 开启NPU设备共享模式。参考[昇腾官方文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531)。对于910B芯片，HDK驱动25.5.0及以上版本才支持共享模式。例如，设置设备0所有芯片的容器共享模式为使能：

   ```bash
   npu-smi set -t device-share -i 0 -d 1
   ```

   >![image](figures/icon-notice.gif) **注意：**
   >
   >设备共享模式是后续Volcano调度和device plugin上报资源的前置条件，必须在部署vNPU组件前完成配置，否则Pod可能因节点未开启共享而调度失败。`npu-smi`命令中`-i`指定设备ID、`-d`指定共享模式（1为使能），请按实际设备替换。支持的芯片型号及HDK版本配套关系详见[附录-支持的NPU型号](#支持的npu型号)。

3. 创建xpu Namespace。

   ```bash
   kubectl apply -f charts/yaml/namespace.yaml
   ```

4. 通过Helm Chart部署npu-client-update、npu-device-plugin、xpu-exporter。

   ```bash
   helm install vxpu oci://cr.openfuyao.cn/charts/vxpu --version 1.0.0
   ```

   >![image](figures/icon-note.gif) **说明：**
   >
   >- `oci://cr.openfuyao.cn/charts/vxpu`会从远端OCI仓库直接拉取Chart；离线环境请改用本地tgz包，见[离线部署](#离线部署)小节。
   >- `--version`需与实际发布的Chart版本号匹配，本文以`1.0.0`为例，请以[vNPU仓库](https://gitcode.com/openFuyao/vNPU)发布为准。

5. 在主机OS上创建Volcano日志目录并设置权限。Volcano容器以UID 1000运行，需赋予目录写权限。

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

6. 通过`charts/yaml/volcano-deployment.yaml`部署Volcano。

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```

#### 离线部署

1. 为需要部署vNPU组件的节点增加label。

   ```bash
   kubectl label node {nodename} huawei.com/vnpu=ready
   ```

2. 开启NPU设备共享模式。参考[昇腾官方文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531)。对于910B芯片，HDK驱动25.5.0及以上版本才支持共享模式。例如，设置设备0所有芯片的容器共享模式为使能：

   ```bash
   npu-smi set -t device-share -i 0 -d 1
   ```

   >![image](figures/icon-notice.gif) **注意：**
   >
   >设备共享模式是后续Volcano调度和device plugin上报资源的前置条件，必须在部署vNPU组件前完成配置，否则Pod可能因节点未开启共享而调度失败。`npu-smi`命令中`-i`指定设备ID、`-d`指定共享模式（1为使能），请按实际设备替换。支持的芯片型号及HDK版本配套关系详见[附录-支持的NPU型号](#支持的npu型号)。

3. 创建xpu Namespace。

   ```bash
   kubectl apply -f charts/yaml/namespace.yaml
   ```

4. 导入离线镜像包。以containerd本地镜像为例，镜像包名称请以实际为准。

   ```shell
   ctr -n k8s.io images import acl_client_update-1.0.0-aarch64.tar
   ctr -n k8s.io images import npu_device_plugin-1.0.0-aarch64.tar
   ctr -n k8s.io images import xpu_exporter-1.0.0-aarch64.tar
   ctr -n k8s.io images import vc_controller_manager-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc_scheduler-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc_webhook_manager-1.15.0-aarch64.tar
   ```

5. 通过Helm Chart部署npu-client-update、npu-device-plugin、xpu-exporter。

   ```bash
   helm upgrade --install vxpu vxpu-1.0.0.tgz -n xpu --wait
   ```

6. 在主机OS上创建Volcano日志目录并设置权限。Volcano容器以UID 1000运行，需赋予目录写权限。

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

7. 通过`charts/yaml/volcano-deployment.yaml`部署Volcano。

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```

## 使用vNPU

### 前提条件

- Kubernetes集群已就绪。
- 节点已安装昇腾HDK驱动（支持的芯片型号及HDK版本配套关系详见[附录-支持的NPU型号](#支持的npu型号)）。
- 已部署Ascend Docker Runtime（建议7.2.RC1+版本）。
- 已按[安装](#安装)章节完成vNPU组件部署，且节点已添加`huawei.com/vnpu=ready`标签。
- 业务Pod所在集群中Volcano可用，且`spec.schedulerName`设置为`volcano`。可通过以下命令确认Volcano组件运行状态：

  ```bash
  kubectl get deployment -n volcano-system
  kubectl get pod -n volcano-system
  ```
  
  预期回显：volcano-system命名空间下的所有Pod为running状态。

### 背景信息

业务Pod只需声明所需的vNPU资源，无需修改业务代码。vNPU的调度器（volcano-xpu-plugin）和device plugin（xpu-device-plugin）负责自动调度Pod到合适的节点、合适的NPU设备上并挂载设备。Pod可通过注解配置切分模式（soft/hard）、AI CPU分配策略、软切分算力调度策略、节点级和设备级调度策略，以及精确指定dieID和NPU设备类型。

软切分AI Core调度策略对比如下。

**表1** 软切分AI Core调度策略对比

| 模式名称 | 特点描述 |
| :--- | :--- |
| 固定配额模式（`fixed-share`） | 各vNPU按配比严格执行时间片。a. 若有暂未分配的vNPU资源，则在最后一个vNPU时间片消耗完时，睡眠一段时间，此时NPU将闲置空转。b. 某vNPU上无任务运行，仍会按照给定比例消耗完该vNPU的时间片。 |
| 弹性模式（`elastic`） | 在各vNPU按配比执行时间片的基础上，优化调度逻辑以提升整卡利用率。a. 最后一个vNPU时间片消耗完，马上将NPU使用权释放给下一个vNPU，跳过睡眠逻辑。b. 在某vNPU上无任务运行，则直接跳过未消耗完的时间片，切换至另一个vNPU。c. 时间片借用机制：正在执行的vNPU在识别到卡资源空闲时，允许其他vNPU借用该时间片内浪费的资源，使卡上资源不被浪费。 |
| 争抢模式（`best-effort`） | vNPU之间自行争抢NPU资源，该模式的资源利用率最高，但无法保证vNPU的QoS。 |

调度策略对比如下。

**表2** 调度策略对比

| 策略名称 | 特点描述 |
| :--- | :--- |
| 紧凑调度（`binpack`） | 优先将Pod/vNPU集中分配到少数节点/设备上，以提高单节点/设备的资源利用率，降低碎片化。 |
| 打散调度（`spread`） | 优先将Pod/vNPU分散分配到更多节点/设备上，以提高负载均衡和故障隔离能力。 |

### 使用限制

- **软切分**：一个容器最多只能申请一个vNPU；一个容器只能有一个进程可以使用vNPU；只支持切分和限制AI Core及显存容量，不支持AI CPU、VPC、VDEC、JPEGD等核的切分控制；业务容器不允许为特权容器。
- **支持的NPU芯片型号**：310P3、910B4（软切分+硬切分均支持）；910C暂不支持（详见[附录-支持的NPU型号](#支持的npu型号)）。
- `huawei.com/vnpu-number`取值范围为1 ~ 单节点物理设备总数；大于1表示分配整卡；若为软/硬切分，则只能为1。
- `huawei.com/vnpu-cores`单位为1%；软切分`fixed-share`/`elastic`策略取值范围1~100，`best-effort`策略取值范围0~100；硬切分取值范围1~100；不配置则默认100。
- `huawei.com/vnpu-memory.1Gi`单位为1Gi；软切分不允许配置为0，不配置则默认使用单卡显存总量；硬切分无需配置。
- 910B芯片软/硬切分要求HDK驱动25.5.0及以上版本才支持共享模式。
- 业务Pod的`spec.schedulerName`必须为`volcano`。

>![image](figures/icon-notice.gif) **注意：**
>
>- 业务容器不允许为特权容器，否则软切分劫持库无法生效，资源隔离失效。
>- 软切分场景一个容器只允许一个进程使用vNPU，多进程并发使用同一vNPU可能导致AI Core时间片调度异常。
>- `huawei.com/vnpu-memory.1Gi`在软切分模式下不允许配置为0，否则Pod调度失败。

### 配置说明

Pod支持的全部注解与资源字段如下。

**表3** Pod vNPU配置字段说明

| 配置项 | 功能 | 取值说明 |
| :--- | :--- | :--- |
| `huawei.com/vnpu-mode` | 配置切分模式 | `soft`：软切分；`hard`：硬切分。可选，默认为`soft`。 |
| `huawei.com/vnpu-hard-aicpu-level` | 配置硬切分模式AI CPU分配策略 | `low`：低配；`high`：高配。可选，默认为`low`。 |
| `huawei.com/vnpu-soft-scheduler-policy` | 软切分算力调度模式 | `fixed-share`：固定配额；`elastic`：弹性模式；`best-effort`：争抢模式。可选，默认为`fixed-share`。 |
| `huawei.com/vnpu-pod-node-scheduler-policy` | 节点级调度策略 | `binpack`：紧凑调度，优先将Pod集中到少数节点；`spread`：打散调度，优先将Pod分散到更多节点。可选，默认为`binpack`。 |
| `huawei.com/vnpu-pod-device-scheduler-policy` | 设备级调度策略 | `binpack`：紧凑调度，优先将vNPU集中到少数NPU设备；`spread`：打散调度，优先将vNPU分散到更多NPU设备。可选，默认为`binpack`。 |
| `huawei.com/vnpu-use-npu-dieID` | 指定NPU卡的dieID | 取值为NPU卡的dieID，例如`"37E96E64-20C10A5A-8C747124-6E78030A-BF003019"`。dieID可查看节点注解`huawei.com/node-vnpu-register`的第2个字段，查看命令如下。可选，默认为空，匹配任意dieID。 |
| `huawei.com/vnpu-use-npu-type` | 指定NPU设备类型 | 取值为`"NPU-ASCEND-{芯片类型}"`，例如`"NPU-ASCEND-310P3"`。设备类型可查看节点注解`huawei.com/node-vnpu-register`的第5个字段，查看命令如下。可选，默认为空，匹配任意设备类型。 |
| `spec.schedulerName` | 指定调度器名称 | 必须为`volcano`。 |
| `huawei.com/vnpu-number` | vNPU数量 | 必选。取值范围：1 ~ 单节点物理设备总数。如果大于1，表示分配整卡；如果是软/硬切分，则只能为1。 |
| `huawei.com/vnpu-cores` | AI Core使用率 | 可选。单位为1%。不配置则默认为100。软切分`fixed-share`/`elastic`策略取值范围1~100；`best-effort`策略取值范围0~100；硬切分取值范围1~100。 |
| `huawei.com/vnpu-memory.1Gi` | 显存大小 | 可选。单位为1Gi。软切分不配置则默认使用单卡显存总量，不允许配置为0；硬切分无需配置。 |

查看节点注解`huawei.com/node-vnpu-register`中NPU设备详情（dieID、芯片类型等字段）的命令示例：

```bash
# 输出节点 vnpu-register 注解的完整内容
kubectl get node {nodename} -o jsonpath='{.metadata.annotations.huawei\.com/node-vnpu-register}'
# 输出字段之间以分号分隔，第2个字段为dieID，第5个字段为芯片类型
```

>![image](figures/icon-note.gif) **说明：**
>
>- `resources.limits`与`resources.requests`中相同字段的取值必须一致，否则kubelet校验失败。
>- 软切分模式申请整卡（`huawei.com/vnpu-number` > 1）不被支持，请使用硬切分或显式声明`huawei.com/vnpu-mode: "hard"`。
>- `huawei.com/vnpu-use-npu-dieID`与`huawei.com/vnpu-use-npu-type`配合使用可精确指定目标NPU；单独使用其中之一时，匹配维度仅限该字段。

### 配置样例

**样例1**：软切分固定配额模式，声明1个vNPU、50%的AI Core、8Gi显存。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-soft-scheduler-policy: "fixed-share"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {业务镜像}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
        huawei.com/vnpu-memory.1Gi: 8
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
        huawei.com/vnpu-memory.1Gi: 8
```

**样例2**：软切分争抢模式，只限制显存为8Gi，不限制AI Core。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-soft-scheduler-policy: "best-effort"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {业务镜像}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 0
        huawei.com/vnpu-memory.1Gi: 8
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 0
        huawei.com/vnpu-memory.1Gi: 8
```

**样例3**：硬切分低配模式，申请50%的AI Core。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"
    huawei.com/vnpu-hard-aicpu-level: "low"
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {业务镜像}
    resources:
      limits:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50  # AI Core百分比
      requests:
        huawei.com/vnpu-number: 1
        huawei.com/vnpu-cores: 50
```

**样例4**：声明使用两张整卡。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  schedulerName: volcano
  containers:
  - name: test-container
    image: {业务镜像}
    resources:
      limits:
        huawei.com/vnpu-number: 2
      requests:
        huawei.com/vnpu-number: 2
```

**样例5**：完整配置示例（同时设置切分模式、调度策略、指定dieID和设备类型）。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"                                                          # 切分模式
    huawei.com/vnpu-hard-aicpu-level: "low"                                               # 硬切分模式AI CPU分配策略
    huawei.com/vnpu-soft-scheduler-policy: "elastic"                                     # 软切分算力调度模式
    huawei.com/vnpu-pod-node-scheduler-policy: "spread"                                  # 节点级调度策略
    huawei.com/vnpu-pod-device-scheduler-policy: "binpack"                              # 设备级调度策略
    huawei.com/vnpu-use-npu-dieID: "37E96E64-20C10A5A-8C747124-6E78030A-BF003019"        # 指定NPU卡的dieID
    huawei.com/vnpu-use-npu-type: "NPU-ASCEND-310P3"                                      # 指定NPU设备类型
spec:
  schedulerName: volcano  # 指定调度器名称为volcano
  containers:
  - name: test-container
    image: {业务镜像}
    resources:
      limits:
        huawei.com/vnpu-number: 1        # 必选，vNPU数量
        huawei.com/vnpu-cores: 5        # 可选，容器可使用的aicore使用率
        huawei.com/vnpu-memory.1Gi: 8   # 可选，容器可使用的显存
      requests:
        huawei.com/vnpu-number: 1        # 和limits配置相同
        huawei.com/vnpu-cores: 5        # 和limits配置相同
        huawei.com/vnpu-memory.1Gi: 8   # 和limits配置相同
```

### 操作步骤

1. 设置节点切分模式。

   vNPU支持软切分和硬切分两种模式，可按节点粒度设置切分模式。节点默认为软切分模式。通过Helm部署npu-device-plugin服务时可通过`values.yaml`或Helm命令行参数设置节点切分模式。

   - 方式一：在`values.yaml`中设置节点切分模式，然后通过Helm部署。

     ```yaml
     vnpuNodeMode:
       worker1:        # 节点名
         mode: "hard"  # 节点切分模式，hard|soft，默认为soft
       worker2:
         mode: "soft"
     ```

   - 方式二：通过Helm命令行参数设置。

     ```bash
     helm upgrade --install vxpu ./vxpu-1.0.0.tgz -n xpu --set vnpuNodeMode.worker1.mode="hard"
     ```

     >![image](figures/icon-note.gif) **说明：**
     >
     >- 节点名（`worker1`）需与`kubectl get nodes`输出的节点名一致。
     >- `vnpuNodeMode`未配置的节点默认使用软切分（`soft`）模式。
     >- 同一节点切分模式不可与Pod注解`huawei.com/vnpu-mode`冲突，否则Pod调度失败。

2. 创建使用vNPU的Pod。

   业务Pod在`spec.schedulerName`指定为`volcano`后，通过`resources.limits`/`resources.requests`声明所需的vNPU数量、AI Core使用率、显存大小，通过`metadata.annotations`配置切分模式与调度策略。完整字段说明见[配置说明](#配置说明)，可用样例见[配置样例](#配置样例)。

### 后续操作

- 部署完成后，通过`kubectl get pod -n xpu`查看npu-client-update、npu-device-plugin、xpu-exporter组件运行状态，确认DaemonSet就绪。
- 检查节点注解`huawei.com/node-vnpu=ready`是否存在，以及节点注解`huawei.com/node-vnpu-register`中是否已上报NPU设备详情（dieID、芯片类型等字段）。查看命令见[配置说明](#配置说明)小节。
- 如需对接监控系统，请参见[vNPU指标监控](#相关操作)小节配置Prometheus抓取xpu-exporter。
- 如需调整某节点的切分模式，通过Helm重新部署时设置`vnpuNodeMode.<nodename>.mode`即可，无需重建节点。切分模式变更仅对新调度到该节点的Pod生效，已运行Pod不受影响；如需让已运行Pod应用新的切分模式，请重启对应Pod。

### 相关操作

xpu-exporter支持采集vNPU相关的监控指标，对接Prometheus实现对NPU资源的精细化监控。NPU整卡指标可安装昇腾MindCluster的NPU Exporter采集。vNPU支持采集的指标如下。

**表4** vNPU监控指标

| 指标名 | 说明 | label |
| :--- | :--- | :--- |
| `xpu_vnpu_mode` | vNPU的切分模式 | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_soft_scheduler_policy` | vNPU的软切分调度模式，硬切分不涉及 | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_mem_util` | 每个vNPU的内存使用率 | npuUUid, nodeName, nodeIp, podUid, cntrName, vnpuId, vnpuCoreLimit, vnpuMemLimit |
| `xpu_vnpu_num` | 每个NPU卡上分配的vNPU数量 | nodeName, nodeIp, npuUUid |
| `xpu_vnpu_pod_num` | 每个NPU上分配了vNPU的Pod数量 | nodeName, nodeIp, npuUUid |

>![image](figures/icon-note.gif) **说明：**
>
>- 上述指标由xpu-exporter以Prometheus exposition格式暴露在`/metrics`端点，默认侦听端口请以部署Chart的`values.yaml`为准。
>- NPU整卡指标（温度、功耗、整卡显存等）需另行安装昇腾MindCluster的NPU Exporter采集，与vNPU指标互补。
>- `xpu_vnpu_soft_scheduler_policy`仅在软切分模式下有值，硬切分节点不暴露此指标。

通过`kubectl describe node {nodename}`查看节点上`huawei.com/vnpu-*`资源的Capacity/Allocatable，确认NPU切分资源已正确注册。节点注解`huawei.com/node-vnpu-register`中包含每张NPU卡的dieID、芯片类型等详情，Pod可通过`huawei.com/vnpu-use-npu-dieID`和`huawei.com/vnpu-use-npu-type`精确指定落位。查看节点注解的命令见[配置说明](#配置说明)小节。

## 附录

### 代码结构

- **volcano-xpu-plugin**：Volcano插件，实现NPU虚拟化资源调度。
- **xpu-device-plugin**：设备插件，实现NPU资源上报和分配。
- **client_update**：负责软切分劫持库的部署。
- **xpu-exporter**：负责vNPU指标的上报。
- **ci**：编译构建脚本。
- **charts**：vNPU组件部署相关的文件。

### 支持的NPU型号

vNPU支持的NPU芯片型号及软件版本配套关系如下。

**表5** 支持的NPU型号及软件版本配套

| 芯片型号 | 软切分 | 硬切分 | HDK版本配套 | CANN版本配套（软切分） |
| :--- | :--- | :--- | :--- | :--- |
| 310P3 | 支持 | 支持 | 25.5.0及以上版本 | 8.5.0、9.1.0 |
| 910B4 | 支持 | 支持 | 25.5.0及以上版本 | 8.5.0、9.1.0 |
| 910C | 不支持 | 不支持 | NA | NA |

>![image](figures/icon-note.gif) **说明：**
>
>以上芯片型号及软件版本配套已经过充分验证，不在此范围内的软硬件未经验证，可能存在兼容性问题。310P3与910B4芯片软/硬切分均要求HDK驱动25.5.0及以上版本才支持设备共享模式。

### 支持的Volcano版本

当前支持Volcano 1.15.0版本，最新版本待规划。
