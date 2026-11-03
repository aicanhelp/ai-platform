## 编译

### 工具准备

安装必要的工具和软件：

- 工具：git、docker >= 20.10、helm、Go >= 1.25.0、gcc、cmake、make、patch、objcopy
- 软件：CANN、HDK（需与NPU型号匹配）

### 下载代码

```shell
git clone https://gitcode.com/openFuyao/vNPU.git
cd vNPU
git submodule update --init --recursive
```

### 本地构建

#### 设置 HDK 与 CANN 环境变量

本地构建前需要确保 HDK 和 CANN 的环境变量已正确设置。构建脚本 `build.sh` 会自动检测，但推荐在运行脚本前手动设置：

**CANN 环境变量设置**

```shell
# 通过 set_env.sh 设置，设置后 ASCEND_HOME_PATH 等环境变量自动生效
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

**HDK 环境变量设置**

```shell
# 驱动安装在默认路径时无需设置（脚本默认查找 /usr/local/Ascend）
# 驱动安装在非默认路径时，需指定 ENPU_ASCEND_DRIVER_PATH：
export ENPU_ASCEND_DRIVER_PATH=/usr/local/Ascend
```

**版本一致性要求（重要）**：构建环境的 CANN 与 HDK 版本应与目标 NPU 型号匹配，否则产物在目标 NPU 上可能出现兼容性问题（如运行异常、符号缺失、API 不匹配）。

```shell
cd vNPU

# 构建全部组件二进制 + 全部镜像 tar + chart包
./ci/build.sh

# 构建指定组件二进制 + 对应镜像 tar
./ci/build.sh acl-client           # 软切分劫持库 + 客户端工具
./ci/build.sh npu-device-plugin     # NPU 设备插件
./ci/build.sh xpu-exporter          # vNPU 监控导出器
./ci/build.sh volcano               # volcano 相关组件

# 构建多个指定组件
./ci/build.sh npu-device-plugin xpu-exporter

# 单独打chart包
./ci/build.sh chart
```

产物输出：
- 二进制/库文件：`ci/output/`
- 镜像 tar：`ci/output/images/`
- chart 包：`ci/output/chart/`

> 注意：
> - npu-device-plugin、xpu-exporter 的镜像名与 `charts/vnpu/values.yaml` 默认配置一致，镜像导入节点后可直接使用 chart 默认配置部署，无需修改镜像名。
> - acl-client 镜像名不带 NPU 型号后缀（`acl-client-update`），与 chart 默认镜像名（如 `acl-client-update-910b`）不一致，需自行重命名镜像或修改 chart 配置，处理方式见"离线部署"章节步骤5。
> - volcano 镜像为短名，与 `charts/yaml/volcano-deployment.yaml` 的镜像引用一致。

## 在线部署

### 前置条件

1. 已准备Kubernetes集群。
2. 已以root用户安装昇腾HDK驱动。

### 操作步骤

1. 部署Ascend Docker Runtime，建议使用7.2.RC1+版本。以7.2.RC1版本为例，详细部署方式请参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)。
2. 为需要部署vnpu组件的节点增加label：`kubectl label node {nodename} huawei.com/vnpu=ready`。
3. 创建xpu Namespace：`kubectl apply -f charts/yaml/namespace.yaml`。
4. 通过helm chart部署npu-client-update、npu-device-plugin、xpu-exporter：`helm install vnpu oci://cr.openfuyao.cn/charts/vnpu --version 0.0.0-latest`。

   > **注意**：Helm chart 中的 acl-client 镜像默认使用 **910B** 型号。如果使用其他型号的 NPU 卡，需通过 `--set` 指定对应镜像名或修改yaml中对应字段：
   > 
   > | NPU 型号 | `npuClientUpdate.image.name` 取值 |
   > |----------|-----------------------------------|
   > | 910B     | `acl-client-update-910b`（默认）  |
   > | 310P     | `acl-client-update-310p`          |
   > | 910C     | `acl-client-update-910c`          |
   > 
   > 例如，部署 310P 型号：
   > ```shell
   > helm install vnpu oci://cr.openfuyao.cn/charts/vnpu --version 0.0.0-latest --set npuClientUpdate.image.name=acl-client-update-310p
   > ```
5. 先使用root用户在主机OS上创建日志目录并设置权限（Volcano容器以UID 1000运行，需赋予目录写权限）：

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

   再通过charts/yaml/volcano-deployment.yaml部署volcano：

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```
6. 参考[昇腾官方文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531)开启设备共享模式。对于910b芯片，HDK驱动25.5.0及以上版本才支持共享模式。例如，设置设备0所有芯片的容器共享模式为使能：`npu-smi set -t device-share -i 0 -d 1`。

## 离线部署

离线部署的产物来源支持两种：发布的软件包，或本地构建产物（`./ci/build.sh` 产出，镜像tar位于 `ci/output/images/`，chart包位于 `ci/output/chart/`）。
chart拉取命令：
```shell
    helm pull oci://cr.openfuyao.cn/charts/vnpu --version ${version}
```

### 前置条件

1. 已准备Kubernetes集群。
2. 已以root用户安装昇腾HDK驱动。

### 操作步骤

1. 部署Ascend Docker Runtime，建议使用7.2.RC1+版本。以7.2.RC1版本为例，详细部署方式请参考[昇腾官方文档](https://www.hiascend.com/document/detail/zh/mindcluster/72rc1/clustersched/dlug/dlug_installation_017.html)。

2. 为需要部署vnpu组件的节点增加label：`kubectl label node {nodename} huawei.com/vnpu=ready`。

3. 创建xpu Namespace：`kubectl apply -f charts/yaml/namespace.yaml`。

4. 在每个需要部署vnpu组件的节点导入镜像包。镜像来源二选一：

   - 发布软件包：使用提前下载的镜像tar。
   - 本地构建产物：使用 `ci/output/images/` 目录下的镜像tar。

   以containerd本地镜像为例：
   ```shell
   # 镜像包名称请以实际为准（aarch64/x86_64按实际架构）
   ctr -n k8s.io images import acl-client-update-1.0.0-aarch64.tar
   ctr -n k8s.io images import npu-device-plugin-1.0.0-aarch64.tar
   ctr -n k8s.io images import xpu-exporter-1.0.0-aarch64.tar
   ctr -n k8s.io images import vc-controller-manager-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc-scheduler-1.15.0-aarch64.tar
   ctr -n k8s.io images import vc-webhook-manager-1.15.0-aarch64.tar
   ```

5. 通过helm chart部署npu-client-update、npu-device-plugin、xpu-exporter。chart包来源二选一：

   - 使用发布软件包中的chart包：

     ```shell
     helm upgrade --install vnpu vnpu-1.0.0.tgz -n xpu --wait
     ```

   - 使用本地构建产出的chart包：

     ```shell
     helm upgrade --install vnpu ci/output/chart/vnpu-1.0.0.tgz -n xpu --wait
     ```

   > **注意**：Helm chart 中的 acl-client 镜像默认使用 **910B** 型号。如果使用其他型号的 NPU 卡，需通过 `--set` 指定对应镜像名：
   > 
   > | NPU 型号 | `npuClientUpdate.image.name` 取值 |
   > |----------|-----------------------------------|
   > | 910B     | `acl-client-update-910b`（默认）  |
   > | 310P     | `acl-client-update-310p`          |
   > | 910C     | `acl-client-update-910c`          |
   > 
   > 例如，部署 310P 型号：
   > ```shell
   > helm upgrade --install vnpu vnpu-1.0.0.tgz -n xpu --wait --set npuClientUpdate.image.name=acl-client-update-310p
   > ```
   >
   > 本地构建的acl-client镜像不带NPU型号后缀（`acl-client-update`），与chart默认镜像名不一致，导入节点后需自行处理，任选其一：
   >
   > - 重命名镜像为chart默认镜像名（以910B为例，310P/910C对应修改）：
   >
   >   ```shell
   >   ctr -n k8s.io images tag cr.openfuyao.cn/openfuyao/vnpu/acl-client-update:1.0.0 cr.openfuyao.cn/openfuyao/vnpu/acl-client-update-910b:1.0.0
   >   ```
   >
   > - 或部署时通过 `--set` 修改chart镜像名（chart包来源按上文任选）：
   >
   >   ```shell
   >   helm upgrade --install vnpu ci/output/chart/vnpu-1.0.0.tgz -n xpu --wait --set npuClientUpdate.image.name=acl-client-update
   >   ```

6. 先使用root用户在主机OS上创建日志目录并设置权限（Volcano容器以UID 1000运行，需赋予目录写权限）：

   ```bash
   mkdir -p /var/log/volcano-{admission,controller,scheduler}
   chown -R 1000:1000 /var/log/volcano-{admission,controller,scheduler}
   ```

   将charts/yaml/volcano-deployment.yaml中镜像版本${version}修改为具体发布版本后，再通过charts/yaml/volcano-deployment.yaml部署volcano：

   ```bash
   kubectl apply -f charts/yaml/volcano-deployment.yaml
   ```

7. 参考[昇腾官方文档](https://support.huawei.com/enterprise/zh/doc/EDOC1100540371/d18b0230?idPath=23710424|251366513|254884019|261408772|261457531)开启设备共享模式。对于910b芯片，HDK驱动25.5.0及以上版本才支持共享模式。例如，设置设备0所有芯片的容器共享模式为使能：`npu-smi set -t device-share -i 0 -d 1`。

### 设置节点切分模式

vNPU支持软切分和硬切分两种模式，可按节点粒度设置，默认为软切分。部署npu-device-plugin时，可通过values.yaml或Helm命令行参数指定节点切分模式。

在values.yaml中设置：

```yaml
vnpuNodeMode:
  worker1:  # 节点名
    mode: "hard"  # 切分模式，hard|soft，默认为soft
  worker2:
    mode: "soft"
```

或通过Helm命令行参数设置：

```shell
helm upgrade --install vnpu ./vnpu-1.0.0.tgz -n xpu --set vnpuNodeMode.worker1.mode="hard"
```

## Pod使用NPU

### 配置说明

业务Pod只需声明所需的vNPU资源，无需修改业务代码，vNPU的调度器和device plugin负责自动调度Pod到合适的节点合适的NPU设备上并挂载设备。

支持Pod设置切分模式，按照vNPU的数量、AICore、显存申请资源等。

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
    huawei.com/vnpu-mode: "hard"  # 切分模式
    huawei.com/vnpu-hard-aicpu-level: "low" # 硬切分模式AI CPU分配策略
    huawei.com/vnpu-soft-scheduler-policy: "elastic"  # 软切分算力调度模式
    huawei.com/vnpu-pod-node-scheduler-policy: "spread"  # 节点级调度策略
    huawei.com/vnpu-pod-device-scheduler-policy: "binpack"  # 设备级调度策略
    huawei.com/vnpu-use-npu-dieID: "37E96E64-20C10A5A-8C747124-6E78030A-BF003019"  # 指定NPU卡的dieID
    huawei.com/vnpu-use-npu-type: "NPU-ASCEND-310P3"  # 指定NPU设备类型
spec:
    schedulerName: volcano  # 指定调度器名称为volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 1  # 必选，vnpu数量。
            huawei.com/vnpu-cores: 5  # 可选，容器可使用的aicore使用率
            huawei.com/vnpu-memory.1Gi: 8 # 可选，容器可使用的显存
          requests:
            huawei.com/vnpu-number: 1  # 和limits配置相同
            huawei.com/vnpu-cores: 5   # 和limits配置相同
            huawei.com/vnpu-memory.1Gi: 8 # 和limits配置相同
```

配置说明：

| 配置项                                      | 功能                        | 取值说明                                                     |
| ------------------------------------------- | --------------------------- | ------------------------------------------------------------ |
| huawei.com/vnpu-mode                        | 配置切分模式                | soft: 软切分<br/>hard: 硬切分<br/>可选，默认为soft           |
| huawei.com/vnpu-hard-aicpu-level            | 配置硬切分模式AICPU分配策略 | low: 低配<br/>high: 高配<br/>可选，默认为low                 |
| huawei.com/vnpu-soft-scheduler-policy       | 软切分算力调度模式          | fixed-share: 固定配额<br/>elastic: 弹性模式<br/>best-effort: 争抢模式<br/>可选，默认为fixed-share |
| huawei.com/vnpu-pod-node-scheduler-policy   | 节点级调度策略              | binpack: 紧凑调度，优先将Pod集中到少数节点<br/>spread: 打散调度，优先将Pod分散到更多节点<br/>可选，默认为binpack |
| huawei.com/vnpu-pod-device-scheduler-policy | 设备级调度策略              | binpack: 紧凑调度，优先将vNPU集中到少数NPU设备<br/>spread: 打散调度，优先将vNPU分散到更多NPU设备<br/>可选，默认为binpack |
| huawei.com/vnpu-use-npu-dieID              | 指定NPU卡的dieID           | 取值为NPU卡的dieID，例如"37E96E64-20C10A5A-8C747124-6E78030A-BF003019"。<br/>dieID可查看node注解huawei.com/node-vnpu-register的第2个字段。<br/>可选，默认为空，匹配任意dieID。 |
| huawei.com/vnpu-use-npu-type               | 指定NPU设备类型             | 取值为"NPU-ASCEND-{芯片类型}"，例如"NPU-ASCEND-310P3"。<br/>设备类型可查看node注解huawei.com/node-vnpu-register的第5个字段。<br/>可选，默认为空，匹配任意设备类型。 |
| spec.schedulerName                          | 指定调度器名称              | 必须为volcano                                                |
| huawei.com/vnpu-number                      | vnpu数量                    | 必选。取值范围：1 ~ 单节点物理设备总数。如果大于1，表示分配整卡。如果是软/硬切分，则只能为1。 |
| huawei.com/vnpu-cores                       | aicore使用率                | 可选。单位为1%。不配置则默认为100。对于软切分，fixed-share和elastic策略取值范围为1~100；对于best-effort策略，取值范围为0~100。对于硬切分，取值范围为1~100。 |
| huawei.com/vnpu-memory.1Gi                  | 显存大小                    | 可选。单位为1Gi。对于软切分，不配置则默认表示使用单卡显存总量，不允许配置为0。对于硬切分，无需配置。 |

软切分算力调度模式说明：

| 模式名称                    | 特点描述                                                     |
| :-------------------------- | :----------------------------------------------------------- |
| 固定配额模式（fixed-share） | 各vNPU按配比严格执行时间片。 a. 若有暂未分配的vNPU资源，则在最后一个vNPU时间片消耗完时，睡眠一段时间。此时NPU将闲置空转。 b. 某vNPU上无任务运行，仍会按照给定比例消耗完该vNPU的时间片。 |
| 弹性模式（elastic）         | 由于固定配额模式可能出现NPU资源浪费，在此模式下，在各vNPU按配比执行时间片的基础上，为提高卡资源利用率，优化了调度逻辑。 a. 最后一个vNPU时间片消耗完，马上将NPU使用权释放给下一个vNPU，跳过睡眠逻辑。 b. 在某vNPU上无任务运行，则直接跳过未消耗完的时间片，切换至另一个vNPU。 c. 时间片借用机制：正在执行的vNPU在识别到卡资源空闲的情况下，允许其他vNPU借用该时间片内浪费的资源，使卡上资源不被浪费，进而提升整卡利用率。 |
| 争抢模式（best-effort）     | vNPU之间自行争抢NPU资源，该模式的资源利用率最高，但无法保证vNPU的QoS。 |

调度策略说明：

| 策略名称 | 特点描述 |
| :------------- | :----------------------------------------------------------- |
| 紧凑调度（binpack） | 优先将Pod/vNPU集中分配到少数节点/设备上，以提高单节点/设备的资源利用率，降低碎片化。 |
| 打散调度（spread） | 优先将Pod/vNPU分散分配到更多节点/设备上，以提高负载均衡和故障隔离能力。 |

### 配置样例

例如，Pod使用软切分固定配额模式，声明1个vNPU，50%的AICore，8G显存：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
spec:
    schedulerName: volcano
    containers:
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

使用软切分争抢模式，只限制显存为8G，不限制AICore：

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

使用硬切分，申请50%的AICore：

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
        resources:
          limits:
            huawei.com/vnpu-number: 1  
            huawei.com/vnpu-cores: 50  # AI Core百分比
          requests:
            huawei.com/vnpu-number: 1
            huawei.com/vnpu-cores: 50
```

Pod声明使用两张整卡：

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: testpod
  annotations:
spec:
    schedulerName: volcano
    containers:
        resources:
          limits:
            huawei.com/vnpu-number: 2 
          requests:
            huawei.com/vnpu-number: 2
```

## vNPU指标监控

xpu-exporter支持采集vNPU监控指标并对接Prometheus，实现对NPU资源的精细化监控。NPU整卡指标可安装昇腾MindCluster的NPU Exporter采集。

支持的指标：

| 指标名                         | 说明                               | label                                                        |	 
| ------------------------------ | ---------------------------------- | ------------------------------------------------------------ |	 
| xpu_vnpu_mode                  | vNPU的切分模式                     | nodeName,nodeIp,npuUUid                                      |	 
| xpu_vnpu_soft_scheduler_policy | vNPU的软切分调度模式，硬切分不涉及 | nodeName,nodeIp,npuUUid                                      |	 
| xpu_vnpu_mem_util              | 每个vnpu的内存使用率               | npuUUid,nodeName,nodeIp,podUid,cntrName,vnpuId,vnpuCoreLimit,vnpuMemLimit |	 
| xpu_vnpu_num                   | 每个NPU卡上分配的vnpu数量          | nodeName,nodeIp,npuUUid                                      |	 
| xpu_vnpu_pod_num               | 每个NPU上分配了vnpu的pod数量       | nodeName,nodeIp,npuUUid                                      |
