# NPUClusterPolicy CRD 参考

> **Note**：`NPUClusterPolicy` 是 npu-operator 的**唯一入口 API**——集群级、单例（名字必须为 `cluster`）。Spec 共 17 个顶层段：operator 自身 + DaemonSet 公共配置 + 14 个必需的组件段 + 3 个可选的功能段（`volcano` / `vnpu` / `dra`）。
>
> 架构背景见 [architecture.md](architecture.md)。本文字段说明均以 `api/v1/npuclusterpolicy_types.go` 与生成的 CRD 为准。

---

## 1. CRD 元信息

| 项 | 值 |
|---|---|
| Group / Version / Kind | `npu.openfuyao.com` / `v1` / `NPUClusterPolicy` |
| 作用域 | **Cluster**（非命名空间，`+kubebuilder:resource:scope=Cluster`） |
| 单例约束 | CEL 校验：`self.metadata.name == 'cluster'` |
| Status 子资源 | 启用 |
| 打印列 | `Status`（`.status.phase`）、`Age` |
| 短名 | 无 |
| 类型定义 | `api/v1/npuclusterpolicy_types.go` |
| 生成的 CRD | `config/crd/bases/npu.openfuyao.com_npuclusterpolicies.yaml` 与 `charts/npu-operator/crds/npu.openfuyao.com_npuclusterpolicies.yaml`（内容一致，约 2.7 万行，因完整内嵌 K8s Volume/EnvVar 等 schema） |

### 单例约束的实现

```yaml
x-kubernetes-validations:
  - rule: self.metadata.name == 'cluster'
    message: Only one NPUClusterPolicy with name 'cluster' can be created in a cluster
```

由 API Server 侧 CEL 校验直接拒绝，无需 operator 介入。整个集群只有一份策略；operator 的 Reconcile 以该单例为唯一调和对象。

## 2. Spec 顶层结构

```yaml
spec:
  operator:       OperatorSpec     # operator 行为（RuntimeClass 名）
  daemonsets:     DaemonsetsSpec   # 所有 DaemonSet 的公共配置
  driver:         DriverSpec       # 1. NPU 驱动            [必需]
  ociRuntime:     OCIRuntimeSpec   # 2. 容器运行时           [必需]
  volcano:        VolcanoSpec      # 3. Volcano flavor/运行身份/admission [可选]
  vcscheduler:    SchedulerSpec    # 4. Volcano 调度器       [必需]
  vccontroller:   ControllerSpec   # 5. Volcano 控制器       [必需]
  trainer:        TrainerSpec      # 6. AscendJob 训练算子   [必需]
  devicePlugin:   DevicePluginSpec # 7. 设备插件            [必需]
  nodeD:          NodeDSpec        # 8. 节点守护            [必需]
  exporter:       ExporterSpec     # 9. 监控导出            [必需]
  clusterd:       ClusterDSpec     # 10. ClusterD           [必需]
  rscontroller:   RSControllerSpec # 11. 韧性控制器          [必需]
  mindiotft:      MindIOTFTSpec    # 12. MindIO 容错        [必需]
  mindioacp:      MindIOACPSpec    # 13. MindIO Checkpoint  [必需]
  vnpu:           VNPUSpec         # 14. vNPU 组件组         [可选，默认不管理]
  dra:            DRASpec          # 15. DRA 组件组          [可选，默认不管理]
```

**必需性**：除 `volcano`、`vnpu`、`dra`（带 `omitempty`）外，其余 14 段在生成的 CRD 中均为 required（`spec` 本身可整体省略，但省略时各段取零值而非默认值）。各段内部字段基本都有默认值，最小合法实例只需 `metadata.name: cluster`。生产环境通常由 Helm 从 `values.yaml` 渲染完整实例（`charts/npu-operator/templates/npuclusterpolicy.yaml`）。

## 3. 公共子结构

以下结构被多个组件 Spec 复用。

### 3.1 `ImageSpec` —— 镜像描述

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `registry` | string | `""` | 镜像仓库地址 |
| `repository` | string | `""` | 仓库路径 |
| `tag` | string | `""` | 标签 |
| `imagePullPolicy` | string | `IfNotPresent` | 拉取策略 |
| `imagePullSecrets` | []string | `[]` | 拉取密钥名列表 |

**`Name()` 拼接规则**（`ImageSpec.Name()`）：

| registry | tag | repository | 结果 |
|---|---|---|---|
| 空 | 空 | 非空 | `repository` |
| 空 | 非空 | 非空 | `repository:tag` |
| 非空 | 非空 | 非空 | `registry/repository:tag` |
| 非空 | 空 | 非空 | `registry/repository:`（以冒号结尾，无效） |
| 任意 | 任意 | 空 | `""`（空串，transformer 跳过注入，保留清单默认镜像） |

### 3.2 `CommandSpec` —— 启动命令

```yaml
commandSpec:
  command: ["/bin/sh", "-c"]   # 覆盖容器 ENTRYPOINT（可选）
  args:    ["--foo=bar"]       # 覆盖容器 CMD（可选）
```

部分组件（如 DRA kubelet-plugin）在用户未提供 commandSpec 时由 operator 生成默认参数。

### 3.3 `PlacementSpec` —— 调度位置（inline）

以内联（`json:",inline"`）方式出现在大多数组件 Spec 中：

| 字段 | 说明 |
|---|---|
| `nodeSelector` | 附加到 operand Pod 的节点选择器（与清单已有 selector 合并） |
| `tolerations` | 附加到 operand Pod 的容忍（追加，不清除清单原有项） |

**注意**：device-plugin / vnpu-device-plugin / dra-kubelet-plugin / vnpu-client-update / dra-vcannrt-installer 的 `nodeSelector` 参与组件冲突检测（见 [architecture.md §7](architecture.md#7-组件冲突检测)），需保证互斥组件选中不重叠的节点组。

### 3.4 `ResourceRequirements` —— 资源配额

```yaml
resources:
  limits:   {cpu: "2", memory: "3Gi"}
  requests: {cpu: "100m", memory: "100Mi"}
```

直接复用 `corev1.ResourceList`，支持任意 K8s 资源名。

### 3.5 `VolumesSpec` —— 卷扩展

```yaml
volumeSpec:
  volumes:      []corev1.Volume        # 完整 K8s Volume schema
  volumeMounts: []corev1.VolumeMount   # 完整 K8s VolumeMount schema
```

### 3.6 `LogRotate` —— 日志轮转

| 字段 | 类型 | 默认 | 范围 | 说明 |
|---|---|---|---|---|
| `logLevel` | enum | `info` | debug/info/warning/error/critical | 映射数值 -1~3 |
| `logFile` | string | 各组件不同 | — | 日志文件路径 |
| `rotate` | int32 | `30` | 1~30 | 轮转保留份数 |
| `maxAge` | int32 | `7` | 7~700 | 保留天数 |
| `compress` | bool | `false` | — | 是否压缩 |

数值映射：`debug=-1  info=0  warning=1  error=2  critical=3`（`LogLevel.Value()`）。

配置通过环境变量注入容器：`NPU_OPERATOR_LOG_LEVEL` / `NPU_OPERATOR_LOG_FILE` / `NPU_OPERATOR_ROTATE` / `NPU_OPERATOR_MAX_AGE` / `NPU_OPERATOR_COMPRESS`。

**CEL 限制**：`devicePlugin`、`nodeD`、`exporter`、`driver`、`clusterd`、`rscontroller` 六个组件的 `logRotate` 带 XValidation `!has(self.compress) || !self.compress`，即**不允许开启 compress**（底层组件不支持）；`trainer` 不受限。

### 3.7 `ServiceMonitorSpec`

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `enabled` | bool | `false` | 是否创建 Prometheus ServiceMonitor（当前仅 vNPU exporter 支持） |

## 4. 各组件 Spec 详解

除特别说明外，各组件均支持：`managed`（默认 `true`）、`imageSpec`、`env`、`commandSpec`、`volumeSpec`、`resources`、`logRotate`、inline `placement` 的子集。

### 4.1 `operator`（OperatorSpec）

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `runtimeClass` | string | `ascend` | RuntimeClass 名称与 handler，operator 据此创建 RuntimeClass 资源 |
| `imageSpec` | ImageSpec | `{}` | 当前 controller 不消费该字段；operator 自身镜像由 Helm 在 `values.yaml` 的 `images.operator` 中配置 |

### 4.2 `daemonsets`（DaemonsetsSpec）—— DaemonSet 公共配置

| 字段 | 说明 |
|---|---|
| `labels` / `annotations` | 合并到所有 DaemonSet 及其 Pod 模板 |
| `nodeSelector` | **复制合并**到所有 DaemonSet Pod 模板的 nodeSelector |
| `tolerations` | **追加**到所有 DaemonSet Pod 模板 |
| `imageSpec` | 当前 controller 不消费该字段（各组件镜像独立配置） |

### 4.3 `driver`（DriverSpec）

| 字段 | 说明 |
|---|---|
| `managed` / `imageSpec` / `env` / `logRotate` / `placement` | 常规 |
| `initImageSpec` | init 容器镜像（**真正执行驱动安装的容器**） |

`logRotate.logFile` 默认 `/var/log/mindx-dl/driver/driver.log`；禁 compress。常用 env：`DRIVER_VERSION`、`HOST_DRIVER_SOURCE_PATH`。

### 4.4 `ociRuntime`（OCIRuntimeSpec）

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `managed` | 默认 `true` | |
| `interval` | 默认 `300`，范围 **60~1800** | 健康检查间隔（秒） |
| `imageSpec` | — | 主容器镜像（inject） |
| `initRuntimeImageSpec` | — | init 容器：ascend-docker-runtime |
| `initConfigImageSpec` | — | init 容器：npu-container-toolkit |
| `placement` | inline | |

### 4.5 `devicePlugin`（DevicePluginSpec）

全量字段：`managed` + `placement` + `initImageSpec` + `imageSpec` + `env` + `commandSpec` + `volumeSpec` + `resources` + `logRotate`（logFile 默认 `/var/log/mindx-dl/devicePlugin/devicePlugin.log`，禁 compress）。

### 4.6 `nodeD`（NodeDSpec）

除常规字段外，两个专属调优参数：

| 字段 | 默认 | 范围 | 说明 |
|---|---|---|---|
| `heartbeatInterval` | `5` | 1~300 | 心跳上报间隔（秒） |
| `pollInterval` | `60` | 60~600 | 硬件故障扫描间隔（秒） |

`logRotate.logFile` 默认 `/var/log/mindx-dl/noded/noded.log`，禁 compress。`nodeD.managed` 切换会同步增删**所有节点**的 `nodeDEnable=on` 标签。

### 4.7 `exporter`（ExporterSpec）

常规字段 + `logRotate`（logFile 默认 `/var/log/mindx-dl/npu-exporter/npu-exporter.log`，禁 compress）。

### 4.8 `volcano`（VolcanoSpec）—— flavor 与公共配置

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `flavor` | enum `mindcluster`/`vnpu`/`external`，**缺省 mindcluster** | 决定管理哪组 Volcano 资产（组件矩阵见 [architecture.md §3.2](architecture.md#32-volcano-组件按-flavor-加载57-个)） |
| `runAsUser` / `runAsGroup` | *int64 | 覆盖 Volcano 工作负载 Pod 及全部容器的运行身份（uid=0 时同时设置 `runAsNonRoot=false`） |
| `admission.enabled` | *bool，缺省时仅 vnpu flavor 视为 true | 是否管理 vnpu-admission（仅 vnpu flavor 下生效） |
| `admission.imageSpec` / `initImageSpec` | — | admission 主/ init 镜像（init 镜像缺省取 busybox，可用 operator 环境变量覆盖） |
| `admission.log.{dir,file,level}` | — | admission 日志目录/文件/级别 |

### 4.9 `vcscheduler`（SchedulerSpec）

| 字段 | 说明 |
|---|---|
| `managed` / `initImageSpec` / `imageSpec` / `env` / `commandSpec` / `volumeSpec` / `resources` / `placement` | 常规 |
| `schedulerConfigMap` | **完整的 volcano-scheduler.conf 文本**（string 逃逸舱）。operator 写入 ConfigMap `volcano-scheduler-configmap`；mindcluster flavor 下 `volcano-npu` 插件名会被按镜像版本+节点架构改写为 `volcano-npu_<version>_<arch>` |

注：mindcluster flavor 下本开关控制标准 `volcano-scheduler` Deployment；vnpu flavor 下同时控制 vNPU 版 scheduler；external flavor 下不管理任何 scheduler。

### 4.10 `vccontroller`（ControllerSpec）

与 SchedulerSpec 结构相同（无 schedulerConfigMap）。开关的 flavor 语义同上（对应 volcano-controllers / vNPU vc-controller-manager）。

### 4.11 `trainer`（TrainerSpec）

常规字段 + `logRotate`（默认 logFile `/var/log/mindx-dl/ascend-operator/ascend-operator.log`；**允许 compress**）。此组件为 AscendJob 训练算子（Deployment `ascend-operator-manager`），Chart 同时安装 `ascendjobs.crd.yaml`。

### 4.12 `clusterd`（ClusterDSpec）

常规字段 + `logRotate`（logFile 默认 `/var/log/mindx-dl/clusterd/clusterd.log`，禁 compress）。

### 4.13 `rscontroller`（RSControllerSpec）

字段精简：`managed` + `logRotate` + `resources` + `imageSpec` + `commandSpec` + `placement`（**无 env / volumeSpec**；logFile 默认 `/var/log/mindx-dl/resilience-controller/run.log`，禁 compress）。

### 4.14 `mindiotft`（MindIOTFTSpec）

极简：`managed` + `imageSpec` + `env` + `placement`。本质是安装脚本容器（hostPID/hostNetwork），无 resources/logRotate/commandSpec。

> **Note**：mindio 清单默认不含 nodeSelector，会调度到所有节点（含纯 CPU 节点），容器内脚本通过检测 `/opt/openFuyao/mindio/*.zip` 决定是否真正安装；可用 `placement.nodeSelector` 收敛范围。

### 4.15 `mindioacp`（MindIOACPSpec）

在 MindIOTFTSpec 基础上增加：

| 字段 | 说明 |
|---|---|
| `version` | MindIO ACP 版本号（如 `7.3.0`）。注意：当前 operator 不消费该字段，实际版本由容器内脚本自行检测 |

### 4.16 `vnpu`（VNPUSpec）—— vNPU 组件组

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `managed` | **默认 `false`** | 总开关；开启后 NPU 节点被打上 `huawei.com/vnpu=ready` 标签 |
| `placement` | inline | 组内组件共享的 placement |
| `updateStrategy` | string | DaemonSet 更新策略（如 `RollingUpdate`） |
| `priorityClassName` | string | vNPU Pod 优先级类 |
| `nodeModeConfig` / `virtualTemplateConfig` | string | 写入 `vnpu.node-mode.config` / `vnpu.virtual-template.config` ConfigMap |
| `clientUpdate` | VNPUComponentSpec（子开关默认 true） | 客户端更新 DaemonSet |
| `devicePlugin` | VNPUDevicePluginSpec（默认 true） | vNPU 设备插件 |
| `devicePlugin.deviceSplitCount` | **默认 `20`** | 每物理 NPU 暴露的 vNPU 设备数 |
| `devicePlugin.loggingConsole` | 默认 `true` | 是否输出到控制台 |
| `exporter` | VNPUExporterSpec（**子开关默认 false**） | vNPU exporter |
| `exporter.port` | 默认 `8082` | 指标端口 |
| `exporter.https` | 默认 `false` | 是否启用 HTTPS |
| `exporter.serviceMonitor.enabled` | 默认 `false` | 创建 ServiceMonitor |

`VNPUComponentSpec` 含：`managed` + `imageSpec` + `env` + `commandSpec` + `volumeSpec` + `resources` + `placement`。

### 4.17 `dra`（DRASpec）—— DRA 组件组

| 字段 | 类型/默认 | 说明 |
|---|---|---|
| `managed` | **默认 `false`** | 总开关 |
| `driverName` | 默认 `npu.huawei.com` | DRA 驱动名，用于 DeviceClass 名称与 CEL selector 改写 |
| `deviceProfile` | 默认 `npu` | DRA driver 的 device profile |
| `cdiRoot` | 默认 `/var/run/cdi` | CDI spec 根目录 |
| `healthcheckPort` | 默认 `-1`（禁用） | 健康检查端口，<0 关闭 |
| `numDevices` | — | gpu profile 生成的设备数 |
| `kubeconfig` / `kubeAPIQPS` / `kubeAPIBurst` / `featureGates` | — | 透传给 DRA driver |
| `enableHardVNPU` | — | **已废弃**：硬 vNPU 模式由 DRA profile 配置或节点标签决定 |
| `kubeletPluginsDirectoryPath` | 默认 `/var/lib/kubelet/plugins` | kubelet 插件目录 |
| `kubeletRegistrarDirectoryPath` | 默认 `/var/lib/kubelet/plugins_registry` | registrar 目录 |
| `placement` | inline | 组内共享 placement |
| `kubeletPlugin` | DRAComponentSpec（默认 true） | DaemonSet `npu-dra-kubeletplugin` |
| `deviceClasses.managed` | 默认 `true` | 是否管理 DeviceClass |
| `deviceClasses.templates[]` | `{name, selectorCEL}` | 自定义 DeviceClass 模板（name 与 CEL 表达式均必填，由 operator 调和时校验，非 CRD schema 强制） |
| `softVNPU` | DRASoftVNPUSpec（默认 true） | 软 vNPU 支持（ConfigMap + 卷） |
| `softVNPU.shareCount` | 默认 `16` | 每物理 NPU 的软共享实例数 |
| `softVNPU.softShareShmDir` | 默认 `/shared` | 共享 shm 目录 |
| `softVNPU.softShareDevConfigDir` | 默认 `/etc/enpu` | 设备配置宿主机目录 |
| `softVNPU.npuProfileConfigPath` | 默认 `/etc/dra/dra.config` | 容器内 profile 配置路径 |
| `softVNPU.profileConfig` | string | `dra.config` ConfigMap 内容 |
| `softVNPU.chipCapabilitiesConfigMap` / `chipCapabilitiesNamespace` | 默认 `vnpu-template-config` / — | capabilities ConfigMap 引用 |
| `softVNPU.chipCapabilities` | string | `capabilities.yaml` ConfigMap 内容 |
| `softVNPU.vcannrtInstaller` | DRAComponentSpec（默认 true） | vCANN-RT 安装 DaemonSet |
| `webhook` | DRAWebhookSpec（默认 true） | DRA 校验 webhook（Deployment `dra-webhook` + Service + ValidatingWebhookConfiguration） |
| `webhook.port` | 默认 `443` | webhook 监听端口 |
| `webhook.tlsCertFile` / `tlsPrivateKeyFile` | 默认 `/tls/tls.crt` / `/tls/tls.key` | 证书路径 |
| `webhook.caBundle` | string | 注入 ValidatingWebhookConfiguration 的 CA bundle |
| `webhook.failurePolicy` | enum `Ignore`/`Fail`，默认 `Fail` | webhook 失败策略 |

内置 DeviceClass（`dra/deviceclasses` 资产，驱动名可被 `driverName` 改写）：`full.npu.huawei.com`、`elastic.npu.huawei.com`、`fixed.npu.huawei.com`、`best-effort.npu.huawei.com`、`hard.npu.huawei.com`。operator 每次调和会对带 `openfuyao.com/dra-deviceclass-managed=true` 注解且 OwnerReference 匹配的 DeviceClass 做垃圾回收（含关联 ResourceSlice）。

`DRAComponentSpec` 含：`managed` + `imageSpec` + `env` + `commandSpec` + `volumeSpec` + `resources` + `placement`。

## 5. Status 结构

```yaml
status:
  phase: Ready | NotReady
  namespace: npu-operator-system        # operator 安装命名空间
  conditions:                           # 标准 metav1.Condition
    - type: Ready                       # Ready 与 Error 永远一真一假
      status: "True"
      reason: Reconciled                # 见下表
      message: "..."
    - type: Error
      status: "False"
      ...
  componentStatuses:                    # 最多 21 项，逐组件
    - name: /var/lib/npu-operator/components/driver
      state:
        type: running                   # pending|deploying|running|terminated|unmanaged
        reason: Reconciled
        message: ""
      prevState:                        # 上一状态（调试用）
        type: deploying
```

### 5.1 整体 Phase 与 Conditions

| Phase | 进入条件 |
|---|---|
| `Ready` | 所有被管理组件 `running` 或 `unmanaged`（无 NPU 节点的集群也可能 Ready，见 reason） |
| `NotReady` | 任一组件调和出错 / 非 running / 检测到组件冲突 |

常见 `reason`：`Reconciled`（全部调和成功）、`Reconciling`、`ReconcileFailed`、`NoNFDLabels`（Ready，缺 NFD 标签）、`NoNPUNodes`（Ready，无受支持 CRI 的 NPU 节点）、`ComponentConflict`（NotReady，组件选择器重叠，**不会自动重试**，需人工修改 placement）、`ComponentNotReady`。

### 5.2 组件五态（ComponentPhase）

| Phase | 含义 |
|---|---|
| `pending` | 待部署 / 部署失败待重试（如 `PodNotRunning`） |
| `deploying` | 正在部署（调和开始时先置此态） |
| `running` | 已部署且就绪（**集群无 NPU 节点时跳过创建的 DaemonSet 也保持此态**） |
| `terminated` | 已终止待重启（inspect 出错） |
| `unmanaged` | 用户关闭管理，资源已回收 |

多个资源状态合并时按严重度取最高：`pending > terminated > deploying > running > unmanaged`。

### 5.3 `componentStatuses` 名称映射

状态项名称默认为组件目录全路径；vNPU / DRA / Volcano-vnpu 子组件使用去前缀短名；部分基础支撑组件不产生状态项：

| 组件目录 | 状态项名称 |
|---|---|
| driver / oci-runtime / device-plugin / trainer / noded / clusterd / resilience-controller / npu-exporter / mindio/mindiotft / mindio/mindioacp / volcano/volcano-controller / volcano/volcano-scheduler | `/var/lib/npu-operator/components/<原路径>` |
| `vnpu/client-update` | `/var/lib/npu-operator/components/vnpu-client-update` |
| `vnpu/device-plugin` | `…/vnpu-device-plugin` |
| `vnpu/exporter`（含 exporter-servicemonitor） | `…/vnpu-exporter` |
| `volcano/vnpu-controller` / `volcano/vnpu-scheduler` / `volcano/vnpu-admission` | `…/volcano-vnpu-controller` 等 |
| `dra/kubelet-plugin` / `dra/vcannrt-installer` / `dra/webhook` | `…/dra-kubelet-plugin` 等 |
| `vnpu/namespace`、`volcano/vnpu-crds`、`volcano/vnpu-namespaces`、`dra/deviceclasses`、`dra/soft-vnpu` | **无状态项** |

状态列表每次调和按组件加载顺序重建，仅在内容变化时才写 API。

## 6. 字段变更的传播路径速查

| 修改 CR 的… | operator 的反应 |
|---|---|
| `spec.<X>.managed: true→false` | 组件资源按 canDelete（OwnerReference 匹配本实例）逐个删除；仍被其他受管组件引用的资源被跳过；状态置 `unmanaged` |
| `spec.volcano.flavor` | 切换 Volcano 资产组；旧 flavor 由本 CR 创建的资源被回收；旧 flavor 默认镜像不会被误用（防串扰 guard） |
| `spec.<X>.imageSpec.*` | transformer 重算镜像名 → `CreateOrUpdate` → 滚动升级；结果哈希更新到 `openfuyao.com/last-applied-hash` |
| `spec.<X>.env` / `resources` / `volumeSpec` / `commandSpec` | 按名替换/追加注入对应容器字段 → 更新工作负载 |
| `spec.<X>.placement.nodeSelector` | 合并进 Pod 模板；**device-plugin 类组件变更会重新触发冲突检测** |
| `spec.vcscheduler.schedulerConfigMap` | 重写 ConfigMap `volcano-scheduler-configmap`（含按架构改写插件名） |
| `spec.nodeD.managed` | 同步增删所有节点的 `nodeDEnable=on` 标签 |
| `spec.vnpu.managed` | 同步增删 NPU 节点的 `huawei.com/vnpu=ready` 标签 |
| `spec.dra.deviceClasses.templates` | 渲染自定义 DeviceClass；回收不再期望的托管 DeviceClass 及其 ResourceSlice |

## 7. 常见操作

```bash
# 查看整体状态
kubectl get npuclusterpolicy cluster -o jsonpath='{.status.phase}'

# 列出所有组件状态
kubectl get npuclusterpolicy cluster \
  -o jsonpath='{range .status.componentStatuses[*]}{.name}{"\t"}{.state.type}{"\t"}{.state.reason}{"\n"}{end}'

# 临时关闭某组件（如 npu-exporter）
kubectl patch npuclusterpolicy cluster --type=merge -p '{"spec":{"exporter":{"managed":false}}}'

# 修改某组件镜像 tag
kubectl patch npuclusterpolicy cluster --type=merge \
  -p '{"spec":{"devicePlugin":{"imageSpec":{"tag":"v7.4.0"}}}}'

# 开启 vNPU / DRA（两者与 mindcluster device-plugin 存在冲突检测约束，注意 placement 隔离）
kubectl patch npuclusterpolicy cluster --type=merge -p '{"spec":{"vnpu":{"managed":true}}}'
```

修改 CR 后 operator 通常在一个调和周期内（秒级）感知并应用变更；`GenerationChangedPredicate` 保证纯 status 更新不会触发额外调和。
