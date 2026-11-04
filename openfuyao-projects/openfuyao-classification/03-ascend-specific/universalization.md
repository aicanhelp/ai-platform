# 03 昇腾抽象与国产芯片扩展设计

## 目标

把昇腾 NPU / KAE / vNPU 相关代码从“仅支持华为昇腾”改造成**通用异构算力接入框架的样板实现**，使其它国产芯片（寒武纪、海光、燧原等）可以：

1. 复用同一套 CRD 描述设备类型、能力、策略；
2. 复用同一套节点发现、调度增强、控制台扩展机制；
3. 只编写厂商-specific 的驱动安装、设备插件/DRA 插件、NFD 规则、健康检查逻辑。

## 现状问题

### 1. CRD 直接绑定厂商

- `npu-operator` 定义 `NPUClusterPolicy`；
- `kae-operator` 定义 `KAEPolicy`；
- 其它国产芯片若照此模式，会各自新增一组 `XXXPolicy` CRD，平台需要逐一学习和集成。

### 2. 节点发现重复

- `npu-feature-discovery`、`kae-operator` 各自实现节点特征发现；
- 通用 NFD 虽被引入，但昇腾-specific 规则写在 `npu-operator/charts/npu-operator/templates/nodefeaturerules.yaml` 里，其它芯片无法复用这套模板机制。

### 3. 调度与 NUMA 数据壁垒

- `npu-operator` 在 `assets/volcano/` 中放置昇腾-specific 的 Volcano 调度配置；
- `volcano-config-service` 的 `Oenuma` CRD 定义的 NUMA 字段无人填充，说明当前厂商拓扑数据没有统一入口。

### 4. UI/监控未抽象

- `eagle-eye` 针对昇腾 NPU 做监控；
- 控制台没有通用“设备视图”扩展点；
- 各厂商若有自己的子前端，只能独立通过 `ConsolePlugin` 注册，但没有统一的数据模型。

## 改造方案

### 方案一：通用设备接入 CRD

#### 新增 Group：`devices.openfuyao.com/v1`

```yaml
apiVersion: devices.openfuyao.com/v1
kind: DeviceClass
metadata:
  name: npu.ascend.com
spec:
  displayName: "昇腾 NPU"
  vendor: "Huawei"
  category: "NPU"
  resourceNames:
    - huawei.com/Ascend310P
    - huawei.com/Ascend910
```

```yaml
apiVersion: devices.openfuyao.com/v1
kind: DeviceProfile
metadata:
  name: ascend-910b
spec:
  deviceClassRef: npu.ascend.com
  modelName: "Ascend 910B"
  capabilities:
    compute: "256 TOPS"
    memory: "32 GiB"
    supportsVirtualization: true
  runtime:
    driverImage: ascend-driver-installer:v1.0
    containerToolkitImage: npu-container-toolkit:v1.0
    draPluginImage: npu-dra-plugin:v1.0
    devicePluginImage: npu-device-plugin:v1.0
  features:
    - name: numa-aware
      defaultEnabled: true
  scheduler:
    resourceName: huawei.com/Ascend910
    volcanoQueueConfig: |-
      # 昇腾 Volcano queue snippet
  monitoring:
    exporterImage: eagle-eye:v1.0
    metricsPort: 8080
```

```yaml
apiVersion: devices.openfuyao.com/v1
kind: NodeDeviceInventory
metadata:
  name: node-01
spec:
  nodeName: node-01
  devices:
    - deviceProfileRef: ascend-910b
      deviceID: "npu0"
      health: Healthy
      capacity:
        huawei.com/Ascend910: "1"
        huawei.com/AscendVNP: "4"
      topology:
        numaNode: 0
```

```yaml
apiVersion: devices.openfuyao.com/v1
kind: DevicePolicy
metadata:
  name: npu-default
spec:
  deviceClassRef: npu.ascend.com
  virtualization:
    mode: vNPU    # none / vNPU / MIG / SR-IOV
    defaultSlice: 1
```

### 方案二：通用设备生命周期控制器

新增一个通用 `device-operator`（或复用/扩展 `node-feature-discovery` + 几个小控制器），职责包括：

1. 根据 `DeviceProfile` 中的 `runtime.*Image` 生成节点 DaemonSet：
   - 驱动安装器；
   - 设备插件 / DRA 插件；
   - 容器工具包。
2. 根据 `DeviceProfile` 中的 `features` 生成通用 `NodeFeatureRule`，由 NFD 应用。
3. 收集各厂商 device plugin 上报的 `NodeDeviceInventory`。
4. 调和 `DevicePolicy`，决定虚拟化切分、健康检查、故障隔离策略。
5. 把 `NodeDeviceInventory` 中的 `topology` 转换为 Volcano 可消费的 `Numatopology` 或 `NodeConfig` / `Oenuma` 数据。

#### 厂商 operator 的新定位

| 当前组件 | 当前定位 | 目标定位 |
|---|---|---|
| `npu-operator` | 昇腾 NPU 总控，定义私有 CRD | 昇腾专用实现：封装 `DeviceProfile`、驱动镜像、volcano snippet、NFD 规则 |
| `kae-operator` | 昇腾 KAE 总控 | 昇腾 KAE 专用实现，复用同一框架 |
| `npu-driver-installer` | 节点驱动安装 | 被 `device-operator` 按 `DeviceProfile.runtime.driverImage` 调度执行 |
| `npu-feature-discovery` | 节点特征发现 | 被通用 NFD 框架替代或作为 NFD 规则输入源 |
| `eagle-eye` | 昇腾监控 | 被 `DeviceProfile.monitoring.exporterImage` 替代，通用监控框架抓取指标 |
| `npu-operator/assets/volcano/*` | 昇腾 Volcano 配置 | 迁移到 `DeviceProfile.scheduler` 中作为可插拔调度模板 |

### 方案三：节点发现规则模板化

把各厂商的 `nodefeaturerules.yaml` 抽象为：

```yaml
apiVersion: devices.openfuyao.com/v1
kind: DeviceFeatureRule
metadata:
  name: ascend-npu-feature-rule
spec:
  deviceClassRef: npu.ascend.com
  deviceProfileRef: ascend-910b
  nfdRule: |-
    name: "ascend-npu"
    labels:
      "ascend-npu": "true"
    matchFeatures:
      - feature: pci.device
            matchOn:
              - class: {op: In, value: ["12d8"]}
```

通用 NFD 控制器加载所有 `DeviceFeatureRule`，按 `DeviceClass` 生成对应的 `NodeFeatureRule`。

### 方案四：调度与 NUMA 数据提供者接口

为 Volcano / NUMA 感知调度引入统一的拓扑提供者：

```go
type DeviceTopologyProvider interface {
    GetNodeDeviceTopology(nodeName string) (*NodeDeviceTopology, error)
}
```

- 昇腾实现：读取 `npu-feature-discovery` 或昇腾 driver 提供的节点信息，写入 `NodeDeviceInventory`；
- 其它厂商实现：同样写入标准 `NodeDeviceInventory`；
- `volcano-config-service` / `volcano-ext` 统一从 `NodeDeviceInventory` 与 `Numatopology` 合并生成调度决策，不再读取 `Oenuma` 等私有开关之外的字段。

具体改造：
- 把 `Oenuma` 中的 `Spec.Updateenable` 保留为兼容开关；
- 新增 `Oenuma` 数据填充逻辑：从 `NodeDeviceInventory.topology` 自动填充 `Node` / `Numa` / `PodAffinity`；
- 或者弃用 `Oenuma` 的私有 group，统一使用 `nodeinfo.volcano.sh` 的 `Numatopology`。

### 方案五：控制台设备视图扩展点

为控制台提供通用设备视图框架：

```yaml
apiVersion: console.openfuyao.com/v1beta1
kind: ConsolePlugin
metadata:
  name: ascend-device-plugin
spec:
  displayName: "昇腾设备管理"
  route: /devices/ascend
  entry: /static/plugins/ascend/index.js
  deviceBinding:
    deviceClassRef: npu.ascend.com
  resources:
    - apiGroup: devices.openfuyao.com
      kind: NodeDeviceInventory
```

- `console-service` 识别带 `deviceBinding` 的 `ConsolePlugin`，将其挂载到“设备管理”主菜单下；
- 子前端通过统一 REST/K8s 接口读取 `DeviceClass` / `DeviceProfile` / `NodeDeviceInventory`，无需了解昇腾私有 CRD；
- 为每个厂商子前端提供统一卡片、列表、监控图表模板，降低重复开发。

### 方案六：国产芯片扩展 SOP

其它国产芯片接入时，只需提供：

1. 一个 `DeviceClass`；
2. 一个或多个 `DeviceProfile`（含驱动/运行时/调度/监控镜像）；
3. 一个厂商-specific 小 operator 或 DaemonSet，负责：
   - 安装驱动；
   - 注册设备插件/DRA 插件；
   - 把设备信息写入 `NodeDeviceInventory`。
4. 一个 `ConsolePlugin`（可选），用于设备视图。

平台其余能力（节点发现、NUMA 调度、Volcano 配置、控制台挂载、监控采集）全部复用。

## CRD 变更概览

| 变更 | 新增/修改 | 影响组件 |
|---|---|---|
| `DeviceClass` | 新增 | 通用 device-operator、console-service |
| `DeviceProfile` | 新增 | 通用 device-operator、厂商 operator |
| `NodeDeviceInventory` | 新增 | 通用 device-operator、volcano-config-service、console-service |
| `DevicePolicy` | 新增 | 通用 device-operator、厂商 operator |
| `DeviceFeatureRule` | 新增 | node-feature-discovery 通用控制器 |
| `Oenuma` / `Numatopology` | 修改或替换 | volcano-config-service、volcano-ext |
| `NPUClusterPolicy` | 逐步弃用 | 替换为 `DeviceProfile` |
| `KAEPolicy` | 逐步弃用 | 替换为 `DeviceProfile` |

## 实施顺序建议

1. 定义 `devices.openfuyao.com` CRD（DeviceClass / DeviceProfile / NodeDeviceInventory / DevicePolicy / DeviceFeatureRule）；
2. 开发最小通用 `device-operator`：
   - 读取 DeviceProfile；
   - 生成驱动 DaemonSet；
   - 生成通用 NodeFeatureRule；
   - 汇总 NodeDeviceInventory；
3. 改造 `npu-operator` 作为第一个 DeviceProfile 供应商（保留兼容模式）；
4. 改造 `kae-operator` 作为第二个 DeviceProfile 示例；
5. 选择一个非昇腾国产芯片（或虚拟的国产设备）做第三个实现，验证框架通用性；
6. 在 `volcano-config-service` 中接入 `NodeDeviceInventory` 拓扑数据，替代 `Oenuma` 手写逻辑；
7. 在控制台增加通用“设备管理”菜单和 `ConsolePlugin` 扩展点。
