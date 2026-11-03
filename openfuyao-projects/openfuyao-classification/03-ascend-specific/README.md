# 03 昇腾专有组件

本目录收录所有名称、CRD、驱动、设备插件、虚拟化配置、监控采集规则等直接绑定华为昇腾（Ascend NPU）/ KAE / vNPU 的组件。

## 包含组件

| 目录 | 说明 |
|---|---|
| `eagle-eye` | 昇腾 NPU 运维/监控工具。 |
| `kae-device-plugin` | 昇腾 KAE 加速设备插件。 |
| `kae-driver-installer` | 昇腾 KAE 驱动安装器。 |
| `kae-operator` | 昇腾 KAE 策略与生命周期管理。 |
| `npu-container-toolkit` | 昇腾容器运行时工具包。 |
| `npu-dra-plugin` | 昇腾 DRA（Dynamic Resource Allocation）插件。 |
| `npu-driver-installer` | 昇腾 NPU 驱动安装器。 |
| `npu-feature-discovery` | 昇腾专用节点特征发现，配合通用 NFD 使用。 |
| `npu-node-provision` | 昇腾节点初始化/资源准备。 |
| `npu-operator` | 昇腾 NPU 总控 operator，含 Volcano / vNPU / NFD 子 chart 等。 |
| `vNPU` | 昇腾虚拟 NPU 能力。 |

## 边界说明

- 这些组件依赖或创建昇腾专有 CRD，例如 `NPUClusterPolicy`（npu-operator）和 `KAEPolicy`（kae-operator）。
- 它们通常通过 Helm 子 chart 引入通用组件：`node-feature-discovery`、`volcano` 等。这些通用依赖的源码已归入 `02-base-platform`，这里只保留昇腾侧特有配置/规则（如 `nodefeaturerules.yaml`、昇腾 Volcano 资源、昇腾资源 YAML）的分析与引用。
- `openFuyao` 平台本身不感知这些组件内部的数据结构；它们以标准 K8s 资源和节点 label 被平台间接使用。
