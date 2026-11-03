# 03 昇腾专有组件详细分析

## 1. 总体定位

昇腾组件是 openFuyao 平台的**异构算力供应商插件**。它们不修改平台通用能力，而是通过以下方式接入：

1. 厂商自己提供 operator + CRD + Helm chart；
2. 安装器/应用市场负责把 chart 部署到目标集群；
3. 设备通过标准 K8s 设备插件 / DRA / 节点标签暴露给调度器；
4. 控制台不针对昇腾写专有页面，若有 UI 也是子应用通过 `ConsolePlugin` 挂载。

## 2. NPU 组件

### 2.1 核心 operator

- `npu-operator`：昇腾 NPU 总控。
  - 自带 `NodeFeatureRule`（`charts/npu-operator/templates/nodefeaturerules.yaml`）用于给含 NPU 的节点打昇腾-specific label。
  - 子 chart 包括上游 `node-feature-discovery` 与昇腾-specific 的 `npu-feature-discovery`。
  - `assets/volcano/` 下包含昇腾-specific 的 Volcano / vNPU 资源和配置。
  - `values.yaml` 中 `nfd.nodefeaturerules: false` 表示默认不启用昇腾 NFD 规则。

### 2.2 驱动与运行时

- `npu-driver-installer`：在节点上安装/维护昇腾驱动。
- `npu-container-toolkit`：把 NPU 设备能力注入容器。
- `npu-node-provision`：昇腾节点初始化。
- `npu-dra-plugin`：基于 K8s 1.26+ DRA 框架分配 NPU。
- `npu-feature-discovery`：昇腾专用节点特征发现组件，向 NFD / K8s 暴露昇腾设备信息。

### 2.3 虚拟化

- `vNPU`：昇腾虚拟 NPU。`npu-operator/assets/volcano/` 中也包含 vNPU 相关调度资源。

### 2.4 监控

- `eagle-eye`：昇腾 NPU 运维/监控，通常暴露 Prometheus 指标或提供昇腾-specific 告警规则。

## 3. KAE 组件

| 组件 | 说明 |
|---|---|
| `kae-operator` | 昇腾 KAE（Kunpeng Acceleration Engine）策略控制器，定义 `KAEPolicy` 等 CRD。 |
| `kae-device-plugin` | 把 KAE 加速设备注册为 K8s 扩展资源。 |
| `kae-driver-installer` | 安装 KAE 驱动。 |

`kae-operator` 的 chart 也依赖上游 `node-feature-discovery` 作为子 chart。  
这再次说明：NFD 是通用能力，昇腾规则是其中一份可选规则。

## 4. 与通用平台的关系

| 通用平台能力 | 昇腾组件如何使用 |
|---|---|
| `node-feature-discovery`（02） | 作为子 chart 引入，并额外加载昇腾 `nodefeaturerules.yaml`。 |
| Volcano 调度器（02） | `npu-operator/assets/volcano/` 提供昇腾 Volcano / vNPU 配置。注意这不是 `volcano-ext` / `volcano-config-service` 管理的范围。 |
| `ConsolePlugin`（02） | 昇腾子前端如果要进控制台，会通过通用 `ConsolePlugin` 注册；当前 `npu-operator` 没有对应 `ConsolePlugin`。 |
| 安装器 addons（01） | 昇腾 chart 可作为应用市场或安装器 addon 被部署。 |

## 5. 通用化方向

1. 把昇腾 CRD（`NPUClusterPolicy`、`KAEPolicy`）作为厂商实现样例；
2. 抽象出通用设备接入框架：
   - 通用 `DeviceProfile` CRD：描述某种加速卡的驱动、运行时、资源名、虚拟化模式；
   - 通用 `NodeDeviceInventory`：节点设备清单；
3. 昇腾-specific 的 NFD 规则、Volcano 资源、监控规则作为 `DeviceProfile` 的默认模板；
4. `npu-operator`、`kae-operator` 等保留为具体厂商实现，但都去实现同一个通用接口或 CRD。

## 6. 复制范围说明

本目录完整复制上述组件的源码目录。同时会在根分析中记录哪些文件属于昇腾-specific 规则（如 `nodefeaturerules.yaml`、`assets/volcano/*`），以便后续通用化时抽取模板。
