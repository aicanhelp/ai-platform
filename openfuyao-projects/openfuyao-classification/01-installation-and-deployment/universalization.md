# 01 通用 K8s 部署工具设计

## 1. 目标与定位

把 `bkeadm` / `cluster-api-provider-bke` / `bkeagent` / `installer-service` / `bke-console-*` 这一套升级成**通用 K8s 部署与运维工具**。

**目标不是简单地去掉 openFuyao 标签。** 开源界已经有 kubeadm、kubespray、sealos 等工具能“把 K8s 装起来”，再做一次没有竞争力。我们要做的是：

> **以 openFuyao 在超大规模集群、异构算力、复杂控制面拆分上的工程实践为起点，把复杂 K8s 拓扑的部署和运维工作工程化、声明化、可复用化。** 让万级节点、多 etcd、多 apiserver 组、按 API 资源分层路由、异构算力就绪等场景，从“靠专家手工拼装”变成“靠配置文件和向导一键交付”。

换句话说：

| 模式 | 产物 |
|---|---|
| 原生 K8s 模式 | 只带基础 addon（kube-proxy、CNI、CoreDNS）的标准 K8s 集群 |
| openFuyao 平台模式 | 在 K8s 之上安装 openFuyao 管理平台（第二层） |
| 设备就绪模式 | 在任意 K8s 上安装设备接入框架与具体厂商算力组件（第三层） |
| 超大规模模式 | 多 etcd 拆分、多 apiserver 组、按 API 资源分组负载均衡的万级节点控制面 |

该工具需要与上层平台解耦、与具体品牌解耦、与具体硬件解耦，只负责“把 K8s 集群按指定拓扑和参数部署好并交付给后续平台层”。

## 2. 我们为什么基于 openFuyao 做，而不是从零做一个工具

openFuyao 已经沉淀了很多一般社区工具不具备的东西：

| 已有能力 | 对“通用 K8s 部署工具”的价值 |
|---|---|
| `bkeadm` + `cluster-api-provider-bke` + `bkeagent` 的声明式集群/节点生命周期 | 作为工程化安装引擎的骨架 |
| 已验证的 **20,000 节点**控制面拆分实践：多 etcd、多 apiserver 组、按 API 资源七层分组 | 这是复杂 K8s 部署的核心卖点 |
| `npu-operator` / `vNPU` / `volcano-config-service` 等异构算力部署经验 | 让部署工具直接支持“AI/HPC 就绪”的 K8s 拓扑 |
| `bke-console-website` + `installer-service` 引导控制台 | 保留向导式安装体验，但只做部署相关功能 |
| `openfuyao-system-controller` 等平台组件一键装配 | 作为可选 addon 保留，而非强制默认安装 |

**改造不是抛弃这些**，而是把它们从“只能装 openFuyao 平台”的专用模式，升级为“可以装任意 K8s 发行版 + 任意上层平台”的通用模式。

## 3. 与开源工具的差异

| 能力 | kubeadm / kubespray 等 | 通用 K8s 部署工具（目标） |
|---|---|---|
| 安装几十节点标准 K8s | ✅ | ✅ |
| 声明式/CRD 驱动的集群生命周期 | 部分有 | ✅（BKECluster/BKENode） |
| 万级节点控制面拓扑部署 | ❌ / 需大量手工 | ✅（多 etcd、多 apiserver 组、七层分组 LB） |
| 节点批量发现、模板化注册 | 弱 | ✅ |
| 可插拔 addon / 组件目录 | 弱 | ✅ |
| 向导式 Web 控制台 + CLI | 不一定 | ✅（可选 `bke-console-website`） |
| 与 openFuyao 管理平台天然集成 | ❌ | ✅（作为可选 addon） |
| 与设备接入框架配合部署异构算力 | 弱 | ✅ |

我们要保留并强化的，是**复杂控制面拓扑的工程化部署能力**和企业级生命周期管理与运维能力。

## 4. 现状问题

### 4.1 制品源硬编码

- `bkeadm/assets/offline-artifacts.yaml` / `online-artifacts.yaml` 中充斥：
  - `cr.openfuyao.cn/openfuyao`
  - `openfuyao.obs.cn-north-4.myhuaweicloud.com`
  - `hub.oepkgs.net/openfuyao`
- 本地 registry / repo 初始化脚本默认使用这些源。

### 4.2 控制面拓扑缺乏声明式配置

当前 `BKECluster` 主要描述节点和 addon，对 etcd、apiserver、load balancer、kubelet 调优等控制面拓扑参数描述不足。如果要部署 20,000 节点级别的控制面拆分架构，需要大量手工修改或外部脚本配合。

### 4.3 引导控制台与 openFuyao 平台强耦合

- `bkeadm init` 默认会部署 `bke-console-service` / `bke-console-website` / `oauth-*` / `marketplace-*` / `monitoring-*` 等 openFuyao 平台组件；
- `bkeadm/pkg/initialize/bkeconsole/` 的 `consts.sh`、`installConsole.sh` 中镜像/chart/命名空间/CRD 域名/证书 signer 全部硬编码成 `openfuyao`。

### 4.4 Addon 是目录硬映射，无法插拔

- `BKECluster.spec.clusterConfig.addons` 通过 `bke-manifests/kubernetes/<name>/<version>/<name>.yaml` 硬编码映射；
- 新组件必须修改目录结构或 `installer-service` 默认列表，扩展性不足。

### 4.5 认证体系预设

- `oauth-server` / `oauth-proxy` / `oauth-webhook` 在 bootstrap 阶段默认被拉起；
- 部署工具不应强制自带认证系统，应允许外部 OIDC 或关闭认证。

## 5. 核心能力模型

```text
输入
  │
  ├─ 制品源配置（ArtifactSource）
  ├─ 控制面拓扑配置（ControlPlaneTopology）
  ├─ 节点清单 / 节点池（BKECluster / BKENode / NodePool）
  ├─ addon / 组件目录（ComponentCatalog / ClusterComponent）
  └─ 可选认证配置（AuthProvider）
       │
       ▼
  通用 K8s 部署工具
       │
       ▼
  输出
  ├─ 原生 / openFuyao 平台 K8s 集群
  ├─ 可扩展的控制面拓扑（单/多 etcd、多 apiserver 组、LB 分组）
  ├─ 节点级 agent（bkeagent）持续 reconcile 集群状态
  └─ 可选的引导控制台 / installer Web UI
```

## 6. 改造方案

### 6.1 制品源外部化

#### 新增 CRD / 配置对象

```yaml
apiVersion: config.openfuyao.com/v1alpha1
kind: ArtifactSource
metadata:
  name: default
spec:
  registries:
    - name: k8s
      url: registry.io/k8s
    - name: openfuyao
      url: registry.io/openfuyao
      optional: true
  chartRepos:
    - name: stable
      url: https://charts.example.cn
  packageRepos:
    - name: k8s-yum
      type: yum
      baseUrl: https://pkgs.example.cn/yum
  binaryBaseUrl: https://binaries.example.cn
```

#### 改造点

1. `bkeadm/assets/offline-artifacts.yaml` 中所有仓库 URL 改为通过 `ArtifactSource` 或 CLI flag 注入，默认模板不填死任何 openFuyao 源。
2. `bkeadm init` 新增 flag：`--artifact-source <file>`；未指定时要求用户显式填写或使用纯本地内置源。
3. `cluster-api-provider-bke` 渲染 `Command` 时读取 `BKECluster.spec.artifactSourceRef`。
4. `bkeadm/pkg/initialize/initialize.go` 本地 registry 初始化脚本不再硬编码上游源。

### 6.2 控制面拓扑可配置（核心能力）

新增 `ControlPlaneTopology` 配置段（可内嵌在 `BKECluster.spec.controlPlane` 或作为独立 CRD）。

```yaml
apiVersion: config.openfuyao.com/v1alpha1
kind: ControlPlaneTopology
metadata:
  name: large-scale
spec:
  etcd:
    mode: split          # single / stacked / split
    clusters:
      - name: pods
        endpoints:
          - https://etcd-pods-1:2379
          - https://etcd-pods-2:2379
          - https://etcd-pods-3:2379
      - name: events
        endpoints:
          - https://etcd-events-1:2379
          - https://etcd-events-2:2379
          - https://etcd-events-3:2379
      - name: data
        endpoints: [...]
    serverOverrides:
      - prefix: /registry/pods
        cluster: pods
      - prefix: /registry/events
        cluster: events
      - prefix: /registry/leases
        cluster: events
  apiServers:
    groups:
      - name: pods
        replicaCount: 6
        listenPort: 6443
      - name: nodes
        replicaCount: 3
      - name: core
        replicaCount: 3
    loadBalancer:
      type: haproxy   # haproxy / external
      layer7Routes:
        - pathPrefix: /api/v1/pods
          backendGroup: pods
        - pathPrefix: /api/v1/nodes
          backendGroup: nodes
        - pathPrefix: /api/v1/services
          defaultBackendGroup: core
  kubelet:
    qps: 50
    burst: 100
    statusUpdateFrequency: 20s
  apiServer:
    maxRequestsInflight: 800
    maxMutatingRequestsInflight: 400
```

#### 改造点

1. `cluster-api-provider-bke` 生成控制面 `Command` 时支持渲染多 etcd 集群与 `--etcd-servers-overrides`。
2. 部署工具支持为不同 apiserver 组生成差异化的启动参数，并配置前端 HAProxy/nginx/外部 LB 的七层规则模板。
3. 默认最小拓扑（单 etcd / 单 apiserver / 默认参数）与大规模拓扑使用同一套 schema，通过规模模板切换。
4. 调优参数写入节点 `bkeagent` 可读取的 `NodeConfig` 或 `KubeadmConfig` 补丁。

### 6.3 默认 addon 中立化与组件目录

#### 最小默认 addon 集合

```yaml
spec:
  clusterConfig:
    addons:
      - kubeproxy
      - coredns
      - calico
```

`openfuyao-system-controller` 从默认集合中移除，变为可选 addon：

```yaml
      - name: openfuyao-system-controller
        enabled: false
        catalogRef: openfuyao
```

#### Addon 注册机制 CRD 化

```yaml
apiVersion: config.openfuyao.com/v1alpha1
kind: ClusterComponent
metadata:
  name: openfuyao-console
spec:
  displayName: openFuyao 管理平台
  category: platform
  source:
    type: helm
    chart: openfuyao-system-controller
    repoRef: openfuyao
  dependencies:
    - coredns
    - calico
```

```yaml
apiVersion: config.openfuyao.com/v1alpha1
kind: ComponentCatalog
metadata:
  name: default
spec:
  sources:
    - name: bke-manifests
      type: local
      basePath: /opt/bke-manifests/kubernetes
    - name: openfuyao-oci
      type: oci
      url: oci://cr.openfuyao.cn/openfuyao/components
```

#### 改造点

1. `BKECluster.spec.clusterConfig.addons` 升级为对象数组，支持 `name`、`catalogRef`、`enabled`、`parameters`。
2. `cluster-api-provider-bke` 的 addon applier 按 `ComponentCatalog` + `ClusterComponent` 解析来源，保留 `bke-manifests` 本地 catalog 兼容。
3. `installer-service` 暴露 `/components` API，前端可列出、启用/禁用 addon。
4. 引导控制台只展示与部署相关的 addon，openFuyao 平台组件作为可选 addon 出现。

### 6.4 认证中立化

1. `bkeadm init` 新增 `--without-auth` / `--auth-provider <name>`：
   - `noop`：不部署 oauth，也不做 webhook 集成；
   - `oauth`：部署内置 oauth-server/webhook（当前 openFuyao 行为，作为可选 addon）；
   - `oidc`：使用外部 OIDC 配置注入 `kube-apiserver --oidc-*` 参数。
2. `installer-service` 的认证/鉴权中间件抽象为 `AuthProvider` 接口。
3. 原生 K8s 模式下默认使用 `noop` 或外部 OIDC。

### 6.5 引导控制台与品牌解耦

1. 引导控制台（`bke-console-*`）是部署工具的可选 UI，不是 openFuyao 管理平台的入口：
   - 只展示 cluster、node、addon、控制面拓扑、部署进度等生命周期内容；
   - 不预装 `marketplace`、`monitoring`、`logging`、`web-terminal` 等第二层服务。
2. 提供 `bkeadm init --without-console` / `--console-provider <name>`：
   - `none`：完全 CLI 驱动；
   - `bke-console`：使用默认引导控制台；
   - `custom`：通过 `--console-chart` / `--console-values` 替换为第三方控制台。
3. 引导控制台的 logo、标题、文档链接等通过 ConfigMap / values 抽成可配置，默认保持中性。
4. 镜像、chart、命名空间、证书 signer、CRD 域名等全部参数化，当前 openFuyao 值作为一组默认 profile 提供`。

### 6.6 输出产物分层

| 输出 | 说明 |
|---|---|
| `GenericCluster` | 原生 K8s，只带基础 addon，无 openFuyao 组件 |
| `OpenFuyaoCluster` | 原生 K8s + openFuyao 管理平台 addon |
| `LargeScaleCluster` | 多 etcd / 多 apiserver 组 / 七层 LB 的万级节点就绪拓扑 |
| `DeviceReadyCluster` | 任意上述模式 + 设备接入框架 + 具体厂商设备组件 |

## 7. CRD / 配置变更

| 变更 | 新增/修改 | 影响组件 | 说明 |
|---|---|---|---|
| `ArtifactSource` | 新增 | `bkeadm`、`installer-service` | 制品源外部化 |
| `ControlPlaneTopology` | 新增 | `bkeadm`、`cluster-api-provider-bke` | 控制面拆分与调优 |
| `NodePool` / `BKENodeTemplate` | 新增 | `cluster-api-provider-bke` | 批量节点模板化注册 |
| `ClusterComponent` | 新增 | `cluster-api-provider-bke`、`installer-service` | addon 注册 |
| `ComponentCatalog` | 新增 | `cluster-api-provider-bke`、`installer-service` | addon 目录 |
| `BKECluster` | 修改 | `installer-service`、`cluster-api-provider-bke` | 增加 controlPlane、artifactSourceRef、authProvider 等字段 |
| `AuthProvider` | 新增 | `installer-service`、前端中间件 | 认证可替换 |

## 8. 实施顺序

1. **制品源参数化**：把 `bkeadm` 和 `bkeadm/pkg/initialize/bkeconsole` 中的 registry / yum / chart / image 源抽到 `ArtifactSource` profile；低侵入，先改常量文件与脚本模板。
2. **引导控制台解耦**：把 `installConsole.sh` 中的 chart 列表拆成“部署工具必选项 + openFuyao 平台可选项”，默认不装 openFuyao 平台全家桶；同步把命名空间/CRD 域名/证书 signer 变量化。
3. **addon CRD 化**：`installer-service` 和 `cluster-api-provider-bke` 支持 `ClusterComponent` / `ComponentCatalog`，此时 openFuyao 平台组件作为可选 addon。
4. **控制面拓扑 CRD 化**：支持 `ControlPlaneTopology` 描述多 etcd、多 apiserver 组、LB 分组规则；生成对应的 `kubeadm` 配置与 HAProxy 配置模板。
5. **认证抽象**：`installer-service` 接入 `AuthProvider` 接口；`bkeadm init` 支持 `--auth-provider`。
6. **验证**：
   - 最小模式：原生 K8s，无控制台、无 openFuyao 组件；
   - 默认最小控制台：带引导 UI 的原生 K8s；
   - openFuyao 平台模式：K8s + 第二层全部组件；
   - 大规模模式：多 etcd、多 apiserver 组、七层分组（仿真环境验证）。

## 9. 与第二、三层的关系

```text
通用 K8s 部署工具
        │
        ├─ 输出原生 K8s ─────────────────────────► 用户 / 第三方平台
        │
        ├─ 输出 openFuyao 平台模式 K8s ───────────► 通用 openFuyao 管理平台（第二层）
        │
        └─ 输出设备就绪型 K8s ────────────────────► 设备接入框架（第三层）
```

第二层和第三层都是**部署工具的可选 addon**。

- 第二层负责在 K8s 之上提供：控制台、应用市场、组件参考库、组件安装部署、用户权限、可观测、调度增强、AI workload 等。
- 第三层负责定义：设备模型 CRD、异构算力生命周期接口、调度扩展点，由各厂商 operator 实现。

通用 K8s 部署工具不关心各 addon 内部逻辑，只负责按 `ComponentCatalog` 把它们可靠地部署、升级、维护在目标 K8s 上。
