# 02 基础平台独立化部署设计

## 目标

让 `02-base-platform` 中的控制台、插件框架、应用市场、Volcano 调度增强、NUMA 感知、Ray、日志/监控/多集群等能力能够**独立部署、单组件可用**，不依赖 openFuyao 安装器或昇腾组件。

## 现状问题

### 1. 认证体系假设

- 多个基础平台服务假定 `oauth-server`、`oauth-webhook`、`oauth-proxy` 已在集群中运行，并依赖其签发的 token 做身份校验。
- 当用户只想部署“一个控制台框架”时，强制安装 oauth 系统过于沉重。

### 2. 服务间地址硬编码

- `console-service` 可能通过固定 service 名（如 `marketplace-service.openfuyao-system`、`plugin-management-service.openfuyao-system`）调用下游服务。
- 各子前端可能被硬编码为预期与其它 openFuyao 组件在同一控制台内运行。

### 3. 命名空间耦合

- 大量服务默认部署在 `openfuyao-system-controller` 安装的命名空间（如 `openfuyao-system`）。
- 独立使用时应允许用户自定义 namespace 和 release 名。

### 4. 缺少顶层入口编排

- 没有一份“openFuyao 基础平台 Helm 元 chart”允许用户一键安装完整平台或只启用部分能力。

## 改造方案

### 方案一：统一认证抽象

#### 新增 `AuthProvider` 配置

每个服务通过环境变量或 `ConfigMap` 接收认证配置：

```yaml
auth:
  mode: disabled    # disabled / oauth / oidc
  oauth:
    serverUrl: http://oauth-server.openfuyao-system.svc
    webhookUrl: https://oauth-webhook.openfuyao-system.svc/validate
  oidc:
    issuerUrl: https://idp.example.com
    clientId: openfuyao-console
    groupsClaim: groups
```

#### 改造点

1. 在 `console-service`、`plugin-management-service`、`marketplace-service`、`application-management-service`、`multi-cluster-service`、`web-terminal-service` 中新增 `AuthProvider` 接口。
2. `disabled` 模式下不做身份校验，仅依赖 K8s RBAC（适合内网或已经由网关统一做鉴权的场景）。
3. `oauth` 模式保留现有 `oauth-server` 调用链路。
4. `oidc` 模式支持外部 OIDC Provider，控制台后端只做 JWT 校验。
5. 各服务的 `values.yaml` 暴露 `auth.mode`，默认 `disabled` 或 `oidc`，不再默认要求 oauth。

### 方案二：依赖服务地址可配置 + 可关闭

#### 为 `console-service` 建立依赖配置

```yaml
dependencies:
  pluginManagement:
    enabled: true
    serviceUrl: http://plugin-management-service:8080
  marketplace:
    enabled: false
    serviceUrl: http://marketplace-service:8080
  monitoring:
    enabled: false
    serviceUrl: http://monitoring-service:8080
  multiCluster:
    enabled: false
    serviceUrl: http://multi-cluster-service:8080
  userManagement:
    enabled: false
    serviceUrl: http://user-management-operator:8080
```

#### 改造点

1. `console-service` 在启动时读取依赖配置，未启用的子系统不在菜单/路由中展示。
2. 各子服务调用失败时给出降级行为（如隐藏图标或提示未启用）。
3. 所有子前端依然通过 `ConsolePlugin` CRD 注册；`console-website` 只加载当前集群中存在的 `ConsolePlugin`。
4. `plugin-management-service` 保持独立：它只监听 K8s 中 `ConsolePlugin` CRD，本身不依赖 console。
5. `marketplace-service` 保持独立：只解析 `HelmChartRepository` CRD 与 chart 源。
6. `application-management-service` 若依赖 marketplace，通过 `serviceUrl` 注入，自身无强依赖。

### 方案三：命名空间与发布名可配置

1. 所有 `02-base-platform` 组件的 chart 中移除对 `openfuyao-system` 的硬编码：
   - 使用 Helm `Release.Namespace`；
   - 使用 `{{ .Release.Name }}` 做资源命名前缀；
   - `ConsolePlugin` 引用子前端时可通过 `namespace` 参数定位。
2. `plugin-management-service` 监听所有 namespace 的 `ConsolePlugin`，或按配置限定 namespace list。
3. `HelmChartRepository` 也不限定命名空间，允许用户在目标 namespace 下创建。

### 方案四：基础平台 Helm 元 chart

新增 `charts/openfuyao-base-platform`（或 `platform-suite`）：

```yaml
dependencies:
  - name: console-service
    condition: console.enabled
  - name: console-website
    condition: console.enabled
  - name: plugin-management-service
    condition: console.enabled
  - name: marketplace-service
    condition: marketplace.enabled
  - name: user-management-operator
    condition: userManagement.enabled
  - name: volcano-ext
    condition: volcano.enabled
  - name: volcano-config-service
    condition: volcano.enabled
  - name: ray-service
    condition: ray.enabled
  - name: ray-website
    condition: ray.enabled
  - name: monitoring-service
    condition: monitoring.enabled
  - name: logging-operator
    condition: logging.enabled
  - name: multi-cluster-service
    condition: multiCluster.enabled
```

用户可单独安装任意组件，也可通过元 chart 统一安装：

```bash
# 只安装控制台框架
helm install platform openfuyao/openfuyao-base-platform \
  --set console.enabled=true

# 安装完整平台
helm install platform openfuyao/openfuyao-base-platform \
  --set console.enabled=true,marketplace.enabled=true,volcano.enabled=true,ray.enabled=true
```

### 方案五：版本与配置漂移管理

1. 把 `ComponentVersion` / `ClusterVersion` CRD 也用于基础平台组件版本管理。
2. 允许 `values.yaml` 覆盖全局 registry，并保持 `cr.openfuyao.cn` 作为默认值但不唯一。
3. 为独立使用场景发布 `values-offline.yaml`、`values-oidc.yaml` 等示例。

## 组件独立部署矩阵

| 组件 | 独立可部署 | 外部依赖 | 需要改造点 |
|---|---|---|---|
| `console-service` / `console-website` | 是 | K8s + 可选子服务 | 认证抽象、依赖开关、命名空间 |
| `plugin-management-service` | 是 | K8s API | 已独立，只需确认 namespace 监听 |
| `marketplace-service` | 是 | K8s API + chart 源 | 已独立 |
| `application-management-service` | 是 | 可选 marketplace | 服务地址可配置 |
| `user-management-operator` | 是 | K8s | 认证抽象 |
| `web-terminal-service` | 是 | K8s API | 已独立 |
| `volcano-ext` / `volcano-config-service` | 是 | K8s API + Volcano | 已独立 |
| `volcano-config-website` | 是 | console host | 子前端 |
| `ray-service` / `ray-website` | 是 | K8s + KubeRay | 已独立，只需可选配置 |
| `monitoring-*` / `logging-*` / `victoriametrics-controller` | 是 | K8s + 可观测组件 | 命名空间与认证 |
| `multi-cluster-service` / `multi-cluster-website` | 是 | K8s API + 多个集群 | 认证与依赖 |
| `node-feature-discovery` | 是 | 节点 + K8s | 已独立，通用 NFD 规则可默认启用 |

## 阶段建议

1. 先在 `console-service` 引入 `AuthProvider` 和依赖开关；
2. 同时为所有 02 组件整理 `values.yaml`，去掉硬编码 namespace；
3. 创建 `openfuyao-base-platform` 元 chart；
4. 验证单组件 `helm install` 是否可用；
5. 最后把元 chart 接入 `01` 的 `ClusterComponent` 安装流程。
