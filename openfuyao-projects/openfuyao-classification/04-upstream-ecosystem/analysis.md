# 04 Kubernetes 生态上游依赖详细分析

## 1. 为什么单独分类

openFuyao 仓库不仅包含自研业务代码，还内嵌或引用了多个 Kubernetes 生态核心上游项目的 Fork/镜像。 它们体量巨大但无 openFuyao 业务逻辑，若与自研代码混在一起，会干扰对平台能力边界和通用化改造的分析。

## 2. 各上游项目定位

| 项目 | 定位 | 由谁部署/消费 |
|---|---|---|
| `kubernetes` | K8s 发行版源码 | 由 `bkeadm` + `cluster-api-provider-bke` 部署到目标集群。 |
| `containerd` | 容器运行时 | 由节点初始化 `Command` 与 package repo 安装到目标节点。 |
| `runc` | OCI 容器运行时 | 与 containerd 一起作为节点运行的一部分。 |
| `etcd` | 控制面数据存储 | 作为 K8s 主节点堆栈的一部分自动拉起。 |
| `coredns` | 集群 DNS | 默认 addon，与 `kubeproxy`、`calico` 一起由安装器 apply。 |
| `kwok` | 虚拟节点/压力测试 | 用于大规模测试，与生产运行时无关。 |

## 3. 与三类的关系

- **01 安装部署类**：决定这些上游组件的版本、镜像、rpm 包、部署顺序。
- **02 通用基础平台类**：把它们作为集群基础设施使用，例如 console-service 依赖 coredns、monitoring-service 依赖 coredns/etcd 等。
- **03 昇腾专有类**：依赖 K8s 标准设备插件/DRA/容器调度机制，从而与上游 K8s/containerd 交互。

## 4. 未来可做的通用化工作

1. 提炼每个上游项目的 openFuyao 补丁清单；
2. 评估哪些补丁是通用增强（例如特定 cgroup 设置、网络插件参数）可下沉为 `ClusterVersion` / `ComponentVersion` 配置；
3. 明确 release/package metadata 来源，逐步把版本管理从硬编码目录迁移到 CRD 或制品仓库清单。

## 5. 复制范围

本次完整复制了这些项目的源码目录（`.git` 除外），保留了完整的上游目录结构。
