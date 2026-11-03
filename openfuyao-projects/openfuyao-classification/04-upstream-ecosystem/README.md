# 04 Kubernetes 生态上游依赖 / Fork

本目录收录 openFuyao 维护或集成的 Kubernetes 生态上游项目。这些不是 openFuyao 自研业务代码， but 是运行平台所必需的底层运行时、网络、存储、虚拟节点或 K8s 发行版本身。

## 包含组件

| 目录 | 说明 | 与 openFuyao 的关系 |
|---|---|---|
| `kubernetes` | Kubernetes 主仓库 Fork。 | 作为目标集群的 K8s 发行版被安装器部署。 |
| `containerd` | 容器运行时 Fork。 | 安装阶段部署到目标节点。 |
| `runc` | 容器运行时 low-level 工具。 | 与 containerd 一起作为节点运行时。 |
| `coredns` | 集群 DNS 组件 Fork。 | 作为基础平台 addon 部署。 |
| `etcd` | 分布式键值存储 Fork。 | K8s 控制面的默认后端存储。 |
| `kwok` | K8s Without Kubelet 虚拟节点工具。 | 用于大规模测试/模拟。 |

## 边界说明

- 这些代码保持上游结构，只做了必要的构建/集成适配，没有 openFuyao 业务逻辑。
- 它们不是昇腾专有，也不属于某个单一 openFuyao 服务。
- 功能分类（01/02/03）通过 `bke-manifests`、Helm chart 或容器镜像间接使用它们。
- 若其中某些项目包含 openFuyao 特定补丁，未来应把补丁清单整理到 `analysis.md` 中，便于评估是否应抽离到平台层 CRD/配置里。
