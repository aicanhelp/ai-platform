# sig-installation

欢迎来到sig-installation！

## SIG组目标

sig-installation旨在构建一个开放、易用的Kubernetes集群安装体系，实现零基础用户一键安装Kubernetes集群，目标如下：

1. **简化复杂部署**：降低用户在多样化环境中部署符合最佳实践的生产级Kubernetes集群的门槛。
2. **统一生命周期管理**：提供从创建到日常运维的标准化工具链，实现管理的**一致性和可重复性**。
3. **实现自动化与可扩展性**：通过如`Cluster API`，用声明式API自动化管理大规模集群的整个生命周期，并支持横向扩展到管理任意数量的集群。

## 项目规划RoadMap（2026/01/13更新）

| 规划能力 | 规划目标 | 计划发布 |
| --- | --- | --- |
| 前置检查 | - 提供完备的环境前置检查工具，提前识别集群部署失败点 | 26.X |
| 声明式版本安装/升级框架 | - 版本可追溯、升级可编排、兼容性可校验 | 26.X |
| 集群级/节点级配置分层管理 | - 基于Node拆分支持节点的运维能力，包括但不限于节点自定义配置变更、节点级安装rpm包。| 26.X |
| 安装升级框架增强 | - 支持Helm Chart及二进制包规范接入 | 26.X |
| 组件自动化生命周期管理(Operator化) | - 开箱即用，组件全生命周期自管理 | 27.X |
| k8s组件部署增强 | - 支持外接etcd集群<br> - 支持关键组件kube-controller-manager/scheduler/apiserver按规划节点部署 | 27.X |
| 简化引导节点架构 | - 降低部署对专用节点依赖，释放资源用于业务负载，降低交付成本 | 27.X |
| 支持组件参数可配置                | - 支持安装集群的containerd参数可配置；<br>- 支持安装集群的kubelet参数可配置。      | 25.12 |
| 支持集群自定义证书和kubeconfig能力降级 | - 支持用户导入全局CA证书签发集群使用的证书；<br>- 支持不同K8s组件的kubeconfig签发。    | 25.12 |
| 支持集群安装自定义仓库              | - 支持镜像仓和二进制仓的自定义配置，并基于此安装集群。                             | 25.12 |
| 引导节点提供用户管理能力             | - 引导节点提供用户管理能力，便于小集群安装使用（集群节点数<10）。                      | 25.09 |
| 完全支持离线安装                 | - 完全支持集群的离线安装。                                           | 25.09 |

## 如何贡献代码

请参考[贡献指南](CONTRIBUTING.md)

## 社区会议

sig-installation开放的，我们的用户、贡献者和维护者不断改进它。正是因为您，我们才能够构建出更好的部署工具。加入我们！

- 安装部署SIG例行每周三上午10:00-11:00召开，可以在社区官网获取具体会议议程：[https://www.openfuyao.cn/zh/meeting/](https://www.openfuyao.cn/zh/meeting/)。
- SIG例行会议议题申报：https://etherpad.openfuyao.cn/p/release-management。
- 可以通过社区官网订阅邮箱：[installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn)订阅SIG例会和获取最新SIG资讯。

## 提案提交&归档

安装部署SIG相关提案(OFEP)请以PR提交至：https://gitcode.com/openFuyao/ofep/tree/main/ofeps/sig-Installation

## 仓库清单（2026/01/13更新）

repository地址：

- [sig-installarion](https://gitcode.com/openFuyao/sig-installation)
- [bkeadm](https://gitcode.com/openFuyao/bkeadm)
- [cluster-api-provider-bke](https://gitcode.com/openFuyao/cluster-api-provider-bke)
- [bkeagent](https://gitcode.com/openFuyao/bkeagent)
- [bkecommon](https://gitcode.com/openFuyao/bkecommon)
- [bke-manifests](https://gitcode.com/openFuyao/bke-manifests)
