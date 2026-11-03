# sig-large-scale-cluster

欢迎来到sig-large-scale-cluster！

## 工作目标和范围

- **目标：** 致力于提供稳定的超大规模集群，单集群聚焦Kubernetes核心组件优化、AI作业调度优化与TCP store键链优化，突破Kubernetes宣称的纳管上限，支撑128k卡（1.6万节点）集群。面向未来50万卡集群规模，提供多集群多级调度等技术来进一步提升调度性能。
- **范围：** 聚焦超大规模下网络、监控、AI作业调度、镜像拉取加速、AI训练稳定性提升、Kubernetes核心组件优化与集群的自动化部署。

## 组织例会

sig-large-scale-cluster是开放的，我们的用户、贡献者和维护者不断改进它。正是因为您，我们才能够构建出更稳定、易用的超大规模集群。加入我们！

- 我们会在会议上分享最新的进展、解决项目问题、演示新功能、讨论未来规划等，加入sig-large-scale-cluster的例会：

  - 会议时间：每周四上午 10:00~11:30。
  - 会议链接：[在openFuyao官网社区会议中获取](https://www.openfuyao.cn/zh/meeting/)。
  - 会议记录：[会议纪要](https://etherpad.openfuyao.cn/p/sig-Large-Scale-Cluster)。

## 成员

### Maintainer列表

- Luo Gangyi[@luogangyi](https://gitcode.com/luogangyi)  
- Zhu Guihua[@osmcn](https://gitcode.com/osmcn)  
- SHI Zilong[@shizilong](https://gitcode.com/shizilong)  

## 联系方式

- 邮件列表：[订阅sig-large-scale-cluster邮件](https://mailweb.openfuyao.cn/mailman3/lists/large-scale-cluster@list.openfuyao.cn/)。

## 项目清单

repository地址请见[code-repository-info](./docs/code-repository-info.md)文件。

## 目录结构说明

- 本仓库目录结构说明

| 目录名             | 说明                       |
|-----------------|--------------------------|
| best-practices/ | SIG内部最佳实践总结。              |
| design/         | 已定稿的设计文档，包括架构设计、特性设计、评审结果等。 |
| discussions/    | 存放非正式、探索性、开放性讨论内容。        |
| docs/           | 通用文档，如使用指南、FAQ、操作手册等。     |
| proposals/      | 需求分析、未来规划、提案文档。           |
| principles/     | 技术原理解析、底层机制讲解文档。          |
| references/     | SIG内引用本社区或其他社区的流程、规范文档。   |

## 参与贡献代码

请参考[贡献指南](CONTRIBUTING.md)
