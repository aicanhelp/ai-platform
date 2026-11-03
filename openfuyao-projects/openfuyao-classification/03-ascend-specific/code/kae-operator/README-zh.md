# kae-operator

Kubernetes可通过设备插件框架访问鲲鹏 [KAE 加速器](https://www.hikunpeng.com/developer/techArticles/20230906-4?envFlag=1)
等特殊硬件资源。但是，配置和管理带有这些硬件资源的节点需要配置多个软件组件，如驱动程序、容器运行时或其他库，这既困难又容易出错。

KAE Operator使用Kubernetes中的 [operator framework](https://cloud.redhat.com/blog/introducing-the-operator-framework)
来自动管理配置KAE所需的所有软件组件。这些组件包括 KAE 驱动程序、Kubernetes KAE 设备插件、节点标签管理等。

KAE Operator允许Kubernetes集群的管理员像管理集群中的CPU节点一样管理KAE节点。

请注意，KAE Operator特别适用于Kubernetes集群需要快速扩展的场景，例如部署额外的KAE节点，以及管理底层软件组件的生命周期。

## 入门

### 先决条件

- go version v1.21.0+
- docker version 17.03+
- kubectl version v1.11.3+
- Access to a Kubernetes v1.11.3+ cluster

### 使用

请查阅[产品文档](./docs/KAEOperator.md)
