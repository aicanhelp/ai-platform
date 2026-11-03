# kae-operator

Kubernetes can access Kunpeng [KAE accelerators](https://www.hikunpeng.com/developer/techArticles/20230906-4?envFlag=1)
and other special hardware resources through device plugin framework. However, configuring and managing nodes with these hardware resources requires configuring multiple software components, such as driver programs, container runtime, or other libraries, which is both difficult and error-prone.

KAE Operator uses the [operator framework](https://cloud.redhat.com/blog/introducing-the-operator-framework) in Kubernetes to automatically manage all software components required by KAE. These components include KAE driver program, Kubernetes KAE device plugin, node label management, etc.

KAE Operator allows Kubernetes cluster administrators to manage KAE nodes like managing CPU nodes in the cluster.

Please note that KAE Operator is particularly suitable for scenarios where Kubernetes clusters need rapid expansion, such as deploying additional KAE nodes, and managing the lifecycle of underlying software components.

## Prerequisites

- go version v1.21.0+.
- docker version 17.03+.
- kubectl version v1.11.3+.
- Access to a Kubernetes v1.11.3+ cluster.

## Usage

Please refer to [Product Documentation](./docs/KAEOperator.md)