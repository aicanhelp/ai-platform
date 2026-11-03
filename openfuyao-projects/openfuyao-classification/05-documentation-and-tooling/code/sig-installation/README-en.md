# sig-installation

Welcome to sig-installation!

## SIG Group Goals

sig-installation aims to build an open and easy-to-use Kubernetes cluster installation system, enabling zero-foundation users to install Kubernetes clusters with one click. The goals are as follows:

1. **Simplify complex deployment**: Lower the threshold for users to deploy production-grade Kubernetes clusters that follow best practices in diverse environments.
2. **Unified lifecycle management**: Provide a standardized toolchain from creation to daily operations, achieving **consistency and reproducibility** in management.
3. **Achieve automation and scalability**: Use declarative APIs through tools like `Cluster API` to automate the entire lifecycle of large-scale clusters, and support horizontal scaling to manage any number of clusters.

## Project Planning RoadMap (Updated 2026/01/13)

| **Planning Capability** | **Planning Goal** | **Planned Release** |
| --- | --- | --- |
| **Pre-checks** | - Provide comprehensive environment pre-check tools to identify cluster deployment failure points in advance | 26.X |
| **Declarative version install/upgrade framework** | - Version traceability, upgrade orchestration, compatibility verification | 26.X |
| **Cluster-level/node-level layered configuration management** | - Based on Node splitting to support node-level O&M capabilities, including but not limited to custom configuration changes and node-level RPM package installation | 26.X |
| **Enhanced install/upgrade framework** | - Support Helm Chart and binary package standard integration | 26.X |
| **Automated component lifecycle management (Operator-based)** | - Out-of-the-box, full lifecycle self-management of components | 27.X |
| **Enhanced Kubernetes component deployment** | - Support external etcd cluster<br> - Support planned deployment of key components (kube-controller-manager / scheduler / apiserver) on designated nodes | 27.X |
| **Simplified bootstrap node architecture** | - Reduce dependency on dedicated nodes, free resources for business workloads, lower delivery costs | 27.X |
| **Configurable component parameters** | - Support configurable containerd parameters during cluster installation<br>- Support configurable kubelet parameters during cluster installation | 25.12 |
| **Custom certificates and kubeconfig downgrade capability** | - Support user-imported global CA certificates for cluster-issued certificates<br>- Support kubeconfig issuance for different Kubernetes components | 25.12 |
| **Custom repository support for cluster installation** | - Support custom configuration of image and binary repositories, and install clusters based on them | 25.12 |
| **Bootstrap node user management capability** | - Bootstrap node provides user management capability, convenient for small cluster installation (cluster size <10 nodes) | 25.09 |
| **Full offline installation support** | - Fully support offline cluster installation | 25.09 |

## How to Contribute Code

Please refer to [Contribution Guide](CONTRIBUTING.md)

## Community Meetings

sig-installation is open, and our users, contributors, and maintainers continuously improve it. It is because of you that we can build better deployment tools. Join us!

- Installation and deployment SIG regular meetings are held every Wednesday morning at 10:00-11:00. You can get specific meeting agendas on the community official website: [https://www.openfuyao.cn/en/meeting/](https://www.openfuyao.cn/en/meeting/)
- SIG regular meeting topic application: https://etherpad.openfuyao.cn/p/release-management.
- You can subscribe to email through the community official website: [installation@list.openfuyao.cn](mailto:installation@list.openfuyao.cn) to subscribe to SIG regular meetings and get the latest SIG information.

## Proposal Submission & Archiving

Installation and deployment SIG related proposals (OFEP) should be submitted as PRs to: https://gitcode.com/openFuyao/ofep/tree/main/ofeps/sig-Installation.

## Repository List (Updated 2026/01/13)

repository address:

- [sig-installarion](https://gitcode.com/openFuyao/sig-installation)
- [bkeadm](https://gitcode.com/openFuyao/bkeadm)
- [cluster-api-provider-bke](https://gitcode.com/openFuyao/cluster-api-provider-bke)
- [bkeagent](https://gitcode.com/openFuyao/bkeagent)
- [bkecommon](https://gitcode.com/openFuyao/bkecommon)
- [bke-manifests](https://gitcode.com/openFuyao/bke-manifests)