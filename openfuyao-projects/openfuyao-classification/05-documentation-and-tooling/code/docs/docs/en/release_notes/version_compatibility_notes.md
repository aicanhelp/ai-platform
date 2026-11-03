---
sidebar_position: 5
---

# Version Compatibility Notes

## Release Artifacts

openFuyao release artifacts are installation packages, which can be obtained from the official website ["Package Download"](https://www.openfuyao.cn/en/download/).

**Table 1** openFuyao distribution v26.09 installation package list

<table>
  <thead>
    <tr>
      <th>Installation Method</th>
      <th>Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowSpan="2">Cluster-API</td>
      <td>bkeadm_Linux_amd64.tar.gz</td>
      <td>openFuyao installation package for x86_64 architecture</td>
    </tr>
    <tr>
      <td>bkeadm_Linux_arm64.tar.gz</td>
      <td>openFuyao installation package for ARM64 architecture</td>
    </tr>
  </tbody>
</table>

**Table 2** openFuyao distribution v26.09 Chart package list
| **Package Name** | **Description** |
|---|---|
| infernex | AI inference acceleration suite |
| eagle-eye | AI inference observability |
| hermes-router | AI inference intelligent routing |
| cache-indexer | AI inference KVCache index management |
| pd-orchestrator | Elastic scheduling component set |
| weight-dispatcher | Model weight distribution |
| kae-operator | KAE access enablement |
| logging-package | Logging component |
| colocation-package | Online-offline co-location scheduling enhancement |
| many-core-orchestrator | Many-core co-location scheduling |
| matrixagent | Memory borrowing node-side functionality |
| matrixcontroller | Memory borrowing center-side functionality |
| matrixshm | Shared memory CSI |
| superpod-exporter | Super node observability |
| monitoring-dashboard | Custom monitoring dashboard |
| multi-cluster-service | Multi-cluster management |
| npu-operator | NPU access enablement |
| numa-affinity-package | NUMA affinity scheduling |
| ray-package | openFuyao/ray |
| compliance-operator | Compliance scanning |
| opensandbox | Sandbox access |
| fluxsandbox | High-performance sandbox scheduling engine |
| vNPU | NPU virtualization |
| npu-dra-plugin | DRA plugin compatible with Ascend NPU |
| kubevirt | Container VM co-management component |

> ![icon](figures/icon-note.gif)**Note:**<br />
> Some incubating components are not yet equipped with Chart packages; their artifacts only provide images or require manual build/compilation. For components that provide Chart packages, their image-related configurations are included in the Chart packages.

**Table 3** openFuyao distribution v26.09 standalone component image list

| Image Name | Image Address | Description        |
|---|---|-----------|
| npu-dra-plugin | cr.openfuyao.cn/npu-dra-plugin:1.0.0 | DRA plugin compatible with Ascend NPU |
| ub-network-device-plugin | cr.openfuyao.cn/ub-network-device-plugin:1.0.0 | Lingqu network device plugin  |

High-density container images need to be built manually. Build guidance is provided in the [User Guide](https://gitcode.com/openFuyao/sig-container-platform/blob/master/docs/en/kunpeng_confidential_container/user_guide/kunpeng_confidential_container.md).

## Operating System Version Compatibility

**Table 4** openFuyao distribution v26.09 supported scenarios

| Operating System      | Version            | Architecture           | Linux Kernel Version |
|-----------|---------------|--------------|-----------|
| openEuler | 20.03 LTS     | ARM64, x86_64 | 4.19      |
| openEuler | 20.03 LTS SP3 | ARM64, x86_64 | 4.19      |
| openEuler | 22.03 LTS SP3 | ARM64, x86_64 | 5.10      |
| openEuler | 24.03 LTS SP1 | ARM64, x86_64 | 6.6       |
| Ubuntu    | 22.04         | ARM64, x86_64 | 5.15      |
| UOS       | 20 1070e      | ARM64, x86_64 | 4.19, 6.6 |

> ![icon](./figures/icon-note.gif) **Note:**<br />
>
> - Mixed deployment of different OSs within the same cluster is not supported.
>
> - Mixed deployment of different architectures (ARM64, x86_64) is supported.

## Kubernetes Version Compatibility

The openFuyao distribution v26.09 installs openFuyao Kubernetes by default and provides full support for `Kubernetes 1.34`.
- kube-apiserver: v1.34.3
- kube-controller-manager: v1.34.3
- kube-scheduler: v1.34.3
- etcd: v3.6.7
- kube-proxy: v1.34.3
- kubelet: v1.34.3
- kubectl: v1.34.3
- containerd: v2.1.1

> ![icon](./figures/icon-note.gif) **Note:**<br />
> For the operating system and Kubernetes version requirements of each feature component in openFuyao, refer to the user guide of each feature component.
