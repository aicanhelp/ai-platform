---
sidebar_position: 5
---

# 版本配套说明

## 发布件

openFuyao发布件为安装软件包，安装软件包可以从官网[“软件包下载”](https://www.openfuyao.cn/zh/download/)处获取。

**表 1** openFuyao发行版v26.09版本安装软件包列表

<table>
  <thead>
    <tr>
      <th>安装方式</th>
      <th>名称</th>
      <th>描述</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowSpan="2">Cluster-API</td>
      <td>bkeadm_Linux_amd64.tar.gz</td>
      <td>x86_64架构的openFuyao安装包</td>
    </tr>
    <tr>
      <td>bkeadm_Linux_arm64.tar.gz</td>
      <td>ARM64架构的openFuyao安装包</td>
    </tr>
  </tbody>
</table>

**表 2** openFuyao发行版v26.09版本Chart包列表
| **包名称** | **描述** |
|---|---|
| infernex | AI推理加速套件 |
| eagle-eye | AI推理可观测 |
| hermes-router | AI推理智能路由 |
| cache-indexer | AI推理KVCache索引管理 |
| pd-orchestrator | 弹性调度组件集 |
| weight-dispatcher | 模型权重分发 |
| kae-operator | KAE接入使能 |
| logging-package | 日志组件 |
| colocation-package | 在离线混部调度增强 |
| many-core-orchestrator | 众核混部调度 |
| matrixagent | 内存借用节点侧功能 |
| matrixcontroller | 内存借用中心侧功能 |
| matrixshm | 共享内存CSI |
| superpod-exporter | 超节点可观测 |
| monitoring-dashboard | 自定义监控看板 |
| multi-cluster-service | 多集群管理 |
| npu-operator | NPU接入使能 |
| numa-affinity-package | numa亲和调度 |
| ray-package | openFuyao/ray |
| compliance-operator | 合规扫描 |
| opensandbox | 沙箱接入 |
| fluxsandbox | 高性能沙箱调度引擎 |
| vNPU | NPU虚拟化 |
| npu-dra-plugin | 亲和昇腾NPU的DRA插件 |
| kubevirt | 容器虚机共管组件 |

> ![输入图片说明](figures/icon-note.gif)**说明：**<br />
> 部分孵化期组件暂未配备Chart包，制品仅提供镜像或需手动构建编译；提供Chart包的组件，其镜像相关配置包含Chart包中。

**表 3** openFuyao发行版v26.09版本独立组件镜像列表

| 镜像名称 | 镜像地址 | 描述        |
|---|---|-----------|
| npu-dra-plugin | cr.openfuyao.cn/npu-dra-plugin:1.0.0 | 亲和昇腾NPU的DRA插件 |
| ub-network-device-plugin | cr.openfuyao.cn/ub-network-device-plugin:1.0.0 | 灵衢网络设备插件  |

高密容器镜像需手动构建，[用户指南](https://gitcode.com/openFuyao/sig-container-platform/blob/master/docs/zh/kunpeng_confidential_container/user_guide/kunpeng_confidential_container.md)中提供了构建指导

## 操作系统版本配套说明

**表 4** openFuyao发行版v26.09版本支持场景

| 操作系统      | 版本            | 架构           | Linux内核版本 |
|-----------|---------------|--------------|-----------|
| openEuler | 20.03 LTS     | ARM64、x86_64 | 4.19      |
| openEuler | 20.03 LTS SP3 | ARM64、x86_64 | 4.19      |
| openEuler | 22.03 LTS SP3 | ARM64、x86_64 | 5.10      |
| openEuler | 24.03 LTS SP1 | ARM64、x86_64 | 6.6       |
| Ubuntu    | 22.04         | ARM64、x86_64 | 5.15      |
| UOS       | 20 1070e      | ARM64、x86_64 | 4.19、6.6 |

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
>
> - 不支持同一集群内不同OS混部。
>
> - 支持不同架构（ARM64、x86_64）的混部。

## Kubernetes版本配套说明

openFuyao发行版v26.09版本默认安装openFuyao Kubernetes，对`Kubernetes 1.34`版本提供完整支持。
- kube-apiserver：v1.34.3
- kube-controller-manager：v1.34.3
- kube-scheduler：v1.34.3
- etcd：v3.6.7
- kube-proxy：v1.34.3
- kubelet：v1.34.3
- kubectl：v1.34.3
- containerd：v2.1.1

> ![输入图片说明](./figures/icon-note.gif) **说明：**<br />
> openFuyao中各特性组件对操作系统和Kubernetes版本的要求，参见各特性组件的用户指南。
