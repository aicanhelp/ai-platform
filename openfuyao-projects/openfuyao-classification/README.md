# openFuyao 代码分类分析

本目录按照五大维度对 openFuyao 仓库中的组件进行重新组织和分析：

1. **安装部署相关**（`01-installation-and-deployment`）
2. **openFuyao 通用基础平台**（`02-base-platform`）—— 不含昇腾等厂商专有逻辑
3. **昇腾专有**（`03-ascend-specific`）
4. **Kubernetes 生态上游依赖 / Fork**（`04-upstream-ecosystem`）—— `containerd`、`coredns`、`etcd`、`kubernetes`、`kwok`、`runc`
5. **文档、社区、治理与工程工具**（`05-documentation-and-tooling`）—— `docs`、各 `sig-*`、`ofep`、`release-management`、测试工具等

## 分类原则

- **安装部署类**：负责把 openFuyao 管理面从零部署到目标集群，或在节点生命周期中执行初始化命令。核心特征是它们使用 `bke.bocloud.com`、`bkeagent.bocloud.com`、`config.openfuyao.com` 等车openFuyao 自有安装器 CRD。
- **通用基础平台类**：控制台、插件框架、应用市场、Volcano 调度增强、NUMA 感知、Ray 集成、日志/监控/多集群等。这些能力与任何单一厂商芯片无关，面向所有异构算力。
- **昇腾专类**：名称、CRD、驱动、设备插件、虚拟化配置、监控采集规则等直接绑定华为昇腾 NPU / KAE / vNPU 的组件。

## 目录结构

```text
openfuyao-classification/
├── README.md                        # 本文件
├── 01-installation-and-deployment/
│   ├── README.md                    # 该大类总览
│   ├── analysis.md                  # 详细分析
│   └── code/                        # 复制进来的源码
├── 02-base-platform/
│   ├── README.md
│   ├── analysis.md
│   └── code/
└── 03-ascend-specific/
    ├── README.md
    ├── analysis.md
    └── code/
```

## 未复制源码的巨型上游依赖

仓库中还包含 Kubernetes、containerd、etcd、runc、coredns、kwok 等纯上游项目或 Fork。它们体量巨大且并非 openFuyao 自研逻辑，因此只在对应分类的 `analysis.md` 中列为底层依赖，**没有复制完整源码**。

## 参考文档

本目录已携带此前撰写的两份总览文档：

- `openfuyao-crd-analysis.md`：完整 CRD 与插件框架分析
- `openfuyao-projects-analysis.md`：通用管理框架提取思路
