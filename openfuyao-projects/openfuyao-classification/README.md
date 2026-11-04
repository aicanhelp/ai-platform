# openFuyao 代码分类分析

本目录按照五大维度对 openFuyao 仓库中的组件进行重新组织和分析：

1. **安装部署相关**（`01-installation-and-deployment`）
2. **openFuyao 通用基础平台**（`02-base-platform`）—— 不含昇腾等厂商专有逻辑
3. **昇腾专有**（`03-ascend-specific`）
4. **Kubernetes 生态上游依赖 / Fork**（`04-upstream-ecosystem`）—— `containerd`、`coredns`、`etcd`、`kubernetes`、`kwok`、`runc`
5. **文档、社区、治理与工程工具**（`05-documentation-and-tooling`）—— `docs`、各 `sig-*`、`ofep`、`release-management`、测试工具等

## 分类原则

- **安装部署类**：负责把 openFuyao 管理面从零部署到目标集群，或在节点生命周期中执行初始化命令。核心特征是它们使用 `bke.bocloud.com`、`bkeagent.bocloud.com`、`config.openfuyao.com` 等 openFuyao 自有安装器 CRD。
- **通用基础平台类**：控制台、插件框架、应用市场、Volcano 调度增强、NUMA 感知、Ray 集成、日志/监控/多集群等。这些能力与任何单一厂商芯片无关，面向所有异构算力。
- **昇腾专类**：名称、CRD、驱动、设备插件、虚拟化配置、监控采集规则等直接绑定华为昇腾 NPU / KAE / vNPU 的组件。
- **上游生态类**：`kubernetes`、`containerd`、`etcd`、`coredns`、`runc`、`kwok` 等 openFuyao 维护或集成的上游 Fork/依赖。它们是运行平台的底层支撑，但不含 openFuyao 自研业务逻辑。
- **文档与治理工具类**：`docs`、各 `sig-*`、`ofep`、`community`、`release-management`、测试与质量工具等，不直接构成生产运行时，但对项目组织、贡献流程和发布治理很重要。

## 目录结构

```text
openfuyao-classification/
├── README.md                        # 本文件
├── openfuyao-crd-analysis.md        # 总览：CRD/插件框架分析
├── openfuyao-projects-analysis.md   # 总览：通用框架提取思路
├── 01-installation-and-deployment/
│   ├── README.md                    # 该大类总览
│   ├── analysis.md                  # 详细分析
│   └── code/                        # 复制进来的源码
├── 02-base-platform/
│   ├── README.md
│   ├── analysis.md
│   └── code/
├── 03-ascend-specific/
│   ├── README.md
│   ├── analysis.md
│   └── code/
├── 04-upstream-ecosystem/
│   ├── README.md
│   ├── analysis.md
│   └── code/                        # Kubernetes / containerd / coredns 等上游 Fork
└── 05-documentation-and-tooling/
    ├── README.md
    ├── analysis.md
    └── code/                        # docs、社区、SIG、测试、开发者工具等
```

## 复制范围

- `01 ~ 03` 完整复制了对应功能组件的源码（`.git` 除外）。
- `04` 完整复制了 6 个上游 Fork/依赖项目的源码（`.git` 除外）。
- `05` 完整复制了文档、社区治理、SIG、测试及开发者工具的源码（`.git` 除外）。
- 仓库中已不存在未归入这五类的顶层目录。

## 参考文档

本目录已携带此前撰写的两份总览文档：

- `openfuyao-crd-analysis.md`：完整 CRD 与插件框架分析
- `openfuyao-projects-analysis.md`：通用管理框架提取思路
