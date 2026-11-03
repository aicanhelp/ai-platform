---
title: 【安全】支持K8S CIS与STIG不合规项整改
ofep-number: 0006
authors:
  - "@openfuyao-security"
owning-sig: security-committee
status: provisional
creation-date: 2026-08-03
reviewers:
approvers:
see-also:
  - "oFEP-0003"
replaces:
---

# oFEP-0006: 【安全】支持K8S CIS与STIG不合规项整改

<!-- toc -->

- [发布签核清单](#发布签核清单)
- [摘要](#摘要)
- [动机](#动机)
  - [目标](#目标)
  - [非目标](#非目标)
- [提案](#提案)
  - [用户故事（可选）](#用户故事可选)
  - [注释/约束/警告（可选）](#注释约束警告可选)
  - [风险与缓解措施](#风险与缓解措施)
- [设计细节](#设计细节)
  - [1. 静态 Pod Manifest 文件权限](#1-静态-pod-manifest-文件权限)
  - [2. kube-apiserver](#2-kube-apiserver)
  - [3. kube-controller-manager](#3-kube-controller-manager)
  - [4. kube-scheduler](#4-kube-scheduler)
  - [5. etcd](#5-etcd)
  - [6. kubelet](#6-kubelet)
  - [测试计划](#测试计划)
    - [先决条件测试更新](#先决条件测试更新)
    - [e2e 测试](#e2e-测试)
  - [毕业标准](#毕业标准)
  - [升级/降级策略](#升级降级策略)
  - [版本倾斜策略](#版本倾斜策略)
- [DFX 设计](#dfx-设计)
  - [可测试性](#可测试性)
  - [可诊断性](#可诊断性)
  - [可运维性](#可运维性)
  - [可靠性](#可靠性)
  - [性能](#性能)
  - [安全性](#安全性)
  - [兼容性](#兼容性)
- [生产准备情况评审问卷](#生产准备情况评审问卷)
  - [功能启用和回滚](#功能启用和回滚)
  - [推出、升级和回滚规划](#推出升级和回滚规划)
  - [监控要求](#监控要求)
  - [依赖项](#依赖项)
  - [可扩展性](#可扩展性)
  - [故障排除](#故障排除)
- [实施历史](#实施历史)
- [缺点](#缺点)
- [替代方案](#替代方案)

<!-- /toc -->

## 发布签核清单

标记有（R）的项目在达到里程碑/发布之前是必需的。

- \[ ]（R）发布里程碑中的增强问题，链接到 openfuyao/ofep 中的 oFEP 目录
- [ ] (R) oFEP 审批者已批准 oFEP 状态为“可实施”
- [ ] (R) 设计细节已适当记录
- \[ ]（R）测试计划已到位，并考虑了 SIG 架构和 SIG 测试的输入
- [ ] (R) 毕业标准已设定
- [ ] (R) 生产准备情况审查完成
- [ ] (R) 生产准备情况审查已获批准
- [ ] “实施历史”部分已更新里程碑

## 摘要

本提案基于 **compliance operator的CIS/STIG 的不合规项进行整改**，将现有扫描结果中的 `FAIL`  项，沉淀为一个**独立的安全加固与回退工具方案**，用于帮助用户在安装完成后按需执行安全整改。

本提案的核心定位是：

- 不做默认强制加固
- 不直接集成到现有安装部署流程
- 以独立工具脚本方式提供
- 支持一键式整改与一键式回退

本提案聚焦于将整改需求规范化、产品化，而不是仅保留一份人工执行命令清单。

## 动机

### 目标

- 将现有 `CIS + STIG` 扫描报告中的整改项沉淀为统一的安全加固方案，避免每次人工逐条执行命令。
- 提供一个独立的安全加固工具，在用户环境上执行安全加固整改。
- 工具需要支持：
  - 备份整改前配置
  - 回滚到整改前状态
  - 输出整改结果与失败原因
- 整改后系统功能应保持正常，并能通过Compliance Operator复扫验证整改效果。
- 对于不适合自动化整改、可能影响系统稳定性的项，要求单独标记并走评审备案。

### 非目标

- 不在本提案中强制要求所有安装实例默认执行加固。
- 不在本提案中实现 web UI，仅提供命令行工具方案。
- 不在本提案中定义完整的 CIS/STIG 基线规则内容，仅聚焦当前已识别的不合规项整改落地。
- 不承诺自动整改所有扫描项；对高风险或需人工判断的项，仅输出评审建议。

## 提案

总体方案如下：

- 提供一个独立的 **K8S CIS/STIG 整改工具**，以脚本或脚本集合形式交付。
- 用户在工具配置文件中填写环境信息后，显式执行整改。
- 工具按整改分组完成以下动作：
  - 修改静态 Pod manifest 权限
  - 调整 kube-apiserver / kube-controller-manager / kube-scheduler / etcd / kubelet 相关参数或配置
  - 修正相关目录、证书、配置文件的权限与属主
- 工具在整改前自动备份原始配置，在用户选择回退时恢复原始状态。
- 工具执行结束后输出整改报告，并建议用户重新执行一轮 `CIS + STIG` 扫描确认结果。

### 用户故事（可选）

#### 故事 1

安全管理员在一套已安装完成的集群上执行 `CIS + STIG` 扫描，发现若干不合规项。管理员不希望手工逐条修改配置，而是使用整改工具完成批量整改，并在必要时一键回退。

#### 故事 2

平台运维希望将“安装部署”和“安全加固”解耦。系统默认以兼容性优先，不强制修改安全参数；当用户需要加强安全基线时，再显式运行整改工具。

### 注释/约束/警告（可选）

- “加固”不是必须步骤，不能作为当前安装部署的前置依赖。
- 整改工具必须是独立工具，不与主安装流程强耦合。

### 风险与缓解措施

- 风险：某些整改项会影响控制面组件启动参数，导致集群异常。
  - 缓解：整改前必须备份；整改后逐组件做健康检查；提供一键回退。
- 风险：用户误以为整改后即可完全满足所有 CIS/STIG 条款。
  - 缓解：工具输出“已整改/未整改/需人工处理”三类结果，并要求复扫验证。工具的日志中不要输出任何敏感信息

## 设计细节

### 1. 静态 Pod Manifest 文件权限

涉及目录：`/etc/kubernetes/manifests/`

| 组件                      | 配置文件                                                     | 检查项                             | 来源          | 当前内容              | 原始要求                                                                             | 建议修改                                                                  |
| ----------------------- | -------------------------------------------------------- | ------------------------------- | ----------- | ----------------- | -------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| kube-apiserver          | `/etc/kubernetes/manifests/kube-apiserver.yaml`          | API Server Pod 规范文件权限过宽         | CIS `1.1.1` | `permissions=644` | 文件权限应为 `600` 或更严格，目的是限制非特权用户读取或篡改控制面静态 Pod 配置，降低 API Server 参数、证书挂载和安全开关被恶意修改的风险 | 执行 `chmod 600 /etc/kubernetes/manifests/kube-apiserver.yaml`          |
| kube-controller-manager | `/etc/kubernetes/manifests/kube-controller-manager.yaml` | Controller Manager Pod 规范文件权限过宽 | CIS `1.1.3` | `permissions=644` | 文件权限应为 `600` 或更严格，目的是防止控制器参数、凭据路径和集群控制逻辑被未授权读取或修改，保护控制面完整性                       | 执行 `chmod 600 /etc/kubernetes/manifests/kube-controller-manager.yaml` |
| kube-scheduler          | `/etc/kubernetes/manifests/kube-scheduler.yaml`          | Scheduler Pod 规范文件权限过宽          | CIS `1.1.5` | `permissions=644` | 文件权限应为 `600` 或更严格，目的是避免调度器配置、认证信息和监听参数被低权限用户获取或篡改，减少调度面被利用的风险                    | 执行 `chmod 600 /etc/kubernetes/manifests/kube-scheduler.yaml`          |
| etcd                    | `/etc/kubernetes/manifests/etcd.yaml`                    | etcd Pod 规范文件权限过宽               | CIS `1.1.7` | `permissions=644` | 文件权限应为 `600` 或更严格，目的是防止 etcd 启动参数、证书路径和存储配置被未授权读取或改写，保护集群核心数据存储的安全边界             | 执行 `chmod 600 /etc/kubernetes/manifests/etcd.yaml`                    |

### 2. kube-apiserver

配置文件：`/etc/kubernetes/manifests/kube-apiserver.yaml`

| 检查项                                           | 来源               | 当前内容                                                                                        | 原始要求                                                                                                                                                                                                                             | 建议修改                                                                                                                                                                                                                          |
| --------------------------------------------- | ---------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 未设置 `--kubelet-certificate-authority`         | CIS `1.2.5`      | 报告展示的当前启动参数中未出现 `--kubelet-certificate-authority`                                           | 必须显式设置 `--kubelet-certificate-authority=<ca-file>`，用于建立 API Server 与 kubelet 之间的 TLS 信任；这样可以防止 API Server 连接到伪造或不受信任的 kubelet，降低中间人攻击和节点冒充风险                                                                                     | 在 manifest 中增加 `--kubelet-certificate-authority=/etc/kubernetes/pki/ca.crt` 或你的实际 CA 路径                                                                                                                                       |
| 未启用 `NodeRestriction` 准入插件                    | CIS `1.2.14`     | 当前 `--enable-admission-plugins` 未出现，或未包含 `NodeRestriction`                                  | `--enable-admission-plugins` 必须包含 `NodeRestriction`；这样可以限制 kubelet 只能修改与自身节点相关的对象，防止被攻陷的节点越权操作其他节点或 Pod 资源                                                                                                                       | 增加或修改为 `--enable-admission-plugins=...,NodeRestriction,...`                                                                                                                                                                   |
| `service-account-extend-token-expiration` 未关闭 | CIS `1.2.30`     | 报告展示的当前启动参数中未出现 `--service-account-extend-token-expiration=false`                           | 必须显式设置 `--service-account-extend-token-expiration=false`；这样可以避免令牌被长期延展使用，缩短凭据泄露后的可利用时间窗口，降低横向移动风险                                                                                                                                | 在 manifest 中增加 `--service-account-extend-token-expiration=false`                                                                                                                                                              |
| TLS 最低版本未满足 STIG                              | STIG `SV-242378` | 扫描报告未直接给出当前值，但判定 API Server 未满足最小 TLS 版本要求；从当前启动参数看未显式设置 `--tls-min-version`                | 必须使用 `TLS 1.2` 或更高版本，通常要求显式设置 `--tls-min-version=VersionTLS12`；这样可以禁用已知存在弱点的旧版 SSL/TLS 协议，保护 API 通信的机密性和完整性                                                                                                                      | 在 manifest 中增加 `--tls-min-version=VersionTLS12`                                                                                                                                                                               |
| TLS Cipher Suites 不满足 STIG                    | STIG `SV-242418` | 当前参数中仅看到 `TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256` 两项 | 必须使用 STIG 允许的批准套件，至少包含 `TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384`；这样可以避免使用弱加密套件，提升 API Server 对传输窃听和降级攻击的防护能力 | 将 `--tls-cipher-suites` 补齐为 STIG 要求的 4 个套件                                                                                                                                                                                    |
| 审计策略内容不满足 STIG                                | STIG `SV-242403` | 当前已设置 `--audit-policy-file=/etc/kubernetes/audit-policy.yaml`，但策略文件内容仍被判定不满足要求              | 必须生成能记录事件类型、事件源、结果、用户及关联容器等审计记录；STIG 最低示例要求在审计策略中覆盖较高审计级别，例如 `RequestResponse`；这样可以提升事后追踪、异常调查和安全取证能力                                                                                                                            | 检查并更新 `/etc/kubernetes/audit-policy.yaml`，至少确保 `apiVersion: audit.k8s.io/v1`、`kind: Policy`、`rules` 覆盖关键请求；必要时按 STIG 示例采用 `RequestResponse` 级别，谨慎，这个级别记录内容太多，只做参考，不是必须，这条的要求是有记录策略，不是要求强制记录到requestResponse级别，这个记录东西太多，有其他风险。 |
| 未正确限制请求超时                                     | STIG `SV-242438` | 当前启动参数中未看到 `--request-timeout`                                                              | `--request-timeout` 必须为大于 `0` 的有效值，用于限制请求占用时间；这样可以减少长连接或异常请求持续占用 API Server 资源，降低拒绝服务和资源耗尽风险                                                                                                                                     | 在 manifest 中增加 `--request-timeout=1m` 或符合你环境要求的正值                                                                                                                                                                             |

#### kube-apiserver 建议重点检查参数

```yaml
- --kubelet-certificate-authority=/etc/kubernetes/pki/ca.crt
- --enable-admission-plugins=NodeRestriction,...
- --service-account-extend-token-expiration=false
- --tls-min-version=VersionTLS12
- --tls-cipher-suites=TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256,TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384,TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384
- --audit-policy-file=/etc/kubernetes/audit-policy.yaml
- --request-timeout=1m
```

### 3. kube-controller-manager

配置文件：`/etc/kubernetes/manifests/kube-controller-manager.yaml`

| 检查项              | 来源                             | 当前内容                                                 | 原始要求                                                                                                             | 建议修改                                |
| ---------------- | ------------------------------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 绑定地址暴露到非本地地址     | CIS `1.3.7` + STIG `SV-242385` | 当前为 `--bind-address=192.168.100.140`                 | `--bind-address` 必须为 `127.0.0.1`，或不显式暴露到非回环地址；这样可以减少控制器管理接口暴露面，避免外部网络直接探测或访问其监听端口                                | 改为 `--bind-address=127.0.0.1`       |
| TLS 最低版本未满足 STIG | STIG `SV-242376`               | 扫描报告未直接给出当前值，但判定未满足要求；当前参数中未显式设置 `--tls-min-version` | 必须使用 `TLS 1.2` 或更高版本，通常要求显式设置 `--tls-min-version=VersionTLS12`；这样可以避免 Controller Manager 使用过时协议，降低通信被降级、窃听或劫持的风险 | 增加 `--tls-min-version=VersionTLS12` |

#### kube-controller-manager 建议重点检查参数

```yaml
- --bind-address=127.0.0.1
- --tls-min-version=VersionTLS12
```

### 4. kube-scheduler

配置文件：`/etc/kubernetes/manifests/kube-scheduler.yaml`

| 检查项              | 来源                             | 当前内容                                                 | 原始要求                                                                                                 | 建议修改                                |
| ---------------- | ------------------------------ | ---------------------------------------------------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------- |
| 绑定地址暴露到非本地地址     | CIS `1.4.2` + STIG `SV-242384` | 当前为 `--bind-address=192.168.100.140`                 | `--bind-address` 必须为 `127.0.0.1`，或不显式暴露到非回环地址；这样可以减少调度器管理接口暴露面，降低被外部扫描、探测或滥用的概率                      | 改为 `--bind-address=127.0.0.1`       |
| TLS 最低版本未满足 STIG | STIG `SV-242377`               | 扫描报告未直接给出当前值，但判定未满足要求；当前参数中未显式设置 `--tls-min-version` | 必须使用 `TLS 1.2` 或更高版本，通常要求显式设置 `--tls-min-version=VersionTLS12`；这样可以提升 Scheduler 通信链路的机密性和完整性，避免使用弱协议 | 增加 `--tls-min-version=VersionTLS12` |

#### kube-scheduler 建议重点检查参数

```yaml
- --bind-address=127.0.0.1
- --tls-min-version=VersionTLS12
```

### 5. etcd

涉及对象：

- 配置文件：`/etc/kubernetes/manifests/etcd.yaml`
- 数据目录：以 `ps -ef | grep etcd` 中的 `--data-dir` 实际值为准，当前通常可按 `/var/lib/etcd` 理解

| 检查项                 | 来源               | 当前内容                                         | 原始要求                                                                                        | 建议修改                                                                                                                 |
| ------------------- | ---------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `etcd` 数据目录权限不满足要求  | CIS `1.1.11`     | 扫描报告未直接给出当前值，但判定目录权限未满足要求                    | `etcd` 数据目录权限必须为 `700` 或更严格；这样可以限制只有授权用户和进程访问集群核心状态数据，降低密钥、对象元数据和状态信息泄露的风险                  | 查出 `--data-dir` 实际路径后执行 `chmod 700 <data-dir>`，例如 `chmod 700 /var/lib/etcd`                                          |
| `etcd` 数据目录属主不满足要求  | CIS `1.1.12`     | 扫描报告未直接给出当前值，但判定目录属主不是 `etcd:etcd`           | `etcd` 数据目录属主属组必须为 `etcd:etcd`；这样可以确保只有 etcd 服务账户拥有必要访问权限，减少其他账户误用或恶意操作数据目录的机会              | 执行 `chown etcd:etcd <data-dir>`，例如 `chown etcd:etcd /var/lib/etcd`                                                   |
| 启用了不合规的自动 TLS       | STIG `SV-242379` | 扫描报告未直接给出当前值，但判定 `--auto-tls` 未满足要求          | `--auto-tls` 必须设置为 `false`，应使用正式证书而不是自动生成 TLS；这样可以避免使用缺乏统一信任管理的自签自动证书，提升 etcd 通信的可验证性和抗冒充能力 | 在 `etcd.yaml` 中显式设置 `--auto-tls=false`                                                                               |
| 启用了不合规的 peer 自动 TLS | STIG `SV-242380` | 扫描报告未直接给出当前值，但判定 `--peer-auto-tls` 未满足要求     | `--peer-auto-tls` 必须设置为 `false`，集群间 TLS 也应使用正式证书；这样可以增强 etcd 节点间身份校验，防止伪造节点或不可信对端加入通信       | 在 `etcd.yaml` 中显式设置 `--peer-auto-tls=false`                                                                          |
| `etcd` 数据文件属主不满足要求  | STIG `SV-242445` | 扫描报告未直接给出当前值，但判定 `/var/lib/etcd/*` 文件属主不符合要求 | `etcd` 数据文件属主属组必须为 `etcd:etcd`；这样可以把对 etcd 数据文件的修改权限限定在服务账户范围内，防止其他本地账户篡改控制面数据              | 执行 `chown etcd:etcd /var/lib/etcd/*`；如数据目录不同，请替换为实际路径这条谨慎，先看看能实现不，涉及etcd的运行，强行增加etcd用户，还要修改etcd的runasuser，可能会影响系统运行。 |
| `etcd` 数据文件权限不满足要求  | STIG `SV-242459` | 扫描报告未直接给出当前值，但判定 `/var/lib/etcd/*` 文件权限不符合要求 | `etcd` 数据文件权限必须为 `644` 或更严格；这样可以减少不必要写权限，降低核心状态数据被本地恶意进程或误操作破坏的风险                           | 执行 `chmod 644 /var/lib/etcd/*`；如数据目录不同，请替换为实际路径                                                                      |

#### etcd 建议先确认的运行参数

```bash
ps -ef | grep etcd
```

重点确认：

- `--data-dir`
- `--auto-tls`
- `--peer-auto-tls`

### 6. kubelet

涉及对象：

- kubelet 启动参数 / `systemd` drop-in
- kubelet 配置文件：以 `ps -ef | grep kubelet` 中 `--config` 指向的路径为准，常见是 `/var/lib/kubelet/config.yaml`
- kubelet 使用的 CA、证书、私钥文件

| 检查项                       | 来源               | 当前内容                                                                | 原始要求                                                                                                                             | 建议修改                                                        |
| ------------------------- | ---------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| 可能启用了 `hostname override` | STIG `SV-242404` | 该项为 `UNKNOWN`，报告要求进一步确认；需检查 kubelet 启动参数中是否存在 `--hostname-override` | kubelet 不应启用 `--hostname-override`；这样可以避免节点以伪造主机名注册，防止破坏节点身份与证书校验的一致性，降低冒充和错误绑定风险                                                | 检查 `systemd` 配置和 kubelet 启动参数，若存在则删除该参数                     |
| kubelet 配置文件属主不正确         | STIG `SV-242406` | 扫描报告未直接给出当前值，但判定 kubelet 配置文件不属于 `root`                             | kubelet 配置文件必须由 `root:root` 拥有；这样可以防止本地非特权用户篡改节点运行参数，避免节点被恶意放宽安全限制                                                               | 找到 `--config` 指向的文件后执行 `chown root:root <kubelet-config>`   |
| 未正确配置 `clientCAFile`      | STIG `SV-242420` | 扫描报告未直接给出当前值，但判定 `clientCAFile` 未满足要求；同时提示需要从命令行参数迁移到配置文件           | kubelet 必须设置 `clientCAFile`，指向批准的组织 CA 文件；不应仅依赖 `--client-ca-file` 命令行参数；这样可以确保 kubelet 只信任受控 CA 签发的客户端证书，提升节点访问控制和双向 TLS 身份校验能力 | 在 kubelet 配置文件中设置 `clientCAFile: /path/to/ca.crt`，并确保路径可读   |
| kubelet CA 文件权限不正确        | STIG `SV-242449` | 扫描报告未直接给出当前值，但判定 `clientCAFile` 指向的文件权限不满足要求                        | kubelet CA 文件权限必须为 `644` 或更严格；这样可以避免 CA 文件被不当修改，同时保留 kubelet 正常读取能力，保护信任链配置完整性                                                   | 对 `clientCAFile` 指向的文件执行 `chmod 644 <ca-file>`              |
| kubelet CA 文件属主不正确        | STIG `SV-242450` | 扫描报告未直接给出当前值，但判定 `clientCAFile` 指向的文件不属于 `root`                     | kubelet CA 文件属主属组必须为 `root:root`；这样可以把信任根文件的控制权限定给系统管理员，降低本地用户替换 CA 造成信任劫持的风险                                                    | 对 `clientCAFile` 指向的文件执行 `chown root:root <ca-file>`        |
| 未正确配置 `tlsPrivateKeyFile` | STIG `SV-242424` | 扫描报告未直接给出当前值，但判定 kubelet 未满足私钥配置要求                                  | kubelet 必须通过配置文件设置 `tlsPrivateKeyFile`，指向正确的私钥文件；这样可以确保 kubelet 通过受管私钥提供服务端身份，避免临时或错误密钥导致身份校验失效                                  | 在 kubelet 配置文件中设置 `tlsPrivateKeyFile: /path/to/kubelet.key` |
| 未正确配置 `tlsCertFile`       | STIG `SV-242425` | 扫描报告未直接给出当前值，但判定 kubelet 未满足证书配置要求                                  | kubelet 必须通过配置文件设置 `tlsCertFile`，指向正确的服务证书文件；这样可以保证 kubelet 对外服务时使用受信证书，增强节点接口通信的机密性与身份真实性                                       | 在 kubelet 配置文件中设置 `tlsCertFile: /path/to/kubelet.crt`       |
| 未启用内核默认保护                 | STIG `SV-242434` | 扫描报告未直接给出当前值，但判定 `protectKernelDefaults` 未满足要求                      | kubelet 必须设置 `protectKernelDefaults: true`；这样可以防止 Pod 或节点运行时轻易覆盖宿主机内核默认安全参数，降低宿主机被容器逃逸或错误内核调优影响的风险                               | 在 kubelet 配置文件中设置 `protectKernelDefaults: true`             |
| 流式连接空闲超时不合规               | STIG `SV-245541` | 扫描报告未直接给出当前值，但判定 `streamingConnectionIdleTimeout` 不满足要求             | kubelet 必须设置 `streamingConnectionIdleTimeout`，建议值为 `5m`，不能被关闭；这样可以避免空闲流式连接长期存在并被滥用，减少资源占用和未授权会话持续驻留的风险                           | 在 kubelet 配置文件中设置 `streamingConnectionIdleTimeout: 5m`      |

#### kubelet 建议重点检查配置

```yaml
clientCAFile: /etc/kubernetes/pki/ca.crt
tlsCertFile: /var/lib/kubelet/pki/kubelet.crt
tlsPrivateKeyFile: /var/lib/kubelet/pki/kubelet.key
protectKernelDefaults: true
streamingConnectionIdleTimeout: 5m
```

### kubelet 建议先确认的运行参数

```bash
ps -ef | grep kubelet
```

重点确认：

- `--config`
- `--hostname-override`
- `--client-ca-file`
- `--tls-cert-file`
- `--tls-private-key-file`


### DFX 设计

本提案交付形态为独立的整改/回退工具，主要在运维窗口由用户显式触发执行，不属于长期驻留服务，因此 DFX 重点放在可测试性、可诊断性、可运维性、可靠性、安全性与兼容性。

#### 可测试性

设计方案：

- 支持 `dry-run` 模式：仅输出将要修改的文件、字段与权限变更，不落盘修改。

验收标准：

- `dry-run` 输出包含：命中规则列表、拟修改的目标路径、拟修改的关键字段/权限、将被写入的值或权限位。


#### 可诊断性

设计方案：

- 对失败项输出可定位的最小必要信息：目标文件路径、、错误码等信息。
- 明确日志与报告不输出敏感信息（证书内容、私钥、token、完整 kubeconfig 等），仅输出必要的路径与摘要信息。

验收标准：

- 任意一条规则执行失败时，输出中可明确定位到“哪条规则、哪一步骤、哪个目标对象、失败原因”，并给出可操作的下一步建议。

#### 可运维性

设计方案：

- 工具提供清晰的命令入口与参数模型：`apply`（整改）、`rollback`（回退）。
- 明确备份策略：备份目录命名规则、保留策略、覆盖策略，以及如何基于备份进行回退与人工恢复。

验收标准：

- 对同一集群重复执行 `apply` 不应造成不可控的重复改写；重复执行应得到一致结果或明确提示“已是目标状态”。

#### 可靠性

不涉及：该工具为离线工具，不提供双机热备等可靠性机制，仅在运维窗口由用户显式触发执行。

#### 性能

不涉及：该工具为离线运维工具，不承载在线高并发请求路径，本提案不做专项性能优化设计。

#### 安全性

设计方案：

- 默认最小化变更面：仅修改已明确映射到 CIS/STIG 的目标项；。
- 对备份与报告文件设置合理访问控制，可控制收敛到只有root可以访问，避免因备份导致敏感配置被非授权读取。

验收标准：

- 执行日志中不出现敏感内容明文；抽样检查包含证书/私钥相关项时，输出仅允许出现路径与必要摘要。
- 对工具修改过的目标文件，权限/属主满足整改目标且不引入更宽松的权限。


#### 兼容性

设计方案：

不涉及：该工具为离线运维工具，不依赖于任何网络服务或外部配置，所以不存在兼容历史版本的问题。

### 测试计划

\[ ] 我/我们理解，相关组件所有者可能会要求在实现前补充或重构测试。

##### 先决条件测试更新

- 准备一套openFuyao社区发行版的环境
- 部署上Compliance operator，执行 `CIS + STIG` 扫描

  <br />

#

##### e2e 测试

- 执行整改工具后重新运行 `CIS + STIG` 扫描，验证整改项减少。
- 验证回退后集群恢复到整改前状态。
- <br />

### 毕业标准

- Alpha：能覆盖当前已识别的主要整改项，并支持备份、整改、回退、结果输出。
- Beta：在典型部署环境完成验证，整改后系统功能正常，复扫结果符合预期。
- Stable：形成稳定的独立安全整改工具与运维文档，支持生产场景使用。

### 升级/降级策略

- 不涉及，这个是一个独立的安全加固工具

### 版本倾斜策略

- 不涉及，工具不随openFuyao发行版发布，独立演进。



## 生产准备情况评审问卷

### 功能启用和回滚

- 启用：用户手工运行整改工具，不自动启用。
- 回滚：用户手工运行回退工具，恢复整改前状态。

### 推出、升级和回滚规划

- 整改失败可能导致控制面组件无法启动，因此必须先做备份并分组执行。
- 回滚不应影响未被本工具修改的其他系统配置。

### 监控要求

<br />

- 整改完成后需校验以下健康信号：
  - 控制面组件进程/静态 Pod 状态
  - kubelet 正常运行
  - etcd 正常工作

### 依赖项

### 可扩展性

- 后续可扩展：
  - 输出整改前后差异报告

    <br />

### 故障排除

- 若整改后组件异常，优先查看：
  - 工具执行日志
  - 被修改的 manifest/config 差异
  - kubelet 日志
  - 控制面静态 Pod 状态
- 若复扫结果未改善，优先检查：
  - 是否恢复至整改前效果
  - <br />

    <br />

## 实施历史

- 2026-08-03：基于扫描整改汇总文档整理形成 oFEP 初稿（provisional）。

## 缺点

## 替代方案

- 仅保留一份人工整改清单，不提供工具：实现简单，但执行成本高、易漏项、不可回退。

  <br />

##

