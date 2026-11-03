# 更新 etcd 静态 Pod 配置

> 适用对象：**仅更新 etcd 静态 Pod 配置**（`etcd.extraArgs`）的场景。
> 若需要更新 `kube-apiserver` / `kube-controller-manager` / `kube-scheduler` 的参数，请参阅
> [更新管控面静态 Pod 配置](./ensure_master_config_update_operation_guide.md)。
> 设计说明见 [声明式集群静态 Pod 配置更新方案](../../../../design/支持静态pod配置热更新/声明式集群静态pod配置更新方案.md)。

---

## 1. 功能概述

etcd 是 Kubernetes 集群的底层键值存储，几乎所有集群状态都保存在其中。在集群运行过程中，运维人员有时需要调整 etcd 的运行参数，例如：调大 `quota-backend-bytes`（后端数据库配额）、调整心跳 / 选举超时、调整快照频率等。

本特性面向**集群运维人员 / SRE**，适用于以下场景：**在 etcd 版本不变、无需执行完整升级流程的前提下，仅修改 etcd 的运行参数（`extraArgs`），使变更安全、最小化地生效。**

本特性提供的能力：

- **声明式**：直接修改 `BKECluster`（`BKECluster` / bc）中 `etcd.extraArgs`（`kubectl edit` / `kubectl patch` / `kubectl apply` 任一方式均可），由控制器自动下发，无需登录节点手工修改文件。
- **最小化影响**：只重新生成 etcd 的 manifest，不影响管控面三组件、不影响业务节点。
- **可控**：逐 etcd 节点串行更新，单节点成功就绪后再更新下一个，降低风险。

### 1.1 相关概念

- **静态 Pod 是什么**：静态 Pod 的 manifest 是节点上 `/etc/kubernetes/manifests/` 目录下的普通 YAML 文件（如 `/etc/kubernetes/manifests/etcd.yaml`）。该文件由**节点上的 kubelet 直接读取并管理**，**不经过 API Server**：它并非像普通 Pod 那样通过 `kubectl` 提交至集群、再由调度器调度，而是由 kubelet 持续监听该目录，一旦文件内容发生变化即自动重建对应的 Pod（先删除旧 Pod，再创建新 Pod）。
- **为什么和普通 Pod 不同**：普通 Pod 的创建、更新与删除均由 API Server 及相应的控制器管理，修改配置通常通过 `kubectl edit` 或更新 Deployment 完成。而静态 Pod 的"配置"即磁盘上的 manifest 文件，修改参数的本质是**改写该文件并由 kubelet 重新拉起组件**；静态 Pod 不保存在 etcd 中，也不受任何 Workload 控制器管理，因而不具备"更新 Deployment 即触发滚动更新"的能力，变更必须直接作用于节点文件系统。
- **etcd 也是静态 Pod**：管控面四组件 `kube-apiserver`、`kube-controller-manager`、`kube-scheduler`、`etcd` 全部以静态 Pod 方式运行（前三者由 [更新管控面静态 Pod 配置](./ensure_master_config_update_operation_guide.md) 处理）。这正是**为何需要独立特性来变更 etcd 参数**：该特性的本质是"在每个 etcd 节点上改写 `/etc/kubernetes/manifests/etcd.yaml`，并等待 kubelet 重新拉起 etcd"，而非常规的 Pod 变更流程。

### 1.2 特点

- **独立触发**：不依赖、也不影响管控面三组件，支持 etcd 独立部署/单独挂载的场景。
- **不做 etcd 备份**：与版本升级不同，纯配置更新不触发备份动作。
- **逐节点串行**：前一个 etcd 节点成功并 Ready 后，才下发下一个节点。
- **失败可重试**：只有全部 etcd 节点成功后才回写 `status.etcdConfigHash`；失败保留旧基线，下轮自动重试。

### 1.3 执行流程（内部实现，供排查参考）

1. 修改 bc（`kubectl apply` / `kubectl edit` / `kubectl patch` 任一方式）→ Webhook `validateStaticPodConfigUpdate` 校验：管控面三组件必须同时存在、etcd 不可变字段未变更（校验不通过则本次写操作直接失败）。
2. capbke Controller `Reconcile` → `PhaseFlow` 依次调用各 phase 的 `NeedExecute`。
3. `EnsureEtcdConfigUpdate.NeedExecute` 判定：控制面已初始化、etcd/K8s 版本已对齐、存在 etcd 节点、期望哈希非空、基线非空且**与期望哈希不同** → 触发。
4. `Execute`：逐 etcd 节点串行下发 `UpgradeEtcdConfig` 命令并等待。
5. 节点 bkeagent：`updateEtcdConfig()` → 记录当前 etcd Pod Hash → 重新渲染 `/etc/kubernetes/manifests/etcd.yaml` → `waitComponentReady`（等 Pod Hash 变化 + Pod Running）。
6. **全部** etcd 节点成功后回写 `status.etcdConfigHash`；任一节点失败则保留旧基线，下轮 reconcile 自动重试。

---

## 2. 前提条件

| 编号 | 条件 | 检查方法 |
| --- | --- | --- |
| 1 | 集群已部署完成且健康（控制面已初始化） | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}'` |
| 2 | **基线哈希已建立**：`status.etcdConfigHash` 非空 | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'` |
| 3 | 集群存在 etcd 角色的 BKENode | `kubectl get bkenode -l ... `（外置 etcd 集群无 etcd BKENode，本特性不生效） |
| 4 | 准备变更的**只有** `etcd.extraArgs`，`etcdVersion` 与 `kubernetesVersion` **保持不变** | 见 [4.2 修改 etcd 参数](#42-修改-etcd-参数) |
| 5 | 集群当前没有正在进行的其它 phase（升级、扩缩容等） | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}'` |

> **关于条件 2**：新建集群在完成一次健康检查（`EnsureCluster`）后会自动建立基线；**存量集群**升级到含本特性的版本后，也需要先经过一轮健康检查才会写入基线。若 `status.etcdConfigHash` 为空，修改参数**不会**触发更新（这是为了防止部署阶段误刷静态 Pod）。

> **关于「基线哈希」（`status.etcdConfigHash`）**：它记录的是**上一次成功下发到节点静态 Pod 的那份配置**的哈希值，即"已生效配置基线"。哈希由**所有参与哈希计算的字段**计算得出，其中 `etcd.extraArgs` 真正渲染进 manifest，而 `etcd.extraVolumes` 虽**参与哈希**（改动会改变哈希值，从而触发本 phase），但**不会渲染**到 manifest。etcd 侧**目前仍会校验哈希**——只有期望哈希与基线哈希不一致时才会触发本 phase。每次 `apply` 后，控制器会用当前 `spec` 算出**期望哈希**；只有当**期望哈希 ≠ 基线哈希**时才会重新下发。更新全部 etcd 节点成功后，基线会被回写为最新期望哈希；一旦对齐，后续将不再重复触发。

---

## 3. 使用限制

### 3.1 只有 `extraArgs` 真正生效

etcd 静态 Pod 的 manifest 模板当前**只渲染 `extraArgs`**：

```yaml
# utils/bkeagent/mfutil/tmpl/k8s/etcd.yaml.tmpl（节选）
- command:
  - etcd
  ...
  {{- range . | extraArgs }}
  - --{{ . }}
  {{- end }}
```

`extraArgs` 中的每一项按 `--<key>=<value>` 追加到 etcd 启动命令末尾；与模板内置参数冲突时 **用户参数 last-wins（覆盖内置值）**。

下列字段虽然**参与哈希计算**（改动会触发更新流程），但**不会渲染到 manifest**，运维时请**不要**依赖它们：

> <span style="color:red;font-weight:bold;">注意：对于未渲染到 manifest 的字段（如 `etcd.extraVolumes`），更新流程仍会完整执行 `waitComponentReady`，并将新的哈希值回写至 Status，流程显示为"成功"；但目标配置实际并未生效（例如卷并未挂载）。此后哈希已对齐，本 phase 不再触发重试，最终呈现为"配置已修改、流程显示成功、但实际未生效"的结果。</span>

| 字段 | 参与哈希 | 是否渲染到 manifest |
| --- | :---: | :---: |
| `etcd.extraArgs` | ✅ | ✅ **唯一真正生效** |
| `etcd.extraVolumes` | ✅ | ❌ 社区暂未落地 |
| `etcd.dataDir` | ❌ | ❌ 引导期不可变，Webhook 拒绝变更 |
| `etcd.serverCertSANs` | ❌ | ❌ 引导期不可变，Webhook 拒绝变更 |
| `etcd.peerCertSANs` | ❌ | ❌ 引导期不可变，Webhook 拒绝变更 |

此外，etcd 侧**目前仍会校验哈希**：只有期望哈希与 `status.etcdConfigHash` 基线不一致时才会触发本 phase；哈希一致后不再重复执行。

修改 `kubernetesVersion`、`etcdVersion`、`kubelet.*`、`networking.*`、各类 Repo 等字段**不会**触发本 phase。

### 3.2 参数有效性必须人工核对

本特性**仅负责"发现配置变化 → 重新下发 manifest"**，**不校验参数语义**。以下各项均需运维人员人工确认：

1. **参数名/取值错误不会被拦截**：`--heartbeat-interval` 拼写错误、取值越界、单位使用错误，Webhook 与哈希计算均不会报错；但 etcd 进程会因非法参数启动失败，`waitComponentReady` 超时后命令失败，**期间 etcd 不可用，整个集群不可写**。
2. **覆盖内置参数风险极高**：`extraArgs` 会覆盖模板内置的同名参数（如 `--data-dir`、`--listen-client-urls`、`--name`、`--initial-cluster` 等）。误覆盖会直接导致 etcd 无法加入集群或数据目录错乱。
3. **etcd 参数影响面大**：心跳/选举超时、配额、压缩策略等参数需结合集群规模、磁盘性能、网络时延综合评估，错误的取值可能造成 Leader 频繁切换甚至数据不可用。
4. **无任何预检查/dry-run**：当前没有参数白名单、没有语法校验、没有变更演练能力。

**建议规范**：

- 先在测试集群验证参数，确认 etcd 能正常 Ready 且集群可读写，再应用于生产环境；
- 一次仅调整**一类参数**，避免多参数叠加导致无法定位；
- apply 前逐条比对参数名与取值（对照当前 etcd 版本的官方文档）；
- 选择**业务低峰**执行，并预留回退时间窗；
- apply 后按 [4.4 验证方法](#44-验证方法) 逐项验证。

---

## 4. 使用教程

> 下文以管理集群为例。`BKECluster` CR 的位置：
> - **管理集群**：在管理集群中修改；
> - **业务集群**：在管理集群中修改。
>
> 请将 `<cluster>`、`<ns>` 替换为实际的集群名与命名空间。

### 4.1 记录当前配置（用于回退）

```bash
kubectl get bkecluster <cluster> -n <ns> -o yaml > bkecluster-backup-$(date +%Y%m%d%H%M).yaml

# 记录当前基线哈希
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'

# 举例如下：
#root@test:~/bkeadmTest# kubectl get bc -A
#NAMESPACE     NAME                  PHASE          STATE     CLUSTER STATUS   ENDPOINT          ENDPOINT PORT   VERSION        AGENT STATUS   AGE
#bke-cluster   bke-cluster-manager   ClusterReady   Healthy   Ready            192.168.122.190   6443            v1.34.3-of.1   1/1            5m25s
#root@test:~/bkeadmTest# kubectl get bkecluster bke-cluster-manager -n bke-cluster -o jsonpath='{.status.etcdConfigHash}{"\n"}'
#a31b319c40d9c92292aa0e6e4f4addf2b50458333e853be5bca071ada2004440
```

### 4.2 修改 etcd 参数

修改 bc 的具体方式不影响本特性的触发逻辑：`kubectl edit`、`kubectl patch` 与 `kubectl apply -f <新配置 YAML 文件>` **三者效果等价**，本质都是变更 `BKECluster` 的 `spec`。只要 `spec.clusterConfig.cluster.etcd.extraArgs` 发生变化，且满足 [2. 前提条件](#2-前提条件) 中的条件，控制器即会重新协调并触发本 phase。

**方式一：直接 edit（推荐，便于人工核对）**

```bash
kubectl edit bkecluster <cluster> -n <ns>
```

定位 `spec.clusterConfig.cluster.etcd.extraArgs`，修改目标参数：

```yaml
spec:
  clusterConfig:
    cluster:
      etcd:
        extraArgs:
          heartbeat-interval: "500"      # 示例：心跳间隔（毫秒）
          election-timeout: "2500"       # 示例：选举超时（毫秒）
          quota-backend-bytes: "2147483648"   # 示例：后端存储配额（字节）
```

**方式二：patch（适合自动化）**

```bash
kubectl patch bkecluster <cluster> -n <ns> --type=merge -p '
{
  "spec": {
    "clusterConfig": {
      "cluster": {
        "etcd": {
          "extraArgs": {
            "heartbeat-interval": "500",
            "election-timeout": "2500"
          }
        }
      }
    }
  }
}'
```

**方式三：apply 新 YAML 文件（适合配置版本化管理）**

```bash
kubectl apply -f new-bc.yaml
```

> 说明：`new-bc.yaml` 为已修改 `etcd.extraArgs` 的 `BKECluster` 清单，可由 [4.1 记录当前配置（用于回退）](#41-记录当前配置用于回退) 中的备份文件修改得到。

> **注意**
> - `extraArgs` 是**全量覆盖**语义，patch merge 仅合并 map 中的同名 key，删除某个参数需将该 key 显式置为 `null`（edit 删除该行或使用 JSON merge patch 传入 `null`）。
> - 不要在同一个 apply 中同时修改 `etcdVersion` 或 `kubernetesVersion`，否则本 phase 将不执行，改由版本升级 phase 处理。
> - 不要改动 `dataDir` / `serverCertSANs` / `peerCertSANs`，会被 Webhook 直接拒绝。

### 4.3 观察执行进度

```bash
# 1) 集群 phase（返回结果应为 EnsureEtcdConfigUpdate）
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}{"\n"}'

# 2) 下发的命令
kubectl get command -l bke.bocloud.com/cluster-command

# 3) 控制器日志（reason 关键字：EnsureEtcdConfigUpdate）
cat /var/log/openFuyao/bke-controller-manager/<bke-controller-manager-pod-name> | grep -i "EnsureEtcdConfigUpdate"
# 4) 节点 bkeagent 日志，有以下返回值即可
# 关键字：update etcd static pod config / re-generate etcd manifest / etcd config update success
cat /var/log/openFuyao/bkeagent.log | grep -i "update etcd static pod config"

```

### 4.4 验证方法

```bash
# 1) 节点上 etcd manifest 已更新
# 以 root 用户登录目标 etcd 节点，查看 manifest 目录下各组件 YAML 文件的更新时间
cd /etc/kubernetes/manifests/
ls -l
# 输出示例（etcd.yaml 的更新时间应晚于变更前的记录）
# total 24
# drwxr-xr-x 2 root root 4096 Sep 14 18:27 ./
# drwxr-xr-x 4 root root 4096 Sep 14 18:27 ../
# -rw-r--r-- 1 root root 1941 Sep 14 18:55 etcd.yaml
# -rw-r--r-- 1 root root 3426 Sep 14 18:27 kube-apiserver.yaml
# -rw-r--r-- 1 root root 2519 Sep 14 18:27 kube-controller-manager.yaml
# -rw-r--r-- 1 root root 1590 Sep 14 18:27 kube-scheduler.yaml

# 2) etcd 静态 Pod 实际启动参数已生效
kubectl -n kube-system get pod etcd-<node> -o yaml | grep -A 40 'command:'

# 3) etcd Pod 处于 Running
kubectl -n kube-system get pod -l component=etcd -o wide

# 4) 集群健康
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}{"\n"}'

# 5) 基线哈希已刷新
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'
```

> **关于第 5 项**：此处返回的哈希值应与 [4.1 记录当前配置（用于回退）](#41-记录当前配置用于回退) 中记录的哈希结果**不同**；若与该处记录一致，说明本 phase 未触发或配置未实际下发。

**通过标准**：基线哈希已变化且稳定不再反复触发；所有 etcd Pod Running；集群 `clusterStatus` 为 Ready。

---

## 5. 失败排查与处理

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `kubectl apply` 直接报错 `spec.clusterConfig.cluster.etcd.dataDir: Forbidden` | 修改了引导期不可变字段 | 撤销该字段的修改；这类变更需要重建控制面，无法通过配置更新完成 |
| `kubectl apply` 报错 `APIServer/ControllerManager/Scheduler must all be present` | 管控面三组件中有组件配置被删除/置空 | 补全三组件配置后重新 apply |
| apply 成功但长时间无命令、哈希不变 | ① `status.etcdConfigHash` 为空（基线未建立）；② 改动字段不参与哈希；③ 版本字段同时被改动 | 依次排查：等待一轮健康检查建立基线；确认只改 `extraArgs`；确认未改版本 |
| 命令下发后失败，日志出现 `EtcdConfigUpdateFailed` | ① 参数非法导致 etcd 启动失败；② 等待 Pod Hash 变化超时 | 查看节点 bkeagent 日志与 `crictl logs`；按 [6. 回退方法](#6-回退方法) 回退 |
| etcd Pod CrashLoopBackOff | `extraArgs` 参数名/取值错误，或覆盖了内置关键参数 | 回退配置；核对参数后重试 |
| 部分节点成功、部分失败 | 个别节点状态异常（磁盘空间不足、端口占用等） | 修复节点后等待自动重试（旧基线保留，会持续重试） |
| 修改了 `extraVolumes`，流程显示成功但卷实际未挂载 | 该字段参与哈希（会触发更新流程）但**不渲染到 manifest**；流程照常回写新哈希后即对齐，不再重试 | 该字段暂不支持此方式变更，请勿依赖；撤销改动，仅保留 `extraArgs` 变更 |

排查命令：

```bash
# 命令对象与失败节点
kubectl get command -l bke.bocloud.com/cluster-command -o yaml

# 控制器侧失败原因
kubectl -n <capbke-ns> logs deploy/capbke-controller-manager | grep -i EtcdConfigUpdateFailed

# 节点 bkeagent 日志
ssh <etcd-node> 'journalctl -u bkeagent -f'

# 容器运行时内查看 etcd 容器日志
ssh <etcd-node> 'crictl ps -a | grep etcd; crictl logs <container-id>'
```

### 5.1 EnsureEtcdConfigUpdate 协调失败后的恢复手段

> <span style="color:red;font-weight:bold;">重要：etcd 配置更新失败后，目前暂无恢复手段。既无自动回滚，也无法手动重新协调，只能以预防为主。</span>

<span style="color:red;font-weight:bold;">etcd 配置更新失败后，目前暂无恢复手段。</span>

与管控面不同，`EnsureEtcdConfigUpdate` 当前<span style="color:red;font-weight:bold;">无法通过手动触发的方式重新协调</span>（例如为 bc 添加 retry 注解的方式对本场景不适用）；一旦因参数非法等原因导致 etcd 启动失败，集群不可写、管控面随之失效，<span style="color:red;font-weight:bold;">无法在本集群内自愈</span>。因此请务必以预防为主：

- 严格按 [3.2 参数有效性必须人工核对](#32-参数有效性必须人工核对) 人工核对参数名与取值，先在测试集群验证，再应用于生产环境；
- 重要变更请在操作前自行备份 etcd 数据；
- 预留足够的回退时间窗，并选择业务低峰执行。

> <span style="color:red;font-weight:bold;">注意</span>
> - <span style="color:red;font-weight:bold;">管理集群自管理场景同样无可用恢复手段</span>：若故障发生在<span style="color:red;font-weight:bold;">管理集群切换监听后</span>的场景，本集群内部无法完成恢复，<span style="color:red;font-weight:bold;">必须依赖上层集群处于健康状态</span>才能介入处理。

---

## 6. 回退方法

本特性**不提供自动回滚**。回退方式为：**将 `extraArgs` 恢复为原值并重新 apply**，触发一次新的配置更新。

### 6.1 回退示例（以哈希值变化说明）

> 以下哈希值为便于阅读进行了缩略，实际值为更长的十六进制摘要；判断回退是否成功的关键在于哈希值是否与原基线一致。

假设某集群 etcd 基线哈希 `status.etcdConfigHash` 的**初始值**为 `2b5f9c1a`（对应首次成功下发的 `extraArgs`：`--quota-backend-bytes=2147483648`）。

| 阶段 | etcd.extraArgs 取值 | 期望哈希 | 说明 |
| --- | --- | --- | --- |
| 初始基线 | `--quota-backend-bytes=2147483648` | `2b5f9c1a` | 首次下发后写入 `status.etcdConfigHash` |
| 变更后 | `--quota-backend-bytes=8589934592` | `c7e3a0d4` | 期望哈希 ≠ 基线，触发本 phase 的配置更新；全部 etcd 节点更新后 `status.etcdConfigHash` 被回写为 `c7e3a0d4` |
| 回退后 | `--quota-backend-bytes=2147483648`（恢复原值） | `2b5f9c1a` | 期望哈希 `2b5f9c1a` ≠ 当前基线 `c7e3a0d4`，再次触发更新；更新完成后 `status.etcdConfigHash` 回写为 `2b5f9c1a`，与初始基线一致 |

回退完成的判定标准：**将 `extraArgs` 恢复为原值并重新 apply，待全部 etcd 节点更新结束后，`status.etcdConfigHash` 应重新等于初始基线 `2b5f9c1a`**。即回退后的哈希值应与原基线哈希一致，表明 etcd 静态 Pod 配置已恢复至变更前的状态。

```bash
# 1) 用备份恢复
kubectl apply -f bkecluster-backup-<时间戳>.yaml

# 或：使用 edit 手动恢复原参数（例如将 etcd.extraArgs 恢复为 --quota-backend-bytes=2147483648）
kubectl edit bkecluster <cluster> -n <ns>

# 2) 观察回退是否完成：哈希应回到初始基线 2b5f9c1a
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.etcdConfigHash}{"\n"}'
kubectl -n kube-system get pod -l component=etcd -o wide
```

回退同样会重建 etcd manifest 并等待 Ready，验证方法同 [4.4 验证方法](#44-验证方法)。

---

## 7. 注意事项汇总

1. <span style="color:red;font-weight:bold;">只对 `extraArgs` 生效</span>：`extraVolumes` 等字段虽参与哈希（改动会触发更新流程），但不会渲染进 manifest，最终呈现为<span style="color:red;font-weight:bold;">"配置已修改、流程显示成功、但实际未生效"</span>的结果；此类热更新请勿依赖（见 [3. 使用限制](#3-使用限制)）。
2. **参数有效性需人工核对**：程序不校验参数语义，错误的参数会直接导致 etcd 无法启动。
3. **不做 etcd 备份**：配置更新路径不触发备份；重要变更请自行先备份 etcd 数据。
4. **不保证不中断**：etcd 静态 Pod 重建期间集群不可写，请在业务低峰执行。
5. **不可变字段禁止变更**：`dataDir` / `serverCertSANs` / `peerCertSANs` 变更会被 Webhook 拒绝，需重建控制面。
6. **不要同时变更版本**：`etcdVersion` / `kubernetesVersion` 变化时，本 phase 将不执行，改由版本升级 phase 处理。
7. **外置 etcd 不生效**：没有 etcd 角色 BKENode 时不会触发。
8. **无自动回滚**：回退方式为重新 apply 原有配置。
9. **只影响 etcd 节点**：管控面节点不受本 phase 影响。
10. <span style="color:red;font-weight:bold;">失败暂无恢复手段</span>：etcd 配置更新失败后既无自动回滚，也无可用的手动重试手段；管理集群自管理自身 BKECluster（即 bc）时同样无法在本集群内自愈，<span style="color:red;font-weight:bold;">必须依赖上层集群健康，请务必以预防为主</span>（见 [5.1 EnsureEtcdConfigUpdate 协调失败后的恢复手段](#51-ensureetcdconfigupdate-协调失败后的恢复手段)）。
