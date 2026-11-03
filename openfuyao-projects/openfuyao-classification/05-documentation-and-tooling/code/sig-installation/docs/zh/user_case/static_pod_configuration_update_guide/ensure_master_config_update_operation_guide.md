# 更新管控面静态 Pod 配置

> 适用对象：**更新管控面静态 Pod（`kube-apiserver` / `kube-controller-manager` / `kube-scheduler`）配置**（`extraArgs`）的场景。
> 若需要更新 etcd 参数，请参阅 [更新 etcd 静态 Pod 配置](./ensure_etcd_config_update_operation_guide.md)。
> 设计说明见 [声明式集群静态 Pod 配置更新方案](../../../../design/支持静态pod配置热更新/声明式集群静态pod配置更新方案.md)。

---

## 1. 功能概述

管控面三组件（`kube-apiserver`、`kube-controller-manager`、`kube-scheduler`）是 Kubernetes 集群最核心的组件。在集群运行过程中，运维人员常常需要调整它们的运行参数，例如：增大 apiserver 的审计日志保留天数、调整 scheduler 的调度行为、收紧 controller-manager 的鉴权参数等。

本特性面向**集群运维人员 / SRE**，适用于以下场景：**在集群版本不变、无需执行完整升级流程的前提下，仅修改管控面三组件的运行参数（`extraArgs`），使变更安全、最小化地生效。**

它提供的能力：

- **声明式**：直接修改 `BKECluster`（`BKECluster` / bc）中管控面组件的 `extraArgs`（`kubectl edit` / `kubectl patch` / `kubectl apply` 任一方式均可），由控制器自动下发，无需登录节点手工修改文件。
- **最小化影响**：仅重新生成发生变更的组件 manifest，不影响未变更组件、不影响 etcd、不影响业务节点。
- **可控**：逐 Master 节点串行更新，单节点成功就绪后再更新下一个，降低风险。

### 1.1 相关概念

- **静态 Pod 是什么**：静态 Pod 的 manifest 是节点上 `/etc/kubernetes/manifests/` 目录下的普通 YAML 文件（如 `/etc/kubernetes/manifests/kube-apiserver.yaml`）。该文件由**节点上的 kubelet 直接读取并管理**，**不经过 API Server**：它并非像普通 Pod 那样通过 `kubectl` 提交至集群、再由调度器调度，而是由 kubelet 持续监听该目录，一旦文件内容发生变化即自动重建对应的 Pod（先删除旧 Pod，再创建新 Pod）。
- **为什么和普通 Pod 不同**：普通 Pod 的创建、更新与删除均由 API Server 及相应的控制器管理，修改配置通常通过 `kubectl edit` 或更新 Deployment 完成。而静态 Pod 的"配置"即磁盘上的 manifest 文件，修改参数的本质是**改写该文件并由 kubelet 重新拉起组件**；静态 Pod 不保存在 etcd 中，也不受任何 Workload 控制器管理，因而不具备"更新 Deployment 即触发滚动更新"的能力，变更必须直接作用于节点文件系统。
- **管控面三组件均为静态 Pod**：`kube-apiserver`、`kube-controller-manager`、`kube-scheduler`全部以静态 Pod 方式运行（etcd 由 [更新 etcd 静态 Pod 配置](./ensure_etcd_config_update_operation_guide.md) 单独处理）。这正是**为何需要独立特性来变更这些组件的参数**：该特性的本质是"在每个管控面节点上改写 `/etc/kubernetes/manifests/` 下对应组件的 manifest，并等待 kubelet 重新拉起组件"，而非常规的 Pod 变更流程。

### 1.2 特点

- **组件级最小化**：改 apiserver 参数不会连带重刷 controller-manager / scheduler。
- **不涉及 etcd**：etcd 的配置更新由独立流程负责，本 phase 不会触碰 etcd manifest，也**不做 etcd 备份**。
- **逐节点串行**：前一个 Master 节点成功并 Ready 后，才下发下一个节点。
- **失败可重试**：只有全部 Master 节点成功后才回写 `status.controlPlaneConfigHash`；失败保留旧基线，下轮自动重试。

### 1.3 执行流程（内部实现，供排查参考）

1. 修改 bc（`kubectl apply` / `kubectl edit` / `kubectl patch` 任一方式）→ Webhook `validateStaticPodConfigUpdate` 校验：管控面三组件必须同时存在、etcd 不可变字段未变更（校验不通过则本次写操作直接失败）。
2. capbke Controller `Reconcile` → `PhaseFlow` 依次调用各 phase 的 `NeedExecute`（`EnsureEtcdConfigUpdate` 排在 `EnsureMasterConfigUpdate` 之前）。
3. `EnsureMasterConfigUpdate.NeedExecute` 判定：控制面已初始化、spec与status中 K8s/etcd 版本已对齐、存在 Master 节点、期望哈希非空、基线非空且**与期望哈希不同** → 触发。
4. `Execute`：触发 `reconcileMasterConfigUpdate()`。
5. 逐 Master 节点串行下发 `UpgradeControlPlaneConfig` 命令，携带 `controlPlaneConfigOldHash` / `controlPlaneConfigNewHash`（**这两个参数仅本 phase 使用，其他 phase 不受影响**）并等待完成。
6. 节点 bkeagent：`updateControlPlaneConfig(old, new)` → 记录三组件当前 Pod Hash → 逐组件比对得到变化集合 → 按 `kube-apiserver → kube-controller-manager → kube-scheduler` 顺序：未变化则跳过（日志 `config unchanged, skip`），变化则重新渲染 `/etc/kubernetes/manifests/<component>.yaml` → `waitComponentReady`（等 Pod Hash 变化 + Pod Running）。
7. **全部** Master 节点成功后回写 `status.controlPlaneConfigHash = newHash`；任一节点失败则保留旧基线，下轮 reconcile 自动重试。

---

## 2. 前提条件

| 编号 | 条件 | 检查方法 |
| --- | --- | --- |
| 1 | 集群已部署完成且健康（控制面已初始化） | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}'` |
| 2 | **基线哈希已建立**：`status.controlPlaneConfigHash` 非空 | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'` |
| 3 | 集群存在 Master 角色节点 | `kubectl get bkenode` |
| 4 | **管控面三组件配置同时存在**（`apiServer` / `controllerManager` / `scheduler`），且本次只改 `extraArgs` | 见 [4.2 修改管控面组件参数](#42-修改管控面组件参数) |
| 5 | `kubernetesVersion` 与 `etcdVersion` **保持不变** | 见 [4.2 修改管控面组件参数](#42-修改管控面组件参数) |
| 6 | 集群当前没有正在进行的其它 phase（升级、扩缩容等） | `kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}'` |

> **关于条件 2**：新建集群在完成一次健康检查（`EnsureCluster`）后会自动建立基线；**存量集群**升级到含本特性的版本后，也需先经过一轮健康检查才会写入基线。若 `status.controlPlaneConfigHash` 为空，修改参数**不会**触发更新（防止部署阶段误刷静态 Pod）。

> **关于「基线哈希」（`status.controlPlaneConfigHash`）**：它记录的是**上一次成功下发到节点静态 Pod 的那份配置**的哈希值，即"已生效配置基线"。哈希由**所有参与哈希计算的字段**计算得出，其中包括各组件的 `extraArgs`，以及虽然参与哈希、但当前并未渲染进 manifest 的 `extraVolumes` / `certSANs` / `apiEndpoint`。其中**只有 `extraArgs` 真正渲染进 manifest**；其余字段参与哈希（改动会改变哈希值，从而触发本 phase），但不会写入 manifest。每次 `apply` 后，控制器会用当前 `spec` 算出**期望哈希**；只有当**期望哈希 ≠ 基线哈希**时，本 phase 才会触发重新下发。新建/存量集群完成首次健康检查（`EnsureCluster`）后才会写入该基线——目的是避免在部署或早期阶段因哈希缺失而误刷静态 Pod。更新全部节点成功后，基线会被回写为最新期望哈希；一旦对齐，后续将不再重复触发。

---

## 3. 使用限制

### 3.1 只有 `extraArgs` 真正生效

三个管控面组件的 manifest 模板当前**只渲染 `extraArgs`**：

```yaml
# utils/bkeagent/mfutil/tmpl/k8s/kube-apiserver.yaml.tmpl（节选）
- command:
  - kube-apiserver
  ...
  {{- range . | extraArgs }}
  - --{{ . }}
  {{- end }}
```

`extraArgs` 中每一项按 `--<key>=<value>` 追加到组件启动命令末尾；与模板内置参数冲突时 **用户参数 last-wins（覆盖内置值）**。

| 组件 | 字段 | 参与哈希 | 是否渲染到 manifest |
| --- | --- | :---: | :---: |
| `apiServer` | `extraArgs` | ✅ | ✅ **真正生效** |
| `apiServer` | `extraVolumes` | ✅ | ❌ 社区暂未落地 |
| `apiServer` | `certSANs` | ✅ | ❌ 社区暂未落地 |
| `apiServer` | `apiEndpoint`（host/port） | ✅ | ❌ 社区暂未落地 |
| `controllerManager` | `extraArgs` | ✅ | ✅ **真正生效** |
| `controllerManager` | `extraVolumes` | ✅ | ❌ 社区暂未落地 |
| `scheduler` | `extraArgs` | ✅ | ✅ **真正生效** |
| `scheduler` | `extraVolumes` | ✅ | ❌ 社区暂未落地 |
| `etcd` | 全部字段 | ❌ | 由 etcd phase 独立处理 |

**结论：运维时只调整 `extraArgs`。**

> <span style="color:red;font-weight:bold;">注意：对于未渲染到 manifest 的字段（如 `extraVolumes` / `certSANs` / `apiEndpoint`），更新流程仍会完整执行 `waitComponentReady`，并将新的哈希值回写至 Status，流程显示为"成功"；但目标配置实际并未生效（例如卷并未挂载）。此后哈希已对齐，本 phase 不再触发重试，最终呈现为"配置已修改、流程显示成功、但实际未生效"的结果。</span>

修改 `kubernetesVersion`、`etcdVersion`、`kubelet.*`、`networking.*`、各类 Repo 等字段**不会**触发本 phase。

### 3.2 参数有效性必须人工核对

本特性**仅负责"发现配置变化 → 重新下发 manifest"**，**不校验参数语义**。以下各项均需运维人员人工确认：

1. **参数名/取值错误不会被拦截**：`--max-requests-inflight` 拼写错误、取值越界、布尔值使用非标准取值（如中文字面量）等，Webhook 与哈希计算均不会报错；组件会因非法参数启动失败（CrashLoopBackOff），`waitComponentReady` 超时后命令失败，**期间管控面不可用，集群无法被管理**。
2. **覆盖内置参数风险极高**：`extraArgs` 会覆盖模板内置的同名参数（如 apiserver 的 `--secure-port`、`--advertise-address`、`--etcd-servers`、`--service-cluster-ip-range` 等）。误覆盖会直接导致 apiserver 不可用。
3. **apiserver 参数影响面最大**：限流、审计、FeatureGates、证书相关参数一旦配置错误，会导致整个集群 API 不可用。
4. **无任何预检查/dry-run**：当前没有参数白名单、没有语法校验、没有变更演练能力。

**建议规范**：

- 先在测试集群验证参数，确认组件能正常 Ready、集群功能正常，再应用于生产环境；
- **一次仅修改一个组件的一类参数**，便于定位；
- apply 前逐条比对参数名与取值（对照当前 Kubernetes 版本的官方参数文档）；
- 选择**业务低峰**执行，并预留回退时间窗；
- 多 Master 场景下注意：本 phase **逐节点串行**，更新期间管控面容量下降，需确认剩余节点可承载流量；
- apply 后按 [4.4 验证方法](#44-验证方法) 逐项验证。

---

## 4. 使用教程

> 下文以管理集群为例。`BKECluster` CR 的位置：
> - **管理集群**：在管理集群自身中修改（完成监听切换后，管理集群自行管理自己的 `BKECluster` / bc）；
> - **业务集群**：在管理集群中修改。
>
> 请将 `<cluster>`、`<ns>` 替换为实际的集群名与命名空间。

### 4.1 记录当前配置（用于回退）

```bash
kubectl get bkecluster <cluster> -n <ns> -o yaml > bkecluster-backup-$(date +%Y%m%d%H%M).yaml

# 记录当前基线哈希
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
# 举例：
#root@test:~/bkeadmTest# kubectl get bc -A
#NAMESPACE     NAME                  PHASE          STATE     CLUSTER STATUS   ENDPOINT          ENDPOINT PORT   VERSION        AGENT STATUS   AGE
#bke-cluster   bke-cluster-manager   ClusterReady   Healthy   Ready            192.168.122.190   6443            v1.34.3-of.1   1/1            5m25s
#root@test:~# kubectl get bkecluster bke-cluster-manager -n bke-cluster -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
#{"apiServer":"a9960a43cd92903e44af103123bfc11912d95beaae553175796e160788c9964d","controllerManager":"23e5bb4df921abc205b24146ab796f09c329dfafdf07eb84254bcd63cb6598d6","scheduler":"c4587d74be70eb5b6f634df4d421fabf0156cc79043568db59f0a340b326dfd5"}
```

### 4.2 修改管控面组件参数

修改 bc 的具体方式不影响本特性的触发逻辑：`kubectl edit`、`kubectl patch` 与 `kubectl apply -f <新配置 YAML 文件>` **三者效果等价**，本质都是变更 `BKECluster` 的 `spec`。只要 `spec.clusterConfig.cluster` 下对应组件的 `extraArgs` 发生变化，且满足 [2. 前提条件](#2-前提条件) 中的条件，控制器即会重新协调并触发本 phase。

**方式一：直接 edit（推荐，便于人工核对）**

```bash
kubectl edit bkecluster <cluster> -n <ns>
```

定位 `spec.clusterConfig.cluster` 下的 `apiServer` / `controllerManager` / `scheduler`：

```yaml
spec:
  clusterConfig:
    cluster:
      apiServer:
        extraArgs:
          max-requests-inflight: "800"        # 示例：限流
          max-mutating-requests-inflight: "400"
          feature-gates: "RemoveSelfLink=false"
      controllerManager:
        extraArgs:
          concurrent-deployment-syncs: "10"   # 示例：并发同步数
          kube-api-qps: "50"
      scheduler:
        extraArgs:
          kube-api-qps: "50"                  # 示例：QPS 限制
```

> **重要**：`apiServer` / `controllerManager` / `scheduler` **三个配置段必须同时存在**。任一段缺失或置空，Webhook 会直接拒绝 apply（`APIServer/ControllerManager/Scheduler must all be present`）。若某组件暂时不需要额外参数，保留空段即可，例如 `scheduler: {}` 或 `scheduler: {extraArgs: {}}`。

**方式二：patch（适合自动化）**

```bash
kubectl patch bkecluster <cluster> -n <ns> --type=merge -p '
{
  "spec": {
    "clusterConfig": {
      "cluster": {
        "apiServer": {
          "extraArgs": {
            "max-requests-inflight": "800",
            "max-mutating-requests-inflight": "400"
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

> 说明：`new-bc.yaml` 为已修改 `extraArgs` 的 `BKECluster` 清单，可由 [4.1 记录当前配置（用于回退）](#41-记录当前配置用于回退) 中的备份文件修改得到。

> **注意**
> - `extraArgs` 是**全量覆盖**语义，merge patch 只合并同名 key；删除某个参数需显式删除该 key（edit 删行或 JSON merge patch 传 `null`）。
> - 不要在同一个 apply 中同时修改 `kubernetesVersion` 或 `etcdVersion`，否则本 phase 将不执行，改由版本升级 phase 处理。
> - 不要删除/置空任何一个管控面组件配置段。

### 4.3 观察执行进度

```bash
# 1) 集群 phase
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.phase}{"\n"}'

# 2) 下发的命令（phase 应为 UpgradeControlPlaneConfig）
kubectl get command -l bke.bocloud.com/cluster-command

# 3) 控制器日志（reason 关键字：EnsureMasterConfigUpdate）
cat /var/log/openFuyao/bke-controller-manager/<bke-controller-manager-pod-name> | grep -i "EnsureMasterConfigUpdate"

# 4) 节点 bkeagent 日志
# 关键字：update control plane static pod config / component <name> config changed, regenerate manifest
#         component <name> config unchanged, skip / component <name> config update success
cat /var/log/openFuyao/bkeagent.log | grep -i "update control plane static pod config"
```

### 4.4 验证方法

```bash
# 1) 节点上 manifest 已更新
# 以 root 用户登录目标 Master 节点，查看 manifest 目录下各组件 YAML 文件的更新时间
cd /etc/kubernetes/manifests/
ls -l
# 输出示例（发生变更的组件 YAML 文件更新时间应晚于其余文件）
# total 24
# drwxr-xr-x 2 root root 4096 Sep 14 18:27 ./
# drwxr-xr-x 4 root root 4096 Sep 14 18:27 ../
# -rw-r--r-- 1 root root 1941 Sep 14 18:27 etcd.yaml
# -rw-r--r-- 1 root root 3426 Sep 14 18:55 kube-apiserver.yaml
# -rw-r--r-- 1 root root 2519 Sep 14 18:27 kube-controller-manager.yaml
# -rw-r--r-- 1 root root 1590 Sep 14 18:27 kube-scheduler.yaml

# 2) 静态 Pod 实际启动参数已生效
kubectl -n kube-system get pod kube-apiserver-<node> -o yaml | grep -A 60 'command:'
kubectl -n kube-system get pod kube-controller-manager-<node> -o yaml | grep -A 40 'command:'
kubectl -n kube-system get pod kube-scheduler-<node> -o yaml | grep -A 30 'command:'

# 3) 三个管控面组件全部 Running（所有 Master 节点）
kubectl -n kube-system get pod -l tier=control-plane -o wide

# 4) 未改动的组件未被重启（AGE 应远大于刚更新的组件）
kubectl -n kube-system get pod -l tier=control-plane -o wide

# 5) 集群健康
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.clusterStatus}{"\n"}'

# 6) 基线哈希已刷新
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
```

> **关于第 6 项**：此处返回的哈希值应与 [4.1 记录当前配置（用于回退）](#41-记录当前配置用于回退) 中记录的哈希结果**不同**；若与该处记录一致，说明本 phase 未触发或配置未实际下发。

**通过标准**：基线哈希已变化且稳定不再反复触发；三个管控面组件在所有 Master 节点均为 Running；未修改的组件未被重启；集群 `clusterStatus` 为 Ready；`kubectl get nodes` 正常。

---

## 5. 失败排查与处理

| 现象 | 可能原因 | 处理 |
| --- | --- | --- |
| `kubectl apply` 报错 `APIServer/ControllerManager/Scheduler must all be present` | 三组件中某段被删除或置空 | 补全该组件配置段（可保留空段）后重新 apply |
| `kubectl apply` 报错 etcd 字段 `Forbidden` | 同时修改了 etcd 的 `dataDir` / `serverCertSANs` / `peerCertSANs` | 撤销 etcd 不可变字段的修改 |
| apply 成功但长时间无命令、哈希不变 | ① `status.controlPlaneConfigHash` 为空（基线未建立）；② 改动字段不参与哈希；③ 同时改了版本字段 | 等待一轮健康检查建立基线；确认只改 `extraArgs`；确认未改版本 |
| 命令失败，日志出现 `MasterConfigUpdateFailed` | ① 参数非法导致组件启动失败；② 等待 Pod Hash 变化超时 | 立即查看节点 bkeagent 日志与容器日志；按 [6. 回退方法](#6-回退方法) 回退 |
| `kube-apiserver` CrashLoopBackOff | `extraArgs` 参数名/取值错误，或覆盖了内置关键参数 | 回退配置；核对参数后重试 |
| 只有部分 Master 节点更新完成 | 某节点异常（磁盘、端口、kubelet 状态等） | 修复节点后等待自动重试（旧基线保留，会持续重试） |
| 修改了 `extraVolumes` / `certSANs` / `apiEndpoint`，流程显示成功但配置实际未生效（卷未挂载） | 这些字段参与哈希（会触发更新流程）但**不渲染到 manifest**；流程照常回写新哈希后即对齐，不再重试 | 此类字段当前不支持热更新，请勿依赖；撤销改动，仅保留 `extraArgs` 变更；确需挂载该类配置请重建控制面 |

排查命令：

```bash
# 命令对象与失败节点
kubectl get command -l bke.bocloud.com/cluster-command -o yaml

# 控制器侧失败原因
kubectl -n <capbke-ns> logs deploy/capbke-controller-manager | grep -i MasterConfigUpdateFailed

# 节点 bkeagent 日志
ssh <master-node> 'journalctl -u bkeagent -f'

# 容器运行时内查看组件容器日志
ssh <master-node> 'crictl ps -a | grep -E "kube-apiserver|kube-controller-manager|kube-scheduler"; crictl logs <container-id>'
```

### 5.1 EnsureMasterConfigUpdate 协调失败后的恢复手段（手动重试）

> <span style="color:red;font-weight:bold;">重要前提：必须先将 bc 修正为正确配置，再手动触发重新协调；否则只会重复失败并再次耗尽重试次数。</span>

`EnsureMasterConfigUpdate` 协调失败后（例如参数非法导致组件 CrashLoopBackOff、组件长时间未 Ready 导致命令超时），可按以下顺序恢复：

1. <span style="color:red;font-weight:bold;">等待 phase 重试次数耗尽</span>：等待当前轮次的重试结束，此时集群 `CLUSTER STATUS` 变为 `UpgradeFailed`（协调已停止，不再持续消耗重试次数）。
2. <span style="color:red;font-weight:bold;">将 bc 修正为正确配置</span>：依据上表定位原因后，将 `spec.clusterConfig.cluster` 下对应组件的 `extraArgs` 恢复为原值或修正为正确取值（参考 [6. 回退方法](#6-回退方法)）。该步骤不可省略。
3. **再次触发协调**：为 bc 添加 retry 注解，手动触发新一轮协调：

```bash
kubectl annotate bkecluster <cluster> -n <ns> bke.bocloud.com/retry="" --overwrite
```

执行后按 [4.3 观察执行进度](#43-观察执行进度) 观察 phase 与命令下发情况，并按 [4.4 验证方法](#44-验证方法) 验证。

> <span style="color:red;font-weight:bold;">注意</span>
> - <span style="color:red;font-weight:bold;">修正配置是前提</span>：若配置项仍为非法取值，重新协调只会再次失败并重复消耗重试次数。
> - <span style="color:red;font-weight:bold;">管理集群自管理场景例外（无可用恢复手段）</span>：管理集群完成监听切换后，由其自身的 controller 协调自己的 bc。若此时本集群管控面已不可用（无法在本集群内执行上述 kubectl 操作），则<span style="color:red;font-weight:bold;">本集群内部无可用恢复手段，必须依赖上层集群处于健康状态</span>才能介入处理。

---

## 6. 回退方法

本特性**不提供自动回滚**。回退方式为：**将 `extraArgs` 恢复为原值并重新 apply**，触发一次新的配置更新。

### 6.1 回退示例（以哈希值变化说明）

> 以下哈希值为便于阅读进行了缩略，实际值为更长的十六进制摘要；判断回退是否成功的关键在于哈希值是否与原基线一致。

假设某管理集群控制面基线哈希 `status.controlPlaneConfigHash` 的**初始值**为 `9f3a2b7c`（对应首次成功下发的 `extraArgs`：apiserver `--audit-log-maxage=7`，controller-manager 与 scheduler 为默认值）。

| 阶段 | apiserver.extraArgs 取值 | 期望哈希 | 说明 |
| --- | --- | --- | --- |
| 初始基线 | `--audit-log-maxage=7` | `9f3a2b7c` | 首次下发后写入 `status.controlPlaneConfigHash` |
| 变更后 | `--audit-log-maxage=30` | `4d8e1a6c` | 期望哈希 ≠ 基线，触发本 phase 的配置更新；滚动更新后 `status.controlPlaneConfigHash` 被回写为 `4d8e1a6c` |
| 回退后 | `--audit-log-maxage=7`（恢复原值） | `9f3a2b7c` | 期望哈希 `9f3a2b7c` ≠ 当前基线 `4d8e1a6c`，再次触发滚动更新；更新完成后 `status.controlPlaneConfigHash` 回写为 `9f3a2b7c`，与初始基线一致 |

回退完成的判定标准：**将 `extraArgs` 恢复为原值并重新 apply，待全部 Master 节点滚动更新结束后，`status.controlPlaneConfigHash` 应重新等于初始基线 `9f3a2b7c`**。即回退后的哈希值应与原基线哈希一致，表明静态 Pod 配置已恢复至变更前的状态。

```bash
# 1) 用备份恢复
kubectl apply -f bkecluster-backup-<时间戳>.yaml

# 或：使用 edit 手动恢复原参数（例如将 apiserver.extraArgs 恢复为 --audit-log-maxage=7）
kubectl edit bkecluster <cluster> -n <ns>

# 2) 观察回退是否完成：哈希应回到初始基线 9f3a2b7c
kubectl get bkecluster <cluster> -n <ns> -o jsonpath='{.status.controlPlaneConfigHash}{"\n"}'
kubectl -n kube-system get pod -l tier=control-plane -o wide
```

回退同样会重建对应组件的 manifest 并等待 Ready，验证方法同 [4.4 验证方法](#44-验证方法)。

---

## 7. 注意事项汇总

1. <span style="color:red;font-weight:bold;">只对 `extraArgs` 生效</span>：`extraVolumes` / `certSANs` / `apiEndpoint` 等虽参与哈希（改动会触发更新流程），但不会渲染进 manifest，最终呈现为<span style="color:red;font-weight:bold;">"配置已修改、流程显示成功、但实际未生效"</span>的结果；此类热更新请勿依赖（见 [3. 使用限制](#3-使用限制)）。
2. **参数有效性需人工核对**：程序不校验参数语义，错误的参数会直接导致管控面组件无法启动。
3. **管控面三组件必须同时存在**：任一配置段缺失会被 Webhook 拒绝。
4. **不做 etcd 备份、不涉及 etcd**：etcd 配置变更由独立的 etcd 配置更新流程负责。
5. **不保证不中断**：静态 Pod 重建期间对应组件短暂不可用；且本 phase **逐 Master 节点串行**，期间管控面容量下降，多 Master 场景请评估容量。
6. **不要同时变更版本**：`kubernetesVersion` / `etcdVersion` 变化时，本 phase 将不执行，改由版本升级 phase 处理。
7. **组件级最小化仅对"发生变更的组件"生效**：判断依据是新旧两个哈希结构体的逐组件比对，未发生变更的组件会被跳过。
8. **无自动回滚**：回退方式为重新 apply 原有配置。
9. **只影响 Master 节点**：Worker 节点不受本 phase 影响；kubelet 参数也不在本特性范围内。
10. **失败可手动重试**：协调失败后等待重试次数耗尽（`CLUSTER STATUS` 为 `UpgradeFailed`），将配置修正后通过 `bke.bocloud.com/retry` 注解重新触发协调；<span style="color:red;font-weight:bold;">但管理集群自管理自身 BKECluster（即 bc）且管控面已不可用时，本集群内部无可用恢复手段，必须依赖上层集群健康</span>（见 [5.1 EnsureMasterConfigUpdate 协调失败后的恢复手段（手动重试）](#51-ensuremasterconfigupdate-协调失败后的恢复手段手动重试)）。
