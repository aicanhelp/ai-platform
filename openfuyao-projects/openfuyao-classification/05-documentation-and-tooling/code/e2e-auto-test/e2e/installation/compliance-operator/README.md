# compliance-hardening E2E 用例追溯与手动验证指南

## 背景

已评审基线包含 17 条特性级集成测试用例。下文列出的 XLSX 是唯一主基准，其中 O 列决定每条用例的分组。

集成测试用例（FIT）是一个经过评审的业务场景，Ginkgo 的 `It(...)` 是对应的自动化执行项。进入 E2E 的每条 FIT/DFX 均对应一个 `It(...)`，用例内的多个步骤和预期结果通过 `By(...)` 与 `Expect(...)` 保留。FIT-16、FIT-17 按评审表 O/P 列由产品侧 Bats UT 独立覆盖，不进入本套件。

这些分组会改变集群状态，所以套件使用以下有序生命周期。外层 `SIGDescribe` 带
`installation`、`post-init` 等标签，并以 `Ordered` 保持状态顺序。仓库通用标签中
`post-init` 表示引导节点已就绪。Runner 已完整导出 `GUIDE_NODE_*`、`TEST_NODE2~5_*` 和
`WORKLOAD_CLUSTER_VIP_*` 的必需字段时，本套件只使用进程环境变量且不读取 `.env`；必需字段
不完整时才加载 `e2e/.env` 兼容本地执行。`KUBECONFIG` 为可选配置：未提供时直接使用
GUIDE_NODE 的默认 kubeconfig；提供时优先将其作为 GUIDE_NODE 路径使用，远端不可用时再从
Runner 读取、内联其中引用的证书与密钥并安全复制到 GUIDE_NODE；两处均不可用时回退 GUIDE_NODE
默认配置。最终选中的 kubeconfig 必须连接到完成 `bke init` 且 BKE 管理组件 Ready 的
management environment：

```text
E2E 外：一次性部署 bootstrap / management environment 并执行 bke init，后续多轮复用
  -> BeforeAll：只读确认现有 management API、BKECluster CRD 和 bke-controller-manager Ready
  -> 通过现有 management cluster 创建 TEST_NODE2~5 组成的 3M1W workload cluster
  -> controlPlaneEndpoint、workload kubeconfig、entry context 均校验使用 WORKLOAD_CLUSTER_VIP_HOST:PORT
  -> worker kubelet 使用 VIP IP，或使用在 worker 上解析到该 VIP 的 HA 域名；端口必须匹配且禁止解析到 master IP
  -> entry node 无工具时，从 Runner 当前 E2E checkout 的 testdata 上传最小脚本制品
  -> cluster.yaml 不存在时，根据 TEST_NODE2~5_* / COMPLIANCE_NODE2~5_* 安全生成
  -> DFX-01 配置防护
  -> E2E-B：在 S0 状态执行 dry-run
  -> E2E-A：从 S0 状态进入加固态
  -> DFX-02：重复加固
  -> E2E-C：回退到 S0 状态
  -> DFX-03：重复回退
  -> DFX-04：真实加固/回退产物敏感信息审查
  -> AfterAll：通过 management cluster 删除本轮 workload cluster 并清理生成的配置/kubeconfig
  -> bootstrap / management environment 及其 kubeconfig 保留，不执行 bke reset 或重新 init
```

## 源码与资料位置

- 被测功能源码：`/root/compliance-operator/compliance-hardening`。初始 E2E 重构阶段只读；用户后续分别明确授权修复 `lib/health_check.sh`、`lib/cluster.sh` 和 `lib/backup.sh`
- 已评审测试用例：`/root/compliance-operator/compliance-hardening/tests/fit/CIS-STIG安全加固特性级集成测试用例-v0.9.xlsx`
- E2E 源码：`/root/e2e-auto-test/e2e/installation/compliance-operator`
- E2E Go module 根目录：`/root/e2e-auto-test/e2e`

E2E 仓库在 `installation/compliance-operator/testdata/compliance-hardening` 保存最小运行制品，
Runner checkout 本仓库后即可使用。流水线也可通过
`COMPLIANCE_HARDENING_SOURCE_DIR` 显式覆盖为 Runner 上的其他 `compliance-hardening` 目录或
`compliance-operator` 仓库根目录。远程 entry node 上的工具副本路径固定为
`/root/compliance-operator/compliance-hardening`，定义在 `utils/const.go`。上传制品只包含
`bin/`、`lib/` 和 `config/hardening-rules.yaml`，不会复制 Runner 上可能含凭据的
`config/cluster.yaml`。

## XLSX 核对结果与分组

`FIT用例` 工作表从第 2 行到第 18 行正好包含 17 条用例。O 列标题为 `E2E合并组`，实际分组如下：

```text
17 条已评审 FIT
├── E2E-A：加固主链路
│   ├── FIT-01 .. FIT-09
│   └── FIT-15
├── E2E-B：dry-run 预检
│   └── FIT-10 .. FIT-11
├── E2E-C：回退闭环
│   └── FIT-12 .. FIT-14
└── UT（独立执行，不进入 E2E）
    └── FIT-16 .. FIT-17

额外 DFX
└── DFX-01 .. DFX-04
```

E2E-A 的编号不连续是 XLSX 的真实结果。XLSX 中 FIT-15 是审计策略用例，其 O 列为 `E2E-A 加固主链路`。不能将它重新编号为 FIT-10，否则会破坏与已评审用例的追溯关系。

## 最终 E2E 结构

完整套件包含 19 个 Ginkgo 测试点：15 条进入 E2E 的已评审 FIT 与 4 条额外 DFX 均保持一对一映射。FIT-16、FIT-17 是独立 Bats UT，不计入 Ginkgo 测试点。

```text
[E2E-A] Hardening Main Flow
├── FIT-01 集群编排与已评审整改值
├── FIT-02 kube-bench 与 OpenSCAP 复扫对比
├── FIT-03 bind-address 与 livenessProbe 同步
├── FIT-04 每个 master 的 etcd data-dir owner/group 保持不变
├── FIT-05 hostname-override 保留
├── FIT-06 worker 仅执行 g6
├── FIT-07 Markdown 与 JSON 报告完整性
├── FIT-08 业务负载与 Kubernetes API 可用性
├── FIT-09 加固产物关联与留痕
└── FIT-15 audit-policy 内容保持不变

[E2E-B] Dry-Run Precheck
├── FIT-10 按节点/角色预览并对比真实旧值
└── FIT-11 指纹与备份数量保持不变

[E2E-C] Rollback Loop
├── FIT-12 统一回退、S0 恢复、复扫和日志
├── FIT-13 不存在备份的拒绝与零副作用
└── FIT-14 确认块展示与取消

[DFX] 额外用例
├── DFX-01 非法集群配置防护
├── DFX-02 重复加固幂等
├── DFX-03 重复回退幂等
└── DFX-04 真实加固/回退产物敏感信息审查
```

方括号内的英文名称是代码中的真实 Ginkgo 名称，保持英文是为了与测试输出和筛选标签一致。

## FIT 映射

| 原始 FIT | XLSX O 列 | 最终分组 | 自动化测试点 | 说明 |
|---|---|---|---|---|
| FIT-01 | E2E-A 加固主链路 | E2E-A | 完成集群加固；校验 master 已评审整改值 | 混合全局/节点级 SSH 凭据、统一时间戳、master 串行、worker 并行 |
| FIT-02 | E2E-A 加固主链路 | E2E-A | 加固后 kube-bench 对比；加固后 OpenSCAP 对比 | 剩余失败项必须属于已评审人工处理集合 |
| FIT-03 | E2E-A 加固主链路 | E2E-A | 校验 controller-manager、scheduler 的 bind/probe 同步 | 同时检查控制面 Pod 健康状态 |
| FIT-04 | E2E-A 加固主链路 | E2E-A | 对比每个 master 的 etcd owner/group 前后值 | 不再统一要求 `root:root` |
| FIT-05 | E2E-A 加固主链路 | E2E-A | 保留正在使用的 hostname override 和节点身份 | 节点必须保持 Ready |
| FIT-06 | E2E-A 加固主链路 | E2E-A | 校验 worker 输出包含 `g6_kubelet`；校验 worker 的 g6 配置 | worker 不应存在控制面 manifest |
| FIT-07 | E2E-A 加固主链路 | E2E-A | 校验 Markdown/JSON 报告结构和节点状态 | 包含人工处理提醒及旧值到新值明细 |
| FIT-08 | E2E-A 加固主链路 | E2E-A | 校验业务副本、UID 行为和 ConfigMap API 操作 | 最终所有节点必须 Ready |
| FIT-09 | E2E-A 加固主链路 | E2E-A | 关联加固日志、节点日志、报告和备份 | E2E-C 同时将回退日志作为回退证据校验 |
| FIT-10 | E2E-B dry-run预检 | E2E-B | 校验按节点/角色预览；将 CIS 1.1.1 旧值与 `stat` 对比 | 必须在 S0 状态执行 |
| FIT-11 | E2E-B dry-run预检 | E2E-B | 对比 dry-run 前后的全部指纹和备份数量 | dry-run 不得创建备份或写入配置 |
| FIT-12 | E2E-C 回退闭环 | E2E-C | 执行统一回退；对比 S0 指纹和 Ready；对比扫描；校验回退日志 | 依赖 E2E-A 生成的时间戳和备份 |
| FIT-13 | E2E-C 回退闭环 | E2E-C | 在确认前拒绝不存在的时间戳；校验零副作用 | 预期退出码为 1 |
| FIT-14 | E2E-C 回退闭环 | E2E-C | 展示每个节点的目标备份并取消 | 取消不得修改节点 |
| FIT-15 | E2E-A 加固主链路 | E2E-A | 对比 audit-policy MD5 并检查报告状态 | 该规则只校验内容，不执行整改 |
| FIT-16 | —（UT独立，不进E2E） | UT | 不属于 Ginkgo E2E | 产品 Bats：SSH 预检失败后立即停止 |
| FIT-17 | —（UT独立，不进E2E） | UT | 不属于 Ginkgo E2E | 产品 Bats：记录逻辑不输出敏感信息；真实产物由额外 DFX-04 审查 |

## etcd data-dir 修复

旧 E2E 对所有 master 使用同一个配置路径，并统一断言加固后的 owner/group 为 `root:root`。当不同 master 的原始状态不一致时，该断言是错误的。例如真实基线为：

```text
master1 root:root
master2 etcd:etcd
master3 root:root
```

现在 E2E 会在每个 master 上独立执行以下流程：

1. 读取 `/etc/kubernetes/manifests/etcd.yaml`。
2. 定位名为 `etcd` 的容器，从 `command` 或 `args` 中解析 `--data-dir=/path` 或 `--data-dir /path`。
3. 如果 manifest、参数、绝对路径、目录或 owner/group 无法获取，立即失败。
4. 加固前保存 `{node -> 检测路径, owner/group}`。
5. 加固后重新检测路径和 owner/group。
6. 将每个节点与它自己的基线比较，并在失败信息中输出 node、路径、before 和 after。

FIT-04 不比较目录 permission 前后相等，因为已评审需求要求 CIS 1.1.11 将目录权限整改为 `700`。原有 g5 加固后权限断言仍会校验该要求。

节点指纹和扫描挂载也改为使用动态检测的宿主机路径。OpenSCAP 仍将该动态宿主机路径挂载到容器内的 `/var/lib/etcd`，因为 STIG 内容依赖该容器内路径。这是路径适配，不是对宿主机路径的假设。

## 额外 DFX

XLSX 不包含这 4 条 DFX。两个仓库中都没有找到独立的 DFX 评审源文件。目前可用的来源是重构前已有的 E2E 代码和 README，因此本次只保留、整理和命名已有覆盖，不凭空创建新的测试目标。

| DFX | 已有覆盖 | 本次处理 |
|---|---|---|
| DFX-01 | 文件不存在、YAML 非法、缺少凭据、零副作用 | 保留覆盖，增加独立标签和英文测试名称 |
| DFX-02 | 重复加固、合规报告状态、空 changes、指纹不变、节点 Ready | 保留覆盖，增加独立标签和英文测试名称 |
| DFX-03 | 重复回退直接判定无差异、恢复态指纹不变 | 修复产品差量判定后，严格要求一次重复回退即为 no-op |
| DFX-04 | entry node 编排/节点结果日志、Markdown/JSON 报告及各节点 `changes.json` 敏感信息审查 | 合并为一个测试点；密码经 stdin 传入审查进程，逐节点检查真实备份产物 |

DFX-04 与已评审 FIT-17 的安全目标相近，但自动化层级不同：FIT-17 按 XLSX O/P 列继续由产品侧 Bats UT 独立覆盖，DFX-04 则验证完整加固和回退链路生成的真实产物。

## 修改文件

| 文件 | 原作用 | 修改内容和原因 |
|---|---|---|
| `compliance_operator_suite_test.go` | Suite 初始化和有序生命周期 | 记录 XLSX 分组和独立 DFX 集合；将 suite 名称改为英文 |
| `hardening_main_path_test.go` | 加固生命周期 | 增加 FIT 名称和标签、FIT-03/FIT-06 独立测试点及逐节点 etcd 基线 |
| `hardening_dryrun_test.go` | dry-run 生命周期 | 将测试点映射到 FIT-10/FIT-11，并使用带分组的名称 |
| `hardening_rollback_test.go` | 回退生命周期 | 将测试点映射到 FIT-12/FIT-13/FIT-14，并使用英文名称 |
| `hardening_config_guard_test.go` | DFX 配置防护 | 增加 DFX 分组、用例标签和英文名称 |
| `hardening_idempotency_test.go` | DFX 加固/回退幂等 | 增加 DFX 分组、用例标签和英文名称 |
| `sensitive_info_audit_test.go` | DFX 产物审查 | 增加 DFX 分组、用例标签和英文名称 |
| `hardening_helpers.go` | 公共断言和生命周期 helper | g5 断言改为使用动态检测的 etcd 路径 |
| `utils/ssh_functions.go` | SSH 与指纹 helper | 增加严格的 manifest 动态检测；消除指纹采集期间的并发 map 写入 |
| `utils/scan_functions.go` | CIS/STIG 扫描 helper | 扫描挂载使用 entry master 动态检测的宿主机路径 |
| `utils/config.go`、`utils/const.go` | 环境配置 | 删除固定的 etcd 宿主机路径选项和默认值 |
| `../.env.template` | E2E 公共环境变量模板 | 删除固定 etcd data-dir 配置并说明动态检测方式 |
| `README.md` | E2E 使用说明 | 使用 XLSX 追溯关系和本指南替换旧的 CSV/聚合用例说明 |
| 产品 `lib/health_check.sh` | 节点与集群健康检查 | Ready 未达阈值时按 `ENV_HEALTH_CHECK_TIMEOUT` 每 15 秒重试，超时后才失败 |
| 产品 `lib/cluster.sh` | 集群编排与远端命令构造 | 校验健康检查超时，并在 harden/rollback 时传给远端节点进程 |
| 产品 `lib/backup.sh` | 备份与文件级回退 | `changes.json` 中的历史变更对象仅在当前内容与备份不同时进入恢复队列，保证重复回退直接返回 no-op |

产品目录修改了上述三个经用户分别明确授权的文件。产品目录中原有的 `config/cluster.yaml.before-e2e.20260909-143813` 未修改。

### DFX-03 重复回退修复

原 `restore_file` 只要在 `changes.json` 中看到 `manifest-arg` 或 `kubelet-field` 历史记录，就会无条件复制对应备份文件。`changes.json` 说明加固时曾经修改过什么，不能证明重复回退时当前文件仍与备份不同。这会导致内容已经一致的 manifest 和 kubelet 配置在每次回退时仍被复制，持续输出 `ROLLBACK_NOOP=0` 并触发无意义的组件重启。

修复后，`changes.json` 只负责提供候选路径。候选文件必须满足“当前文件不存在，或者 `cmp` 确认当前内容与备份不同”才会加入恢复队列。DFX-03 因此严格执行一次重复回退，并要求四个节点全部返回 `ROLLBACK_NOOP=1`、没有 `ROLLBACK_NOOP=0`、没有文件恢复，同时保持 S0 指纹不变。

### FIT-08 worker Pod UID 的环境约束

XLSX 明确要求 worker 上的业务 Pod UID 在加固前后不变。E2E 不会把 worker UID 变化降级为通过。若失败，E2E 会输出 before/after Pod 快照、各 worker kubelet 的 API endpoint 和近期驱逐事件。

已评审用例备注所依据的环境是 worker kubelet 连接高可用 VIP。BKE 可能在 kubelet kubeconfig 中写入 VIP IP，也可能写入 `master.bocloud.com` 一类 HA 域名，因此不能只按 kubeconfig 中的 host 字符串判断。执行 FIT-08 前必须从运行中 kubelet 的命令行解析真实 `--kubeconfig`，再在 worker 本机解析活动 API endpoint。

E2E 的失败诊断会动态读取 kubelet 进程的 `--kubeconfig` 参数，再通过该 kubeconfig 的活动 context 获取 API server，同时输出 kubeconfig 路径、context 和 server。它不会再假定 kubelet 固定使用 `/etc/kubernetes/kubelet.conf`。

E2E-A 会在修改集群状态前检查：端口必须等于 `WORKLOAD_CLUSTER_VIP_PORT`；host 是域名时，其 worker 本机解析结果必须包含 `WORKLOAD_CLUSTER_VIP_HOST`，且不得包含任何 master IP。这样既支持 BKE 的 HA 域名表示，也能阻止 worker 实际绕过 VIP 直连 master。

## 手动验证

下面的命令全部由用户手动执行。本次重构没有执行这些命令。

### Step 1：进入 E2E Go module

```bash
cd /root/e2e-auto-test/e2e
```

### Step 2：确认测试环境

从 entry node 执行，或按下面方式通过 SSH 执行：

```bash
ssh root@192.168.200.225 'kubectl config current-context'
ssh root@192.168.200.225 'kubectl get nodes -o wide'
ssh root@192.168.200.225 'kubectl -n kube-system get pods -o wide'
ssh root@192.168.200.225 'command -v yq jq sshpass kubectl nerdctl'
for node in 192.168.200.225 192.168.200.226 192.168.200.227 192.168.200.228; do ssh root@"$node" 'hostname; true'; done
```

确认 `e2e/.env` 包含正确的连接配置。远程 entry node 未部署工具时，E2E 默认从当前仓库的
`testdata/compliance-hardening` 打包并上传最小脚本制品；设置
`COMPLIANCE_HARDENING_SOURCE_DIR` 时优先使用显式指定的产品源码。远端已有完整工具时直接复用。
已有 `cluster.yaml` 时，E2E 会保留并校验其节点清单；文件
不存在时，E2E 根据解析后的 `TEST_NODE2~5_*` / `COMPLIANCE_NODE2~5_*` 配置自动生成，
权限设为 `600`，凭据仅通过 SSH stdin 写入远端且不会输出到 Ginkgo 日志。

### Step 3：保存 E2E 前的 etcd 基线

```bash
set -o pipefail
for node in 192.168.200.225 192.168.200.226 192.168.200.227; do
  result="$(ssh root@"$node" 'data_dir="$(yq eval -r '\''.spec.containers[] | select(.name == "etcd") | (.command[]?, .args[]?)'\'' /etc/kubernetes/manifests/etcd.yaml | awk '\''/^--data-dir=/{sub(/^--data-dir=/,"");print;exit} take_next{print;exit} $0=="--data-dir"{take_next=1}'\'')"; test -n "$data_dir"; test -d "$data_dir"; printf "data-dir=%s owner/group=%s" "$data_dir" "$(stat -c %U:%G "$data_dir")"')" || exit 1
  printf 'node=%s %s\n' "$node" "$result"
done | tee /tmp/compliance-etcd-before.txt
```

命令必须为每个 master 生成一条包含节点 IP、非空路径和 owner/group 的记录。由于当前四台机器的 hostname 都是 `localhost.localdomain`，基线使用 IP 区分节点。任何路径或 `stat` 结果缺失都表示前置条件失败。

### Step 4：代码级检查

```bash
cd /root/e2e-auto-test/e2e
gofmt -d installation/compliance-operator/*.go installation/compliance-operator/utils/*.go
go test -c -o /tmp/compliance-operator-e2e.test ./installation/compliance-operator
go vet ./installation/compliance-operator/...
```

`go test -c` 只编译测试二进制，不执行 Ginkgo suite。当前仓库没有为 compliance-operator 配置 lint target，请勿使用不存在的 Make target。

产品仓库仍保留 FIT-16/FIT-17 的补充 Bats 单元测试，与当前真实集群 E2E 相互独立：

UT 额外要求 `bats-core >= 1.5`。先检查：

```bash
command -v bats && bats --version
```

如果提示 `bats: command not found`，表示 UT 尚未执行。按照产品仓库 README，可选择以下一种安装方式。

在线安装：

```bash
cd /tmp
git clone https://github.com/bats-core/bats-core.git
cd bats-core
./install.sh /usr/local
bats --version
```

离线安装：先在一台已安装 Bats 的机器上打包并传到当前机器，然后由当前机器的 root 用户解压：

```bash
tar czf bats.tgz -C / usr/local/bin/bats usr/local/libexec/bats-core usr/local/lib/bats-core
scp bats.tgz root@<当前测试机IP>:/tmp/
tar xzf /tmp/bats.tgz -C /
bats --version
```

确认版本后再执行：

```bash
cd /root/compliance-operator/compliance-hardening
bats tests/unit/test_fit_gaps.bats
```

### Step 5：手动执行 E2E-A

该命令会将集群从 S0 变为加固态，并在结束时保留加固态：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='E2E-A && post-init' ./installation/compliance-operator
```

### Step 6：手动执行 E2E-B

只能在集群处于 S0 状态时执行：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='E2E-B && post-init' ./installation/compliance-operator
```

### Step 7：手动执行 E2E-C

应在 E2E-A 后执行。固定工作目录 `/tmp/compliance-e2e` 中必须存在时间戳和 S0 指纹文件：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='E2E-C && post-init' ./installation/compliance-operator
```

### Step 8：手动执行单条已评审 FIT

FIT-01 对应一个自动化测试点。该命令会执行加固，并在结束时保留加固态：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='FIT-01 && post-init' ./installation/compliance-operator
```

其他 FIT 也有追溯标签。FIT-02、FIT-12 等只校验生命周期后续状态的用例依赖前序阶段和持久化状态，不能作为独立的环境准备流程执行。

### Step 9：手动执行 DFX

不能使用单独的 `DFX` filter 在一次独立运行中正确执行全部 4 条 DFX，因为 DFX-02 要求加固态，DFX-03 要求已恢复的 S0 状态，DFX-04 还要求完整加固与回退产物已生成。执行全部 DFX 的受支持方式是运行 Step 10 的完整有序流程，它会在正确的阶段执行 4 条 DFX。

可以独立执行的 DFX-01 示例：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='DFX-01 && post-init' ./installation/compliance-operator
```

处于对应生命周期状态时，其他单条 DFX 的标签分别为 `DFX-02`、`DFX-03` 和 `DFX-04`。DFX-04 不支持在没有前序完整流程产物的环境中独立执行。

### Step 10：手动执行完整 compliance-operator E2E

这是推荐命令。完整有序 suite 会在结束前将集群恢复到 S0：

```bash
cd /root/e2e-auto-test/e2e
ginkgo -v --label-filter='post-init' ./installation/compliance-operator
```

### Step 11：校验 etcd owner/group

在 E2E-A 完成后、E2E-C 开始前，再次执行 Step 3 的循环，将最后的输出文件改为 `/tmp/compliance-etcd-after.txt`，然后比较：

```bash
diff -u /tmp/compliance-etcd-before.txt /tmp/compliance-etcd-after.txt
```

预期关系是：

```text
master1: before == after
master2: before == after
master3: before == after
```

预期结果不是所有 master 都等于 `root:root`。如果失败，需要保留 node、检测到的路径、before、after、完整 Ginkgo 输出和 hardening 日志。

### Step 12：E2E 后环境检查

以下检查均为只读：

```bash
ssh root@192.168.200.225 'kubectl get nodes -o wide'
ssh root@192.168.200.225 'kubectl -n kube-system get pods -o wide'
ssh root@192.168.200.225 'kubectl get crd | grep -i compliance || true'
ssh root@192.168.200.225 'kubectl get jobs,pods -A | grep -iE "compliance|hardening" || true'
for node in 192.168.200.225 192.168.200.226 192.168.200.227 192.168.200.228; do ssh root@"$node" 'hostname; test ! -e /var/lib/compliance-hardening/backups/.harden-origin; rc=$?; stat -c "%a %U:%G %n" /etc/kubernetes/manifests/*.yaml 2>/dev/null || true; exit $rc'; done
ssh root@192.168.200.225 'kubectl -n kube-system get pods -l component=etcd -o wide'
```

如果只执行了部分用例并导致集群停留在加固态，先读取 entry node 上的 `/tmp/compliance-e2e/harden-ts.txt`，人工确认时间戳。E2E-A 会在产品命令返回后立即保存这个时间戳，即使后续 `exit=0` 断言失败也会保留。先执行以下只读检查：

```bash
ssh root@192.168.200.225 'cat /tmp/compliance-e2e/harden-ts.txt 2>/dev/null || true'
for node in 192.168.200.225 192.168.200.226 192.168.200.227 192.168.200.228; do
  ssh root@"$node" 'printf "%s " "$(hostname)"; find /var/lib/compliance-hardening/backups -mindepth 1 -maxdepth 1 -type d -printf "%f\n" 2>/dev/null | sort | tail -5'
done
```

对于使用旧版 E2E 产生、尚未写入 `harden-ts.txt` 的失败现场，从 entry node 最新的编排日志中查找时间戳：

```bash
ssh root@192.168.200.225 'log=$(ls -1t /var/log/compliance-hardening/cluster-harden-*.log 2>/dev/null | head -1); echo "log=$log"; test -n "$log" && grep "本次集群加固备份时间戳" "$log" | tail -1'
```

只有当同一时间戳的备份目录在 4 个节点都存在时，才手动执行产品回退：

```bash
ssh root@192.168.200.225 'cd /root/compliance-operator/compliance-hardening && ts="$(cat /tmp/compliance-e2e/harden-ts.txt)" && test -n "$ts" && bash bin/k8s-hardening.sh cluster-rollback config/cluster.yaml -y --backup-subdir "$ts"'
```

如果本轮失败发生在时间戳留存修复之前，`harden-ts.txt` 不存在，则先从最新日志自动提取时间戳、打印并人工核对，再使用该值。不要把示例占位文字当成时间戳执行：

```bash
TS=$(ssh root@192.168.200.225 'log=$(ls -1t /var/log/compliance-hardening/cluster-harden-*.log 2>/dev/null | head -1); test -n "$log" && grep "本次集群加固备份时间戳" "$log" | tail -1' | grep -oE '[0-9]{8}-[0-9]{6}' | tail -1)
printf '待核对的统一备份时间戳：%s\n' "$TS"
test -n "$TS"

for node in 192.168.200.225 192.168.200.226 192.168.200.227 192.168.200.228; do
  ssh root@"$node" "test -d /var/lib/compliance-hardening/backups/$TS && echo '$node EXISTS' || echo '$node MISSING'"
done

# 仅在人工确认时间戳正确且四个节点全部为 EXISTS 后执行：
ssh root@192.168.200.225 "cd /root/compliance-operator/compliance-hardening && bash bin/k8s-hardening.sh cluster-rollback config/cluster.yaml -y --backup-subdir '$TS'"
```

执行回退前必须人工检查时间戳。本指南只提供命令，本次重构没有执行 rollback。

### 产品健康检查问题及修复

如果节点侧加固日志在固定等待 60 秒后报告 `节点未达多数就绪：2/4 Ready`，本轮加固已经修改了当前 master，随后因产品健康检查失败而停止；不要直接重跑 E2E-A。

原产品的 `lib/health_check.sh::check_cluster` 仅在 `kubectl get nodes` 完全没有返回节点时重试。一旦能够查询到节点但 Ready 数不足多数，例如 `2/4 Ready`，它会立即返回失败。函数读取了 `ENV_HEALTH_CHECK_TIMEOUT`，但没有用该值控制这个恢复等待。`lib/cluster.sh` 也没有把该变量传给远端 harden/rollback 进程。

修复后，无响应和 Ready 未达阈值都会每 15 秒重新采样，最长等待 `cluster.health_check_timeout`。单节点要求 `1/1 Ready`，HA 集群要求多数 Ready。编排同时把该超时传入远端 harden 和 rollback。E2E 继续保留 `cluster-harden exit=0` 强断言，不把真正的超时失败降级为通过。

先检查当前节点和控制面状态，并从最新编排日志确认统一备份时间戳：

```bash
ssh root@192.168.200.225 'kubectl get nodes -o wide; kubectl -n kube-system get pods -o wide'
ssh root@192.168.200.225 'log=$(ls -1t /var/log/compliance-hardening/cluster-harden-*.log 2>/dev/null | head -1); echo "log=$log"; test -n "$log" && grep -E "本次集群加固备份时间戳|节点未达多数就绪|健康检查失败|加固失败" "$log" | tail -20'
for node in 192.168.200.225 192.168.200.226 192.168.200.227 192.168.200.228; do
  ssh root@"$node" 'echo "node=$SSH_CONNECTION"; systemctl is-active kubelet; test -e /var/lib/compliance-hardening/backups/.harden-origin && cat /var/lib/compliance-hardening/backups/.harden-origin || true'
done
```

确认故障原因并使用本轮统一时间戳完成回退后，先把本地修改的三个产品文件同步到 entry node 的 `/root/compliance-operator/compliance-hardening/lib/`，再重新建立 Step 3 基线并运行 E2E-A。若恢复期间出现 `2/4 Ready`，日志应显示“节点暂未达就绪阈值”并继续采样；只有配置超时耗尽后才应失败。

## 验证状态

| 验证项 | 状态 | 命令 |
|---|---|---|
| 格式差异 | PASS | `gofmt -d installation/compliance-operator/*.go installation/compliance-operator/utils/*.go` |
| 仅编译检查 | PASS | `go test -c -o /tmp/compliance-operator-e2e.test ./installation/compliance-operator` |
| Go Vet | PASS | `go vet ./installation/compliance-operator/...` |
| 补充 Bats UT | NOT RUN - waiting for user validation | 从产品仓库执行 `bats tests/unit/test_fit_gaps.bats` |
| E2E-A | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='E2E-A && post-init' ./installation/compliance-operator` |
| E2E-B | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='E2E-B && post-init' ./installation/compliance-operator` |
| E2E-C | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='E2E-C && post-init' ./installation/compliance-operator` |
| 单条 FIT-01 | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='FIT-01 && post-init' ./installation/compliance-operator` |
| DFX-04 | NOT RUN - waiting for user validation | 随完整有序流程审查真实加固与回退产物 |
| 全部 DFX 独立运行 | NOT SUPPORTED | 使用完整有序流程，让 DFX-02～DFX-04 获得各自需要的状态和产物 |
| 单条 DFX-01 | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='DFX-01 && post-init' ./installation/compliance-operator` |
| Full E2E | NOT RUN - waiting for user validation | `ginkgo -v --label-filter='post-init' ./installation/compliance-operator` |
