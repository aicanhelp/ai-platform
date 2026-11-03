# 控制面 E2E 测试

覆盖 flux-sandbox **控制面**（Controller + Scheduler + CRD）的功能用例，对应
`flux-sandbox-test.md` 中的 FS-FUNC-001 ~ FS-FUNC-039（共 32 个功能用例）。

与数据面 e2e（`../` 包）不同，本测试**假设控制面已在集群中部署就绪**，测试夹具
只做"验证 /readyz + CRD 兜底 apply + 建立客户端 + port-forward"。

## 前置条件（通用）

1. Kubernetes 集群已部署，以下组件 **Running**：
   - `flux-sandbox-controller`（Deployment，label `app=flux-sandbox-controller`）
   - `flux-sandbox-scheduler`（StatefulSet，label `app=flux-sandbox-scheduler`）
   - CRD 已安装：`make crd-apply`
2. Controller `/readyz` 返回 `{"ready":true}`，Scheduler `/readyz` 返回 200。
3. 以下镜像已加载到集群节点（离线环境必须，在线环境需可拉取）：
   - Agent：chart/ConfigMap 配置的镜像名（如 `cr.openfuyao.cn/openfuyao/flux-sandbox/agent:latest`），
     以及内置回退名 `flux-sandbox/agent:latest` —— FS-FUNC-037 删除 ConfigMap 后 controller 会回退到
     该内置默认镜像；节点侧 tag 需写成 CRI 规范化名 `docker.io/flux-sandbox/agent:latest`
     （两段名不加 `library/` 前缀，否则 kubelet 判定镜像不存在而触发远端拉取）；
   - execd：`opensandbox/execd:v1.0.20`；resource-reserve：`registry.k8s.io/pause:3.10`；
     测试沙箱：`docker.io/library/busybox:latest`。
4. namespace `flux-system` 已创建。
5. 本机 `kubectl` 可访问集群（`~/.kube/config` 或 `E2E_KUBECONFIG`）。

> 部分用例对集群拓扑有额外要求（多 scheduler、多节点、controller flag 等），不满足时会 `t.Skip` 并说明原因。

## 运行

```bash
# 运行全部控制面功能用例（假设集群已就绪）
make test-e2e-cp

# 只跑某一组用例
make test-e2e-cp -- -run TestSandboxGroup
make test-e2e-cp -- -run TestFastPath
make test-e2e-cp -- -run TestAgentPod

# 只跑单条用例（-run 接任意用例名或 Go test 正则；排查失败、复跑抖动用例时使用）
make test-e2e-cp -- -run TestRuntime_ExecdPing
make test-e2e-cp -- -run TestSandboxFault_AutoRecreate

# 运行破坏性用例（会 scale scheduler、删 ConfigMap，用后自动恢复）
E2E_DESTRUCTIVE=1 make test-e2e-cp

# 运行大规模分片用例（需 controller 启动参数 --max-pod-per-sub-sandbox-group=2）
E2E_BIG_SHARDING=1 make test-e2e-cp -- -run TestSharding
```

### 全量执行（一条命令跑完所有用例）

裸 `make test-e2e-cp` 只执行非门控用例：FS-FUNC-030/037（破坏性）、FS-FUNC-039（大规模分片）、
FS-FUNC-013（E2B）默认 `t.Skip`。要一次性执行全部用例：

```bash
make test-e2e-cp-full
```

等价于手工执行：

```bash
E2E_DESTRUCTIVE=1 E2E_BIG_SHARDING=1 E2E_E2B_ENABLED=1 E2E_SCHEDULER_REPLICAS=3 E2E_CLEANUP=1 \
  go test -tags=e2e -count=1 -v -timeout=40m ./test/e2e/controlplane/
```

全量执行在通用前置条件之外还需要（不满足时对应用例 `t.Skip`，而 FS-FUNC-039 会直接 FAIL）：

1. scheduler 副本数 >= 2（chart 默认 3），且 controller 启动参数
   `--expected-scheduler-replicas` 与实际副本数一致（FS-FUNC-003/026/030 需要）；
2. controller 启动参数 `--max-pod-per-sub-sandbox-group=2`（仅 FS-FUNC-039 需要；
   其余用例的 agentPodMax <= 2，不受该参数影响）；
3. E2B 用例（FS-FUNC-013）要求节点存在 E2B CRI shim（`/run/cri-multiplex.sock`）。

单节点集群 FS-FUNC-027（跨节点路由）因拓扑限制固定 `t.Skip`，即单节点全量成功的标准结果为
30 PASS + 1 SKIP，一次全量约 15~20 分钟。

其它说明：

- 所有 e2e make 目标均带 `-count=1`：e2e 直打真实集群，禁用 go test 结果缓存，
  避免上一次的 PASS 缓存掩盖本次真实回归；
- 破坏性用例（030/037）会 scale scheduler、删 ConfigMap 并重启 controller，用例自身在
  cleanup 中恢复全部状态；测试夹具在 controller Pod 被替换后会自动重建 port-forward 与
  gRPC 连接，因此 037 之后同一进程内的用例不受影响；
- 已知抖动：FS-FUNC-023（TestSandboxFault_AutoRecreate）依赖 agent 心跳超时（默认 60s）
  触发 Lost 检测，长时间持续负载下 scheduler reconcile 变慢时，Lost 可能超出 240s 等待窗口
  （单独重跑该用例即可通过；同组 LostManual 耗时同步变长可佐证为负载因素而非产品缺陷）。

## 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `E2E_KUBECONFIG` | `~/.kube/config` | kubeconfig 路径 |
| `E2E_CP_NAMESPACE` | `flux-system` | 控制面命名空间 |
| `E2E_CONTROLLER_POD` | (自动按 label 发现) | Controller Pod 名 |
| `E2E_SCHEDULER_REPLICAS` | `3` | 期望 scheduler 副本数；<2 时多 scheduler 用例跳过 |
| `E2E_AGENT_IMAGE` | `flux-sandbox/agent:latest` | Agent 镜像 |
| `E2E_EXECD_IMAGE` | `opensandbox/execd:v1.0.20` | execd 镜像 |
| `E2E_SANDBOX_IMAGE` | `docker.io/library/busybox:latest` | 测试沙箱镜像 |
| `E2E_RESERVE_IMAGE` | `registry.k8s.io/pause:3.10` | resource-reserve sidecar 镜像 |
| `E2E_DESTRUCTIVE` | (空) | 非空才跑破坏性用例（FS-FUNC-030、037） |
| `E2E_BIG_SHARDING` | (空) | 非空才跑大规模分片用例（FS-FUNC-039） |
| `E2E_CLEANUP` | (空) | 非空则在 teardown 删除测试创建的 SandboxGroup |

## 用例清单

| 文件 | 用例编号 | 说明 |
|------|----------|------|
| `sandboxgroup_test.go` | FS-FUNC-001~005 | SandboxGroup 创建/状态回写、SSG 拆分、Round-Robin、资源继承、级联删除 |
| `agentpod_test.go` | FS-FUNC-006,007,011,012,013,014 | Agent Pod 双容器结构、env 注入、自愈、execd init 容器、E2B 不注入、containerd 无 execd 失败 |
| `fastpath_test.go` | FS-FUNC-015,016,018,020 | FastPath gRPC 创建、CR 异步补写、删除幂等、容量满拒绝 |
| `fault_test.go` | FS-FUNC-022,023,024 | Agent Pod 删除后 Lost（Manual）、AutoRecreate 重调度、孤儿检测无误删 |
| `router_test.go` | FS-FUNC-025,026,027 | Router allocated 最小优先、Scheduler 隔离、跨节点通信 |
| `health_test.go` | FS-FUNC-028,029,030 | Controller/Scheduler 健康检查、Scheduler NotReady 闸门 |
| `runtime_e2e_test.go` | FS-FUNC-031,032,033,035 | containerd execd /ping、/command、Agent /status、重复创建幂等 |
| `controller_config_test.go` | FS-FUNC-036,037 | ConfigMap 默认配置生效、缺失时回退默认值 |
| `resource_test.go` | FS-FUNC-038 | 资源预留双容器隔离 |
| `sharding_test.go` | FS-FUNC-039 | 大规模分片（4 SSG × 2 Agent Pod × Sandbox） |

## 测试隔离与清理

- 每个用例用 `cpName(t)` 派生唯一 SandboxGroup / Sandbox CR 名，`t.Cleanup` 注册删除
  SandboxGroup（foreground 删除，级联清 SSG + Agent Pod + Sandbox）。
- 破坏性用例（030/037）默认 `t.Skip`，需 `E2E_DESTRUCTIVE=1` 才跑，并在 cleanup 中
  恢复 scheduler 副本数 / 重建 ConfigMap。
- 多 scheduler/多节点用例（003/026/027）在 `E2E_SCHEDULER_REPLICAS<2` 时 `t.Skip`。

## 不覆盖

- 性能用例（FS-PERF-*）：由 `tools/sandbox-benchmark` 工具覆盖（见其 README）。
- 数据面 Agent HTTP API 详测：由数据面 e2e（`../` 包）覆盖。
