# Checkpoint & Restore E2E 测试指南

本文档介绍如何构建、部署和运行 Checkpoint 与 Restore 功能的端到端（E2E）测试。

---

## 1. 前置条件

- Kubernetes 集群（v1.28+），`kubectl` 已配置
- Docker 或其他容器运行时
- Go 1.24+
- Ginkgo v2（测试框架）
- 集群中已部署 `sandbox-system` namespace
- `kata-qemu` RuntimeClass 已创建（Restore 测试中 RESTORE-E2E-008 需要）

---

## 2. 构建镜像

项目根目录下通过 Makefile 提供三个组件的镜像构建 target。

### 2.1 agent-sandbox-controller

```bash
# 使用默认镜像名 agent-sandbox-controller:latest
make docker-build-controller

# 指定自定义镜像名和 tag
make docker-build-controller CONTROLLER_IMG=<registry>/agent-sandbox-controller:<tag>
```

Dockerfile: `dockerfiles/agent-sandbox-controller.Dockerfile`（多阶段构建，内含 Go 编译）

### 2.2 sandbox-manager

```bash
# 使用默认镜像名 sandbox-manager:latest
make docker-build-manager

# 指定自定义镜像名和 tag
make docker-build-manager MANAGER_IMG=<registry>/sandbox-manager:<tag>
```

Dockerfile: `dockerfiles/sandbox-manager.Dockerfile`（多阶段构建，内含 Go 编译）

### 2.3 node-agent

node-agent 需要先本地编译二进制，再构建镜像：

```bash
# 第一步：编译二进制
CGO_ENABLED=0 GOOS=linux go build -o bin/node-agent ./cmd/node-agent/

# 第二步：构建镜像（注意 context 是 bin/ 目录）
docker build -f dockerfiles/node-agent.Dockerfile -t node-agent:latest bin/

# 指定自定义镜像名
docker build -f dockerfiles/node-agent.Dockerfile -t <registry>/node-agent:<tag> bin/
```

Dockerfile: `dockerfiles/node-agent.Dockerfile`（基于 alpine:3.20，直接 COPY 预编译二进制）

### 2.4 推送镜像

构建完成后将镜像推送到集群可访问的 registry：

```bash
docker push <registry>/agent-sandbox-controller:<tag>
docker push <registry>/sandbox-manager:<tag>
docker push <registry>/node-agent:<tag>
```

### 2.5 自定义 kata-shim（Restore 功能需要）

Restore 功能依赖自定义 kata-shim（含 `detectRestoreSnapshot` 逻辑），需要在已有 kata-containers 源码基础上编译。

```bash
# 进入 kata-containers 源码目录
cd /path/to/kata-containers/src/runtime

# 编译 containerd-shim-kata-v2
make containerd-shim-v2

# 验证生成的二进制文件（约 67MB，包含 restore 逻辑）
ls -lh containerd-shim-kata-v2
```

---

## 3. 部署

### 3.1 一键部署（推荐）

使用 `hack/deploy.sh` 脚本一键部署所有组件：

```bash
bash hack/deploy.sh
```

部署顺序：

| 步骤 | 组件 | 说明 |
|------|------|------|
| 1/6 | CRD | `kubectl apply -f config/crd/bases/` |
| 2/6 | sandbox-controller-manager | `kubectl apply -k config/default/`（含 RBAC、Webhook） |
| 3/6 | sandbox-manager | `kubectl apply -k config/sandbox-manager/`（含 RBAC、Ingress） |
| 4/6 | node-agent | DaemonSet + RBAC + NRI ConfigMap |
| 5/6 | 等待 Pods Ready | 自动等待各组件就绪 |
| 6/6 | 验证 | 打印 Pod 状态和 controller 日志 |


### 3.2 确认 Feature Gate 已开启

controller 部署清单中默认包含 `--feature-gates=SandboxCheckpoint=true`，确认方式：

```bash
kubectl get deploy sandbox-controller-manager -n sandbox-system \
  -o jsonpath='{.spec.template.spec.containers[0].args}' | tr ',' '\n' | grep feature
```

预期输出包含：`--feature-gates=SandboxCheckpoint=true`

### 3.3 确认 node-agent 就绪

```bash
kubectl get ds sandbox-node-agent -n sandbox-system
```

确认 `READY` 列与 `DESIRED` 列数值一致。

### 3.4 部署自定义 kata-shim

前提：节点已通过 kata-deploy 安装了标准 kata-containers。

```bash
# 1. 备份原始 shim
sudo cp /opt/kata/bin/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2.bak

# 2. 替换 shim（先移动再复制，避免 "Text file busy"）
sudo mv /opt/kata/bin/containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2.old
sudo cp containerd-shim-kata-v2 /opt/kata/bin/containerd-shim-kata-v2

# 3. 重启 containerd
sudo systemctl restart containerd

# 4. 验证替换成功（新文件约 67MB，原始约 46MB）
ls -lh /opt/kata/bin/containerd-shim-kata-v2
```

### 3.5 确认 Restore Feature Gate 已开启

controller 部署清单中需包含 `--feature-gates=SandboxRestore=true`：

```bash
kubectl get deploy sandbox-controller-manager -n sandbox-system \
  -o jsonpath='{.spec.template.spec.containers[0].args}' | tr ',' '\n' | grep feature
```

预期输出包含：`--feature-gates=SandboxCheckpoint=true` 和 `--feature-gates=SandboxRestore=true`

---

## 4. 测试用例概览

测试文件：`test/e2e/checkpoint_test.go`

测试分为两个 Describe 块：**Checkpoint**（控制器级别）和 **Snapshot File Persistence**（存储后端级别）。

### 4.1 Checkpoint 控制器测试（13 个用例）

| 用例 ID | 名称 | 覆盖功能 |
|---------|------|---------|
| CP-E2E-001 | checkpoint end-to-end lifecycle | 完整生命周期：创建 Sandbox → 创建 Checkpoint → 验证 Creating → Succeeded，验证 archiveLocation、archiveSize、keepRunning |
| CP-E2E-002 | non-kata runtime failure | 非 Kata 运行时的 Sandbox 创建 Checkpoint 应失败，验证 `RuntimeDetectionFailed` 错误消息 |
| CP-E2E-003 | nonexistent sandbox failure | 引用不存在的 Sandbox 应失败，验证 `SourceSandboxNotFound` 错误消息 |
| CP-E2E-004 | without sandboxName failure | 未指定 sandboxName 应失败，验证 `sandboxName is required` 错误消息 |
| CP-E2E-005 | TTL completion time | 设置 TTL 的 Checkpoint 成功后应记录 completionTime |
| CP-E2E-006 | TTL auto-deletion | 设置短 TTL (15s) 后，GC 控制器应在 TTL 过期后自动删除 Checkpoint |
| CP-E2E-007 | manual delete finalizer cleanup | 手动删除 Succeeded 的 Checkpoint 应触发 finalizer 清理归档文件并移除 finalizer |
| CP-E2E-008 | target container | 多容器 Sandbox 中通过 `targetContainer` 指定特定容器进行快照，验证 containerID 匹配 |
| CP-E2E-009 | conditions state transition | 验证成功后 Conditions 状态：Archived=True、Ready=True、Progressing 被移除 |
| CP-E2E-010 | succeeded checkpoint idempotency | Succeeded 状态的 Checkpoint 在触发额外 Reconcile 后状态保持不变（幂等性） |
| CP-E2E-011 | TTL day format | 验证 `1d` 天数格式的 TTL 能被正确解析且不会提前过期 |
| CP-E2E-012 | FeatureGate disabled | 禁用 `SandboxCheckpoint` FeatureGate 后，Checkpoint CR 不被处理（phase 为空、无 finalizer、无 conditions） |
| CP-E2E-013 | Node-Agent unreachable retry | node-agent 不可达时 Checkpoint 保持 Pending（不 Failed），node-agent 恢复后自动成功 |

### 4.2 存储后端测试（13 个用例，Label: `storage-persistence`）

| 用例 ID | 名称 | 覆盖功能 |
|---------|------|---------|
| SP-E2E-001 | local backend full lifecycle | Local 后端完整生命周期：Upload → WriteMetadata → GetChecksum → Download → Delete |
| SP-E2E-002 | CSI backend full lifecycle | CSI 后端完整生命周期 |
| SP-E2E-003 | NFS backend full lifecycle | NFS 后端完整生命周期 |
| SP-E2E-004 | registry multi-backend coexistence | 多后端注册表共存：同时注册 Local/CSI/NFS 并分别 Upload |
| SP-E2E-005 | config-based backend initialization | 基于环境变量配置初始化后端（含大小写不敏感、未知类型报错） |
| SP-E2E-006 | checksum integrity verification | SHA256 校验和完整性验证：正常匹配 + 篡改检测 |
| SP-E2E-007 | metadata round-trip with all fields | Metadata 全字段写入和读取一致性 |
| SP-E2E-008 | delete cleans up archive and metadata | 删除操作同时清理归档文件和 metadata.json（覆盖 Local/CSI/NFS 三种后端） |
| SP-E2E-009 | location format validation | 各后端拒绝非法 location 格式（如 CSI 拒绝 `s3://`、NFS 拒绝 `local://`） |
| SP-E2E-010 | backend constructor validation | 构造函数参数校验：空路径、不存在路径、非目录路径 |
| SP-E2E-011 | config rejects missing env vars | 缺少必要环境变量时 S3/MinIO/CSI/NFS 后端初始化应报错 |
| SP-E2E-012 | atomic write verification | Upload 成功后不留 `.tmp` 临时文件（原子写入） |
| SP-E2E-013 | multiple checkpoints in same namespace | 同一 namespace 下多个 Checkpoint 的归档互相隔离，删除一个不影响其他 |

### 4.3 Restore 控制器测试（10 个用例）

测试文件：`restore_test.go`

| 用例 ID | 名称 | 覆盖功能 |
|---------|------|---------|
| RESTORE-E2E-001 | restore from succeeded checkpoint | 完整 Restore 流程：创建 Sandbox（带 restore-from 注解）→ Restoring → Running，验证 Pod 注解注入、Status 字段（RestoreFromCheckpoint、RestoreCompletionTime、NodeName）、Conditions（Restoring=False/RestoreCompleted、Ready=True） |
| RESTORE-E2E-002 | failed checkpoint | 引用 Failed 状态的 Checkpoint，Sandbox → Failed，Message 包含 "not Succeeded" |
| RESTORE-E2E-003 | nonexistent checkpoint | 引用不存在的 Checkpoint，Sandbox → Failed，Message 包含 "not found" |
| RESTORE-E2E-004 | missing ArchiveLocation | Checkpoint 无 ArchiveLocation，Sandbox → Failed，Message 包含 "no ArchiveLocation" |
| RESTORE-E2E-005 | missing RuntimeType | Checkpoint 无 RuntimeType，Sandbox → Failed，Message 包含 "no RuntimeType" |
| RESTORE-E2E-006 | restore pod annotations | 验证 Restore Pod 注入 4 个注解：archive-location、restore-checkpoint-ref、restore-runtime-type、restore-vmm-type（kata-qemu → qemu） |
| RESTORE-E2E-007 | no restore annotation | 无 restore-from 注解的 Sandbox 正常创建到 Running，无 restore 相关 Status 和 Conditions |
| RESTORE-E2E-008 | invalid archive location | DownloadSnapshot 失败阻断 sandbox 创建（需自定义 kata-shim），Pod 保持 Pending，Sandbox 持续 Restoring，验证 FailedCreatePodSandBox 事件和 Consistently 30s 窗口 |
| RESTORE-E2E-009 | valid archive cold start | 有效 archive 路径，Sandbox 冷启动到 Running，验证 RestoreCompleted 状态 |
| RESTORE-E2E-010 | completion status transition | 完整验证 Restoring → Running 状态转换：Restoring Condition 从 True/PodCreated → False/RestoreCompleted，Message 包含 "Sandbox restored from checkpoint" |

---

## 5. 运行测试

### 5.1 仅运行 Checkpoint 测试

```bash
go test --tags=e2e . -v -ginkgo.v -ginkgo.focus="Checkpoint|Snapshot File Persistence"
```

### 5.2 仅运行 Restore 测试

```bash
go test --tags=e2e . -v -ginkgo.v -ginkgo.focus="Restore"
```

### 5.3 运行全部测试

```bash
go test --tags=e2e . -v -ginkgo.v -ginkgo.focus="Checkpoint|Restore|Snapshot File Persistence"
```
---

## 6. 测试环境自动行为

### BeforeSuite

测试套件启动时自动执行：

1. **检测 Kata 运行时**：检查集群是否存在 `kata-qemu` RuntimeClass，记录到 `hasKataRuntime` 变量
2. **Patch node-agent 到 mock 模式**：将 node-agent DaemonSet 的启动参数修改为 `--containerd-mode=mock`，绕过真实 containerd/kata 运行时依赖
3. **等待 node-agent 滚动更新完成**

### AfterSuite

测试套件结束时自动执行：

1. **恢复 node-agent 原始参数**：将 DaemonSet 的启动参数恢复到测试前状态
2. **等待 node-agent 恢复就绪**

### BeforeEach / AfterEach

每个用例执行前后自动：
- 创建独立 namespace
- 创建测试用 Sandbox 和 Checkpoint 对象
- 清理创建的 Sandbox 和 Checkpoint 资源
- 移除 namespace 内所有 Checkpoint 的 finalizer（`agents.kruise.io/checkpoint`），避免 namespace 删除卡在 Terminating
- 删除 namespace 并等待完全清理
