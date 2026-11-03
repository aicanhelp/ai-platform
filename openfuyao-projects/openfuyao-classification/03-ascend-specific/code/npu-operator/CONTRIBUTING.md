# 贡献指引

感谢参与 npu-operator 的开发。本指引面向代码与文档贡献者；文档维护规范见 [docs/README.md](docs/README.md)。

## 开发环境

- Go 1.24+（以 `go.mod` 为准）
- Docker 或其他容器运行时
- 可访问的 Kubernetes 集群（本地开发可用 kind / minikube）
- 本项目基于 [Kubebuilder](https://book.kubebuilder.io/) 脚手架（v4 layout）

## 开发流程

1. Fork 仓库并从 `master` 创建特性分支
2. 修改代码与文档（文档联动要求见 [docs/README.md](docs/README.md) 第 2 节）
3. 本地验证：

   ```bash
   make test      # 单元测试（含 envtest）
   make lint      # golangci-lint
   make build     # 构建 manager 二进制
   ```

   修改了 CRD（`api/v1/`）或 RBAC 时，先运行 `make manifests generate` 重新生成清单，并将生成产物一并提交；
4. 提交 MR 并填写变更说明；有用户可感知行为变更的，请在 CHANGELOG.md 的 Unreleased 段追加记录。

## 提交与 MR 规范

- Commit / MR 标题使用 `type: 简述` 格式，type 取值：`feat` / `fix` / `docs` / `refactor` / `chore`
- 一个 MR 聚焦一件事，避免混合无关改动
- 生成产物（CRD、RBAC、deepcopy）与源码修改放在同一 MR 中提交

## 评审期望

- CI（lint、test）通过
- 评审以正确性与代码-文档一致性为主
- 评审意见讨论后，由作者跟进更新；文档可在后续 PR 中补齐

## 目录速览

| 路径 | 内容 |
|---|---|
| `api/v1/` | `NPUClusterPolicy` CRD 定义 |
| `internal/controller/` | Reconcile 与渲染管线实现 |
| `config/`、`charts/` | Kustomize 清单与 Helm Chart |
| `docs/design/` | 设计文档 |
