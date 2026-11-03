# openFuyao 统一前置检查验证记录

> 历史记录，已被当前实现取代，不作为本 PR 的验收证据。

2026-08-10 的阶段性输出包含旧 Bundle digest、旧规则数量、fake collector 集成测试和被 logger
前缀污染的 JSON，已从 PR 删除，避免被误用为当前证据。当前验收只以本次提交重新执行的命令输出
和 MR 检查为准。

## 当前验收命令

- `make test-preflight`
- `make verify-preflight-bundle`
- `go mod tidy -diff && go mod verify`
- `cd preflight && go vet ./...`
- `go vet ./pkg/preflightbridge ./pkg/preflightconfig ./tools/preflight-bundle-gen`
- Linux/OrbStack 中构建单个 `bke` 二进制，并验证内置 policy/Bundle、终端宽度、真实 SSH TCP forwarding。

## 证据边界

- `pkg/preflightbridge/e2e_test.go` 使用 fake collector，只证明 adapter、binding、SDK 和 renderer 的进程内集成，不是 CLI/SSH/Kubernetes E2E。
- 不把本地日志或截图提交为长期证据；MR 只记录可复现命令、环境与明确的证据边界。
- development Bundle 不等于正式发布批准。正式 channel 仍需不可变 source revision、制品摘要/签名以及平台测试和审批引用。
- 真实环境结论必须记录二进制 commit、Bundle digest、输入摘要、OrbStack VM 信息、命令、退出码和未解决项。
