# Preflight SDK

`preflight` 是统一前置检查的独立 Go 模块，模块路径为
`gopkg.openfuyao.cn/preflight`。它包含通用 API、CEL 规则引擎、Collector 接口、
内置 Collector 和报告渲染，不依赖 bkeadm 的命令或 Provider 类型。九种内置运算符会映射为
固定 CEL 表达式；自定义规则可以携带受限 CEL 表达式，但只能读取 Collector 返回的 `actual`。

仓库根模块负责 bke CLI、Provider 输入解析和 openFuyao Host Adapter。
根 `go.mod` 通过下面的本地替换在同一仓库内联调 SDK：

```text
replace gopkg.openfuyao.cn/preflight => ./preflight
```

这个边界有三个目的：

- SDK 可以独立测试、版本化，并被其他产品 Host Adapter 复用；
- 产品配置、凭据和安装副作用不会进入通用 SDK；
- 根模块与 SDK 模块的依赖方向由 `arch_test.go` 持续校验。

自定义检查仍遵守同一边界：SDK 的 `collectors/custom` 只定义 Fact JSON、输出上限、脱敏投影和
Host Executor 接口；bkeadm Host Adapter 负责把 `preflight.yaml` 转成规则/实例，并在本机或
SSH 节点执行 `fileContent`/`exec`。Collector 只产生 Fact，不判断 PASS/FAIL；CEL 只计算布尔
结果；引擎统一应用 failure/unknown 策略并生成报告。

`exec` 是受控扩展点，不等同于“框架保证脚本只读”。框架能保证结构化参数、无 shell 拼接、
无隐式 sudo、固定环境、权限检查、超时和输出上限；脚本自身是否只读仍由提供者负责。因此生产
环境应由运维预置并审核脚本，不应把不可信用户提供的脚本直接交给 root preflight 执行。

因此修改后需要分别执行两个模块的测试：

```bash
cd preflight && go test -count=1 ./...
cd .. && go test -count=1 ./pkg/preflightbridge ./pkg/preflightconfig
```

bkeadm 在进程内把内置 `preflight.yaml` 现场编译为 Bundle 后交给 SDK；SDK 本身不读取
bkeadm 的 `preflight.yaml`，也不负责解析 `cluster.yaml` 或 `nodes.yaml`。
