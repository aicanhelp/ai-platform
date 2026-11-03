# Prediction Model 示例制品

> 本目录提供一份可被 **Hermes Prediction Sidecar** 直接加载的预测模型制品示例，目录结构已满足运行时加载要求，开箱即用。

## 示例概览

`Qwen/Qwen3-8B/baseline-conversation-data/` 是为 `Qwen/Qwen3-8B` 生成的一份模型制品：

| 字段           | 值                           |
| -------------- | ---------------------------- |
| `targetModel`  | `Qwen/Qwen3-8B`              |
| `modelVersion` | `baseline-conversation-data` |
| `backend`      | `xgboost`                    |

需要强调的是，它**不是**原始训练数据集，也**不是**训练源码，而是一份已训练完成、可供 Prediction Sidecar 从 `artifactRoot/targetModel/modelVersion` 路径直接加载的产物。

该示例以 `xgboost` 为后端，且只包含 aggregate 场景所需的两个 slot（`aggregate_ttft`、`aggregate_tpot`）。queue state 以 in-model 特征 `isQueued` 表达，`slotFeatureColumns` 与当前 sidecar schema 对齐。

## 目录结构

```text
examples/prediction-model/Qwen/Qwen3-8B/baseline-conversation-data/
├── manifest.json
├── metadata.json
├── report.json
└── slots/
    ├── aggregate_tpot.xgboost.json
    └── aggregate_ttft.xgboost.json
```

各文件职责：

- **`manifest.json`** — 加载阶段最关键的运行时清单。声明制品标识、后端、已启用的 slot，以及 slot 到文件的映射（`slotFiles`）。
- **`metadata.json`** — 运行时元数据。记录特征列、阈值、holdout 配置，以及每个 slot 的状态（含尚未达到阈值、未被启用的 slot）。
- **`report.json`** — 训练报告。记录 promoted slot 的指标，以及加载阶段与样本生成阶段合并后的 `dropReasons`。
- **`slots/`** — 存放 `manifest.json` 所引用的实际模型文件。

## 加载条件：Prediction Sidecar 能否启动

只有满足以下全部条件，Prediction Sidecar 才能成功加载该制品：

- 制品位于 `artifactRoot/targetModel/modelVersion` 路径下；
- Sidecar 的启动参数 `--target-model`、`--model-version` 与 manifest 中声明的值完全一致；
- `manifest.json` 与 `metadata.json` 都存在，且均为合法 JSON；
- 两个文件声明的 `backend` 一致；
- `promotedSlots` 中的每个 slot 都能在 `slotFiles` 中找到映射；
- 每个 promoted slot 对应的文件都真实存在于磁盘；
- 当前 Sidecar 支持制品声明的 `backend`。

> **注意**：「可加载」只代表预测器能成功启动，并不代表 Hermes Router 会对每个请求都使用预测结果——这是两道相互独立的门槛。

## 生效条件：Hermes Router 何时真正使用预测结果

只有同时满足以下条件，Router 才会把预测输出用于路由决策：

- 请求携带非空的 `target_model`；
- 请求的目标模型与已加载制品的目标模型一致；
- 特征提取器能为端点构造出完整的输入；
- Sidecar 具备当前 Pod 类型与 queue family 运行所需的完整 promoted slot 组合：
  - **aggregate**：需要对应 queue family 的 TTFT 与 TPOT 两个 slot；
  - **prefill**：需要对应的 disagg TTFT slot；
  - **decode**：需要对应的 disagg TPOT slot；
- Sidecar 的输出通过 schema 校验，且能映射回已收集的端点。

本示例只提供 `aggregate_ttft` 与 `aggregate_tpot` 两个 slot，因此它**只能影响 aggregate 场景的流量**，无法覆盖 disaggregated 场景所需的 slot 组合。

当特征缺失、输出非法或相关预测不可用时，预测评分器采用 **fail-open** 策略：回退到基于快照的评分，而不是直接让路由失败。

## 最小接入

至少要让 Prediction Sidecar 以与制品一致的标识启动：

```yaml
inferenceExtension:
  prediction:
    enabled: true
    artifactRoot: /models/prediction
    targetModel: Qwen/Qwen3-8B
    modelVersion: baseline-conversation-data
```

同时保证 Unix socket 配置前后一致：

- 共享 Chart helper 会把上述值转换为 `--artifact-root`、`--target-model`、`--model-version` 传给 sidecar；
- sidecar 套接字默认路径为 `/var/run/hermes/prediction.sock`；
- scorer 的 `predictorAddress` 必须使用 `unix://` 地址，例如 `unix:///var/run/hermes/prediction.sock`。

## 训练

完整的训练命令、输入要求、提升阈值（promotion threshold）、后端选择，以及制品的发布行为，参见 [sidecar/prediction/README.md](../../sidecar/prediction/README.md)。