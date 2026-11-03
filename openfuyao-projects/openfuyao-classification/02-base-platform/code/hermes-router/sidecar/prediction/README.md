# Prediction Sidecar 训练指南

> 本文档介绍如何为 Hermes Router 的 prediction sidecar 训练、发布并校验 prediction artifact bundle（下文简称 bundle）。
>
> 如果你只想先看仓库里现成的示例 bundle，请直接参考 [examples/prediction-model/README.md](../../examples/prediction-model/README.md)。

## 训练流程概览

prediction sidecar 提供两个入口命令：

- **`prediction-sidecar`** — 从已发布的 bundle 加载模型，对外提供 gRPC 预测服务。
- **`prediction-train`** — 基于已完成的请求记录，构建一份新的 bundle。

`prediction-train` 的完整流程：

1. 从 `--input` 加载已完成请求的 JSONL 记录；
2. 将这些记录转换成训练样本；
3. 只保留 `target_model` 与 `--target-model` 匹配的样本；
4. 按配置的 holdout ratio，把每个 slot 切分为训练集与验证集；
5. 对样本量达到阈值的 slot，逐个训练模型；
6. 将 `manifest.json`、`metadata.json`、`report.json` 与 `slots/*` 写入 staging 目录；
7. 先写 staging，再将其重命名到 `artifact_root/target_model/model_version`，确保 sidecar 不会读到只写了一半的 bundle。

## 训练数据来源

`prediction-train` 不负责采集请求流量。其训练输入为 Hermes Router 在请求完成后持久化的 **completed request JSONL**。

训练数据的生成与使用流程如下：

1. 在 Router 或 Helm values 中配置 `requestTracking.persistence`；
2. 将 `persistence.enabled` 设置为 `true`。仓库示例 profile 默认值为 `false`，未启用时不会生成训练数据；
3. Router 的 `RequestLifecycleTracker` 在请求完成时，将记录写入 inflight store；
4. inflight store 根据 `flushThreshold` 和 `flushInterval`，将已完成记录追加至 `persistence.outputPath`；
5. 将该 JSONL 文件或其副本、合并结果作为 `prediction-train --input` 的输入。

以下配置与仓库中的 profile 及 chart 保持一致：

```yaml
requestTracking:
  storeName: hermes-inflight
  persistence:
    enabled: true
    flushThreshold: 100
    outputPath: /tmp/hermes-inflight/completed.jsonl
```

因此，示例命令中的 `--input` 应指向上述 `outputPath`。本仓库默认示例路径为 `/tmp/hermes-inflight/completed.jsonl`。

注意：按照当前示例配置，JSONL 文件写入 EPP 容器内部，宿主机默认无法访问。训练前可选择以下任一方案：

1. 从 EPP Pod 中复制已生成的 JSONL 文件到训练环境。
2. 通过 Helm 的 `inferenceExtension.volumes` 和 `inferenceExtension.volumeMounts`，将 `outputPath` 的父目录挂载到节点目录或持久化卷。

例如，下面的 values 覆盖会将容器内的 `/tmp/hermes-inflight/completed.jsonl` 写入 EPP Pod 所在节点的 `/var/lib/hermes-router/request-tracking/completed.jsonl`：

```yaml
inferenceExtension:
  routing:
    requestTracking:
      persistence:
        enabled: true
        outputPath: /tmp/hermes-inflight/completed.jsonl
  volumes:
    - name: request-tracking-data
      hostPath:
        path: /var/lib/hermes-router/request-tracking
        type: DirectoryOrCreate
  volumeMounts:
    - name: request-tracking-data
      mountPath: /tmp/hermes-inflight
```

`hostPath` 仅适用于需要直接访问 EPP 所在节点文件系统的场景。若训练任务在其他节点或其他 Pod 中运行，请改用可被训练任务挂载的 PVC 或其他共享存储。无论使用何种卷类型，都必须确保挂载目录对 EPP 容器的运行用户可写。随后，将容器内路径、复制后的文件路径或共享存储中的对应路径传入 `prediction-train --input`。

## 训练输入要求

`prediction-train` 期望的输入是上述已完成请求的 JSONL：**一行一条 JSON 记录**，且能被当前 loader 解析为已完成请求记录与训练样本。当前 `schemaVersion` 必须为 `1`。

单行记录的简化示例如下（aggregate 场景；真实落盘还可能包含 `prediction`、`selectedRoute`、`missingFeatureReasons` 等字段，训练 loader 主要依赖下面这些键）：

```json
{
  "schemaVersion": 1,
  "requestId": "req-001",
  "model": "Qwen/Qwen3-8B",
  "completedAt": "2026-07-22T03:00:00Z",
  "inputTokenLength": 128,
  "outputLength": 64,
  "ttftMillis": 42.5,
  "totalLatencyMillis": 380.0,
  "predictionInputs": {
    "leader": {
      "inputId": 0,
      "podType": "aggregate",
      "inputTokenLength": 128,
      "prefixCacheScore": 0.25,
      "numRequestWaiting": 2,
      "numRequestRunning": 4,
      "kvCacheUsage": 0.35,
      "inflightTokens": 512,
      "hbmUtilization": 0.4,
      "aiCoreUtilization": 0.55,
      "bandwidthUtilization": 0.2,
      "serverIP": "10.0.0.12"
    }
  }
}
```

字段职责（训练侧）：

| 键 | 作用 |
| --- | --- |
| `schemaVersion` | 必须为 `1`，否则整行丢弃 |
| `model` | 目标模型名；需与 `--target-model` 匹配才会进入训练 |
| `completedAt` | 完成时间，用于样本排序与 holdout 切分 |
| `ttftMillis` | TTFT 标签；缺失或 `<= 0` 时无法生成 TTFT 样本 |
| `totalLatencyMillis` + `outputLength` | 与 `ttftMillis` 一起推导 TPOT 标签 |
| `predictionInputs` | 按角色（`leader` / `prefill` / `decode`）存放端点特征；空或缺角色会导致对应 slot 无样本 |

`predictionInputs.<role>` 中的特征列（如 `kvCacheUsage`、`numRequestWaiting`）会进入模型输入；`podType` 必须是 `aggregate`、`prefill` 或 `decode`。

输入数据的质量会直接决定最终 bundle 的质量：

- 无法解析的记录会在加载阶段被丢弃；
- 在样本生成阶段，若记录缺少所需的 label 或角色相关的预测输入，也会被记入 `dropReasons`；
- `report.json` 汇总的是加载阶段与样本生成阶段合并后的 `dropReasons`，并不等同于严格意义上的“丢弃记录总数”；
- 可用样本越少，能达到 promotion 阈值的 slot 也就越少。

## 完整训练命令

建议在 sidecar 目录下运行训练命令，以便复用项目的脚本入口。先确认 Router 已开启 persistence，并把 `--input` 指向其 `outputPath`：

```bash
cd sidecar/prediction
uv run prediction-train \
  --input /tmp/hermes-inflight/completed.jsonl \
  --artifact-root /tmp/hermes-prediction-artifacts \
  --target-model 'Qwen/Qwen3-8B' \
  --model-version baseline-conversation-data \
  --model-backend xgboost \
  --quantile 0.9 \
  --min-samples-per-slot 1000 \
  --min-holdout-samples-per-slot 200 \
  --holdout-ratio 0.2 \
  --bucket-cap 500
```

其中 `--hyperparams` 必须以 JSON object string 形式传入，例如 `'{"learning_rate": 0.05}'`。

最关键的参数可归为三类：

- **bundle 标识**：`--artifact-root`、`--target-model`、`--model-version`
- **后端行为**：`--model-backend`、`--quantile`、`--hyperparams`
- **slot 可晋升性**：`--min-samples-per-slot`、`--min-holdout-samples-per-slot`、`--holdout-ratio`、`--bucket-cap`

## 制品格式与发布方式

发布后的 bundle 位于 `artifact_root/target_model/model_version`。以上文的示例命令为例，最终目录结构为：

```text
/tmp/hermes-prediction-artifacts/
└── Qwen/
    └── Qwen3-8B/
        └── baseline-conversation-data/
            ├── manifest.json
            ├── metadata.json
            ├── report.json
            └── slots/
```

各文件职责：

- **`manifest.json`** — 运行时标识、后端、promoted slot 名称，以及 slot 文件映射。
- **`metadata.json`** — 特征列表、promotion 阈值、holdout 配置，以及每个 slot 的状态。
- **`report.json`** — promoted slot 的指标，以及加载阶段与样本生成阶段合并后的 `dropReasons`。
- **`slots/`** — 具体的后端模型文件，例如 `*.xgboost.json` 或 `*.lightgbm.txt`。

发布过程刻意设计成两阶段：trainer 先写入 staging 目录；替换已有 bundle 时，先把旧 bundle 移到一旁，再将 staging 重命名到目标位置，最后删除备份。这样做是为了避免 sidecar 在更新过程中读到只写了一半的 live bundle——而**不是**声称从旧版本到新版本是一次原子切换。

## Slot 晋升规则

trainer 不会把每一个概念上的 slot 都导出为可用模型。

只有同时满足以下两个条件，slot 才会被 promote：

- 该 slot 的总样本数不低于 `minSamplesPerSlot`；
- 该 slot 的 holdout 样本数不低于 `minHoldoutSamplesPerSlot`。

未达到任一阈值的 slot 仍会保留在 `metadata.json` 中，并标记 `status: "below_threshold"` 与 `failingThreshold`，但不会被写入 `manifest.json` 的 `promotedSlots`。

## 运行时加载条件

sidecar 启动时，只有同时满足以下条件，bundle 才是可加载的：

- 启动时传入了 `--artifact-root`、`--target-model`、`--model-version`；
- bundle 位于 `artifactRoot/targetModel/modelVersion`；
- `manifest.json` 与 `metadata.json` 都存在，且均为合法 JSON；
- `manifest.json` 中的 `targetModel` 与 `--target-model` 一致；
- `manifest.json` 中的 `modelVersion` 与 `--model-version` 一致；
- `manifest.json` 与 `metadata.json` 都声明了 `backend`，且两边取值完全一致；
- `backend` 取值只能是 `xgboost` 或 `lightgbm`；
- `promotedSlots` 中的每个 slot 都能在 `slotFiles` 中找到映射；
- 每个映射出的 slot 文件都真实存在于磁盘。

只要任一检查失败，sidecar 会以 prediction not-ready 状态启动，对预测请求返回 failed-precondition 错误，而不会提供基于制品的预测。

## 与 Hermes Router 的边界

bundle 可加载，是 prediction-guided routing 的**必要**条件，但并不**充分**。

只有同时满足以下条件，Hermes Router 才会真正使用预测输出：

- 请求携带非空的 `target_model`；
- 请求中的 `target_model` 与当前已加载 bundle 的目标模型一致；
- 特征提取器能够构造出完整的端点输入；
- 当前 Pod 类型与 queue family 所需的完整 slot 组合确实就绪：aggregate 需要对应 queue family 的 TTFT 与 TPOT 两个 slot；prefill 需要对应的 disagg TTFT slot；decode 需要对应的 disagg TPOT slot；
- scorer 收到的输出通过 schema 校验，且能映射回已收集的输入。

只要其中任意一项不成立，预测评分器就会 fail open 到本地的 snapshot scoring，而不会直接中断路由。

## FAQ

### 为什么跑完 `prediction-train`，却没有可用模型？

命令成功退出，只说明训练流程跑完了；**只有样本量达标的 slot 才会真正产出模型**。每个 slot 需同时满足：

- 总样本数 ≥ `--min-samples-per-slot`（默认 `1000`）
- holdout 样本数 ≥ `--min-holdout-samples-per-slot`（默认 `200`）

未达标的 slot 不会写入 `slots/`，只会在 `metadata.json` 里标记为 `below_threshold`，并用 `failingThreshold` 标明卡在哪一项。

可以这样排查：

1. 打开 `metadata.json`，看各 slot 的 `samples`、`holdoutSamples`、`failingThreshold`。
2. 打开 `report.json` 的 `dropReasons`：解析失败或标签不完整时，有效样本会远少于 JSONL 行数。
3. 确认 `--target-model` 与 JSONL 里的 `model` 一致，否则对应样本会被直接过滤。
4. 本地冒烟可临时调低上述阈值；正式训练仍建议按目标流量规模保留足够样本。

### 为什么采集到的记录里 `predictionInputs` 是空的？

`predictionInputs` 是请求调度时组装好、随 completed 记录一起落盘的，**不是** persistence 事后补出来的。任一必需信号缺失时，该角色就不会写入 `predictionInputs`；同条记录里通常能看到 `missingFeatureReasons`。

常见原因：

| `missingFeatureReasons` | 含义 |
| --- | --- |
| `missing endpoint address` | 端点地址为空 |
| `missing endpoint metrics` | 缺少 backend metrics |
| `missing inflight load` | 缺少 inflight 负载 |
| `missing npu state` | 尚未采到 NPU 状态 |
| `stale npu state` | NPU 数据已超过 `npuExporter.staleAfter`（示例 `5s`）而过期 |

优先核对：npu-exporter 是否正常、label/port/path 是否匹配，以及 scrape 间隔是否明显短于 `staleAfter`。所有角色都失败时，JSONL 里可能根本没有 `predictionInputs` 字段，此时以同条记录的 `missingFeatureReasons` 为准。

补充两点：

- 请求若没有 tokenized prompt，特征提取会整单跳过：`predictionInputs` 为空，且不一定带 `missingFeatureReasons`。
- `prefixCacheScore` 拿不到时通常记为 `0`。若该字段长期全是 `0`，应检查 cache indexer / prefix cache，否则会影响训练特征质量。

`predictionInputs` 为空时，训练侧会大量丢弃样本，最终也容易表现为“训不出模型”。

