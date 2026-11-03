# Prediction Model 示例

本目录提供一份可直接挂载的 **prediction artifact bundle**，供 [Hermes Prediction Sidecar](../../sidecar/prediction/README.md) 加载。将制品放到 `artifactRoot/<targetModel>/<modelVersion>/`，并在推理扩展配置中填写匹配的标识即可启用预测。

## 快速开始

1. 将本目录下的 `Qwen/` 复制到运行 EPP 的节点本地目录，例如 `/path/to/prediction-models`，最终路径应为：

   ```text
   /path/to/prediction-models/Qwen/Qwen3-32B/aggregate-Qwen-Qwen3-32B/
   ```

2. 在 Helm values 中启用 prediction，并把 `modelVolume.hostPath.path` 设为你在节点上存放制品的目录：

```yaml
inferenceExtension:
  prediction:
    enabled: true
    targetModel: Qwen/Qwen3-32B
    modelVersion: aggregate-Qwen-Qwen3-32B
    modelVolume:
      hostPath:
        path: /path/to/prediction-models
        type: Directory
```

Chart helper 会把上述字段转换为 sidecar 的 `--artifact-root`、`--target-model`、`--model-version`，并将卷挂载到 pod 内的固定制品路径。评分器需通过 Unix socket 连接 sidecar，例如 `unix:///var/run/hermes/prediction.sock`（sidecar 默认监听 `/var/run/hermes/prediction.sock`）。

## 制品说明

路径 `Qwen/Qwen3-32B/aggregate-Qwen-Qwen3-32B/` 包含以下 bundle：

| 字段            | 值                         |
| --------------- | -------------------------- |
| `targetModel`   | `Qwen/Qwen3-32B`           |
| `modelVersion`  | `aggregate-Qwen-Qwen3-32B` |
| `backend`       | `xgboost`                  |
| `bundleVersion` | `2`                        |

已 promoted 的 slot：

| Slot              | 用途                          |
| ----------------- | ----------------------------- |
| `aggregate_ttft`  | aggregate 场景的首 token 时延 |
| `aggregate_tpot`  | aggregate 场景的逐 token 时延 |

排队相关特征（如 `numRequestWaiting`、`isQueued`）已内化为模型输入，因此同一份 bundle 同时覆盖有排队与无排队负载。

## 适用范围

**适用于**

- 目标模型为 `Qwen/Qwen3-32B` 的请求
- **aggregate** 部署模式（需要 TTFT + TPOT 两个 slot）

**不适用于**

- disaggregated prefill / decode（本 bundle 未包含 `disagg_ttft`、`disagg_tpot` slot）

bundle 成功加载只表示 sidecar 可以对外提供预测；Router 是否采用预测结果还取决于请求特征、slot 组合与输出校验等运行时条件。预测不可用时会 **fail-open** 回退到基于快照的评分，而不会中断路由。完整加载条件与 Router 使用边界见 [sidecar/prediction/README.md](../../sidecar/prediction/README.md#运行时加载条件)。

## 目录结构

```text
examples/prediction-model/Qwen/Qwen3-32B/aggregate-Qwen-Qwen3-32B/
├── manifest.json      # 运行时加载清单（标识、backend、slot 映射）
├── metadata.json      # 特征列、阈值与 slot 状态
├── report.json        # 训练指标与 dropReasons 汇总
└── slots/
    ├── aggregate_ttft.xgboost.json
    └── aggregate_tpot.xgboost.json
```

## 进一步了解

- **训练自己的 bundle** — 命令、输入格式、晋升阈值与发布流程：[sidecar/prediction/README.md](../../sidecar/prediction/README.md)
- **制品格式与文件职责** — 同上文档的「制品格式与发布方式」一节
- **训练指标** — 见本目录下 `report.json`
