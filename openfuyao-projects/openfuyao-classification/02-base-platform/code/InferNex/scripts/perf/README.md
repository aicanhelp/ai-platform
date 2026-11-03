# InferNex 自动化性能测试工具使用说明

`scripts/perf` 提供一套围绕 **`aiperf`** 的自动化压测脚本，用于在 Kubernetes 环境中完成 InferNex 的部署重建、就绪等待、指标采集、压测执行与结果归档。

**压测工具说明：** 本仓库通过开源工具 [**aiperf**](https://github.com/ai-dynamo/aiperf) 发起压测。`aiperf` 的安装步骤、`aiperf profile` 完整参数、数据集格式和端点类型，请以其官方文档为准。本文档仅说明本目录脚本的编排方式与配置项；如需新增压测参数，请在 `auto_aiperf.conf` 的 `AIPERF_CMD` 中按需调整。

本目录的核心入口文件如下：

- `auto_aiperf.sh`
- `auto_aiperf.conf`

**结果数据位置：** 默认归档到 `${PROJECT_DIR}/artifacts`（可在 `auto_aiperf.conf` 中通过 `ARTIFACTS_DIR` 调整）。

## 功能概览

`auto_aiperf.sh` 会循环执行以下流程：

1. 检查是否已有 `aiperf profile` 进程在运行。
2. 卸载并重新安装目标 Helm release。
3. 等待命名空间内 Pod 全部达到 `Running + Ready`。
4. 后台启动 `collect_kv_transfer_metrics.py` 采集 decode/prefill 日志。
5. 执行一次 `aiperf profile`。
6. 等待采集脚本根据空闲超时自动退出。
7. 归档本轮压测产物。
8. 休眠后进入下一轮。

这套流程适合做长时间稳定性压测、回归压测，以及带自动归档的持续实验。

## 依赖要求

运行前请先确认下面这些依赖已经就绪：

- `kubectl`
  用于查询和检查 Pod、Service 状态。
- `helm`
  用于安装和卸载目标 release。
- `python3`
  用于运行指标采集脚本。
- [**aiperf**](https://github.com/ai-dynamo/aiperf)
  用于发起实际压测；安装方式与 CLI 参数请参考上游仓库文档。
- 可访问的 Kubernetes 集群，并且对目标命名空间具备足够权限。

运行前还需确认以下检查项：

- `collect_kv_transfer_metrics.py` 文件存在，且可由 `python3` 正常执行；
- `AIPERF_CMD` 中引用的数据文件（如 trace）路径在 `PROJECT_DIR` 下有效；
- `GATEWAY_SERVICE_NAME` 和 `GATEWAY_SERVICE_PORT` 对应实际可访问的压测入口。

## 快速开始

### 1. 检查配置

先打开 `auto_aiperf.conf`，至少确认以下配置正确：

- `NAMESPACE`
- `RELEASE_NAME`
- `CHART_DIR`
- `PROJECT_DIR`
- `AIPERF_CMD`

### 2. 直接运行

在仓库根目录执行：

```bash
cd scripts/perf
bash auto_aiperf.sh
```

### 3. 使用自定义配置

可以通过命令行传入配置文件：

```bash
bash auto_aiperf.sh /path/to/your_auto_aiperf.conf
```

也可以通过环境变量指定：

```bash
AUTO_AIPERF_CONFIG=/path/to/your_auto_aiperf.conf bash auto_aiperf.sh
```

配置优先级如下：

1. 命令行传入的配置文件路径
2. 环境变量 `AUTO_AIPERF_CONFIG`
3. 同目录默认配置 `auto_aiperf.conf`

## 关键配置说明

所有运行参数都集中在 `auto_aiperf.conf` 中。通常只需要优先关注下面几项。

- **`NAMESPACE`**：目标 Kubernetes 命名空间，`kubectl`、`helm`、Pod 就绪检查和安装均在该命名空间中。
- **`RELEASE_NAME`**：目标 Helm release 名称，脚本会对这个 release 执行卸载和安装。
- **`CHART_DIR`**：Helm chart 的本地路径，脚本通过它执行 `helm install "${RELEASE_NAME}" "${CHART_DIR}"`。
- **`PROJECT_DIR`**：项目根目录；如果仓库布局变化或脚本路径变化，需要显式改成真实项目根。
- **`ARTIFACTS_DIR`**：每轮压测结果的归档根目录，默认 `${PROJECT_DIR}/artifacts`。
- **`BASE_RESULT_DIR_NAME`**：`aiperf` 输出结果目录名。脚本会在 `ARTIFACTS_DIR` 下查找该目录并在每轮结束后加时间戳归档。该值必须与 `aiperf` 实际输出目录一致。
- **`AIPERF_CMD`**：`aiperf profile` 的命令主体，例如：

```bash
AIPERF_CMD=(
  aiperf profile
  --input-file traces/xxx.jsonl
  --custom-dataset-type mooncake_trace
  ...
)
```

  使用说明：

  - 不要使用单行字符串；
  - 不要在这里手动加 `--url`，脚本会在运行时自动获取网关 `url`；
  - `--input-file` 等路径相对于 `PROJECT_DIR` 解析。

## 日志采集说明

### 功能

- 在压测期间持续采集 `decode/prefill` Pod 日志
- 统计 decode 的 KV 传输耗时
- 统计 vLLM 引擎状态（如 `GPU KV cache usage`、吞吐、队列等）
- 实时跟踪目标 Pod 日志
- 长时间压测场景下持续采集，支持空闲超时自动退出（默认 10 秒）
- Pod 重启时自动尝试拉取 `--previous` 日志并落盘，便于排错

### 输出位置

- **终端实时输出**：采集进度和最终统计结果
- **本地文件输出**：重启/崩溃相关日志保存到 `--output-dir`，默认 `./decode_crash_logs`

### 输出内容

- **Cost 分布（decode）**
  - 每个 decode Pod 的 `Total / Min / Max / Avg / P10~P99`
  - 多个 decode Pod 时，额外输出整体聚合分布
- **引擎状态统计（decode + prefill）**
  - 每个 Pod 的关键指标 `Min / Max / Avg / Last`（如 KV cache、吞吐、Running/Waiting）
- **重启事件信息**
  - 事件时间、Pod 名、崩溃日志文件路径、崩溃前日志片段

## 常见问题

### 终端长时间无输出或等待 Pod 就绪

多数情况不是卡死，而是在等待 Pod Ready。可以执行：

```bash
kubectl -n <NAMESPACE> get pods -w
```

如果看到 `PodInitializing`、`0/1`、`ContainerCreating` 等状态，说明仍在启动中。

### `bad substitution` 或数组相关报错

重点检查 `AIPERF_CMD` 是否仍然是 bash 数组格式，并确认你是用 `bash auto_aiperf.sh` 执行，而不是 `sh auto_aiperf.sh`。

### 结果目录与归档配置不一致

优先检查：

- `BASE_RESULT_DIR_NAME`
- `ARTIFACTS_DIR`
- `aiperf` 实际输出目录名

这三者必须一致或可正确对应。

## 相关文件

- `auto_aiperf.sh`
  主入口脚本，负责调度整轮流程。
- `auto_aiperf.conf`
  配置文件，定义部署、路径、轮询与 `aiperf` 参数。
- `collect_kv_transfer_metrics.py`
  采集 decode/prefill 日志，输出 KV 传输 cost 与引擎状态统计。
