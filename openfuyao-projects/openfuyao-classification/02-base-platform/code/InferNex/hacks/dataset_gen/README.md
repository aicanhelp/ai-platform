# dataset_gen

基于 AIPerf 的 `PromptGenerator`（Shakespeare 语料，内置 o200k_base tokenizer）生成 AIPerf JSONL 格式的测试数据集。需预装 `aiperf` 库。

```bash
pip install aiperf
```

## gen_single_turn.py

生成固定长度单轮数据集：N 个不重复 prompt × 每个重复 R 次，打乱后输出。

```bash
# 默认：120 类型 × 4 次 = 480 条，ISL=8192 tokens
python3 gen_single_turn.py

# 自定义
python3 gen_single_turn.py --num-types 50 --repeat 3 --isl 4096 --output my_data.jsonl --seed 42
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-types` | 模板数 | 120 |
| `--repeat` | 每模板重复次数 | 4 |
| `--isl` | 输入 token 长度 | 8192 |
| `--output` | 输出文件 | `random_dataset.jsonl` |
| `--seed` | 随机种子 | 42 |

### 输入/输出说明

每行 JSON 是一条独立请求，不携带上下文：

```json
{"text": "<固定 ISL tokens 的输入文本>"}
```

**输入（`text`）**：单条输入文本，由 `PromptGenerator` 基于 Shakespeare 语料生成，长度固定为 `--isl` tokens。所有输入互不重复；生成后按 `--repeat` 复制、整体打乱，因此同一条文本可能出现多次（缓存友好，用于模拟重复请求）。

**输出**：脚本不生成回复内容。AIPerf 按 `--custom-dataset-type single_turn` 解析后，由后端模型实时生成回复，输出长度由 profile 时的生成参数（如 `--osl`）决定。

## gen_multi_turn.py

生成固定长度多轮会话数据集：每个 session N 轮，每轮输入固定 ISL tokens、输出上限固定 OSL，输入全局去重。

```bash
# 默认：120 会话 × 4 轮，每轮 ISL=1024 / OSL=1024 tokens
python3 gen_multi_turn.py --output random_1k_mt_s42.jsonl

# 自定义
python3 gen_multi_turn.py --num-sessions 50 --turns 4 --isl 8192 --osl 1024 --output my_data.jsonl --seed 42
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num-sessions` | 会话数 | 120 |
| `--turns` | 每会话轮数 | 4 |
| `--isl` | 每轮输入 token 长度 | 1024 |
| `--osl` | 每轮回复最大 token 数 | 1024 |
| `--output` | 输出文件 | `random_multi_turn_dataset.jsonl` |
| `--seed` | 随机种子 | 42 |

### 输入/输出说明

每行 JSON 表示一个完整会话（session），每个会话内包含 N 轮（turn）对话：

```json
{"session_id": "s42_0000", "turns": [
  {"text": "<第 1 轮新增提问，固定 1024 tokens>", "output_length": 1024},
  {"text": "<第 2 轮新增提问，固定 1024 tokens>", "output_length": 1024},
  {"text": "<第 3 轮新增提问，固定 1024 tokens>", "output_length": 1024},
  {"text": "<第 4 轮新增提问，固定 1024 tokens>", "output_length": 1024}
]}
```

**输入（`text`）**：每轮「新增提问」的文本，由 `PromptGenerator` 基于 Shakespeare 语料生成，每轮长度固定为 `--isl` tokens，不含先前任何输入输出。整个数据集共 `num-sessions × turns` 条输入，全局去重（逐条校验 token 数、丢弃重复），避免请求命中 KV cache 导致性能数据失真。

**输出（`output_length`）**：每轮回复的生成上限，AIPerf 将其转为请求的 `max_tokens`/`max_completion_tokens` 参数，所有轮取同一值 `--osl`。若不设置，回复长度由 profile 时的 `--osl` 等生成参数决定。

**上下文累积**：AIPerf 默认按 `deltas_without_responses` 模式处理多轮——每轮 `text` 只是当轮增量，请求发送时会把历史累积起来。第 i 轮实际发送给模型的完整上下文 ≈ 前 i-1 轮的「用户输入 + 模型回复」+ 当前轮新输入。以默认 4 轮 × 1k、每轮回复 1k 为例，第 4 轮请求的完整上下文约为 4k 输入 + 3k 回复（不含系统提示等固定开销）。

生成的 JSONL 可直接作为 AIPerf 自定义数据集使用，格式与用法见 [Custom Dataset Guide](https://docs.nvidia.com/aiperf/tutorials/datasets-inputs/custom-dataset-guide)。
