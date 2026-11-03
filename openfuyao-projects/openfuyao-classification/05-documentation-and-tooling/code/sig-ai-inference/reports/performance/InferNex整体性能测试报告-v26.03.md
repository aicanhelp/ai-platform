# InferNex整体性能测试报告-v26.03

## 测试场景与数据集

- **工具&智能体场景**：该场景中大语言模型（LLM）部署为工具或智能体来执行任务，近年来持续增多。这类任务的典型特点是，会采用预先设计、通常篇幅较长且高度重复的系统提示词。我们使用公开的[Mooncake toolagent trace](https://github.com/kvcache-ai/Mooncake/blob/main/FAST25-release/traces/toolagent_trace.jsonl)数据集进行验证，共有23608条请求、理论前缀命中率约 63%，输入长度均值约 8596 token，输出均值约 182 token，测试时按真实时间戳发送请求，聚合测试启用2.5倍加速，实际并发约为 **14.46条/秒**；PD 分离测试未启用加速，实际并发约为 **5.78条/秒**。
- **多轮对话系统提示词复用场景**：该场景中LLM会与多位用户持续对话。用户的单个会话中包含多次请求，每个用户预先定义一段专属长文本规则（如编码规范、项目信息），且不同用户的规则互不相同。数据集包含32个独立用户，共有960条请求，系统提示词长度为4096token，个性化问题长度为128token，**并发为16条/秒。**

## 实验环境

### 硬件集群
- **NPU**: 昇腾 910B4 32G × 16 卡
- **拓扑**: 2台昇腾Atlas A2，每节点 8 卡

### 软件版本
- **推理引擎**: vLLM 0.13.0
- **InferNex**: 0.22.2

### 模型配置 

模型使用`Qwen/Qwen3-8B`，关键参数如下表：

| 参数 | 配置值 | 说明 |
|:---|:---|:---|
| `maxModelLen` | 16,384 | 模型最大上下文长度 |
| `maxNumBatchedTokens` | 40,960 | 批处理 token 上限 |
| `gpuMemoryUtilization` | 0.8 | 显存利用率上限 |

## InferNex测试与部署规格

| 规格      | 描述                 | 架构   | hermes-router | 路由策略        | cache-indexer | mooncake-store | elastic-scaler | eagle-eye |
| --------- | -------------------- | ------ | ------------- | --------------- | ------------- | -------- | -------------- | --------- |
| 聚合-基线 | 不开启性能优化       | 聚合   | ✅| random          | ❌| ❌| ❌| ❌|
| 聚合-kv    | 开启kvcache感知能力  | 聚合   | ✅| kv-cache-aware  | ✅| ❌| ❌| ❌|
| 聚合-m-kv    | 增加多级kvcache管理能力  | 聚合   | ✅| kv-cache-aware  | ✅| ✅| ❌| ❌|
| PD-基线  | 不开启性能优化       | PD分离 | ✅| random          | ❌| ❌| ❌| ❌|
| PD-kv      | 开启kvcache感知能力  | PD分离 | ✅| kv-cache-aware  | ✅| ❌| ❌| ❌|
| PD-m-kv      | 增加多级kvcache管理能力  | PD分离 | ✅| kv-cache-aware  | ✅| ✅| ❌| ❌|

InferNex测试规格如上表所示，**部署规格**如下：
- 聚合模式部署规格：共4个推理实例，单个实例TP=4，DP=1，PP=1，共使用2台物理机16卡。
- PD分离模式部署规格：共4个推理实例，每个推理实例跨机部署1个Prefill节点与1个Decode节点，并行策略均为TP=2，DP=1，PP=1，4实例共使用2台物理机16卡。

> tips： PD分离架构（1P1D）与聚合单实例的部署规格均为4卡。但在PD分离架构中，Prefill与Decode 节点各占2卡，实际可用于存储KV Cache的HBM资源约为聚合模式的一半，这使得PD分离场景下更容易触发KV Cache换入换出，从而影响推理性能，因此横向对比二者时性能存在一定差异。

## 工具&智能体场景

### 聚合架构性能结果

| 测试规格           |请求成功率| TTFT(avg) | TTFT(P99) | TPOT(avg) | TPOT(P99) | TPS | E2EL(avg) | E2EL(P99) |
| -------------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| 聚合-基线          | ~99.9% | 7.90s        | 42.16s       | 135ms      | 632 ms      | 81232.52tok/s | 24.92s       |  125.44s      |
| 聚合-kv          | ~99.9% | 4.95s       | 15.01s       | 209ms    | 1590 ms     | 82453.02tok/s | 22.64s       | 114.57s      |
| 聚合-m-kv | ~99.9% | 9.59s        | 25.10s       | 195ms    | 1434 ms      | 81846.52tok/s | 28.20s       | 127.33s      |

聚合架构使用KVCache Aware路由策略时，InferNex在E2E延迟、TTFT两个指标上都有性能提升：
- E2E延迟：平均耗时22.64s（对比基线降低9%），p99耗时114.57s（对比基线降低8%）；
- TTFT：平均耗时4.95s（对比基线降低37%），p99耗时15.01s（对比基线降低64%）；
- TPOT：平均耗时209ms，p99耗时1590ms，对比基线明显上升，分析发现在工具&智能体场景中，聚合架构下使用KVCache Aware策略时后端实例的请求数分布往往不均，分配到更多请求数的实例的Decode负载高，导致TPOT耗时整体增长，且尾部耗时明显增加。
- TPS：对比基线基本持平（提升1.5%），整体吞吐能力保持稳定。

进一步开启mooncake后发现性能对比`聚合-kv`有一定劣化，分析发现每次请求都会增加一次KVCache查询，且当前cache-indexer组件仅支持感知HBM中的KVCache，无法充分复用mooncake管理的内存中的KVCache。


<!-- ![ttft_percentiles_bars_with_lines](https://hackmd.io/_uploads/r1-2Mnssbg.png) -->
<!-- ![ttft_and_request_latency_percentiles_bars_with_lines](https://hackmd.io/_uploads/S1KFYcNh-e.png) -->
![ttft_and_tpot_InferNex_Mooncake_Toolagent_Agg_Qwen3_8B](https://hackmd.io/_uploads/BJfenQ5nbx.png)


### PD分离架构性能结果

| 测试规格           | 成功率    | TTFT(avg) | TTFT(P99) | TPOT(avg) | TPOT(P99) | TPS                 | E2EL(avg) | E2EL(P99) |
| -------------- | ------ | --------- | --------- | --------- | --------- | ------------------- | --------- | --------- |
| PD-基线          | 99.96% | 12.65s    | 204.34s   | 24ms      | 41ms      | 33625.31 tokens/sec | 16.37s    | 207.20s   |
| PD-kv          | 100%   | 9.66s     | 49.44s    | 25ms      | 41ms      | 33640.16 tokens/sec | 13.30s    | 65.09s    |
| PD-m-kv | 100%   | 9.83s     | 64.23s    | 24ms      | 38ms      | 33634.26 tokens/sec | 13.46s    | 72.46s    |



在PD分离架构下使用KVCache Aware路由策略时，InferNex的性能同样有所优化：
- E2E延迟：平均耗时13.30s（对比基线降低19%），p99耗时65.09s（对比基线降低68%）；
- TTFT：平均耗时9.66s（对比基线降低24%），p99耗时49.44s（对比基线降低75%）；
- TPOT：平均耗时25ms，p99耗时41ms,对比基线基本持平；
- TPS：对比基线提升0.04%，基本持平。

进一步开启mooncake后发现性能与`PD-kv`基本持平，平均TTFT上升1.7%，TPOT降低4%，E2EL上升1.2%，TPS表现基本相同。

<!-- ![ttft_percentiles_pd_disaggregation_bars_with_lines](https://hackmd.io/_uploads/rkIazhisZg.png) -->
<!-- ![ttft_and_request_latency_pd_disaggregation_bars_with_lines](https://hackmd.io/_uploads/SylPP943be.png) -->

![ttft_and_tpot_pd_disaggregation_bars_with_lines](https://hackmd.io/_uploads/HycZnX53bx.png)


## 多轮对话系统提示词复用场景

### 聚合架构性能结果

| 测试规格  | 成功率  | TTFT(avg) | TTFT(P99) | TPOT(avg) | TPOT(P99) | TPS                | E2EL(avg) | E2EL(P99) |
| ----- | ---- | --------- | --------- | --------- | --------- | ------------------ | --------- | --------- |
| 聚合-随机 | 100% | 0.39s     | 5.01s     | 20ms      | 33ms      | 69030.1 tokens/sec | 3.61s     | 9.35s     |
| 聚合-kv | 100% | 0.20s     | 0.37s     | 18ms      | 24ms      | 66743.9 tokens/sec | 3.01s     | 4.01s     |



多轮对话场景中，聚合架构下使用KVCache Aware路由策略时InferNex提升明显：
- E2E延迟：平均耗时3.01s（对比基线降低约17%），p99耗时4.01s（对比基线降低57%）;
- TTFT：平均耗时0.21s（对比基线降低46%），p99耗时0.37s（对比基线降低93%）；
- TPOT：平均耗时18ms（对比基线降低10%），p99耗时24ms（对比基线降低27%）；
- TPS：对比基线下降约3.3%，基本持平。

<!-- ![ttft_and_latency_percentiles_from_benchmark_logs](https://hackmd.io/_uploads/H12JY6Q2-e.png) -->
![ttft_and_tpot_InferNex_Multi_Turn_Agg_Qwen3_8B](https://hackmd.io/_uploads/B1k1TQ53Zl.png)


### PD分离架构性能结果

| 测试规格      | 成功率  | TTFT(avg) | TTFT(P99) | TPOT(avg) | TPOT(P99) | TPS                | E2EL(avg) | E2EL(P99) |
| --------- | ---- | --------- | --------- | --------- | --------- | ------------------ | --------- | --------- |
| pd-random | 100% | 11.12s    | 80.86s    | 24ms      | 43ms      | 19332.9 tokens/sec | 15.02s    | 83.39s    |
| pd-kv     | 100% | 8.11s     | 79.49s    | 24ms      | 32ms      | 24095.1 tokens/sec | 11.99s    | 82.51s    |


PD分离架构下InferNex使用KVCache Aware路由策略的表现如下：
- E2E延迟：平均耗时 11.99s（对比基线降低约20%），p99 耗时82.51s（对比基线降低1%）;
- TTFT：平均耗时 8.11s（对比基线降低27%），p99 耗时79.49s（对比基线降低2%）；
- TPOT：平均耗时24ms（对比基线持平），p99耗时32ms（对比基线降低约26%）；
- TPS：对比基线提升约25%。

<!-- ![ttft_and_latency_pd_disaggregation_from_benchmark_logs](https://hackmd.io/_uploads/SJObt6mhWx.png) -->
![ttft_and_tpot_InferNex_Multi_Turn_PD_Qwen3_8B](https://hackmd.io/_uploads/BJX-amc3bx.png)

