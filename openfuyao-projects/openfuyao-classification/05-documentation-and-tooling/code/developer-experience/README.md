# 🚀 Hermes Router

<div align="center">

**智能AI推理路由引擎 - 让每一次推理请求都找到最佳归宿**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Go Version](https://img.shields.io/badge/go-%3E%3D1.21-blue.svg)](https://go.dev/)

</div>

## 📖 项目简介

Hermes Router 是一个专为 AI 推理场景设计的**智能路由解决方案**，如同希腊神话中的信使赫尔墨斯一般，为您的推理请求找到最快、最优的路径。

### ✨ 核心特性

- 🎯 **智能路由策略**：支持多种路由算法，包括基于 KV Cache 的感知路由
- ⚡ **资源感知调度**：实时监控负载与资源状态，动态优化请求分发
- 🔧 **灵活配置**：插件化架构，支持自定义路由策略
- 🏗️ **多架构支持**：同时支持聚合架构和 Prefill-Decode (PD) 分离架构
- 📊 **性能优化**：专为大模型推理场景优化，提升资源利用率和响应速度

---

## ⚙️ EPP 配置指南

EPP (Endpoint Picker) 配置位于 `/charts/inferencepool/values.yaml` 文件中，配置路径如下：

```yaml
inferenceExtension:
  # ... 其他配置项
  pluginsCustomConfig:
    # 👇 在此处添加 EPP 配置
```

### 📋 路由策略详解

我们提供三种开箱即用的路由策略，满足不同架构需求：

#### 1️⃣ 聚合架构 - KV Cache 感知路由

> 💡 **适用场景**：传统聚合架构，通过 KV Cache 命中率优化性能

**策略特点**：
- 🎯 智能匹配缓存命中率，减少重复计算
- ⚖️ 多维度权重平衡（缓存未命中率、XPU 使用率、等待队列）
- 🔗 与 Cache Indexer 服务集成，实时获取缓存状态

```yaml
epp-aggregate-kv-cache-aware.yaml: |
  apiVersion: inference.networking.x-k8s.io/v1alpha1
  kind: EndpointPickerConfig
  plugins:
  - type: scorer-aggregate-kv-cache-aware
    parameters:
      kvCacheHitNotRateWeight: 1.0
      xpuCacheUsageWeight: 1.0
      waitingRequestWeight: 1.0
      kvCacheManagerIP: cache-indexer-service
      kvCacheManagerPort: 8080
      kvCacheManagerPath: /match_sort
      kvCacheManagerTimeout: 5000000000
  - type: picker-min-random
  schedulingProfiles:
  - name: default
    plugins:
    - pluginRef: scorer-aggregate-kv-cache-aware
    - pluginRef: picker-min-random
```

#### 2️⃣ PD 架构 - KV Cache 感知路由

> 💡 **适用场景**：Prefill-Decode 分离架构，精细化管理预填充和解码阶段

**策略特点**：
- 🔀 自动区分 Prefill 和 Decode 节点
- 🎭 独立评分机制，分别优化两阶段性能
- 🏷️ 基于 Pod Label 智能过滤和分组
- 🔄 自动添加 PD 相关请求头，简化客户端逻辑

```yaml
epp-pd-kv-cache-aware.yaml: |
  apiVersion: inference.networking.x-k8s.io/v1alpha1
  kind: EndpointPickerConfig
  plugins:
  - type: filter-by-pd-label
    parameters:
      pdLabelName: openfuyao.com/pdRole
      pdGroupLabelName: openfuyao.com/pdGroupID
      prefillValue: prefill
      decodeValue: decode
      leaderValue: leader
  - type: scorer-pd-kv-cache-aware
    parameters:
      kvCacheHitNotRateWeight: 1.0
      xpuCacheUsageWeight: 1.0
      prefillWaitingRequestWeight: 1.0
      decodeWaitingRequestWeight: 1.0
      prefillPodScoreWeight: 1.0
      decodePodScoreWeight: 1.0
      kvCacheManagerIP: cache-indexer-service
      kvCacheManagerPort: 8080
      kvCacheManagerPath: /match_sort
      kvCacheManagerTimeout: 5000000000
  - type: picker-pd-kv-cache-aware
  - type: pd-header-handler
  schedulingProfiles:
  - name: default
    plugins:
    - pluginRef: filter-by-pd-label
    - pluginRef: scorer-pd-kv-cache-aware
    - pluginRef: picker-pd-kv-cache-aware
    - pluginRef: pd-header-handler
```

#### 3️⃣ PD 架构 - Bucket 桶策略路由

> 💡 **适用场景**：基于请求长度分桶，优化不同尺寸请求的处理效率

**策略特点**：
- 🪣 按序列长度智能分桶（默认阈值：200 tokens）
- 📉 指数衰减算法，平滑负载分布
- ⚖️ Alpha-Beta 权重调节，灵活控制调度策略
- 🎯 长短请求分流，避免队头阻塞

```yaml
  epp-pd-bucket.yaml: |
    apiVersion: inference.networking.x-k8s.io/v1alpha1
    kind: EndpointPickerConfig
    plugins:
    - type: filter-by-pd-label
      parameters:
        pdLabelName: openfuyao.com/pdRole
        pdGroupLabelName: openfuyao.com/pdGroupID
        prefillValue: prefill
        decodeValue: decode
        leaderValue: leader
    - type: scorer-pd-bucket
      parameters:
        alpha: 0.5
        beta: 0.5
        decayFactor: 0.5
        bucketSeperateLength: 200
    - type: pd-header-handler
    schedulingProfiles:
    - name: default
      plugins:
      - pluginRef: filter-by-pd-label
      - pluginRef: scorer-pd-bucket
        weight: 1
      - pluginRef: pd-header-handler
```

---

## 🛠️ 本地开发指南

### 🐳 镜像构建

#### 📦 构建参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `COMMIT` | Git 提交哈希值，用于版本追踪 | `abc123...` |
| `VERSION` | 组件版本号 | `0.0.0-latest` |
| `SOURCE_DATE_EPOCH` | 镜像 rootfs 时间戳，确保可重现构建 | `1733385600` |

#### 🚀 构建命令

##### 选项 1: 构建并推送至 OCI 镜像仓库

> 💡 **适用场景**：生产环境部署，需要将镜像推送到远程仓库

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  **参数说明**：
  - `<path/to/dockerfile>`：Dockerfile 文件路径
  - `<oci/repository>`：OCI 镜像仓库地址
  - `<tag>`：镜像标签

##### 选项 2: 构建并导出为本地 Tarball

> 💡 **适用场景**：离线环境部署，需要将镜像打包为 tar 文件传输

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  **参数说明**：
  - `<path/to/dockerfile>`：Dockerfile 文件路径
  - `<oci/repository>`：OCI 镜像仓库地址
  - `<tag>`：镜像标签
  - `<path/to/oci-layout.tar>`：输出的 tar 包路径

##### 选项 3: 构建并导出镜像文件系统

> 💡 **适用场景**：需要直接查看或修改镜像文件系统内容

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>
  <details>
  <summary>使用<code>nerdctl</code></summary>

  ```bash
  nerdctl build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  **参数说明**：
  - `<path/to/dockerfile>`：Dockerfile 文件路径
  - `<path/to/output>`：输出目录路径（按平台分离存储）

---

### 📊 Helm Chart 构建

#### 📦 打包 Chart

将 Helm Chart 打包为 `.tgz` 格式：

```bash
helm package <path/to/chart> -u \
    --version=0.0.0-latest \
    --app-version=openFuyao-v25.09
```

**参数说明**：
- `<path/to/chart>`：Chart 文件夹路径
- `--version`：Chart 版本号
- `--app-version`：应用版本号
- `-u`：更新依赖

#### 🚢 推送至 OCI 仓库

将打包好的 Chart 推送到 OCI 镜像仓库：

```bash
helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
```

**参数说明**：
- `<path/to/chart.tgz>`：打包后的 Chart 文件路径
- `<oci/repository>`：OCI 仓库地址
- `<tag>`：Chart 版本标签

---

## 🤝 贡献指南

我们欢迎所有形式的贡献！无论是报告 Bug、提出新功能建议，还是提交代码改进。

## 📄 许可证

本项目采用 Apache 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

---

<div align="center">

**Made with ❤️ by the Hermes Router Team**

如有问题或建议，欢迎提交 [Issue](../../issues) 或 [Pull Request](../../pulls)

</div>