
# 在离线混部

在离线混部是openFuyao提供的一套完整的云原生混合部署和资源超卖解决方案，通过统一的混部管理平台和用户友好的可视化界面，支持在线/离线业务混合部署，保障在线业务在使用高峰时期的调度，同时使能离线业务在在线业务低谷时期使用超卖资源，大幅提升集群的资源利用率和业务部署密度。本仓库是openFuyao在离线混部后端组件，对外提供sevice服务接口，包括混部监控信息接口，混部节点管理的添加、移除，混部调度策略配置。

## 特性介绍

### 核心能力

在离线混部提供基础混部能力和高级混部特性：

**基础混部能力**：
- **业务多QoS分级管理**：支持HLS（高延迟敏感）、LS（延迟敏感）、BE（尽力而为）三级QoS分级。
- **混部场景调度增强**：保证离线业务调度到使用超卖资源的节点上。
- **基于rubik的节点混部引擎**：支持配置水位线驱逐功能。
- **统一配置管理**：使用统一的configmap进行混部配置管理（节点管理、volcano、rubik配置管理）。
- **超卖资源管理**：超卖agent部署范围与超卖资源计算修正。
- **混部基础监控**：提供混部基础监控能力。

**高级混部特性**：
- **基于NUMA拓扑感知的调度增强**：支持NUMA亲和性调度优化。
- **基于NRI机制的无侵入混部**：利用containerd NRI机制实现容器资源管理。
- **CPU弹性限流**：支持CPU资源动态调整。
- **内存异步回收**：基于QoS级别的内存回收机制。
- **访存带宽限制**：限制BE级别Pod对内存带宽和CPU缓存的占用。
- **PSI干扰检测**：基于系统压力指标的干扰检测和处理。

### 性能提升效果

基于实际测试验证的性能数据：

- **资源利用率提升**：测试场景下，资源利用率提高30%-50%，QPS下降不超过5%。
- **NUMA亲和优化**：开启NUMA亲和的NRI插件后，测试场景下平均时延降低40%+，吞吐量提高70%+。
- **混部稳定性**：通过QoS分级和水位线驱逐机制，确保在线业务性能稳定。

### 能力范围

本特性由以下核心组件（代码仓）组成，提供完整的混部解决方案：

+ **colocation-management**：混部工作负载准入控制和超卖资源管理。
+ **colocation-service**：混部监控信息接口和混部管理后端服务。
+ **colocation-website**：混部统计可视化和配置管理前端界面。

主要功能包括：
+ 支持对不同QoS分级业务的优先级调度和负载均衡调度。
+ 支持单机上在线业务对离线业务的cpu和memory的QoS压制。
+ 支持单机上基于cpu/memory的水位线对离线业务进行驱逐重调度。
+ 支持CPU弹性限流、内存异步回收、访存带宽限制和PSI干扰检测等高级混部特性。
+ 提供集群级和节点级混部资源监控和可视化展示。
+ 支持通过Web界面查看节点混部能力、配置混部策略参数。
+ 提供用户友好的方式查看和开启/关闭高级混部功能。
+ 支持混部节点一键开启/关闭，实时查看操作结果。

## 安装说明

### 前提条件

1. **系统环境要求**
   - Kubernetes v1.21及以上版本。
   - containerd v1.7.0及以上版本。
   - kube-prometheus v1.19及以上版本。
   - volcano v1.9.0及以上版本。

2. **安装volcano调度器**
   
   ```shell
   helm repo add volcano-sh https://volcano-sh.github.io/helm-charts
   helm repo update
   helm install volcano volcano-sh/volcano --version 1.9.0 -n volcano-system --create-namespace
   ```

3. **配置kubelet**（如需支持HLS级别Pod绑核）
   - 配置CPU管理策略为static。
   - 开启NUMA亲和策略。

4. **配置containerd NRI功能**
   编辑`/etc/containerd/config.toml`添加NRI配置：
   ```toml
   [plugins."io.containerd.nri.v1.nri"]
   disable = false
   disable_connections = false
   plugin_config_path="/etc/nri/conf.d"
   plugin_path="/opt/nri/plugins"
   ```

### 开始安装

#### 通过openFuyao平台安装

1. 登录openFuyao平台，选择“应用市场 > 应用列表”。
2. 搜索“colocation-package”扩展组件。
3. 点击“部署”，配置应用名称、版本和命名空间。
4. 在Values.yaml中配置部署参数。
5. 完成部署后在“扩展组件管理”中管理该组件。

#### 独立部署

1. 拉取helm包：
   
   ```shell
   helm pull oci://cr.openfuyao.cn/charts/colocation-package --version xxx
   ```

2. 解压并修改配置：
   
   ```shell
   tar -zxvf colocation-package-xxx.tgz
   vim colocation-package/values.yaml
   ```

3. 安装：
   
   ```shell
   helm install colocation-package ./
   ```

### openFuyao平台使用

部署完成后，可通过openFuyao平台的“算力优化中心 > 在离线混部”访问以下功能：

- **概览**：查看在离线混部工作流程。
- **混部策略配置**：管理混部节点和配置调度策略参数。
- **混部监控**：查看集群级和节点级混部监控数据。

## 本地构建

### 镜像构建

#### 构建参数

- `GOPRIVATE`：配置Go语言私有仓库，相当于`GOPRIVATE`环境变量。
- `COMMIT`：当前git commit的哈希值。
- `VERSION`：组件版本。
- `SOURCE_DATE_EPOCH`：镜像rootfs的时间戳。

#### 构建命令

- 构建并推送到指定OCI仓库。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=image,name=<oci/repository>:<tag>,oci-mediatypes=true,rewrite-timestamp=true,push=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
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
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag。

- 构建并导出OCI Layout到本地tarball。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=oci,name=<oci/repository>:<tag>,dest=<path/to/oci-layout.tar>,rewrite-timestamp=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
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
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest \
      --build-arg=SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`<oci/repository>`为镜像地址，`<tag>`为镜像tag，`path/to/oci-layout.tar`为tar包路径。

- 构建并导出镜像rootfs到本地目录。

  <details open>
  <summary>使用<code>docker</code></summary>

  ```bash
  docker buildx build . -f <path/to/dockerfile> \
      -o type=local,dest=<path/to/output>,platform-split=true \
      --platform=linux/amd64,linux/arm64 \
      --provenance=false \
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
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
      --build-arg=GOPRIVATE=gopkg.openfuyao.cn \
      --build-arg=COMMIT=$(git rev-parse HEAD) \
      --build-arg=VERSION=0.0.0-latest
  ```

  </details>

  其中，`<path/to/dockerfile>`为Dockerfile路径`./build/Dockerfile`，`path/to/output`为本地目录路径。

### Helm Chart构建

- 打包Helm Chart。

  ```bash
  helm package <path/to/chart> -u \
      --version=0.0.0-latest \
      --app-version=openFuyao-v25.09
  ```

  其中，`<path/to/chart>`为Chart文件夹路径。

- 推送Chart包到指定OCI仓库。

  ```bash
  helm push <path/to/chart.tgz> oci://<oci/repository>:<tag>
  ```

  其中，`<path/to/chart.tgz>`为Chart包路径，`<oci/repository>`为Chart包推送地址，`<tag>`为Chart包tag。

## 安全能力说明

openFuyao v26.06 及往期版本仅提供基础安全功能：服务间双向认证（mTLS）与组件级用户认证鉴权（RBAC）需使用者/集成伙伴根据自身部署环境自行适配与加固（如引入 cert-manager、做统一证书、部署鉴权中间件、配置 NetworkPolicy 收缩可达范围等）。当前聚焦基础功能交付，安全防护需用户自行加强，后续版本安全能力会逐步规划落地。

## 外部接口审计日志缺失清单

本服务以下外部接口均无法记录操作（审计）日志，请下游开发者在二次开发时自行补充审计日志能力，以支撑问题追溯与安全合规。

- **根因**：`pkg/service/server.go` 的 `init()` 中直接使用 `go-restful` 生成 `WebService` 并暴露端口，整个生命周期未挂载任何请求拦截器（Filter / Middleware）。
- **改造入口**：在 `pkg/service/server.go` 的 `init()` 方法中，通过 `ws.Filter(自定义审计逻辑)` 或 `container.Filter(...)` 统一拦截所有请求并输出日志。

| 接口地址 | 访问方法 | 风险等级 | 说明 |
| :--- | :--- | :--- | :--- |
| `/rest/colocation/v1/colocation-configs/` | GET, POST | 高 | POST 涉及全局混部配置写入；该服务同时缺乏统一鉴权层 |
| `/rest/colocation/v1/colocation-nodes/{name}` | PATCH | 高 | 允许强制切换节点混部状态，需保留操作人审计追踪 |
| `/rest/colocation/v1/colocation-status/` | GET | 低 | 只读接口，合规场景下需记录访问日志以备安全溯源 |
| `/rest/colocation/v1/colocation-metrics/nodes/{node}` | GET | 低 | 只读接口，合规场景下需记录访问日志以备安全溯源 |
| `/rest/colocation/v1/colocation-metrics/cluster` | GET | 低 | 只读接口，合规场景下需记录访问日志以备安全溯源 |