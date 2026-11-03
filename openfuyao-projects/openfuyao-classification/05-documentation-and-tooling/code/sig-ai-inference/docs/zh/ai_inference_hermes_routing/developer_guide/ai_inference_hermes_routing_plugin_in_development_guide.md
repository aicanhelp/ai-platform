# AI推理赫尔墨斯路由插件开发指南

## 特性介绍

Hermes-router是一个Kubernetes（K8s）原生的AI推理智能路由方案，用于接收用户推理请求并转发至合适的推理服务后端。

作为GIE框架下的EPP插件，Hermes-router仅关注请求的转发过程，本指南介绍开发者如何基于EPP框架进行自定义路由策略开发。

## 约束与限制

**推理引擎限制**

Hermes-router对推理引擎的支持范围与GIE保持一致。
其中，依赖全局KVCache信息的KVCache aware与时延预测相关能力需要配合cache-indexer使用。当前cache-indexer最新版本为0.22.0，这类能力目前主要在vLLM场景完成验证，因此开发和联调时建议优先选择vLLM作为推理后端。

其他插件在理论上支持多种推理引擎，但目前主要验证了vLLM作为推理引擎的场景。

**开发限制**

在本文档中，一个EPP实例代表一种路由策略，一个路由策略通常由多个类型的插件组成，插件开发需遵循[GIE框架规范](https://gateway-api-inference-extension.sigs.k8s.io/)。

**部署限制**

- 仅支持在Kubernetes环境中部署运行。
- 插件需在本地完成开发，构建为镜像后才可部署运行。
- 如需变更路由策略配置，需在修改配置后重启EPP方可生效。

## 环境准备

### 环境要求

#### 硬件要求

Hermes-router开发环境对硬件无特殊要求，建议按照如下进行配置。

- CPU：4核及以上
- 内存：8GB及以上
- 磁盘：20GB及以上可用空间

#### 软件要求

- 操作系统：Linux
- Go环境：Go 1.25或更高版本
- Docker：Docker 20.10+或兼容的容器运行时（如nerdctl）
- Kubernetes集群：用于部署和测试
- kubectl：用于与K8s集群交互
- Helm：3.0+版本，用于部署Hermes-router
- 如使用基于Gateway的联调方式：需准备支持Gateway API Inference Extension的开源网关，如Istio 1.28及以上版本

#### 依赖组件

- 基础依赖
  - 推理后端服务：集群中需已部署vLLM等推理引擎服务。
  - 如使用基于Gateway的部署或联调方式：需安装Gateway API CRDs和Inference Extension CRDs。
- 额外依赖
  - 若要开发或验证KVCache aware与时延预测相关能力：需部署cache-indexer 0.22.0及以上版本，用于提供全局KVCache感知与命中率计算。

> ![note](../user_guide/figures/icon-note.gif) **说明：**<br>
> 如需快速搭建包含推理后端、智能路由、全局KVCache管理等组件的完整开发联调环境，可通过InferNex进行一键集成部署，具体部署方法请参考[InferNex用户指南](../../ai_inference_infernex/user_guide/ai_inference_infernex.md)。

### 搭建环境

1. 克隆代码仓库。

   ```bash
   git clone https://gitcode.com/openFuyao/hermes-router.git
   cd hermes-router
   ```

2. 配置Go开发环境。

   ```bash
   go version

   # 加速依赖下载（可选）
   go env -w GOPROXY=https://goproxy.cn,direct
   ```

3. 安装基于Gateway联调所需的网关与CRDs（可选）。

   ```bash
   ISTIO_VERSION=1.28.0
   curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -

   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.0/standard-install.yaml

   ./istio-$ISTIO_VERSION/bin/istioctl install \
     --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true \
     --set values.gateways.istio-ingressgateway.type=NodePort
   ```

   如需继续安装Inference Extension CRDs和Gateway入口资源，请参见[安装配套组件](../user_guide/ai_inference_hermes_routing.md#安装配套组件)完成配置。

4. 构建Hermes-router镜像。

   ```bash
   cd hermes-router

   docker build -t hermes-router:dev -f build/Dockerfile .

   # 或使用nerdctl
   nerdctl build -t hermes-router:dev -f build/Dockerfile .
   ```

### 验证环境

1. 验证Gateway联调依赖是否安装成功（可选）。

   ```bash
   kubectl get pods -n istio-system
   kubectl get crd inferencepools.inference.networking.x-k8s.io
   ```

2. 以standalone模式部署Hermes-router进行开发验证。

   ```bash
   cd hermes-router

   # 下载standalone模式所需的Helm子Chart依赖
   helm dependency build ./charts/standalone

   # 部署Hermes-router到hermes-test命名空间
   # --set参数说明：
   #   createInferencePool=false：跳过自动创建InferencePool，由用户手动管理
   #   endpointSelector：指定推理后端Pod的标签选择器，需与实际Pod标签匹配
   #   targetPorts：推理后端服务侦听的端口号
   helm install hermes-router ./charts/standalone \
     --namespace hermes-test --create-namespace \
     -f ./examples/profiles/pd-random.yaml \
     --set inferenceExtension.image.repository=hermes-router \
     --set inferenceExtension.image.tag=dev \
     --set inferenceExtension.endpointsServer.createInferencePool=false \
     --set inferenceExtension.endpointsServer.endpointSelector='openfuyao.com/model=qwen-qwen3-8b' \
     --set inferenceExtension.endpointsServer.targetPorts=8000

   # 检查EPP Pod是否正常运行
   kubectl get pods -n hermes-test -l epp=hermes-router
   # 查看Hermes-router Service，确认入口端口（默认8081）已正常暴露
   kubectl get svc -n hermes-test hermes-router
   ```

**预期结果：**

- EPP Pod状态为Running。
- Service正常创建，并默认暴露HTTP入口端口`8081`。
- 容器日志中无明显报错信息。

## 开发自定义路由策略

### 使用场景概述

当预置的路由策略无法满足特定业务需求时，开发者可以通过实现自定义插件来扩展路由能力。典型场景包括：

- 业务特定路由规则：根据业务标签、用户ID等自定义属性进行路由。
- 性能优化需求：针对特定硬件或网络环境优化路由算法。
- 混合架构支持：支持聚合架构和PD分离架构的混合部署。

### 系统架构

Hermes-router基于GIE的EPP框架实现，当前插件按照职责边界划分为`Scheduling`、`RequestControl`和`Datalayer`三层。分别负责路由决策、请求处理控制和外部数据接入，通过统一的插件注册机制协同完成一次推理请求的处理。

**表1** Hermes-router插件分层说明

| 层级 | 目录 | 功能描述 | 示例 |
|------|------|---------|------|
| Scheduling | `pkg/epp/framework/plugins/scheduling` | 直接参与候选端点的过滤、评分和选择。 | `pd-group-filter`、`kvcache-aware-scorer`、`random-picker` |
| RequestControl | `pkg/epp/framework/plugins/requestcontrol` | 负责请求级数据生产、请求生命周期跟踪以及转发前处理。 | `tokenizer`、`request-lifecycle-tracker`、`pd-header-handler` |
| Datalayer | `pkg/epp/framework/plugins/datalayer` | 从外部系统拉取、抽取并暴露可供调度使用的属性。 | `npu-exporter-data-source`、`npu-exporter-extractor` |

在开发者最常接触的调度链路中，Scheduling层又可继续分为如下三类能力。

**表2** Scheduling层插件类型说明

| 插件类型 | 功能描述 | 示例 |
|---------|---------|------|
| Filter插件 | 过滤候选端点列表。 | `pd-group-filter`：过滤无效PD Group。<br>`prefix-cache-filter`：按照前缀缓存亲和性过滤候选实例。 |
| Scorer插件 | 对候选端点进行评分。 | `kvcache-aware-scorer`：基于KVCache相关指标评分。<br>`pd-bucket-scorer`：基于PD分桶负载评分。<br>`prediction-scorer`：基于预测结果评分。 |
| Picker插件 | 根据评分或策略选择最终端点。 | `random-picker`：随机选择实例。<br>`min-score-picker`：选择最低分实例。<br>`softmax-potc-picker`：按Softmax概率进行选择。 |

RequestControl和Datalayer中的常用插件类型如表3所示。

**表3** 请求控制与数据层插件类型说明

| 插件类型 | 所属层级 | 功能描述 | 示例 |
|---------|---------|---------|------|
| DataProducer插件 | RequestControl | 为请求生成调度所需的上下文数据。 | `pd-group-producer`、`prefix-cache-producer`、`prediction-feature-extractor` |
| Lifecycle插件 | RequestControl | 跟踪请求生命周期，维护在途请求等状态。 | `request-lifecycle-tracker` |
| PreRequest插件 | RequestControl | 在请求转发前补充目标后端所需信息。 | `pd-header-handler` |
| Source/Extractor插件 | Datalayer | 从外部系统采集数据并抽取为路由属性。 | `npu-exporter-data-source`、`npu-exporter-extractor` |

**插件执行流程**

1. **请求进入EPP**：RequestControl层插件根据请求内容生成或补充请求级数据，例如分词结果、PD Group信息、前缀缓存信息和预测特征等。
2. **外部数据接入**：Datalayer层插件按需从外部数据源拉取指标，并将其抽取为可供调度使用的属性。
3. **调度决策**：Scheduling层插件按照`schedulingProfiles`中定义的顺序执行过滤、评分和选择，得到最终目标端点。
4. **转发前处理**：PreRequest类插件在请求转发前写入请求头或路由元信息，然后将请求发送至目标推理后端。

以下链接提供了开发自定义路由策略所需的相关文档：

- [hermes-router代码仓库](https://gitcode.com/openFuyao/hermes-router)：Hermes-router的源代码仓库，包含插件实现示例和开发框架，开发者可参考现有插件代码进行开发。
- [GIE框架规范](https://gateway-api-inference-extension.sigs.k8s.io/)：Gateway API Inference Extension（GIE）的官方规范文档，定义了EPP插件的接口规范和开发标准。
- [EPP Architecture Proposal](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal)：Endpoint Picker Plugin（EPP）架构提案，详细说明了EPP插件的架构设计、执行流程和扩展机制。

### 开发步骤

1. 设计路由策略。

   开发者需要先明确路由策略目标，并拆解出需要落在Scheduling、RequestControl或Datalayer中的插件职责，本指南略过策略设计本身。

2. 开发路由插件。

   开发者在完成路由策略设计后，需要根据处理职责将逻辑拆分为若干EPP规范的插件并实现。下面以用于过滤候选端点的Filter插件为例，展示当前版本的EPP插件开发流程。

   2.1. 创建Filter代码文件。

   建议在`pkg/epp/framework/plugins/scheduling/filter/<plugin-name>/plugin.go`下创建新的Filter插件，并在同目录补充`plugin_test.go`。

   2.2. 定义Filter插件类型常量和结构体。

   ```go
   const PluginType = "my-filter"

   type Filter struct {
      typedName fwkplugin.TypedName
   }

   var _ fwkscheduling.Filter = (*Filter)(nil)
   ```

   2.3. 实现结构体构造函数和`TypedName()`方法。

   ```go
   func New(name string) *Filter {
      return &Filter{typedName: fwkplugin.TypedName{Type: PluginType, Name: name}}
   }

   func (f *Filter) TypedName() fwkplugin.TypedName {
      return f.typedName
   }
   ```

   2.4. 实现工厂函数。

   ```go
   func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
      if err := parseParameters(raw); err != nil {
         return nil, fmt.Errorf("my filter %q: %w", name, err)
      }
      return New(name), nil
   }
   ```

   2.5. 实现`Filter`接口中的`Filter()`方法。

   ```go
   func (f *Filter) Filter(
      ctx context.Context,
      state *fwkscheduling.CycleState,
      request *fwkscheduling.LLMRequest,
      endpoints []fwkscheduling.Endpoint,
   ) []fwkscheduling.Endpoint {
      _ = ctx
      _ = state
      _ = request
      return endpoints
   }
   ```

   2.6. 在`pkg/epp/register.go`中注册新增插件。

   ```go
   func RegisterAllPlugins() {
      fwkplugin.Register(myfilter.PluginType, myfilter.Factory)
      // ... 其他插件
   }
   ```

   2.7. 补充参数解析和单元测试。

   当前仓库中的插件通常会通过`json.RawMessage`解析参数，并使用`plugin_test.go`验证参数、过滤结果或评分逻辑。开发者可以参考`pd-group-filter`、`prefix-cache-filter`等现有实现组织测试用例。

   至此一个Filter插件开发完成。开发者可以按照相同模式开发Scorer、Picker、PreRequest、Lifecycle或Datalayer相关插件，只需替换对应的目录和接口。

3. 定义路由策略。

   当前版本中，开发者可以按照如下两种方式接入路由策略。

   - 复用内置路由能力：通过`inferenceExtension.pluginsConfigFile: default-plugins.yaml`和`inferenceExtension.routing`选择Hermes-router已经内置的调度链。
   - 编排自定义插件链：通过`inferenceExtension.pluginsConfigFile`指定自定义配置文件名，并通过`inferenceExtension.pluginsCustomConfig`提供`EndpointPickerConfig`内容。

   内置路由能力的配置示例如下。

   ```yaml
   inferenceExtension:
     pluginsConfigFile: default-plugins.yaml
     routing:
       deploymentMode: aggregate
       profile: random
   ```

   自定义插件链的配置示例如下。

   ```yaml
   inferenceExtension:
     pluginsConfigFile: custom-plugins.yaml
     pluginsCustomConfig:
       custom-plugins.yaml: |
         apiVersion: inference.networking.x-k8s.io/v1alpha1
         kind: EndpointPickerConfig
         plugins:
         - name: my-filter
           type: my-filter
         - name: random-picker
           type: random-picker
         schedulingProfiles:
         - name: default
           plugins:
           - pluginRef: my-filter
           - pluginRef: random-picker
   ```

   在自定义配置中，开发者可以根据需要进一步补充`dataLayer`、多条`schedulingProfiles`或其他插件参数配置。

至此开发者已完成路由策略的开发。

### 调测验证

完成插件开发后，建议优先通过standalone模式进行快速验证；如需验证与网关的集成效果，再切换到Gateway模式联调。

1. standalone模式快速验证。

   standalone模式下，用户可以直接访问带有`epp=<RELEASE_NAME>`标签的Pod。该Pod内的envoy sidecar会在`8081`端口接收推理请求，并将请求交由EPP完成路由。下面以release名称`hermes-router`为例进行说明。

   ```shell
   RELEASE_NAME=hermes-router
   EPP_POD_IP=$(kubectl get pods -n <NAMESPACE> -l epp=${RELEASE_NAME} -o jsonpath='{.items[0].status.podIP}')
   ```

   发送推理请求。

   ```shell
   curl -X POST http://${EPP_POD_IP}:8081/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
       "model": "Qwen/Qwen3-8B",
       "messages": [
         {"role": "user", "content": "你好"}
       ],
       "max_tokens": 100,
       "temperature": 0.7,
       "stream": false
     }'
   ```

2. 可选：基于Gateway的联调验证。

   ```shell
   kubectl get svc -n <NAMESPACE> -l gateway.networking.k8s.io/gateway-name=inference-gateway
   kubectl port-forward -n <NAMESPACE> service/<gateway-service-name> 8000:80
   ```

   在另一终端发送推理请求。

   ```shell
   curl -X POST http://localhost:8000/v1/chat/completions \
   -H "Content-Type: application/json" \
   -d '{
      "model": "Qwen/Qwen3-8B",
      "messages": [
         {"role": "user", "content": "你好"}
      ],
      "max_tokens": 100,
      "temperature": 0.7,
      "stream": false
   }'
   ```

3. 如请求未按预期路由，建议检查以下内容。

   - 自定义插件是否已经在`pkg/epp/register.go`中完成注册。
   - `pluginsConfigFile`是否指向了正确的配置文件。
   - `schedulingProfiles`中的插件引用顺序是否符合预期。
   - EPP Pod日志中是否存在参数解析、插件初始化或转发相关报错。

开发完成后，请参照[用户指南](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md)中的**安装**章节进行部署及验证工作。
