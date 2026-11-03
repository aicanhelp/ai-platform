# AI推理弹性伸缩插件开发指南

## 特性介绍

Elastic Scaler是AI推理PD-Orchestrator的核心组件，采用插件化架构支持用户自定义扩缩容决策算法及自定义资源管理逻辑。

Elastic Scaler提供两类插件接口：
- 扩缩容决策算法插件（ScalingAlgorithm）：基于指标数据计算目标副本数，支持多种自定义算法。
- 资源管理插件（ResourceHandler）：支持任意自定义资源类型接入扩缩体系，处理资源的副本数获取与更新。

详情请参见[Elastic Scaler用户指南](../user_guide/ai_inference_elastic_scaling.md)。

## 约束与限制

本章节梳理当前版本Elastic Scaler插件的能力边界及使用限制，明确插件适用范围。

### 功能限制

**扩缩容决策算法插件**
   - 当前版本中，自定义算法调用实现路径还未实现。
   - 自定义算法可以成功注册到`DefaultAlgorithmManager`，但在实际扩缩容决策时不会被调用。
   - 临时方案：可等待后续版本完善算法调用路径，或直接修改`AlgorithmManager.CalculateDesiredReplicas()` 实现算法分发逻辑。

### 操作限制

**注册限制**
   - 尝试注册同名算法时，`RegisterAlgorithm()`会返回错误。
   - 尝试注册相同apiVersion/kind的资源Handler时，`RegisterResourceHandler()`会panic。
   - 避免在多个`init()`函数中重复注册。

**性能考虑**
   - 插件中的计算逻辑应尽量高效，避免阻塞主控制器循环。
   - 复杂的算法计算建议异步处理或使用缓存。
   - 频繁的API调用可能影响K8s API Server性能。

**潜在风险**
   - 错误的扩缩容算法可能导致资源过度扩容或缩容，影响业务稳定性。
   - 资源Handler实现错误可能导致目标资源状态不一致。
   - 建议在测试环境充分验证后再部署到生产环境。

## 环境准备

本章节介绍在开始Elastic Scaler插件开发前需要完成的环境和工具准备工作。

### 环境要求

本小节说明进行Elastic Scaler插件开发与调测所需的硬件与软件环境配置。

#### 硬件要求

Elastic Scaler插件开发环境对硬件无特殊要求，建议按照如下进行配置：

- **CPU**：4核及以上
- **内存**：8GB及以上
- **磁盘**：20GB及以上可用空间

#### 软件要求

- **操作系统**：Linux
- **Go环境**：Go 1.21或更高版本
- **Docker**：Docker 20.10+或兼容的容器运行时（如nerdctl）
- **Kubernetes集群**：用于部署和测试
- **kubectl**：用于与K8s集群交互

### 搭建环境

1. 克隆代码仓库。

   ```bash
   git clone https://gitcode.com/openFuyao/elastic-scaler.git
   cd elastic-scaler
   ```

2. 配置Go开发环境。

   ```bash
   # 检查Go版本
   go version

   # 设置Go代理（可选，加速依赖下载）
   go env -w GOPROXY=https://goproxy.cn,direct
   ```

3. 安装依赖。

   ```bash
   # 下载依赖
   go mod download

   # 验证依赖
   go mod verify
   ```

### 检验环境是否搭建成功

1. 验证Go环境。

   ```bash
   go version
   # 应输出类似：go version go1.21.x Linux/amd64
   ```

2. 验证依赖完整性。

   ```bash
   go mod verify
   # 应输出：all modules verified
   ```

3. 验证编译能力。

   ```bash
   make build
   # 或
   go build -o bin/elastic-scaler ./cmd/elastic-scaler
   # 编译成功表示环境搭建成功
   ```

4. 验证K8s连接（可选，用于测试）。

   ```bash
   kubectl cluster-info
   kubectl get nodes
   # 应能正常显示集群信息
   ```


## 开发扩缩容决策算法插件

本章节说明如何基于ScalingAlgorithm接口实现并集成自定义扩缩容决策算法插件。

### 使用场景概述

扩缩容决策算法插件适用于需要根据特定业务逻辑或指标模式来决定扩缩容策略的场景，例如：
- 基于预测性扩缩容：根据历史负载数据预测未来需求。
- 多指标联合决策：同时考虑CPU、内存、QPS等多个指标。
- 成本优化扩缩容：在满足SLA的前提下最小化资源成本。
- 业务感知扩缩容：根据业务高峰期、促销活动等业务特征调整策略。

### 组件与接口说明

**表1** 系统组件职责
| 系统组件名 | 描述 |
|--------|------|
| ElasticScaler CR | 定义扩缩容策略，包括算法名称、配置参数和指标源。|
| AlgorithmManager | 管理所有已注册的算法插件，负责算法分发。 |
| ScalingAlgorithm | 算法插件接口，由开发者实现具体的扩缩容逻辑。|


**表2** 决策算法插件接口说明
| 接口名 | 描述 |
|--------|------|
| CalculateDesiredReplicas | 基于指标数据和上下文计算期望副本数。 |

### 开发步骤

1. 创建算法实现文件。

   在`pkg/elasticscaler/scaling/`目录下创建新的算法实现文件，例如`custom_algorithm.go`。

   ```go
   package scaling

   import (
      "context"
      "fmt"

      escontext "gitcode.com/openFuyao/elastic-scaler/pkg/elasticscaler/context"
   )

   // CustomAlgorithm implements ScalingAlgorithm interface.
   type CustomAlgorithm struct {
      // 算法配置参数
      threshold float64
      cooldownSeconds int32
   }

   var _ ScalingAlgorithm = &CustomAlgorithm{}

   // NewCustomAlgorithm returns a new CustomAlgorithm instance.
   func NewCustomAlgorithm() ScalingAlgorithm {
      return &CustomAlgorithm{
         threshold:      0.8,
         cooldownSeconds: 60,
      }
   }

   // CalculateDesiredReplicas calculates desired replicas based on metrics.
   func (a *CustomAlgorithm) CalculateDesiredReplicas(
      ctx context.Context,
      processingCtx *escontext.ScalingAlgorithmContext,
   ) (int32, error) {
      if processingCtx == nil || processingCtx.ElasticScaler == nil {
         return 0, fmt.Errorf("processing context or Elastic is nil")
      }

      // 从上下文中获取当前指标
      // metrics := processingCtx.Metrics
      // currentReplicas := processingCtx.CurrentReplicas

      // TODO: 实现自定义扩缩算法逻辑
      // 示例：基于阈值计算期望副本数
      // if metricValue > a.threshold {
      //     return currentReplicas + 1
      // }

      return 0, nil
   }
   ```


   > ![](./figures/icon-notice.gif)**注意**：
   >
   > - 在`CalculateDesiredReplicas`方法中添加充分的错误处理和边界检查。
   > - 使用klog记录关键计算步骤和中间结果，便于调测。
   > - 确保返回的副本数在[minReplicas，maxReplicas]范围内。
   > - 考虑实现冷却时间（cooldown）机制，避免频繁扩缩容。

2. 注册算法。

   在算法实现的`init()`函数中注册算法到`DefaultAlgorithmManager`。

   ```go
   func init() {
      if err := DefaultAlgorithmManager.RegisterAlgorithm("custom", NewCustomAlgorithm()); err != nil {
         // panic(err)
         fmt.Println("register custom algorithm failed", err)
      }
   }
   ```

   **表3** 注册方法说明

   | 参数 | 类型 | 说明 |
   |------|------|------|
   | name | string | 算法名称，用于在ElasticScaler CR中引用。 |
   | algorithm | ScalingAlgorithm | ScalingAlgorithm接口实现。 |

   **返回值**：error， 如果同名算法已注册则返回错误。

   >![](./figures/icon-notice.gif)**注意**：
   >
   >- 算法名称必须唯一，建议使用具有业务含义的名称。
   >- 注册失败时可以选择panic或记录日志，根据需求决定。
   >- 避免在多个init()函数中重复注册同一算法。

3. 配置ElasticScaler使用自定义算法。

   创建ElasticScaler CRD资源，指定使用自定义算法。

   ```yaml
   apiVersion: elasticscaler.io/v1alpha1
   kind: ElasticScaler
   metadata:
   name: custom-scaling-example
   namespace: ai-inference
   spec:
   # 伸缩目标
   targetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: vllm-inference

   # 最小/最大副本数
   minReplicas: 2
   maxReplicas: 10

   # 指标驱动触发配置
   trigger:
      type: MetricsTrigger
      metricsTrigger:
         # 使用自定义算法
         scalingAlgorithm: custom

         # 算法配置
         algorithmConfig:
         threshold: 0.8
         cooldownSeconds: 60

         # 指标配置
         metrics:
         - type: External
         external:
            metric:
               name: qps
            target:
               type: AverageValueValue
               averageValue: "100"
   ```

### 调测验证

本小节介绍自定义扩缩容算法插件在集群中的验证步骤及排查思路。

#### 验证算法注册

1. 执行如下命令，启动Elastic Scaler控制器。

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f
   ```

2. 执行如下命令，查看日志确认算法注册成功。

   ```
   # 应看到类似输出
   register custom algorithm failed <nil>
   ```

#### 验证算法调用

1. 执行如下命令，创建测试ElasticScaler资源。

   ```bash
   kubectl apply -f test-elastic-scaler.yaml
   ```

2. 执行如下命令，查看ElasticScaler状态。

   ```bash
   kubectl get elasticscalers -n ai-inference -o yaml
   ```

3. 执行如下命令，查看控制器日志，确认算法被调用。

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f | grep CustomAlgorithm
   ```

> ![](./figures/icon-notice.gif)**注意**：
> 
> - 由于当前版本算法调用路径尚未实现，可能无法看到实际的算法调用日志。
> - 建议在算法实现中添加klog日志，便于后续调测。
> - 可以通过单元测试验证算法逻辑的正确性。

## 开发资源管理插件

本章节说明如何实现自定义资源管理插件，并将任意Kubernetes资源接入Elastic Scaler扩缩容体系。

### 使用场景概述

资源管理插件适用于需要将自定义K8s资源类型接入Elastic Scaler扩缩容体系的场景，例如：
- 自定义工作负载资源：如自定义的推理服务CRD。
- 需要特殊副本管理的资源：如需要同时更新多个字段的复杂资源。
- 跨资源副本协调：如需要同时管理多个关联资源的副本数。

### 组件与接口说明

**表4** 系统组件职责
| 系统组件名 | 描述 |
|--------|------|
| ElasticScaler CR | 定义目标资源引用（apiVersion、kind、name）。 |
| HandlerFactory | 管理所有已注册的资源Handler，根据目标资源类型创建对应的Handler实例。 |
| ResourceHandler | 资源Handler接口，由开发者实现具体的资源管理逻辑。 |

**表5** 资源管理插件接口说明
| 接口名 | 描述 |
|--------|------|
| Supports | 检查Handler是否支持给定的资源类型。 |
| GetCurrentReplicas | 获取目标资源的当前副本数。 |
| UpdateReplicas | 更新目标资源的副本数。 |
| GetPodList | 获取目标资源管理的Pod列表。 |

### 开发步骤

1. 创建资源Handler实现文件。

   在`pkg/elasticscaler/resource/`目录下创建新的资源Handler实现文件，例如`custom_resource_handler.go`。

   ```go
   package resource

   import (
      "context"
      "fmt"

      "sigs.k8s.io/controller-runtime/pkg/client"
   )

   // CustomResourceHandler implements ResourceHandler for custom resources.
   type CustomResourceHandler struct {
      Client client.Client
   }

   var _ ResourceHandler = &CustomResourceHandler{}

   // NewCustomResourceHandler returns a new CustomResourceHandler instance.
   func NewCustomResourceHandler(c client.Client) ResourceHandler {
      return &CustomResourceHandler{Client: c}
   }

   // Supports checks if the handler supports the given resource type.
   func (h *CustomResourceHandler) Supports(targetRef corev1.ObjectReference) bool {
      // TODO: 判断是否支持该资源类型
      // 示例：只支持特定的apiVersion和kind
      return targetRef.APIVersion == "custom.example.com/v1alpha1" &&
         targetRef.Kind == "CustomResource"
   }

   // GetCurrentReplicas gets the current number of replicas.
   func (h *CustomResourceHandler) GetCurrentReplicas(
      ctx context.Context,
      targetRef corev1.ObjectReference,
   ) (int32, error) {
      // TODO: 根据你的CRD结构读取当前replicas（spec或status）
      // 示例：从spec.replicas或status.replicas字段读取
      return 0, fmt.Errorf("not implemented")
   }

   // UpdateReplicas updates the number of replicas.
   func (h *CustomResourceHandler) UpdateReplicas(
      ctx context.Context,
      targetRef corev1.ObjectReference,
      desired int32,
   ) error {
      // TODO: 更新你的CRD的spec.replicas或类似字段
      // 示例：通过client.Update()或client.Patch()更新资源
      return fmt.Errorf("not implemented")
   }

   // GetPodList gets the list of pods managed by the target resource.
   func (h *CustomResourceHandler) GetPodList(
      ctx context.Context,
      targetRef corev1.ObjectReference,
   ) (*corev1.PodList, error) {
      // TODO: 根据CRD中定义的selector列出Pod
      selector := labels.Everything()
      podList := &corev1.PodList{}
      if err := h.Client.List(ctx, podList, &client.ListOptions{
         Namespace:     targetRef.Namespace,
         LabelSelector: selector,
      }); err != nil {
         return nil, err
      }
      return podList, nil
   }
   ```

   > ![](./figures/icon-notice.gif)**注意**：
   > 
   > - 在`Supports`方法中精确匹配apiVersion和kind，避免误匹配。
   > - `UpdateReplicas`建议使用Patch而非Update，减少冲突。
   > - `GetPodList`应使用资源定义的selector标签，避免列出无关Pod。
   > - 所有方法都应添加充分的错误处理和日志记录。

2. 注册资源Handler。

   在资源Handler实现的`init()`函数中注册到HandlerFactory。

   ```go
   func init() {
      RegisterResourceHandler(apiVersion, kind, func(c client.Client) ResourceHandler {
         return NewCustomResourceHandler(c)
      })
   }
   ```

   **表6** 注册方法说明

   | 参数 | 类型 | 说明 |
   |------|------|------|
   | apiVersion | string | 自定义资源的API版本。 |
   | kind | string | 自定义资源的Kind。 |
   | factory | HandlerFactory | HandlerFactory函数，用于创建ResourceHandler实例。 |

   > ![](./figures/icon-notice.gif)**注意**：
   > 
   > - apiVersion和kind的组合必须唯一。
   > - 建议使用完整的apiVersion（包括group/version）。
   > - 避免在多个init()函数中重复注册同一资源类型。
   > - factory函数应该轻量级，避免在注册时进行重量级操作。

3. 配置ElasticScaler使用自定义资源。

   创建ElasticScaler CRD资源，指定使用自定义资源。

   ```yaml
   apiVersion: elasticscaler.io/v1alpha1
   kind: ElasticScaler
   metadata:
   name: custom-resource-scaling
   namespace: ai-inference
   spec:
   # 伸缩目标（自定义资源）
   targetRef:
      apiVersion: custom.example.com/v1alpha1
      kind: CustomResource
      name: my-custom-resource
      namespace: ai-inference

   # 最小/最大副本数
   minReplicas: 2
   maxReplicas: 10

   # 指标驱动触发配置
   trigger:
      type: MetricsTrigger
      metricsTrigger:
         scalingAlgorithm: HPA

         # 指标配置
         metrics:
         - type: Resource
         resource:
            metricsName: cpu
            target:
               type: Utilization
               averageUtilization: 70
   ```

### 调测验证

本小节介绍自定义资源管理插件在实际环境中的验证方法和关键检查项。

#### 验证Handler注册

1. 执行如下命令，启动Elastic Scaler控制器。

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f
   ```

2. 查看日志确认Handler注册成功（无panic表示注册成功）。

#### 验证资源管理

1. 执行如下命令，创建测试自定义资源。

   ```bash
   kubectl apply -f test-custom-resource.yaml
   ```

2. 执行如下命令，创建ElasticScaler资源。

   ```bash
   kubectl apply -f test-elastic-scaler.yaml
   ```

3. 执行如下命令，查看ElasticScaler状态。

   ```bash
   kubectl get elasticscalers -n ai-inference -o yaml
   ```

4. 执行如下命令，查看目标资源副本数是否正确更新。

   ```bash
   kubectl get customresources -n ai-inference -o yaml
   ```

> ![](./figures/icon-notice.gif)**注意**：
> 
> - 确保自定义资源已正确安装CRD。
> - 确保ElasticScaler有权限读写目标资源（RBAC配置）。
> - 建议在测试环境充分验证后再部署到生产环境。

## FAQ

本章节汇总Elastic Scaler插件开发与使用过程中常见问题及参考解决方案。

1. **算法注册失败**

   **现象描述**<br>
   算法注册失败，控制器日志显示错误信息。

   **可能原因**
   - 尝试注册同名算法。
   - 算法名称为空。
   - 算法实例为nil。

   **解决办法**
   1. 使用唯一的算法名称，建议使用具有业务含义的名称。
   2. 检查是否已在其他地方注册相同名称的算法。
   3. 查看控制器日志中的详细错误信息。
   4. 确保算法实例正确初始化。

2. **资源Handler注册失败**

   **现象描述**<br>
   资源Handler注册失败，控制器panic。

   **可能原因**
   - 尝试注册相同`apiVersion/kind`的资源Handler。
   - `apiVersion`或`kind`为空。
   - factory函数为nil。

   **解决办法**
   1. 确保`apiVersion`和`kind`组合唯一。
   2. 检查是否已在其他地方注册相同资源类型。
   3. 避免在多个`init()`函数中重复注册。
   4. 确保factory函数正确实现且不为nil。

3. **算法未被调用**

   **现象描述**<br>
   算法注册成功，但在实际扩缩容决策时未被调用。

   **可能原因**
   - 当前版本算法调用实现路径尚未实现。
   - `AlgorithmManager.CalculateDesiredReplicas()` 方法的核心逻辑尚未实现。

   **解决办法**
   1. 等待后续版本完善算法调用路径。
   2. 或直接修改`AlgorithmManager.CalculateDesiredReplicas()` 实现算法分发逻辑。
   3. 参考代码位置：`pkg/elasticscaler/scaling/algorithm.go`（第55-68行）。

4. **自定义资源无法扩缩容**

   **现象描述**<br>
   自定义资源Handler已注册，但ElasticScaler无法正确管理资源副本数。

   **可能原因**
   - `ResourceHandler`接口实现不完整。
   - `Supports()`方法判断逻辑错误。
   - RBAC权限不足。
   - 自定义资源CRD未正确安装。

   **解决办法**
   1. 确保实现完整的`ResourceHandler`接口（所有4个方法）。
   2. 正确处理`GetCurrentReplicas`和`UpdateReplicas`。
   3. `Supports()`方法精确匹配`apiVersion`和`Kind`。
   4. 确保ElasticScaler有权限读写目标资源（检查RBAC配置）。
   5. 确保自定义资源CRD已正确安装到集群。

5. **如何调测算法计算逻辑**

   **现象描述**<br>
   需要调测算法计算逻辑，但不确定如何获取中间结果。

   **可能原因**<br>
   算法实现复杂，需要查看中间计算过程。

   **解决办法**
   1. 在算法实现中添加klog日志。
      ```go
      import "k8s.io/klog/v2"

      func (a *CustomAlgorithm) CalculateDesiredReplicas(...) (int32, error) {
         klog.Info("CustomAlgorithm: CalculateDesiredReplicas called")
         klog.Infof("Processing context: %+v", processingCtx)
         // 算法逻辑...
         klog.Infof("Calculated desired replicas: %d", desiredReplicas)
         return desiredReplicas, nil
      }
      ```
   2. 查看ElasticScaler CR的`status`字段。
   3. 查看控制器日志中的指标数据。
   4. 使用单元测试验证算法逻辑。

6. **如何支持Scale子资源**

   **现象描述**<br>
   需要支持K8s Scale子资源（如`/scale`）。

   **可能原因**<br>
   需要实现更标准的资源副本管理方式。

   **解决办法**
   参考`DefaultCustomResourceHandler`的实现：
   1. 优先使用Scale子资源（如`/scale`）。
   2. 如果Scale子资源不可用，回退到`spec.replicas`或`status.replicas`。
   3. 使用`unstructured.Unstructured`处理通用资源。
   4. 参考代码位置：`pkg/elasticscaler/resource/default_custom_resource_handler.go`。

## 附录

本章节提供Elastic Scaler插件开发相关的参考资料与延伸阅读链接。

### 参考资源

- **elastic-scaler代码仓库**：https://gitcode.com/openFuyao/elastic-scaler
  - `pkg/elasticscaler/scaling/`：扩缩算法实现示例。
  - `pkg/elasticscaler/resource/`：资源Handler实现示例。
  - `pkg/elasticscaler/context/`：上下文接口定义。

- **OFEP 0030提案**：https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0030-ofep-通用扩缩容决策框架设计提案.md
  - 详细的设计说明和架构设计。
  - 接口定义和使用场景。
