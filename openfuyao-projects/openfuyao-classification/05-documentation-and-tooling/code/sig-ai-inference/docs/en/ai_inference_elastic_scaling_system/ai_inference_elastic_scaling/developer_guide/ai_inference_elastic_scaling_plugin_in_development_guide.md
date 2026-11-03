# AI Inference Elastic Scaling Plugin Developer Guide

## Feature Introduction

Elastic Scaler is a core component of the AI inference PD-Orchestrator, adopting a plugin-based architecture to support user-defined scaling decision algorithms and custom resource management logic.

Elastic Scaler provides two types of plugin interfaces:
- Scaling Decision Algorithm Plugin (ScalingAlgorithm): Calculates the target number of replicas based on metric data, supporting various custom algorithms.
- Resource Management Plugin (ResourceHandler): Supports any custom resource type integration into the scaling system, handling replica count retrieval and updates for resources.

For details, see the [Elastic Scaler User Guide](../user_guide/ai_inference_elastic_scaling.md).

## Constraints and Limitations

This section outlines the capability boundaries and usage limitations of the Elastic Scaler plugin in the current version, clarifying the scope of plugin applicability.

### Functional Limitations

**Scaling Decision Algorithm Plugin**
   - In the current version, the custom algorithm invocation implementation path has not yet been implemented.
   - Custom algorithms can be successfully registered to `DefaultAlgorithmManager`, but will not be invoked during actual scaling decisions.
   - Temporary workaround: Wait for subsequent versions to complete the algorithm invocation path, or directly modify `AlgorithmManager.CalculateDesiredReplicas()` to implement the algorithm dispatch logic.

### Operational Limitations

**Registration Limitations**
   - When attempting to register an algorithm with the same name, `RegisterAlgorithm()` will return an error.
   - When attempting to register a resource Handler with the same apiVersion/kind, `RegisterResourceHandler()` will panic.
   - Avoid duplicate registrations in multiple `init()` functions.

**Performance Considerations**
   - Computation logic in plugins should be as efficient as possible, avoiding blocking the main controller loop.
   - Complex algorithm computations are recommended to be handled asynchronously or with caching.
   - Frequent API calls may affect K8s API Server performance.

**Potential Risks**
   - Incorrect scaling algorithms may lead to resource over-scaling or under-scaling, affecting business stability.
   - Incorrect resource Handler implementation may lead to inconsistent target resource states.
   - It is recommended to fully validate in a test environment before deploying to a production environment.

## Environment Preparation

This section describes the environment and tool preparation work required before starting Elastic Scaler plugin development.

### Environment Requirements

This subsection describes the hardware and software environment configurations required for Elastic Scaler plugin development and debugging.

#### Hardware Requirements

The Elastic Scaler plugin development environment has no special hardware requirements. Recommended configuration is as follows:

- **CPU**: 4 cores or above
- **Memory**: 8GB or above
- **Disk**: 20GB or above of available space

#### Software Requirements

- **Operating System**: Linux
- **Go Environment**: Go 1.21 or higher
- **Docker**: Docker 20.10+ or compatible container runtime (such as nerdctl)
- **Kubernetes Cluster**: For deployment and testing
- **kubectl**: For interacting with the K8s cluster

### Setting Up the Environment

1. Clone the code repository.

   ```bash
   git clone https://gitcode.com/openFuyao/elastic-scaler.git
   cd elastic-scaler
   ```

2. Configure the Go development environment.

   ```bash
   # Check Go version
   go version

   # Set Go proxy (optional, accelerates dependency downloads)
   go env -w GOPROXY=https://goproxy.cn,direct
   ```

3. Install dependencies.

   ```bash
   # Download dependencies
   go mod download

   # Verify dependencies
   go mod verify
   ```

### Verifying Environment Setup

1. Verify the Go environment.

   ```bash
   go version
   # Should output something like: go version go1.21.x Linux/amd64
   ```

2. Verify dependency integrity.

   ```bash
   go mod verify
   # Should output: all modules verified
   ```

3. Verify compilation capability.

   ```bash
   make build
   # Or
   go build -o bin/elastic-scaler ./cmd/elastic-scaler
   # Successful compilation indicates environment setup is complete
   ```

4. Verify K8s connection (optional, for testing).

   ```bash
   kubectl cluster-info
   kubectl get nodes
   # Should display cluster information normally
   ```


## Developing a Scaling Decision Algorithm Plugin

This section describes how to implement and integrate a custom scaling decision algorithm plugin based on the ScalingAlgorithm interface.

### Use Case Overview

The scaling decision algorithm plugin is applicable to scenarios that require scaling strategies based on specific business logic or metric patterns, for example:
- Predictive scaling: Predicting future demand based on historical load data.
- Multi-metric joint decision-making: Considering CPU, memory, QPS, and other metrics simultaneously.
- Cost-optimized scaling: Minimizing resource costs while satisfying SLA requirements.
- Business-aware scaling: Adjusting strategies based on business characteristics such as peak periods and promotional activities.

### Component and Interface Description

**Table 1** System Component Responsibilities
| System Component Name | Description |
|--------|------|
| ElasticScaler CR | Defines scaling strategies, including algorithm name, configuration parameters, and metric sources.|
| AlgorithmManager | Manages all registered algorithm plugins, responsible for algorithm dispatch. |
| ScalingAlgorithm | Algorithm plugin interface, implemented by developers with specific scaling logic.|

**Table 2** Decision Algorithm Plugin Interface Description
| Interface Name | Description |
|--------|------|
| CalculateDesiredReplicas | Calculates the desired number of replicas based on metric data and context. |

### Development Steps

1. Create an algorithm implementation file.

   Create a new algorithm implementation file under the `pkg/elasticscaler/scaling/` directory, for example `custom_algorithm.go`.

   ```go
   package scaling

   import (
      "context"
      "fmt"

      escontext "gitcode.com/openFuyao/elastic-scaler/pkg/elasticscaler/context"
   )

   // CustomAlgorithm implements ScalingAlgorithm interface.
   type CustomAlgorithm struct {
      // Algorithm configuration parameters
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

      // Get current metrics from context
      // metrics := processingCtx.Metrics
      // currentReplicas := processingCtx.CurrentReplicas

      // TODO: Implement custom scaling algorithm logic
      // Example: Calculate desired replicas based on threshold
      // if metricValue > a.threshold {
      //     return currentReplicas + 1
      // }

      return 0, nil
   }
   ```


   > ![](./figures/icon-notice.gif)**Note**:
   >
   > - Add sufficient error handling and boundary checks in the `CalculateDesiredReplicas` method.
   > - Use klog to record key computation steps and intermediate results for debugging convenience.
   > - Ensure the returned replica count is within the [minReplicas, maxReplicas] range.
   > - Consider implementing a cooldown mechanism to avoid frequent scaling.

2. Register the algorithm.

   Register the algorithm to `DefaultAlgorithmManager` in the `init()` function of the algorithm implementation.

   ```go
   func init() {
      if err := DefaultAlgorithmManager.RegisterAlgorithm("custom", NewCustomAlgorithm()); err != nil {
         // panic(err)
         fmt.Println("register custom algorithm failed", err)
      }
   }
   ```

   **Table 3** Registration Method Description

   | Parameter | Type | Description |
   |------|------|------|
   | name | string | Algorithm name, used for reference in ElasticScaler CR. |
   | algorithm | ScalingAlgorithm | ScalingAlgorithm interface implementation. |

   **Return value**: error, returns an error if an algorithm with the same name is already registered.

   >![](./figures/icon-notice.gif)**Note**:
   >
   >- The algorithm name must be unique; it is recommended to use a name with business significance.
   >- On registration failure, you can choose to panic or log, depending on requirements.
   >- Avoid duplicate registration of the same algorithm in multiple init() functions.

3. Configure ElasticScaler to use the custom algorithm.

   Create an ElasticScaler CRD resource specifying the use of the custom algorithm.

   ```yaml
   apiVersion: elasticscaler.io/v1alpha1
   kind: ElasticScaler
   metadata:
   name: custom-scaling-example
   namespace: ai-inference
   spec:
   # Scaling target
   targetRef:
      apiVersion: apps/v1
      kind: Deployment
      name: vllm-inference

   # Minimum/maximum replicas
   minReplicas: 2
   maxReplicas: 10

   # Metric-driven trigger configuration
   trigger:
      type: MetricsTrigger
      metricsTrigger:
         # Use custom algorithm
         scalingAlgorithm: custom

         # Algorithm configuration
         algorithmConfig:
         threshold: 0.8
         cooldownSeconds: 60

         # Metric configuration
         metrics:
         - type: External
         external:
            metric:
               name: qps
            target:
               type: AverageValueValue
               averageValue: "100"
   ```

### Debugging and Verification

This subsection describes the verification steps and troubleshooting approaches for custom scaling algorithm plugins in the cluster.

#### Verifying Algorithm Registration

1. Run the following command to start the Elastic Scaler controller.

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f
   ```

2. Run the following command to check logs and confirm successful algorithm registration.

   ```
   # Should see output similar to
   register custom algorithm failed <nil>
   ```

#### Verifying Algorithm Invocation

1. Run the following command to create a test ElasticScaler resource.

   ```bash
   kubectl apply -f test-elastic-scaler.yaml
   ```

2. Run the following command to check ElasticScaler status.

   ```bash
   kubectl get elasticscalers -n ai-inference -o yaml
   ```

3. Run the following command to check controller logs and confirm that the algorithm is invoked.

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f | grep CustomAlgorithm
   ```

> ![](./figures/icon-notice.gif)**Note**:
> 
> - Due to the algorithm invocation path not yet being implemented in the current version, actual algorithm invocation logs may not be visible.
> - It is recommended to add klog logs in the algorithm implementation for subsequent debugging convenience.
> - Algorithm logic correctness can be verified through unit tests.

## Developing a Resource Management Plugin

This section describes how to implement a custom resource management plugin and integrate any Kubernetes resource into the Elastic Scaler scaling system.

### Use Case Overview

The resource management plugin is applicable to scenarios that require integrating custom K8s resource types into the Elastic Scaler scaling system, for example:
- Custom workload resources: Such as custom inference service CRDs.
- Resources requiring special replica management: Such as complex resources that need to update multiple fields simultaneously.
- Cross-resource replica coordination: Such as managing replica counts of multiple associated resources simultaneously.

### Component and Interface Description

**Table 4** System Component Responsibilities
| System Component Name | Description |
|--------|------|
| ElasticScaler CR | Defines target resource reference (apiVersion, kind, name). |
| HandlerFactory | Manages all registered resource Handlers, creating corresponding Handler instances based on target resource type. |
| ResourceHandler | Resource Handler interface, implemented by developers with specific resource management logic. |

**Table 5** Resource Management Plugin Interface Description
| Interface Name | Description |
|--------|------|
| Supports | Checks whether the Handler supports the given resource type. |
| GetCurrentReplicas | Gets the current replica count of the target resource. |
| UpdateReplicas | Updates the replica count of the target resource. |
| GetPodList | Gets the list of Pods managed by the target resource. |

### Development Steps

1. Create a resource Handler implementation file.

   Create a new resource Handler implementation file under the `pkg/elasticscaler/resource/` directory, for example `custom_resource_handler.go`.

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
      // TODO: Determine whether this resource type is supported
      // Example: Only support specific apiVersion and kind
      return targetRef.APIVersion == "custom.example.com/v1alpha1" &&
         targetRef.Kind == "CustomResource"
   }

   // GetCurrentReplicas gets the current number of replicas.
   func (h *CustomResourceHandler) GetCurrentReplicas(
      ctx context.Context,
      targetRef corev1.ObjectReference,
   ) (int32, error) {
      // TODO: Read current replicas from your CRD structure (spec or status)
      // Example: Read from spec.replicas or status.replicas field
      return 0, fmt.Errorf("not implemented")
   }

   // UpdateReplicas updates the number of replicas.
   func (h *CustomResourceHandler) UpdateReplicas(
      ctx context.Context,
      targetRef corev1.ObjectReference,
      desired int32,
   ) error {
      // TODO: Update your CRD's spec.replicas or similar field
      // Example: Update resource via client.Update() or client.Patch()
      return fmt.Errorf("not implemented")
   }

   // GetPodList gets the list of pods managed by the target resource.
   func (h *CustomResourceHandler) GetPodList(
      ctx context.Context,
      targetRef corev1.ObjectReference,
   ) (*corev1.PodList, error) {
      // TODO: List Pods based on the selector defined in the CRD
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

   > ![](./figures/icon-notice.gif)**Note**:
   > 
   > - Precisely match apiVersion and kind in the `Supports` method to avoid mismatches.
   > - It is recommended to use Patch rather than Update in `UpdateReplicas` to reduce conflicts.
   > - `GetPodList` should use the selector labels defined in the resource to avoid listing unrelated Pods.
   > - All methods should have sufficient error handling and logging.

2. Register the resource Handler.

   Register it to HandlerFactory in the `init()` function of the resource Handler implementation.

   ```go
   func init() {
      RegisterResourceHandler(apiVersion, kind, func(c client.Client) ResourceHandler {
         return NewCustomResourceHandler(c)
      })
   }
   ```

   **Table 6** Registration Method Description

   | Parameter | Type | Description |
   |------|------|------|
   | apiVersion | string | API version of the custom resource. |
   | kind | string | Kind of the custom resource. |
   | factory | HandlerFactory | HandlerFactory function, used to create ResourceHandler instances. |

   > ![](./figures/icon-notice.gif)**Note**:
   > 
   > - The combination of apiVersion and kind must be unique.
   > - It is recommended to use the complete apiVersion (including group/version).
   > - Avoid duplicate registration of the same resource type in multiple init() functions.
   > - The factory function should be lightweight; avoid performing heavyweight operations during registration.

3. Configure ElasticScaler to use the custom resource.

   Create an ElasticScaler CRD resource specifying the use of the custom resource.

   ```yaml
   apiVersion: elasticscaler.io/v1alpha1
   kind: ElasticScaler
   metadata:
   name: custom-resource-scaling
   namespace: ai-inference
   spec:
   # Scaling target (custom resource)
   targetRef:
      apiVersion: custom.example.com/v1alpha1
      kind: CustomResource
      name: my-custom-resource
      namespace: ai-inference

   # Minimum/maximum replicas
   minReplicas: 2
   maxReplicas: 10

   # Metric-driven trigger configuration
   trigger:
      type: MetricsTrigger
      metricsTrigger:
         scalingAlgorithm: HPA

         # Metric configuration
         metrics:
         - type: Resource
         resource:
            metricsName: cpu
            target:
               type: Utilization
               averageUtilization: 70
   ```

### Debugging and Verification

This subsection describes the verification methods and key checkpoints for custom resource management plugins in the actual environment.

#### Verifying Handler Registration

1. Run the following command to start the Elastic Scaler controller.

   ```bash
   kubectl logs -n ai-inference -l control-plane=elastic-scaler-controller-manager -f
   ```

2. Check logs to confirm successful Handler registration (no panic indicates successful registration).

#### Verifying Resource Management

1. Run the following command to create a test custom resource.

   ```bash
   kubectl apply -f test-custom-resource.yaml
   ```

2. Run the following command to create an ElasticScaler resource.

   ```bash
   kubectl apply -f test-elastic-scaler.yaml
   ```

3. Run the following command to check ElasticScaler status.

   ```bash
   kubectl get elasticscalers -n ai-inference -o yaml
   ```

4. Run the following command to check whether the target resource replica count has been correctly updated.

   ```bash
   kubectl get customresources -n ai-inference -o yaml
   ```

> ![](./figures/icon-notice.gif)**Note**:
> 
> - Ensure the custom resource CRD has been correctly installed.
> - Ensure ElasticScaler has permission to read and write the target resource (RBAC configuration).
> - It is recommended to fully validate in a test environment before deploying to a production environment.

## FAQ

This section summarizes common issues and reference solutions during Elastic Scaler plugin development and usage.

1. **Algorithm Registration Failure**

   **Symptom Description**<br>
   Algorithm registration fails, and the controller log shows error messages.

   **Possible Causes**
   - Attempting to register an algorithm with the same name.
   - Algorithm name is empty.
   - Algorithm instance is nil.

   **Solutions**
   1. Use a unique algorithm name; it is recommended to use a name with business significance.
   2. Check whether an algorithm with the same name has already been registered elsewhere.
   3. Review detailed error information in the controller logs.
   4. Ensure the algorithm instance is correctly initialized.

2. **Resource Handler Registration Failure**

   **Symptom Description**<br>
   Resource Handler registration fails, and the controller panics.

   **Possible Causes**
   - Attempting to register a resource Handler with the same `apiVersion/kind`.
   - `apiVersion` or `kind` is empty.
   - factory function is nil.

   **Solutions**
   1. Ensure the `apiVersion` and `kind` combination is unique.
   2. Check whether the same resource type has already been registered elsewhere.
   3. Avoid duplicate registrations in multiple `init()` functions.
   4. Ensure the factory function is correctly implemented and not nil.

3. **Algorithm Not Invoked**

   **Symptom Description**<br>
   Algorithm is registered successfully but not invoked during actual scaling decisions.

   **Possible Causes**
   - The algorithm invocation implementation path has not yet been implemented in the current version.
   - The core logic of the `AlgorithmManager.CalculateDesiredReplicas()` method has not yet been implemented.

   **Solutions**
   1. Wait for subsequent versions to complete the algorithm invocation path.
   2. Or directly modify `AlgorithmManager.CalculateDesiredReplicas()` to implement the algorithm dispatch logic.
   3. Reference code location: `pkg/elasticscaler/scaling/algorithm.go` (lines 55-68).

4. **Custom Resource Cannot Scale**

   **Symptom Description**<br>
   Custom resource Handler has been registered, but ElasticScaler cannot correctly manage the resource replica count.

   **Possible Causes**
   - `ResourceHandler` interface implementation is incomplete.
   - `Supports()` method judgment logic is incorrect.
   - Insufficient RBAC permissions.
   - Custom resource CRD has not been correctly installed.

   **Solutions**
   1. Ensure complete implementation of the `ResourceHandler` interface (all 4 methods).
   2. Correctly handle `GetCurrentReplicas` and `UpdateReplicas`.
   3. `Supports()` method should precisely match `apiVersion` and `Kind`.
   4. Ensure ElasticScaler has permission to read and write the target resource (check RBAC configuration).
   5. Ensure the custom resource CRD has been correctly installed in the cluster.

5. **How to Debug Algorithm Computation Logic**

   **Symptom Description**<br>
   Need to debug algorithm computation logic but unsure how to obtain intermediate results.

   **Possible Causes**<br>
   Algorithm implementation is complex, and intermediate computation processes need to be examined.

   **Solutions**
   1. Add klog logs in the algorithm implementation.
      ```go
      import "k8s.io/klog/v2"

      func (a *CustomAlgorithm) CalculateDesiredReplicas(...) (int32, error) {
         klog.Info("CustomAlgorithm: CalculateDesiredReplicas called")
         klog.Infof("Processing context: %+v", processingCtx)
         // Algorithm logic...
         klog.Infof("Calculated desired replicas: %d", desiredReplicas)
         return desiredReplicas, nil
      }
      ```
   2. Check the `status` field of the ElasticScaler CR.
   3. Check metric data in the controller logs.
   4. Use unit tests to verify algorithm logic.

6. **How to Support Scale Subresource**

   **Symptom Description**<br>
   Need to support K8s Scale subresource (such as `/scale`).

   **Possible Causes**<br>
   Need to implement a more standard resource replica management approach.

   **Solutions**
   Refer to the `DefaultCustomResourceHandler` implementation:
   1. Prefer using the Scale subresource (such as `/scale`).
   2. If the Scale subresource is unavailable, fall back to `spec.replicas` or `status.replicas`.
   3. Use `unstructured.Unstructured` for handling generic resources.
   4. Reference code location: `pkg/elasticscaler/resource/default_custom_resource_handler.go`.

## Appendix

This section provides reference materials and further reading links related to Elastic Scaler plugin development.

### Reference Resources

- **elastic-scaler code repository**: https://gitcode.com/openFuyao/elastic-scaler
  - `pkg/elasticscaler/scaling/`: Scaling algorithm implementation examples.
  - `pkg/elasticscaler/resource/`: Resource Handler implementation examples.
  - `pkg/elasticscaler/context/`: Context interface definitions.

- **OFEP 0030 Proposal**: https://gitcode.com/openFuyao/ofep/blob/main/ofeps/sig-ai-inference/0030-ofep-通用扩缩容决策框架设计提案.md
  - Detailed design descriptions and architecture design.
  - Interface definitions and use cases.
