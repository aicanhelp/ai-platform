# AI Inference Hermes Routing Plugin Development Guide

## Feature Introduction

Hermes-router is a Kubernetes (K8s) native AI inference intelligent routing solution, used for receiving user inference requests and forwarding them to appropriate inference service backends.

As an EPP plugin under the GIE framework, Hermes-router only focuses on the request forwarding process. This guide introduces how developers can develop custom routing strategies based on the EPP framework.

## Constraints and Limitations

**Inference Engine Limitations**

Hermes-router's inference engine support scope is consistent with GIE.
Among these, KVCache aware and latency prediction-related capabilities that depend on global KVCache information need to be used with cache-indexer. The current cache-indexer latest version is 0.22.0; these capabilities have been primarily validated in vLLM scenarios, so vLLM is recommended as the inference backend for development and integration.

Other plugins theoretically support multiple inference engines, but currently mainly vLLM as the inference engine has been validated.

**Development Limitations**

In this document, one EPP instance represents one routing strategy; a routing strategy typically consists of multiple types of plugins. Plugin development must follow the [GIE Framework Specification](https://gateway-api-inference-extension.sigs.k8s.io/).

**Deployment Limitations**

- Only supports deployment and running in Kubernetes environments.
- Plugins must be developed locally and built into images before deployment and running.
- If routing strategy configuration needs to be changed, EPP must be restarted after modifying configuration for changes to take effect.

## Environment Preparation

### Environment Requirements

#### Hardware Requirements

Hermes-router development environment has no special hardware requirements; recommended configuration is as follows.

- CPU: 4 cores or above
- Memory: 8GB or above
- Disk: 20GB or above available space

#### Software Requirements

- Operating system: Linux
- Go environment: Go 1.25 or higher version
- Docker: Docker 20.10+ or compatible container runtime (such as nerdctl)
- Kubernetes cluster: for deployment and testing
- kubectl: for interacting with K8s cluster
- Helm: 3.0+ version, for deploying Hermes-router
- If using gateway-based integration method: need an open-source gateway supporting Gateway API Inference Extension, such as Istio 1.28 or above

#### Dependency Components

- Basic dependencies
  - Inference backend service: vLLM or other inference engine services must be deployed in the cluster.
  - If using gateway-based deployment or integration method: need to install Gateway API CRDs and Inference Extension CRDs.
- Additional dependencies
  - To develop or validate KVCache aware and latency prediction-related capabilities: need to deploy cache-indexer 0.22.0 or above, for providing global KVCache awareness and hit rate calculation.

> ![note](../user_guide/figures/icon-note.gif) **Note:**<br>
> To quickly set up a complete development and integration environment including inference backend, intelligent routing, global KVCache management and other components, one-click integrated deployment can be done via InferNex; see the [InferNex User Guide](../../ai_inference_infernex/user_guide/ai_inference_infernex.md) for specific deployment methods.

### Setting Up the Environment

1. Clone the code repository.

   ```bash
   git clone https://gitcode.com/openFuyao/hermes-router.git
   cd hermes-router
   ```

2. Configure Go development environment.

   ```bash
   go version

   # Accelerate dependency download (optional)
   go env -w GOPROXY=https://goproxy.cn,direct
   ```

3. Install gateway and CRDs needed for gateway-based integration (optional).

   ```bash
   ISTIO_VERSION=1.28.0
   curl -L https://istio.io/downloadIstio | ISTIO_VERSION=${ISTIO_VERSION} sh -

   kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.4.0/standard-install.yaml

   ./istio-$ISTIO_VERSION/bin/istioctl install \
     --set values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true \
     --set values.gateways.istio-ingressgateway.type=NodePort
   ```

   To continue installing Inference Extension CRDs and Gateway entry resources, see [Installing Supporting Components](../user_guide/ai_inference_hermes_routing.md#installing-supporting-components) to complete configuration.

4. Build Hermes-router image.

   ```bash
   cd hermes-router

   docker build -t hermes-router:dev -f build/Dockerfile .

   # Or use nerdctl
   nerdctl build -t hermes-router:dev -f build/Dockerfile .
   ```

### Verifying Environment

1. Verify gateway integration dependencies are installed successfully (optional).

   ```bash
   kubectl get pods -n istio-system
   kubectl get crd inferencepools.inference.networking.x-k8s.io
   ```

2. Deploy Hermes-router in standalone mode for development verification.

   ```bash
   cd hermes-router

   # Download Helm sub-Chart dependencies needed for standalone mode
   helm dependency build ./charts/standalone

   # Deploy Hermes-router to hermes-test namespace
   # --set parameter description:
   #   createInferencePool=false: Skip auto-creating InferencePool, managed manually by user
   #   endpointSelector: Label selector for inference backend Pods, must match actual Pod labels
   #   targetPorts: Port number that inference backend service listens on
   helm install hermes-router ./charts/standalone \
     --namespace hermes-test --create-namespace \
     -f ./examples/profiles/pd-random.yaml \
     --set inferenceExtension.image.repository=hermes-router \
     --set inferenceExtension.image.tag=dev \
     --set inferenceExtension.endpointsServer.createInferencePool=false \
     --set inferenceExtension.endpointsServer.endpointSelector='openfuyao.com/model=qwen-qwen3-8b' \
     --set inferenceExtension.endpointsServer.targetPorts=8000

   # Check EPP Pod running status
   kubectl get pods -n hermes-test -l epp=hermes-router
   # View Hermes-router Service, confirm entry port (default 8081) is properly exposed
   kubectl get svc -n hermes-test hermes-router
   ```

**Expected Results:**

- EPP Pod status is Running.
- Service is created normally, default exposing HTTP entry port `8081`.
- Container logs show no obvious error messages.

## Developing Custom Routing Strategies

### Use Case Overview

When preset routing strategies cannot meet specific business needs, developers can extend routing capabilities by implementing custom plugins. Typical scenarios include:

- Business-specific routing rules: route based on business tags, user ID, and other custom attributes.
- Performance optimization needs: optimize routing algorithms for specific hardware or network environments.
- Mixed architecture support: support mixed deployment of aggregate architecture and PD disaggregation architecture.

### System Architecture

Hermes-router is implemented based on the GIE EPP framework; current plugins are divided into `Scheduling`, `RequestControl`, and `Datalayer` layers by responsibility boundary. They handle routing decisions, request processing control, and external data access respectively, coordinating through a unified plugin registration mechanism to complete processing of one inference request.

**Table 1** Hermes-router Plugin Layer Description

| Layer | Directory | Function Description | Example |
|------|------|---------|------|
| Scheduling | `pkg/epp/framework/plugins/scheduling` | Directly participates in candidate endpoint filtering, scoring, and selection. | `pd-group-filter`, `kvcache-aware-scorer`, `random-picker` |
| RequestControl | `pkg/epp/framework/plugins/requestcontrol` | Responsible for request-level data production, request lifecycle tracking, and pre-forwarding processing. | `tokenizer`, `request-lifecycle-tracker`, `pd-header-handler` |
| Datalayer | `pkg/epp/framework/plugins/datalayer` | Pulls, extracts, and exposes attributes from external systems for scheduling use. | `npu-exporter-data-source`, `npu-exporter-extractor` |

In the scheduling chain that developers most frequently interact with, the Scheduling layer can be further divided into three capability types.

**Table 2** Scheduling Layer Plugin Type Description

| Plugin Type | Function Description | Example |
|---------|---------|------|
| Filter plugin | Filters candidate endpoint list. | `pd-group-filter`: Filters invalid PD Groups.<br>`prefix-cache-filter`: Filters candidate instances by prefix cache affinity. |
| Scorer plugin | Scores candidate endpoints. | `kvcache-aware-scorer`: Scores based on KVCache-related metrics.<br>`pd-bucket-scorer`: Scores based on PD bucket load.<br>`prediction-scorer`: Scores based on prediction results. |
| Picker plugin | Selects final endpoint based on scores or strategy. | `random-picker`: Randomly selects instances.<br>`min-score-picker`: Selects the lowest-score instance.<br>`softmax-potc-picker`: Selects by Softmax probability. |

Common plugin types in RequestControl and Datalayer are shown in Table 3.

**Table 3** Request Control and Data Layer Plugin Type Description

| Plugin Type | Layer | Function Description | Example |
|---------|---------|---------|------|
| DataProducer plugin | RequestControl | Generates context data needed for scheduling for a request. | `pd-group-producer`, `prefix-cache-producer`, `prediction-feature-extractor` |
| Lifecycle plugin | RequestControl | Tracks request lifecycle, maintains in-flight request and other states. | `request-lifecycle-tracker` |
| PreRequest plugin | RequestControl | Supplements information needed by target backend before request forwarding. | `pd-header-handler` |
| Source/Extractor plugin | Datalayer | Collects data from external systems and extracts it into routing attributes. | `npu-exporter-data-source`, `npu-exporter-extractor` |

**Plugin Execution Flow**

1. **Request enters EPP**: RequestControl layer plugins generate or supplement request-level data based on request content, such as tokenization results, PD Group information, prefix cache information, and prediction features.
2. **External data access**: Datalayer layer plugins pull metrics from external data sources as needed and extract them into attributes available for scheduling.
3. **Scheduling decision**: Scheduling layer plugins execute filtering, scoring, and selection in the order defined in `schedulingProfiles`, obtaining the final target endpoint.
4. **Pre-forwarding processing**: PreRequest plugins write request headers or routing metadata before request forwarding, then send the request to the target inference backend.

The following links provide documentation needed for developing custom routing strategies:

- [hermes-router Code Repository](https://gitcode.com/openFuyao/hermes-router): Hermes-router source code repository, containing plugin implementation examples and development framework; developers can reference existing plugin code for development.
- [GIE Framework Specification](https://gateway-api-inference-extension.sigs.k8s.io/): Gateway API Inference Extension (GIE) official specification documentation, defining EPP plugin interface specifications and development standards.
- [EPP Architecture Proposal](https://github.com/kubernetes-sigs/gateway-api-inference-extension/tree/main/docs/proposals/0683-epp-architecture-proposal): Endpoint Picker Plugin (EPP) architecture proposal, detailing EPP plugin architecture design, execution flow, and extension mechanisms.

### Development Steps

1. Design routing strategy.

   Developers need to first clarify routing strategy objectives and decompose plugin responsibilities that fall into Scheduling, RequestControl, or Datalayer; this guide omits strategy design itself.

2. Develop routing plugins.

   After completing routing strategy design, developers need to decompose logic into several EPP-spec plugins and implement them based on processing responsibilities. Below, a Filter plugin for filtering candidate endpoints is used as an example to demonstrate the current version EPP plugin development flow.

   2.1. Create Filter code file.

   It is recommended to create a new Filter plugin under `pkg/epp/framework/plugins/scheduling/filter/<plugin-name>/plugin.go` and add `plugin_test.go` in the same directory.

   2.2. Define Filter plugin type constant and struct.

   ```go
   const PluginType = "my-filter"

   type Filter struct {
      typedName fwkplugin.TypedName
   }

   var _ fwkscheduling.Filter = (*Filter)(nil)
   ```

   2.3. Implement struct constructor and `TypedName()` method.

   ```go
   func New(name string) *Filter {
      return &Filter{typedName: fwkplugin.TypedName{Type: PluginType, Name: name}}
   }

   func (f *Filter) TypedName() fwkplugin.TypedName {
      return f.typedName
   }
   ```

   2.4. Implement factory function.

   ```go
   func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
      if err := parseParameters(raw); err != nil {
         return nil, fmt.Errorf("my filter %q: %w", name, err)
      }
      return New(name), nil
   }
   ```

   2.5. Implement `Filter()` method in the `Filter` interface.

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

   2.6. Register the new plugin in `pkg/epp/register.go`.

   ```go
   func RegisterAllPlugins() {
      fwkplugin.Register(myfilter.PluginType, myfilter.Factory)
      // ... other plugins
   }
   ```

   2.7. Supplement parameter parsing and unit tests.

   Current repository plugins typically parse parameters via `json.RawMessage` and use `plugin_test.go` to validate parameters, filtering results, or scoring logic. Developers can reference `pd-group-filter`, `prefix-cache-filter` and other existing implementations to organize test cases.

   A Filter plugin is now complete. Developers can develop Scorer, Picker, PreRequest, Lifecycle, or Datalayer plugins following the same pattern, replacing only the corresponding directory and interface.

3. Define routing strategy.

   In the current version, developers can connect routing strategies via the following two methods.

   - Reuse built-in routing capabilities: Select Hermes-router built-in scheduling chains via `inferenceExtension.pluginsConfigFile: default-plugins.yaml` and `inferenceExtension.routing`.
   - Orchestrate custom plugin chains: Specify a custom configuration file name via `inferenceExtension.pluginsConfigFile` and provide `EndpointPickerConfig` content via `inferenceExtension.pluginsCustomConfig`.

   Built-in routing capability configuration example:

   ```yaml
   inferenceExtension:
     pluginsConfigFile: default-plugins.yaml
     routing:
       deploymentMode: aggregate
       profile: random
   ```

   Custom plugin chain configuration example:

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

   In custom configuration, developers can further supplement `dataLayer`, multiple `schedulingProfiles`, or other plugin parameter configurations as needed.

   Routing strategy development is now complete.

### Testing and Verification

After completing plugin development, it is recommended to first perform quick verification via standalone mode; if gateway integration effects need to be verified, then switch to Gateway mode integration.

1. Standalone mode quick verification.

    In standalone mode, users can directly access Pods with `epp=<RELEASE_NAME>` label. The envoy sidecar in that Pod receives inference requests on port `8081` and forwards them to EPP for routing. Below, release name `hermes-router` is used as an example.

    ```shell
    RELEASE_NAME=hermes-router
    EPP_POD_IP=$(kubectl get pods -n <NAMESPACE> -l epp=${RELEASE_NAME} -o jsonpath='{.items[0].status.podIP}')
    ```

    Send inference request.

    ```shell
    curl -X POST http://${EPP_POD_IP}:8081/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-8B",
        "messages": [
          {"role": "user", "content": "hello"}
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": false
      }'
    ```

2. Optional: Gateway-based integration verification.

    ```shell
    kubectl get svc -n <NAMESPACE> -l gateway.networking.k8s.io/gateway-name=inference-gateway
    kubectl port-forward -n <NAMESPACE> service/<gateway-service-name> 8000:80
    ```

    Send inference request in another terminal.

    ```shell
    curl -X POST http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
       "model": "Qwen/Qwen3-8B",
       "messages": [
          {"role": "user", "content": "hello"}
       ],
       "max_tokens": 100,
       "temperature": 0.7,
       "stream": false
    }'
    ```

3. If requests are not routed as expected, check the following.

    - Whether custom plugin is registered in `pkg/epp/register.go`.
    - Whether `pluginsConfigFile` points to the correct configuration file.
    - Whether plugin reference order in `schedulingProfiles` matches expectations.
    - Whether EPP Pod logs contain parameter parsing, plugin initialization, or forwarding-related errors.

After development is complete, refer to the **Installation** section in the [User Guide](https://gitcode.com/openFuyao/sig-ai-inference/blob/main/docs/zh/ai_inference_hermes_routing/user_guide/ai_inference_hermes_routing.md) for deployment and verification.