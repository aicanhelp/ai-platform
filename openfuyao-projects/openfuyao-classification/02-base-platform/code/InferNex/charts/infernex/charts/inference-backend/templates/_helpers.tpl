{{/*
openFuyao standard image reference helper.
Usage: {{ list . "image-key" | include "helpers.image.name" }}
*/}}
{{- define "helpers.image.name" -}}
{{- $ctx := index . 0 -}}
{{- $image := index . 1 | get $ctx.Values.images -}}
{{- $image.repository }}:{{ $image.tag | default $ctx.Chart.AppVersion }}{{ $image.digest | default "" | empty | ternary "" (print "@sha256:" $image.digest) }}
{{- end -}}

{{/*
Image with explicit tag from values (third-party / pinned versions, not Chart.AppVersion).
Usage: {{ list . "redis" | include "helpers.image.name.pinned" }}
*/}}
{{- define "helpers.image.name.pinned" -}}
{{- $ctx := index . 0 -}}
{{- $key := index . 1 -}}
{{- $image := $key | get $ctx.Values.images -}}
{{- $tag := required (printf "images.%s.tag is required" $key) $image.tag -}}
{{- $image.repository }}:{{ $tag }}{{ $image.digest | default "" | empty | ternary "" (print "@sha256:" $image.digest) }}
{{- end -}}

{{/*
vLLM chart name
*/}}
{{- define "vllm.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
vLLM chart
*/}}
{{- define "vllm.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
vLLM instance inference service name
*/}}
{{- define "vllm.instance.deploymentName" -}}
{{- .service.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
vLLM instance service name
*/}}
{{- define "vllm.instance.serviceName" -}}
{{- .service.name | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
LWS spec.replicas: reads node.replicas when set (including 0); otherwise dict "default" (defaults to 1).
Pass dict: node (e.g. .pd.prefill), optional "default".
*/}}
{{- define "vllm.lwsReplicas" -}}
{{- $node := index . "node" -}}
{{- $def := index . "default" | default 1 -}}
{{- if hasKey $node "replicas" -}}
{{- index $node "replicas" -}}
{{- else -}}
{{- $def -}}
{{- end -}}
{{- end }}

{{/*
Common labels for a vllm service instance
Used on workload metadata (e.g. LeaderWorkerSet, Deployment) and related Service / Pod template labels.
*/}}
{{- define "vllm.instance.labels" -}}
helm.sh/chart: {{ include "vllm.chart" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
app.kubernetes.io/instance: {{ printf "%s-%s" .Release.Name .service.name }}
app.kubernetes.io/name: {{ include "vllm.name" . }}
{{- end }}

{{/*
Selector labels for a vllm service instance
Used where stable selectors are needed (e.g. Service spec.selector, LWS/worker Pod template labels, proxy Deployment).
*/}}
{{- define "vllm.instance.selectorLabels" -}}
app.kubernetes.io/name: {{ include "vllm.name" . }}
app.kubernetes.io/instance: {{ printf "%s-%s" .Release.Name .service.name }}
{{- end }}

{{/*
Service labels
*/}}
{{- define "vllm.service.labels" -}}
{{- include "vllm.instance.labels" . }}
app.kubernetes.io/component: service
{{- end }}

{{/*
Get complete vllm inference engine image.
Third-party engine images (e.g. vllm-ascend) use an explicit tag from values, not Chart.AppVersion.
*/}}
{{- define "vllm.image" -}}
{{- list . "inferenceEngine" | include "helpers.image.name.pinned" -}}
{{- end -}}

{{/*
Get complete proxy server image.
*/}}
{{- define "vllm.proxy.image" -}}
{{- list . "proxyServer" | include "helpers.image.name" -}}
{{- end -}}

{{/*
Merged pd.proxyServer config. The proxy and its access logs stay enabled by
default. Endpoint filtering follows accessLog.excludedEndpoints from values.
Pass dict: proxyServer (optional .pd.proxyServer map).
*/}}
{{- define "vllm.pdProxyServerConfig" -}}
{{- $accessLogDefaults := dict "enabled" true "excludedEndpoints" (list) -}}
{{- $defaults := dict "enabled" true "discoveryInterval" 10 "accessLog" $accessLogDefaults -}}
{{- if .proxyServer -}}
{{- mergeOverwrite $defaults .proxyServer | toJson -}}
{{- else -}}
{{- $defaults | toJson -}}
{{- end -}}
{{- end -}}

{{/*
Get complete model downloader image.
*/}}
{{- define "vllm.modelDownloader.image" -}}
{{- list . "modelDownloader" | include "helpers.image.name" -}}
{{- end -}}

{{/*
Get redis image (pinned third-party version).
*/}}
{{- define "vllm.redis.image" -}}
{{- list . "redis" | include "helpers.image.name.pinned" -}}
{{- end -}}

{{/*
Get yaml/json tool initContainer image (pinned third-party version).
*/}}
{{- define "vllm.yamlTool.image" -}}
{{- list . "yamlTool" | include "helpers.image.name.pinned" -}}
{{- end -}}

{{/*
Get image pull policy
*/}}
{{- define "vllm.image.pullPolicy" -}}
{{- if and .Values.global.image .Values.global.image.pullPolicy }}
{{- .Values.global.image.pullPolicy }}
{{- else }}
{{- "IfNotPresent" }}
{{- end }}
{{- end }}

{{/*
Get model name for labels (formatted)
*/}}
{{- define "vllm.modelName.label" -}}
{{- $modelName := "" }}
{{- if and .Values.global .Values.global.modelName }}
{{- $modelName = .Values.global.modelName }}
{{- end }}
{{- if $modelName }}
{{- $modelName | replace "/" "-" | replace "." "-" | lower }}
{{- else }}
{{- "" }}
{{- end }}
{{- end }}

{{/*
Calculate resources for vLLM inference node
Merges service-level and node-level resources, with node-level overriding service-level
Adds NPU card resource from tensorParallelSize * pipelineParallelSize * dataParallelSizeLocal (unless cardCount set).
*/}}
{{- define "vllm.node.resources" -}}
{{- $baseResources := .baseResources | default (dict "requests" dict "limits" dict) }}
{{- $nodeResources := .nodeResources }}
{{- $tensorParallelSize := .tensorParallelSize | default 1 | int }}
{{- $pipelineParallelSize := .pipelineParallelSize | default 1 | int }}
{{- $dataParallelSizeLocal := .dataParallelSizeLocal | default 1 | int }}
{{- /* cardCount may be 0 (no extended resource); explicit 0 must not fall through to the formula. */ -}}
{{- $cardCount := 0 }}
{{- /*
Semantics:
- If cardCount is not configured (invalid / no value), fall back to tp*dp*pp.
- If cardCount is configured (including 0), it will be used as-is.
*/}}
{{- if hasKey . "cardCount" }}
{{- $cardCount = index . "cardCount" | int }}
{{- else }}
{{- $cardCount = mul $tensorParallelSize (mul $pipelineParallelSize $dataParallelSizeLocal) }}
{{- end }}
{{- $cardResourceName := "huawei.com/Ascend910" }}
{{- if and .Values .Values.inferenceDevice }}
{{- $cardResourceName = .Values.inferenceDevice }}
{{- end }}
{{- $baseRequests := $baseResources.requests | default dict }}
{{- $baseLimits := $baseResources.limits | default dict }}
{{- /* Clone base maps to avoid cross-node mutation (prefill -> decode). */}}
{{- $requests := dict }}
{{- $limits := dict }}
{{- range $key, $value := $baseRequests }}
{{- $requests = set $requests $key $value }}
{{- end }}
{{- range $key, $value := $baseLimits }}
{{- $limits = set $limits $key $value }}
{{- end }}
{{- if $nodeResources }}
{{- $nodeRequests := $nodeResources.requests | default dict }}
{{- $nodeLimits := $nodeResources.limits | default dict }}
{{- range $key, $value := $nodeRequests }}
{{- $requests = set $requests $key $value }}
{{- end }}
{{- range $key, $value := $nodeLimits }}
{{- $limits = set $limits $key $value }}
{{- end }}
{{- end }}
{{- if gt $cardCount 0 }}
{{- $requests = set $requests $cardResourceName (toString $cardCount) }}
{{- $limits = set $limits $cardResourceName (toString $cardCount) }}
{{- end }}
{{- $result := dict "requests" $requests "limits" $limits }}
{{- $result | toYaml }}
{{- end }}

{{/*
Calculate resources for prefill node
*/}}
{{- define "vllm.prefill.resources" -}}
{{- $baseResources := .service.resources | default (dict "requests" dict "limits" dict) }}
{{- $opts := dict "baseResources" $baseResources "nodeResources" .service.pd.prefill.resources "tensorParallelSize" .service.pd.prefill.tensorParallelSize "dataParallelSizeLocal" (.service.pd.prefill.dataParallelSizeLocal | default 1) "pipelineParallelSize" (.service.pd.prefill.pipelineParallelSize | default 1) "Values" .Values }}
{{- if hasKey .service.pd.prefill "cardCount" }}
{{- $_ := set $opts "cardCount" .service.pd.prefill.cardCount }}
{{- end }}
{{- include "vllm.node.resources" $opts }}
{{- end }}

{{/*
Calculate resources for decode node
*/}}
{{- define "vllm.decode.resources" -}}
{{- $baseResources := .service.resources | default (dict "requests" dict "limits" dict) }}
{{- $opts := dict "baseResources" $baseResources "nodeResources" .service.pd.decode.resources "tensorParallelSize" .service.pd.decode.tensorParallelSize "dataParallelSizeLocal" (.service.pd.decode.dataParallelSizeLocal | default 1) "pipelineParallelSize" (.service.pd.decode.pipelineParallelSize | default 1) "Values" .Values }}
{{- if hasKey .service.pd.decode "cardCount" }}
{{- $_ := set $opts "cardCount" .service.pd.decode.cardCount }}
{{- end }}
{{- include "vllm.node.resources" $opts }}
{{- end }}

{{/*
Calculate resources for aggregated node
*/}}
{{- define "vllm.aggregated.resources" -}}
{{- $baseResources := .service.resources | default (dict "requests" dict "limits" dict) }}
{{- $opts := dict "baseResources" $baseResources "nodeResources" .service.aggregated.resources "tensorParallelSize" .service.aggregated.tensorParallelSize "dataParallelSizeLocal" (.service.aggregated.dataParallelSizeLocal | default 1) "pipelineParallelSize" (.service.aggregated.pipelineParallelSize | default 1) "Values" .Values }}
{{- if hasKey .service.aggregated "cardCount" }}
{{- $_ := set $opts "cardCount" .service.aggregated.cardCount }}
{{- end }}
{{- include "vllm.node.resources" $opts }}
{{- end }}

{{/*
Merge base environment variables (global, inference-backend, and service-level)
These are shared across all pods in the service
*/}}
{{- define "vllm.baseEnv" -}}
{{- $globalEnv := .Values.global.env | default list }}
{{- $inferenceBackendEnv := .Values.env | default list }}
{{- $serviceEnv := .service.env | default list }}
{{- if and .service.kvTransferConfig .service.kvTransferConfig.mooncake }}
{{- $configPath := .service.kvTransferConfig.mooncake.configPath | default "/app/mooncake.json" }}
{{- $mooncakeEnv := list (dict "name" "MOONCAKE_CONFIG_PATH" "value" $configPath) }}
{{- $serviceEnv = concat $serviceEnv $mooncakeEnv }}
{{- end }}
{{- concat $globalEnv $inferenceBackendEnv $serviceEnv | toYaml }}
{{- end }}

{{/*
Pod metadata environment variables (POD_NAME, POD_IP)
These are automatically injected by Kubernetes for all pods
*/}}
{{- define "vllm.podEnv" -}}
- name: POD_NAME
  valueFrom:
    fieldRef:
      fieldPath: metadata.name
- name: POD_IP
  valueFrom:
    fieldRef:
      fieldPath: status.podIP
{{- end }}

{{/*
Node-specific environment variables (for prefill/decode/aggregated nodes)
These are specific to each node type and defined in values.yaml under pd.prefill.env, pd.decode.env, or aggregated.env
*/}}
{{- define "vllm.nodeEnv" -}}
{{- $nodeEnv := .nodeEnv | default list }}
{{- if $nodeEnv }}
{{- $nodeEnv | toYaml }}
{{- end }}
{{- end }}

{{/*
Get base volumeMounts from values.yaml (inference-backend and service-level)
Merges ascend, model (auto-generated from global.cachePath), and service-level volumeMounts, with enable switch for ascend
*/}}
{{- define "vllm.baseMounts" -}}
{{- $ascendMounts := list }}
{{- if and .Values.volumeMounts.ascend .Values.volumeMounts.ascend.enable }}
{{- $ascendMounts = .Values.volumeMounts.ascend.mounts | default list }}
{{- end }}
{{- $modelMounts := list (dict "name" "rootcache" "mountPath" "/root/.cache") }}
{{- $serviceMounts := .service.volumeMounts | default list }}
{{- $allMounts := concat $ascendMounts $modelMounts $serviceMounts }}
{{- if $allMounts }}
{{- $allMounts | toYaml }}
{{- end }}
{{- end }}

{{/*
Get base volumes from values.yaml (inference-backend and service-level)
Merges ascend, model (auto-generated from global.cachePath), and service-level volumes, with enable switch for ascend
*/}}
{{- define "vllm.baseVolumes" -}}
{{- $ascendVolumes := list }}
{{- if and .Values.volumes.ascend .Values.volumes.ascend.enable }}
{{- $ascendVolumes = .Values.volumes.ascend.mounts | default list }}
{{- end }}
{{- $modelVolumes := list (dict "name" "rootcache" "hostPath" (dict "path" .Values.global.cachePath)) }}
{{- $serviceVolumes := .service.volumes | default list }}
{{- $allVolumes := concat $ascendVolumes $modelVolumes $serviceVolumes }}
{{- if $allVolumes }}
{{- $allVolumes | toYaml }}
{{- end }}
{{- end }}

{{/*
Get inference role label value from inferenceRoleLabels config
*/}}
{{- define "vllm.pdRole.label" -}}
{{- $role := .role | default "" }}
{{- if and $.Values.inferenceRoleLabels (index $.Values.inferenceRoleLabels $role) }}
{{- index $.Values.inferenceRoleLabels $role }}
{{- else }}
{{- $role }}
{{- end }}
{{- end }}

{{/*
Check if any service has mooncake store enabled
Returns "true" if at least one enabled service has kvTransferConfig.mooncake.use_store set to true, otherwise returns empty string
*/}}
{{- define "vllm.hasMooncake" -}}
{{- $hasMooncake := false }}
{{- range .Values.services }}
{{- if and .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store }}
{{- $hasMooncake = true }}
{{- end }}
{{- end }}
{{- if $hasMooncake }}
{{- "true" }}
{{- end }}
{{- end }}

{{/*
Whether to create the shared Mooncake Master Deployment and Service.
The setting is taken from the first enabled service using Mooncake store.
master.create defaults to true to preserve the existing behavior. Set it to
false to use an externally managed Mooncake Master; configure that master's
endpoint in kvTransferConfig.mooncake.config.master_server_address.
*/}}
{{- define "vllm.mooncake.master.create" -}}
{{- $picked := false -}}
{{- $create := true -}}
{{- range .Values.services -}}
{{- if and (not $picked) .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store -}}
{{- $picked = true -}}
{{- if and .kvTransferConfig.mooncake.master (hasKey .kvTransferConfig.mooncake.master "create") -}}
{{- $create = .kvTransferConfig.mooncake.master.create -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if and (not $picked) .Values.mooncakeMaster (hasKey .Values.mooncakeMaster "create") -}}
{{- $create = .Values.mooncakeMaster.create -}}
{{- end -}}
{{- if $create -}}
true
{{- end -}}
{{- end }}

{{/*
Whether to create the shared Mooncake metadata server (Redis) Deployment and
Service. The setting is taken from the first enabled service using Mooncake
store. metadataServer.create defaults to true to preserve the existing
behavior. Set it to false to use an externally managed Redis instance; configure
its endpoint in kvTransferConfig.mooncake.config.metadata_server.
*/}}
{{- define "vllm.mooncake.metadataServer.create" -}}
{{- $picked := false -}}
{{- $create := true -}}
{{- range .Values.services -}}
{{- if and (not $picked) .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store -}}
{{- $picked = true -}}
{{- if and .kvTransferConfig.mooncake.metadataServer (hasKey .kvTransferConfig.mooncake.metadataServer "create") -}}
{{- $create = .kvTransferConfig.mooncake.metadataServer.create -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $create -}}
true
{{- end -}}
{{- end }}

{{/*
Get mooncake metadata server (Redis) resources from the first enabled service
with kvTransferConfig.mooncake.use_store and metadataServer.resources defined.
The Redis Deployment is shared per release; only one resource profile is applied.
*/}}
{{- define "vllm.mooncake.metadataServer.resources" -}}
{{- $picked := false -}}
{{- $resources := dict -}}
{{- range .Values.services -}}
{{- if and (not $picked) .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store -}}
{{- if and .kvTransferConfig.mooncake.metadataServer .kvTransferConfig.mooncake.metadataServer.resources -}}
{{- $picked = true -}}
{{- $resources = .kvTransferConfig.mooncake.metadataServer.resources -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if $picked -}}
{{- toYaml $resources -}}
{{- end -}}
{{- end }}

{{/*
Get mooncake master shared config from the first enabled service
with kvTransferConfig.mooncake.use_store and master defined.
Falls back to legacy top-level .Values.mooncakeMaster if present.
*/}}
{{- define "vllm.mooncake.master.config" -}}
{{- $picked := false -}}
{{- $master := dict -}}
{{- range .Values.services -}}
{{- if and (not $picked) .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store .kvTransferConfig.mooncake.master -}}
{{- $picked = true -}}
{{- $master = .kvTransferConfig.mooncake.master -}}
{{- end -}}
{{- end -}}
{{- if $picked -}}
{{- toYaml $master -}}
{{- else if .Values.mooncakeMaster -}}
{{- toYaml .Values.mooncakeMaster -}}
{{- end -}}
{{- end }}

{{/*
Get mooncake master resources from the first enabled service
with kvTransferConfig.mooncake.use_store and master.resources defined.
The master Deployment is shared per release; only one resource profile is applied.
*/}}
{{- define "vllm.mooncake.master.resources" -}}
{{- $picked := false -}}
{{- $resources := dict -}}
{{- range .Values.services -}}
{{- if and (not $picked) .enabled .kvTransferConfig .kvTransferConfig.mooncake .kvTransferConfig.mooncake.use_store -}}
{{- if and .kvTransferConfig.mooncake.master .kvTransferConfig.mooncake.master.resources -}}
{{- $picked = true -}}
{{- $resources = .kvTransferConfig.mooncake.master.resources -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- if not $picked -}}
{{- $resources = dict "requests" (dict "cpu" "8") "limits" (dict "cpu" "8") -}}
{{- end -}}
{{- toYaml $resources -}}
{{- end }}

{{/*
Build kv-transfer-config JSON string
Supports two modes:
1. If connectorConfigJson is provided, use it directly (full JSON string, user has full control)
2. If connectorConfig is provided, convert YAML to JSON and auto-fill node-specific fields

Parameters:
  - kvTransferConfig: the kvTransferConfig object from values
  - service: the service object
  - nodeType: "prefill", "decode", or "aggregated"
*/}}
{{- define "vllm.kvTransferConfig.json" -}}
{{- $kvTransferConfig := .kvTransferConfig }}
{{- $service := .service }}
{{- $nodeType := .nodeType }}
{{- /* Calculate kv_role based on nodeType */}}
{{- $kvRole := "kv_producer" }}
{{- if eq $nodeType "decode" }}
{{- $kvRole = "kv_consumer" }}
{{- else if eq $nodeType "aggregated" }}
{{- $kvRole = "kv_both" }}
{{- end }}
{{- /* Calculate kv_rank based on nodeType */}}
{{- $kvRank := 0 }}
{{- if eq $nodeType "decode" }}
{{- $kvRank = 1 }}
{{- end }}

{{- if $kvTransferConfig.connectorConfigJson }}
{{- /* Mode 1: User provided full JSON string, use it directly */}}
{{- $kvTransferConfig.connectorConfigJson }}
{{- else if $kvTransferConfig.connectorConfig }}
{{- /* Mode 2: User provided YAML config, convert to JSON and add node-specific fields */}}
{{- $config := $kvTransferConfig.connectorConfig }}
{{- /* Create a new dict to ensure we can modify it */}}
{{- $finalConfig := dict }}
{{- /* Copy existing config fields */}}
{{- range $key, $value := $config }}
{{- $finalConfig = set $finalConfig $key $value }}
{{- end }}
{{- /* Add node-specific fields if not already present */}}
{{- if not $finalConfig.kv_role }}
{{- $finalConfig = set $finalConfig "kv_role" $kvRole }}
{{- end }}
{{- if not $finalConfig.engine_id }}
{{- $finalConfig = set $finalConfig "engine_id" "'$POD_NAME'" }}
{{- end }}
{{- /* Add kv_rank only for prefill and decode, not for aggregated */}}
{{- if and (ne $nodeType "aggregated") (not $finalConfig.kv_rank) }}
{{- $finalConfig = set $finalConfig "kv_rank" $kvRank }}
{{- end }}
{{- /* Process kv_connector_extra_config */}}
{{- $extraConfig := $finalConfig.kv_connector_extra_config | default dict }}
{{- /* Create a new dict to hold the processed extra config */}}
{{- $processedExtraConfig := dict }}
{{- /* Copy all existing fields from extraConfig */}}
{{- range $key, $value := $extraConfig }}
{{- if ne $key "connectors" }}
{{- $processedExtraConfig = set $processedExtraConfig $key $value }}
{{- end }}
{{- end }}
{{- /* Calculate prefill/decode sizes for kv_connector_extra_config (tp/dp only; pp not injected here yet). */}}
{{- $prefillTpSize := 1 }}
{{- $prefillDpSize := 1 }}
{{- $decodeTpSize := 1 }}
{{- $decodeDpSize := 1 }}
{{- if eq $service.mode "pd" }}
{{- $prefillTpSize = $service.pd.prefill.tensorParallelSize | default 1 | int }}
{{- $prefillDpSize = $service.pd.prefill.dataParallelSize | default 1 | int }}
{{- $decodeTpSize = $service.pd.decode.tensorParallelSize | default 1 | int }}
{{- $decodeDpSize = $service.pd.decode.dataParallelSize | default 1 | int }}
{{- else if eq $service.mode "aggregated" }}
{{- $prefillTpSize = $service.aggregated.tensorParallelSize | default 1 | int }}
{{- $prefillDpSize = $service.aggregated.dataParallelSize | default 1 | int }}
{{- $decodeTpSize = $service.aggregated.tensorParallelSize | default 1 | int }}
{{- $decodeDpSize = $service.aggregated.dataParallelSize | default 1 | int }}
{{- end }}
{{- $prefillConfig := dict "tp_size" $prefillTpSize "dp_size" $prefillDpSize }}
{{- $decodeConfig := dict "tp_size" $decodeTpSize "dp_size" $decodeDpSize }}
{{- /* Calculate base port based on nodeType. Base 28000 keeps the zmq control-plane port
      outside the mooncake transfer engine port range (20000-27000) to avoid conflicts. */}}
{{- $basePort := 28000 }}
{{- if eq $nodeType "decode" }}
{{- $basePort = 28001 }}
{{- end }}
{{- /* If kv_connector_extra_config has connectors (expect to be MultiConnector), add default kv_port, kv_role and prefill/decode to each connector */}}
{{- if $extraConfig.connectors }}
{{- $connectors := list }}
{{- range $connector := $extraConfig.connectors }}
{{- $newConnector := dict }}
{{- /* Copy all fields from connector */}}
{{- range $ckey, $cvalue := $connector }}
{{- $newConnector = set $newConnector $ckey $cvalue }}
{{- end }}
{{- /* Add default kv_role for each connector if not present (preserve user-configured kv_role) */}}
{{- if not $newConnector.kv_role }}
{{- $newConnector = set $newConnector "kv_role" $kvRole }}
{{- end }}
{{- /* Add default kv_port if not present (preserve user-configured ports) */}}
{{- if not $newConnector.kv_port }}
{{- $newConnector = set $newConnector "kv_port" (printf "%d" $basePort) }}
{{- end }}
{{- /* Process connector's kv_connector_extra_config */}}
{{- $connectorExtraConfig := $newConnector.kv_connector_extra_config | default dict }}
{{- /* Add prefill/decode sizes to connector's extra_config if not present */}}
{{- if not $connectorExtraConfig.prefill }}
{{- $connectorExtraConfig = set $connectorExtraConfig "prefill" $prefillConfig }}
{{- end }}
{{- if not $connectorExtraConfig.decode }}
{{- $connectorExtraConfig = set $connectorExtraConfig "decode" $decodeConfig }}
{{- end }}
{{- $newConnector = set $newConnector "kv_connector_extra_config" $connectorExtraConfig }}
{{- $connectors = append $connectors $newConnector }}
{{- end }}
{{- $processedExtraConfig = set $processedExtraConfig "connectors" $connectors }}
{{- /* For MultiConnector, also add prefill/decode to top-level for compatibility */}}
{{- if not $processedExtraConfig.prefill }}
{{- $processedExtraConfig = set $processedExtraConfig "prefill" $prefillConfig }}
{{- end }}
{{- if not $processedExtraConfig.decode }}
{{- $processedExtraConfig = set $processedExtraConfig "decode" $decodeConfig }}
{{- end }}
{{- /* For MultiConnector, add kv_port to top-level if not present (always use default port) */}}
{{- if not $finalConfig.kv_port }}
{{- $finalConfig = set $finalConfig "kv_port" (printf "%d" $basePort) }}
{{- end }}
{{- else }}
{{- /* Single connector mode: add prefill/decode sizes to top-level extra_config if not present */}}
{{- if not $processedExtraConfig.prefill }}
{{- $processedExtraConfig = set $processedExtraConfig "prefill" $prefillConfig }}
{{- end }}
{{- if not $processedExtraConfig.decode }}
{{- $processedExtraConfig = set $processedExtraConfig "decode" $decodeConfig }}
{{- end }}
{{- /* For single connector, add kv_port if not present in top-level config (use default port) */}}
{{- if not $finalConfig.kv_port }}
{{- $finalConfig = set $finalConfig "kv_port" (printf "%d" $basePort) }}
{{- end }}
{{- end }}
{{- $finalConfig = set $finalConfig "kv_connector_extra_config" $processedExtraConfig }}
{{- $finalConfig | toJson }}
{{- end }}
{{- end }}

{{/*
LWS / multi-DP: shell fragment setting DP_PARALLEL_FLAGS (Helm-expanded dpSize / dpLocal / rpcPort).
When dpSize>1 only: sets START_RANK then DP_PARALLEL_FLAGS with hybrid LB flags; backslash-newlines inside
"..." are for readable generated YAML only (bash still produces one flag string).
Dict keys: dpSize, dpLocal, rpcPort (same semantics as the role's dataParallelRpcPort; may be empty).
*/}}
{{- define "vllm.dpParallelConfig" -}}
{{- $ds := .dpSize | int -}}
{{- $dl := .dpLocal | int -}}
{{- $rpc := .rpcPort | default "" | toString -}}
{{- if gt $ds 1 }}
START_RANK=$(( ${LWS_WORKER_INDEX:-0} * {{ $dl }} ))
DP_PARALLEL_FLAGS="--data-parallel-size {{ $ds }} \
 --data-parallel-size-local {{ $dl }} \
 --data-parallel-start-rank ${START_RANK} \
 --data-parallel-address ${LWS_LEADER_ADDRESS} \
 --data-parallel-rpc-port {{ $rpc }} \
 --data-parallel-hybrid-lb"
{{- else }}
DP_PARALLEL_FLAGS="--data-parallel-size {{ $ds }}"
{{- end }}
{{- end }}
