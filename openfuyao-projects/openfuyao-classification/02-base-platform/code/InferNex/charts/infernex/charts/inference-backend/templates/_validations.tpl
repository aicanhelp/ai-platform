{{/*
Validate required configuration values
*/}}
{{- define "vllm.validate" -}}
{{- $errors := list }}

{{/* Validate global configuration */}}
{{- if not .Values.global }}
{{- $errors = append $errors "global is required" }}
{{- else }}
{{- if not .Values.global.modelName }}
{{- $errors = append $errors "global.modelName is required" }}
{{- end }}
{{- if not .Values.global.cachePath }}
{{- $errors = append $errors "global.cachePath is required" }}
{{- end }}
{{- end }}

{{/* Validate inference engine image configuration */}}
{{- if not .Values.images.inferenceEngine }}
{{- $errors = append $errors "images.inferenceEngine is required" }}
{{- else }}
{{- if not .Values.images.inferenceEngine.repository }}
{{- $errors = append $errors "images.inferenceEngine.repository is required" }}
{{- end }}
{{- if not .Values.images.inferenceEngine.tag }}
{{- $errors = append $errors "images.inferenceEngine.tag is required" }}
{{- end }}
{{- end }}

{{/* Validate proxy server image configuration */}}
{{- if not .Values.images.proxyServer }}
{{- $errors = append $errors "images.proxyServer is required" }}
{{- else }}
{{- if not .Values.images.proxyServer.repository }}
{{- $errors = append $errors "images.proxyServer.repository is required" }}
{{- end }}
{{- end }}

{{/* Validate redis image configuration */}}
{{- if .Values.images.redis }}
{{- if not .Values.images.redis.repository }}
{{- $errors = append $errors "images.redis.repository is required when images.redis is defined" }}
{{- end }}
{{- if not .Values.images.redis.tag }}
{{- $errors = append $errors "images.redis.tag is required when images.redis is defined" }}
{{- end }}
{{- end }}

{{/* Validate yaml tool image configuration */}}
{{- if not .Values.images.yamlTool }}
{{- $errors = append $errors "images.yamlTool is required" }}
{{- else }}
{{- if not .Values.images.yamlTool.repository }}
{{- $errors = append $errors "images.yamlTool.repository is required" }}
{{- end }}
{{- if not .Values.images.yamlTool.tag }}
{{- $errors = append $errors "images.yamlTool.tag is required" }}
{{- end }}
{{- end }}

{{/* Validate services configuration */}}
{{- $hasEnabledService := false }}
{{- range .Values.services }}
{{- if .enabled }}
{{- $hasEnabledService = true }}
{{- end }}
{{/* Validate service mode must be 'pd' or 'aggregated' */}}
{{- if .mode }}
{{- if and (ne .mode "pd") (ne .mode "aggregated") }}
{{- $errors = append $errors (printf "services[%s].mode must be 'pd' or 'aggregated', got '%s'" (.name | default "unknown") .mode) }}
{{- end }}
{{- else }}
{{- $errors = append $errors (printf "services[%s].mode is required" (.name | default "unknown")) }}
{{- end }}
{{/* kvTransferConfig is required for pd services */}}
{{- if and .enabled (eq .mode "pd") (not .kvTransferConfig) }}
{{- $errors = append $errors (printf "services[%s].kvTransferConfig is required when mode is 'pd'" (.name | default "unknown")) }}
{{- end }}
{{/* dataParallelSize must be divisible by dataParallelSizeLocal (default 1 = one global rank span per Pod). */}}
{{- if and .enabled (eq .mode "pd") }}
{{- $dps := int (.pd.prefill.dataParallelSize | default 1) }}
{{- $dpl := int (.pd.prefill.dataParallelSizeLocal | default 1) }}
{{- if lt $dpl 1 }}
{{- $errors = append $errors (printf "services[%s].pd.prefill.dataParallelSizeLocal must be >= 1" (.name | default "unknown")) }}
{{- else if and (gt $dps 0) (ne (mod $dps $dpl) 0) }}
{{- $errors = append $errors (printf "services[%s].pd.prefill.dataParallelSize (%d) must be divisible by dataParallelSizeLocal (%d)" (.name | default "unknown") $dps $dpl) }}
{{- end }}
{{- $dpsd := int (.pd.decode.dataParallelSize | default 1) }}
{{- $dpld := int (.pd.decode.dataParallelSizeLocal | default 1) }}
{{- if lt $dpld 1 }}
{{- $errors = append $errors (printf "services[%s].pd.decode.dataParallelSizeLocal must be >= 1" (.name | default "unknown")) }}
{{- else if and (gt $dpsd 0) (ne (mod $dpsd $dpld) 0) }}
{{- $errors = append $errors (printf "services[%s].pd.decode.dataParallelSize (%d) must be divisible by dataParallelSizeLocal (%d)" (.name | default "unknown") $dpsd $dpld) }}
{{- end }}
{{- end }}
{{/* multi-DP: dataParallelRpcPort is required between ranks; dataParallelAddress comes from LWS env (not user-set). */}}
{{- if and .enabled (eq .mode "pd") (gt (int (.pd.prefill.dataParallelSize | default 1)) 1) }}
{{- if not .pd.prefill.dataParallelRpcPort }}
{{- $errors = append $errors (printf "services[%s].pd.prefill.dataParallelRpcPort is required when pd.prefill.dataParallelSize>1 " (.name | default "unknown")) }}
{{- end }}
{{- end }}
{{- if and .enabled (eq .mode "pd") (gt (int (.pd.decode.dataParallelSize | default 1)) 1) }}
{{- if not .pd.decode.dataParallelRpcPort }}
{{- $errors = append $errors (printf "services[%s].pd.decode.dataParallelRpcPort is required when pd.decode.dataParallelSize>1" (.name | default "unknown")) }}
{{- end }}
{{- end }}
{{- if and .enabled (eq .mode "aggregated") }}
{{- $dpsa := int (.aggregated.dataParallelSize | default 1) }}
{{- $dpla := int (.aggregated.dataParallelSizeLocal | default 1) }}
{{- if lt $dpla 1 }}
{{- $errors = append $errors (printf "services[%s].aggregated.dataParallelSizeLocal must be >= 1" (.name | default "unknown")) }}
{{- else if and (gt $dpsa 0) (ne (mod $dpsa $dpla) 0) }}
{{- $errors = append $errors (printf "services[%s].aggregated.dataParallelSize (%d) must be divisible by dataParallelSizeLocal (%d) for LWS sizing" (.name | default "unknown") $dpsa $dpla) }}
{{- end }}
{{- end }}
{{- if and .enabled (eq .mode "aggregated") (gt (int (.aggregated.dataParallelSize | default 1)) 1) }}
{{- if not .aggregated.dataParallelRpcPort }}
{{- $errors = append $errors (printf "services[%s].aggregated.dataParallelRpcPort is required when aggregated.dataParallelSize>1 (LWS)" (.name | default "unknown")) }}
{{- end }}
{{- end }}
{{/* Reject removed structured engine fields wherever they are still present,
     including disabled services and inactive mode branches. */}}
{{- $serviceName := .name | default "unknown" }}
{{- with .pd }}
{{- with .prefill }}
{{- $prePath := printf "services[%s].pd.prefill" $serviceName }}
{{- $legacyPre := include "vllm.validateLegacyFields" (dict "nodePath" $prePath "node" .) }}
{{- if $legacyPre }}{{- $errors = append $errors $legacyPre }}{{- end }}
{{- end }}
{{- with .decode }}
{{- $decPath := printf "services[%s].pd.decode" $serviceName }}
{{- $legacyDec := include "vllm.validateLegacyFields" (dict "nodePath" $decPath "node" .) }}
{{- if $legacyDec }}{{- $errors = append $errors $legacyDec }}{{- end }}
{{- end }}
{{- end }}
{{- with .aggregated }}
{{- $aggPath := printf "services[%s].aggregated" $serviceName }}
{{- $legacyAgg := include "vllm.validateLegacyFields" (dict "nodePath" $aggPath "node" .) }}
{{- if $legacyAgg }}{{- $errors = append $errors $legacyAgg }}{{- end }}
{{- end }}

{{/* Validate active engine options and conflicts with chart-managed parameters. */}}
{{- if and .enabled (eq .mode "pd") }}
{{- $prePath := printf "services[%s].pd.prefill" (.name | default "unknown") }}
{{- $maxTokensPre := include "vllm.validateMaxNumBatchedTokens" (dict "nodePath" $prePath "extraArgs" .pd.prefill.extraArgs "tensorParallelSize" (.pd.prefill.tensorParallelSize | default 1)) }}
{{- if $maxTokensPre }}{{- $errors = append $errors $maxTokensPre }}{{- end }}
{{- $errPre := include "vllm.validateExtraArgs" (dict "nodePath" $prePath "extraArgs" .pd.prefill.extraArgs) }}
{{- if $errPre }}{{- $errors = append $errors $errPre }}{{- end }}
{{- $decPath := printf "services[%s].pd.decode" (.name | default "unknown") }}
{{- $maxTokensDec := include "vllm.validateMaxNumBatchedTokens" (dict "nodePath" $decPath "extraArgs" .pd.decode.extraArgs "tensorParallelSize" (.pd.decode.tensorParallelSize | default 1)) }}
{{- if $maxTokensDec }}{{- $errors = append $errors $maxTokensDec }}{{- end }}
{{- $errDec := include "vllm.validateExtraArgs" (dict "nodePath" $decPath "extraArgs" .pd.decode.extraArgs) }}
{{- if $errDec }}{{- $errors = append $errors $errDec }}{{- end }}
{{- end }}
{{- if and .enabled (eq .mode "aggregated") }}
{{- $aggPath := printf "services[%s].aggregated" (.name | default "unknown") }}
{{- $maxTokensAgg := include "vllm.validateMaxNumBatchedTokens" (dict "nodePath" $aggPath "extraArgs" .aggregated.extraArgs "tensorParallelSize" (.aggregated.tensorParallelSize | default 1)) }}
{{- if $maxTokensAgg }}{{- $errors = append $errors $maxTokensAgg }}{{- end }}
{{- $errAgg := include "vllm.validateExtraArgs" (dict "nodePath" $aggPath "extraArgs" .aggregated.extraArgs) }}
{{- if $errAgg }}{{- $errors = append $errors $errAgg }}{{- end }}
{{- end }}
{{- end }}

{{/* At least one service must be enabled */}}
{{- if not $hasEnabledService }}
{{- $errors = append $errors "at least one service with enabled=true is required" }}
{{- end }}

{{/* If there are any validation errors, fail with all error messages */}}
{{- if $errors }}
{{- $errorMsg := join "\n" $errors }}
{{- printf "\nVALIDATION ERROR:\n%s\n" $errorMsg | fail }}
{{- end }}
{{- end }}

{{/*
Validate that extraArgs does not duplicate parameters managed by structured
fields or auto-generated by the chart.

Each parameter must have exactly one configuration source. If a conflict is
detected, validation fails immediately instead of silently overriding one value
with another. Neither structured-field precedence nor extraArgs precedence is used.

Args (dict):
  - nodePath: human readable config path, e.g. "services[qwen].pd.prefill"
  - extraArgs: the extraArgs list of a vLLM node (prefill / decode / aggregated)

Returns:
  Error message string if any conflict is found, otherwise empty string.
*/}}
{{- define "vllm.validateExtraArgs" -}}
{{- $nodePath := .nodePath -}}
{{- $errors := list -}}
{{- $reserved := list
  "--served-model-name"
  "--port"
  "--tensor-parallel-size"
  "--pipeline-parallel-size"
  "--data-parallel-size"
  "--data-parallel-size-local"
  "--data-parallel-rpc-port"
  "--data-parallel-start-rank"
  "--data-parallel-address"
  "--data-parallel-hybrid-lb"
  "--kv-transfer-config"
  "--kv-events-config" -}}
{{- $prefixCachingEnabled := false -}}
{{- $prefixCachingDisabled := false -}}
{{- range .extraArgs | default list -}}
{{- $flag := regexReplaceAll "(=|\\s).*" (trim (toString .)) "" -}}
{{- if eq $flag "--enable-prefix-caching" -}}
{{- $prefixCachingEnabled = true -}}
{{- else if eq $flag "--no-enable-prefix-caching" -}}
{{- $prefixCachingDisabled = true -}}
{{- end -}}
{{- if has $flag $reserved -}}
{{- $errors = append $errors (printf "%s.extraArgs: parameter %s conflicts with a structured field or a chart-managed vLLM parameter; duplicate configuration in extraArgs is not allowed" $nodePath $flag) -}}
{{- end -}}
{{- end -}}
{{- if and $prefixCachingEnabled $prefixCachingDisabled -}}
{{- $errors = append $errors (printf "%s.extraArgs: --enable-prefix-caching and --no-enable-prefix-caching cannot be configured together" $nodePath) -}}
{{- end -}}
{{- join "\n" $errors -}}
{{- end -}}

{{/*
Reject structured engine fields that were migrated to extraArgs. hasKey is
used deliberately so zero and false values are not silently ignored.

Args (dict):
  - nodePath: human readable config path
  - node: the configuration map of a vLLM node

Returns:
  Newline-separated error messages, or an empty string.
*/}}
{{- define "vllm.validateLegacyFields" -}}
{{- $nodePath := .nodePath -}}
{{- $node := .node | default (dict) -}}
{{- $errors := list -}}
{{- $migrations := list
  (dict "field" "maxModelLen" "flags" "--max-model-len")
  (dict "field" "maxNumBatchedTokens" "flags" "--max-num-batched-tokens")
  (dict "field" "gpuMemoryUtilization" "flags" "--gpu-memory-utilization")
  (dict "field" "blockSize" "flags" "--block-size")
  (dict "field" "enablePrefixCaching" "flags" "--enable-prefix-caching or --no-enable-prefix-caching") -}}
{{- range $migration := $migrations -}}
{{- if hasKey $node $migration.field -}}
{{- $errors = append $errors (printf "%s.%s has been removed; migrate it to %s in %s.extraArgs" $nodePath $migration.field $migration.flags $nodePath) -}}
{{- end -}}
{{- end -}}
{{- join "\n" $errors -}}
{{- end -}}

{{/*
Validate --max-num-batched-tokens when it is explicitly configured. Only the
canonical `--max-num-batched-tokens <positive-integer>` form is accepted. The
chart does not assume an upstream/default value when the option is absent.

Args (dict):
  - nodePath: human readable config path
  - extraArgs: the extraArgs list of a vLLM node
  - tensorParallelSize: tensor parallel size for the node

Returns:
  Error message string, or an empty string.
*/}}
{{- define "vllm.validateMaxNumBatchedTokens" -}}
{{- $nodePath := .nodePath -}}
{{- $flag := "--max-num-batched-tokens" -}}
{{- $flagPattern := printf "^%s([[:space:]]|=|$)" $flag -}}
{{- $valuePattern := printf "^%s [1-9][0-9]*$" $flag -}}
{{- $matches := list -}}
{{- range .extraArgs | default list -}}
{{- $arg := trim (toString .) -}}
{{- if regexMatch $flagPattern $arg -}}
{{- $matches = append $matches $arg -}}
{{- end -}}
{{- end -}}
{{- $count := len $matches -}}
{{- if gt $count 1 -}}
{{- printf "%s.extraArgs: parameter %s must be specified at most once, got %d occurrences" $nodePath $flag $count -}}
{{- else if eq $count 1 -}}
{{- $arg := index $matches 0 -}}
{{- if not (regexMatch $valuePattern $arg) -}}
{{- printf "%s.extraArgs: parameter %s must use the format \"%s <positive-integer>\" in one list item" $nodePath $flag $flag -}}
{{- else -}}
{{- $tokens := int (trimPrefix (printf "%s " $flag) $arg) -}}
{{- $tp := int (.tensorParallelSize | default 1) -}}
{{- if and (gt $tp 0) (ne (mod $tokens $tp) 0) -}}
{{- printf "%s.extraArgs: %s (%d) must be a multiple of tensorParallelSize (%d)" $nodePath $flag $tokens $tp -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

