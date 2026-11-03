{{/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */}}

{{- define "gateway-api-inference-extension.routing.state" -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- dict
  "enabled" (not (empty $routing.profile))
      "deploymentMode" (default "aggregate" $routing.deploymentMode)
      "profile" (default "" $routing.profile)
  | toYaml -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.tokenizerSocketPath" -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- $tokenizer := default (dict) $routing.tokenizer -}}
{{- $sidecar := default (dict) .Values.inferenceExtension.tokenizer -}}
{{- if not (empty $tokenizer.socketPath) -}}
{{ $tokenizer.socketPath }}
{{- else if or (default false $sidecar.enabled) (not (empty $sidecar.socketPath)) -}}
{{ include "inference-extension.tokenizer.socketPath" . }}
{{- end -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.validate" -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- $state := include "gateway-api-inference-extension.routing.state" . | fromYaml -}}
{{- $tokenizer := default (dict) $routing.tokenizer -}}
{{- $tokenizerSocketPath := include "gateway-api-inference-extension.routing.tokenizerSocketPath" . | trim -}}
{{- $requestTracking := default (dict) $routing.requestTracking -}}
{{- $requestPersistence := default (dict) $requestTracking.persistence -}}
{{- $cacheIndexer := default (dict) $routing.cacheIndexer -}}
{{- $decodeFirst := default (dict) $routing.decodeFirst -}}
{{- $prefillDecider := default (dict) $decodeFirst.prefillDecider -}}
{{- $prefixCacheFilter := default (dict) $routing.prefixCacheFilter -}}
{{- $npuExporter := default (dict) $routing.npuExporter -}}
{{- if $state.enabled -}}
  {{- $usesInflightTracking := or (eq $state.profile "prediction") (eq $state.profile "kv-cache-aware") (eq $state.profile "decode-first") (and (eq $state.profile "bucket") (eq $state.deploymentMode "pd")) -}}
  {{- $usesTokenizer := or (eq $state.profile "prediction") (eq $state.profile "kv-cache-aware") (eq $state.profile "decode-first") -}}
  {{- if .Values.inferenceExtension.latencyPredictor.enabled -}}
    {{- fail "inferenceExtension.latencyPredictor and inferenceExtension.routing cannot be enabled together" -}}
  {{- end -}}
  {{- if not (has $state.deploymentMode (list "aggregate" "pd")) -}}
    {{- fail (printf "unsupported inferenceExtension.routing.deploymentMode %q" $state.deploymentMode) -}}
  {{- end -}}
  {{- if not (has $state.profile (list "random" "kv-cache-aware" "bucket" "prediction" "decode-first")) -}}
    {{- fail (printf "unsupported inferenceExtension.routing.profile %q" $state.profile) -}}
  {{- end -}}
  {{- if and (eq $state.deploymentMode "aggregate") (has $state.profile (list "bucket" "decode-first")) -}}
    {{- fail (printf "routing profile %q requires deploymentMode=pd" $state.profile) -}}
  {{- end -}}
  {{- if and (eq $state.profile "kv-cache-aware") (empty $cacheIndexer.address) -}}
    {{- fail "inferenceExtension.routing.cacheIndexer.address must be set when routing.profile=kv-cache-aware" -}}
  {{- end -}}
  {{- if and (eq $state.profile "decode-first") (empty $cacheIndexer.address) -}}
    {{- fail "inferenceExtension.routing.cacheIndexer.address must be set when routing.profile=decode-first" -}}
  {{- end -}}
  {{- if and (eq $state.profile "decode-first") (not (has (default "prefix-based" $prefillDecider.type) (list "always-disagg" "prefix-based"))) -}}
    {{- fail (printf "unsupported inferenceExtension.routing.decodeFirst.prefillDecider.type %q" (default "prefix-based" $prefillDecider.type)) -}}
  {{- end -}}
  {{- if and (eq $state.profile "prediction") (not .Values.inferenceExtension.prediction.enabled) (ne (default "active" .Values.inferenceExtension.prediction.predictionMode) "shadow") -}}
    {{- fail "inferenceExtension.prediction.enabled must be true when routing.profile=prediction (unless predictionMode=shadow, which only collects features and needs no prediction sidecar)" -}}
  {{- end -}}
  {{- if and $usesTokenizer (or (empty $tokenizer.model) (empty $tokenizerSocketPath)) -}}
    {{- fail (printf "inferenceExtension.routing.tokenizer.model and tokenizer.socketPath must be set when routing.profile=%s" $state.profile) -}}
  {{- end -}}
  {{- if and $usesInflightTracking (empty $requestTracking.storeName) -}}
    {{- fail "inferenceExtension.routing.requestTracking.storeName must be set when routing consumes inflight tracking" -}}
  {{- end -}}
  {{- if and (eq $state.profile "prediction") (empty $cacheIndexer.address) -}}
    {{- fail "inferenceExtension.routing.cacheIndexer.address must be set when routing.profile=prediction because prefix-cache-filter requires prefix-cache-producer" -}}
  {{- end -}}
  {{- if and (hasKey $cacheIndexer "blockSize") (le (int $cacheIndexer.blockSize) 0) -}}
    {{- fail "inferenceExtension.routing.cacheIndexer.blockSize must be greater than 0" -}}
  {{- end -}}
  {{- if and (eq $state.profile "prediction") (or (empty $npuExporter.exporterNamespace) (empty $npuExporter.exporterPodLabelSelector) (le (int (default 0 $npuExporter.exporterPort)) 0) (empty $npuExporter.path) (empty $npuExporter.cacheTTL) (empty $npuExporter.scrapeTimeout) (empty $npuExporter.staleAfter)) -}}
    {{- fail "inferenceExtension.routing.npuExporter.exporterNamespace, exporterPodLabelSelector, exporterPort, path, cacheTTL, scrapeTimeout, and staleAfter must be set when routing.profile=prediction" -}}
  {{- end -}}
  {{- if and $usesInflightTracking (default false (index $requestPersistence "enabled")) (empty (index $requestPersistence "outputPath")) -}}
    {{- fail "inferenceExtension.routing.requestTracking.persistence.outputPath must be set when persistence.enabled=true" -}}
  {{- end -}}
{{- end -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.valueOrDefault" -}}
{{- $values := index . "values" -}}
{{- $key := index . "key" -}}
{{- $default := index . "default" -}}
{{- if hasKey $values $key -}}
{{ index $values $key }}
{{- else -}}
{{ $default }}
{{- end -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.predictionScorerParameters" -}}
{{- $prediction := default (dict) .Values.inferenceExtension.prediction -}}
predictorAddress: {{ printf "unix://%s" $prediction.socketPath | quote }}
predictionMode: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "predictionMode" "default" "active") | quote }}
timeout: {{ $prediction.timeout }}
maxBatchSize: {{ $prediction.maxBatchSize }}
ttftWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "ttftWeight" "default" 0.6) }}
tpotWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "tpotWeight" "default" 0.4) }}
kvWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "kvWeight" "default" 1.0) }}
queueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "queueWeight" "default" 1.0) }}
prefixWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "prefixWeight" "default" 1.0) }}
inflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prediction "key" "inflightWeight" "default" 1.0) }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.prefixCacheProducerParameters" -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- $cacheIndexer := default (dict) $routing.cacheIndexer -}}
cacheIndexerAddress: {{ $cacheIndexer.address | quote }}
{{- if hasKey $cacheIndexer "blockSize" }}
blockSize: {{ $cacheIndexer.blockSize }}
{{- end }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.pdLabelParameters" -}}
{{- $pd := default (dict) .Values.inferenceExtension.routing.pd -}}
pdLabelName: {{ default "openfuyao.com/pdRole" $pd.pdLabelName | quote }}
pdGroupLabelName: {{ default "openfuyao.com/pdGroupID" $pd.pdGroupLabelName | quote }}
prefillValue: {{ default "prefill" $pd.prefillValue | quote }}
decodeValue: {{ default "decode" $pd.decodeValue | quote }}
leaderValue: {{ default "leader" $pd.leaderValue | quote }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.pdTargetPodSelectorParameters" -}}
{{- $root := index . "root" -}}
{{- $roleValue := index . "roleValue" -}}
{{- $routing := default (dict) $root.Values.inferenceExtension.routing -}}
{{- $pd := default (dict) $routing.pd -}}
targetPodSelector:
  exists:
  - {{ default "openfuyao.com/pdGroupID" $pd.pdGroupLabelName | quote }}
  in:
    {{ default "openfuyao.com/pdRole" $pd.pdLabelName | quote }}:
    - {{ $roleValue | quote }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.requestTrackingPlugins" -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- $requestTracking := default (dict) $routing.requestTracking -}}
{{- $requestPersistence := default (dict) $requestTracking.persistence -}}
- type: request-lifecycle-tracker
  parameters:
    storeName: {{ $requestTracking.storeName | quote }}
    persistence:
      enabled: {{ default false $requestPersistence.enabled }}
      {{- if $requestPersistence.enabled }}
      flushThreshold: {{ default 500 $requestPersistence.flushThreshold }}
      {{- with $requestPersistence.flushInterval }}
      flushInterval: {{ . | quote }}
      {{- end }}
      outputPath: {{ $requestPersistence.outputPath | quote }}
      {{- end }}
- type: inflight-data-producer
  parameters:
    storeName: {{ $requestTracking.storeName | quote }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.requestTrackingPluginRefs" -}}
- pluginRef: request-lifecycle-tracker
- pluginRef: inflight-data-producer
{{- end -}}

{{- define "gateway-api-inference-extension.routing.commonPlugins" -}}
- type: metrics-data-source
  parameters:
    scheme: {{ .Values.inferenceExtension.metricsDataSource.scheme | default "http" | quote }}
    path: {{ .Values.inferenceExtension.metricsDataSource.path | default "/metrics" | quote }}
    insecureSkipVerify: {{ .Values.inferenceExtension.metricsDataSource.insecureSkipVerify | default true }}
- type: core-metrics-extractor
  {{- $modelServerType := .Values.inferencePool.modelServerType | default "vllm" }}
  {{- if and .Values.inferenceExtension.endpointsServer (not .Values.inferenceExtension.endpointsServer.createInferencePool) }}
    {{- $modelServerType = .Values.inferenceExtension.endpointsServer.modelServerType | default "vllm" }}
  {{- end }}
  {{- if ne $modelServerType "vllm" }}
  parameters:
    defaultEngine: {{ $modelServerType | quote }}
  {{- end }}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.strategyPlugins" -}}
{{- $state := include "gateway-api-inference-extension.routing.state" . | fromYaml -}}
{{- $routing := default (dict) .Values.inferenceExtension.routing -}}
{{- $pd := default (dict) $routing.pd -}}
{{- $kvCacheAware := default (dict) $routing.kvCacheAware -}}
{{- $bucket := default (dict) $routing.bucket -}}
{{- $tokenizer := default (dict) $routing.tokenizer -}}
{{- $tokenizerSocketPath := include "gateway-api-inference-extension.routing.tokenizerSocketPath" . | trim -}}
{{- $requestTracking := default (dict) $routing.requestTracking -}}
{{- $requestPersistence := default (dict) $requestTracking.persistence -}}
{{- $cacheIndexer := default (dict) $routing.cacheIndexer -}}
{{- $cacheIndexerPrefill := default (dict) $cacheIndexer.prefill -}}
{{- $cacheIndexerDecode := default (dict) $cacheIndexer.decode -}}
{{- $decodeFirst := default (dict) $routing.decodeFirst -}}
{{- $prefillDecider := default (dict) $decodeFirst.prefillDecider -}}
{{- $decodeKVCacheAware := default (dict) $decodeFirst.decodeKVCacheAware -}}
{{- $prefixCacheFilter := default (dict) $routing.prefixCacheFilter -}}
{{- $npuExporter := default (dict) $routing.npuExporter -}}
{{- $prediction := default (dict) .Values.inferenceExtension.prediction -}}
{{- if and $state.enabled (eq $state.profile "random") (eq $state.deploymentMode "aggregate") }}
- type: uniform-random-picker
{{- else if and $state.enabled (eq $state.profile "random") (eq $state.deploymentMode "pd") }}
- type: pd-group-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.pdLabelParameters" . | indent 4 }}
- type: pd-group-filter
- type: uniform-random-picker
- type: pd-header-handler
{{- else if and $state.enabled (eq $state.profile "decode-first") (eq $state.deploymentMode "pd") }}
- type: tokenizer
  parameters:
    model: {{ $tokenizer.model | quote }}
    socketPath: {{ $tokenizerSocketPath | quote }}
    {{- with $tokenizer.backend }}
    backend: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.tokenizerSource }}
    tokenizerSource: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.initTimeout }}
    initTimeout: {{ . | quote }}
    {{- end }}
    {{- if hasKey $tokenizer "initRetries" }}
    initRetries: {{ $tokenizer.initRetries }}
    {{- end }}
- type: pd-group-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.pdLabelParameters" . | indent 4 }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: prefix-cache-producer
  name: prefill-prefix-cache
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
    l1Weight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $cacheIndexerPrefill "key" "l1Weight" "default" 0.7) }}
    l3Weight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $cacheIndexerPrefill "key" "l3Weight" "default" 0.3) }}
{{ include "gateway-api-inference-extension.routing.pdTargetPodSelectorParameters" (dict "root" . "roleValue" (default "prefill" $pd.prefillValue)) | indent 4 }}
- type: prefix-cache-producer
  name: decode-prefix-cache
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
    l1Weight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $cacheIndexerDecode "key" "l1Weight" "default" 1.0) }}
    l3Weight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $cacheIndexerDecode "key" "l3Weight" "default" 0.0) }}
{{ include "gateway-api-inference-extension.routing.pdTargetPodSelectorParameters" (dict "root" . "roleValue" (default "decode" $pd.decodeValue)) | indent 4 }}
- type: pd-group-filter
- type: decode-kvcache-aware-scorer
  parameters:
    prefixWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $decodeKVCacheAware "key" "prefixWeight" "default" 1.0) }}
    runningWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $decodeKVCacheAware "key" "runningWeight" "default" 1.0) }}
    kvUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $decodeKVCacheAware "key" "kvUsageWeight" "default" 1.0) }}
    queueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $decodeKVCacheAware "key" "queueWeight" "default" 1.0) }}
    inflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $decodeKVCacheAware "key" "inflightWeight" "default" 1.0) }}
- type: maximum-score-picker
  name: decode-picker
- type: pd-kvcache-aware-scorer
  parameters:
    {{- if hasKey $kvCacheAware "prefillKVUsageWeight" }}
    prefillKVUsageWeight: {{ index $kvCacheAware "prefillKVUsageWeight" }}
    {{- else }}
    prefillKVUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillKVWeight" "default" 1.0) }}
    {{- end }}
    prefillPrefixWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillPrefixWeight" "default" 1.0) }}
    prefillQueueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillQueueWeight" "default" 1.0) }}
    prefillInflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillInflightWeight" "default" 1.0) }}
    decodeRunningWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeRunningWeight" "default" 1.0) }}
    {{- if hasKey $kvCacheAware "decodeKVUsageWeight" }}
    decodeKVUsageWeight: {{ index $kvCacheAware "decodeKVUsageWeight" }}
    {{- else }}
    decodeKVUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeKVWeight" "default" 1.0) }}
    {{- end }}
    decodeQueueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeQueueWeight" "default" 1.0) }}
    decodeInflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeInflightWeight" "default" 1.0) }}
    prefillScoreWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillScoreWeight" "default" 1.0) }}
    decodeScoreWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeScoreWeight" "default" 1.0) }}
- type: maximum-score-picker
  name: pd-picker
{{- if eq (default "prefix-based" $prefillDecider.type) "always-disagg" }}
- type: always-disagg-prefill-decider
  name: prefill-decider
{{- else }}
- type: prefix-based-prefill-decider
  name: prefill-decider
  parameters:
    minHitRatio: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prefillDecider "key" "minHitRatio" "default" 0.8) }}
    maxUncachedTokens: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prefillDecider "key" "maxUncachedTokens" "default" 128) }}
{{- end }}
- type: disagg-profile-handler
  parameters:
    decodeProfileName: decode
    pdProfileName: pd
    prefillDeciderPluginRef: prefill-decider
- type: pd-header-handler
{{- else if and $state.enabled (eq $state.profile "kv-cache-aware") (eq $state.deploymentMode "aggregate") }}
- type: tokenizer
  parameters:
    model: {{ $tokenizer.model | quote }}
    socketPath: {{ $tokenizerSocketPath | quote }}
    {{- with $tokenizer.backend }}
    backend: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.tokenizerSource }}
    tokenizerSource: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.initTimeout }}
    initTimeout: {{ . | quote }}
    {{- end }}
    {{- if hasKey $tokenizer "initRetries" }}
    initRetries: {{ $tokenizer.initRetries }}
    {{- end }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: prefix-cache-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
- type: kvcache-aware-scorer
  parameters:
    kvCacheNotHitRateWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "kvCacheNotHitRateWeight" "default" 1.0) }}
    xpuCacheUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "xpuCacheUsageWeight" "default" 1.0) }}
    waitingRequestWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "waitingRequestWeight" "default" 1.0) }}
    inflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "inflightWeight" "default" 1.0) }}
- type: maximum-score-picker
{{- else if and $state.enabled (eq $state.profile "kv-cache-aware") (eq $state.deploymentMode "pd") }}
- type: tokenizer
  parameters:
    model: {{ $tokenizer.model | quote }}
    socketPath: {{ $tokenizerSocketPath | quote }}
    {{- with $tokenizer.backend }}
    backend: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.tokenizerSource }}
    tokenizerSource: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.initTimeout }}
    initTimeout: {{ . | quote }}
    {{- end }}
    {{- if hasKey $tokenizer "initRetries" }}
    initRetries: {{ $tokenizer.initRetries }}
    {{- end }}
- type: pd-group-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.pdLabelParameters" . | indent 4 }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: prefix-cache-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
{{ include "gateway-api-inference-extension.routing.pdTargetPodSelectorParameters" (dict "root" . "roleValue" (default "prefill" $pd.prefillValue)) | indent 4 }}
- type: pd-group-filter
- type: pd-kvcache-aware-scorer
  parameters:
    {{- if hasKey $kvCacheAware "prefillKVUsageWeight" }}
    prefillKVUsageWeight: {{ index $kvCacheAware "prefillKVUsageWeight" }}
    {{- else }}
    prefillKVUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillKVWeight" "default" 1.0) }}
    {{- end }}
    prefillPrefixWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillPrefixWeight" "default" 1.0) }}
    prefillQueueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillQueueWeight" "default" 1.0) }}
    prefillInflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillInflightWeight" "default" 1.0) }}
    decodeRunningWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeRunningWeight" "default" 1.0) }}
    {{- if hasKey $kvCacheAware "decodeKVUsageWeight" }}
    decodeKVUsageWeight: {{ index $kvCacheAware "decodeKVUsageWeight" }}
    {{- else }}
    decodeKVUsageWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeKVWeight" "default" 1.0) }}
    {{- end }}
    decodeQueueWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeQueueWeight" "default" 1.0) }}
    decodeInflightWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeInflightWeight" "default" 1.0) }}
    prefillScoreWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "prefillScoreWeight" "default" 1.0) }}
    decodeScoreWeight: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $kvCacheAware "key" "decodeScoreWeight" "default" 1.0) }}
- type: maximum-score-picker
- type: pd-header-handler
{{- else if and $state.enabled (eq $state.profile "bucket") (eq $state.deploymentMode "pd") }}
- type: pd-group-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.pdLabelParameters" . | indent 4 }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: pd-group-filter
- type: pd-bucket-scorer
  parameters:
    alpha: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $bucket "key" "alpha" "default" 1.0) }}
    beta: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $bucket "key" "beta" "default" 2.0) }}
    decayFactor: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $bucket "key" "decayFactor" "default" 0.99) }}
    bucketSeparateLength: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $bucket "key" "bucketSeparateLength" "default" 200) }}
    {{- with $bucket.tpSizeLabelKey }}
    tpSizeLabelKey: {{ . | quote }}
    {{- end }}
- type: minimum-score-picker
- type: pd-header-handler
{{- else if and $state.enabled (eq $state.profile "prediction") (eq $state.deploymentMode "aggregate") }}
- type: tokenizer
  parameters:
    model: {{ $tokenizer.model | quote }}
    socketPath: {{ $tokenizerSocketPath | quote }}
    {{- with $tokenizer.backend }}
    backend: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.tokenizerSource }}
    tokenizerSource: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.initTimeout }}
    initTimeout: {{ . | quote }}
    {{- end }}
    {{- if hasKey $tokenizer "initRetries" }}
    initRetries: {{ $tokenizer.initRetries }}
    {{- end }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: prefix-cache-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
- type: prediction-feature-extractor
- type: prefix-cache-filter
  parameters:
    threshold: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prefixCacheFilter "key" "threshold" "default" 0.5) }}
- type: prediction-scorer
  parameters:
{{ include "gateway-api-inference-extension.routing.predictionScorerParameters" . | indent 4 }}
- type: softmax-potc-picker
- type: endpoint-notification-source
- type: npu-exporter-data-source
  parameters:
    exporterNamespace: {{ $npuExporter.exporterNamespace | quote }}
    exporterPodLabelSelector: {{ $npuExporter.exporterPodLabelSelector | quote }}
    exporterPort: {{ $npuExporter.exporterPort }}
    path: {{ $npuExporter.path | quote }}
    cacheTTL: {{ $npuExporter.cacheTTL | quote }}
    scrapeTimeout: {{ $npuExporter.scrapeTimeout | quote }}
- type: npu-exporter-extractor
  parameters:
    staleAfter: {{ $npuExporter.staleAfter | quote }}
{{- else if and $state.enabled (eq $state.profile "prediction") (eq $state.deploymentMode "pd") }}
- type: tokenizer
  parameters:
    model: {{ $tokenizer.model | quote }}
    socketPath: {{ $tokenizerSocketPath | quote }}
    {{- with $tokenizer.backend }}
    backend: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.tokenizerSource }}
    tokenizerSource: {{ . | quote }}
    {{- end }}
    {{- with $tokenizer.initTimeout }}
    initTimeout: {{ . | quote }}
    {{- end }}
    {{- if hasKey $tokenizer "initRetries" }}
    initRetries: {{ $tokenizer.initRetries }}
    {{- end }}
{{ include "gateway-api-inference-extension.routing.requestTrackingPlugins" . }}
- type: pd-group-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.pdLabelParameters" . | indent 4 }}
- type: prefix-cache-producer
  parameters:
{{ include "gateway-api-inference-extension.routing.prefixCacheProducerParameters" . | indent 4 }}
{{ include "gateway-api-inference-extension.routing.pdTargetPodSelectorParameters" (dict "root" . "roleValue" (default "prefill" $pd.prefillValue)) | indent 4 }}
- type: prediction-feature-extractor
  parameters:
    pdLabelName: {{ default "openfuyao.com/pdRole" $pd.pdLabelName | quote }}
    prefillValue: {{ default "prefill" $pd.prefillValue | quote }}
    decodeValue: {{ default "decode" $pd.decodeValue | quote }}
- type: pd-group-filter
- type: prefix-cache-filter
  parameters:
    threshold: {{ include "gateway-api-inference-extension.routing.valueOrDefault" (dict "values" $prefixCacheFilter "key" "threshold" "default" 0.5) }}
- type: prediction-scorer
  parameters:
{{ include "gateway-api-inference-extension.routing.predictionScorerParameters" . | indent 4 }}
- type: softmax-potc-picker
- type: pd-header-handler
- type: endpoint-notification-source
- type: npu-exporter-data-source
  parameters:
    exporterNamespace: {{ $npuExporter.exporterNamespace | quote }}
    exporterPodLabelSelector: {{ $npuExporter.exporterPodLabelSelector | quote }}
    exporterPort: {{ $npuExporter.exporterPort }}
    path: {{ $npuExporter.path | quote }}
    cacheTTL: {{ $npuExporter.cacheTTL | quote }}
    scrapeTimeout: {{ $npuExporter.scrapeTimeout | quote }}
- type: npu-exporter-extractor
  parameters:
    staleAfter: {{ $npuExporter.staleAfter | quote }}
{{- end -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.dataLayer" -}}
{{- $state := include "gateway-api-inference-extension.routing.state" . | fromYaml -}}
{{- if and $state.enabled (eq $state.profile "prediction") }}
sources:
- pluginRef: metrics-data-source
  extractors:
  - pluginRef: core-metrics-extractor
- pluginRef: endpoint-notification-source
  extractors:
  - pluginRef: request-lifecycle-tracker
- pluginRef: npu-exporter-data-source
  extractors:
  - pluginRef: npu-exporter-extractor
{{- end -}}
{{- end -}}

{{- define "gateway-api-inference-extension.routing.schedulingProfiles" -}}
{{- $state := include "gateway-api-inference-extension.routing.state" . | fromYaml -}}
{{- if and $state.enabled (eq $state.profile "decode-first") (eq $state.deploymentMode "pd") }}
- name: decode
  plugins:
  - pluginRef: pd-group-filter
  - pluginRef: decode-kvcache-aware-scorer
  - pluginRef: decode-picker
- name: pd
  plugins:
  - pluginRef: pd-group-filter
  - pluginRef: pd-kvcache-aware-scorer
  - pluginRef: pd-picker
{{- else }}
- name: default
  plugins:
  {{- if and $state.enabled (eq $state.profile "random") (eq $state.deploymentMode "aggregate") }}
  - pluginRef: uniform-random-picker
  {{- else if and $state.enabled (eq $state.profile "random") (eq $state.deploymentMode "pd") }}
  - pluginRef: pd-group-producer
  - pluginRef: pd-group-filter
  - pluginRef: uniform-random-picker
  - pluginRef: pd-header-handler
  {{- else if and $state.enabled (eq $state.profile "kv-cache-aware") (eq $state.deploymentMode "aggregate") }}
  - pluginRef: tokenizer
{{ include "gateway-api-inference-extension.routing.requestTrackingPluginRefs" . | indent 2 }}
  - pluginRef: prefix-cache-producer
  - pluginRef: kvcache-aware-scorer
  - pluginRef: maximum-score-picker
  {{- else if and $state.enabled (eq $state.profile "kv-cache-aware") (eq $state.deploymentMode "pd") }}
  - pluginRef: tokenizer
  - pluginRef: pd-group-producer
{{ include "gateway-api-inference-extension.routing.requestTrackingPluginRefs" . | indent 2 }}
  - pluginRef: prefix-cache-producer
  - pluginRef: pd-group-filter
  - pluginRef: pd-kvcache-aware-scorer
  - pluginRef: maximum-score-picker
  - pluginRef: pd-header-handler
  {{- else if and $state.enabled (eq $state.profile "bucket") (eq $state.deploymentMode "pd") }}
  - pluginRef: pd-group-producer
{{ include "gateway-api-inference-extension.routing.requestTrackingPluginRefs" . | indent 2 }}
  - pluginRef: pd-group-filter
  - pluginRef: pd-bucket-scorer
  - pluginRef: minimum-score-picker
  - pluginRef: pd-header-handler
  {{- else if and $state.enabled (eq $state.profile "prediction") (eq $state.deploymentMode "aggregate") }}
  - pluginRef: tokenizer
{{ include "gateway-api-inference-extension.routing.requestTrackingPluginRefs" . | indent 2 }}
  - pluginRef: prefix-cache-producer
  - pluginRef: prediction-feature-extractor
  - pluginRef: prefix-cache-filter
  - pluginRef: prediction-scorer
  - pluginRef: softmax-potc-picker
  {{- else if and $state.enabled (eq $state.profile "prediction") (eq $state.deploymentMode "pd") }}
  - pluginRef: tokenizer
{{ include "gateway-api-inference-extension.routing.requestTrackingPluginRefs" . | indent 2 }}
  - pluginRef: pd-group-producer
  - pluginRef: prefix-cache-producer
  - pluginRef: prediction-feature-extractor
  - pluginRef: pd-group-filter
  - pluginRef: prefix-cache-filter
  - pluginRef: prediction-scorer
  - pluginRef: softmax-potc-picker
  - pluginRef: pd-header-handler
  {{- end }}
{{- end }}
{{- end -}}
