{{/*
Common labels
*/}}
{{- define "gateway-api-inference-extension.labels" -}}
app.kubernetes.io/name: {{ include "gateway-api-inference-extension.name" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Inference extension name
*/}}
{{- define "gateway-api-inference-extension.name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-epp
{{- end -}}

{{/*
InferencePool name.
Defaults to the release name to preserve the upstream GIE chart behavior.
Hermes Router may override it with inferencePool.name, or derive it from
global.modelName for model-oriented releases.
*/}}
{{- define "gateway-api-inference-extension.inferencepool.name" -}}
{{- if and .Values.inferencePool .Values.inferencePool.name -}}
{{- .Values.inferencePool.name | replace "/" "-" | replace "." "-" | lower | trim | trunc 63 | trimSuffix "-" -}}
{{- else if and .Values.global .Values.global.modelName -}}
{{- .Values.global.modelName | replace "/" "-" | replace "." "-" | lower | trim | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- .Release.Name -}}
{{- end -}}
{{- end -}}

{{/*
Cluster RBAC unique name
*/}}
{{- define "gateway-api-inference-extension.cluster-rbac-name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 }}
{{- $ns := .Release.Namespace | default "default" | lower | trim | trunc 40 }}
{{- printf "%s-%s-epp" $base $ns | quote | trunc 84 }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
{{- /* Check if endpointsServer exists AND if createInferencePool is false */ -}}
{{- if and .Values.inferenceExtension.endpointsServer (not .Values.inferenceExtension.endpointsServer.createInferencePool) -}}
{{- /* LOGIC FOR STANDALONE EPP MODE */ -}}
epp: {{ include "gateway-api-inference-extension.name" . }}
{{- else -}}
{{- /* LOGIC FOR PARENT (INFERENCEPOOL) MODE */ -}}
inferencepool: {{ include "gateway-api-inference-extension.name" . }}
{{- end -}}
{{- end -}}

{{/*
Mode labels
*/}}
{{- define "gateway-api-inference-extension.modeLabels" -}}
{{- if and .Values.inferenceExtension.endpointsServer (not .Values.inferenceExtension.endpointsServer.createInferencePool) -}}
inference.networking.k8s.io/igw-mode: standalone
{{- else -}}
inference.networking.k8s.io/igw-mode: inferencepool
{{- end -}}
{{- end -}}


{{/*
Create a default fully qualified app name for inferenceGateway.
*/}}
{{- define "gateway-api-inference-extension.gateway.fullname" -}}
  {{- if .Values.experimentalHttpRoute.inferenceGatewayName -}}
    {{- .Values.experimentalHttpRoute.inferenceGatewayName | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- printf "%s-inference-gateway" .Release.Name| trunc 63 | trimSuffix "-" -}}
  {{- end -}}
{{- end -}}

{{/*
Resolve a container image reference from the top-level images map.
Usage: image: '{{ list . "component-1" | include "helpers.image.name" }}'
*/}}
{{- define "helpers.image.name" -}}
{{- $ctx := index . 0 -}}
{{- $imageKey := index . 1 -}}
{{- $image := $imageKey | get $ctx.Values.images -}}
{{- required (printf "images.%s is required" $imageKey) $image.repository -}}:{{ $image.tag | default $ctx.Chart.AppVersion }}{{ $image.digest | default "" | empty | ternary "" (print "@sha256:" $image.digest) }}
{{- end -}}

{{/*
Resolve imagePullPolicy from the top-level images map entry.
Usage: {{ list . "component-1" | include "helpers.image.pullPolicy" }}
*/}}
{{- define "helpers.image.pullPolicy" -}}
{{- $ctx := index . 0 -}}
{{- $imageKey := index . 1 -}}
{{- $image := $imageKey | get $ctx.Values.images -}}
{{- include "gateway-api-inference-extension.imagePullPolicy" (dict "root" $ctx "localPolicy" (default "" $image.pullPolicy)) -}}
{{- end -}}

{{/*
Resolve the imagePullPolicy for chart-managed containers.
global.image.pullPolicy overrides container-local settings when set.
*/}}
{{- define "gateway-api-inference-extension.imagePullPolicy" -}}
{{- $root := index . "root" -}}
{{- $localPolicy := default "" (index . "localPolicy") -}}
{{- $defaultPolicy := default "IfNotPresent" (index . "defaultPolicy") -}}
{{- $globalPolicy := "" -}}
{{- if and $root.Values.global $root.Values.global.image $root.Values.global.image.pullPolicy -}}
{{- $globalPolicy = $root.Values.global.image.pullPolicy -}}
{{- end -}}
{{- if $globalPolicy -}}
{{- $globalPolicy -}}
{{- else if $localPolicy -}}
{{- $localPolicy -}}
{{- else -}}
{{- $defaultPolicy -}}
{{- end -}}
{{- end -}}
