{{/*
validate inferencePool.name is required
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.name" -}}
{{- $hasInferencePoolName := not (empty $.Values.inferencePool.name) -}}
{{- $hasGlobalModelName := and $.Values.global (not (empty $.Values.global.modelName)) -}}
{{- if not (or $hasInferencePoolName $hasGlobalModelName) }}
{{- fail ".Values.inferencePool.name or .Values.global.modelName is required" }}
{{- end -}}
{{- end -}}
