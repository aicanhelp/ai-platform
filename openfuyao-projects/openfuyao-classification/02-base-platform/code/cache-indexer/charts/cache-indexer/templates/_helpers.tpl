{{/*
Standard helpers for cache-indexer chart (Go rewrite).
*/}}

{{- define "cache-indexer.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "cache-indexer.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}

{{- define "cache-indexer.labels" -}}
app.kubernetes.io/name: {{ include "cache-indexer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
app.kubernetes.io/component: cache-indexer
{{- end -}}

{{- define "cache-indexer.selectorLabels" -}}
app.kubernetes.io/name: {{ include "cache-indexer.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "cache-indexer.image.pullPolicy" -}}
{{- if and .Values.global .Values.global.image .Values.global.image.pullPolicy }}
{{- .Values.global.image.pullPolicy }}
{{- else if (index .Values.images "cache-indexer").pullPolicy }}
{{- (index .Values.images "cache-indexer").pullPolicy }}
{{- else }}
{{- "IfNotPresent" }}
{{- end }}
{{- end -}}

{{- define "cache-indexer.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "cache-indexer.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}
