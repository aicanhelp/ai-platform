{{/*
Expand the name of the chart.
*/}}
{{- define "ub-network-bandwidth-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "ub-network-bandwidth-operator.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "ub-network-bandwidth-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Common labels.
*/}}
{{- define "ub-network-bandwidth-operator.labels" -}}
helm.sh/chart: {{ include "ub-network-bandwidth-operator.chart" . }}
{{ include "ub-network-bandwidth-operator.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels.
*/}}
{{- define "ub-network-bandwidth-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "ub-network-bandwidth-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
Target namespace.
*/}}
{{- define "ub-network-bandwidth-operator.namespace" -}}
{{- default .Release.Namespace .Values.namespace.name -}}
{{- end -}}
