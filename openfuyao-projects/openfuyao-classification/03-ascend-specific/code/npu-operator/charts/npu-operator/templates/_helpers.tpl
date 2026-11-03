{{/*
Expand the name of the chart.
*/}}
{{- define "npu-operator.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "npu-operator.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "npu-operator.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "npu-operator.labels" -}}
helm.sh/chart: {{ include "npu-operator.chart" . }}
{{ include "npu-operator.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "npu-operator.selectorLabels" -}}
app.kubernetes.io/name: {{ include "npu-operator.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "npu-operator.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "npu-operator.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{- define "helpers.image.name" }}
{{- $ctx := index . 0 -}}
{{- $image := index . 1 | get $ctx.Values.images -}}
{{- $image.repository }}:{{ $image.tag | default $ctx.Chart.AppVersion }}{{ $image.digest | default "" | empty | ternary "" (print "@sha256:" $image.digest) }}
{{- end }}

{{- define "npu-operator.operand-labels" -}}
helm.sh/chart: {{ include "npu-operator.chart" . }}
app.kubernetes.io/managed-by: {{ include "npu-operator.name" . }}
{{- if .Values.daemonsets.labels }}
{{ toYaml .Values.daemonsets.labels }}
{{- end }}
{{- end -}}

{{- define "npu-operator.imageSpec" -}}
{{- $image := index .root.Values.images .image -}}
registry: ""
repository: '{{ $image.repository }}'
tag: '{{ $image.tag | default .root.Chart.AppVersion }}'
imagePullPolicy: {{ $image.pullPolicy }}
imagePullSecrets:
{{- with .values.imagePullSecrets }}
{{ toYaml . | nindent 2 }}
{{- end }}
{{- end -}}

{{- define "npu-operator.optionalImageSpec" -}}
{{- $image := index .root.Values.images .image | default dict -}}
{{- $secrets := .values.imagePullSecrets | default list -}}
{{- if or $image.repository $image.tag $image.pullPolicy $secrets }}
registry: ""
{{- with $image.repository }}
repository: '{{ . }}'
{{- end }}
{{- if $image.repository }}
tag: '{{ $image.tag | default .root.Chart.AppVersion }}'
{{- else if $image.tag }}
tag: '{{ $image.tag }}'
{{- end }}
{{- with $image.pullPolicy }}
imagePullPolicy: {{ . }}
{{- end }}
{{- with $secrets }}
imagePullSecrets:
{{ toYaml . | nindent 2 }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "npu-operator.placementSpec" -}}
{{- with .nodeSelector }}
nodeSelector:
{{ toYaml . | indent 2 }}
{{- end }}
{{- with .tolerations }}
tolerations:
{{ toYaml . | indent 2 }}
{{- end }}
{{- end -}}

{{- define "npu-operator.commandSpec" -}}
{{- with .commandSpec }}
commandSpec:
{{- if .command }}
  command: {{ .command | toJson }}
{{- end }}
{{- if .args }}
  args: {{ .args | toJson }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "npu-operator.envSpec" -}}
{{- with .env }}
env:
{{ toYaml . | indent 2 }}
{{- end }}
{{- end -}}

{{- define "npu-operator.volumeSpec" -}}
{{- with .volumeSpec }}
volumeSpec:
{{- with .volumes }}
  volumes:
{{ toYaml . | indent 4 }}
{{- end }}
{{- with .volumeMounts }}
  volumeMounts:
{{ toYaml . | indent 4 }}
{{- end }}
{{- end }}
{{- end -}}

{{- define "npu-operator.resourcesSpec" -}}
{{- with .resources }}
resources:
{{ toYaml . | indent 2 }}
{{- end }}
{{- end -}}
