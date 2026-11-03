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

{{- define "infernex-bridge.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{- define "infernex-bridge.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := include "infernex-bridge.name" . -}}
{{- if eq .Release.Name $name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "infernex-bridge.serviceAccountName" -}}
{{- if .Values.serviceAccount.create -}}
{{- default (include "infernex-bridge.fullname" .) .Values.serviceAccount.name -}}
{{- else -}}
{{- default "default" .Values.serviceAccount.name -}}
{{- end -}}
{{- end -}}

{{/*
Default image pull policy from global.image.pullPolicy.
*/}}
{{- define "infernex-bridge.image.pullPolicy" -}}
{{- if and .Values.global.image .Values.global.image.pullPolicy }}
{{- .Values.global.image.pullPolicy }}
{{- else }}
{{- "IfNotPresent" }}
{{- end }}
{{- end -}}
