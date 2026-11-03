{{/*
Extract the port number from the listen address.
*/}}
{{- define "extractPort" -}}
{{- $listenAddress := .Values.web.listenAddress -}}
{{- if hasPrefix ":" $listenAddress -}}
{{- $port := (regexFind ":(\\d+)" $listenAddress) -}}
{{- if $port -}}
{{- $port | trimPrefix ":" -}}
{{- end -}}
{{- else -}}
{{- $port := $listenAddress -}}
{{- if $port -}}
{{- $port -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{- define "helpers.image.name" }}
{{- $ctx := index . 0 -}}
{{- $image := index . 1 | get $ctx.Values.images -}}
{{- $image.repository }}:{{ $image.tag | default $ctx.Chart.AppVersion }}{{ $image.digest | default "" | empty | ternary "" (print "@sha256:" $image.digest) }}
{{- end }}

