{{- define "e2e-logging-extension.name" -}}
e2e-logging-extension
{{- end -}}

{{- define "e2e-logging-extension.fullname" -}}
{{ include "e2e-logging-extension.name" . }}
{{- end -}}
