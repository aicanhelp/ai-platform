{{- define "eagle-eye.namespaceExists" -}}
  {{- $namespace := "eagle-eye" -}}
  {{- $ns := lookup "v1" "Namespace" "" $namespace -}}
  {{- if not $ns -}}
    false
  {{- else -}}
    true
  {{- end -}}
{{- end -}}
