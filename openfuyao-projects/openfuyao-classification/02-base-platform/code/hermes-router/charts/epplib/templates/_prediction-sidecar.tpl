{{/*
Prediction sidecar templates.

Three reusable defines:
  - inference-extension.prediction.container       Container spec for the
                                                   prediction sidecar.
  - inference-extension.prediction.eppVolumeMount  Shared UDS socket mount for
                                                   the EPP main container.
  - inference-extension.prediction.podVolume       Pod-level emptyDir backing
                                                   the shared socket.

Each define is a no-op when prediction is disabled, so callers in
_deployment.yaml can include unconditionally and let the templates self-gate.
*/}}

{{- define "inference-extension.prediction.socketPath" -}}
{{- $cfg := .Values.inferenceExtension.prediction | default dict -}}
{{- $cfg.socketPath | default "/var/run/hermes/prediction.sock" -}}
{{- end -}}

{{- define "inference-extension.prediction.socketVolumeName" -}}prediction-socket-volume{{- end -}}

{{/*
artifactRoot is the in-pod directory the bundle is mounted at and passed to the
sidecar as --artifact-root. It is an internal convention, not a user-facing knob:
the same value drives both the mountPath and the CLI flag so they cannot drift.
Power users may still override it via inferenceExtension.prediction.artifactRoot.
*/}}
{{- define "inference-extension.prediction.artifactRoot" -}}
{{- $cfg := .Values.inferenceExtension.prediction | default dict -}}
{{- $cfg.artifactRoot | default "/models/hermes" -}}
{{- end -}}

{{- define "inference-extension.prediction.modelVolumeName" -}}prediction-model{{- end -}}

{{- define "inference-extension.prediction.container" -}}
{{- $cfg := .Values.inferenceExtension.prediction | default dict -}}
{{- if $cfg.enabled }}
{{- $socketPath := include "inference-extension.prediction.socketPath" . -}}
{{- $socketDir := dir $socketPath -}}
{{- $volName := include "inference-extension.prediction.socketVolumeName" . -}}
{{- $artifactRoot := include "inference-extension.prediction.artifactRoot" . -}}
{{- $modelVolName := include "inference-extension.prediction.modelVolumeName" . }}
- name: prediction
  image: '{{ list . "hermes-router-prediction" | include "helpers.image.name" }}'
  imagePullPolicy: {{ list . "hermes-router-prediction" | include "helpers.image.pullPolicy" }}
  args:
    - --socket-path
    - {{ $socketPath | quote }}
    - --artifact-root
    - {{ $artifactRoot | quote }}
  {{- with $cfg.targetModel }}
    - --target-model
    - {{ . | quote }}
  {{- end }}
  {{- with $cfg.modelVersion }}
    - --model-version
    - {{ . | quote }}
  {{- end }}
  {{- with $cfg.extraArgs }}
  {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $cfg.env }}
  env:
  {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $cfg.resources }}
  resources:
  {{- toYaml . | nindent 4 }}
  {{- end }}
  volumeMounts:
    - name: {{ $volName }}
      mountPath: {{ $socketDir | quote }}
    - name: {{ $modelVolName }}
      mountPath: {{ $artifactRoot | quote }}
  {{- with $cfg.volumeMounts }}
  {{- toYaml . | nindent 4 }}
  {{- end }}
{{- end }}
{{- end -}}

{{- define "inference-extension.prediction.eppVolumeMount" -}}
{{- $cfg := .Values.inferenceExtension.prediction | default dict -}}
{{- if $cfg.enabled }}
- name: {{ include "inference-extension.prediction.socketVolumeName" . }}
  mountPath: {{ dir (include "inference-extension.prediction.socketPath" .) | quote }}
{{- end }}
{{- end -}}

{{- define "inference-extension.prediction.podVolume" -}}
{{- $cfg := .Values.inferenceExtension.prediction | default dict -}}
{{- if $cfg.enabled }}
- name: {{ include "inference-extension.prediction.socketVolumeName" . }}
  emptyDir: {}
- name: {{ include "inference-extension.prediction.modelVolumeName" . }}
  {{- with $cfg.modelVolume }}
  {{- toYaml . | nindent 2 }}
  {{- else }}
  emptyDir: {}
  {{- end }}
{{- with $cfg.volumes }}
{{ tpl (toYaml .) $ }}
{{- end }}
{{- end }}
{{- end -}}
