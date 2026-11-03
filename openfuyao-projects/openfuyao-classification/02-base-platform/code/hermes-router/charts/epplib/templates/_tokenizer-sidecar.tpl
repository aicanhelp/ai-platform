{{/*
Tokenizer sidecar templates. Mirrors the prediction sidecar pattern.

Three reusable defines:
  - inference-extension.tokenizer.container       Container spec for the
                                                  tokenizer sidecar.
  - inference-extension.tokenizer.eppVolumeMount  Shared UDS socket mount for
                                                  the EPP main container.
  - inference-extension.tokenizer.podVolume       Pod-level emptyDir backing
                                                  the shared socket.

Each define self-gates on .Values.inferenceExtension.tokenizer.enabled so
callers can include unconditionally.
*/}}

{{- define "inference-extension.tokenizer.socketPath" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- $cfg.socketPath | default "/tmp/tokenizer/tokenizer-uds.socket" -}}
{{- end -}}

{{- define "inference-extension.tokenizer.socketDir" -}}{{ dir (include "inference-extension.tokenizer.socketPath" .) }}{{- end -}}

{{- define "inference-extension.tokenizer.provider" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- $cfg.provider | default "huggingface" -}}
{{- end -}}

{{- define "inference-extension.tokenizer.socketVolumeName" -}}tokenizer-uds{{- end -}}

{{/*
Resolve the tokenizer thread-pool size (parallel-tokenize worker threads).
Precedence: explicit .threadPoolSize > derived from resources.limits.cpu
(one worker per whole core, floored, minimum 1) > empty (sidecar default).
Handles both core ("4") and millicpu ("3500m") CPU-limit forms.
*/}}
{{- define "inference-extension.tokenizer.threadPoolSize" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- if $cfg.threadPoolSize -}}
{{- $cfg.threadPoolSize -}}
{{- else -}}
{{- $limits := ($cfg.resources | default dict).limits | default dict -}}
{{- $cpu := $limits.cpu -}}
{{- if $cpu -}}
{{- $cpuStr := printf "%v" $cpu -}}
{{- if hasSuffix "m" $cpuStr -}}
{{- max 1 (div (trimSuffix "m" $cpuStr | int) 1000) -}}
{{- else -}}
{{- max 1 (int (floor (float64 $cpuStr))) -}}
{{- end -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
HTTP /healthz listen port. 8082 matches the sidecar default and the kserve
scheduler preset. Explicit 0 disables the endpoint; hasKey is required so
Helm does not treat 0 as empty and fall back to 8082.
*/}}
{{- define "inference-extension.tokenizer.healthPort" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- if hasKey $cfg "healthPort" -}}
{{- int $cfg.healthPort -}}
{{- else -}}
8082
{{- end -}}
{{- end -}}

{{- define "inference-extension.tokenizer.container" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- if $cfg.enabled }}
{{- $socketPath := include "inference-extension.tokenizer.socketPath" . -}}
{{- $socketDir := dir $socketPath -}}
{{- $healthPort := include "inference-extension.tokenizer.healthPort" . | int -}}
- name: tokenizer
  image: '{{ list . "hermes-router-tokenizer" | include "helpers.image.name" }}'
  imagePullPolicy: {{ list . "hermes-router-tokenizer" | include "helpers.image.pullPolicy" }}
  {{- if $cfg.args }}
  args:
    {{- range $cfg.args }}
    - {{ . | quote }}
    {{- end }}
  {{- else }}
  args:
    - --socket-path
    - {{ $socketPath | quote }}
    - --health-port
    - {{ $healthPort | quote }}
    {{- with $cfg.extraArgs }}
    {{- toYaml . | nindent 4 }}
    {{- end }}
  {{- end }}
  {{- if $cfg.env }}
  env:
  {{- toYaml $cfg.env | nindent 4 }}
  {{- else }}
  env:
    - name: TOKENIZER_PROVIDER
      value: {{ include "inference-extension.tokenizer.provider" . | quote }}
    {{- $threadPoolSize := include "inference-extension.tokenizer.threadPoolSize" . | trim }}
    {{- $extraDefinesThreadPool := false }}
    {{- range ($cfg.extraEnv | default list) }}
    {{- if eq (.name | default "") "TOKENIZER_THREAD_POOL_SIZE" }}{{ $extraDefinesThreadPool = true }}{{ end }}
    {{- end }}
    {{- if and $threadPoolSize (not $extraDefinesThreadPool) }}
    - name: TOKENIZER_THREAD_POOL_SIZE
      value: {{ $threadPoolSize | quote }}
    {{- end }}
  {{- with $cfg.extraEnv }}
  {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- end }}
  {{- with $cfg.resources }}
  resources:
  {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- if $healthPort }}
  ports:
    - name: health
      containerPort: {{ $healthPort }}
      protocol: TCP
  {{- with $cfg.startupProbe }}
  startupProbe:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $cfg.livenessProbe }}
  livenessProbe:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- with $cfg.readinessProbe }}
  readinessProbe:
    {{- toYaml . | nindent 4 }}
  {{- end }}
  {{- end }}
  volumeMounts:
    - name: {{ include "inference-extension.tokenizer.socketVolumeName" . }}
      mountPath: {{ $socketDir | quote }}
  {{- range $mount := $cfg.volumeMounts }}
    - name: {{ required "inferenceExtension.tokenizer.volumeMounts[].name is required" $mount.name }}
      mountPath: {{ required "inferenceExtension.tokenizer.volumeMounts[].mountPath is required" $mount.mountPath | quote }}
      {{- with $mount.subPath }}
      subPath: {{ . | quote }}
      {{- end }}
      {{- with $mount.subPathExpr }}
      subPathExpr: {{ . | quote }}
      {{- end }}
      {{- with $mount.readOnly }}
      readOnly: {{ . }}
      {{- end }}
      {{- with $mount.mountPropagation }}
      mountPropagation: {{ . | quote }}
      {{- end }}
  {{- end }}
{{- end }}
{{- end -}}

{{- define "inference-extension.tokenizer.eppVolumeMount" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- if $cfg.enabled }}
- name: {{ include "inference-extension.tokenizer.socketVolumeName" . }}
  mountPath: {{ include "inference-extension.tokenizer.socketDir" . | quote }}
{{- end }}
{{- end -}}

{{- define "inference-extension.tokenizer.podVolume" -}}
{{- $cfg := .Values.inferenceExtension.tokenizer | default dict -}}
{{- if $cfg.enabled }}
- name: {{ include "inference-extension.tokenizer.socketVolumeName" . }}
  emptyDir: {}
{{- with $cfg.volumes }}
{{ tpl (toYaml .) $ }}
{{- end }}
{{- end }}
{{- end -}}
