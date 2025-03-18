{{/*
Common labels
*/}}
{{- define "gateway-api-inference-extension.labels" -}}
app.kubernetes.io/name: {{ include "gateway-api-inference-extension.name" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Inference extension name
*/}}
{{- define "gateway-api-inference-extension.name" -}}
{{- $base := .Values.inferencePool.name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-epp
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
app: {{ include "gateway-api-inference-extension.name" . }}
{{- end -}}
