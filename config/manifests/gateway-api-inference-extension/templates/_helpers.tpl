{{/*
Common labels
*/}}
{{- define "gateway-api-inference-extension.labels" -}}
app.kubernetes.io/name: epp-{{ .Values.inferencePool.name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Inference extension name
*/}}
{{- define "gateway-api-inference-extension.name" -}}
{{- $base := .Values.inferencePool.name | default "default-pool" | lower | trim | trunc 40 -}}
epp-{{ $base }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
app: epp-{{ .Values.inferencePool.name }}
{{- end -}}
