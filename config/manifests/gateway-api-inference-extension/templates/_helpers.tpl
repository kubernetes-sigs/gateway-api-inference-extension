{{/*
Common labels
*/}}
{{- define "gateway-api-inference-extension.labels" -}}
app.kubernetes.io/name: {{ .Values.inferenceExtension.name }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
app: {{ .Values.inferenceExtension.name }}
{{- end -}}