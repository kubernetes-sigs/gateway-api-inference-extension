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
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 -}}
{{ $base }}-epp
{{- end -}}

{{/*
Cluster RBAC unique name
*/}}
{{- define "gateway-api-inference-extension.cluster-rbac-name" -}}
{{- $base := .Release.Name | default "default-pool" | lower | trim | trunc 40 }}
{{- $ns := .Release.Namespace | default "default" | lower | trim | trunc 40 }}
{{- printf "%s-%s-epp" $base $ns | quote | trunc 84 }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
{{- /* Check if endpointsServer exists AND if standalone is true */ -}}
{{- if and .Values.inferenceExtension.endpointsServer .Values.inferenceExtension.endpointsServer.standalone -}}
{{- /* LOGIC FOR STANDALONE EPP MODE */ -}}
epp: {{ include "gateway-api-inference-extension.name" . }}
{{- else -}}
{{- /* LOGIC FOR PARENT (INFERENCEPOOL) MODE */ -}}
inferencepool: {{ include "gateway-api-inference-extension.name" . }}
{{- end -}}
{{- end -}}
