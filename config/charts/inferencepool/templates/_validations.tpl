{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" -}}
{{- if or (empty $.Values.inferencePool.modelServers) (not $.Values.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- end -}}

{{/*
EPP resource validations
*/}}
{{- define "gateway-api-inference-extension.validations.epp.resources" -}}
{{- if not .Values.inferenceExtension.resources }}
{{- fail ".Values.inferenceExtension.resources is required. EPP is a critical component that must have resource requests set." }}
{{- end }}
{{- if not .Values.inferenceExtension.resources.requests }}
{{- fail ".Values.inferenceExtension.resources.requests is required. EPP is a critical component that must have resource requests set." }}
{{- end }}
{{- $_ := required ".Values.inferenceExtension.resources.requests.cpu is required. EPP is a critical component that must have CPU requests set." .Values.inferenceExtension.resources.requests.cpu }}
{{- $_ := required ".Values.inferenceExtension.resources.requests.memory is required. EPP is a critical component that must have memory requests set." .Values.inferenceExtension.resources.requests.memory }}
{{- end -}}
