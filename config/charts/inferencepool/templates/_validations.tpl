{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" -}}
{{- if or (empty $.Values.inferencePool.modelServers) (not $.Values.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- if or (not $.Values.inferenceExtension.resources) (not $.Values.inferenceExtension.resources.requests) (not $.Values.inferenceExtension.resources.requests.cpu) }}
{{- fail ".Values.inferenceExtension.resources.requests.cpu is required" }}
{{- end }}
{{- if or (not $.Values.inferenceExtension.resources) (not $.Values.inferenceExtension.resources.requests) (not $.Values.inferenceExtension.resources.requests.memory) }}
{{- fail ".Values.inferenceExtension.resources.requests.memory is required" }}
{{- end }}
{{- end -}}
