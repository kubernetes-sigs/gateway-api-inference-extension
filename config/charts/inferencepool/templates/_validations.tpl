{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" -}}
{{- if or (empty $.Values.global.inferencePool.modelServers) (not $.Values.global.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.global.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- end -}}
