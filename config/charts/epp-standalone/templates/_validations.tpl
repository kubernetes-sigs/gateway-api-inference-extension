{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" }}
{{- if not (and .Values.inferenceExtension.endpointsServer .Values.inferenceExtension.endpointsServer.standalone) }}
{{- if or (empty $.Values.inferencePool.modelServers) (not $.Values.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- end }}
{{- end -}}
