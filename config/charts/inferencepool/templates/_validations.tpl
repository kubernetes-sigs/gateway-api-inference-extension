{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" -}}
{{- if not $.Values.inferencePool.name }}
{{- fail "missing .Values.inferencePool.name" }}
{{- end }}


{{- if or (empty $.Values.inferencePool.modelServers) (not $.Values.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- end -}}
