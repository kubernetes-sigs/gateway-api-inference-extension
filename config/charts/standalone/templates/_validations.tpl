{{/*
common validations
*/}}
{{- define "gateway-api-inference-extension.validations.inferencepool.common" }}
{{- if and .Values.inferenceExtension.endpointsServer .Values.inferenceExtension.endpointsServer.createInferencePool }}
{{- if or (empty $.Values.inferencePool.modelServers) (not $.Values.inferencePool.modelServers.matchLabels) }}
{{- fail ".Values.inferencePool.modelServers.matchLabels is required" }}
{{- end }}
{{- end }}
{{- end -}}

{{/*
standalone validations
*/}}
{{- define "gateway-api-inference-extension.validations.standalone" -}}
{{- $sidecar := .Values.inferenceExtension.sidecar | default dict -}}
{{- if $sidecar.enabled -}}
  {{- $proxyType := default "envoy" ($sidecar.proxyType | default "envoy") | lower -}}
  {{- if not (or (eq $proxyType "envoy") (eq $proxyType "agentgateway")) -}}
    {{- fail (printf ".Values.inferenceExtension.sidecar.proxyType must be one of [envoy, agentgateway], got %q" $proxyType) -}}
  {{- end -}}
  {{- if eq $proxyType "agentgateway" -}}
    {{- if and .Values.inferenceExtension.endpointsServer .Values.inferenceExtension.endpointsServer.createInferencePool -}}
      {{- fail ".Values.inferenceExtension.endpointsServer.createInferencePool=false is required when proxyType=agentgateway; standalone agentgateway currently supports only service-backed routing" -}}
    {{- end -}}
    {{- $agentgateway := index $sidecar "agentgateway" | default dict -}}
    {{- $service := index $agentgateway "service" | default dict -}}
    {{- $serviceName := index $service "name" | default "" -}}
    {{- $serviceCreate := index $service "create" | default true -}}
    {{- if empty $serviceName -}}
      {{- fail ".Values.inferenceExtension.sidecar.agentgateway.service.name is required when proxyType=agentgateway" -}}
    {{- end -}}
    {{- if $serviceCreate -}}
      {{- $selectorLabels := include "gateway-api-inference-extension.agentgateway.modelServiceSelectorLabels" . -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
