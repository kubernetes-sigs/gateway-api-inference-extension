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
    {{- if hasKey $service "port" -}}
      {{- fail ".Values.inferenceExtension.sidecar.agentgateway.service.port has been replaced by .Values.inferenceExtension.sidecar.agentgateway.service.ports" -}}
    {{- end -}}
    {{- if empty $serviceName -}}
      {{- fail ".Values.inferenceExtension.sidecar.agentgateway.service.name is required when proxyType=agentgateway" -}}
    {{- end -}}
    {{- $targetPorts := include "gateway-api-inference-extension.standaloneEndpointTargetPorts" . -}}
    {{- $servicePorts := include "gateway-api-inference-extension.agentgateway.modelServicePorts" . -}}
    {{- if ne $targetPorts $servicePorts -}}
      {{- fail (printf ".Values.inferenceExtension.sidecar.agentgateway.service.ports must match .Values.inferenceExtension.endpointsServer.targetPorts when proxyType=agentgateway, got service ports %q and target ports %q" $servicePorts $targetPorts) -}}
    {{- end -}}
    {{- $listenerPort := include "gateway-api-inference-extension.standaloneProxyListenerPort" . -}}
    {{- $flags := .Values.inferenceExtension.flags | default dict -}}
    {{- if and (hasKey $flags "secure-serving") (ne (toString (index $flags "secure-serving")) "false") -}}
      {{- fail ".Values.inferenceExtension.flags.secure-serving must be false when proxyType=agentgateway; standalone agentgateway uses plaintext gRPC to EPP over localhost" -}}
    {{- end -}}
    {{- if $serviceCreate -}}
      {{- $selectorLabels := include "gateway-api-inference-extension.agentgateway.modelServiceSelectorLabels" . -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
