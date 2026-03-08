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
      {{- fail ".Values.inferenceExtension.endpointsServer.createInferencePool=false is required when proxyType=agentgateway; standalone agentgateway uses EPP endpoint discovery with a logical service backend" -}}
    {{- end -}}
    {{- if hasKey $sidecar "agentgateway" -}}
      {{- fail ".Values.inferenceExtension.sidecar.agentgateway is no longer supported; standalone agentgateway derives its logical backend from endpointsServer settings" -}}
    {{- end -}}
    {{- $_ := include "gateway-api-inference-extension.standaloneEndpointTargetPorts" . -}}
    {{- $_ := include "gateway-api-inference-extension.standaloneProxyListenerPort" . -}}
    {{- $flags := .Values.inferenceExtension.flags | default dict -}}
    {{- if and (hasKey $flags "secure-serving") (ne (toString (index $flags "secure-serving")) "false") -}}
      {{- fail ".Values.inferenceExtension.flags.secure-serving must be false when proxyType=agentgateway; standalone agentgateway uses plaintext gRPC to EPP over localhost" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- end -}}
