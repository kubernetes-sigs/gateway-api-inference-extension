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
{{- /* Check if endpointsServer exists AND if createInferencePool is false */ -}}
{{- if and .Values.inferenceExtension.endpointsServer (not .Values.inferenceExtension.endpointsServer.createInferencePool) -}}
{{- /* LOGIC FOR STANDALONE EPP MODE */ -}}
epp: {{ include "gateway-api-inference-extension.name" . }}
{{- else -}}
{{- /* LOGIC FOR PARENT (INFERENCEPOOL) MODE */ -}}
inferencepool: {{ include "gateway-api-inference-extension.name" . }}
{{- end -}}
{{- end -}}

{{/*
Mode labels
*/}}
{{- define "gateway-api-inference-extension.modeLabels" -}}
{{- if and .Values.inferenceExtension.endpointsServer (not .Values.inferenceExtension.endpointsServer.createInferencePool) -}}
inference.networking.k8s.io/igw-mode: standalone
{{- else -}}
inference.networking.k8s.io/igw-mode: inferencepool
{{- end -}}
{{- end -}}


{{/*
Create a default fully qualified app name for inferenceGateway.
*/}}
{{- define "gateway-api-inference-extension.gateway.fullname" -}}
  {{- if .Values.experimentalHttpRoute.inferenceGatewayName -}}
    {{- .Values.experimentalHttpRoute.inferenceGatewayName | trunc 63 | trimSuffix "-" -}}
  {{- else -}}
    {{- printf "%s-inference-gateway" .Release.Name| trunc 63 | trimSuffix "-" -}}
  {{- end -}}
{{- end -}}

{{/*
Return the standalone sidecar proxy type.
*/}}
{{- define "gateway-api-inference-extension.sidecarProxyType" -}}
{{- $sidecar := .Values.inferenceExtension.sidecar | default dict -}}
{{- default "envoy" ($sidecar.proxyType | default "envoy") | lower -}}
{{- end -}}

{{/*
Return the resolved sidecar configuration for the current chart.
Standalone uses proxy presets merged with explicit sidecar overrides.
*/}}
{{- define "gateway-api-inference-extension.sidecar" -}}
{{- $sidecar := deepCopy (.Values.inferenceExtension.sidecar | default dict) -}}
{{- $resolved := $sidecar -}}
{{- if eq .Chart.Name "standalone" -}}
  {{- $proxyType := include "gateway-api-inference-extension.sidecarProxyType" . -}}
  {{- $presets := index $sidecar "presets" | default dict -}}
  {{- $preset := deepCopy ((index $presets $proxyType) | default dict) -}}
  {{- $resolved = mergeOverwrite $preset $sidecar -}}
{{- end -}}
{{- $resolved = omit $resolved "agentgateway" "presets" "proxyType" -}}
{{- toYaml $resolved -}}
{{- end -}}

{{/*
Return the rendered sidecar ConfigMap data.
*/}}
{{- define "gateway-api-inference-extension.sidecarConfigMapData" -}}
{{- $sidecar := include "gateway-api-inference-extension.sidecar" . | fromYaml | default dict -}}
{{- $configMap := index $sidecar "configMap" | default dict -}}
{{- $data := deepCopy ((index $configMap "data") | default dict) -}}
{{- if and (eq .Chart.Name "standalone") (eq (include "gateway-api-inference-extension.sidecarProxyType" .) "agentgateway") -}}
  {{- $generated := dict "config.yaml" (include "gateway-api-inference-extension.sidecar.agentgatewayConfig" .) -}}
  {{- $data = mergeOverwrite $data $generated -}}
{{- end -}}
{{- toYaml $data -}}
{{- end -}}

{{/*
Render labels from the standalone endpoint selector for the generated model Service.
Only equality-based selectors are supported because Service selectors are a map.
*/}}
{{- define "gateway-api-inference-extension.agentgateway.modelServiceSelectorLabels" -}}
{{- $selector := .Values.inferenceExtension.endpointsServer.endpointSelector | default "" -}}
{{- if empty $selector -}}
  {{- fail ".Values.inferenceExtension.endpointsServer.endpointSelector is required when creating an agentgateway model Service" -}}
{{- end -}}
{{- range $raw := splitList "," $selector }}
  {{- $part := trim $raw -}}
  {{- $kv := splitList "=" $part -}}
  {{- if ne (len $kv) 2 -}}
    {{- fail (printf ".Values.inferenceExtension.endpointsServer.endpointSelector must use comma-separated key=value labels when creating an agentgateway model Service, got %q" $selector) -}}
  {{- end -}}
  {{- $key := trim (index $kv 0) -}}
  {{- $value := trim (index $kv 1) -}}
  {{- if or (empty $key) (empty $value) -}}
    {{- fail (printf ".Values.inferenceExtension.endpointsServer.endpointSelector must use non-empty key=value labels when creating an agentgateway model Service, got %q" $selector) -}}
  {{- end -}}
{{- printf "%s: %s\n" ($key | quote) ($value | quote) -}}
{{- end -}}
{{- end -}}

{{/*
Render the default standalone agentgateway sidecar config template.
*/}}
{{- define "gateway-api-inference-extension.sidecar.agentgatewayConfig" -}}
{{- $sidecarValues := .Values.inferenceExtension.sidecar | default dict -}}
{{- $agentgateway := index $sidecarValues "agentgateway" | default dict -}}
{{- $service := index $agentgateway "service" | default dict -}}
{{- $serviceName := index $service "name" | default "" -}}
{{- $serviceNamespace := index $service "namespace" | default .Release.Namespace -}}
{{- $servicePort := index $service "port" | default 8000 -}}
{{- $listenerPort := 8081 -}}
{{- $servicePorts := .Values.inferenceExtension.extraServicePorts | default list -}}
{{- if gt (len $servicePorts) 0 -}}
  {{- $listenerPort = int ((index $servicePorts 0).targetPort | default (index $servicePorts 0).port) -}}
{{- end -}}
config:
  statsAddr: "0.0.0.0:15020"
  readinessAddr: "0.0.0.0:15021"
binds:
- port: {{ $listenerPort }}
  listeners:
  - name: default
    protocol: HTTP
    routes:
    - name: standalone-epp
      matches:
      - path:
          pathPrefix: /
      backends:
      - service:
          name: {{ printf "%s/%s" $serviceNamespace $serviceName | quote }}
          port: {{ $servicePort }}
        policies:
          inferenceRouting:
            endpointPicker:
              host: {{ printf "127.0.0.1:%v" (.Values.inferenceExtension.extProcPort | default 9002) | quote }}
services:
- name: {{ $serviceName | quote }}
  namespace: {{ $serviceNamespace | quote }}
  hostname: {{ $serviceName | quote }}
  vips: []
  ports:
    {{ $servicePort }}: {{ $servicePort }}
{{- end -}}
