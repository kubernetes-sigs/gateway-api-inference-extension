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
Normalize a scalar, comma-separated string, or list of ports into a
comma-separated numeric string.
*/}}
{{- define "gateway-api-inference-extension.normalizedPortList" -}}
{{- $path := .path -}}
{{- $value := .value -}}
{{- if empty $value -}}
  {{- fail (printf "%s is required" $path) -}}
{{- end -}}
{{- $rawPorts := list -}}
{{- if kindIs "slice" $value -}}
  {{- $rawPorts = $value -}}
{{- else -}}
  {{- $rawPorts = splitList "," (toString $value) -}}
{{- end -}}
{{- $ports := list -}}
{{- range $raw := $rawPorts -}}
  {{- $rawString := trim (toString $raw) -}}
  {{- if not (regexMatch "^[0-9]+$" $rawString) -}}
    {{- fail (printf "%s must contain only numeric ports, got %q" $path $rawString) -}}
  {{- end -}}
  {{- $port := int $rawString -}}
  {{- if or (lt $port 1) (gt $port 65535) -}}
    {{- fail (printf "%s must contain ports between 1 and 65535, got %d" $path $port) -}}
  {{- end -}}
  {{- $ports = append $ports (toString $port) -}}
{{- end -}}
{{- if eq (len $ports) 0 -}}
  {{- fail (printf "%s must contain at least one port" $path) -}}
{{- end -}}
{{- join "," $ports -}}
{{- end -}}

{{/*
Return the standalone proxy listener port exposed by the EPP Service.
The port is selected by the Service port named "http" so selection is
deterministic even when additional Service ports are configured.
*/}}
{{- define "gateway-api-inference-extension.standaloneProxyListenerPort" -}}
{{- $servicePorts := .Values.inferenceExtension.extraServicePorts | default list -}}
{{- $found := false -}}
{{- $listenerPort := "" -}}
{{- $targetPort := "" -}}
{{- $hasTargetPort := false -}}
{{- range $index, $servicePort := $servicePorts -}}
  {{- if eq (toString (index $servicePort "name")) "http" -}}
    {{- if $found -}}
      {{- fail ".Values.inferenceExtension.extraServicePorts must contain exactly one port named \"http\" when proxyType=agentgateway" -}}
    {{- end -}}
    {{- $found = true -}}
    {{- if not (hasKey $servicePort "port") -}}
      {{- fail (printf ".Values.inferenceExtension.extraServicePorts[%d].port is required for the port named \"http\"" $index) -}}
    {{- end -}}
    {{- $listenerPort = index $servicePort "port" -}}
    {{- if hasKey $servicePort "targetPort" -}}
      {{- $hasTargetPort = true -}}
      {{- $targetPort = index $servicePort "targetPort" -}}
    {{- end -}}
  {{- end -}}
{{- end -}}
{{- if not $found -}}
  {{- fail ".Values.inferenceExtension.extraServicePorts must contain exactly one port named \"http\" when proxyType=agentgateway" -}}
{{- end -}}
{{- if kindIs "slice" $listenerPort -}}
  {{- fail ".Values.inferenceExtension.extraServicePorts[name=http].port must be a single numeric port" -}}
{{- end -}}
{{- $listenerPortString := trim (toString $listenerPort) -}}
{{- if not (regexMatch "^[0-9]+$" $listenerPortString) -}}
  {{- fail (printf ".Values.inferenceExtension.extraServicePorts[name=http].port must be numeric, got %q" $listenerPortString) -}}
{{- end -}}
{{- $listenerPortNumber := int $listenerPortString -}}
{{- if or (lt $listenerPortNumber 1) (gt $listenerPortNumber 65535) -}}
  {{- fail (printf ".Values.inferenceExtension.extraServicePorts[name=http].port must be between 1 and 65535, got %d" $listenerPortNumber) -}}
{{- end -}}
{{- if $hasTargetPort -}}
  {{- $targetPortString := trim (toString $targetPort) -}}
  {{- if and (ne $targetPortString $listenerPortString) (ne $targetPortString "http") -}}
    {{- fail (printf ".Values.inferenceExtension.extraServicePorts[name=http].targetPort must be omitted, %q, or \"http\" when proxyType=agentgateway, got %q" $listenerPortString $targetPortString) -}}
  {{- end -}}
{{- end -}}
{{- $listenerPortString -}}
{{- end -}}

{{/*
Return the standalone EPP model-server target ports.
*/}}
{{- define "gateway-api-inference-extension.standaloneEndpointTargetPorts" -}}
{{- include "gateway-api-inference-extension.normalizedPortList" (dict "path" ".Values.inferenceExtension.endpointsServer.targetPorts" "value" .Values.inferenceExtension.endpointsServer.targetPorts) -}}
{{- end -}}

{{/*
Return the agentgateway model Service ports.
*/}}
{{- define "gateway-api-inference-extension.agentgateway.modelServicePorts" -}}
{{- $sidecarValues := .Values.inferenceExtension.sidecar | default dict -}}
{{- $agentgateway := index $sidecarValues "agentgateway" | default dict -}}
{{- $service := index $agentgateway "service" | default dict -}}
{{- include "gateway-api-inference-extension.normalizedPortList" (dict "path" ".Values.inferenceExtension.sidecar.agentgateway.service.ports" "value" (index $service "ports")) -}}
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
  {{- if eq $proxyType "agentgateway" -}}
    {{- $listenerPort := include "gateway-api-inference-extension.standaloneProxyListenerPort" . | int -}}
    {{- $ports := index $resolved "ports" | default list -}}
    {{- $resolvedPorts := list (dict "containerPort" $listenerPort "name" "http") -}}
    {{- range $index, $port := $ports -}}
      {{- if gt $index 0 -}}
        {{- $resolvedPorts = append $resolvedPorts $port -}}
      {{- end -}}
    {{- end -}}
    {{- $_ := set $resolved "ports" $resolvedPorts -}}
  {{- end -}}
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
{{- $servicePorts := splitList "," (include "gateway-api-inference-extension.agentgateway.modelServicePorts" .) -}}
{{- $backendPort := index $servicePorts 0 -}}
{{- $listenerPort := include "gateway-api-inference-extension.standaloneProxyListenerPort" . | int -}}
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
          port: {{ $backendPort }}
        policies:
          inferenceRouting:
            endpointPicker:
              host: {{ printf "127.0.0.1:%v" (.Values.inferenceExtension.extProcPort | default 9002) | quote }}
            destinationMode: passthrough
services:
- name: {{ $serviceName | quote }}
  namespace: {{ $serviceNamespace | quote }}
  hostname: {{ $serviceName | quote }}
  vips: []
  ports:
    {{- range $servicePort := $servicePorts }}
    {{ $servicePort }}: {{ $servicePort }}
    {{- end }}
{{- end -}}
