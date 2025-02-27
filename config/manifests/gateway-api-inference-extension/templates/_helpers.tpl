{{- define "httpRoute.name" -}}
llm-route-{{ .Release.Name }}
{{- end -}}

{{- define "backend.name" -}}
backend-{{ .Release.Name }}
{{- end -}}

{{- define "gatewayClass.name" -}}
inference-gateway-{{ .Release.Name }}
{{- end -}}

{{- define "gateway.name" -}}
inference-gateway-{{ .Release.Name }}
{{- end -}}

{{- define "envoyExtensionPolicy.name" -}}
ext-proc-policy-{{ .Release.Name }}
{{- end -}}

{{- define "envoyPatchPolicy.name" -}}
custom-response-patch-policy-{{ .Release.Name }}
{{- end -}}

{{/*
Selector labels
*/}}
{{- define "gateway-api-inference-extension.selectorLabels" -}}
app: {{ include "gateway-api-inference-extension.name" . }}
{{- end -}}

{{- define "clusterRole.name" -}}
inference-extension-{{ .Release.Namespace }}-{{ .Release.Name }}
{{- end -}}

{{- define "backendTrafficPolicy.name" -}}
high-connection-route-policy-{{ .Release.Name }}
{{- end -}}

{{- define "gateway-api-inference-extension.name" -}}
inference-gateway-ext-proc-{{ .Release.Name }}
{{- end -}}
