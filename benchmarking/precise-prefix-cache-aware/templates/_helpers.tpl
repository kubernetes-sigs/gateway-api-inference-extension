{{/*
Expand the name of the chart.
*/}}
{{- define "precise-prefix-cache-aware.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "precise-prefix-cache-aware.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "precise-prefix-cache-aware.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "precise-prefix-cache-aware.labels" -}}
helm.sh/chart: {{ include "precise-prefix-cache-aware.chart" . }}
{{ include "precise-prefix-cache-aware.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "precise-prefix-cache-aware.selectorLabels" -}}
app.kubernetes.io/name: {{ include "precise-prefix-cache-aware.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Config Mount Path
*/}}
{{- define "precise-prefix-cache-aware.configMount" -}}
{{- print "/etc/inference-perf" -}}
{{- end }}

{{/*
Hugging Face Secret Name
*/}}
{{- define "precise-prefix-cache-aware.hfSecret" -}}
{{- printf "%s-hf-secret" (include "precise-prefix-cache-aware.fullname" .) -}}
{{- end }}

{{/*
Hugging Face Secret Key
*/}}
{{- define "precise-prefix-cache-aware.hfKey" -}}
{{- print "token" -}}
{{- end }}