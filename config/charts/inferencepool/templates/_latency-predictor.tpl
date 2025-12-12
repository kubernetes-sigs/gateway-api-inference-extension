{{/*
Latency Predictor Env
Supports both legacy mode (single predictor) and PD mode (multiple predictors)
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.env" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
{{- if .Values.inferenceExtension.latencyPredictor.pdMode.enabled }}
{{/* PD Mode: Generate environment variables for each predictor type */}}
{{- range $predictorName, $predictorConfig := .Values.inferenceExtension.latencyPredictor.pdMode.predictors }}
- name: {{ $predictorName | upper }}_TRAINING_URL
  value: "http://localhost:{{ $predictorConfig.trainingServer.port }}"
- name: {{ $predictorName | upper }}_PREDICTION_URL
  value: "{{- $count := int $predictorConfig.predictionServers.count -}}
          {{- $startPort := int $predictorConfig.predictionServers.startPort -}}
          {{- range $i := until $count -}}
            {{- if $i }},{{ end }}http://localhost:{{ add $startPort $i }}
          {{- end }}"
{{- end }}
{{- else }}
{{/* Legacy Mode: Single predictor environment variables */}}
- name: PREDICTION_SERVER_URL
  value: "{{- $count := int .Values.inferenceExtension.latencyPredictor.predictionServers.count -}}
          {{- $startPort := int .Values.inferenceExtension.latencyPredictor.predictionServers.startPort -}}
          {{- range $i := until $count -}}
            {{- if $i }},{{ end }}http://localhost:{{ add $startPort $i }}
          {{- end }}"
- name: TRAINING_SERVER_URL
  value: "http://localhost:{{ .Values.inferenceExtension.latencyPredictor.trainingServer.port }}"
{{- end }}
{{- range $key, $value := .Values.inferenceExtension.latencyPredictor.eppEnv }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Latency Predictor Sidecar Containers
Supports both legacy mode (single predictor) and PD mode (multiple predictors)
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.containers" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
{{- if .Values.inferenceExtension.latencyPredictor.pdMode.enabled }}
{{/* PD Mode: Create training and prediction servers for each predictor type */}}
{{- range $predictorName, $predictorConfig := .Values.inferenceExtension.latencyPredictor.pdMode.predictors }}
# Training Server for {{ $predictorName }} predictor
- name: training-server-{{ $predictorName }}
  image: {{ $predictorConfig.trainingServer.image.hub | default $.Values.inferenceExtension.latencyPredictor.trainingServer.image.hub }}/{{ $predictorConfig.trainingServer.image.name | default $.Values.inferenceExtension.latencyPredictor.trainingServer.image.name }}:{{ $predictorConfig.trainingServer.image.tag | default $.Values.inferenceExtension.latencyPredictor.trainingServer.image.tag }}
  imagePullPolicy: {{ $predictorConfig.trainingServer.image.pullPolicy | default $.Values.inferenceExtension.latencyPredictor.trainingServer.image.pullPolicy }}
  command: ["uvicorn"]
  args:
  - "training_server:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "{{ $predictorConfig.trainingServer.port }}"
  ports:
  - containerPort: {{ $predictorConfig.trainingServer.port }}
    name: train-{{ $predictorName }}
  livenessProbe:
    httpGet:
      path: {{ $predictorConfig.trainingServer.livenessProbe.httpGet.path | default "/healthz" }}
      port: {{ $predictorConfig.trainingServer.port }}
    initialDelaySeconds: {{ $predictorConfig.trainingServer.livenessProbe.initialDelaySeconds | default 30 }}
    periodSeconds: {{ $predictorConfig.trainingServer.livenessProbe.periodSeconds | default 20 }}
  readinessProbe:
    httpGet:
      path: {{ $predictorConfig.trainingServer.readinessProbe.httpGet.path | default "/readyz" }}
      port: {{ $predictorConfig.trainingServer.port }}
    initialDelaySeconds: {{ $predictorConfig.trainingServer.readinessProbe.initialDelaySeconds | default 45 }}
    periodSeconds: {{ $predictorConfig.trainingServer.readinessProbe.periodSeconds | default 10 }}
  resources:
    {{- toYaml ($predictorConfig.trainingServer.resources | default $.Values.inferenceExtension.latencyPredictor.trainingServer.resources) | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "gateway-api-inference-extension.name" $ }}-latency-predictor-{{ $predictorName }}-training
  env:
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "training-{{ $predictorName }}"
  volumeMounts:
  - name: training-server-{{ $predictorName }}-storage
    mountPath: /models
{{- range $i := until (int $predictorConfig.predictionServers.count) }}
# Prediction Server {{ add $i 1 }} for {{ $predictorName }} predictor
- name: prediction-server-{{ $predictorName }}-{{ add $i 1 }}
  image: {{ $predictorConfig.predictionServers.image.hub | default $.Values.inferenceExtension.latencyPredictor.predictionServers.image.hub }}/{{ $predictorConfig.predictionServers.image.name | default $.Values.inferenceExtension.latencyPredictor.predictionServers.image.name }}:{{ $predictorConfig.predictionServers.image.tag | default $.Values.inferenceExtension.latencyPredictor.predictionServers.image.tag }}
  imagePullPolicy: {{ $predictorConfig.predictionServers.image.pullPolicy | default $.Values.inferenceExtension.latencyPredictor.predictionServers.image.pullPolicy }}
  command: ["uvicorn"]
  args: ["prediction_server:app", "--host", "0.0.0.0", "--port", "{{ add $predictorConfig.predictionServers.startPort $i }}"]
  ports:
  - containerPort: {{ add $predictorConfig.predictionServers.startPort $i }}
    name: pred-{{ $predictorName }}-{{ add $i 1 }}
  livenessProbe:
    httpGet:
      path: {{ $predictorConfig.predictionServers.livenessProbe.httpGet.path | default "/healthz" }}
      port: {{ add $predictorConfig.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $predictorConfig.predictionServers.livenessProbe.initialDelaySeconds | default 15 }}
    periodSeconds: {{ $predictorConfig.predictionServers.livenessProbe.periodSeconds | default 15 }}
  readinessProbe:
    httpGet:
      path: {{ $predictorConfig.predictionServers.readinessProbe.httpGet.path | default "/readyz" }}
      port: {{ add $predictorConfig.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $predictorConfig.predictionServers.readinessProbe.initialDelaySeconds | default 10 }}
    periodSeconds: {{ $predictorConfig.predictionServers.readinessProbe.periodSeconds | default 5 }}
    failureThreshold: {{ $predictorConfig.predictionServers.readinessProbe.failureThreshold | default 10 }}
  resources:
    {{- toYaml ($predictorConfig.predictionServers.resources | default $.Values.inferenceExtension.latencyPredictor.predictionServers.resources) | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "gateway-api-inference-extension.name" $ }}-latency-predictor-{{ $predictorName }}-prediction
  env:
  - name: PREDICT_PORT
    value: "{{ add $predictorConfig.predictionServers.startPort $i }}"
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "prediction-{{ $predictorName }}-{{ add $i 1 }}"
  - name: TRAINING_SERVER_URL
    value: "http://localhost:{{ $predictorConfig.trainingServer.port }}"
  volumeMounts:
  - name: prediction-server-{{ $predictorName }}-{{ add $i 1 }}-storage
    mountPath: /server_models
{{- end }}
{{- end }}
{{- else }}
{{/* Legacy Mode: Single predictor containers */}}
# Training Server Sidecar Container
- name: training-server
  image: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.hub }}/{{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.name }}:{{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.tag }}
  imagePullPolicy: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.pullPolicy }}
  command: ["uvicorn"]
  args:
  - "training_server:app"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "{{ .Values.inferenceExtension.latencyPredictor.trainingServer.port }}"
  ports:
  - containerPort: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.port }}
    name: training-port
  livenessProbe:
    {{- toYaml .Values.inferenceExtension.latencyPredictor.trainingServer.livenessProbe | nindent 4 }}
  readinessProbe:
    {{- toYaml .Values.inferenceExtension.latencyPredictor.trainingServer.readinessProbe | nindent 4 }}
  resources:
    {{- toYaml .Values.inferenceExtension.latencyPredictor.trainingServer.resources | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "gateway-api-inference-extension.name" . }}-latency-predictor-training
  env:
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "training"
  volumeMounts:
  - name: training-server-storage
    mountPath: /models
{{- range $i := until (int .Values.inferenceExtension.latencyPredictor.predictionServers.count) }}
# Prediction Server Sidecar Container {{ add $i 1 }}
- name: prediction-server-{{ add $i 1 }}
  image: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.hub }}/{{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.name }}:{{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.tag }}
  imagePullPolicy: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.pullPolicy }}
  command: ["uvicorn"]
  args: ["prediction_server:app", "--host", "0.0.0.0", "--port", "{{ add $.Values.inferenceExtension.latencyPredictor.predictionServers.startPort $i }}"]
  ports:
  - containerPort: {{ add $.Values.inferenceExtension.latencyPredictor.predictionServers.startPort $i }}
    name: predict-port-{{ add $i 1 }}
  livenessProbe:
    httpGet:
      path: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.livenessProbe.httpGet.path }}
      port: {{ add $.Values.inferenceExtension.latencyPredictor.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.livenessProbe.initialDelaySeconds }}
    periodSeconds: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.livenessProbe.periodSeconds }}
  readinessProbe:
    httpGet:
      path: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.readinessProbe.httpGet.path }}
      port: {{ add $.Values.inferenceExtension.latencyPredictor.predictionServers.startPort $i }}
    initialDelaySeconds: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.readinessProbe.initialDelaySeconds }}
    periodSeconds: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.readinessProbe.periodSeconds }}
    failureThreshold: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.readinessProbe.failureThreshold }}
  resources:
    {{- toYaml $.Values.inferenceExtension.latencyPredictor.predictionServers.resources | nindent 4 }}
  envFrom:
  - configMapRef:
      name: {{ include "gateway-api-inference-extension.name" $ }}-latency-predictor-prediction
  env:
  - name: PREDICT_PORT
    value: "{{ add $.Values.inferenceExtension.latencyPredictor.predictionServers.startPort $i }}"
  - name: POD_NAME
    valueFrom:
      fieldRef:
        fieldPath: metadata.name
  - name: SERVER_TYPE
    value: "prediction-{{ add $i 1 }}"
  - name: TRAINING_SERVER_URL
    value: "http://localhost:{{ $.Values.inferenceExtension.latencyPredictor.trainingServer.port }}"
  volumeMounts:
  - name: prediction-server-{{ add $i 1 }}-storage
    mountPath: /server_models
{{- end }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Latency Predictor Volumes
Supports both legacy mode (single predictor) and PD mode (multiple predictors)
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.volumes" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
{{- if .Values.inferenceExtension.latencyPredictor.pdMode.enabled }}
{{/* PD Mode: Create volumes for each predictor type */}}
{{- range $predictorName, $predictorConfig := .Values.inferenceExtension.latencyPredictor.pdMode.predictors }}
- name: training-server-{{ $predictorName }}-storage
  emptyDir:
    sizeLimit: {{ $predictorConfig.trainingServer.volumeSize | default $.Values.inferenceExtension.latencyPredictor.trainingServer.volumeSize }}
{{- range $i := until (int $predictorConfig.predictionServers.count) }}
- name: prediction-server-{{ $predictorName }}-{{ add $i 1 }}-storage
  emptyDir:
    sizeLimit: {{ $predictorConfig.predictionServers.volumeSize | default $.Values.inferenceExtension.latencyPredictor.predictionServers.volumeSize }}
{{- end }}
{{- end }}
{{- else }}
{{/* Legacy Mode: Single predictor volumes */}}
- name: training-server-storage
  emptyDir:
    sizeLimit: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.volumeSize }}
{{- range $i := until (int .Values.inferenceExtension.latencyPredictor.predictionServers.count) }}
- name: prediction-server-{{ add $i 1 }}-storage
  emptyDir:
    sizeLimit: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.volumeSize }}
{{- end }}
{{- end }}
{{- end }}
{{- end }}
