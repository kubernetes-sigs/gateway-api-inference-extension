{{/*
Latency Predictor Env
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.env" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
- name: PREDICTION_SERVER_URL
  value: "{{- $count := int .Values.inferenceExtension.latencyPredictor.predictionServers.count -}}
          {{- $startPort := int .Values.inferenceExtension.latencyPredictor.predictionServers.startPort -}}
          {{- range $i := until $count -}}
            {{- if $i }},{{ end }}http://localhost:{{ add $startPort $i }}
          {{- end }}"
- name: TRAINING_SERVER_URL
  value: "http://localhost:{{ .Values.inferenceExtension.latencyPredictor.trainingServer.port }}"
{{- range $key, $value := .Values.inferenceExtension.latencyPredictor.eppEnv }}
- name: {{ $key }}
  value: {{ $value | quote }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Latency Predictor Sidecar Containers
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.containers" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
# Training Server Sidecar Container
- name: training-server
  image: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.registry }}/{{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.repository }}:{{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.tag }}
  imagePullPolicy: {{ .Values.inferenceExtension.latencyPredictor.trainingServer.image.pullPolicy }}
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
  image: {{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.registry }}/{{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.repository }}:{{ $.Values.inferenceExtension.latencyPredictor.predictionServers.image.tag }}
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

{{/*
Latency Predictor Volumes
*/}}
{{- define "gateway-api-inference-extension.latencyPredictor.volumes" -}}
{{- if .Values.inferenceExtension.latencyPredictor.enabled }}
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
