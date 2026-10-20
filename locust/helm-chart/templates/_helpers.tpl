{{- define "modsim-locust.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "modsim-locust.fullname" -}}
{{- printf "%s-%s" .Release.Name (include "modsim-locust.name" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{- define "modsim-locust.labels" -}}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
app.kubernetes.io/name: {{ include "modsim-locust.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "modsim-locust.selectorLabels" -}}
app.kubernetes.io/name: {{ include "modsim-locust.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
