package resources

import "k8s.io/apimachinery/pkg/types"

const (
	AppBackendNamespace  = "gateway-conformance-app-backend"
	InfraNamespace       = "gateway-conformance-infra"
	PrimaryGatewayName   = "conformance-primary"
	SecondaryGatewayName = "conformance-secondary"

	PrimaryInferencePoolName   = "primary-inference-pool"
	SecondaryInferencePoolName = "secondary-inference-pool"

	PrimaryModelServerAppLabel         = "primary-inference-model-server"
	SecondaryModelServerAppLabel       = "secondary-inference-model-server"
	PrimaryModelServerDeploymentName   = "primary-inference-model-server-deployment"
	SecondaryModelServerDeploymentName = "secondary-inference-model-server-deployment"

	ModelServerPodReplicas = 3
)

var (
	PrimaryGatewayNN   = types.NamespacedName{Name: PrimaryGatewayName, Namespace: InfraNamespace}
	SecondaryGatewayNN = types.NamespacedName{Name: SecondaryGatewayName, Namespace: InfraNamespace}

	PrimaryInferencePoolNN   = types.NamespacedName{Name: PrimaryInferencePoolName, Namespace: AppBackendNamespace}
	SecondaryInferencePoolNN = types.NamespacedName{Name: SecondaryInferencePoolName, Namespace: AppBackendNamespace}

	PrimaryEppDeploymentNN   = types.NamespacedName{Name: "primary-app-endpoint-picker", Namespace: AppBackendNamespace}
	SecondaryEppDeploymentNN = types.NamespacedName{Name: "secondary-app-endpoint-picker", Namespace: AppBackendNamespace}
)
