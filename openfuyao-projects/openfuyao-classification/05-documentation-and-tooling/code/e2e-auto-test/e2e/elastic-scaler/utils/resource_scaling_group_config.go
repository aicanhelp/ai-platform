package utils

import "k8s.io/apimachinery/pkg/runtime/schema"

const (
	RSGHelmReleaseName          = "rsg"
	RSGHelmChart                = "oci://cr.openfuyao.cn/charts/pd-orchestrator"
	RSGHelmVersion              = "0.0.0-latest"
	RSGControllerDeploymentName = "resourcescalinggroup-controller"
	RSGWebhookServiceName       = "webhook-service"
	RSGWebhookTestLabelKey      = "e2e.openfuyao.com/suite"
	RSGWebhookTestLabelValue    = "resourcescalinggroup-webhook"
	RSGValidatingWebhookName    = RSGHelmReleaseName + "-resourcescalinggroup-validating-webhook"
	RSGMutatingWebhookName      = RSGHelmReleaseName + "-resourcescalinggroup-mutating-webhook"

	RSGLWSHelmReleaseName = "rsg-lws"
	RSGLWSHelmChart       = "oci://registry.k8s.io/lws/charts/lws"
	RSGLWSHelmVersion     = "0.8.0"
	RSGLWSNamespace       = "lws-system"

	RSGLWSMutatingWebhook   = "lws-mutating-webhook-configuration"
	RSGLWSValidatingWebhook = "lws-validating-webhook-configuration"
)

// RSG 子 chart 的 post-install hook（生成 webhook 证书 / 注册 webhook）activeDeadlineSeconds 为 1200s，
// helm 默认 --timeout 5m 会在 hook 跑完前先报 "timed out waiting for the condition"，故显式放大。
const RSGHelmTimeout = "20m"

// RSGTestNamespace RSG controller 与测试资源的命名空间，统一指向 e2e 专属命名空间（不再用 chart 默认的 scaling-system）。
var RSGTestNamespace = ElasticScalerE2ENamespace

var ResourceScalingGroupGVR = schema.GroupVersionResource{
	Group:    "autoscaling.openfuyao.com",
	Version:  "v1alpha1",
	Resource: "resourcescalinggroups",
}

var LeaderWorkerSetGVR = schema.GroupVersionResource{
	Group:    "leaderworkerset.x-k8s.io",
	Version:  "v1",
	Resource: "leaderworkersets",
}

var LeaderWorkerSetFallbackGVR = schema.GroupVersionResource{
	Group:    "scheduling.k8s.io",
	Version:  "v1alpha1",
	Resource: "leaderworkersets",
}

var LeaderWorkerSetAPIVersion = "leaderworkerset.x-k8s.io/v1"
