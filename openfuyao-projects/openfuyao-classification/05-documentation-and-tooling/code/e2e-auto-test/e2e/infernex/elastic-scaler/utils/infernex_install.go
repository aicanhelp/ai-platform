package utils

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	infernexUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

// InfernexBackendMode 为 InferNex chart inference-backend.services[0].mode。
type InfernexBackendMode string

const (
	InfernexBackendAggregated InfernexBackendMode = "aggregated"
	InfernexBackendPD         InfernexBackendMode = "pd"
)

const infernexInstallWaitTimeout = 10 * time.Minute

// InstallInfernex 是 elastic-scaler 套件专用安装入口
// gateway / hermes-router / cache-indexer / eagle-eye 关闭
// ES / RSG / Tidal 控制器开启，不提供默认 ES / RSG CR
// 支持聚合与 PD 分离两种后端模式
func InstallInfernex(clientset *kubernetes.Clientset, ctx context.Context, releaseName, namespace string, backend InfernexBackendMode) error {
	if backend != InfernexBackendAggregated && backend != InfernexBackendPD {
		return fmt.Errorf("unsupported infernex backend mode %q, want %q or %q", backend, InfernexBackendAggregated, InfernexBackendPD)
	}

	chart := infernexUtils.ESInfernexHelmChart
	version := infernexUtils.ESInfernexHelmVersion
	args := []string{
		"install",
		releaseName,
		chart,
		"-n", namespace,
		"--create-namespace",
	}
	isLocal := chart != "" && !strings.HasPrefix(chart, "oci://")
	if !isLocal && strings.TrimSpace(version) != "" {
		args = append(args, "--version", version)
	}
	args = append(args, infernexCommonHelmSets()...)
	if backend == InfernexBackendAggregated {
		args = append(args, infernexAggregatedBackendHelmSets()...)
	}

	cmd := exec.Command("helm", args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("helm install infernex %s/%s failed: %v\nOutput: %s", namespace, releaseName, err, string(out))
	}
	GinkgoWriter.Printf("helm install infernex %s/%s (%s) output: %s\n", namespace, releaseName, backend, string(out))

	waitInfernexControllersAndSharedWorkloads(clientset, ctx, releaseName, namespace, infernexInstallWaitTimeout)
	return nil
}

// UninstallInfernex 卸载 elastic-scaler 套件安装的 InferNex release。
func UninstallInfernex(releaseName, namespace string) error {
	cmd := exec.Command("helm", "uninstall", releaseName, "-n", namespace)
	out, err := cmd.CombinedOutput()
	if err != nil {
		if strings.Contains(string(out), "release: not found") {
			GinkgoWriter.Printf("helm release %s/%s 不存在，跳过卸载\n", namespace, releaseName)
			return nil
		}
		return fmt.Errorf("helm uninstall infernex %s/%s failed: %v\nOutput: %s", namespace, releaseName, err, string(out))
	}
	GinkgoWriter.Printf("helm uninstall infernex %s/%s output: %s\n", namespace, releaseName, string(out))
	return nil
}

// CreatePDResourceScalingGroup 在默认 RSG CR 关闭后，按 chart instanceConfig 创建 PD 组 RSG。
func CreatePDResourceScalingGroup(dynamicClient dynamic.Interface, ctx context.Context, namespace, rsgName string) error {
	obj := &unstructured.Unstructured{
		Object: map[string]any{
			"apiVersion": "autoscaling.openfuyao.com/v1alpha1",
			"kind":       "ResourceScalingGroup",
			"metadata": map[string]any{
				"name":      rsgName,
				"namespace": namespace,
			},
			"spec": map[string]any{
				"targetResources": []any{
					esUtils.NewTargetResource("prefill", esUtils.LeaderWorkerSetAPIVersion, "LeaderWorkerSet", infernexUtils.RSGInfernexPrefillName, ""),
					esUtils.NewTargetResource("decode", esUtils.LeaderWorkerSetAPIVersion, "LeaderWorkerSet", infernexUtils.RSGInfernexDecodeName, ""),
				},
				"scalingStrategy": map[string]any{
					"type": "GroupReplication",
					"groupConfig": map[string]any{
						"groupName": "pd",
						"replicas":  int64(1),
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(esUtils.ResourceScalingGroupGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("创建 ResourceScalingGroup %s/%s 失败: %w", namespace, rsgName, err)
	}
	GinkgoWriter.Printf("ResourceScalingGroup %s/%s 已就绪\n", namespace, rsgName)
	return nil
}

// CreateHpaElasticScalerCRForAggregated 创建指向聚合 LeaderWorkerSet 的 HPA ElasticScaler。
// InferNex_FIT_86 使用。
func CreateHpaElasticScalerCRForAggregated(dynamicClient dynamic.Interface, ctx context.Context, namespace, esName string, minReplicas, maxReplicas, cpuUtilization int32) error {
	targetRef := esUtils.ESMetricsTargetRef{
		APIVersion: esUtils.LeaderWorkerSetAPIVersion,
		Kind:       "LeaderWorkerSet",
		Name:       infernexUtils.ResolveInfernexAggregatedLWSName(dynamicClient, ctx),
		Namespace:  namespace,
	}
	metrics := []esUtils.ESMetricSpec{
		{
			Type:               "Resource",
			ResourceName:       "cpu",
			TargetType:         "Utilization",
			AverageUtilization: cpuUtilization,
		},
	}
	if err := esUtils.DeleteElasticScalerCRAndWait(dynamicClient, ctx, esName, namespace); err != nil {
		return err
	}
	return esUtils.CreateElasticScalerCRMetricsTrigger(dynamicClient, ctx, esName, namespace, targetRef, minReplicas, maxReplicas, metrics)
}

// CreateApaElasticScalerCRForAggregated 创建指向聚合 LeaderWorkerSet 的 APA ElasticScaler。
// InferNex_FIT_79 使用。
func CreateApaElasticScalerCRForAggregated(dynamicClient dynamic.Interface, ctx context.Context, namespace, esName string, minReplicas, maxReplicas, cpuUtilization int32) error {
	targetRef := esUtils.ESMetricsTargetRef{
		APIVersion: esUtils.LeaderWorkerSetAPIVersion,
		Kind:       "LeaderWorkerSet",
		Name:       infernexUtils.ResolveInfernexAggregatedLWSName(dynamicClient, ctx),
		Namespace:  namespace,
	}
	metrics := []esUtils.ESApaMetricEntry{
		{
			Type:         "Resource",
			ResourceName: "cpu",
			TargetType:   "Utilization",
			AverageUtil:  cpuUtilization,
		},
	}
	return esUtils.CreateApaElasticScalerCR(dynamicClient, ctx, esName, namespace, targetRef, minReplicas, maxReplicas, metrics)
}

// CreateApaElasticScalerCRForRSG 创建指向 RSG 的 APA ElasticScaler。
func CreateApaElasticScalerCRForRSG(dynamicClient dynamic.Interface, ctx context.Context, namespace, esName, rsgName string, minReplicas, maxReplicas, cpuUtilization int32) error {
	targetRef := esUtils.ESMetricsTargetRef{
		APIVersion: "autoscaling.openfuyao.com/v1alpha1",
		Kind:       "resourcescalinggroup",
		Name:       rsgName,
		Namespace:  namespace,
	}
	metrics := []esUtils.ESApaMetricEntry{
		{
			Type:         "Resource",
			ResourceName: "cpu",
			TargetType:   "Utilization",
			AverageUtil:  cpuUtilization,
		},
	}
	return esUtils.CreateApaElasticScalerCR(dynamicClient, ctx, esName, namespace, targetRef, minReplicas, maxReplicas, metrics)
}

func infernexCommonHelmSets() []string {
	return []string{
		"--set", "inferenceGateway.enabled=false",
		"--set", "hermes-router.enabled=false",
		"--set", "cache-indexer.enabled=false",
		"--set", "eagle-eye.enabled=false",
		"--set", "pd-orchestrator.elastic-scaler.enabled=true",
		"--set", "pd-orchestrator.elastic-scaler.elasticScaler.enabled=false",
		"--set", "pd-orchestrator.resourcescalinggroup.enabled=true",
		"--set", "pd-orchestrator.resourcescalinggroup.instanceConfig.enabled=false",
		"--set", "pd-orchestrator.tidal.enabled=true",
	}
}

func infernexAggregatedBackendHelmSets() []string {
	return []string{
		"--set", "inference-backend.services[0].name=vllm-aggregated-tp2-01",
		"--set", "inference-backend.services[0].enabled=true",
		"--set", "inference-backend.services[0].mode=aggregated",
		"--set", "inference-backend.services[0].service.port=8000",
		"--set", "inference-backend.services[0].kvTransferConfig.connectorConfig.kv_connector=AscendStoreConnector",
		"--set", "inference-backend.services[0].kvTransferConfig.connectorConfig.kv_connector_extra_config.lookup_rpc_port=1",
		"--set", "inference-backend.services[0].kvTransferConfig.connectorConfig.kv_connector_extra_config.backend=mooncake",
		"--set", "inference-backend.services[0].kvTransferConfig.mooncake.configPath=/app/mooncake.json",
		"--set", "inference-backend.services[0].kvTransferConfig.mooncake.use_store=true",
		"--set", "inference-backend.services[0].kvTransferConfig.mooncake.config=local_hostname: \"$POD_IP\"\nmetadata_server: \"redis://redis-service:6379\"\nmaster_server_address: \"mooncake-master-service:30089\"\ndevice_name: \"\"\nprotocol: \"ascend\"\nglobal_segment_size: 42949672960\nuse_ascend_direct: true",
		"--set", "inference-backend.services[0].aggregated.replicas=1",
		"--set", "inference-backend.services[0].aggregated.tensorParallelSize=2",
		"--set", "inference-backend.services[0].aggregated.dataParallelSize=1",
		"--set-json", `inference-backend.services[0].aggregated.extraArgs=["--max-model-len 10000","--max-num-batched-tokens 40960","--gpu-memory-utilization 0.8","--block-size 128","--enable-prefix-caching"]`,
		"--set", "inference-backend.services[0].resources.requests.cpu=4",
		"--set", "inference-backend.services[0].resources.requests.memory=32Gi",
		"--set", "inference-backend.services[0].resources.limits.cpu=8",
		"--set", "inference-backend.services[0].resources.limits.memory=64Gi",
	}
}

func waitInfernexControllersAndSharedWorkloads(clientset *kubernetes.Clientset, ctx context.Context, releaseName, namespace string, timeout time.Duration) {
	scalingNS := esUtils.ElasticScalerHelmNamespace
	controllers := map[string]int32{
		fmt.Sprintf("%s-elastic-scaler-controller-manager", releaseName): 1,
		"resourcescalinggroup-controller":                                1,
		"tidal-controller-manager":                                       1,
	}
	for name, replicas := range controllers {
		GinkgoWriter.Printf("等待 Deployment %s/%s ReadyReplicas=%d...\n", scalingNS, name, replicas)
		esUtils.WaitForDeploymentReplicas(clientset, ctx, name, scalingNS, replicas, timeout)
	}

	workloads := map[string]int32{
		"mooncake-master-deployment": 1,
		"redis-server-deployment":    1,
	}
	for name, replicas := range workloads {
		GinkgoWriter.Printf("等待 Deployment %s/%s ReadyReplicas=%d...\n", namespace, name, replicas)
		esUtils.WaitForDeploymentReplicas(clientset, ctx, name, namespace, replicas, timeout)
	}
}
