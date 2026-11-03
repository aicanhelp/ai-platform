package utils

import (
	"context"
	"fmt"
	"time"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// Infernex 场景下的 Tidal / ES 资源名
const (
	InfernexTidalName = "infernex-tidal-rsg"
	InfernexESName    = "infernex-es-tidal-rsg"
	// InfernexTidalDefaultRSGName 与 chart 中 resourcescalinggroup.instanceConfig.name 一致
	InfernexTidalDefaultRSGName = "rsg"
)

// InfernexTidalScaleUpRules 初始只配置扩容规则，避免扩容断言过程中被缩容规则抢先触发
var InfernexTidalScaleUpRules = []esUtils.TidalRule{
	{Name: "scale-up-2", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 2, Description: "每2分钟扩容到2"},
}

// InfernexTidalScaleDownRules 扩容验证完成后再切换到该规则，做缩容验证
var InfernexTidalScaleDownRules = []esUtils.TidalRule{
	{Name: "scale-down-1", Cron: "CRON_TZ=Asia/Shanghai 0 */2 * * * *", Replicas: 1, Description: "每2分钟缩容到1"},
}

// GetDefaultRSGName 获取命名空间中一个默认存在的 RSG 名称
func GetDefaultRSGName(dynamicClient dynamic.Interface, ctx context.Context, namespace string) (string, error) {
	rsgList, err := dynamicClient.Resource(esUtils.ResourceScalingGroupGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", err
	}
	if len(rsgList.Items) == 0 {
		return "", fmt.Errorf("no ResourceScalingGroup found in namespace %s", namespace)
	}
	return rsgList.Items[0].GetName(), nil
}

// CreateInfernexTidalAndESForDefaultRSG 创建 Tidal->RSG 与 ES(StateTrigger->Tidal) 链路
func CreateInfernexTidalAndESForDefaultRSG(dynamicClient dynamic.Interface, ctx context.Context, namespace string) error {
	rsgName, err := GetDefaultRSGName(dynamicClient, ctx, namespace)
	if err != nil {
		return err
	}
	GinkgoWriter.Printf("使用默认 RSG: %s\n", rsgName)

	target := esUtils.TidalTargetRef{
		APIVersion: "autoscaling.openfuyao.com/v1alpha1",
		Kind:       "ResourceScalingGroup",
		Name:       rsgName,
		Namespace:  namespace,
	}
	if err := esUtils.CreateTidalCRWithTargetRulesAndESRef(
		dynamicClient, ctx,
		InfernexTidalName, namespace,
		target, InfernexTidalScaleUpRules,
		InfernexESName, namespace,
	); err != nil {
		return err
	}
	esUtils.VerifyTidalCRExists(dynamicClient, ctx, InfernexTidalName, namespace)
	esUtils.VerifyTidalStatusReady(dynamicClient, ctx, InfernexTidalName, namespace)

	if err := esUtils.CreateStateTriggerESForTidal(dynamicClient, ctx, InfernexESName, namespace, InfernexTidalName, namespace); err != nil {
		return err
	}
	esUtils.VerifyESCRExists(dynamicClient, ctx, InfernexESName, namespace)
	return nil
}

// SwitchInfernexTidalToScaleDownRule 将 Tidal 规则切换为缩容规则
func SwitchInfernexTidalToScaleDownRule(dynamicClient dynamic.Interface, ctx context.Context, namespace string) error {
	return esUtils.UpdateTidalCRRules(dynamicClient, ctx, InfernexTidalName, namespace, InfernexTidalScaleDownRules)
}

// CleanupInfernexTidalAndES 删除本场景创建的 ES 与 Tidal 资源
func CleanupInfernexTidalAndES(dynamicClient dynamic.Interface, ctx context.Context, namespace string) {
	_ = esUtils.DeleteElasticScalerCR(dynamicClient, ctx, InfernexESName, namespace)
	_ = esUtils.DeleteTidalCR(dynamicClient, ctx, InfernexTidalName, namespace)
	if err := dynamicClient.Resource(esUtils.ResourceScalingGroupGVR).Namespace(namespace).Delete(ctx, InfernexTidalDefaultRSGName, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
		GinkgoWriter.Printf("删除默认 RSG 失败: %v\n", err)
	}
}

// CleanupInfernexPDPodsDeployments 清理 pd 场景下可能残留的 prefill/decode Deployment（含旧名）。
func CleanupInfernexPDPodsDeployments(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	names := []string{
		RSGInfernexPrefillName,
		RSGInfernexDecodeName,
		"pd-" + RSGInfernexPrefillName + "-1",
		"pd-" + RSGInfernexDecodeName + "-1",
		"vllm-pd-2p1d-01-prefill",
		"vllm-pd-2p1d-01-decode",
		"pd-vllm-pd-2p1d-01-prefill-1",
		"pd-vllm-pd-2p1d-01-decode-1",
	}
	for _, name := range names {
		err := clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			GinkgoWriter.Printf("删除 Deployment %s/%s 失败: %v\n", namespace, name, err)
		}
	}
}

// UpdateRSGGroupReplicasInNamespace 手动修改 ResourceScalingGroup 的 GroupReplication 副本数（spec.scalingStrategy.groupConfig.replicas），带 409 重试
func UpdateRSGGroupReplicasInNamespace(dynamicClient dynamic.Interface, ctx context.Context, namespace, rsgName string, replicas int64) error {
	const maxRetries = 8
	var lastErr error
	for i := 0; i < maxRetries; i++ {
		obj, err := dynamicClient.Resource(esUtils.ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, rsgName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, replicas, "spec", "scalingStrategy", "groupConfig", "replicas"); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(esUtils.ResourceScalingGroupGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			GinkgoWriter.Printf("已更新 RSG %s/%s groupConfig.replicas=%d\n", namespace, rsgName, replicas)
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		lastErr = err
		time.Sleep(2 * time.Second)
	}
	return lastErr
}

// WaitForInfernexBackendReplicas 等待 prefill/decode LeaderWorkerSet 个数达到目标（每个 RSG group 一套 LWS）。
func WaitForInfernexBackendReplicas(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string, expectedPrefill, expectedDecode int32, timeout time.Duration) {
	Expect(EnsureLeaderWorkerSetAPI(clientset)).NotTo(HaveOccurred())
	Eventually(func() bool {
		prefillReady, err := CountReadyLeaderWorkerSetsByRole(dynamicClient, ctx, namespace, "prefill")
		if err != nil {
			GinkgoWriter.Printf("count prefill LWS failed: %v\n", err)
			return false
		}
		decodeReady, err := CountReadyLeaderWorkerSetsByRole(dynamicClient, ctx, namespace, "decode")
		if err != nil {
			GinkgoWriter.Printf("count decode LWS failed: %v\n", err)
			return false
		}
		GinkgoWriter.Printf("backend LWS prefill=%d/%d decode=%d/%d\n", prefillReady, expectedPrefill, decodeReady, expectedDecode)
		return prefillReady == expectedPrefill && decodeReady == expectedDecode
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ready prefill/decode LeaderWorkerSet 数量应为 %d/%d", expectedPrefill, expectedDecode)
}
