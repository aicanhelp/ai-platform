package utils

import (
	"context"
	"fmt"
	"time"

	esUtils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

var (
	ESInfernexHelmReleaseName = "infernex"
	// ESInfernexHelmChart supports both OCI remote URL and local chart path.
	// Default: OCI remote URL. Set to local path (e.g. "/path/to/infernex-chart") to use local chart.
	ESInfernexHelmChart     = "oci://cr.openfuyao.cn/charts/infernex"
	ESInfernexHelmVersion   = "0.0.0-latest"
	ESInfernexHelmNamespace = "scaling-system"
	// 推理引擎 LeaderWorkerSet / Service 名称（与 helm values 中的 services[0].name 一致）
	ESInfernexDeploymentName = "vllm-aggregated-tp2-01-aggregated"
	// ElasticScaler CR 名称
	ESInfernexESCRName = "infernex-elastic-scaler"
)

const (
	// NamespaceTerminatingWaitTimeout 等待 Terminating namespace 消失，避免 helm 写入失败。
	NamespaceTerminatingWaitTimeout = 10 * time.Minute
)

// Infernex 相关 Deployment 及期望 Ready 副本数（推理引擎为 LeaderWorkerSet，按 role=aggregated 解析）
var infernexDeployments = map[string]int32{
	"infernex-elastic-scaler-controller-manager": 1,
	"mooncake-master-deployment":                 1,
	"redis-server-deployment":                    1,
}

// WaitForNamespaceReadyForInstall 等待 namespace 可被写入：不存在或 Active 即可。
// 若处于 Terminating，则等到完全删除，避免 helm/Create 报 "namespace is being terminated"。
func WaitForNamespaceReadyForInstall(clientset kubernetes.Interface, ctx context.Context, namespace string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for {
		ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			GinkgoWriter.Printf("namespace %s 不存在，可安装\n", namespace)
			return nil
		}
		if err != nil {
			return fmt.Errorf("get namespace %s: %w", namespace, err)
		}
		if ns.DeletionTimestamp == nil && ns.Status.Phase != corev1.NamespaceTerminating {
			GinkgoWriter.Printf("namespace %s 状态为 %s，可安装\n", namespace, ns.Status.Phase)
			return nil
		}
		GinkgoWriter.Printf("等待 namespace %s 结束 Terminating...\n", namespace)
		if time.Now().After(deadline) {
			return fmt.Errorf("等待 namespace %s 结束 Terminating 超时（phase=%s）", namespace, ns.Status.Phase)
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("等待 namespace %s 结束 Terminating: %w", namespace, ctx.Err())
		case <-time.After(2 * time.Second):
		}
	}
}

// EnsureNamespace 确保 namespace 存在且可写。若正在 Terminating，先等删除再创建。
func EnsureNamespace(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if err == nil {
		if ns.DeletionTimestamp == nil && ns.Status.Phase != corev1.NamespaceTerminating {
			return nil
		}
		if err := WaitForNamespaceReadyForInstall(clientset, ctx, namespace, NamespaceTerminatingWaitTimeout); err != nil {
			return err
		}
	} else if !apierrors.IsNotFound(err) {
		return err
	}

	created := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}
	_, err = clientset.CoreV1().Namespaces().Create(ctx, created, metav1.CreateOptions{})
	if err != nil && apierrors.IsAlreadyExists(err) {
		return nil
	}
	return err
}

func ResolveInfernexAggregatedLWSName(dynamicClient dynamic.Interface, ctx context.Context) string {
	name, err := FindLeaderWorkerSetNameByRole(dynamicClient, ctx, ESInfernexHelmNamespace, "aggregated")
	if err != nil {
		GinkgoWriter.Printf("未按 role=aggregated 找到 LWS，使用默认名 %s: %v\n", ESInfernexDeploymentName, err)
		return ESInfernexDeploymentName
	}
	return name
}

// StartLoadTest 开始对推理引擎进行压测：在集群内启动一个压测 Deployment，持续向推理服务发请求
func StartLoadTest(clientset *kubernetes.Clientset, ctx context.Context, duration time.Duration) error {
	// 如果压测 Deployment 已存在，则直接等待其 Ready
	if _, err := clientset.AppsV1().Deployments(ESInfernexHelmNamespace).Get(ctx, "infernex-load-tester", metav1.GetOptions{}); err == nil {
		GinkgoWriter.Println("压测 Deployment 已存在，直接等待其 Ready...")
		esUtils.WaitForDeploymentReplicas(clientset, ctx, "infernex-load-tester", ESInfernexHelmNamespace, 1, duration)
		return nil
	}

	// 构造压测 Deployment：在集群内持续向 vllm 服务发请求（chat completions 接口）
	var replicas int32 = 1
	cmd := []string{
		"sh",
		"-c",
		`while true; do curl -sS -X POST "http://vllm-aggregated-tp2-01:8000/v1/chat/completions" -H "Content-Type: application/json" -d '{"model":"Qwen/Qwen3-8B","messages":[{"role":"user","content":"请介绍openFuyao开源社区"}],"stream":false}' >/dev/null 2>&1 || true; sleep 0.5; done`,
	}

	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "infernex-load-tester",
			Namespace: ESInfernexHelmNamespace,
			Labels: map[string]string{
				"app": "infernex-load-tester",
			},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": "infernex-load-tester"},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": "infernex-load-tester"},
				},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Name:    "load-tester",
							Image:   "curlimages/curl:8.5.0",
							Command: cmd,
						},
					},
				},
			},
		},
	}

	_, err := clientset.AppsV1().Deployments(ESInfernexHelmNamespace).Create(ctx, dep, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		return fmt.Errorf("创建压测 Deployment 失败: %v", err)
	}

	GinkgoWriter.Printf("已创建压测 Deployment %s，等待其 Ready...\n", "infernex-load-tester")
	esUtils.WaitForDeploymentReplicas(clientset, ctx, "infernex-load-tester", ESInfernexHelmNamespace, 1, duration)
	return nil
}

// StopLoadTest 停止压测：删除集群内的压测 Deployment
func StopLoadTest(clientset *kubernetes.Clientset, ctx context.Context) {
	err := clientset.AppsV1().Deployments(ESInfernexHelmNamespace).Delete(ctx, "infernex-load-tester", metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		GinkgoWriter.Printf("删除压测 Deployment 失败: %v\n", err)
	} else {
		GinkgoWriter.Println("压测 Deployment 已删除")
	}
}

// DeleteElasticScalerCRForInfernex 删除推理引擎的 ElasticScaler CR
func DeleteElasticScalerCRForInfernex(dynamicClient dynamic.Interface, ctx context.Context) error {
	return esUtils.DeleteElasticScalerCR(dynamicClient, ctx, ESInfernexESCRName, ESInfernexHelmNamespace)
}

// WaitForInfernexLeaderWorkerSetReplicas 等待推理引擎 LeaderWorkerSet status.readyReplicas 达到目标
func WaitForInfernexLeaderWorkerSetReplicas(dynamicClient dynamic.Interface, ctx context.Context, expected int32, timeout time.Duration) {
	Eventually(func() int64 {
		name := ResolveInfernexAggregatedLWSName(dynamicClient, ctx)
		obj, err := dynamicClient.Resource(esUtils.LeaderWorkerSetGVR).Namespace(ESInfernexHelmNamespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return -1
		}
		ready, err := leaderWorkerSetReadyReplicas(obj)
		if err != nil {
			return -1
		}
		return ready
	}, timeout, 10*time.Second).Should(Equal(int64(expected)), "aggregated LeaderWorkerSet readyReplicas 应达到 %d", expected)
}

// WaitForInfernexDeploymentReplicas 等待推理引擎副本数达到目标（LeaderWorkerSet 架构）
func WaitForInfernexDeploymentReplicas(dynamicClient dynamic.Interface, ctx context.Context, expected int32, timeout time.Duration) {
	WaitForInfernexLeaderWorkerSetReplicas(dynamicClient, ctx, expected, timeout)
}

// WaitForInfernexAllDeploymentsReady 等待 Infernex 控制器/依赖 Deployment 及推理引擎 LeaderWorkerSet 就绪
func WaitForInfernexAllDeploymentsReady(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, timeout time.Duration) {
	for name, replicas := range infernexDeployments {
		GinkgoWriter.Printf("等待 Deployment %s ReadyReplicas=%d...\n", name, replicas)
		esUtils.WaitForDeploymentReplicas(clientset, ctx, name, ESInfernexHelmNamespace, replicas, timeout)
	}
	lwsName := ResolveInfernexAggregatedLWSName(dynamicClient, ctx)
	GinkgoWriter.Printf("等待 LeaderWorkerSet %s 就绪...\n", lwsName)
	WaitForLeaderWorkerSetFullyReady(dynamicClient, ctx, ESInfernexHelmNamespace, lwsName, timeout)
}
