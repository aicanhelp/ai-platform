package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

func InstallElasticScaler() error {
	return RunElasticScalerHelmInstall(false, true)
}

// WaitForNamespaceDeletionIfTerminating 若 namespace 处于 Terminating，则等待其被完全删除。
func WaitForNamespaceDeletionIfTerminating(clientset *kubernetes.Clientset, ctx context.Context, namespace string, timeout, interval time.Duration) error {
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("获取 namespace %s 失败: %w", namespace, err)
	}
	if ns.Status.Phase != corev1.NamespaceTerminating {
		return nil
	}

	GinkgoWriter.Printf("namespace %s 正在删除中，等待其删除完成...\n", namespace)
	deadline := time.Now().Add(timeout)
	for {
		if time.Now().After(deadline) {
			return fmt.Errorf("等待 namespace %s 删除超时（%s）", namespace, timeout)
		}
		_, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			GinkgoWriter.Printf("namespace %s 已删除完成\n", namespace)
			return nil
		}
		if err != nil {
			GinkgoWriter.Printf("查询 namespace %s 状态失败，继续重试: %v\n", namespace, err)
		}
		time.Sleep(interval)
	}
}

var elasticScalerGVR = schema.GroupVersionResource{
	Group:    "elasticscaler.io",
	Version:  "v1alpha1",
	Resource: "elasticscalers",
}

// ESMetricsTargetRef 目标引用（Deployment 等）
type ESMetricsTargetRef struct {
	APIVersion string
	Kind       string
	Name       string
	Namespace  string
}

// ESMetricSpec 单条指标（Resource 类型，如 cpu）
type ESMetricSpec struct {
	Type               string // Resource
	ResourceName       string // cpu / memory
	TargetType         string // Utilization
	AverageUtilization int32
}

func CleanupElasticScalerTest(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("清理 elastic-scaler 测试资源...")
	ns := ElasticScalerTestNamespace
	// 删除 ES CR
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRName, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMultiMetrics, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameUpdate, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameBigMax, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameExistingHPA, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameStatefulSet, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMultiES1, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMultiES2, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, "elastic-scaler-crud", ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMetricsEmpty, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameUnknownTriggerType, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameNoScalingAlgorithm, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameSignalNoEndpoint, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameMetricNoResource, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameExternalMetric, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameExternalServer, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameCustomAlgorithm, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameSignal, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameStateMissingTarget, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameStateBasic, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameCrossNamespace, ESTestNamespaceA)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentUpdate, ns)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, ESTestCRNameConcurrentDelete, ns)

	// StateTrigger 用例：删除 ES、Tidal、Deployment
	_ = DeleteTidalCR(dynamicClient, ctx, ESTestStateTriggerTidalName, ns)
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestStateTriggerDeploymentName, metav1.DeleteOptions{})

	// 删除 Deployment / StatefulSet / Service 等目标资源
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestDeploymentNameExistingHPA, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestDeploymentName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestDeploymentNameCRUD, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestDeploymentNameBigMax, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, ESTestDeploymentNameMultiES, metav1.DeleteOptions{})
	_ = clientset.AppsV1().StatefulSets(ns).Delete(ctx, ESTestStatefulSetName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, ESTestStatefulSetName+"-service", metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ESTestNamespaceB).Delete(ctx, ESTestCrossNamespaceDeploy, metav1.DeleteOptions{})
	_ = DeleteNamespaceIfExists(clientset, ctx, ESTestNamespaceA)
	_ = DeleteNamespaceIfExists(clientset, ctx, ESTestNamespaceB)
	time.Sleep(2 * time.Second)
	_ = HelmUninstall(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
	GinkgoWriter.Println("清理完成")
}

func VerifyElasticScalerPodsRunning(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration, interval time.Duration) {
	GinkgoWriter.Println("验证 elastic-scaler Pod 运行状态...")
	Eventually(func() bool {
		allRunning, err := JudgeElasticScalerPodsAllRunning(clientset, ctx)
		if err != nil {
			GinkgoWriter.Printf("检查失败: %v\n", err)
			return false
		}
		return allRunning
	}, timeout, interval).Should(BeTrue(), "elastic-scaler Pod 应在超时时间内全部变为 Running")
}

func JudgeElasticScalerPodsAllRunning(clientset *kubernetes.Clientset, ctx context.Context) (bool, error) {
	pods, err := FilterElasticScalerPods(clientset, ctx)
	if err != nil {
		return false, fmt.Errorf("获取 elastic-scaler pods 失败: %v", err)
	}
	if len(pods) == 0 {
		return false, nil
	}
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil {
			return false, nil
		}
		if pod.Status.Phase != corev1.PodRunning {
			GinkgoWriter.Printf("elastic-scaler Pod %s/%s phase=%s\n", pod.Namespace, pod.Name, pod.Status.Phase)
			return false, nil
		}
		for _, cs := range pod.Status.ContainerStatuses {
			if !cs.Ready {
				GinkgoWriter.Printf("elastic-scaler Pod %s/%s container %s not ready (waiting=%t restarts=%d)\n",
					pod.Namespace, pod.Name, cs.Name, cs.State.Waiting != nil, cs.RestartCount)
				return false, nil
			}
		}
	}
	return true, nil
}

func FilterElasticScalerPods(clientset *kubernetes.Clientset, ctx context.Context) ([]corev1.Pod, error) {
	podList, err := clientset.CoreV1().Pods(ElasticScalerHelmNamespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, fmt.Errorf("获取 elastic-scaler pods 失败: %v", err)
	}
	var filteredPods []corev1.Pod
	for _, pod := range podList.Items {
		if strings.HasPrefix(pod.Name, ElasticScalerPodPrefix) {
			filteredPods = append(filteredPods, pod)
		}
	}
	return filteredPods, nil
}

// RestartElasticScalerController 重启 ElasticScaler controller Pod，并等待其恢复 Running
func RestartElasticScalerController(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) error {
	GinkgoWriter.Println("重启 elastic-scaler controller Pod...")
	pods, err := FilterElasticScalerPods(clientset, ctx)
	if err != nil {
		return err
	}
	for _, pod := range pods {
		GinkgoWriter.Printf("删除 controller Pod: %s/%s\n", pod.Namespace, pod.Name)
		if err := clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("删除 controller Pod 失败: %w", err)
		}
	}
	// 等待新的 Pod 启动并运行
	VerifyElasticScalerPodsRunning(clientset, ctx, timeout, 10*time.Second)
	return nil
}

// CreateStatefulSetForES 创建 StatefulSet 及其 Headless Service，供 ES 指标驱动用例使用
func CreateStatefulSetForES(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) error {
	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name + "-service",
			Namespace: namespace,
		},
		Spec: corev1.ServiceSpec{
			ClusterIP: "None",
			Selector:  map[string]string{"app": name},
			Ports:     []corev1.ServicePort{},
		},
	}
	if _, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	sts := &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: appsv1.StatefulSetSpec{
			Replicas:    &replicas,
			ServiceName: name + "-service",
			Selector: &metav1.LabelSelector{
				MatchLabels: map[string]string{"app": name},
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"app": name},
				},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:            name + "-sset",
							Image:           esMetricsTestContainerImages[0],
							ImagePullPolicy: corev1.PullIfNotPresent,
							Command:         []string{"sh", "-c", "while true; do :; done"},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("100m"),
								corev1.ResourceMemory: resource.MustParse("100Mi"),
							},
							Limits: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("200m"),
								corev1.ResourceMemory: resource.MustParse("200Mi"),
							},
						},
					}},
				},
			},
		},
	}
	if _, err := clientset.AppsV1().StatefulSets(namespace).Create(ctx, sts, metav1.CreateOptions{}); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return err
	}
	return nil
}

// CreateElasticScalerCRWithUnknownTriggerType 创建 Trigger.Type 为未知值的 ES，用于输入验证测试
func CreateElasticScalerCRWithUnknownTriggerType(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "unknownTriggerType",
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRMetricsTriggerNoScalingAlgorithm 创建缺失 scalingAlgorithm 的 MetricsTrigger ES
func CreateElasticScalerCRMetricsTriggerNoScalingAlgorithm(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						// 故意缺失 scalingAlgorithm 字段
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "Resource",
								"resource": map[string]interface{}{
									"metricsName": "cpu",
									"target": map[string]interface{}{
										"type":               "Utilization",
										"averageUtilization": int64(1),
									},
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateSignalTriggerESWithoutEndpoint 创建 Trigger.Type=SignalTrigger 但缺失 endpoint 的 ES
func CreateSignalTriggerESWithoutEndpoint(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type":          "SignalTrigger",
					"signalTrigger": map[string]interface{}{
						// 故意缺失 endpoint 字段
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRMetricMissingResource 创建 metrics 中 Resource 类型但缺失 resource 字段的 ES
func CreateElasticScalerCRMetricMissingResource(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": "HPA",
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "Resource",
								// 故意缺失 resource 字段
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRWithExternalMetric 创建包含 External 类型指标的 ES（MetricsTrigger/HPA）
func CreateElasticScalerCRWithExternalMetric(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": "HPA",
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "External",
								"external": map[string]interface{}{
									"metricsName": "qps",
									"target": map[string]interface{}{
										"type":  "Value",
										"value": "100",
									},
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRWithExternalServerMetric 创建包含 ExternalServer 类型指标的 ES（MetricsTrigger/HPA）
func CreateElasticScalerCRWithExternalServerMetric(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": "HPA",
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "ExternalServer",
								"externalServer": map[string]interface{}{
									"metricsName": "qps",
									"target":      "1000",
									"endpoint":    "http://localhost:8080/metrics",
									"protocol":    "http",
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRMetricsTriggerCustomAlgorithm 创建使用自定义算法名的 MetricsTrigger ES（scalingAlgorithm=MyAlgo）
func CreateElasticScalerCRMetricsTriggerCustomAlgorithm(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": "MyAlgo",
						"metrics": []interface{}{
							map[string]interface{}{
								"type": "Resource",
								"resource": map[string]interface{}{
									"metricsName": "cpu",
									"target": map[string]interface{}{
										"type":               "Utilization",
										"averageUtilization": int64(1),
									},
								},
							},
						},
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateSignalTriggerES 创建正常配置的 SignalTrigger ES（用于未实现路径拦截测试）
func CreateSignalTriggerES(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef) error {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "SignalTrigger",
					"signalTrigger": map[string]interface{}{
						"endpoint": "https://example.com/webhook",
						"protocol": "https",
					},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateStateTriggerESMissingTarget 创建 StateTrigger ES，TargetRef 指向不存在的 Deployment
func CreateStateTriggerESMissingTarget(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	nonExistTarget := ESMetricsTargetRef{
		APIVersion: "apps/v1",
		Kind:       "Deployment",
		Name:       "non-exist-target",
		Namespace:  namespace,
	}
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": 1,
				"maxReplicas": 2,
				"targetRef": map[string]interface{}{
					"apiVersion": nonExistTarget.APIVersion,
					"kind":       nonExistTarget.Kind,
					"name":       nonExistTarget.Name,
					"namespace":  nonExistTarget.Namespace,
				},
				"trigger": map[string]interface{}{
					"type":         "StateTrigger",
					"stateTrigger": map[string]interface{}{},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateStateTriggerESForTidal 创建 StateTrigger ES，TargetRef 指向 Tidal
func CreateStateTriggerESForTidal(dynamicClient dynamic.Interface, ctx context.Context, esName, esNamespace, tidalName, tidalNamespace string) error {
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      esName,
				"namespace": esNamespace,
			},
			"spec": map[string]interface{}{
				"targetRef": map[string]interface{}{
					"apiVersion": "tidal.io/v1alpha1",
					"kind":       "Tidal",
					"name":       tidalName,
					"namespace":  tidalNamespace,
				},
				"trigger": map[string]interface{}{
					"type":         "StateTrigger",
					"stateTrigger": map[string]interface{}{},
				},
			},
		},
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(esNamespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// UpdateTidalTargetToNonExist 将 Tidal 的 targetRef.name 更新为不存在的目标，模拟执行扩缩过程中目标被删除的场景
func UpdateTidalTargetToNonExist(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	obj, err := dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	targetRef, found, _ := unstructured.NestedMap(obj.Object, "spec", "targetRef")
	if !found || targetRef == nil {
		return fmt.Errorf("tidal %s/%s spec.targetRef not found", namespace, name)
	}
	targetRef["name"] = "non-exist-target"
	if err := unstructured.SetNestedMap(obj.Object, targetRef, "spec", "targetRef"); err != nil {
		return err
	}
	_, err = dynamicClient.Resource(tidalSchedulerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
	return err
}

// PatchDeploymentAnnotation 轻量更新 Deployment annotation，用于高频入队/并发场景压测
func PatchDeploymentAnnotation(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, key, value string) error {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return err
	}
	if dep.Annotations == nil {
		dep.Annotations = map[string]string{}
	}
	dep.Annotations[key] = value
	_, err = clientset.AppsV1().Deployments(namespace).Update(ctx, dep, metav1.UpdateOptions{})
	return err
}

// CreateOrUpdatePlainHPA 为给定 ES 名称预创建一个同名 HPA（<esName>-hpa）
// 用于验证“集群已有 HPA 资源，Controller 使用 CreateOrUpdateHPA 更新”场景。
func CreateOrUpdatePlainHPA(clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, targetKind, targetName string, minReplicas, maxReplicas, cpuUtilization int32) error {
	hpaName := esName + "-hpa"
	min := minReplicas
	hpa := &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      hpaName,
			Namespace: namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       targetKind,
				Name:       targetName,
			},
			MinReplicas: &min,
			MaxReplicas: maxReplicas,
			Metrics: []autoscalingv2.MetricSpec{
				{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricSource{
						Name: corev1.ResourceCPU,
						Target: autoscalingv2.MetricTarget{
							Type:               autoscalingv2.UtilizationMetricType,
							AverageUtilization: &cpuUtilization,
						},
					},
				},
			},
		},
	}

	// 若已存在则更新，避免测试多次运行冲突
	if _, err := clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Get(ctx, hpaName, metav1.GetOptions{}); err == nil {
		_, err = clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Update(ctx, hpa, metav1.UpdateOptions{})
		return err
	} else if !apierrors.IsNotFound(err) {
		return err
	}

	_, err := clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Create(ctx, hpa, metav1.CreateOptions{})
	return err
}

// esMetricsTestContainerImages 供测试 Deployment 使用的镜像，按顺序兜底：
// 首选 nginx:alpine —— 本套件的 Deployment/StatefulSet 用例普遍用它，环境里最可能已备好；
// 备选 ubuntu:latest、busybox:latest（都带 sh，能跑打满 CPU 的循环）。
// 曾经只用 ubuntu:latest：离线或单节点流水线环境里拉不到该镜像时，Pod 卡在 ImagePullBackOff，
// 副本数一直为 0，用例只能等到超时（表现为 Deployment 副本数应达到 1 的失败）。
var esMetricsTestContainerImages = []string{"nginx:alpine", "ubuntu:latest", "busybox:latest"}

// CreateESMetricsTestDeployment 创建指标驱动测试用 Deployment（带 resources，可选高 CPU 容器用于压测扩容）
func CreateESMetricsTestDeployment(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32, highCPU bool) error {
	return createESMetricsTestDeploymentWithImage(clientset, ctx, name, namespace, replicas, highCPU, esMetricsTestContainerImages[0])
}

// createESMetricsTestDeploymentWithImage 用指定镜像创建测试 Deployment；已存在则视为成功（幂等，避免重复创建报错）。
func createESMetricsTestDeploymentWithImage(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32, highCPU bool, image string) error {
	cpuReq := "100m"
	if highCPU {
		cpuReq = "50m"
	}
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": name}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": name}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:            name,
						Image:           image,
						ImagePullPolicy: corev1.PullIfNotPresent,
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse(cpuReq),
								corev1.ResourceMemory: resource.MustParse("100Mi"),
							},
							Limits: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("200m"),
								corev1.ResourceMemory: resource.MustParse("200Mi"),
							},
						},
					}},
				},
			},
		},
	}
	if highCPU {
		// 打满 CPU 的循环：让 CPU utilization 超标以触发扩容
		dep.Spec.Template.Spec.Containers[0].Command = []string{"sh", "-c", "while true; do :; done"}
	}
	_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err == nil {
		return nil
	}
	if !apierrors.IsNotFound(err) {
		return err
	}
	_, err = clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{})
	return err
}

func buildElasticScalerMetricsTriggerObject(name, namespace string, targetRef ESMetricsTargetRef, minReplicas, maxReplicas int32, metrics []ESMetricSpec) *unstructured.Unstructured {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: ESTestDeploymentName, Namespace: namespace}
	}
	metricsSlice := make([]interface{}, 0, len(metrics))
	for _, m := range metrics {
		metricsSlice = append(metricsSlice, map[string]interface{}{
			"type": m.Type,
			"resource": map[string]interface{}{
				"metricsName": m.ResourceName,
				"target": map[string]interface{}{
					"type":               m.TargetType,
					"averageUtilization": int64(m.AverageUtilization),
				},
			},
		})
	}
	return &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "elasticscaler.io/v1alpha1",
			"kind":       "ElasticScaler",
			"metadata": map[string]interface{}{
				"name":      name,
				"namespace": namespace,
			},
			"spec": map[string]interface{}{
				"minReplicas": minReplicas,
				"maxReplicas": maxReplicas,
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": "HPA",
						"metrics":          metricsSlice,
					},
				},
			},
		},
	}
}

// BuildValidMetricsTriggerESObject 构造一个默认合法的 MetricsTrigger ES（HPA + Resource(cpu)），用于 E2E 中按需覆写字段做规格校验。
func BuildValidMetricsTriggerESObject(name, namespace string, targetRef ESMetricsTargetRef) *unstructured.Unstructured {
	metrics := []ESMetricSpec{
		{Type: "Resource", ResourceName: "cpu", TargetType: "Utilization", AverageUtilization: 1},
	}
	return buildElasticScalerMetricsTriggerObject(name, namespace, targetRef, 1, 2, metrics)
}

// CreateElasticScalerCRFromObject 直接创建传入的 ES unstructured 对象（用于规格校验场景）。
func CreateElasticScalerCRFromObject(dynamicClient dynamic.Interface, ctx context.Context, namespace string, obj *unstructured.Unstructured) error {
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// CreateElasticScalerCRMetricsTrigger 创建 MetricsTrigger 类型 ES CR（HPA 算法，单/多指标）
func CreateElasticScalerCRMetricsTrigger(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef, minReplicas, maxReplicas int32, metrics []ESMetricSpec) error {
	obj := buildElasticScalerMetricsTriggerObject(name, namespace, targetRef, minReplicas, maxReplicas, metrics)
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// GetElasticScalerCR 获取 ES CR
func GetElasticScalerCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) (*unstructured.Unstructured, error) {
	return dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
}

// DeleteElasticScalerCRAndWait 删除 ES CR 并等待对象从 API 中消失。
func DeleteElasticScalerCRAndWait(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	if err := DeleteElasticScalerCR(dynamicClient, ctx, name, namespace); err != nil {
		return err
	}
	deadline := time.Now().Add(2 * time.Minute)
	for time.Now().Before(deadline) {
		_, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if apierrors.IsNotFound(err) {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("timeout waiting for ElasticScaler %s/%s deletion", namespace, name)
}

// DeleteElasticScalerCR 删除 ES CR
func DeleteElasticScalerCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	if err == nil {
		GinkgoWriter.Printf("已删除 ElasticScaler CR %s/%s\n", namespace, name)
	}
	return nil
}

// GetHPA 获取命名空间下名为 name 的 HPA（controller 通常以 ES 同名创建）
func GetHPA(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) (*autoscalingv2.HorizontalPodAutoscaler, error) {
	var hpa *autoscalingv2.HorizontalPodAutoscaler
	var err error
	Eventually(func() bool {
		hpa, err = clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Get(ctx, name, metav1.GetOptions{})
		return err == nil && hpa != nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "HPA %s/%s 应被创建", namespace, name)
	return hpa, err
}

// EnsureNoHPA 若存在则删除 HPA（用于“集群中无对应 HPA”用例）
func EnsureNoHPA(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) error {
	err := clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// VerifyESValidatedCondition 校验 ES .status.conditions 存在 Type=Validated, Status=True, Reason=ValidationSucceeded
func VerifyESValidatedCondition(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
		for _, c := range conditions {
			cond, _ := c.(map[string]interface{})
			if cond["type"] == "Validated" && cond["status"] == "True" && cond["reason"] == "ValidationSucceeded" {
				return true
			}
		}
		return false
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ES %s 应存在 Validated=True, Reason=ValidationSucceeded", name)
}

// VerifyESValidationFailedCondition 校验 Reconcile 侧 Validator（含集群态）失败时写入的 Validated=False, Reason=ValidationFailed
func VerifyESValidationFailedCondition(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
		for _, c := range conditions {
			cond, _ := c.(map[string]interface{})
			if cond["type"] == "Validated" && cond["status"] == "False" && cond["reason"] == "ValidationFailed" {
				return true
			}
		}
		return false
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ES %s 应存在 Validated=False, Reason=ValidationFailed", name)
}

// VerifyESCRExists 校验 ES CR 存在且无报错（创建成功）
func VerifyESCRExists(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		_, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		return err == nil
	}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "ElasticScaler %s/%s 应创建成功", namespace, name)
}

// VerifyHPACreated 校验 HPA 被创建
func VerifyHPACreated(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		_, err := GetHPA(clientset, ctx, name, namespace)
		return err == nil
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "HPA %s/%s 应被 controller 创建", namespace, name)
}

// VerifyHPATargetRefMatchesES 校验 HPA targetRef 与 ES 一致（HPA ScaleTargetRef 仅有 kind/name，同命名空间）
func VerifyHPATargetRefMatchesES(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	esKind, _, _ := unstructured.NestedString(es.Object, "spec", "targetRef", "kind")
	esRefName, _, _ := unstructured.NestedString(es.Object, "spec", "targetRef", "name")

	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())
	Expect(hpa.Spec.ScaleTargetRef.Kind).To(Equal(esKind), "HPA ScaleTargetRef.Kind 应与 ES 一致")
	Expect(hpa.Spec.ScaleTargetRef.Name).To(Equal(esRefName), "HPA ScaleTargetRef.Name 应与 ES 一致")
	// HPA 与 ES 同 namespace，targetRef 指向同一命名空间内资源
}

// VerifyESStatusReplicasMatchHPA 校验 ES .status.currentReplicas/desiredReplicas 与 HPA status 一致
func VerifyESStatusReplicasMatchHPA(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())

	cur, _, _ := unstructured.NestedInt64(es.Object, "status", "currentReplicas")
	des, _, _ := unstructured.NestedInt64(es.Object, "status", "desiredReplicas")
	if hpa.Status.CurrentReplicas != 0 {
		Expect(int64(hpa.Status.CurrentReplicas)).To(Equal(cur), "ES currentReplicas 应与 HPA 一致")
	}
	Expect(int64(hpa.Status.DesiredReplicas)).To(Equal(des), "ES desiredReplicas 应与 HPA 一致")
}

// VerifyESConditionsMatchHPA 校验 HPA conditions 被拷贝到 ES（Type/Status/Reason/Message 一致，ObservedGeneration 为 ES generation）
func VerifyESConditionsMatchHPA(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())

	esConds, _, _ := unstructured.NestedSlice(es.Object, "status", "conditions")
	esGen, _, _ := unstructured.NestedInt64(es.Object, "metadata", "generation")
	for _, hpaC := range hpa.Status.Conditions {
		var found bool
		for _, c := range esConds {
			cond, _ := c.(map[string]interface{})
			if cond["type"] == string(hpaC.Type) {
				Expect(cond["status"]).To(Equal(string(hpaC.Status)))
				Expect(cond["reason"]).To(Equal(hpaC.Reason))
				if msg, ok := cond["message"].(string); ok && hpaC.Message != "" {
					Expect(msg).To(Equal(hpaC.Message))
				}
				if obsGen, ok := cond["observedGeneration"].(int64); ok {
					Expect(obsGen).To(Equal(esGen), "ObservedGeneration 应为 ES 当前 generation")
				}
				found = true
				break
			}
		}
		Expect(found).To(BeTrue(), "ES 应包含 HPA condition Type=%s", hpaC.Type)
	}
}

// VerifyESLastScaleTimeMatchHPA 校验 ES .status.lastScaleTime 与 HPA 一致（有则校验存在且合理）
func VerifyESLastScaleTimeMatchHPA(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())
	if hpa.Status.LastScaleTime == nil {
		return
	}
	esLST, found, _ := unstructured.NestedString(es.Object, "status", "lastScaleTime")
	Expect(found).To(BeTrue(), "ES 应有 lastScaleTime")
	Expect(esLST).NotTo(BeEmpty(), "ES lastScaleTime 不应为空")
	// 时间格式可能带时区，仅校验 HPA 与 ES 均存在且 ES 非空
	GinkgoWriter.Printf("HPA LastScaleTime: %v, ES lastScaleTime: %s\n", hpa.Status.LastScaleTime, esLST)
}

// VerifyHPAResourceMetricCPU 校验 HPA 中 Resource 类型指标 name=cpu，target 类型/值与 ES 配置一致
func VerifyHPAResourceMetricCPU(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	metrics, _, _ := unstructured.NestedSlice(es.Object, "spec", "trigger", "metricsTrigger", "metrics")
	var esCPU int32
	for _, m := range metrics {
		mm, _ := m.(map[string]interface{})
		res, _ := mm["resource"].(map[string]interface{})
		if res["metricsName"] == "cpu" {
			if t, ok := res["target"].(map[string]interface{}); ok {
				if u, ok := t["averageUtilization"].(int64); ok {
					esCPU = int32(u)
					break
				}
			}
		}
	}
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())
	var found bool
	for _, m := range hpa.Spec.Metrics {
		if m.Type == autoscalingv2.ResourceMetricSourceType && m.Resource != nil && m.Resource.Name == corev1.ResourceCPU {
			Expect(m.Resource.Target.Type).To(Equal(autoscalingv2.UtilizationMetricType))
			Expect(*m.Resource.Target.AverageUtilization).To(Equal(esCPU), "HPA cpu target 应与 ES 中 averageUtilization 一致")
			found = true
			break
		}
	}
	Expect(found).To(BeTrue(), "HPA 应包含 ResourceMetric cpu")
}

func VerifyHPAResourceMetricMemory(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	metrics, _, _ := unstructured.NestedSlice(es.Object, "spec", "trigger", "metricsTrigger", "metrics")
	var esMemory int32
	for _, m := range metrics {
		mm, _ := m.(map[string]interface{})
		res, _ := mm["resource"].(map[string]interface{})
		if res["metricsName"] == "memory" {
			if t, ok := res["target"].(map[string]interface{}); ok {
				if u, ok := t["averageUtilization"].(int64); ok {
					esMemory = int32(u)
					break
				}
			}
		}
	}
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())
	var found bool
	for _, m := range hpa.Spec.Metrics {
		if m.Type == autoscalingv2.ResourceMetricSourceType && m.Resource != nil && m.Resource.Name == corev1.ResourceMemory {
			Expect(m.Resource.Target.Type).To(Equal(autoscalingv2.UtilizationMetricType))
			Expect(*m.Resource.Target.AverageUtilization).To(Equal(esMemory), "HPA memory target 应与 ES 中 averageUtilization 一致")
			found = true
			break
		}
	}
	Expect(found).To(BeTrue(), "HPA 应包含 ResourceMetric memory")
}

// ListElasticScalerCRNames 列出指定 namespace 下所有 ElasticScaler CR 的名称
func ListElasticScalerCRNames(dynamicClient dynamic.Interface, ctx context.Context, namespace string) ([]string, error) {
	list, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	names := make([]string, 0, len(list.Items))
	for _, item := range list.Items {
		names = append(names, item.GetName())
	}
	return names, nil
}

// UpdateESMaxReplicas 更新 ES CR 的 maxReplicas
func UpdateESMaxReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, newMax int32) error {
	var lastErr error
	for i := 0; i < 5; i++ {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return err
		}
		if err := unstructured.SetNestedField(obj.Object, int64(newMax), "spec", "maxReplicas"); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		// 发生资源版本冲突时，短暂等待后重试，避免与 controller 并发更新导致 409
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
}

// UpdateElasticScalerCR 读取当前 ES，经 mutate 修改 spec/status 等字段后 Update；带 409 冲突重试。
// 用于 E2E：先创建合法 ES 通过 CRD/准入，再 Update 注入非法字段以触发 reconcile 侧 Validator。
func UpdateElasticScalerCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, mutate func(*unstructured.Unstructured) error) error {
	var lastErr error
	for i := 0; i < 5; i++ {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return err
		}
		if err := mutate(obj); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
}

// PatchESAnnotation 对 ES 的 metadata.annotations 做 merge patch（key=value），冲突时重试，用于竞态测试中的轻量 patch
func PatchESAnnotation(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, key, value string) error {
	var lastErr error
	for i := 0; i < 5; i++ {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return err
		}
		meta, _, _ := unstructured.NestedMap(obj.Object, "metadata")
		if meta == nil {
			meta = map[string]interface{}{}
		}
		ann, _ := meta["annotations"].(map[string]interface{})
		if ann == nil {
			ann = map[string]interface{}{}
		}
		ann[key] = value
		meta["annotations"] = ann
		if err := unstructured.SetNestedMap(obj.Object, meta, "metadata"); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		lastErr = err
		time.Sleep(200 * time.Millisecond)
	}
	return lastErr
}

// VerifyHPAMinMaxReplicas 校验 HPA minReplicas/maxReplicas 与 ES 一致
func VerifyHPAMinMaxReplicas(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace string) {
	es, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	minR, _, _ := unstructured.NestedInt64(es.Object, "spec", "minReplicas")
	maxR, _, _ := unstructured.NestedInt64(es.Object, "spec", "maxReplicas")
	hpa, err := GetHPA(clientset, ctx, esName+"-hpa", namespace)
	Expect(err).NotTo(HaveOccurred())
	Expect(*hpa.Spec.MinReplicas).To(Equal(int32(minR)))
	Expect(hpa.Spec.MaxReplicas).To(Equal(int32(maxR)))
}

// VerifyHPANotExists 校验 HPA 已被删除（用于删除 ES 后的垃圾回收验证）
func VerifyHPANotExists(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		_, err := clientset.AutoscalingV2().HorizontalPodAutoscalers(namespace).Get(ctx, name, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "HPA %s/%s 应在删除 ES 后被清理", namespace, name)
}

// VerifyESStateTriggerFailed 校验 StateTrigger 条件为失败（目标资源不存在等场景）
func VerifyESStateTriggerFailed(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		conds, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
		for _, c := range conds {
			cond, _ := c.(map[string]interface{})
			if cond["type"] == "StateTrigger" && cond["status"] == "False" {
				return true
			}
		}
		return false
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "ES %s/%s 应存在 StateTrigger Status=False 条件", namespace, name)
}

// esMetricsDeploymentRetryWindow 扩容类等待超时后的兜底重试窗口。
const esMetricsDeploymentRetryWindow = 2 * time.Minute

// WaitForDeploymentReplicas 等待 Deployment 就绪副本数达到目标值。
//
// 带一次兜底重试：扩容类等待（期望值大于当前就绪数）超时时，先打印 Deployment/Pod 诊断，
// 再删除明显卡住的 Pod（Pending、镜像拉取失败、反复重启等）促使其重建，然后重试等待一次；
// 仍不达标才失败，且失败信息附带完整诊断，避免只报“副本数应达到 N”而看不出原因。
func WaitForDeploymentReplicas(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, expected int32, timeout time.Duration) {
	observed, reached := waitForDeploymentReplicasReached(clientset, ctx, name, namespace, expected, timeout)
	if reached {
		return
	}
	if expected > 0 && observed < expected {
		GinkgoWriter.Printf("Deployment %s/%s 在 %s 内只达到 %d/%d 个就绪副本，诊断信息：\n%s\n",
			namespace, name, timeout, observed, expected, describeESMetricsTestDeployment(clientset, ctx, name, namespace))
		if removed := deleteStuckDeploymentPods(clientset, ctx, name, namespace); removed > 0 {
			GinkgoWriter.Printf("已删除 %d 个卡住的 Pod，重建后重试等待 %s\n", removed, esMetricsDeploymentRetryWindow)
			if _, ok := waitForDeploymentReplicasReached(clientset, ctx, name, namespace, expected, esMetricsDeploymentRetryWindow); ok {
				return
			}
		}
	}
	Fail(fmt.Sprintf("Deployment %s 副本数应达到 %d，实际 %d，诊断信息：\n%s",
		name, expected, observed, describeESMetricsTestDeployment(clientset, ctx, name, namespace)))
}

// waitForDeploymentReplicasReached 轮询等待就绪副本数达标，返回最后一次观测值与是否达标。
func waitForDeploymentReplicasReached(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, expected int32, timeout time.Duration) (int32, bool) {
	deadline := time.Now().Add(timeout)
	for {
		observed := deploymentReadyReplicas(clientset, ctx, name, namespace)
		if observed == expected {
			return observed, true
		}
		if !time.Now().Before(deadline) {
			return observed, false
		}
		time.Sleep(10 * time.Second)
	}
}

func deploymentReadyReplicas(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) int32 {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return -1
	}
	return dep.Status.ReadyReplicas
}

// deleteStuckDeploymentPods 删除明显卡住的 Pod，让 Deployment 重建 Pod：
// 对单节点环境下的调度或镜像拉取抖动做一次兜底。返回删除数量。
func deleteStuckDeploymentPods(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) int {
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: "app=" + name})
	if err != nil {
		return 0
	}
	propagation := metav1.DeletePropagationBackground
	deleted := 0
	for i := range pods.Items {
		pod := pods.Items[i]
		if pod.DeletionTimestamp != nil {
			continue
		}
		reason, stuck := stuckPodReason(pod)
		if !stuck {
			continue
		}
		GinkgoWriter.Printf("删除卡住的 Pod %s/%s（%s）\n", namespace, pod.Name, reason)
		if err := clientset.CoreV1().Pods(namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{PropagationPolicy: &propagation}); err != nil {
			GinkgoWriter.Printf("删除 Pod %s 失败：%v\n", pod.Name, err)
			continue
		}
		deleted++
	}
	return deleted
}

// stuckPodReason 判断 Pod 是否处于明显卡住的状态并给出可读原因。
func stuckPodReason(pod corev1.Pod) (string, bool) {
	switch pod.Status.Phase {
	case corev1.PodPending:
		return "Pending", true
	case corev1.PodFailed:
		return "Failed", true
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.State.Waiting != nil {
			switch cs.State.Waiting.Reason {
			case "ImagePullBackOff", "ErrImagePull", "InvalidImageName", "CreateContainerError", "CrashLoopBackOff":
				return cs.State.Waiting.Reason, true
			}
		}
		if !cs.Ready && cs.RestartCount >= 3 {
			return fmt.Sprintf("restarts=%d", cs.RestartCount), true
		}
	}
	return "", false
}

func UpdateESMetrics(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, metrics []ESMetricSpec) error {
	var lastErr error
	for i := 0; i < 5; i++ {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return err
		}
		metricsSlice := make([]interface{}, 0, len(metrics))
		for _, m := range metrics {
			metricsSlice = append(metricsSlice, map[string]interface{}{
				"type": m.Type,
				"resource": map[string]interface{}{
					"metricsName": m.ResourceName,
					"target": map[string]interface{}{
						"type":               m.TargetType,
						"averageUtilization": int64(m.AverageUtilization),
					},
				},
			})
		}
		if err := unstructured.SetNestedSlice(obj.Object, metricsSlice, "spec", "trigger", "metricsTrigger", "metrics"); err != nil {
			return err
		}
		_, err = dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Update(ctx, obj, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
}

func UpdateDeploymentReplicas(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) error {
	var lastErr error
	for i := 0; i < 8; i++ {
		dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		dep.Spec.Replicas = &replicas
		_, err = clientset.AppsV1().Deployments(namespace).Update(ctx, dep, metav1.UpdateOptions{})
		if err == nil {
			return nil
		}
		if !apierrors.IsConflict(err) {
			return err
		}
		// HPA / controller 可能并发改 Deployment，409 时重新 Get 再 Update
		lastErr = err
		time.Sleep(500 * time.Millisecond)
	}
	return lastErr
}

// UpdateDeploymentReplicasIfExists 同 UpdateDeploymentReplicas；Deployment 不存在时视为已清理，返回 nil。
func UpdateDeploymentReplicasIfExists(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) error {
	err := UpdateDeploymentReplicas(clientset, ctx, name, namespace, replicas)
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}
