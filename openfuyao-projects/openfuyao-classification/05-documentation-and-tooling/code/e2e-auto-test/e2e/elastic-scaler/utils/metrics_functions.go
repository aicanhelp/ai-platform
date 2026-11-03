package utils

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/json"
	"encoding/pem"
	"fmt"
	"math/big"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// InstallElasticScalerForMetrics 安装 elastic-scaler controller（通用扩缩容决策框架 E2E 专用）。
// 与 HPA 路径测试不同：关闭 tidal / RSG 样例 CR，仅保留 controller + CRD + webhook。
func InstallElasticScalerForMetrics() error {
	return ensureElasticScalerForMetrics(false)
}

// EnsureElasticScalerForMetrics 幂等安装/升级 elastic-scaler（focus 单用例或重复跑套件时 BeforeAll 可安全调用）。
func EnsureElasticScalerForMetrics() error {
	return ensureElasticScalerForMetrics(true)
}

func ensureElasticScalerForMetrics(upgradeInstall bool) error {
	return RunElasticScalerHelmInstall(upgradeInstall, false)
}

// VerifyElasticScalerCRDAvailable 确认 elasticscaler.io/v1alpha1 API 已注册。
func VerifyElasticScalerCRDAvailable(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("验证 ElasticScaler CRD/API 可用...")
	Eventually(func() error {
		_, err := clientset.Discovery().ServerResourcesForGroupVersion("elasticscaler.io/v1alpha1")
		return err
	}, 2*time.Minute, 5*time.Second).Should(Succeed(), "elasticscaler.io/v1alpha1 API 应可用")
}

// ESApaMetricEntry 描述 apa 路径 MetricsTrigger 中的单条指标。
type ESApaMetricEntry struct {
	Type         string // Resource | ExternalServer | External | Custom
	ResourceName string // Resource 类型：cpu / memory
	TargetType   string // Resource 类型：Utilization
	AverageUtil  int32  // Resource Utilization 目标

	// ExternalServer 字段（HTTP/HTTPS 拉取 Prometheus 查询响应；endpoint 可含 scheme，protocol 未填时 controller 默认 https）
	ExternalServerName     string
	ExternalServerTarget   string
	ExternalServerEndpoint string
	ExternalServerProtocol string
	ExternalServerQuery    string

	// External 字段（external.metrics.k8s.io；集群无 adapter 时可 mock 返回值）
	ExternalName   string
	ExternalTarget int64

	// Custom 字段（custom.metrics.k8s.io；集群无 adapter 时可 mock 返回值）
	CustomName   string
	CustomTarget int64
}

func buildApaMetricSlice(metrics []ESApaMetricEntry) []interface{} {
	out := make([]interface{}, 0, len(metrics))
	for _, m := range metrics {
		switch m.Type {
		case "Resource":
			out = append(out, map[string]interface{}{
				"type": "Resource",
				"resource": map[string]interface{}{
					"metricsName": m.ResourceName,
					"target": map[string]interface{}{
						"type":               m.TargetType,
						"averageUtilization": int64(m.AverageUtil),
					},
				},
			})
		case "ExternalServer":
			extSrv := map[string]interface{}{
				"metricsName": m.ExternalServerName,
				"target":      m.ExternalServerTarget,
				"endpoint":    m.ExternalServerEndpoint,
			}
			if p := strings.TrimSpace(m.ExternalServerProtocol); p != "" {
				extSrv["protocol"] = p
			}
			if m.ExternalServerQuery != "" {
				extSrv["query"] = m.ExternalServerQuery
			}
			out = append(out, map[string]interface{}{
				"type":           "ExternalServer",
				"externalServer": extSrv,
			})
		case "External":
			out = append(out, map[string]interface{}{
				"type": "External",
				"external": map[string]interface{}{
					"metricsName": m.ExternalName,
					"target": map[string]interface{}{
						"type":         "AverageValue",
						"averageValue": strconv.FormatInt(m.ExternalTarget, 10),
					},
				},
			})
		case "Custom":
			out = append(out, map[string]interface{}{
				"type": "Custom",
				"custom": map[string]interface{}{
					"metricsName": m.CustomName,
					"target": map[string]interface{}{
						"type":         "AverageValue",
						"averageValue": strconv.FormatInt(m.CustomTarget, 10),
					},
				},
			})
		}
	}
	return out
}

// BuildApaMetricsTriggerESObject 构造 scalingAlgorithm=apa 的 MetricsTrigger ES。
func BuildApaMetricsTriggerESObject(name, namespace string, targetRef ESMetricsTargetRef, minReplicas, maxReplicas int32, metrics []ESApaMetricEntry) *unstructured.Unstructured {
	if targetRef.APIVersion == "" {
		targetRef = ESMetricsTargetRef{APIVersion: "apps/v1", Kind: "Deployment", Name: MetricsTestDeployName, Namespace: namespace}
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
				"minReplicas": int64(minReplicas),
				"maxReplicas": int64(maxReplicas),
				"targetRef": map[string]interface{}{
					"apiVersion": targetRef.APIVersion,
					"kind":       targetRef.Kind,
					"name":       targetRef.Name,
					"namespace":  targetRef.Namespace,
				},
				"trigger": map[string]interface{}{
					"type": "MetricsTrigger",
					"metricsTrigger": map[string]interface{}{
						"scalingAlgorithm": MetricsScalingAlgorithmApa,
						"metrics":          buildApaMetricSlice(metrics),
					},
				},
			},
		},
	}
}

// CreateApaElasticScalerCR 创建 apa 算法 MetricsTrigger ES（若已存在则先删除再创建，避免上次失败残留）。
func CreateApaElasticScalerCR(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef, minReplicas, maxReplicas int32, metrics []ESApaMetricEntry) error {
	return CreateApaElasticScalerCRWithAnnotations(dynamicClient, ctx, name, namespace, targetRef, minReplicas, maxReplicas, metrics, nil)
}

// CreateApaElasticScalerCRWithAnnotations 创建带 metadata.annotations 的 apa MetricsTrigger ES。
func CreateApaElasticScalerCRWithAnnotations(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, targetRef ESMetricsTargetRef, minReplicas, maxReplicas int32, metrics []ESApaMetricEntry, annotations map[string]string) error {
	if err := DeleteElasticScalerCRAndWait(dynamicClient, ctx, name, namespace); err != nil {
		return err
	}
	obj := BuildApaMetricsTriggerESObject(name, namespace, targetRef, minReplicas, maxReplicas, metrics)
	if len(annotations) > 0 {
		meta, _, _ := unstructured.NestedMap(obj.Object, "metadata")
		if meta == nil {
			meta = map[string]interface{}{}
		}
		ann := map[string]interface{}{}
		for k, v := range annotations {
			ann[k] = v
		}
		meta["annotations"] = ann
		obj.Object["metadata"] = meta
	}
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

// ValidMetricsPolicyAnnotationsE2E002 返回 ES-E2E-002 合法策略注解（含大小写混合以验证归一化）。
func ValidMetricsPolicyAnnotationsE2E002() map[string]string {
	return map[string]string{
		AnnotationMetricsAggregationType:   "mAx",
		AnnotationMetricsMissingDataPolicy: "IgNoRe",
		AnnotationMetricsWindowSeconds:     "120",
	}
}

// InvalidMetricsPolicyAnnotationsE2E003 返回 ES-E2E-003 非法策略注解（应被忽略并回退默认，不拖垮采集）。
func InvalidMetricsPolicyAnnotationsE2E003() map[string]string {
	return map[string]string{
		AnnotationMetricsWindowSeconds:     "-1",
		AnnotationMetricsAggregationType:     "UnknownType",
		AnnotationMetricsEnableOutlierFilter: "not-a-bool",
	}
}

// PatchESApaExternalServerPromQL Patch apa ES 的首条 ExternalServer 指标 query（PromQL 参数）。
func PatchESApaExternalServerPromQL(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, promQL string) error {
	queryParam := "query=" + url.QueryEscape(promQL)
	return UpdateElasticScalerCR(dynamicClient, ctx, name, namespace, func(u *unstructured.Unstructured) error {
		metrics, found, err := unstructured.NestedSlice(u.Object, "spec", "trigger", "metricsTrigger", "metrics")
		if err != nil || !found || len(metrics) == 0 {
			return fmt.Errorf("metricsTrigger.metrics not found on ES %s/%s", namespace, name)
		}
		updated := false
		for i, m := range metrics {
			metric, ok := m.(map[string]interface{})
			if !ok || metric["type"] != "ExternalServer" {
				continue
			}
			extSrv, ok := metric["externalServer"].(map[string]interface{})
			if !ok {
				continue
			}
			extSrv["query"] = queryParam
			metric["externalServer"] = extSrv
			metrics[i] = metric
			updated = true
			break
		}
		if !updated {
			return fmt.Errorf("no ExternalServer metric found on ES %s/%s", namespace, name)
		}
		return unstructured.SetNestedSlice(u.Object, metrics, "spec", "trigger", "metricsTrigger", "metrics")
	})
}

// PatchESAnnotations 合并 patch ES metadata.annotations。
func PatchESAnnotations(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, annotations map[string]string) error {
	return UpdateElasticScalerCR(dynamicClient, ctx, name, namespace, func(u *unstructured.Unstructured) error {
		meta, _, _ := unstructured.NestedMap(u.Object, "metadata")
		if meta == nil {
			meta = map[string]interface{}{}
		}
		ann, _ := meta["annotations"].(map[string]interface{})
		if ann == nil {
			ann = map[string]interface{}{}
		}
		for k, v := range annotations {
			ann[k] = v
		}
		meta["annotations"] = ann
		return unstructured.SetNestedMap(u.Object, meta, "metadata")
	})
}

// ESMetricsPipelineDetail MetricsPipeline 条件详情。
type ESMetricsPipelineDetail struct {
	Status             string
	Reason             string
	Message            string
	ObservedGeneration int64
	Found              bool
}

func getESMetricsPipelineDetail(obj *unstructured.Unstructured) ESMetricsPipelineDetail {
	conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	for _, c := range conditions {
		cond, _ := c.(map[string]interface{})
		if cond["type"] != "MetricsPipeline" {
			continue
		}
		detail := ESMetricsPipelineDetail{Found: true}
		detail.Status, _ = cond["status"].(string)
		detail.Reason, _ = cond["reason"].(string)
		detail.Message, _ = cond["message"].(string)
		switch v := cond["observedGeneration"].(type) {
		case int64:
			detail.ObservedGeneration = v
		case float64:
			detail.ObservedGeneration = int64(v)
		case json.Number:
			detail.ObservedGeneration, _ = v.Int64()
		}
		return detail
	}
	return ESMetricsPipelineDetail{}
}

// ESCondition 从 status.conditions 中提取指定 Type 的条件。
func ESCondition(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, condType string) (status, reason, message string, found bool, err error) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	if err != nil {
		return "", "", "", false, err
	}
	conditions, _, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	for _, c := range conditions {
		cond, _ := c.(map[string]interface{})
		if cond["type"] != condType {
			continue
		}
		status, _ = cond["status"].(string)
		reason, _ = cond["reason"].(string)
		message, _ = cond["message"].(string)
		return status, reason, message, true, nil
	}
	return "", "", "", false, nil
}

// DefaultResourceCPUMetric 返回 ES-E2E 基线 Resource/cpu 指标（Utilization=50）。
func DefaultResourceCPUMetric() ESApaMetricEntry {
	return ESApaMetricEntry{
		Type:         "Resource",
		ResourceName: "cpu",
		TargetType:   "Utilization",
		AverageUtil:  50,
	}
}

// DefaultMockExternalMetric 返回 External 类型指标 spec（无 adapter 时由 E2E mock 采集返回值）。
func DefaultMockExternalMetric(name string, target int64) ESApaMetricEntry {
	if name == "" {
		name = "qps"
	}
	if target == 0 {
		target = 100
	}
	return ESApaMetricEntry{
		Type:           "External",
		ExternalName:   name,
		ExternalTarget: target,
	}
}

// DefaultMockCustomMetric 返回 Custom 类型指标 spec（无 adapter 时由 E2E mock 采集返回值）。
func DefaultMockCustomMetric(name string, target int64) ESApaMetricEntry {
	if name == "" {
		name = "pods/inference_qps"
	}
	if target == 0 {
		target = 100
	}
	return ESApaMetricEntry{
		Type:         "Custom",
		CustomName:   name,
		CustomTarget: target,
	}
}

// EnsureMetricsServerAvailable 确认 metrics-server（metrics.k8s.io）可用。
func EnsureMetricsServerAvailable(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("验证 metrics-server API 可用...")
	Eventually(func() error {
		_, err := clientset.Discovery().ServerResourcesForGroupVersion("metrics.k8s.io/v1beta1")
		return err
	}, 2*time.Minute, 5*time.Second).Should(Succeed(), "metrics.k8s.io/v1beta1 应可用")
}

// WaitForDeploymentPodMetricsReady 等待 Deployment 当前 Ready 副本的 CPU 指标可被 metrics-server 采集（连续三次采样各间隔 20s，适配慢 scrape 集群）。
func WaitForDeploymentPodMetricsReady(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string, timeout time.Duration) {
	if timeout <= 0 {
		timeout = MetricsPodMetricsWaitTimeout()
	}
	GinkgoWriter.Printf("等待 Deployment %s/%s Pod metrics 就绪（timeout=%s）...\n", namespace, deployName, timeout)
	Eventually(func() bool {
		return deploymentsPodMetricsStable(clientset, ctx, namespace, []string{deployName})
	}, timeout, 15*time.Second).Should(BeTrue(), "Deployment %s/%s Pod metrics 应稳定就绪", namespace, deployName)
}

func isPodReadyForMetrics(pod corev1.Pod) bool {
	if pod.DeletionTimestamp != nil {
		return false
	}
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady && c.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}

func deploymentPodMetricsAvailable(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) bool {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil || dep.Status.ReadyReplicas < 1 || dep.Spec.Selector == nil {
		return false
	}
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: metav1.FormatLabelSelector(dep.Spec.Selector),
	})
	if err != nil {
		return false
	}
	readyPods := make([]corev1.Pod, 0, len(pods.Items))
	for _, pod := range pods.Items {
		if isPodReadyForMetrics(pod) {
			readyPods = append(readyPods, pod)
		}
	}
	if int32(len(readyPods)) < dep.Status.ReadyReplicas {
		GinkgoWriter.Printf("Deployment %s/%s ready pods=%d < readyReplicas=%d\n",
			namespace, deployName, len(readyPods), dep.Status.ReadyReplicas)
		return false
	}
	limit := int(dep.Status.ReadyReplicas)
	if limit > len(readyPods) {
		limit = len(readyPods)
	}
	for i := 0; i < limit; i++ {
		podName := readyPods[i].Name
		raw, err := clientset.RESTClient().Get().
			AbsPath("apis/metrics.k8s.io/v1beta1/namespaces", namespace, "pods", podName).
			Do(ctx).Raw()
		if err != nil {
			GinkgoWriter.Printf("pod %s metrics 未就绪: %v\n", podName, err)
			return false
		}
		var pm struct {
			Containers []struct {
				Name string `json:"name"`
			} `json:"containers"`
		}
		if err := json.Unmarshal(raw, &pm); err != nil || len(pm.Containers) == 0 {
			GinkgoWriter.Printf("pod %s metrics 响应无效或 containers 为空\n", podName)
			return false
		}
	}
	return true
}

var (
	clusterPrometheusNamespace string
	clusterPrometheusService   string
)

func prometheusServiceCandidates() []struct{ namespace, service string } {
	ns := strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_NAMESPACE"))
	if ns == "" {
		ns = DefaultPrometheusNamespace
	}
	if svc := strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_SERVICE")); svc != "" {
		return []struct{ namespace, service string }{{ns, svc}}
	}
	return []struct{ namespace, service string }{
		{ns, DefaultPrometheusService},
		{ns, "kube-prometheus-kube-prome-prometheus"},
		{ns, "prometheus-k8s"},
		// infernex / 部分安装将 Prometheus 放在业务命名空间（Service 名常与 Prometheus CR 同名）
		{"ai-inference", "infernex-qwen32b-kube-prom-prometheus"},
		{"ai-inference", "infernex-kube-prometheus-s-prometheus"},
		{"ai-inference", "kube-prometheus-kube-prome-prometheus"},
		{"ai-inference", "prometheus-k8s"},
		{"test-inference", "infernex-testns-kube-prome-prometheus"},
		{"test-inference", "infernex-qwen32b-kube-prom-prometheus"},
		{"test-inference", "kube-prometheus-kube-prome-prometheus"},
	}
}

func prometheusServiceHasReadyEndpoints(clientset *kubernetes.Clientset, ctx context.Context, namespace, service string) bool {
	eps, err := clientset.CoreV1().Endpoints(namespace).Get(ctx, service, metav1.GetOptions{})
	if err != nil {
		return false
	}
	for _, subset := range eps.Subsets {
		if len(subset.Addresses) == 0 {
			continue
		}
		for _, p := range subset.Ports {
			if p.Port == DefaultPrometheusPort {
				return true
			}
		}
	}
	return false
}

func rememberClusterPrometheus(namespace, service string) {
	clusterPrometheusNamespace = namespace
	clusterPrometheusService = service
}

func clearClusterPrometheusCache() {
	clusterPrometheusNamespace = ""
	clusterPrometheusService = ""
}

func serviceHasPort9090(svc *corev1.Service) bool {
	for _, p := range svc.Spec.Ports {
		if p.Port == DefaultPrometheusPort {
			return true
		}
	}
	return false
}

func isPrometheusQueryServiceName(name string) bool {
	lower := strings.ToLower(name)
	if lower == "prometheus-operated" {
		return false
	}
	if !strings.Contains(lower, "prometheus") && !strings.Contains(lower, "kube-prom") {
		return false
	}
	for _, exclude := range []string{
		"kubelet", "node-exporter", "coredns", "etcd", "scheduler",
		"controller-manager", "kube-proxy", "operator",
	} {
		if strings.Contains(lower, exclude) {
			return false
		}
	}
	return true
}

func scorePrometheusQueryService(svc *corev1.Service) int {
	if !serviceHasPort9090(svc) || !isPrometheusQueryServiceName(svc.Name) {
		return -1
	}
	score := 10
	lower := strings.ToLower(svc.Name)
	if strings.Contains(lower, "kube-prome") || strings.Contains(lower, "kube-prom-prometheus") || lower == "prometheus-k8s" {
		score += 5
	}
	if strings.Contains(lower, "infernex") {
		score += 4
	}
	if strings.HasSuffix(lower, "-prometheus") {
		score += 3
	}
	if svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != "None" {
		score += 2
	}
	return score
}

var infernexPrometheusNamespaces = []string{"ai-inference", "test-inference", DefaultPrometheusNamespace}

// discoverPrometheusByAppLabel 在 infernex/常见命名空间中按 app.kubernetes.io/name=prometheus 查找 Endpoints 已就绪的 Service。
func discoverPrometheusByAppLabel(clientset *kubernetes.Clientset, ctx context.Context) (namespace, service string, ok bool) {
	bestScore := -1
	for _, ns := range infernexPrometheusNamespaces {
		svcs, err := clientset.CoreV1().Services(ns).List(ctx, metav1.ListOptions{
			LabelSelector: "app.kubernetes.io/name=prometheus",
		})
		if err != nil {
			continue
		}
		for i := range svcs.Items {
			svc := &svcs.Items[i]
			score := scorePrometheusQueryService(svc)
			if score <= bestScore {
				continue
			}
			if !prometheusServiceHasReadyEndpoints(clientset, ctx, ns, svc.Name) {
				continue
			}
			bestScore = score
			namespace, service = ns, svc.Name
		}
	}
	return namespace, service, bestScore >= 0
}

// scanClusterForPrometheusService 在所有命名空间中查找 Endpoints 已就绪的 Prometheus 查询 Service。
func scanClusterForPrometheusService(clientset *kubernetes.Clientset, ctx context.Context) (namespace, service string, ok bool) {
	nss, err := clientset.CoreV1().Namespaces().List(ctx, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("列举命名空间失败，无法扫描 Prometheus Service: %v\n", err)
		return "", "", false
	}
	bestScore := -1
	for _, ns := range nss.Items {
		svcs, err := clientset.CoreV1().Services(ns.Name).List(ctx, metav1.ListOptions{})
		if err != nil {
			continue
		}
		for i := range svcs.Items {
			svc := &svcs.Items[i]
			score := scorePrometheusQueryService(svc)
			if score <= bestScore {
				continue
			}
			if !prometheusServiceHasReadyEndpoints(clientset, ctx, ns.Name, svc.Name) {
				continue
			}
			bestScore = score
			namespace, service = ns.Name, svc.Name
		}
	}
	return namespace, service, bestScore >= 0
}

// ResolveClusterPrometheus 探测集群内 Endpoints 已就绪的 Prometheus Service（结果缓存供 Endpoint 构造复用）。
func ResolveClusterPrometheus(clientset *kubernetes.Clientset, ctx context.Context) (namespace, service string, ok bool) {
	if clusterPrometheusService != "" {
		if prometheusServiceHasReadyEndpoints(clientset, ctx, clusterPrometheusNamespace, clusterPrometheusService) {
			return clusterPrometheusNamespace, clusterPrometheusService, true
		}
		GinkgoWriter.Printf(
			"Prometheus Service %s/%s Endpoints 未就绪，重新探测\n",
			clusterPrometheusNamespace, clusterPrometheusService,
		)
		clearClusterPrometheusCache()
	}
	var tried []string
	for _, candidate := range prometheusServiceCandidates() {
		key := candidate.namespace + "/" + candidate.service
		tried = append(tried, key)
		if _, err := clientset.CoreV1().Services(candidate.namespace).Get(ctx, candidate.service, metav1.GetOptions{}); err != nil {
			continue
		}
		if !prometheusServiceHasReadyEndpoints(clientset, ctx, candidate.namespace, candidate.service) {
			GinkgoWriter.Printf("跳过 Prometheus 候选 %s（Service 存在但 Endpoints 未就绪）\n", key)
			continue
		}
		rememberClusterPrometheus(candidate.namespace, candidate.service)
		GinkgoWriter.Printf("已发现 Prometheus Service: %s/%s\n", candidate.namespace, candidate.service)
		logPrometheusProtocolHint()
		return candidate.namespace, candidate.service, true
	}
	if ns, svc, found := discoverPrometheusByAppLabel(clientset, ctx); found {
		rememberClusterPrometheus(ns, svc)
		GinkgoWriter.Printf("已发现 Prometheus Service（label 探测）: %s/%s\n", ns, svc)
		logPrometheusProtocolHint()
		return ns, svc, true
	}
	if ns, svc, found := scanClusterForPrometheusService(clientset, ctx); found {
		rememberClusterPrometheus(ns, svc)
		GinkgoWriter.Printf("已发现 Prometheus Service（集群扫描）: %s/%s\n", ns, svc)
		logPrometheusProtocolHint()
		return ns, svc, true
	}
	GinkgoWriter.Printf(
		"未找到 Endpoints 已就绪的 Prometheus Service（已尝试: %s）；可设置 METRICS_PROMETHEUS_NAMESPACE / METRICS_PROMETHEUS_SERVICE 或 METRICS_PROMETHEUS_ENDPOINT_URL\n",
		strings.Join(tried, ", "),
	)
	return "", "", false
}

// ClusterPrometheusQueryEndpoint 返回集群内 Prometheus /api/v1/query 主机路径（不含 scheme）。
// 需先调用 ResolveClusterPrometheus；亦可通过 METRICS_PROMETHEUS_NAMESPACE / METRICS_PROMETHEUS_SERVICE 覆盖。
// 完整 URL（含 http:// 或 https://）可通过 METRICS_PROMETHEUS_ENDPOINT_URL 覆盖；含 scheme 时 protocol 字段被忽略。
func ClusterPrometheusQueryEndpoint() string {
	ns := clusterPrometheusNamespace
	svc := clusterPrometheusService
	if svc == "" {
		ns = strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_NAMESPACE"))
		if ns == "" {
			ns = DefaultPrometheusNamespace
		}
		svc = strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_SERVICE"))
		if svc == "" {
			svc = DefaultPrometheusService
		}
	}
	return fmt.Sprintf("%s.%s.svc.cluster.local:%d%s", svc, ns, DefaultPrometheusPort, DefaultPrometheusQueryPath)
}

// ClusterPrometheusExternalServerEndpoint 返回 ExternalServer endpoint。
// 优先 METRICS_PROMETHEUS_ENDPOINT_URL（可含 http:// 或 https://）。
// 否则构造集群内 Service URL：未显式设置 METRICS_PROMETHEUS_PROTOCOL 时默认带 http:// scheme
//（kube-prometheus 集群内通常为 HTTP:9090；含 scheme 后 controller 直接使用 HTTP，避免默认 https 误连）。
func ClusterPrometheusExternalServerEndpoint() string {
	if u := strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_ENDPOINT_URL")); u != "" {
		return u
	}
	hostPath := ClusterPrometheusQueryEndpoint()
	if proto := PrometheusQueryProtocol(); proto != "" {
		return hostPath
	}
	return "http://" + hostPath
}

// PrometheusQueryProtocol 返回 ExternalServer protocol（仅 METRICS_PROMETHEUS_PROTOCOL 显式设置时写入 CR）。
// 未设置且 endpoint 无 scheme 时省略 protocol，controller 默认补全为 https。
func PrometheusQueryProtocol() string {
	return strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_PROTOCOL"))
}

func logPrometheusProtocolHint() {
	if strings.Contains(ClusterPrometheusExternalServerEndpoint(), "://") {
		return
	}
	if PrometheusQueryProtocol() == "" {
		return
	}
	GinkgoWriter.Println(
		"提示: endpoint 无 scheme 且已设置 METRICS_PROMETHEUS_PROTOCOL，请确认与集群 Prometheus 实际协议一致",
	)
}

// ResolveExternalServerProtocol 当 endpoint 已含 scheme 时返回空（protocol 被忽略）；否则返回显式 protocol。
func ResolveExternalServerProtocol(endpoint, protocol string) string {
	if strings.Contains(strings.TrimSpace(endpoint), "://") {
		return ""
	}
	return strings.TrimSpace(protocol)
}

// DefaultClusterExternalServerMetric 构造指向集群 Prometheus 的 ExternalServer 指标（真实端点，非 mock）。
func DefaultClusterExternalServerMetric(metricsName, promQL string) ESApaMetricEntry {
	if metricsName == "" {
		metricsName = "prom_test_metric"
	}
	if promQL == "" {
		promQL = strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_TEST_QUERY"))
	}
	endpoint := ClusterPrometheusExternalServerEndpoint()
	return ESApaMetricEntry{
		Type:                   "ExternalServer",
		ExternalServerName:     metricsName,
		ExternalServerTarget:   "100",
		ExternalServerEndpoint: endpoint,
		ExternalServerProtocol: ResolveExternalServerProtocol(endpoint, PrometheusQueryProtocol()),
		ExternalServerQuery:    promQL,
	}
}

// VerifyESHasNoMetricsPolicyAnnotations 确认未设置 elasticscaler.io/metrics.* 策略注解。
func VerifyESHasNoMetricsPolicyAnnotations(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	ann, _, _ := unstructured.NestedStringMap(obj.Object, "metadata", "annotations")
	for k := range ann {
		Expect(k).NotTo(HavePrefix("elasticscaler.io/metrics."),
			"ES %s/%s 不应含 metrics 策略注解 %q", namespace, name, k)
	}
}

// ResourceMetricsWaitRecovery 在 MetricsPipeline 因 pod metrics 暂不可达而 PartialSuccess 时，辅助重试等待 Deployment Pod metrics。
type ResourceMetricsWaitRecovery struct {
	Clientset        *kubernetes.Clientset
	Namespace        string
	DeployName       string
	ExtraDeployNames []string
}

func (r *ResourceMetricsWaitRecovery) deploymentNames() []string {
	if r == nil {
		return nil
	}
	names := make([]string, 0, 1+len(r.ExtraDeployNames))
	if r.DeployName != "" {
		names = append(names, r.DeployName)
	}
	names = append(names, r.ExtraDeployNames...)
	return names
}

func tryRecoverResourcePodMetrics(recovery *ResourceMetricsWaitRecovery, ctx context.Context) {
	if recovery == nil || recovery.Clientset == nil {
		return
	}
	names := recovery.deploymentNames()
	if len(names) == 0 {
		return
	}
	deadline := time.Now().Add(3 * time.Minute)
	for time.Now().Before(deadline) {
		if deploymentsPodMetricsStable(recovery.Clientset, ctx, recovery.Namespace, names) {
			GinkgoWriter.Printf("Deployment Pod metrics 已稳定就绪: %v\n", names)
			return
		}
		time.Sleep(15 * time.Second)
	}
	GinkgoWriter.Printf("Deployment Pod metrics 恢复等待超时（%v），继续轮询 MetricsPipeline\n", names)
}

// deploymentsPodMetricsStable 与 WaitForDeploymentPodMetricsReady 一致：连续 3 次采样均可用（间隔 20s）。
func deploymentsPodMetricsStable(clientset *kubernetes.Clientset, ctx context.Context, namespace string, deployNames []string) bool {
	for i := 0; i < 3; i++ {
		for _, name := range deployNames {
			if !deploymentPodMetricsAvailable(clientset, ctx, namespace, name) {
				return false
			}
		}
		if i < 2 {
			time.Sleep(20 * time.Second)
		}
	}
	return true
}
// VerifyESMetricsCollectionFullSuccess 校验 MetricsPipeline 全量采集成功语义。
// requireCurrentGeneration=true 时要求 MetricsPipeline.observedGeneration 已对齐 metadata.generation（Patch 后必开）。
// recovery 可选：Resource/cpu 路径在 pod metrics 暂不可达时辅助重试。
func VerifyESMetricsCollectionFullSuccess(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, processedCount int, requireCurrentGeneration bool, recovery ...*ResourceMetricsWaitRecovery) {
	msg := fmt.Sprintf("processed %d metric(s)", processedCount)
	var r *ResourceMetricsWaitRecovery
	if len(recovery) > 0 {
		r = recovery[0]
	}
	VerifyESMetricsPipelineCondition(dynamicClient, ctx, name, namespace, "True", MetricsReasonCollectionSucceeded, requireCurrentGeneration, r, msg)
}

// VerifyESGetMetricsFullSuccess ES-E2E-005：GetMetrics 全量成功，MetricsPipeline 无 PartialSuccess / CollectFailures 语义。
func VerifyESGetMetricsFullSuccess(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, processedCount int, requireCurrentGeneration bool, recovery ...*ResourceMetricsWaitRecovery) {
	VerifyESMetricsCollectionFullSuccess(dynamicClient, ctx, name, namespace, processedCount, requireCurrentGeneration, recovery...)
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	detail := getESMetricsPipelineDetail(obj)
	Expect(detail.Found).To(BeTrue(), "ES %s/%s 应存在 MetricsPipeline 条件", namespace, name)
	Expect(detail.Reason).To(Equal(MetricsReasonCollectionSucceeded),
		"ES %s/%s MetricsPipeline Reason 应为全量成功，实际 message=%s", namespace, name, detail.Message)
	Expect(detail.Reason).NotTo(Equal(MetricsReasonPartialSuccess),
		"ES %s/%s 不应出现 MetricsPartialSuccess: %s", namespace, name, detail.Message)
	Expect(detail.Message).NotTo(ContainSubstring("(Error):"),
		"ES %s/%s MetricsPipeline message 不应含 CollectFailures 汇总: %s", namespace, name, detail.Message)
}

// VerifyESResourceMetricsWithPodListSuccess ES-E2E-010：Resource/cpu 全量成功且 PodList 注入正常（无 no pods in metrics context）。
func VerifyESResourceMetricsWithPodListSuccess(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, recovery ...*ResourceMetricsWaitRecovery) {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, name, namespace, 1, true, recovery...)
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	detail := getESMetricsPipelineDetail(obj)
	Expect(detail.Message).NotTo(ContainSubstring("no pods in metrics context"),
		"ES %s/%s MetricsPipeline 不应出现 PodList 为空语义: %s", namespace, name, detail.Message)
	Expect(detail.Message).NotTo(ContainSubstring("no ready pods"),
		"ES %s/%s MetricsPipeline 不应出现无 Ready Pod 语义: %s", namespace, name, detail.Message)
}

// VerifyESGetMetricsPartialSuccess ES-E2E-006：GetMetrics 部分成功，MetricsPipeline 含 CollectFailures 汇总。
func VerifyESGetMetricsPartialSuccess(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string, failedMetricNames ...string) {
	requireCurrentGeneration := true
	VerifyESMetricsPipelineCondition(dynamicClient, ctx, name, namespace, "True", MetricsReasonPartialSuccess, requireCurrentGeneration, nil)
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	detail := getESMetricsPipelineDetail(obj)
	Expect(detail.Found).To(BeTrue(), "ES %s/%s 应存在 MetricsPipeline 条件", namespace, name)
	Expect(detail.Reason).To(Equal(MetricsReasonPartialSuccess),
		"ES %s/%s MetricsPipeline Reason 应为部分成功", namespace, name)
	Expect(detail.Reason).NotTo(Equal(MetricsReasonCollectionSucceeded),
		"ES %s/%s 不应为全量采集成功: %s", namespace, name, detail.Message)
	for _, metricName := range failedMetricNames {
		Expect(detail.Message).To(ContainSubstring(metricName),
			"ES %s/%s MetricsPipeline message 应含失败指标 %q: %s", namespace, name, metricName, detail.Message)
		Expect(detail.Message).To(ContainSubstring(metricName+" (Error):"),
			"ES %s/%s MetricsPipeline message 应含 CollectFailures 格式 %q (Error): ...: %s", namespace, name, metricName, detail.Message)
	}
}

// VerifyESMetricsPipelineCondition 轮询 MetricsPipeline 条件。
func VerifyESMetricsPipelineCondition(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, expectStatus, expectReason string, requireCurrentGeneration bool, recovery *ResourceMetricsWaitRecovery, messageContains ...string) {
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
		if err != nil {
			return false
		}
		gen, _, _ := unstructured.NestedInt64(obj.Object, "metadata", "generation")
		detail := getESMetricsPipelineDetail(obj)
		if !detail.Found {
			GinkgoWriter.Printf("ES %s/%s MetricsPipeline 条件尚未写入，等待 controller Reconcile\n", namespace, name)
			return false
		}
		if requireCurrentGeneration && detail.ObservedGeneration < gen {
			GinkgoWriter.Printf("MetricsPipeline observedGeneration=%d < generation=%d，等待 Reconcile\n",
				detail.ObservedGeneration, gen)
			return false
		}
		if detail.Status != expectStatus || detail.Reason != expectReason {
			if recovery != nil &&
				expectReason == MetricsReasonCollectionSucceeded &&
				detail.Reason == MetricsReasonPartialSuccess &&
				(strings.Contains(detail.Message, "podmetrics") || strings.Contains(detail.Message, "get pod metrics")) {
				GinkgoWriter.Printf("MetricsPipeline 暂为 PartialSuccess（pod metrics 未就绪），尝试恢复: %s\n", detail.Message)
				tryRecoverResourcePodMetrics(recovery, ctx)
			} else if expectReason == MetricsReasonCollectionSucceeded &&
				detail.Reason == MetricsReasonPartialSuccess &&
				strings.Contains(detail.Message, "podmetrics") {
				GinkgoWriter.Printf("MetricsPipeline 暂为 PartialSuccess（pod metrics 未就绪），继续等待: %s\n", detail.Message)
			} else {
				GinkgoWriter.Printf("MetricsPipeline 未达期望: status=%s reason=%s message=%s\n",
					detail.Status, detail.Reason, detail.Message)
			}
			return false
		}
		if expectReason == MetricsReasonCollectionSucceeded {
			if detail.Reason == MetricsReasonPartialSuccess ||
				detail.Reason == "BuildMetricsContextFailed" ||
				detail.Reason == "MetricsCollectionFailed" {
				return false
			}
		}
		for _, sub := range messageContains {
			if sub != "" && !strings.Contains(detail.Message, sub) {
				GinkgoWriter.Printf("MetricsPipeline message 缺少 %q: %s\n", sub, detail.Message)
				return false
			}
		}
		return true
	}, MetricsPipelineWaitTimeout(), 5*time.Second).Should(BeTrue(),
		"ES %s/%s MetricsPipeline 应为 Status=%s Reason=%s", namespace, name, expectStatus, expectReason)
}

// VerifyESScalingCalculatedCondition 轮询 ScalingCalculated 条件。
func VerifyESScalingCalculatedCondition(dynamicClient dynamic.Interface, ctx context.Context, name, namespace, expectStatus, expectReason string) {
	Eventually(func() bool {
		status, reason, _, found, err := ESCondition(dynamicClient, ctx, name, namespace, "ScalingCalculated")
		if err != nil || !found {
			return false
		}
		return status == expectStatus && reason == expectReason
	}, 3*time.Minute, 5*time.Second).Should(BeTrue(),
		"ES %s/%s ScalingCalculated 应为 Status=%s Reason=%s", namespace, name, expectStatus, expectReason)
}

// VerifyESMetricsPipelineAbsentOrFailed 确认 MetricsPipeline 不存在或为失败态（非法 CR 场景）。
func VerifyESMetricsPipelineAbsentOrFailed(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	Eventually(func() bool {
		status, reason, _, found, err := ESCondition(dynamicClient, ctx, name, namespace, "MetricsPipeline")
		if err != nil {
			return false
		}
		if !found {
			return true
		}
		if status == "True" && reason == MetricsReasonCollectionSucceeded {
			return false
		}
		failReasons := []string{"ValidationFailed", "BuildMetricsContextFailed", "MetricsCollectionFailed"}
		for _, r := range failReasons {
			if reason == r {
				return true
			}
		}
		return status == "False"
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(),
		"ES %s/%s MetricsPipeline 不应为 MetricsCollectionSucceeded", namespace, name)
}

// BrokenExternalServerEndpointURL 返回故意失败的 ExternalServer HTTP 端点（用于采集失败 E2E，非 External/Custom mock）。
func BrokenExternalServerEndpointURL(namespace string) string {
	return fmt.Sprintf("%s.%s.svc.cluster.local:%d/broken", BrokenExternalServerSvcName, namespace, BrokenExternalServerPort)
}

// OKPrometheusExternalServerEndpointURL 返回 ES-E2E-006 成功路径 ExternalServer 端点（Prometheus 标准 vector 响应）。
func OKPrometheusExternalServerEndpointURL(namespace string) string {
	return fmt.Sprintf("%s.%s.svc.cluster.local:%d%s", OKPrometheusExternalServerSvcName, namespace, OKPrometheusExternalServerPort, DefaultPrometheusQueryPath)
}

// E2E006PartialSuccessExternalServerMetrics 返回 ES-E2E-006 双 ExternalServer 指标（成功 Prometheus + 失败非 Prometheus body）。
func E2E006PartialSuccessExternalServerMetrics(namespace string) []ESApaMetricEntry {
	return []ESApaMetricEntry{
		{
			Type:                   "ExternalServer",
			ExternalServerName:     MetricsExternalServerInferQPS,
			ExternalServerTarget:   "100",
			ExternalServerEndpoint: OKPrometheusExternalServerEndpointURL(namespace),
			ExternalServerProtocol: "http",
			ExternalServerQuery:    "query=vector(42)",
		},
		{
			Type:                   "ExternalServer",
			ExternalServerName:     MetricsExternalServerBrokenLatency,
			ExternalServerTarget:   "100",
			ExternalServerEndpoint: BrokenExternalServerEndpointURL(namespace),
			ExternalServerProtocol: "http",
		},
	}
}

// E2E007AllCollectFailureExternalServerMetric 返回 ES-E2E-007 单条失败 ExternalServer 指标（非 Prometheus body）。
func E2E007AllCollectFailureExternalServerMetric(namespace string) []ESApaMetricEntry {
	return []ESApaMetricEntry{
		{
			Type:                   "ExternalServer",
			ExternalServerName:     MetricsExternalServerBrokenLatency,
			ExternalServerTarget:   "100",
			ExternalServerEndpoint: BrokenExternalServerEndpointURL(namespace),
			ExternalServerProtocol: "http",
		},
	}
}

// DeployOKPrometheusExternalServerEndpoint 部署返回 Prometheus 标准 vector 成功响应的 HTTP 端点（ExternalServer 成功路径）。
func DeployOKPrometheusExternalServerEndpoint(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMockHTTPServer(clientset, ctx, namespace, mockHTTPServerSpec{
		deployName: OKPrometheusExternalServerDeployName,
		svcName:    OKPrometheusExternalServerSvcName,
		port:       OKPrometheusExternalServerPort,
		configName: OKPrometheusExternalServerDeployName + "-cfg",
		nginxConf: `server {
    listen 8080;
    location /api/v1/query {
        default_type application/json;
        return 200 '{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1710000000,"42"]}]}}';
    }
}`,
	})
}

// WaitForOKPrometheusExternalServerReady 等待 ExternalServer 成功端点 Deployment Ready。
func WaitForOKPrometheusExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	WaitForDeploymentReplicas(clientset, ctx, OKPrometheusExternalServerDeployName, namespace, 1, 3*time.Minute)
}

// DeployBrokenExternalServerEndpoint 部署返回非 Prometheus 响应的 HTTP 端点，用于 ExternalServer 采集失败场景。
func DeployBrokenExternalServerEndpoint(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMockHTTPServer(clientset, ctx, namespace, mockHTTPServerSpec{
		deployName: BrokenExternalServerDeployName,
		svcName:    BrokenExternalServerSvcName,
		port:       BrokenExternalServerPort,
		configName: BrokenExternalServerDeployName + "-cfg",
		nginxConf: `server {
    listen 8080;
    location /broken {
        default_type application/json;
        return 200 '{"value": "not-prometheus"}';
    }
    location /api/v1/query {
        default_type application/json;
        return 200 '{"status":"error","errorType":"bad_data","error":"mock error"}';
    }
}`,
	})
}

type mockHTTPServerSpec struct {
	deployName string
	svcName    string
	port       int32
	configName string
	nginxConf  string
}

func deployMockHTTPServer(clientset *kubernetes.Clientset, ctx context.Context, namespace string, spec mockHTTPServerSpec) error {
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: spec.configName, Namespace: namespace},
		Data: map[string]string{
			"default.conf": spec.nginxConf,
		},
	}
	if _, err := clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	replicas := int32(1)
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: spec.deployName, Namespace: namespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": spec.deployName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": spec.deployName}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nginx",
						Image: "nginx:alpine",
						Ports: []corev1.ContainerPort{{ContainerPort: spec.port}},
						VolumeMounts: []corev1.VolumeMount{{
							Name:      "nginx-conf",
							MountPath: "/etc/nginx/conf.d",
						}},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("10m"),
								corev1.ResourceMemory: resource.MustParse("32Mi"),
							},
						},
					}},
					Volumes: []corev1.Volume{{
						Name: "nginx-conf",
						VolumeSource: corev1.VolumeSource{
							ConfigMap: &corev1.ConfigMapVolumeSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: spec.configName},
							},
						},
					}},
				},
			},
		},
	}
	if _, err := clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: spec.svcName, Namespace: namespace},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": spec.deployName},
			Ports: []corev1.ServicePort{{
				Port:       spec.port,
				TargetPort: intstr.FromInt32(spec.port),
			}},
		},
	}
	if _, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	WaitForDeploymentReplicas(clientset, ctx, spec.deployName, namespace, 1, 3*time.Minute)
	return nil
}

// WaitForBrokenExternalServerReady 等待 ExternalServer 失败端点 Deployment Ready。
func WaitForBrokenExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	WaitForDeploymentReplicas(clientset, ctx, BrokenExternalServerDeployName, namespace, 1, 3*time.Minute)
}

func mockHTTPServerDeploymentReady(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) bool {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil {
		return false
	}
	return dep.Status.ReadyReplicas >= 1
}

// EnsureOKPrometheusExternalServerReady 幂等部署并等待 ExternalServer 成功端点（ES-E2E-006 创建，007 复用）。
func EnsureOKPrometheusExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	if mockHTTPServerDeploymentReady(clientset, ctx, namespace, OKPrometheusExternalServerDeployName) {
		return
	}
	Expect(DeployOKPrometheusExternalServerEndpoint(clientset, ctx, namespace)).NotTo(HaveOccurred())
	WaitForOKPrometheusExternalServerReady(clientset, ctx, namespace)
}

// EnsureBrokenExternalServerReady 幂等部署并等待 ExternalServer 失败端点（ES-E2E-006 创建，007/013 复用）。
func EnsureBrokenExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	if mockHTTPServerDeploymentReady(clientset, ctx, namespace, BrokenExternalServerDeployName) {
		return
	}
	Expect(DeployBrokenExternalServerEndpoint(clientset, ctx, namespace)).NotTo(HaveOccurred())
	WaitForBrokenExternalServerReady(clientset, ctx, namespace)
}

// esMetricsTestDeploymentWaitTimeout 单个镜像等待 Deployment 就绪的时长（镜像冷拉时要留足）。
const esMetricsTestDeploymentWaitTimeout = 3 * time.Minute

// EnsureESMetricsTestDeploymentReady 幂等准备 metrics 框架 targetRef Deployment（已 Ready 则跳过重建，单跑用例时可复用）。
//
// 带兜底重试：某个镜像在环境里拉不到（离线/单节点流水线常见），或上一次运行残留了不可用的 Deployment 时，
// 直接等待必然超时。这里按 esMetricsTestContainerImages 依次尝试，每次先删掉旧 Deployment 再重建；
// 全部失败时打印 Pod 状态、容器等待原因与最近事件再 Fail，便于从流水线日志直接判断是拉镜像、调度还是资源问题。
func EnsureESMetricsTestDeploymentReady(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32, highCPU, waitPodMetrics bool) {
	for _, image := range esMetricsTestContainerImages {
		ready, exists := esMetricsTestDeploymentReady(clientset, ctx, name, namespace, replicas)
		if ready {
			if waitPodMetrics {
				WaitForDeploymentPodMetricsReady(clientset, ctx, namespace, name, MetricsPodMetricsWaitTimeout())
			}
			return
		}
		if exists {
			GinkgoWriter.Printf("Deployment %s/%s 已存在但未达到 %d 个就绪副本，删除后重建（镜像 %s）\n",
				namespace, name, replicas, image)
			Expect(deleteDeploymentAndWaitGone(clientset, ctx, name, namespace)).NotTo(HaveOccurred())
		}
		Expect(createESMetricsTestDeploymentWithImage(clientset, ctx, name, namespace, replicas, highCPU, image)).NotTo(HaveOccurred())
		if waitForESMetricsTestDeploymentReady(clientset, ctx, name, namespace, replicas, esMetricsTestDeploymentWaitTimeout) {
			if waitPodMetrics {
				WaitForDeploymentPodMetricsReady(clientset, ctx, namespace, name, MetricsPodMetricsWaitTimeout())
			}
			return
		}
		GinkgoWriter.Printf("Deployment %s/%s 使用镜像 %s 在 %s 内未就绪，诊断信息：\n%s\n",
			namespace, name, image, esMetricsTestDeploymentWaitTimeout, describeESMetricsTestDeployment(clientset, ctx, name, namespace))
	}
	Fail(fmt.Sprintf("Deployment %s/%s 依次尝试镜像 %v 后仍未就绪，诊断信息：\n%s",
		namespace, name, esMetricsTestContainerImages, describeESMetricsTestDeployment(clientset, ctx, name, namespace)))
}

// esMetricsTestDeploymentReady 返回 Deployment 是否已达期望就绪副本数，以及该 Deployment 是否存在。
func esMetricsTestDeploymentReady(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32) (ready bool, exists bool) {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return false, false
	}
	return dep.Status.ReadyReplicas >= replicas, true
}

// waitForESMetricsTestDeploymentReady 轮询等待副本就绪；超时返回 false（由调用方换兜底镜像重试，避免直接失败）。
func waitForESMetricsTestDeploymentReady(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string, replicas int32, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for {
		if ready, _ := esMetricsTestDeploymentReady(clientset, ctx, name, namespace, replicas); ready {
			return true
		}
		if !time.Now().Before(deadline) {
			return false
		}
		time.Sleep(10 * time.Second)
	}
}

func deleteDeploymentAndWaitGone(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) error {
	propagation := metav1.DeletePropagationBackground
	if err := clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{PropagationPolicy: &propagation}); err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	deadline := time.Now().Add(30 * time.Second)
	for time.Now().Before(deadline) {
		if _, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{}); apierrors.IsNotFound(err) {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return nil
}

// describeESMetricsTestDeployment 汇总 Deployment 与其 Pod 的状态、容器等待原因和最近事件，用于失败时定位原因。
func describeESMetricsTestDeployment(clientset *kubernetes.Clientset, ctx context.Context, name, namespace string) string {
	var b strings.Builder
	if dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{}); err == nil {
		fmt.Fprintf(&b, "  deployment %s/%s replicas=%d ready=%d available=%d\n",
			namespace, name, dep.Status.Replicas, dep.Status.ReadyReplicas, dep.Status.AvailableReplicas)
		for _, c := range dep.Status.Conditions {
			fmt.Fprintf(&b, "    condition %s=%s reason=%s message=%s\n", c.Type, c.Status, c.Reason, c.Message)
		}
	} else {
		fmt.Fprintf(&b, "  deployment %s/%s get failed: %v\n", namespace, name, err)
	}
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: "app=" + name})
	if err != nil {
		fmt.Fprintf(&b, "  pods list failed: %v\n", err)
		return b.String()
	}
	for i := range pods.Items {
		pod := pods.Items[i]
		restarts := int32(0)
		for _, cs := range pod.Status.ContainerStatuses {
			restarts += cs.RestartCount
		}
		fmt.Fprintf(&b, "  pod %s phase=%s node=%s restarts=%d\n", pod.Name, pod.Status.Phase, pod.Spec.NodeName, restarts)
		for _, cs := range pod.Status.ContainerStatuses {
			switch {
			case cs.State.Waiting != nil:
				fmt.Fprintf(&b, "    container %s waiting reason=%s message=%s\n", cs.Name, cs.State.Waiting.Reason, cs.State.Waiting.Message)
			case cs.State.Terminated != nil:
				fmt.Fprintf(&b, "    container %s terminated reason=%s exitCode=%d\n", cs.Name, cs.State.Terminated.Reason, cs.State.Terminated.ExitCode)
			default:
				fmt.Fprintf(&b, "    container %s running=%v ready=%v\n", cs.Name, cs.State.Running != nil, cs.Ready)
			}
		}
		for _, cond := range pod.Status.Conditions {
			if cond.Status == corev1.ConditionTrue {
				continue
			}
			fmt.Fprintf(&b, "    pod condition %s=%s reason=%s message=%s\n", cond.Type, cond.Status, cond.Reason, cond.Message)
		}
	}
	if events, err := clientset.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{}); err == nil {
		shown := 0
		for i := len(events.Items) - 1; i >= 0 && shown < 8; i-- {
			e := events.Items[i]
			if e.InvolvedObject.Kind != "Pod" || !strings.HasPrefix(e.InvolvedObject.Name, name+"-") {
				continue
			}
			fmt.Fprintf(&b, "  event %s %s: %s\n", e.LastTimestamp.Format(time.RFC3339), e.Reason, e.Message)
			shown++
		}
	}
	return b.String()
}

// CleanupMetricsFrameworkTest 清理通用扩缩容决策框架 E2E 资源（含 ES-E2E-016+ 安装的 RSG controller 与 elastic-scaler helm release）。
func CleanupMetricsFrameworkTest(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("清理通用扩缩容决策框架测试资源...")
	CleanupMetricsFrameworkTestResources(dynamicClient, clientset, ctx)
	GinkgoWriter.Printf("helm uninstall %s/%s（RSG controller）...\n", RSGHelmReleaseName, RSGTestNamespace)
	Expect(HelmUninstall(RSGHelmReleaseName, RSGTestNamespace)).NotTo(HaveOccurred())
	GinkgoWriter.Printf("helm uninstall %s/%s（elastic-scaler controller）...\n", ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)
	Expect(HelmUninstall(ElasticScalerHelmReleaseName, ElasticScalerHelmNamespace)).NotTo(HaveOccurred())
	GinkgoWriter.Println("通用扩缩容决策框架测试资源与 helm controller 清理完成")
}

// PrepareMetricsFrameworkTest 套件开始前清理上次失败/中断遗留的 CR 与工作负载（不卸载 helm）。
func PrepareMetricsFrameworkTest(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("清理 metrics-framework 遗留测试资源...")
	CleanupMetricsFrameworkTestResources(dynamicClient, clientset, ctx)
}

func metricsFrameworkDeletePropagation() *metav1.DeletionPropagation {
	propagation := metav1.DeletePropagationBackground
	return &propagation
}

func deleteMetricsFrameworkESCRs(dynamicClient dynamic.Interface, ctx context.Context, namespace string, esNames ...string) {
	for _, name := range esNames {
		_ = DeleteElasticScalerCR(dynamicClient, ctx, name, namespace)
	}
}

func deleteMetricsFrameworkESCRange(dynamicClient dynamic.Interface, ctx context.Context, namespace string, from, to int) {
	for i := from; i <= to; i++ {
		name := fmt.Sprintf("%s%03d", MetricsESNamePrefix, i)
		_ = DeleteElasticScalerCR(dynamicClient, ctx, name, namespace)
	}
}

func deleteMetricsFrameworkDeployment(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	if name == "" {
		return
	}
	_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{
		PropagationPolicy: metricsFrameworkDeletePropagation(),
	})
}

func deleteMetricsFrameworkDeployments(clientset *kubernetes.Clientset, ctx context.Context, namespace string, names ...string) {
	for _, name := range names {
		deleteMetricsFrameworkDeployment(clientset, ctx, namespace, name)
	}
}

func deleteMetricsFrameworkJob(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	if name == "" {
		return
	}
	_ = clientset.BatchV1().Jobs(namespace).Delete(ctx, name, metav1.DeleteOptions{
		PropagationPolicy: metricsFrameworkDeletePropagation(),
	})
}

func deleteMetricsFrameworkService(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	if name == "" {
		return
	}
	_ = clientset.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

func deleteMetricsFrameworkConfigMap(clientset *kubernetes.Clientset, ctx context.Context, namespace, name string) {
	if name == "" {
		return
	}
	_ = clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

func cleanupMetricsFrameworkInferenceService(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName, svcName, probeJob, loadJob string) {
	deleteMetricsFrameworkJob(clientset, ctx, namespace, loadJob)
	deleteMetricsFrameworkJob(clientset, ctx, namespace, probeJob)
	deleteMetricsFrameworkDeployment(clientset, ctx, namespace, deployName)
	deleteMetricsFrameworkService(clientset, ctx, namespace, svcName)
	if deployName != "" {
		deleteMetricsFrameworkConfigMap(clientset, ctx, namespace, deployName+"-cfg")
	}
}

func cleanupMetricsFrameworkExternalServerEndpoints(clientset *kubernetes.Clientset, ctx context.Context, namespace string, includeOK, includeBroken, includeAudit bool) {
	if includeOK {
		deleteMetricsFrameworkDeployment(clientset, ctx, namespace, OKPrometheusExternalServerDeployName)
		deleteMetricsFrameworkService(clientset, ctx, namespace, OKPrometheusExternalServerSvcName)
		deleteMetricsFrameworkConfigMap(clientset, ctx, namespace, OKPrometheusExternalServerDeployName+"-cfg")
	}
	if includeBroken {
		deleteMetricsFrameworkDeployment(clientset, ctx, namespace, BrokenExternalServerDeployName)
		deleteMetricsFrameworkService(clientset, ctx, namespace, BrokenExternalServerSvcName)
		deleteMetricsFrameworkConfigMap(clientset, ctx, namespace, BrokenExternalServerDeployName+"-cfg")
	}
	if includeAudit {
		deleteMetricsFrameworkDeployment(clientset, ctx, namespace, AuditPrometheusExternalServerDeployName)
		deleteMetricsFrameworkService(clientset, ctx, namespace, AuditPrometheusExternalServerSvcName)
		deleteMetricsFrameworkConfigMap(clientset, ctx, namespace, AuditPrometheusExternalServerDeployName+"-cfg")
	}
}

func cleanupMetricsFrameworkTestDeploy(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) {
	deleteMetricsFrameworkDeployment(clientset, ctx, namespace, deployName)
}

func waitForMetricsFrameworkDeploymentsRemoved(clientset *kubernetes.Clientset, ctx context.Context, namespace string, names []string, timeout time.Duration) {
	if len(names) == 0 {
		return
	}
	if timeout <= 0 {
		timeout = 3 * time.Minute
	}
	GinkgoWriter.Printf("等待 Deployment 删除完成: %v（timeout=%s）...\n", names, timeout)
	Eventually(func() bool {
		for _, name := range names {
			_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
			if err == nil {
				return false
			}
			if !apierrors.IsNotFound(err) {
				return false
			}
		}
		return true
	}, timeout, 5*time.Second).Should(BeTrue(), "Deployment 应已删除: %v", names)
}

func cleanupMetricsFrameworkPhaseResources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, label string, deployNames []string, cleanup func(namespace string)) {
	ns := MetricsTestNamespace
	GinkgoWriter.Printf("清理 metrics-framework %s 阶段资源...\n", label)
	cleanup(ns)
	waitForMetricsFrameworkDeploymentsRemoved(clientset, ctx, ns, deployNames, 3*time.Minute)
	GinkgoWriter.Printf("metrics-framework %s 阶段资源清理完成\n", label)
}

// CleanupMetricsFrameworkAfter001To003 ES-E2E-003 结束后释放 001~003 的 ES 与 targetRef Deployment。
func CleanupMetricsFrameworkAfter001To003(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-001~003", []string{MetricsTestDeployName}, func(ns string) {
		deleteMetricsFrameworkESCRange(dynamicClient, ctx, ns, 1, 3)
		cleanupMetricsFrameworkTestDeploy(clientset, ctx, ns, MetricsTestDeployName)
	})
}

// CleanupMetricsFrameworkAfter005 ES-E2E-005 结束后释放 005 的 ES 与高 CPU targetRef Deployment。
func CleanupMetricsFrameworkAfter005(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-005", []string{MetricsTestDeployName005}, func(ns string) {
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName005)
		cleanupMetricsFrameworkTestDeploy(clientset, ctx, ns, MetricsTestDeployName005)
	})
}

// CleanupMetricsFrameworkAfter006To007 ES-E2E-007 结束后释放 006~007 的 ES、targetRef 与 ExternalServer 双端点。
func CleanupMetricsFrameworkAfter006To007(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-006~007", []string{
		MetricsTestDeployName006,
		MetricsTestDeployName007,
		OKPrometheusExternalServerDeployName,
		BrokenExternalServerDeployName,
	}, func(ns string) {
		deleteMetricsFrameworkESCRange(dynamicClient, ctx, ns, 6, 7)
		cleanupMetricsFrameworkTestDeploy(clientset, ctx, ns, MetricsTestDeployName006)
		cleanupMetricsFrameworkTestDeploy(clientset, ctx, ns, MetricsTestDeployName007)
		cleanupMetricsFrameworkExternalServerEndpoints(clientset, ctx, ns, true, true, false)
	})
}

// CleanupMetricsFrameworkAfter008To009 ES-E2E-009 结束后释放 008~009 的 ES 与推理 workload（009 无独立 CR）。
func CleanupMetricsFrameworkAfter008To009(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-008~009", []string{MetricsInferenceDeployName}, func(ns string) {
		StopMetricsInferenceLoadTest(clientset, ctx, MetricsTestNamespace)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, MetricsTestNamespace, MetricsESName008)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, MetricsTestNamespace,
			MetricsInferenceDeployName, MetricsInferenceSvcName,
			MetricsInferenceProbeJobName, MetricsInferenceLoadTestJobName)
	})
}

// CleanupMetricsFrameworkPhase001To009Resources 释放 ES-E2E-001~009 全部遗留资源（010 入口兜底；009 DeferCleanup 已及时清理时可幂等）。
func CleanupMetricsFrameworkPhase001To009Resources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	ns := MetricsTestNamespace
	GinkgoWriter.Println("清理 metrics-framework ES-E2E-001~009 阶段资源（为 010+ 释放集群负担）...")
	StopMetricsInferenceLoadTest(clientset, ctx, ns)
	deleteMetricsFrameworkESCRange(dynamicClient, ctx, ns, 1, 9)
	deleteMetricsFrameworkDeployments(clientset, ctx, ns,
		MetricsTestDeployName,
		MetricsTestDeployName005,
		MetricsTestDeployName006,
		MetricsTestDeployName007,
		MetricsInferenceDeployName,
	)
	cleanupMetricsFrameworkExternalServerEndpoints(clientset, ctx, ns, true, true, false)
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName, MetricsInferenceSvcName,
		MetricsInferenceProbeJobName, MetricsInferenceLoadTestJobName)
	waitForMetricsFrameworkDeploymentsRemoved(clientset, ctx, ns, []string{
		MetricsTestDeployName,
		MetricsTestDeployName005,
		MetricsTestDeployName006,
		MetricsTestDeployName007,
		MetricsInferenceDeployName,
		OKPrometheusExternalServerDeployName,
		BrokenExternalServerDeployName,
	}, 3*time.Minute)
	GinkgoWriter.Println("metrics-framework ES-E2E-001~009 阶段资源清理完成")
}

// CleanupMetricsFrameworkAfter010 ES-E2E-010 结束后释放本用例 ES 与推理 workload。
func CleanupMetricsFrameworkAfter010(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-010", []string{MetricsInferenceDeployName010}, func(ns string) {
		StopMetricsInferenceLoadTest010(clientset, ctx, ns)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName010)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
			MetricsInferenceDeployName010, MetricsInferenceSvcName010,
			MetricsInferenceProbeJobName010, MetricsInferenceLoadTestJobName010)
	})
}

// CleanupMetricsFrameworkAfter011 ES-E2E-011 结束后释放本用例 ES、推理 workload 与审计 Prometheus 端点。
func CleanupMetricsFrameworkAfter011(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-011", []string{
		MetricsInferenceDeployName011,
		AuditPrometheusExternalServerDeployName,
	}, func(ns string) {
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName011)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
			MetricsInferenceDeployName011, MetricsInferenceSvcName011,
			MetricsInferenceProbeJobName011, "")
		cleanupMetricsFrameworkExternalServerEndpoints(clientset, ctx, ns, false, false, true)
	})
}

// CleanupMetricsFrameworkAfter012 ES-E2E-012 结束后释放本用例 ES 与推理 workload。
func CleanupMetricsFrameworkAfter012(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-012", []string{MetricsInferenceDeployName012}, func(ns string) {
		StopMetricsInferenceLoadTest012(clientset, ctx, ns)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName012)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
			MetricsInferenceDeployName012, MetricsInferenceSvcName012,
			MetricsInferenceProbeJobName012, MetricsInferenceLoadTestJobName012)
	})
}

// CleanupMetricsFrameworkAfter013 ES-E2E-013 结束后释放本用例 ES、推理 workload 与 broken ExternalServer 端点。
func CleanupMetricsFrameworkAfter013(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-013", []string{
		MetricsInferenceDeployName013,
		BrokenExternalServerDeployName,
	}, func(ns string) {
		StopMetricsInferenceLoadTest013(clientset, ctx, ns)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName013)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
			MetricsInferenceDeployName013, MetricsInferenceSvcName013,
			MetricsInferenceProbeJobName013, MetricsInferenceLoadTestJobName013)
		cleanupMetricsFrameworkExternalServerEndpoints(clientset, ctx, ns, false, true, false)
	})
}

// CleanupMetricsFrameworkAfter014 ES-E2E-014 结束后释放本用例 ES 与推理 workload。
func CleanupMetricsFrameworkAfter014(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-014", []string{MetricsInferenceDeployName014}, func(ns string) {
		StopMetricsInferenceLoadTest014(clientset, ctx, ns)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName014)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
			MetricsInferenceDeployName014, MetricsInferenceSvcName014,
			MetricsInferenceProbeJobName014, MetricsInferenceLoadTestJobName014)
	})
}

// CleanupMetricsFrameworkAfter015 ES-E2E-015 结束后释放本用例 ES、推理 workload 与 mock custom.metrics APIService。
func CleanupMetricsFrameworkAfter015(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	ns := MetricsTestNamespace
	cleanupMetricsFrameworkPhaseResources(dynamicClient, clientset, ctx, "ES-E2E-015", []string{MetricsInferenceDeployName015}, func(namespace string) {
		StopMetricsInferenceLoadTest015(clientset, ctx, namespace)
		deleteMetricsFrameworkESCRs(dynamicClient, ctx, namespace, MetricsESName015)
		cleanupMetricsFrameworkInferenceService(clientset, ctx, namespace,
			MetricsInferenceDeployName015, MetricsInferenceSvcName015,
			MetricsInferenceProbeJobName015, MetricsInferenceLoadTestJobName015)
	})
	CleanupMockCustomMetricsAPI(dynamicClient, clientset, ctx, ns)
}

// CleanupMetricsFrameworkPhase010To015Resources 016 入口兜底：释放 010~015 全部遗留 workload（不影响 016+ RSG）。
func CleanupMetricsFrameworkPhase010To015Resources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	ns := MetricsTestNamespace
	GinkgoWriter.Println("清理 metrics-framework ES-E2E-010~015 阶段资源（为 RSG 用例释放集群负担）...")
	deleteMetricsFrameworkESCRange(dynamicClient, ctx, ns, 10, 15)
	for _, stop := range []func(*kubernetes.Clientset, context.Context, string){
		StopMetricsInferenceLoadTest010,
		StopMetricsInferenceLoadTest012,
		StopMetricsInferenceLoadTest013,
		StopMetricsInferenceLoadTest014,
		StopMetricsInferenceLoadTest015,
	} {
		stop(clientset, ctx, ns)
	}
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName010, MetricsInferenceSvcName010,
		MetricsInferenceProbeJobName010, MetricsInferenceLoadTestJobName010)
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName011, MetricsInferenceSvcName011,
		MetricsInferenceProbeJobName011, "")
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName012, MetricsInferenceSvcName012,
		MetricsInferenceProbeJobName012, MetricsInferenceLoadTestJobName012)
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName013, MetricsInferenceSvcName013,
		MetricsInferenceProbeJobName013, MetricsInferenceLoadTestJobName013)
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName014, MetricsInferenceSvcName014,
		MetricsInferenceProbeJobName014, MetricsInferenceLoadTestJobName014)
	cleanupMetricsFrameworkInferenceService(clientset, ctx, ns,
		MetricsInferenceDeployName015, MetricsInferenceSvcName015,
		MetricsInferenceProbeJobName015, MetricsInferenceLoadTestJobName015)
	cleanupMetricsFrameworkExternalServerEndpoints(clientset, ctx, ns, false, true, true)
	CleanupMockCustomMetricsAPI(dynamicClient, clientset, ctx, ns)
	waitForMetricsFrameworkDeploymentsRemoved(clientset, ctx, ns, []string{
		MetricsInferenceDeployName010,
		MetricsInferenceDeployName011,
		MetricsInferenceDeployName012,
		MetricsInferenceDeployName013,
		MetricsInferenceDeployName014,
		MetricsInferenceDeployName015,
		AuditPrometheusExternalServerDeployName,
		BrokenExternalServerDeployName,
	}, 3*time.Minute)
	GinkgoWriter.Println("metrics-framework ES-E2E-010~015 阶段资源清理完成")
}

// CleanupMetricsFrameworkAfter019 ES-E2E-019 结束后释放本用例 ES 与独立 RSG workload（保留 016 RSG/ES 供 017/018/020/021）。
func CleanupMetricsFrameworkAfter019(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	ns := MetricsTestNamespace
	GinkgoWriter.Println("清理 metrics-framework ES-E2E-019 阶段资源...")
	deleteMetricsFrameworkESCRs(dynamicClient, ctx, ns, MetricsESName019)
	CleanupMetricsRSGResourceTestResources(dynamicClient, clientset, ctx, ns)
	GinkgoWriter.Println("metrics-framework ES-E2E-019 阶段资源清理完成")
}

// CleanupMetricsFrameworkTestResources 删除 metrics-framework 用例创建的 CR/Deployment/Job 等。
func CleanupMetricsFrameworkTestResources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context) {
	ns := MetricsTestNamespace
	for i := 1; i <= 21; i++ {
		name := fmt.Sprintf("%s%03d", MetricsESNamePrefix, i)
		_ = DeleteElasticScalerCR(dynamicClient, ctx, name, ns)
	}
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsTestDeployName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsTestDeployName005, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsTestDeployName006, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsTestDeployName007, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName010, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName011, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName012, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName013, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName014, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName015, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, MetricsInferenceDeployName017, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName010, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName011, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName012, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName013, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName014, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName015, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, MetricsInferenceSvcName017, metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName010+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName011+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName012+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName013+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName014+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName015+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, MetricsInferenceDeployName017+"-cfg", metav1.DeleteOptions{})
	propagation := metav1.DeletePropagationBackground
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName010, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName010, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName011, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName012, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName012, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName013, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName013, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName014, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName014, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName015, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName015, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceProbeJobName017, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsInferenceLoadTestJobName017, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.BatchV1().Jobs(ns).Delete(ctx, MetricsFrameworkRSGLoadTestJobName, metav1.DeleteOptions{PropagationPolicy: &propagation})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, BrokenExternalServerDeployName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, OKPrometheusExternalServerDeployName, metav1.DeleteOptions{})
	_ = clientset.AppsV1().Deployments(ns).Delete(ctx, AuditPrometheusExternalServerDeployName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, BrokenExternalServerSvcName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, OKPrometheusExternalServerSvcName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(ns).Delete(ctx, AuditPrometheusExternalServerSvcName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, BrokenExternalServerDeployName+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, OKPrometheusExternalServerDeployName+"-cfg", metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(ns).Delete(ctx, AuditPrometheusExternalServerDeployName+"-cfg", metav1.DeleteOptions{})
	CleanupMockCustomMetricsAPI(dynamicClient, clientset, ctx, ns)
	CleanupMetricsFrameworkRSGResources(dynamicClient, clientset, ctx, ns)
	CleanupMetricsRSGResourceTestResources(dynamicClient, clientset, ctx, ns)
	time.Sleep(2 * time.Second)
}

// --- Audit Prometheus ExternalServer（query 拼接审计）---
// AuditPrometheusEndpointWithFixedParams 返回 ES-E2E-011 endpoint（含 tenant/namespace 固定查询参数，不含 PromQL）。
func AuditPrometheusEndpointWithFixedParams(k8sNamespace string) string {
	return fmt.Sprintf("%s.%s.svc.cluster.local:%d%s?tenant=%s&namespace=%s",
		AuditPrometheusExternalServerSvcName, k8sNamespace, AuditPrometheusExternalServerPort,
		DefaultPrometheusQueryPath, AuditPrometheusEndpointTenant, AuditPrometheusEndpointNamespaceParam)
}

// AuditPrometheusExternalServerQueryParam 返回 spec.externalServer.query（PromQL + step，与 endpoint 固定参数分离）。
func AuditPrometheusExternalServerQueryParam() string {
	promQL := PrometheusInferQPSQuery()
	return "query=" + url.QueryEscape(promQL) + "&step=" + AuditPrometheusQueryStep
}

// AuditPrometheusExternalServerMetric 构造 ES-E2E-011 ExternalServer 指标（HTTP 审计 Prometheus 端点，验证 endpoint 固定参数与 query 拼接）。
func AuditPrometheusExternalServerMetric(k8sNamespace string) ESApaMetricEntry {
	return ESApaMetricEntry{
		Type:                   "ExternalServer",
		ExternalServerName:     MetricsExternalServerInferQPS,
		ExternalServerTarget:   MetricsE2E008InferQPSTarget,
		ExternalServerEndpoint: AuditPrometheusEndpointWithFixedParams(k8sNamespace),
		ExternalServerProtocol: "http",
		ExternalServerQuery:    AuditPrometheusExternalServerQueryParam(),
	}
}

// DeployAuditPrometheusExternalServer 部署将 access log 输出到 stdout 的 Prometheus 查询审计端点。
func DeployAuditPrometheusExternalServer(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMockHTTPServer(clientset, ctx, namespace, mockHTTPServerSpec{
		deployName: AuditPrometheusExternalServerDeployName,
		svcName:    AuditPrometheusExternalServerSvcName,
		port:       AuditPrometheusExternalServerPort,
		configName: AuditPrometheusExternalServerDeployName + "-cfg",
		nginxConf: `server {
    listen 8080;
    access_log /dev/stdout combined;
    error_log /dev/stderr warn;
    location /api/v1/query {
        default_type application/json;
        return 200 '{"status":"success","data":{"resultType":"vector","result":[{"metric":{},"value":[1710000000,"150"]}]}}';
    }
}`,
	})
}

// WaitForAuditPrometheusExternalServerReady 等待审计 Prometheus 端点 Deployment Ready。
func WaitForAuditPrometheusExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	WaitForDeploymentReplicas(clientset, ctx, AuditPrometheusExternalServerDeployName, namespace, 1, 3*time.Minute)
}

// EnsureAuditPrometheusExternalServerReady 幂等部署并等待 ES-E2E-011 审计 Prometheus 端点就绪。
func EnsureAuditPrometheusExternalServerReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, AuditPrometheusExternalServerDeployName, metav1.GetOptions{})
	if err == nil && dep.Status.ReadyReplicas >= 1 {
		WaitForServiceEndpointsReady(clientset, ctx, namespace, AuditPrometheusExternalServerSvcName, 2*time.Minute)
		return
	}
	Expect(DeployAuditPrometheusExternalServer(clientset, ctx, namespace)).NotTo(HaveOccurred())
	WaitForAuditPrometheusExternalServerReady(clientset, ctx, namespace)
}

// VerifyExternalServerQueryAppendInAuditLogs 从审计端点 access log 确认 controller GET URL 同时含 endpoint 固定参数与 spec.query。
func VerifyExternalServerQueryAppendInAuditLogs(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	promQL := PrometheusInferQPSQuery()
	endpointParams := map[string]string{
		"tenant":    AuditPrometheusEndpointTenant,
		"namespace": AuditPrometheusEndpointNamespaceParam,
	}
	queryParams := map[string]string{
		"query": promQL,
		"step":  AuditPrometheusQueryStep,
	}

	GinkgoWriter.Println("从审计 Prometheus access log 验证 ExternalServer URL 拼接...")
	Eventually(func() bool {
		logs, err := fetchAuditPrometheusAccessLogs(clientset, ctx, namespace)
		if err != nil || logs == "" {
			return false
		}
		queryLine := findPrometheusQueryGETLine(logs)
		if queryLine == "" {
			GinkgoWriter.Println("access log 尚未出现 GET /api/v1/query 请求，继续等待...")
			return false
		}
		GinkgoWriter.Printf("审计端点捕获请求: %s\n", queryLine)
		if !strings.Contains(queryLine, "GET ") || !strings.Contains(queryLine, DefaultPrometheusQueryPath) {
			GinkgoWriter.Printf("access log 行不符合 Prometheus GET 形态: %s\n", queryLine)
			return false
		}
		for k, v := range endpointParams {
			if !auditLogContainsParam(queryLine, k, v) {
				GinkgoWriter.Printf("access log 缺少 endpoint 固定参数 %s=%s\n", k, v)
				return false
			}
		}
		for k, v := range queryParams {
			if !auditLogContainsParam(queryLine, k, v) {
				GinkgoWriter.Printf("access log 缺少 spec.query 参数 %s=%s\n", k, v)
				return false
			}
		}
		return true
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"审计 Prometheus access log 应包含 endpoint 固定参数与 spec.query 中的 PromQL/step")
}

func fetchAuditPrometheusAccessLogs(clientset *kubernetes.Clientset, ctx context.Context, namespace string) (string, error) {
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app=" + AuditPrometheusExternalServerDeployName,
	})
	if err != nil {
		return "", err
	}
	var b strings.Builder
	tail := int64(300)
	for _, pod := range pods.Items {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}
		raw, err := clientset.CoreV1().Pods(namespace).GetLogs(pod.Name, &corev1.PodLogOptions{
			TailLines: &tail,
		}).DoRaw(ctx)
		if err != nil {
			continue
		}
		b.Write(raw)
	}
	return b.String(), nil
}

func findPrometheusQueryGETLine(logs string) string {
	for _, line := range strings.Split(logs, "\n") {
		line = strings.TrimSpace(line)
		if strings.Contains(line, "GET ") && strings.Contains(line, DefaultPrometheusQueryPath) {
			return line
		}
	}
	return ""
}

func auditLogContainsParam(line, key, value string) bool {
	if strings.Contains(line, key+"="+value) {
		return true
	}
	return strings.Contains(line, key+"="+url.QueryEscape(value))
}

// --- 双指标全量成功（Resource/cpu + infer_qps）---
// ApaResourceAndInferQPSMetrics 返回 ES-E2E-012 双指标：Resource/cpu + ExternalServer/infer_qps（集群 Prometheus）。
func ApaResourceAndInferQPSMetrics() []ESApaMetricEntry {
	return []ESApaMetricEntry{
		DefaultResourceCPUMetric(),
		E2E008InferQPSExternalServerMetric(),
	}
}

// WaitForESApaMultiMetricsFullSuccess ES-E2E-012：等待双指标全量采集成功且 apa 输出与 Deployment 对齐。
func WaitForESApaMultiMetricsFullSuccess(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, processedCount int, minReplicas, maxReplicas int32, recovery *ResourceMetricsWaitRecovery, timeout time.Duration) {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, processedCount, true, recovery)
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired < int64(minReplicas) || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 ∈ [%d,%d]\n", desired, minReplicas, maxReplicas)
			return false
		}
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, int32(desired))
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 双指标全量成功后 apa 输出应与 Deployment %s 对齐且副本 ∈ [%d,%d]",
		namespace, esName, deployName, minReplicas, maxReplicas)
}

// --- 部分成功 + apa 仍输出 ScalingCalculated ---
// PartialSuccessInferenceExternalServerMetrics 返回 ES-E2E-013 双 ExternalServer 指标（集群 Prometheus infer_qps 成功 + broken_latency 失败端点）。
func PartialSuccessInferenceExternalServerMetrics(namespace string) []ESApaMetricEntry {
	return []ESApaMetricEntry{
		E2E008InferQPSExternalServerMetric(),
		{
			Type:                   "ExternalServer",
			ExternalServerName:     MetricsExternalServerBrokenLatency,
			ExternalServerTarget:   "100",
			ExternalServerEndpoint: BrokenExternalServerEndpointURL(namespace),
			ExternalServerProtocol: "http",
		},
	}
}

// WaitForESApaPartialSuccessWithScaling ES-E2E-013：PartialSuccess 下成功指标仍进入 apa，ScalingCalculated 成功且 Deployment 对齐。
func WaitForESApaPartialSuccessWithScaling(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, failedMetricNames []string, minReplicas, maxReplicas int32, timeout time.Duration) {
	VerifyESGetMetricsPartialSuccess(dynamicClient, ctx, esName, namespace, failedMetricNames...)
	VerifyESMetricsPipelineNotCollectionFailed(dynamicClient, ctx, esName, namespace)

	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired < int64(minReplicas) || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 ∈ [%d,%d]\n", desired, minReplicas, maxReplicas)
			return false
		}
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, int32(desired))
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 部分成功时 apa 仍应输出 ScalingCalculated 且 Deployment %s 对齐", namespace, esName, deployName)
}

// VerifyESMetricsPipelineNotCollectionFailed 确认 MetricsPipeline 未落入全链路采集失败（PartialSuccess 场景不应为 MetricsCollectionFailed）。
func VerifyESMetricsPipelineNotCollectionFailed(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	detail := getESMetricsPipelineDetail(obj)
	Expect(detail.Found).To(BeTrue(), "ES %s/%s 应存在 MetricsPipeline 条件", namespace, name)
	Expect(detail.Reason).NotTo(Equal("MetricsCollectionFailed"),
		"ES %s/%s MetricsPipeline 不应为全链路采集失败: %s", namespace, name, detail.Message)
	Expect(detail.Reason).NotTo(Equal("BuildMetricsContextFailed"),
		"ES %s/%s MetricsPipeline 不应为 BuildMetricsContextFailed: %s", namespace, name, detail.Message)
}

// --- 窗口聚合 Max 策略 ---
// WindowMaxAggregationPolicyAnnotations 返回 ES-E2E-014 策略注解（windowSeconds=120、aggregationType=Max）。
func WindowMaxAggregationPolicyAnnotations() map[string]string {
	return map[string]string{
		AnnotationMetricsWindowSeconds:   "120",
		AnnotationMetricsAggregationType: "Max",
	}
}

// ExternalServerMetricWithPromQL 构造带指定 PromQL 的 ExternalServer/infer_qps 指标。
func ExternalServerMetricWithPromQL(promQL string) ESApaMetricEntry {
	m := E2E008InferQPSExternalServerMetric()
	m.ExternalServerQuery = "query=" + url.QueryEscape(promQL)
	return m
}

// VerifyESPhaseMetricsCollectionSuccess 单阶段校验 MetricsPipeline 全量采集成功。
func VerifyESPhaseMetricsCollectionSuccess(dynamicClient dynamic.Interface, ctx context.Context, esName, namespace, phaseLabel string) {
	GinkgoWriter.Printf("[%s] 验证 MetricsPipeline 全量采集成功...\n", phaseLabel)
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)
}

// WaitForESApaWindowMaxRetainedAfterLowTraffic ES-E2E-014 阶段三：低流量后 desiredReplicas 仍反映窗口 Max（> minReplicas）。
func WaitForESApaWindowMaxRetainedAfterLowTraffic(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, minReplicas int32, highPhaseDesired int32, timeout time.Duration) {
	VerifyESPhaseMetricsCollectionSuccess(dynamicClient, ctx, esName, namespace, "低流量阶段")

	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found {
			return false
		}
		if desired <= int64(minReplicas) {
			GinkgoWriter.Printf("低流量阶段 desiredReplicas=%d，期望仍 > min=%d（窗口 Max 应保留高流量样本 %d）\n",
				desired, minReplicas, highPhaseDesired)
			return false
		}
		if highPhaseDesired > minReplicas && desired < int64(highPhaseDesired) {
			GinkgoWriter.Printf("低流量阶段 desiredReplicas=%d，期望仍 >= 高流量阶段 %d\n", desired, highPhaseDesired)
			return false
		}
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, int32(desired))
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 低流量阶段 desiredReplicas 应仍反映窗口内 Max 指标（>= %d，> min=%d）",
		namespace, esName, highPhaseDesired, minReplicas)
}

// GetESDesiredReplicas 读取 ES status.desiredReplicas。
func GetESDesiredReplicas(dynamicClient dynamic.Interface, ctx context.Context, esName, namespace string) int32 {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
	Expect(found).To(BeTrue(), "ES %s/%s status.desiredReplicas 应存在", namespace, esName)
	return int32(desired)
}

// --- Custom 指标 + mock APIService ---
var customMetricsAPIVersions = []string{"v1beta2", "v1beta1"}

var customMetricsAPIServiceGVR = schema.GroupVersionResource{
	Group:    "apiregistration.k8s.io",
	Version:  "v1",
	Resource: "apiservices",
}

// CustomMetricsMetricCandidates 返回待探测的 Custom 指标名列表（METRICS_CUSTOM_METRIC_NAME 可覆盖为单一候选）。
func CustomMetricsMetricCandidates() []string {
	if name := strings.TrimSpace(os.Getenv("METRICS_CUSTOM_METRIC_NAME")); name != "" {
		return []string{name}
	}
	return []string{
		DefaultCustomMetricsMetricName,
		"inference_qps",
		"pods/inference_qps",
		"pods/http_requests",
	}
}

// CustomMetricsTargetValue 返回 Custom 指标 apa target（METRICS_CUSTOM_METRIC_TARGET 可覆盖）。
func CustomMetricsTargetValue() int64 {
	if raw := strings.TrimSpace(os.Getenv("METRICS_CUSTOM_METRIC_TARGET")); raw != "" {
		if v, err := strconv.ParseInt(raw, 10, 64); err == nil && v > 0 {
			return v
		}
	}
	return DefaultCustomMetricsTarget
}

// CustomMetricsMockReturnValue 返回 mock custom.metrics.k8s.io 指标值（高于 target 以触发 apa 扩容）。
func CustomMetricsMockReturnValue() int64 {
	if raw := strings.TrimSpace(os.Getenv("METRICS_CUSTOM_METRIC_MOCK_VALUE")); raw != "" {
		if v, err := strconv.ParseInt(raw, 10, 64); err == nil && v > 0 {
			return v
		}
	}
	target := CustomMetricsTargetValue()
	mockVal := target * 3
	if mockVal < 150 {
		return 150
	}
	return mockVal
}

// ApaCustomMetricEntry 构造 ES-E2E-015 Custom 指标条目。
func ApaCustomMetricEntry(metricName string) ESApaMetricEntry {
	return ESApaMetricEntry{
		Type:         "Custom",
		CustomName:   metricName,
		CustomTarget: CustomMetricsTargetValue(),
	}
}

// ClusterCustomMetricsAvailable 检查 custom.metrics.k8s.io API 是否已注册。
func ClusterCustomMetricsAvailable(clientset *kubernetes.Clientset, ctx context.Context) bool {
	for _, ver := range customMetricsAPIVersions {
		if _, err := clientset.Discovery().ServerResourcesForGroupVersion("custom.metrics.k8s.io/" + ver); err == nil {
			return true
		}
	}
	return false
}

// ResolveDeploymentCustomMetric 探测 Deployment 上可查询的 Custom 指标名（首个成功候选）。
func ResolveDeploymentCustomMetric(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) (metricName string, ok bool) {
	for _, candidate := range CustomMetricsMetricCandidates() {
		if _, err := getDeploymentCustomMetricValue(clientset, ctx, namespace, deployName, candidate); err == nil {
			GinkgoWriter.Printf("已发现 Deployment %s/%s Custom 指标: %s\n", namespace, deployName, candidate)
			return candidate, true
		}
	}
	return "", false
}

// EnsureCustomMetricsReadyForDeployment 确保 ES-E2E-015 可查询 Custom 指标：优先真实 adapter，否则部署 mock APIService。
func EnsureCustomMetricsReadyForDeployment(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace, deployName string) string {
	if ClusterCustomMetricsAvailable(clientset, ctx) {
		if name, ok := ResolveDeploymentCustomMetric(clientset, ctx, namespace, deployName); ok {
			GinkgoWriter.Println("使用集群已有 custom.metrics.k8s.io 与真实指标")
			return name
		}
		GinkgoWriter.Println("custom.metrics.k8s.io 已注册但无目标指标，尝试部署 E2E mock APIService...")
	} else {
		GinkgoWriter.Println("集群未注册 custom.metrics.k8s.io，部署 E2E mock APIService...")
	}

	Expect(DeployMockCustomMetricsAPI(clientset, dynamicClient, ctx, namespace)).To(Succeed())
	WaitForCustomMetricsAPIReady(clientset, dynamicClient, ctx)

	metricName := CustomMetricsMetricCandidates()[0]
	if envName := strings.TrimSpace(os.Getenv("METRICS_CUSTOM_METRIC_NAME")); envName != "" {
		metricName = envName
	}
	VerifyDeploymentCustomMetricQueryable(clientset, ctx, namespace, deployName, metricName)
	return metricName
}

func mockCustomMetricsOpenRestyConf(mockValue int64) string {
	valueStr := strconv.FormatInt(mockValue, 10)
	metricName := DefaultCustomMetricsMetricName
	return fmt.Sprintf(`
server {
    listen %d ssl;
    ssl_certificate /etc/nginx/certs/tls.crt;
    ssl_certificate_key /etc/nginx/certs/tls.key;
    default_type application/json;

    location = /healthz {
        default_type text/plain;
        return 200 'ok';
    }

    location = /apis/custom.metrics.k8s.io/v1beta2 {
        return 200 '{"kind":"APIResourceList","apiVersion":"v1","groupVersion":"custom.metrics.k8s.io/v1beta2","resources":[{"name":"pods","namespaced":true,"kind":"MetricValueList","verbs":["get","list"]},{"name":"deployments.apps","namespaced":true,"kind":"MetricValueList","verbs":["get"]},{"name":"apps.deployments","namespaced":true,"kind":"MetricValueList","verbs":["get"]}]}';
    }

    location ~ ^/apis/custom.metrics.k8s.io/v1beta2/namespaces/[^/]+/(deployments\.apps|apps\.deployments)/[^/]+/[^/]+/?$ {
        content_by_lua_block {
            local ts = os.date("!%%Y-%%m-%%dT%%H:%%M:%%SZ")
            ngx.header.content_type = "application/json"
            ngx.say(string.format('{"apiVersion":"custom.metrics.k8s.io/v1beta2","kind":"MetricValueList","items":[{"apiVersion":"custom.metrics.k8s.io/v1beta2","kind":"MetricValue","metricName":"%s","timestamp":"%%s","value":"%s"}]}', ts))
        }
    }

    location = /apis/custom.metrics.k8s.io/v1beta1 {
        return 200 '{"kind":"APIResourceList","apiVersion":"v1","groupVersion":"custom.metrics.k8s.io/v1beta1","resources":[{"name":"pods","namespaced":true,"kind":"MetricValueList","verbs":["get","list"]},{"name":"deployments.apps","namespaced":true,"kind":"MetricValueList","verbs":["get"]},{"name":"apps.deployments","namespaced":true,"kind":"MetricValueList","verbs":["get"]}]}';
    }

    location ~ ^/apis/custom.metrics.k8s.io/v1beta1/namespaces/[^/]+/(deployments\.apps|apps\.deployments)/[^/]+/[^/]+/?$ {
        content_by_lua_block {
            local ts = os.date("!%%Y-%%m-%%dT%%H:%%M:%%SZ")
            ngx.header.content_type = "application/json"
            ngx.say(string.format('{"apiVersion":"custom.metrics.k8s.io/v1beta1","kind":"MetricValueList","items":[{"apiVersion":"custom.metrics.k8s.io/v1beta1","kind":"MetricValue","metricName":"%s","timestamp":"%%s","value":"%s"}]}', ts))
        }
    }
}
`, MockCustomMetricsAPIServerPort, metricName, valueStr, metricName, valueStr)
}

func generateMockCustomMetricsTLSCert(namespace string) (certPEM, keyPEM []byte, err error) {
	key, err := rsa.GenerateKey(rand.Reader, 2048)
	if err != nil {
		return nil, nil, err
	}
	svcHost := fmt.Sprintf("%s.%s.svc", MockCustomMetricsAPIServerSvcName, namespace)
	template := x509.Certificate{
		SerialNumber: big.NewInt(time.Now().UnixNano()),
		Subject:      pkix.Name{CommonName: MockCustomMetricsAPIServerSvcName},
		NotBefore:    time.Now().Add(-time.Hour),
		NotAfter:     time.Now().Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames: []string{
			MockCustomMetricsAPIServerSvcName,
			fmt.Sprintf("%s.%s", MockCustomMetricsAPIServerSvcName, namespace),
			svcHost,
			fmt.Sprintf("%s.cluster.local", svcHost),
		},
	}
	der, err := x509.CreateCertificate(rand.Reader, &template, &template, &key.PublicKey, key)
	if err != nil {
		return nil, nil, err
	}
	certPEM = pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: der})
	keyPEM = pem.EncodeToMemory(&pem.Block{Type: "RSA PRIVATE KEY", Bytes: x509.MarshalPKCS1PrivateKey(key)})
	return certPEM, keyPEM, nil
}

func ensureMockCustomMetricsTLSSecret(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	certPEM, keyPEM, err := generateMockCustomMetricsTLSCert(namespace)
	if err != nil {
		return err
	}
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: MockCustomMetricsAPIServerTLSSecret, Namespace: namespace},
		Type:       corev1.SecretTypeTLS,
		Data: map[string][]byte{
			corev1.TLSCertKey:       certPEM,
			corev1.TLSPrivateKeyKey: keyPEM,
		},
	}
	if _, err := clientset.CoreV1().Secrets(namespace).Create(ctx, secret, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return err
		}
		secret, getErr := clientset.CoreV1().Secrets(namespace).Get(ctx, MockCustomMetricsAPIServerTLSSecret, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}
		secret.Data[corev1.TLSCertKey] = certPEM
		secret.Data[corev1.TLSPrivateKeyKey] = keyPEM
		_, err = clientset.CoreV1().Secrets(namespace).Update(ctx, secret, metav1.UpdateOptions{})
	}
	return err
}

func mockCustomMetricsDeployment(namespace string) *appsv1.Deployment {
	replicas := int32(1)
	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      MockCustomMetricsAPIServerDeployName,
			Namespace: namespace,
			Labels:    map[string]string{"app": MockCustomMetricsAPIServerDeployName},
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": MockCustomMetricsAPIServerDeployName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": MockCustomMetricsAPIServerDeployName}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "openresty",
						Image: "openresty/openresty:alpine",
						Ports: []corev1.ContainerPort{{ContainerPort: MockCustomMetricsAPIServerPort}},
						VolumeMounts: []corev1.VolumeMount{
							{Name: "nginx-conf", MountPath: "/etc/nginx/conf.d"},
							{Name: "tls-certs", MountPath: "/etc/nginx/certs", ReadOnly: true},
						},
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("10m"),
								corev1.ResourceMemory: resource.MustParse("32Mi"),
							},
						},
					}},
					Volumes: []corev1.Volume{
						{
							Name: "nginx-conf",
							VolumeSource: corev1.VolumeSource{
								ConfigMap: &corev1.ConfigMapVolumeSource{
									LocalObjectReference: corev1.LocalObjectReference{Name: MockCustomMetricsAPIServerConfigName},
								},
							},
						},
						{
							Name: "tls-certs",
							VolumeSource: corev1.VolumeSource{
								Secret: &corev1.SecretVolumeSource{
									SecretName: MockCustomMetricsAPIServerTLSSecret,
								},
							},
						},
					},
				},
			},
		},
	}
}

// DeployMockCustomMetricsAPI 部署 mock custom.metrics.k8s.io HTTPS 服务并注册 APIService。
func DeployMockCustomMetricsAPI(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string) error {
	if err := deleteMockCustomMetricsAPIServiceIfOwned(dynamicClient, ctx); err != nil {
		return err
	}

	mockValue := CustomMetricsMockReturnValue()
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: MockCustomMetricsAPIServerConfigName, Namespace: namespace},
		Data: map[string]string{
			"default.conf": mockCustomMetricsOpenRestyConf(mockValue),
		},
	}
	if _, err := clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	} else if apierrors.IsAlreadyExists(err) {
		cm.Data["default.conf"] = mockCustomMetricsOpenRestyConf(mockValue)
		if _, err := clientset.CoreV1().ConfigMaps(namespace).Update(ctx, cm, metav1.UpdateOptions{}); err != nil {
			return err
		}
	}

	if err := ensureMockCustomMetricsTLSSecret(clientset, ctx, namespace); err != nil {
		return err
	}

	dep := mockCustomMetricsDeployment(namespace)
	dep.Spec.Template.Annotations = map[string]string{
		"e2e-metrics-framework/restartedAt": fmt.Sprintf("%d", time.Now().UnixNano()),
	}
	if _, err := clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{}); err != nil {
		if !apierrors.IsAlreadyExists(err) {
			return err
		}
		existing, getErr := clientset.AppsV1().Deployments(namespace).Get(ctx, MockCustomMetricsAPIServerDeployName, metav1.GetOptions{})
		if getErr != nil {
			return getErr
		}
		dep.ResourceVersion = existing.ResourceVersion
		if _, err := clientset.AppsV1().Deployments(namespace).Update(ctx, dep, metav1.UpdateOptions{}); err != nil {
			return err
		}
	}

	svc := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      MockCustomMetricsAPIServerSvcName,
			Namespace: namespace,
			Labels:    map[string]string{"app": MockCustomMetricsAPIServerDeployName},
		},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": MockCustomMetricsAPIServerDeployName},
			Ports: []corev1.ServicePort{{
				Name:       "https",
				Port:       MockCustomMetricsAPIServerSvcPort,
				TargetPort: intstr.FromInt32(MockCustomMetricsAPIServerPort),
			}},
		},
	}
	if _, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	WaitForDeploymentReplicas(clientset, ctx, MockCustomMetricsAPIServerDeployName, namespace, 1, 3*time.Minute)
	return registerMockCustomMetricsAPIService(dynamicClient, ctx, namespace)
}

func deleteMockCustomMetricsAPIServiceIfOwned(dynamicClient dynamic.Interface, ctx context.Context) error {
	obj, err := dynamicClient.Resource(customMetricsAPIServiceGVR).Get(ctx, MockCustomMetricsAPIServiceName, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}
	labels, _, _ := unstructured.NestedStringMap(obj.Object, "metadata", "labels")
	if labels["app.kubernetes.io/managed-by"] != MockCustomMetricsManagedByLabel {
		return fmt.Errorf("APIService %s 已存在且非 E2E mock 管理，无法覆盖", MockCustomMetricsAPIServiceName)
	}
	return dynamicClient.Resource(customMetricsAPIServiceGVR).Delete(ctx, MockCustomMetricsAPIServiceName, metav1.DeleteOptions{})
}

func registerMockCustomMetricsAPIService(dynamicClient dynamic.Interface, ctx context.Context, namespace string) error {
	obj := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "apiregistration.k8s.io/v1",
			"kind":       "APIService",
			"metadata": map[string]interface{}{
				"name": MockCustomMetricsAPIServiceName,
				"labels": map[string]interface{}{
					"app.kubernetes.io/managed-by": MockCustomMetricsManagedByLabel,
				},
			},
			"spec": map[string]interface{}{
				"service": map[string]interface{}{
					"name":      MockCustomMetricsAPIServerSvcName,
					"namespace": namespace,
					"port":      int64(MockCustomMetricsAPIServerSvcPort),
				},
				"group":                 "custom.metrics.k8s.io",
				"version":               "v1beta2",
				"groupPriorityMinimum":  int64(100),
				"versionPriority":       int64(100),
				"insecureSkipTLSVerify": true,
			},
		},
	}
	_, err := dynamicClient.Resource(customMetricsAPIServiceGVR).Create(ctx, obj, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		_ = dynamicClient.Resource(customMetricsAPIServiceGVR).Delete(ctx, MockCustomMetricsAPIServiceName, metav1.DeleteOptions{})
		_, err = dynamicClient.Resource(customMetricsAPIServiceGVR).Create(ctx, obj, metav1.CreateOptions{})
	}
	return err
}

// WaitForCustomMetricsAPIReady 等待 mock APIService Available 且 custom.metrics.k8s.io 可发现。
func WaitForCustomMetricsAPIReady(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context) {
	GinkgoWriter.Println("等待 mock custom.metrics.k8s.io APIService 就绪...")
	Eventually(func() bool {
		obj, err := dynamicClient.Resource(customMetricsAPIServiceGVR).Get(ctx, MockCustomMetricsAPIServiceName, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("APIService %s 尚未创建: %v\n", MockCustomMetricsAPIServiceName, err)
			return false
		}
		if ClusterCustomMetricsAvailable(clientset, ctx) {
			return true
		}
		conditions, found, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
		if found {
			for _, c := range conditions {
				cond, ok := c.(map[string]interface{})
				if !ok {
					continue
				}
				if cond["type"] == "Available" && cond["status"] != "True" {
					GinkgoWriter.Printf("APIService %s Available=%v reason=%v message=%v\n",
						MockCustomMetricsAPIServiceName, cond["status"], cond["reason"], cond["message"])
				}
			}
		}
		return false
	}, 5*time.Minute, 5*time.Second).Should(BeTrue(), "mock custom.metrics.k8s.io APIService 应 Available 且 API 可发现")
}

type customMetricValueResponse struct {
	MetricName string `json:"metricName"`
	Value      string `json:"value"`
}

type customMetricValueListResponse struct {
	Items []customMetricValueResponse `json:"items"`
}

func customMetricAPIPathSegments(namespace, deployName, metricName string) []string {
	return customMetricAPIPathSegmentsForResource(namespace, deployName, metricName, "deployments.apps")
}

func customMetricAPIPathSegmentsForResource(namespace, deployName, metricName, resourceName string) []string {
	segments := []string{"namespaces", namespace, resourceName, deployName}
	for _, part := range strings.Split(metricName, "/") {
		if part = strings.TrimSpace(part); part != "" {
			segments = append(segments, part)
		}
	}
	return segments
}

func parseCustomMetricValue(raw []byte) (float64, error) {
	var listResp customMetricValueListResponse
	if err := json.Unmarshal(raw, &listResp); err == nil && len(listResp.Items) > 0 {
		return parseCustomMetricScalarValue(listResp.Items[0].Value)
	}
	var resp customMetricValueResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return 0, err
	}
	return parseCustomMetricScalarValue(resp.Value)
}

func parseCustomMetricScalarValue(value string) (float64, error) {
	valRaw := strings.TrimSpace(value)
	if valRaw == "" {
		return 0, fmt.Errorf("empty custom metric value")
	}
	qty, err := resource.ParseQuantity(valRaw)
	if err != nil {
		val, parseErr := strconv.ParseFloat(valRaw, 64)
		if parseErr != nil {
			return 0, fmt.Errorf("parse custom metric value %q: %w", value, err)
		}
		return val, nil
	}
	return qty.AsApproximateFloat64(), nil
}

func getDeploymentCustomMetricValue(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName, metricName string) (float64, error) {
	resourceNames := []string{"deployments.apps", "apps.deployments"}
	var lastErr error
	for _, resourceName := range resourceNames {
		pathSegments := customMetricAPIPathSegmentsForResource(namespace, deployName, metricName, resourceName)
		for _, ver := range customMetricsAPIVersions {
			req := clientset.RESTClient().Get().
				AbsPath(append([]string{"apis", "custom.metrics.k8s.io", ver}, pathSegments...)...)
			raw, err := req.Do(ctx).Raw()
			if err != nil {
				lastErr = err
				continue
			}
			val, err := parseCustomMetricValue(raw)
			if err != nil {
				lastErr = err
				continue
			}
			return val, nil
		}
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("custom metric %q not found for deployment %s/%s", metricName, namespace, deployName)
	}
	return 0, lastErr
}

// VerifyDeploymentCustomMetricQueryable 再次确认 Custom 指标 API 返回非空值（ES 创建后、Reconcile 前）。
func VerifyDeploymentCustomMetricQueryable(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName, metricName string) {
	val, err := getDeploymentCustomMetricValue(clientset, ctx, namespace, deployName, metricName)
	Expect(err).NotTo(HaveOccurred(), "custom.metrics.k8s.io 应能查询指标 %q", metricName)
	Expect(val).To(BeNumerically(">", 0), "custom 指标 %q 应返回正值", metricName)
	GinkgoWriter.Printf("custom.metrics.k8s.io Deployment %s/%s 指标 %s=%v\n", namespace, deployName, metricName, val)
}

// VerifyESNoTargetRefResolutionFailure 确认 MetricsPipeline 未出现 targetRef 解析类错误。
func VerifyESNoTargetRefResolutionFailure(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	detail := getESMetricsPipelineDetail(obj)
	if !detail.Found {
		return
	}
	Expect(detail.Reason).NotTo(Equal("BuildMetricsContextFailed"),
		"ES %s/%s 不应出现 BuildMetricsContextFailed: %s", namespace, name, detail.Message)
	Expect(strings.ToLower(detail.Message)).NotTo(ContainSubstring("targetref"),
		"ES %s/%s MetricsPipeline 不应含 targetRef 解析错误: %s", namespace, name, detail.Message)
}

// WaitForESApaCustomMetricScaling ES-E2E-015：等待 Custom 指标采集成功且 apa 输出与 Deployment 对齐。
func WaitForESApaCustomMetricScaling(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, minReplicas, maxReplicas int32, timeout time.Duration) {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)
	VerifyESNoTargetRefResolutionFailure(dynamicClient, ctx, esName, namespace)
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired < int64(minReplicas) || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 ∈ [%d,%d]\n", desired, minReplicas, maxReplicas)
			return false
		}
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, int32(desired))
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s Custom 指标路径 apa 输出应与 Deployment %s 对齐", namespace, esName, deployName)
}

// DeploymentCustomMetricAvailable 单次探测 Deployment Custom 指标是否可查询（不等待）。
func DeploymentCustomMetricAvailable(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) bool {
	_, ok := ResolveDeploymentCustomMetric(clientset, ctx, namespace, deployName)
	return ok
}

// CleanupMockCustomMetricsAPI 删除 E2E mock custom.metrics.k8s.io 资源。
func CleanupMockCustomMetricsAPI(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	obj, err := dynamicClient.Resource(customMetricsAPIServiceGVR).Get(ctx, MockCustomMetricsAPIServiceName, metav1.GetOptions{})
	if err == nil {
		labels, _, _ := unstructured.NestedStringMap(obj.Object, "metadata", "labels")
		if labels["app.kubernetes.io/managed-by"] == MockCustomMetricsManagedByLabel {
			_ = dynamicClient.Resource(customMetricsAPIServiceGVR).Delete(ctx, MockCustomMetricsAPIServiceName, metav1.DeleteOptions{})
		}
	}
	_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, MockCustomMetricsAPIServerDeployName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Services(namespace).Delete(ctx, MockCustomMetricsAPIServerSvcName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, MockCustomMetricsAPIServerConfigName, metav1.DeleteOptions{})
	_ = clientset.CoreV1().Secrets(namespace).Delete(ctx, MockCustomMetricsAPIServerTLSSecret, metav1.DeleteOptions{})
}

// --- RSG targetRef 基线（GroupReplication + scale 子资源）---
// ESMetricsRSGTargetRef 构造 ES targetRef 指向 ResourceScalingGroup。
func ESMetricsRSGTargetRef(name, namespace string) ESMetricsTargetRef {
	return ESMetricsTargetRef{
		APIVersion: "autoscaling.openfuyao.com/v1alpha1",
		Kind:       "ResourceScalingGroup",
		Name:       name,
		Namespace:  namespace,
	}
}

// RSGBaselineLowQPSExternalServerMetric 低流量 PromQL ExternalServer 指标（基线采集，不触发扩容）。
func RSGBaselineLowQPSExternalServerMetric() ESApaMetricEntry {
	m := E2E008InferQPSExternalServerMetric()
	m.ExternalServerQuery = "query=" + url.QueryEscape(PrometheusInferQPSLowQuery())
	return m
}

// EnsureResourceScalingGroupControllerForMetrics 安装并等待 RSG 控制器就绪（metrics-framework 016+ 专用）。
func EnsureResourceScalingGroupControllerForMetrics(clientset *kubernetes.Clientset, ctx context.Context) {
	Expect(InstallOrUpgradeResourceScalingGroupController()).NotTo(HaveOccurred())
	VerifyResourceScalingGroupControllerReady(clientset, ctx)
}

// EnsureResourceScalingGroupControllerIfNeeded 若 RSG 控制器 Deployment 已 Ready 则跳过 helm upgrade（019 复用 016 安装）。
func EnsureResourceScalingGroupControllerIfNeeded(clientset *kubernetes.Clientset, ctx context.Context) {
	dep, err := clientset.AppsV1().Deployments(RSGTestNamespace).Get(ctx, RSGControllerDeploymentName, metav1.GetOptions{})
	if err == nil && deploymentReady(dep) {
		GinkgoWriter.Println("ResourceScalingGroup controller 已就绪，跳过 helm upgrade")
		return
	}
	EnsureResourceScalingGroupControllerForMetrics(clientset, ctx)
}

func metricsFrameworkRSGDeploymentNames(namespace string) (prefill, decode, prefillSvc, decodeSvc, appName string) {
	caseID := MetricsFrameworkRSGCaseID
	appName = fmt.Sprintf("rsg-app-%02d", caseID)
	prefill = fmt.Sprintf("rsg-%02d-prefill", caseID)
	decode = fmt.Sprintf("rsg-%02d-decode", caseID)
	prefillSvc = fmt.Sprintf("rsg-%02d-prefill-svc", caseID)
	decodeSvc = fmt.Sprintf("rsg-%02d-decode-svc", caseID)
	return prefill, decode, prefillSvc, decodeSvc, appName
}

// MetricsFrameworkRSGDeploymentNames 返回 ES-E2E-016/017 共用的 prefill/decode 资源名。
func MetricsFrameworkRSGDeploymentNames(_ string) (prefill, decode, prefillSvc, decodeSvc, appName string) {
	return metricsFrameworkRSGDeploymentNames("")
}

// DeployMetricsFrameworkRSGEnvironment 部署 prefill/decode 目标 workload 并创建 GroupReplication RSG（replicas=1）。
func DeployMetricsFrameworkRSGEnvironment(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string) (rsgName, prefillName, decodeName string, err error) {
	prefillName, decodeName, prefillSvc, decodeSvc, appName := metricsFrameworkRSGDeploymentNames(namespace)
	rsgName = MetricsFrameworkRSGName
	groupName := MetricsFrameworkRSGGroupName

	if err = CreateRSGTestDeployment(clientset, ctx, prefillName, namespace, appName, "prefill", 1); err != nil {
		return "", "", "", err
	}
	if err = CreateRSGTestDeployment(clientset, ctx, decodeName, namespace, appName, "decode", 1); err != nil {
		return "", "", "", err
	}
	if err = CreateRSGTestService(clientset, ctx, prefillSvc, namespace, appName, "prefill"); err != nil {
		return "", "", "", err
	}
	if err = CreateRSGTestService(clientset, ctx, decodeSvc, namespace, appName, "decode"); err != nil {
		return "", "", "", err
	}

	targetResources := []map[string]any{
		NewTargetResource("prefill", "apps/v1", "Deployment", prefillName, namespace),
		NewTargetResource("decode", "apps/v1", "Deployment", decodeName, namespace),
	}
	items := make([]any, len(targetResources))
	for i, t := range targetResources {
		items[i] = t
	}
	rsg := NewResourceScalingGroup(rsgName, map[string]any{
		"targetResources": items,
		"scalingStrategy": map[string]any{
			"type": "GroupReplication",
			"groupConfig": map[string]any{
				"groupName": groupName,
				"replicas":  int64(1),
			},
		},
	})
	if err = DeleteRSGIfExists(dynamicClient, ctx, rsgName, namespace); err != nil {
		return "", "", "", err
	}
	if err = CreateRSG(dynamicClient, ctx, rsg); err != nil {
		return "", "", "", err
	}
	return rsgName, prefillName, decodeName, nil
}

func DeleteRSGIfExists(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) error {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return nil
	}
	PrepareRSGForDeletion(dynamicClient, ctx, obj)
	return dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Delete(ctx, name, metav1.DeleteOptions{})
}

// WaitForMetricsFrameworkRSGReady 等待 RSG Ready、group-0 打标且 Pod Ready。
func WaitForMetricsFrameworkRSGReady(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, rsgName, prefillName, decodeName, namespace string) {
	VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)
	VerifyDeploymentLabelsContain(clientset, ctx, namespace, prefillName, map[string]string{
		"rsg.io/group-id": "0",
	})
	VerifyDeploymentLabelsContain(clientset, ctx, namespace, decodeName, map[string]string{
		"rsg.io/group-id": "0",
	})
	WaitForDeploymentReplicas(clientset, ctx, prefillName, namespace, 1, 3*time.Minute)
	WaitForDeploymentReplicas(clientset, ctx, decodeName, namespace, 1, 3*time.Minute)
}

// GetRSGGroupConfigReplicas 读取 RSG spec.scalingStrategy.groupConfig.replicas。
func GetRSGGroupConfigReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) int64 {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	replicas, found, err := unstructured.NestedInt64(obj.Object, "spec", "scalingStrategy", "groupConfig", "replicas")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "RSG %s/%s 应有 groupConfig.replicas", namespace, name)
	return replicas
}

// GetRSGCurrentReplicas 读取 RSG status.currentReplicas。
func GetRSGCurrentReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) int64 {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	replicas, found, err := unstructured.NestedInt64(obj.Object, "status", "currentReplicas")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "RSG %s/%s 应有 status.currentReplicas", namespace, name)
	return replicas
}

// GetRSGLastDesiredReplicas 读取 RSG status.lastDesiredReplicas。
func GetRSGLastDesiredReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) int64 {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	replicas, found, err := unstructured.NestedInt64(obj.Object, "status", "lastDesiredReplicas")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "RSG %s/%s 应有 status.lastDesiredReplicas", namespace, name)
	return replicas
}

// GetRSGScaleSubresourceReplicas 读取 RSG scale 子资源 spec/status.replicas。
func GetRSGScaleSubresourceReplicas(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) (specReplicas, statusReplicas int64) {
	scale, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, name, metav1.GetOptions{}, "scale")
	Expect(err).NotTo(HaveOccurred())
	specReplicas, specFound, err := unstructured.NestedInt64(scale.Object, "spec", "replicas")
	Expect(err).NotTo(HaveOccurred())
	Expect(specFound).To(BeTrue(), "RSG %s/%s scale.spec.replicas 应存在", namespace, name)
	statusReplicas, statusFound, err := unstructured.NestedInt64(scale.Object, "status", "replicas")
	Expect(err).NotTo(HaveOccurred())
	Expect(statusFound).To(BeTrue(), "RSG %s/%s scale.status.replicas 应存在", namespace, name)
	return specReplicas, statusReplicas
}

// WaitForMetricsESOnRSGBaseline 等待并校验 ES-E2E-016 基线：Validated、MetricsPipeline、scale 与 RSG/ES status 对齐。
func WaitForMetricsESOnRSGBaseline(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace, prefillName, decodeName string,
	minReplicas, maxReplicas int32,
) {
	VerifyESCRExists(dynamicClient, ctx, esName, namespace)
	VerifyESValidatedCondition(dynamicClient, ctx, esName, namespace)
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)

	Eventually(func() bool {
		groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
		currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
		scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
		if groupReplicas != 1 || currentReplicas != 1 || scaleSpec != 1 || scaleStatus != 1 {
			GinkgoWriter.Printf("RSG %s 副本未对齐: groupConfig=%d current=%d scale.spec=%d scale.status=%d\n",
				rsgName, groupReplicas, currentReplicas, scaleSpec, scaleStatus)
			return false
		}

		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired < int64(minReplicas) || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 ∈ [%d,%d]\n", desired, minReplicas, maxReplicas)
			return false
		}
		status, reason, _, ok, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		if !ok || status != "True" || reason != MetricsReasonScalingSucceeded {
			return false
		}
		return true
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 应与 RSG %s scale 子资源及 groupConfig.replicas 基线对齐", namespace, esName, rsgName)

	scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
	currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(scaleStatus).To(Equal(groupReplicas), "scale.status.replicas 应与 groupConfig.replicas 一致")
	Expect(scaleSpec).To(Equal(groupReplicas), "scale.spec.replicas 应与 groupConfig.replicas 一致")
	Expect(currentReplicas).To(Equal(groupReplicas), "status.currentReplicas 应与 groupConfig.replicas 一致")
	Expect(groupReplicas).To(Equal(int64(1)), "基线 group 数应为 1")

	// 扩缩对象应为 RSG 组数：ES 不应直接改 prefill/decode Deployment spec.replicas（仍由 RSG 管理 group-0）
	prefill, err := clientset.AppsV1().Deployments(namespace).Get(ctx, prefillName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	decode, err := clientset.AppsV1().Deployments(namespace).Get(ctx, decodeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	Expect(prefill.Spec.Replicas).NotTo(BeNil())
	Expect(decode.Spec.Replicas).NotTo(BeNil())
	Expect(*prefill.Spec.Replicas).To(Equal(int32(1)))
	Expect(*decode.Spec.Replicas).To(Equal(int32(1)))
}

// VerifyMetricsESOnRSGBaselineBeforeScaleUp 在 ES-E2E-016 刚执行完后做即时校验，供 017 复用 016 基线而无需重复 5 分钟 Eventually。
func VerifyMetricsESOnRSGBaselineBeforeScaleUp(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace, prefillName, decodeName string,
	minReplicas, maxReplicas int32,
) {
	VerifyESCRExists(dynamicClient, ctx, esName, namespace)
	VerifyESValidatedCondition(dynamicClient, ctx, esName, namespace)

	groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
	currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
	scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(groupReplicas).To(Equal(int64(1)), "017 前置 RSG groupConfig.replicas 应为 1")
	Expect(currentReplicas).To(Equal(int64(1)))
	Expect(scaleSpec).To(Equal(int64(1)))
	Expect(scaleStatus).To(Equal(int64(1)))

	obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	Expect(err).NotTo(HaveOccurred())
	desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
	Expect(found).To(BeTrue(), "ES %s/%s 应有 desiredReplicas", namespace, esName)
	Expect(desired).To(BeNumerically(">=", int64(minReplicas)))
	Expect(desired).To(BeNumerically("<=", int64(maxReplicas)))

	status, reason, _, ok, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
	Expect(ok).To(BeTrue())
	Expect(status).To(Equal("True"))
	Expect(reason).To(Equal(MetricsReasonScalingSucceeded))

	prefill, err := clientset.AppsV1().Deployments(namespace).Get(ctx, prefillName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	decode, err := clientset.AppsV1().Deployments(namespace).Get(ctx, decodeName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	Expect(prefill.Spec.Replicas).NotTo(BeNil())
	Expect(decode.Spec.Replicas).NotTo(BeNil())
	Expect(*prefill.Spec.Replicas).To(Equal(int32(1)))
	Expect(*decode.Spec.Replicas).To(Equal(int32(1)))
}

// CleanupMetricsFrameworkRSGResources 清理 ES-E2E-016 创建的 RSG 与 prefill/decode 测试 workload。
func CleanupMetricsFrameworkRSGResources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	prefillName, decodeName, prefillSvc, decodeSvc, _ := metricsFrameworkRSGDeploymentNames(namespace)
	rsgName := MetricsFrameworkRSGName
	groupName := MetricsFrameworkRSGGroupName
	if obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, rsgName, metav1.GetOptions{}); err == nil {
		PrepareRSGForDeletion(dynamicClient, ctx, obj)
		_ = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Delete(ctx, rsgName, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		fmt.Sprintf("%s-%s-1", groupName, prefillName),
		fmt.Sprintf("%s-%s-2", groupName, prefillName),
		fmt.Sprintf("%s-%s-1", groupName, decodeName),
		fmt.Sprintf("%s-%s-2", groupName, decodeName),
		prefillName,
		decodeName,
	} {
		_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		fmt.Sprintf("%s-%s-1", prefillSvc, groupName),
		fmt.Sprintf("%s-%s-2", prefillSvc, groupName),
		fmt.Sprintf("%s-%s-1", decodeSvc, groupName),
		fmt.Sprintf("%s-%s-2", decodeSvc, groupName),
		prefillSvc,
		decodeSvc,
	} {
		_ = clientset.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		rsgName + "-replica-config",
		rsgName + "-prefill-snapshot",
		rsgName + "-decode-snapshot",
	} {
		_ = clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
}

// --- RSG 组数扩容与 group 克隆 ---
// RSGGroupCloneDeploymentName 返回 group-N 克隆 Deployment 名（groupID>=1）。
func RSGGroupCloneDeploymentName(groupName, targetDeployName string, groupID int) string {
	return fmt.Sprintf("%s-%s-%d", groupName, targetDeployName, groupID)
}

// RSGGroupCloneServiceName 返回 group-N 克隆 Service 名（groupID>=1）。
func RSGGroupCloneServiceName(serviceName, groupName string, groupID int) string {
	return fmt.Sprintf("%s-%s-%d", serviceName, groupName, groupID)
}

// SumRSGTargetReadyReplicas 汇总 RSG 管理范围内某 target（prefill/decode）各组的 readyReplicas。
// groupCount>0 时仅统计 rsg.io/group-id ∈ [0, groupCount)，避免更高 group 克隆尚未删除导致合计偏大（017 常见 3 vs 2）。
func SumRSGTargetReadyReplicas(clientset *kubernetes.Clientset, ctx context.Context, namespace, rsgName, targetName string, groupCount int64) int32 {
	deps, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("rsg.io/name=%s,rsg.io/target-name=%s", rsgName, targetName),
	})
	Expect(err).NotTo(HaveOccurred())
	var sum int32
	for _, dep := range deps.Items {
		if groupCount > 0 {
			gidStr := strings.TrimSpace(dep.Labels["rsg.io/group-id"])
			gid, err := strconv.ParseInt(gidStr, 10, 32)
			if err != nil || gid < 0 || gid >= groupCount {
				continue
			}
		}
		sum += dep.Status.ReadyReplicas
	}
	return sum
}

func logRSGTargetReadyReplicasMismatch(clientset *kubernetes.Clientset, ctx context.Context, namespace, rsgName, targetName string, groupCount int64, prefillSum, decodeSum int32) {
	deps, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: fmt.Sprintf("rsg.io/name=%s", rsgName),
	})
	if err != nil {
		return
	}
	for _, dep := range deps.Items {
		GinkgoWriter.Printf("RSG %s Deployment %s target=%s group-id=%s ready=%d/%v\n",
			rsgName, dep.Name, dep.Labels["rsg.io/target-name"], dep.Labels["rsg.io/group-id"],
			dep.Status.ReadyReplicas, dep.Spec.Replicas)
	}
	GinkgoWriter.Printf("RSG %s readyReplicas 合计（groupCount=%d）prefill=%d decode=%d\n",
		rsgName, groupCount, prefillSum, decodeSum)
}

// waitForRSGTargetReadyReplicaSums 等待有效组范围内 prefill/decode readyReplicas 合计与 groupCount 一致。
func waitForRSGTargetReadyReplicaSums(clientset *kubernetes.Clientset, ctx context.Context, namespace, rsgName string, groupCount int64, timeout time.Duration) {
	Eventually(func() bool {
		prefillSum := SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "prefill", groupCount)
		decodeSum := SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "decode", groupCount)
		if prefillSum != int32(groupCount) || decodeSum != int32(groupCount) {
			logRSGTargetReadyReplicasMismatch(clientset, ctx, namespace, rsgName, "prefill", groupCount, prefillSum, decodeSum)
			return false
		}
		return true
	}, timeout, 5*time.Second).Should(BeTrue(),
		"RSG %s prefill/decode readyReplicas 合计应均为 %d（仅统计 group-id ∈ [0,%d)）",
		rsgName, groupCount, groupCount)
}

// VerifyRSGGroupCloneReady 校验 group-1 克隆 prefill/decode 存在且标签正确。
func VerifyRSGGroupCloneReady(
	clientset *kubernetes.Clientset,
	ctx context.Context,
	namespace, rsgName, groupName, prefillName, decodeName string,
	groupID int,
) {
	prefillClone := RSGGroupCloneDeploymentName(groupName, prefillName, groupID)
	decodeClone := RSGGroupCloneDeploymentName(groupName, decodeName, groupID)
	groupIDStr := fmt.Sprintf("%d", groupID)

	VerifyDeploymentExists(clientset, ctx, namespace, prefillClone)
	VerifyDeploymentExists(clientset, ctx, namespace, decodeClone)
	VerifyDeploymentLabelsContain(clientset, ctx, namespace, prefillClone, map[string]string{
		"rsg.io/group-id":   groupIDStr,
		"rsg.io/group-name": groupName,
		"rsg.io/name":       rsgName,
		"rsg.io/target-name": "prefill",
	})
	VerifyDeploymentLabelsContain(clientset, ctx, namespace, decodeClone, map[string]string{
		"rsg.io/group-id":   groupIDStr,
		"rsg.io/group-name": groupName,
		"rsg.io/name":       rsgName,
		"rsg.io/target-name": "decode",
	})
	WaitForDeploymentReplicas(clientset, ctx, prefillClone, namespace, 1, 5*time.Minute)
	WaitForDeploymentReplicas(clientset, ctx, decodeClone, namespace, 1, 5*time.Minute)
}

// WaitForMetricsRSGScaleUp 等待 ES 经 scale 子资源将 RSG 组数扩至 >minReplicas，并校验 group 克隆与 readyReplicas 汇总。
// expectedGroups 取 ES 收敛后的 status.desiredReplicas（apa 契约下 vector(150)/target50 可能为 3）。
func WaitForMetricsRSGScaleUp(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace, groupName, prefillName, decodeName string,
	minReplicas, maxReplicas int32,
	timeout time.Duration,
) int64 {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)

	var expectedGroups int64
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired <= int64(minReplicas) || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 > %d 且 <= %d\n", desired, minReplicas, maxReplicas)
			return false
		}
		status, reason, _, ok, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		if !ok || status != "True" || reason != MetricsReasonScalingSucceeded {
			return false
		}

		groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
		currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
		scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
		if groupReplicas != desired || currentReplicas != desired ||
			scaleSpec != desired || scaleStatus != desired {
			GinkgoWriter.Printf("RSG %s 副本未对齐: groupConfig=%d current=%d scale.spec=%d scale.status=%d，等待 ES desired=%d\n",
				rsgName, groupReplicas, currentReplicas, scaleSpec, scaleStatus, desired)
			return false
		}
		expectedGroups = desired
		return true
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 应驱动 RSG %s 组数扩至 (%d,%d]", namespace, esName, rsgName, minReplicas, maxReplicas)

	VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, expectedGroups, expectedGroups)
	VerifyESScalingCalculatedCondition(dynamicClient, ctx, esName, namespace, "True", MetricsReasonScalingSucceeded)

	scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(scaleStatus).To(Equal(expectedGroups))
	Expect(scaleSpec).To(Equal(expectedGroups))
	Expect(groupReplicas).To(Equal(expectedGroups))

	for groupID := 1; groupID < int(expectedGroups); groupID++ {
		VerifyRSGGroupCloneReady(clientset, ctx, namespace, rsgName, groupName, prefillName, decodeName, groupID)
	}

	WaitForDeploymentReplicas(clientset, ctx, prefillName, namespace, 1, 5*time.Minute)
	WaitForDeploymentReplicas(clientset, ctx, decodeName, namespace, 1, 5*time.Minute)

	waitForRSGTargetReadyReplicaSums(clientset, ctx, namespace, rsgName, expectedGroups, 3*time.Minute)
	return expectedGroups
}

// VerifyMetricsRSGScaledStateImmediate 在 ES-E2E-017 刚执行完后做即时校验，供 018 复用扩容态而无需重复轮询。
func VerifyMetricsRSGScaledStateImmediate(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	rsgName, namespace string,
	minReplicas int32,
) int64 {
	priorScaledGroups := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(priorScaledGroups).To(BeNumerically(">", int64(minReplicas)),
		"RSG %s 组数应大于 minReplicas=%d", rsgName, minReplicas)
	Expect(SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "prefill", priorScaledGroups)).To(Equal(int32(priorScaledGroups)))
	Expect(SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "decode", priorScaledGroups)).To(Equal(int32(priorScaledGroups)))
	return priorScaledGroups
}

// TriggerRSGScaleUpLoadTest 部署/复用推理服务、Patch 高 PromQL 并启动压测（ES-E2E-017/020 共用）。
// 调用方应 defer StopMetricsInferenceLoadTest017 以清理压测 Job。
func TriggerRSGScaleUpLoadTest(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, namespace string,
) {
	EnsureMetricsInferenceService017Ready(clientset, ctx, namespace)
	Expect(PatchESApaExternalServerPromQL(dynamicClient, ctx, esName, namespace, PrometheusInferQPSQuery())).NotTo(HaveOccurred())
	StartMetricsInferenceLoadTest017(clientset, ctx, namespace)
	RunMetricsInferenceHTTPProbe017(clientset, ctx, namespace)
}

// --- RSG 组数缩容与克隆清理 ---
// VerifyRSGGroupClonesDeleted 确认 group>=1 的克隆 Deployment/Service 已删除。
func VerifyRSGGroupClonesDeleted(
	clientset *kubernetes.Clientset,
	ctx context.Context,
	namespace, groupName, prefillName, decodeName, prefillSvc, decodeSvc string,
	fromGroupID, toGroupID int,
) {
	for groupID := fromGroupID; groupID <= toGroupID; groupID++ {
		prefillClone := RSGGroupCloneDeploymentName(groupName, prefillName, groupID)
		decodeClone := RSGGroupCloneDeploymentName(groupName, decodeName, groupID)
		prefillSvcClone := RSGGroupCloneServiceName(prefillSvc, groupName, groupID)
		decodeSvcClone := RSGGroupCloneServiceName(decodeSvc, groupName, groupID)
		VerifyDeploymentDeleted(clientset, ctx, namespace, prefillClone)
		VerifyDeploymentDeleted(clientset, ctx, namespace, decodeClone)
		VerifyServiceDeleted(clientset, ctx, namespace, prefillSvcClone)
		VerifyServiceDeleted(clientset, ctx, namespace, decodeSvcClone)
	}
}

// WaitForMetricsRSGScaleDown 等待 ES 经 scale 子资源将 RSG 组数缩至 minReplicas，并校验克隆资源清理与 readyReplicas 汇总。
func WaitForMetricsRSGScaleDown(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace, groupName, prefillName, decodeName, prefillSvc, decodeSvc string,
	minReplicas int32,
	priorScaledGroups int64,
	timeout time.Duration,
) {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)

	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired != int64(minReplicas) {
			GinkgoWriter.Printf("ES desiredReplicas=%v，等待 = %d\n", desired, minReplicas)
			return false
		}
		status, reason, _, ok, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		if !ok || status != "True" || reason != MetricsReasonScalingSucceeded {
			return false
		}

		groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
		currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
		scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
		if groupReplicas != int64(minReplicas) || currentReplicas != int64(minReplicas) ||
			scaleSpec != int64(minReplicas) || scaleStatus != int64(minReplicas) {
			GinkgoWriter.Printf("RSG %s 副本未对齐: groupConfig=%d current=%d scale.spec=%d scale.status=%d，等待 = %d\n",
				rsgName, groupReplicas, currentReplicas, scaleSpec, scaleStatus, minReplicas)
			return false
		}

		min := int64(minReplicas)
		if SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "prefill", min) != minReplicas ||
			SumRSGTargetReadyReplicas(clientset, ctx, namespace, rsgName, "decode", min) != minReplicas {
			return false
		}
		return true
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 应驱动 RSG %s 组数缩至 minReplicas=%d", namespace, esName, rsgName, minReplicas)

	VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, int64(minReplicas), int64(minReplicas))
	VerifyESScalingCalculatedCondition(dynamicClient, ctx, esName, namespace, "True", MetricsReasonScalingSucceeded)

	scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(scaleStatus).To(Equal(int64(minReplicas)))
	Expect(scaleSpec).To(Equal(int64(minReplicas)))
	Expect(GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)).To(Equal(int64(minReplicas)))
	Expect(GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)).To(Equal(int64(minReplicas)))

	WaitForDeploymentReplicas(clientset, ctx, prefillName, namespace, minReplicas, 5*time.Minute)
	WaitForDeploymentReplicas(clientset, ctx, decodeName, namespace, minReplicas, 5*time.Minute)
	waitForRSGTargetReadyReplicaSums(clientset, ctx, namespace, rsgName, int64(minReplicas), 3*time.Minute)

	if priorScaledGroups > int64(minReplicas) {
		VerifyRSGGroupClonesDeleted(
			clientset, ctx, namespace, groupName, prefillName, decodeName, prefillSvc, decodeSvc,
			1, int(priorScaledGroups)-1,
		)
	}
}

// --- RSG + Resource/cpu PodList ---
func metricsRSGResourceDeploymentNames() (prefill, decode, prefillSvc, decodeSvc, appName string) {
	caseID := MetricsFrameworkRSGResourceCaseID
	appName = fmt.Sprintf("rsg-app-%02d", caseID)
	prefill = fmt.Sprintf("rsg-%02d-prefill", caseID)
	decode = fmt.Sprintf("rsg-%02d-decode", caseID)
	prefillSvc = fmt.Sprintf("rsg-%02d-prefill-svc", caseID)
	decodeSvc = fmt.Sprintf("rsg-%02d-decode-svc", caseID)
	return prefill, decode, prefillSvc, decodeSvc, appName
}

// createRSGTestDeploymentWithCPURequest 创建带 CPU request 的 RSG 测试 Deployment（供 Resource/cpu 采集）。
func createRSGTestDeploymentWithCPURequest(clientset *kubernetes.Clientset, ctx context.Context, name, namespace, appName, role string, replicas int32) error {
	depLabels := map[string]string{
		"app":                  appName,
		"role":                 role,
		RSGWebhookTestLabelKey: RSGWebhookTestLabelValue,
	}
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    depLabels,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: depLabels},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: depLabels},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{{
						Name:  "nginx",
						Image: "nginx:alpine",
						Resources: corev1.ResourceRequirements{
							Requests: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("100m"),
								corev1.ResourceMemory: resource.MustParse("128Mi"),
							},
							Limits: corev1.ResourceList{
								corev1.ResourceCPU:    resource.MustParse("500m"),
								corev1.ResourceMemory: resource.MustParse("256Mi"),
							},
						},
					}},
				},
			},
		},
	}
	_, err := clientset.AppsV1().Deployments(namespace).Create(ctx, dep, metav1.CreateOptions{})
	return err
}

// DeployMetricsRSGResourceCPUTestEnvironment 部署带 CPU request 的 prefill/decode 与 GroupReplication RSG。
func DeployMetricsRSGResourceCPUTestEnvironment(clientset *kubernetes.Clientset, dynamicClient dynamic.Interface, ctx context.Context, namespace string) (rsgName, prefillName, decodeName, prefillSvc, decodeSvc, appName string, err error) {
	prefillName, decodeName, prefillSvc, decodeSvc, appName = metricsRSGResourceDeploymentNames()
	rsgName = MetricsFrameworkRSGResourceName
	groupName := MetricsFrameworkRSGResourceGroupName

	if err = createRSGTestDeploymentWithCPURequest(clientset, ctx, prefillName, namespace, appName, "prefill", 1); err != nil {
		return "", "", "", "", "", "", err
	}
	if err = createRSGTestDeploymentWithCPURequest(clientset, ctx, decodeName, namespace, appName, "decode", 1); err != nil {
		return "", "", "", "", "", "", err
	}
	if err = CreateRSGTestService(clientset, ctx, prefillSvc, namespace, appName, "prefill"); err != nil {
		return "", "", "", "", "", "", err
	}
	if err = CreateRSGTestService(clientset, ctx, decodeSvc, namespace, appName, "decode"); err != nil {
		return "", "", "", "", "", "", err
	}

	targetResources := []map[string]any{
		NewTargetResource("prefill", "apps/v1", "Deployment", prefillName, namespace),
		NewTargetResource("decode", "apps/v1", "Deployment", decodeName, namespace),
	}
	items := make([]any, len(targetResources))
	for i, t := range targetResources {
		items[i] = t
	}
	rsg := NewResourceScalingGroup(rsgName, map[string]any{
		"targetResources": items,
		"scalingStrategy": map[string]any{
			"type": "GroupReplication",
			"groupConfig": map[string]any{
				"groupName": groupName,
				"replicas":  int64(1),
			},
		},
	})
	if err = DeleteRSGIfExists(dynamicClient, ctx, rsgName, namespace); err != nil {
		return "", "", "", "", "", "", err
	}
	if err = CreateRSG(dynamicClient, ctx, rsg); err != nil {
		return "", "", "", "", "", "", err
	}
	return rsgName, prefillName, decodeName, prefillSvc, decodeSvc, appName, nil
}

// WaitForMetricsRSGResourceReady 等待 RSG Ready 且 prefill/decode Pod Ready。
func WaitForMetricsRSGResourceReady(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, rsgName, prefillName, decodeName, namespace string) {
	VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, 1, 1)
	WaitForDeploymentReplicas(clientset, ctx, prefillName, namespace, 1, 3*time.Minute)
	WaitForDeploymentReplicas(clientset, ctx, decodeName, namespace, 1, 3*time.Minute)
}

// VerifyRSGStatusSelectorParsable 确认 RSG status.selector 非空且可解析。
func VerifyRSGStatusSelectorParsable(dynamicClient dynamic.Interface, ctx context.Context, rsgName, namespace, appName string) string {
	VerifyRSGStatusSelectorContains(dynamicClient, ctx, rsgName, []string{
		"app=" + appName,
		RSGWebhookTestLabelKey + "=" + RSGWebhookTestLabelValue,
	})
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, rsgName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())
	selector, found, err := unstructured.NestedString(obj.Object, "status", "selector")
	Expect(err).NotTo(HaveOccurred())
	Expect(found).To(BeTrue(), "RSG %s/%s status.selector 应存在", namespace, rsgName)
	Expect(selector).NotTo(BeEmpty())
	_, err = labels.Parse(selector)
	Expect(err).NotTo(HaveOccurred(), "RSG status.selector 应可解析为 label selector: %q", selector)
	return selector
}

// CountPodsMatchingSelector 统计 namespace 内匹配 selector 的 Running Pod 数。
func CountPodsMatchingSelector(clientset *kubernetes.Clientset, ctx context.Context, namespace, selector string) int {
	sel, err := labels.Parse(selector)
	Expect(err).NotTo(HaveOccurred())
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: sel.String(),
	})
	Expect(err).NotTo(HaveOccurred())
	count := 0
	for _, pod := range pods.Items {
		if pod.Status.Phase == corev1.PodRunning {
			count++
		}
	}
	return count
}

// StartRSGPrefillDecodeHTTPLoad 向 prefill/decode Service 持续发 HTTP 请求，制造 CPU 使用。
func StartRSGPrefillDecodeHTTPLoad(clientset *kubernetes.Clientset, ctx context.Context, namespace, prefillSvc, decodeSvc string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsFrameworkRSGLoadTestJobName)
	WaitForServiceEndpointsReady(clientset, ctx, namespace, prefillSvc, 2*time.Minute)
	WaitForServiceEndpointsReady(clientset, ctx, namespace, decodeSvc, 2*time.Minute)

	prefillURL := fmt.Sprintf("http://%s.%s.svc.cluster.local", prefillSvc, namespace)
	decodeURL := fmt.Sprintf("http://%s.%s.svc.cluster.local", decodeSvc, namespace)
	loadScript := fmt.Sprintf(`while true; do
  curl -s -o /dev/null %q || true
  curl -s -o /dev/null %q || true
  sleep 0.2
done`, prefillURL, decodeURL)

	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: MetricsFrameworkRSGLoadTestJobName, Namespace: namespace},
		Spec: batchv1.JobSpec{
			BackoffLimit: func() *int32 { v := int32(0); return &v }(),
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:    "load",
						Image:   "curlimages/curl:8.5.0",
						Command: []string{"sh", "-c", loadScript},
					}},
				},
			},
		},
	}
	_, err := clientset.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Println("RSG prefill/decode HTTP 压测已启动")
}

// StopRSGPrefillDecodeHTTPLoad 停止 ES-E2E-019 RSG HTTP 压测。
func StopRSGPrefillDecodeHTTPLoad(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsFrameworkRSGLoadTestJobName)
}

// WaitForMetricsRSGResourceCPUMetricScaling 等待 RSG targetRef 下 Resource/cpu 全量采集成功且 ScalingCalculated 收敛。
func WaitForMetricsRSGResourceCPUMetricScaling(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace string,
	minReplicas, maxReplicas int32,
	recovery *ResourceMetricsWaitRecovery,
	timeout time.Duration,
) {
	Eventually(func() bool {
		tryRecoverResourcePodMetrics(recovery, ctx)
		status, reason, message, ok, err := ESCondition(dynamicClient, ctx, esName, namespace, "MetricsPipeline")
		if err != nil || !ok {
			return false
		}
		if status != "True" || reason != MetricsReasonCollectionSucceeded {
			if reason == MetricsReasonPartialSuccess && message != "" {
				GinkgoWriter.Printf("MetricsPipeline 暂为 PartialSuccess: %s\n", message)
			}
			return false
		}
		if !strings.Contains(message, "processed 1 metric(s)") {
			return false
		}
		if strings.Contains(message, "no pods in metrics context") {
			return false
		}
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired < int64(minReplicas) || desired > int64(maxReplicas) {
			return false
		}
		scStatus, scReason, _, scOK, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		if !scOK || scStatus != "True" || scReason != MetricsReasonScalingSucceeded {
			return false
		}
		groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
		return groupReplicas == desired
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s Resource/cpu 应对 RSG %s 全量采集成功", namespace, esName, rsgName)

	VerifyESResourceMetricsWithPodListSuccess(dynamicClient, ctx, esName, namespace, recovery)
	VerifyESScalingCalculatedCondition(dynamicClient, ctx, esName, namespace, "True", MetricsReasonScalingSucceeded)
}

// CleanupMetricsRSGResourceTestResources 清理 ES-E2E-019 创建的 RSG 与 prefill/decode 测试 workload。
func CleanupMetricsRSGResourceTestResources(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	StopRSGPrefillDecodeHTTPLoad(clientset, ctx, namespace)
	prefillName, decodeName, prefillSvc, decodeSvc, _ := metricsRSGResourceDeploymentNames()
	rsgName := MetricsFrameworkRSGResourceName
	groupName := MetricsFrameworkRSGResourceGroupName
	if obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, rsgName, metav1.GetOptions{}); err == nil {
		PrepareRSGForDeletion(dynamicClient, ctx, obj)
		_ = dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Delete(ctx, rsgName, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		fmt.Sprintf("%s-%s-1", groupName, prefillName),
		fmt.Sprintf("%s-%s-2", groupName, prefillName),
		fmt.Sprintf("%s-%s-1", groupName, decodeName),
		fmt.Sprintf("%s-%s-2", groupName, decodeName),
		prefillName,
		decodeName,
	} {
		_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		fmt.Sprintf("%s-%s-1", prefillSvc, groupName),
		fmt.Sprintf("%s-%s-2", prefillSvc, groupName),
		fmt.Sprintf("%s-%s-1", decodeSvc, groupName),
		fmt.Sprintf("%s-%s-2", decodeSvc, groupName),
		prefillSvc,
		decodeSvc,
	} {
		_ = clientset.CoreV1().Services(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
	for _, name := range []string{
		rsgName + "-replica-config",
		rsgName + "-prefill-snapshot",
		rsgName + "-decode-snapshot",
	} {
		_ = clientset.CoreV1().ConfigMaps(namespace).Delete(ctx, name, metav1.DeleteOptions{})
	}
}

// --- ES/RSG 双控制器状态收敛 ---
// RSGScaleConvergenceBaseline 扩缩前 RSG/scale 基线快照。
type RSGScaleConvergenceBaseline struct {
	GroupConfigReplicas int64
	CurrentReplicas     int64
	ScaleStatusReplicas int64
}

// RecordRSGScaleConvergenceBaseline 记录 ES-E2E-020 扩缩前 RSG 状态。
func RecordRSGScaleConvergenceBaseline(dynamicClient dynamic.Interface, ctx context.Context, rsgName, namespace string) RSGScaleConvergenceBaseline {
	_, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	return RSGScaleConvergenceBaseline{
		GroupConfigReplicas: GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace),
		CurrentReplicas:     GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace),
		ScaleStatusReplicas: scaleStatus,
	}
}

// GetESCurrentReplicas 读取 ES status.currentReplicas。
func GetESCurrentReplicas(dynamicClient dynamic.Interface, ctx context.Context, esName, namespace string) (int64, bool) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	if err != nil {
		return 0, false
	}
	current, found, _ := unstructured.NestedInt64(obj.Object, "status", "currentReplicas")
	return current, found
}

func rsgReadyReconciled(dynamicClient dynamic.Interface, ctx context.Context, rsgName, namespace string) bool {
	obj, err := dynamicClient.Resource(ResourceScalingGroupGVR).Namespace(namespace).Get(ctx, rsgName, metav1.GetOptions{})
	if err != nil {
		return false
	}
	conditions, found, _ := unstructured.NestedSlice(obj.Object, "status", "conditions")
	if !found {
		return false
	}
	for _, c := range conditions {
		cond, _ := c.(map[string]interface{})
		if cond["type"] != "Ready" {
			continue
		}
		return cond["status"] == "True" && cond["reason"] == "Reconciled"
	}
	return false
}

// WaitForESRSGScaleConvergence 触发扩容后等待 ES 写入 RSG scale 且双控制器状态在 timeout 内收敛。
func WaitForESRSGScaleConvergence(
	dynamicClient dynamic.Interface,
	clientset *kubernetes.Clientset,
	ctx context.Context,
	esName, rsgName, namespace, groupName, prefillName, decodeName string,
	minReplicas, maxReplicas int32,
	priorGroupReplicas int64,
	timeout time.Duration,
) int64 {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)

	var expectedGroups int64
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
		if !found || desired <= priorGroupReplicas || desired > int64(maxReplicas) {
			GinkgoWriter.Printf("[020] ES desiredReplicas=%v，等待 > %d 且 <= %d\n", desired, priorGroupReplicas, maxReplicas)
			return false
		}

		scStatus, scReason, scMsg, scOK, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		if !scOK || scStatus != "True" || scReason != MetricsReasonScalingSucceeded {
			return false
		}
		lowerMsg := strings.ToLower(scMsg)
		if !strings.Contains(lowerMsg, "current") || !strings.Contains(lowerMsg, "desired") {
			GinkgoWriter.Printf("[020] ScalingCalculated message 等待含 current/desired: %s\n", scMsg)
			return false
		}

		groupReplicas := GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)
		if groupReplicas != desired {
			GinkgoWriter.Printf("[020] RSG groupConfig.replicas=%d，等待 = ES desired=%d\n", groupReplicas, desired)
			return false
		}

		currentReplicas := GetRSGCurrentReplicas(dynamicClient, ctx, rsgName, namespace)
		lastDesired := GetRSGLastDesiredReplicas(dynamicClient, ctx, rsgName, namespace)
		if currentReplicas != desired || lastDesired != desired {
			GinkgoWriter.Printf("[020] RSG current=%d lastDesired=%d，等待 = %d\n", currentReplicas, lastDesired, desired)
			return false
		}

		_, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
		if scaleStatus != desired {
			GinkgoWriter.Printf("[020] scale.status.replicas=%d，等待 = %d\n", scaleStatus, desired)
			return false
		}

		esCurrent, esCurrentFound := GetESCurrentReplicas(dynamicClient, ctx, esName, namespace)
		if !esCurrentFound || esCurrent != scaleStatus {
			GinkgoWriter.Printf("[020] ES currentReplicas=%v（found=%v），等待 = scale.status=%d\n", esCurrent, esCurrentFound, scaleStatus)
			return false
		}

		if !rsgReadyReconciled(dynamicClient, ctx, rsgName, namespace) {
			GinkgoWriter.Printf("[020] RSG %s Ready 条件尚未 Reconciled\n", rsgName)
			return false
		}

		expectedGroups = desired
		return true
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 写入 RSG scale 后应在 %s 内与 RSG %s 双控制器状态收敛", namespace, esName, timeout, rsgName)

	VerifyRSGReadyCondition(dynamicClient, ctx, rsgName, "True", "Reconciled")
	VerifyRSGStatusReplicas(dynamicClient, ctx, rsgName, expectedGroups, expectedGroups)
	VerifyESScalingCalculatedCondition(dynamicClient, ctx, esName, namespace, "True", MetricsReasonScalingSucceeded)

	scaleSpec, scaleStatus := GetRSGScaleSubresourceReplicas(dynamicClient, ctx, rsgName, namespace)
	Expect(scaleSpec).To(Equal(expectedGroups))
	Expect(scaleStatus).To(Equal(expectedGroups))
	Expect(GetRSGGroupConfigReplicas(dynamicClient, ctx, rsgName, namespace)).To(Equal(expectedGroups))
	Expect(GetESDesiredReplicas(dynamicClient, ctx, esName, namespace)).To(Equal(int32(expectedGroups)))
	esCurrent, found := GetESCurrentReplicas(dynamicClient, ctx, esName, namespace)
	Expect(found).To(BeTrue())
	Expect(esCurrent).To(Equal(scaleStatus))

	for groupID := 1; groupID < int(expectedGroups); groupID++ {
		VerifyRSGGroupCloneReady(clientset, ctx, namespace, rsgName, groupName, prefillName, decodeName, groupID)
	}
	waitForRSGTargetReadyReplicaSums(clientset, ctx, namespace, rsgName, expectedGroups, 3*time.Minute)

	return expectedGroups
}

// --- RSG targetRef 唯一性校验 ---
const (
	rsgTargetRefConflictSubstring1 = "already manages the same target"
	rsgTargetRefConflictSubstring2 = "another ElasticScaler"
)

// AttemptCreateDuplicateRSGTargetRefES 尝试创建与 owner 相同 RSG targetRef 的第二条 ES（metrics 可不同）。
func AttemptCreateDuplicateRSGTargetRefES(
	dynamicClient dynamic.Interface,
	ctx context.Context,
	duplicateESName, namespace string,
	targetRef ESMetricsTargetRef,
	minReplicas, maxReplicas int32,
) error {
	_ = DeleteElasticScalerCR(dynamicClient, ctx, duplicateESName, namespace)
	obj := BuildApaMetricsTriggerESObject(
		duplicateESName, namespace, targetRef, minReplicas, maxReplicas,
		[]ESApaMetricEntry{DefaultResourceCPUMetric()},
	)
	_, err := dynamicClient.Resource(elasticScalerGVR).Namespace(namespace).Create(ctx, obj, metav1.CreateOptions{})
	return err
}

func isTargetRefUniquenessError(message string) bool {
	lower := strings.ToLower(message)
	return strings.Contains(lower, strings.ToLower(rsgTargetRefConflictSubstring1)) ||
		strings.Contains(lower, strings.ToLower(rsgTargetRefConflictSubstring2))
}

// VerifyDuplicateRSGTargetRefESRejected 校验第二条 ES 因 targetRef 冲突被拒绝（admission 4xx 或 Validated=False）。
func VerifyDuplicateRSGTargetRefESRejected(
	dynamicClient dynamic.Interface,
	ctx context.Context,
	duplicateESName, ownerESName, namespace string,
	createErr error,
) {
	if createErr != nil {
		Expect(isTargetRefUniquenessError(createErr.Error())).To(BeTrue(),
			"第二条 ES 创建应因 targetRef 冲突被拒绝: %v", createErr)
		Consistently(func() bool {
			_, err := GetElasticScalerCR(dynamicClient, ctx, duplicateESName, namespace)
			return apierrors.IsNotFound(err)
		}, 10*time.Second, 1*time.Second).Should(BeTrue(),
			"冲突 ES %s/%s 不应被持久化", namespace, duplicateESName)
		GinkgoWriter.Printf("第二条 ES 在 Create 阶段被拒绝: %v\n", createErr)
		return
	}

	GinkgoWriter.Println("第二条 ES Create 成功，等待 Reconcile 侧 Validator 写入 ValidationFailed")
	VerifyESValidationFailedCondition(dynamicClient, ctx, duplicateESName, namespace)
	Eventually(func() bool {
		_, reason, message, ok, err := ESCondition(dynamicClient, ctx, duplicateESName, namespace, "Validated")
		if err != nil || !ok {
			return false
		}
		if reason != "ValidationFailed" || !strings.Contains(strings.ToLower(message), "already manages") {
			GinkgoWriter.Printf("Validated 条件 message=%q reason=%q\n", message, reason)
			return false
		}
		return strings.Contains(message, ownerESName)
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(),
		"冲突 ES Validated 消息应说明 targetRef 已被 %s 管理", ownerESName)
	_ = DeleteElasticScalerCR(dynamicClient, ctx, duplicateESName, namespace)
}

// VerifyOwnerESRSGTargetRefUnaffected 校验 owner ES 在冲突尝试后仍可正常 Reconcile。
func VerifyOwnerESRSGTargetRefUnaffected(dynamicClient dynamic.Interface, ctx context.Context, ownerESName, namespace string) {
	VerifyESCRExists(dynamicClient, ctx, ownerESName, namespace)
	VerifyESValidatedCondition(dynamicClient, ctx, ownerESName, namespace)
	VerifyESScalingCalculatedCondition(dynamicClient, ctx, ownerESName, namespace, "True", MetricsReasonScalingSucceeded)
	Eventually(func() bool {
		status, reason, message, ok, err := ESCondition(dynamicClient, ctx, ownerESName, namespace, "MetricsPipeline")
		if err != nil || !ok {
			return false
		}
		return status == "True" && reason == MetricsReasonCollectionSucceeded &&
			strings.Contains(message, "processed")
	}, 3*time.Minute, 5*time.Second).Should(BeTrue(),
		"owner ES %s/%s MetricsPipeline 不应因冲突 ES 创建而破坏", namespace, ownerESName)
}

// 指标类型说明（切分 ExternalServer 与 External，勿混淆）：
//   - Resource：依赖 metrics-server（metrics.k8s.io），走真实 Pod 资源指标。
//   - ExternalServer：controller HTTP(S) 访问 Prometheus /api/v1/query，须真实端点；protocol/endpoint 均支持 http/https，未填 protocol 且无 scheme 时默认 https。
//   - External：external.metrics.k8s.io；集群无 adapter 时，E2E 可对返回值 mock。
//   - Custom：custom.metrics.k8s.io；集群无 adapter 时，E2E 部署 mock APIService + nginx 固定返回值。
