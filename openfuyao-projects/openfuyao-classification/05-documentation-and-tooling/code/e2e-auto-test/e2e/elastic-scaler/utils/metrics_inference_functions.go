package utils

import (
	"context"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors 	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// PrometheusInferQPSQuery 返回 infer_qps 的 PromQL（环境变量 METRICS_PROMETHEUS_INFER_QPS_QUERY / METRICS_PROMETHEUS_TEST_QUERY 可覆盖）。
func PrometheusInferQPSQuery() string {
	for _, key := range []string{"METRICS_PROMETHEUS_INFER_QPS_QUERY", "METRICS_PROMETHEUS_TEST_QUERY"} {
		if q := strings.TrimSpace(os.Getenv(key)); q != "" {
			return q
		}
	}
	return "vector(150)"
}

// PrometheusInferQPSLowQuery 返回 ES-E2E-009 低流量 PromQL（触发 apa 缩容；METRICS_PROMETHEUS_INFER_QPS_LOW_QUERY 可覆盖）。
// 默认 vector(1)：target=50、downTolerance=0.1 时需 stableValue/target < 0.9；留足余量避免窗口聚合边界抖动。
func PrometheusInferQPSLowQuery() string {
	if q := strings.TrimSpace(os.Getenv("METRICS_PROMETHEUS_INFER_QPS_LOW_QUERY")); q != "" {
		return q
	}
	return "vector(1)"
}

// ClusterPrometheusAvailable 检查集群是否存在 Endpoints 已就绪的 Prometheus Service（单次探测，不等待）。
func ClusterPrometheusAvailable(clientset *kubernetes.Clientset, ctx context.Context) bool {
	_, _, ok := ResolveClusterPrometheus(clientset, ctx)
	return ok
}

func waitForClusterPrometheus(clientset *kubernetes.Clientset, ctx context.Context, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if _, _, ok := ResolveClusterPrometheus(clientset, ctx); ok {
			return true
		}
		select {
		case <-ctx.Done():
			return false
		case <-time.After(10 * time.Second):
		}
	}
	return false
}

var (
	clusterPrometheusChecked bool
	clusterPrometheusOK      bool
)

// SkipUnlessClusterPrometheusAvailable 轮询等待 Prometheus Endpoints 就绪；超时则 Skip（避免 Endpoints 稍后就绪时误 Skip 008/012）。
// 套件内首次探测结果会缓存，后续 008/011/012/013/015/016/017/020 等用例不再重复等待。
func SkipUnlessClusterPrometheusAvailable(clientset *kubernetes.Clientset, ctx context.Context) {
	if clusterPrometheusChecked {
		if !clusterPrometheusOK {
			Skip("集群未部署可发现的 Prometheus Service，跳过本用例")
		}
		return
	}
	timeout := ClusterPrometheusWaitTimeout()
	GinkgoWriter.Printf("等待集群 Prometheus Service Endpoints 就绪（最长 %s，超时则 Skip）...\n", timeout)
	clusterPrometheusOK = waitForClusterPrometheus(clientset, ctx, timeout)
	clusterPrometheusChecked = true
	if !clusterPrometheusOK {
		Skip("集群未部署可发现的 Prometheus Service（Endpoints 在 " + timeout.String() + " 内未就绪），跳过本用例")
	}
}

// EnsureClusterPrometheusAvailable 确认集群 Prometheus Service Endpoints 就绪（ES-E2E-008+ ExternalServer 路径前置）。
func EnsureClusterPrometheusAvailable(clientset *kubernetes.Clientset, ctx context.Context) {
	GinkgoWriter.Println("验证集群 Prometheus Service Endpoints 就绪...")
	Eventually(func() bool {
		_, _, ok := ResolveClusterPrometheus(clientset, ctx)
		return ok
	}, ClusterPrometheusWaitTimeout(), 10*time.Second).Should(BeTrue(), "集群内应存在 Endpoints 已就绪且可访问的 Prometheus Service")
}

// E2E008InferQPSExternalServerMetric 构造 ES-E2E-008 ExternalServer/infer_qps 指标（集群 Prometheus；endpoint 默认含 http:// scheme）。
func E2E008InferQPSExternalServerMetric() ESApaMetricEntry {
	promQL := PrometheusInferQPSQuery()
	endpoint := ClusterPrometheusExternalServerEndpoint()
	return ESApaMetricEntry{
		Type:                   "ExternalServer",
		ExternalServerName:     MetricsExternalServerInferQPS,
		ExternalServerTarget:   MetricsE2E008InferQPSTarget,
		ExternalServerEndpoint: endpoint,
		ExternalServerProtocol: ResolveExternalServerProtocol(endpoint, PrometheusQueryProtocol()),
		ExternalServerQuery:    "query=" + url.QueryEscape(promQL),
	}
}

// DeployMetricsInferenceService 部署轻量推理服务（HTTP 200 模拟 /v1/chat/completions）。
func DeployMetricsInferenceService(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName,
		svcName:         MetricsInferenceSvcName,
		withCPURequests: false,
	})
}

// EnsureMetricsInferenceServiceReady 幂等部署并等待 ES-E2E-008/009 推理服务就绪（009 复用 008 已扩容 Deployment）。
func EnsureMetricsInferenceServiceReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	if metricsInferenceDeploymentReady(clientset, ctx, namespace, MetricsInferenceDeployName) {
		WaitForServiceEndpointsReady(clientset, ctx, namespace, MetricsInferenceSvcName, 2*time.Minute)
		return
	}
	Expect(DeployMetricsInferenceService(clientset, ctx, namespace)).NotTo(HaveOccurred())
}

func ensureMetricsInferenceServiceReady(clientset *kubernetes.Clientset, ctx context.Context, namespace string, names metricsInferenceServiceNames) {
	if metricsInferenceDeploymentReady(clientset, ctx, namespace, names.deployName) {
		WaitForServiceEndpointsReady(clientset, ctx, namespace, names.svcName, 2*time.Minute)
		return
	}
	Expect(deployMetricsInferenceService(clientset, ctx, namespace, names)).NotTo(HaveOccurred())
}

// EnsureMetricsInferenceService010Ready 幂等部署并等待 ES-E2E-010 推理服务就绪（轻量 CPU request，供 Resource/cpu 采集）。
func EnsureMetricsInferenceService010Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames010())
	WaitForDeploymentPodMetricsReady(clientset, ctx, namespace, MetricsInferenceDeployName010, MetricsPodMetricsWaitTimeout())
}

// DeployMetricsInferenceService010 部署 ES-E2E-010 专用推理服务（轻量 CPU request，供 Resource/cpu 采集）。
func DeployMetricsInferenceService010(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames010())
}

func metricsInferenceServiceNames010() metricsInferenceServiceNames {
	return metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName010,
		svcName:         MetricsInferenceSvcName010,
		withCPURequests: true,
		cpuRequest:      MetricsInference010CPURequest,
		memRequest:      MetricsInference010MemRequest,
		cpuLimit:        MetricsInference010CPULimit,
		memLimit:        MetricsInference010MemLimit,
	}
}

// EnsureMetricsInferenceService011Ready 幂等部署并等待 ES-E2E-011 推理服务就绪。
func EnsureMetricsInferenceService011Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName011,
		svcName:         MetricsInferenceSvcName011,
		withCPURequests: false,
	})
}

// DeployMetricsInferenceService011 部署 ES-E2E-011 专用推理服务（独立 targetRef）。
func DeployMetricsInferenceService011(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName011,
		svcName:         MetricsInferenceSvcName011,
		withCPURequests: false,
	})
}

// EnsureMetricsInferenceService012Ready 幂等部署并等待 ES-E2E-012 推理服务就绪（含 Pod metrics 可采集）。
func EnsureMetricsInferenceService012Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName012,
		svcName:         MetricsInferenceSvcName012,
		withCPURequests: true,
	})
	WaitForDeploymentPodMetricsReady(clientset, ctx, namespace, MetricsInferenceDeployName012, MetricsPodMetricsWaitTimeout())
}

// DeployMetricsInferenceService012 部署 ES-E2E-012 专用推理服务（含 CPU request，供 Resource/cpu 采集）。
func DeployMetricsInferenceService012(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName012,
		svcName:         MetricsInferenceSvcName012,
		withCPURequests: true,
	})
}

// EnsureMetricsInferenceService013Ready 幂等部署并等待 ES-E2E-013 推理服务就绪。
func EnsureMetricsInferenceService013Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName013,
		svcName:         MetricsInferenceSvcName013,
		withCPURequests: false,
	})
}

// DeployMetricsInferenceService013 部署 ES-E2E-013 专用推理服务（独立 targetRef）。
func DeployMetricsInferenceService013(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName013,
		svcName:         MetricsInferenceSvcName013,
		withCPURequests: false,
	})
}

// EnsureMetricsInferenceService014Ready 幂等部署并等待 ES-E2E-014 推理服务就绪。
func EnsureMetricsInferenceService014Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName014,
		svcName:         MetricsInferenceSvcName014,
		withCPURequests: false,
	})
}

// DeployMetricsInferenceService014 部署 ES-E2E-014 专用推理服务（独立 targetRef）。
func DeployMetricsInferenceService014(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName014,
		svcName:         MetricsInferenceSvcName014,
		withCPURequests: false,
	})
}

// EnsureMetricsInferenceService015Ready 幂等部署并等待 ES-E2E-015 推理服务就绪。
func EnsureMetricsInferenceService015Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	ensureMetricsInferenceServiceReady(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName015,
		svcName:         MetricsInferenceSvcName015,
		withCPURequests: false,
	})
}

// DeployMetricsInferenceService015 部署 ES-E2E-015 专用推理服务（Custom 指标路径）。
func DeployMetricsInferenceService015(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName015,
		svcName:         MetricsInferenceSvcName015,
		withCPURequests: false,
	})
}

// DeployMetricsInferenceService017 部署 ES-E2E-017 专用推理服务（RSG 扩容压测）。
func DeployMetricsInferenceService017(clientset *kubernetes.Clientset, ctx context.Context, namespace string) error {
	return deployMetricsInferenceService(clientset, ctx, namespace, metricsInferenceServiceNames{
		deployName:      MetricsInferenceDeployName017,
		svcName:         MetricsInferenceSvcName017,
		withCPURequests: false,
	})
}

// EnsureMetricsInferenceService017Ready 幂等部署并等待 ES-E2E-017/020 推理服务就绪（018 停止压测后 020 可复用）。
func EnsureMetricsInferenceService017Ready(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	if metricsInferenceDeploymentReady(clientset, ctx, namespace, MetricsInferenceDeployName017) {
		WaitForServiceEndpointsReady(clientset, ctx, namespace, MetricsInferenceSvcName017, 2*time.Minute)
		return
	}
	Expect(DeployMetricsInferenceService017(clientset, ctx, namespace)).NotTo(HaveOccurred())
}

func metricsInferenceDeploymentReady(clientset *kubernetes.Clientset, ctx context.Context, namespace, deployName string) bool {
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil {
		return false
	}
	return dep.Status.ReadyReplicas >= 1
}

type metricsInferenceServiceNames struct {
	deployName      string
	svcName         string
	withCPURequests bool
	cpuRequest      string
	memRequest      string
	cpuLimit        string
	memLimit        string
}

func metricsInferenceContainerResources(names metricsInferenceServiceNames) corev1.ResourceRequirements {
	cpuReq, memReq, cpuLim, memLim := "100m", "128Mi", "500m", "256Mi"
	if names.cpuRequest != "" {
		cpuReq = names.cpuRequest
	}
	if names.memRequest != "" {
		memReq = names.memRequest
	}
	if names.cpuLimit != "" {
		cpuLim = names.cpuLimit
	}
	if names.memLimit != "" {
		memLim = names.memLimit
	}
	return corev1.ResourceRequirements{
		Requests: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(cpuReq),
			corev1.ResourceMemory: resource.MustParse(memReq),
		},
		Limits: corev1.ResourceList{
			corev1.ResourceCPU:    resource.MustParse(cpuLim),
			corev1.ResourceMemory: resource.MustParse(memLim),
		},
	}
}

func deployMetricsInferenceService(clientset *kubernetes.Clientset, ctx context.Context, namespace string, names metricsInferenceServiceNames) error {
	cmName := names.deployName + "-cfg"
	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Name: cmName, Namespace: namespace},
		Data: map[string]string{
			"default.conf": `server {
    listen 8000;
    location /health {
        return 200 'ok';
    }
    location /v1/chat/completions {
        default_type application/json;
        return 200 '{"id":"metrics-inference","object":"chat.completion","choices":[{"message":{"role":"assistant","content":"ok"}}]}';
    }
}`,
		},
	}
	if _, err := clientset.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}

	replicas := int32(1)
	container := corev1.Container{
		Name:  "nginx",
		Image: "nginx:alpine",
		Ports: []corev1.ContainerPort{{ContainerPort: MetricsInferencePort}},
		VolumeMounts: []corev1.VolumeMount{{
			Name: "nginx-conf", MountPath: "/etc/nginx/conf.d",
		}},
	}
	if names.withCPURequests {
		container.Resources = metricsInferenceContainerResources(names)
		// CPU limit 下 nginx 默认 worker_processes=auto 会在多核节点起大量 worker 导致卡死
		container.Env = []corev1.EnvVar{{Name: "NGINX_ENTRYPOINT_WORKER_PROCESSES_AUTOTUNE", Value: "1"}}
	}
	dep := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: names.deployName, Namespace: namespace},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{MatchLabels: map[string]string{"app": names.deployName}},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: map[string]string{"app": names.deployName}},
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{container},
					Volumes: []corev1.Volume{{
						Name: "nginx-conf",
						VolumeSource: corev1.VolumeSource{
							ConfigMap: &corev1.ConfigMapVolumeSource{
								LocalObjectReference: corev1.LocalObjectReference{Name: cmName},
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
		ObjectMeta: metav1.ObjectMeta{Name: names.svcName, Namespace: namespace},
		Spec: corev1.ServiceSpec{
			Selector: map[string]string{"app": names.deployName},
			Ports: []corev1.ServicePort{{
				Port:       MetricsInferencePort,
				TargetPort: intstr.FromInt32(MetricsInferencePort),
			}},
		},
	}
	if _, err := clientset.CoreV1().Services(namespace).Create(ctx, svc, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
		return err
	}
	WaitForDeploymentReplicas(clientset, ctx, names.deployName, namespace, 1, 3*time.Minute)
	WaitForServiceEndpointsReady(clientset, ctx, namespace, names.svcName, 3*time.Minute)
	return nil
}

// WaitForServiceEndpointsReady 等待 Service 至少有一个可用 Endpoint。
func WaitForServiceEndpointsReady(clientset *kubernetes.Clientset, ctx context.Context, namespace, serviceName string, timeout time.Duration) {
	GinkgoWriter.Printf("等待 Service %s/%s Endpoints 就绪...\n", namespace, serviceName)
	Eventually(func() bool {
		ep, err := clientset.CoreV1().Endpoints(namespace).Get(ctx, serviceName, metav1.GetOptions{})
		if err != nil {
			return false
		}
		for _, subset := range ep.Subsets {
			if len(subset.Addresses) > 0 && len(subset.Ports) > 0 {
				return true
			}
		}
		return false
	}, timeout, 5*time.Second).Should(BeTrue(), "Service %s/%s 应有可用 Endpoints", namespace, serviceName)
}

func waitForJobDeleted(clientset *kubernetes.Clientset, ctx context.Context, namespace, jobName string) {
	Eventually(func() bool {
		_, err := clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{})
		return apierrors.IsNotFound(err)
	}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "Job %s/%s 应已删除", namespace, jobName)
}

func deleteMetricsInferenceJob(clientset *kubernetes.Clientset, ctx context.Context, namespace, jobName string) {
	propagation := metav1.DeletePropagationBackground
	_ = clientset.BatchV1().Jobs(namespace).Delete(ctx, jobName, metav1.DeleteOptions{
		PropagationPolicy: &propagation,
	})
	waitForJobDeleted(clientset, ctx, namespace, jobName)
}

func metricsInferenceServiceURL(namespace string) string {
	return metricsInferenceServiceURLFor(namespace, MetricsInferenceSvcName)
}

func metricsInferenceServiceURLFor(namespace, svcName string) string {
	return fmt.Sprintf("http://%s.%s.svc.cluster.local:%d", svcName, namespace, MetricsInferencePort)
}

// RunMetricsInferenceHTTPProbe 发送推理请求并验证 HTTP 200（含 Endpoints 就绪后重试）。
func RunMetricsInferenceHTTPProbe(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName, MetricsInferenceProbeJobName)
}

// RunMetricsInferenceHTTPProbe010 ES-E2E-010 推理探针。
func RunMetricsInferenceHTTPProbe010(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName010, MetricsInferenceProbeJobName010)
}

// RunMetricsInferenceHTTPProbe011 ES-E2E-011 推理探针。
func RunMetricsInferenceHTTPProbe011(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName011, MetricsInferenceProbeJobName011)
}

// RunMetricsInferenceHTTPProbe012 ES-E2E-012 推理探针。
func RunMetricsInferenceHTTPProbe012(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName012, MetricsInferenceProbeJobName012)
}

// StartMetricsInferenceLoadTest012 ES-E2E-012 推理压测（制造 CPU 使用并产生推理流量）。
func StartMetricsInferenceLoadTest012(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName012, MetricsInferenceLoadTestJobName012)
}

// StopMetricsInferenceLoadTest012 停止 ES-E2E-012 推理压测。
func StopMetricsInferenceLoadTest012(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName012)
}

// RunMetricsInferenceHTTPProbe013 ES-E2E-013 推理探针。
func RunMetricsInferenceHTTPProbe013(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName013, MetricsInferenceProbeJobName013)
}

// StartMetricsInferenceLoadTest013 ES-E2E-013 推理压测（制造推理流量）。
func StartMetricsInferenceLoadTest013(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName013, MetricsInferenceLoadTestJobName013)
}

// StopMetricsInferenceLoadTest013 停止 ES-E2E-013 推理压测。
func StopMetricsInferenceLoadTest013(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName013)
}

// RunMetricsInferenceHTTPProbe014 ES-E2E-014 推理探针。
func RunMetricsInferenceHTTPProbe014(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName014, MetricsInferenceProbeJobName014)
}

// StartMetricsInferenceLoadTest014 ES-E2E-014 推理压测。
func StartMetricsInferenceLoadTest014(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName014, MetricsInferenceLoadTestJobName014)
}

// StopMetricsInferenceLoadTest014 停止 ES-E2E-014 推理压测。
func StopMetricsInferenceLoadTest014(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName014)
}

// RunMetricsInferenceHTTPProbe015 ES-E2E-015 推理探针。
func RunMetricsInferenceHTTPProbe015(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName015, MetricsInferenceProbeJobName015)
}

// StartMetricsInferenceLoadTest015 ES-E2E-015 推理压测。
func StartMetricsInferenceLoadTest015(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName015, MetricsInferenceLoadTestJobName015)
}

// StopMetricsInferenceLoadTest015 停止 ES-E2E-015 推理压测。
func StopMetricsInferenceLoadTest015(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName015)
}

// RunMetricsInferenceHTTPProbe017 ES-E2E-017 推理探针。
func RunMetricsInferenceHTTPProbe017(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	runMetricsInferenceHTTPProbe(clientset, ctx, namespace, MetricsInferenceSvcName017, MetricsInferenceProbeJobName017)
}

// StartMetricsInferenceLoadTest017 ES-E2E-017 推理压测（驱动 infer_qps 超阈值）。
func StartMetricsInferenceLoadTest017(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName017, MetricsInferenceLoadTestJobName017)
}

// StopMetricsInferenceLoadTest017 停止 ES-E2E-017 推理压测。
func StopMetricsInferenceLoadTest017(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName017)
}

// esMetricsHTTPClientImages 推理探针/压测 Job 使用的 HTTP 客户端镜像，按顺序兜底：
// 首选 nginx:alpine —— 本套件其它用例普遍使用它，环境里最可能已备好（该镜像自带 curl，Alpine/BusyBox 变体也有 wget）；
// 备选 curlimages/curl（官方 curl 镜像）、busybox。离线或单节点环境拉不到某个镜像时，不再直接超时失败。
var esMetricsHTTPClientImages = []string{"nginx:alpine", "curlimages/curl:8.5.0", "busybox:latest"}

// httpPostShell curl/wget 自适应的 POST 片段：调用 http_post <url> 输出 HTTP 状态码（200 成功，其余输出 000）。
const httpPostShell = `if command -v curl >/dev/null 2>&1; then
  http_post() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 -X POST "$1" -H "Content-Type: application/json" -d '{}' || echo 000; }
elif command -v wget >/dev/null 2>&1; then
  http_post() { if wget -q -O /dev/null -T 5 --post-data='{}' --header="Content-Type: application/json" "$1" >/dev/null 2>&1; then echo 200; else echo 000; fi; }
else
  echo "image has neither curl nor wget" >&2
  exit 3
fi
`

// createMetricsInferenceJob 用指定镜像创建探针/压测 Job（先删除同名 Job，保证幂等重试）。
func createMetricsInferenceJob(clientset *kubernetes.Clientset, ctx context.Context, namespace, jobName, containerName, image, script string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, jobName)
	backoff := int32(0)
	job := &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{Name: jobName, Namespace: namespace},
		Spec: batchv1.JobSpec{
			BackoffLimit: &backoff,
			Template: corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{
					RestartPolicy: corev1.RestartPolicyNever,
					Containers: []corev1.Container{{
						Name:    containerName,
						Image:   image,
						Command: []string{"sh", "-c", script},
					}},
				},
			},
		},
	}
	_, err := clientset.BatchV1().Jobs(namespace).Create(ctx, job, metav1.CreateOptions{})
	Expect(err).NotTo(HaveOccurred())
}

// waitForMetricsInferenceJobOutcome 等待 Job 达成目标状态：wantSucceeded=true 时等 Job 成功，
// 否则等 Pod Running。返回 (是否达成, 是否疑似镜像拉取问题)；疑似镜像问题时立即返回，便于调用方换兜底镜像。
func waitForMetricsInferenceJobOutcome(clientset *kubernetes.Clientset, ctx context.Context, namespace, jobName string, wantSucceeded bool, timeout time.Duration) (bool, bool) {
	deadline := time.Now().Add(timeout)
	for {
		if wantSucceeded {
			if j, err := clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{}); err == nil && j.Status.Succeeded >= 1 {
				return true, false
			}
		}
		pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + jobName})
		if err == nil {
			for i := range pods.Items {
				pod := pods.Items[i]
				if !wantSucceeded && pod.Status.Phase == corev1.PodRunning {
					return true, false
				}
				if reason, ok := podImagePullProblem(pod); ok {
					GinkgoWriter.Printf("Job %s/%s 的 Pod %s 镜像拉取异常：%s\n", namespace, jobName, pod.Name, reason)
					return false, true
				}
			}
		}
		if !time.Now().Before(deadline) {
			return false, false
		}
		time.Sleep(5 * time.Second)
	}
}

// podImagePullProblem 判断 Pod 是否卡在镜像拉取（返回可读原因）。
func podImagePullProblem(pod corev1.Pod) (string, bool) {
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.State.Waiting == nil {
			continue
		}
		switch cs.State.Waiting.Reason {
		case "ImagePullBackOff", "ErrImagePull", "InvalidImageName":
			return fmt.Sprintf("reason=%s message=%s", cs.State.Waiting.Reason, cs.State.Waiting.Message), true
		}
	}
	return "", false
}

// describeMetricsInferenceJob 汇总 Job 状态与 Pod 状态、容器等待原因、日志，用于失败时定位原因。
func describeMetricsInferenceJob(clientset *kubernetes.Clientset, ctx context.Context, namespace, jobName string) string {
	var b strings.Builder
	if j, err := clientset.BatchV1().Jobs(namespace).Get(ctx, jobName, metav1.GetOptions{}); err == nil {
		fmt.Fprintf(&b, "  job %s/%s active=%d succeeded=%d failed=%d\n", namespace, jobName, j.Status.Active, j.Status.Succeeded, j.Status.Failed)
	} else {
		fmt.Fprintf(&b, "  job %s/%s get failed: %v\n", namespace, jobName, err)
	}
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: "job-name=" + jobName})
	if err != nil {
		fmt.Fprintf(&b, "  pods list failed: %v\n", err)
		return b.String()
	}
	for i := range pods.Items {
		pod := pods.Items[i]
		fmt.Fprintf(&b, "  pod %s phase=%s node=%s\n", pod.Name, pod.Status.Phase, pod.Spec.NodeName)
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
		if logs, err := clientset.CoreV1().Pods(namespace).GetLogs(pod.Name, &corev1.PodLogOptions{}).DoRaw(ctx); err == nil && len(logs) > 0 {
			fmt.Fprintf(&b, "    logs: %s\n", tailString(string(logs), 2000))
		}
	}
	if events, err := clientset.CoreV1().Events(namespace).List(ctx, metav1.ListOptions{}); err == nil {
		shown := 0
		for i := len(events.Items) - 1; i >= 0 && shown < 8; i-- {
			e := events.Items[i]
			if e.InvolvedObject.Kind != "Pod" || !strings.HasPrefix(e.InvolvedObject.Name, jobName+"-") {
				continue
			}
			fmt.Fprintf(&b, "  event %s %s: %s\n", e.LastTimestamp.Format(time.RFC3339), e.Reason, e.Message)
			shown++
		}
	}
	return b.String()
}

func runMetricsInferenceHTTPProbe(clientset *kubernetes.Clientset, ctx context.Context, namespace, svcName, jobName string) {
	WaitForServiceEndpointsReady(clientset, ctx, namespace, svcName, 2*time.Minute)

	probeURL := metricsInferenceServiceURLFor(namespace, svcName) + "/v1/chat/completions"
	probeScript := httpPostShell + fmt.Sprintf(`i=0
while [ $i -lt 30 ]; do
  code=$(http_post %q)
  if [ "$code" = "200" ]; then echo "probe ok"; exit 0; fi
  i=$((i+1))
  sleep 2
done
exit 1`, probeURL)

	lastImage := ""
	for _, image := range esMetricsHTTPClientImages {
		lastImage = image
		createMetricsInferenceJob(clientset, ctx, namespace, jobName, "probe", image, probeScript)
		ok, imageProblem := waitForMetricsInferenceJobOutcome(clientset, ctx, namespace, jobName, true, 90*time.Second)
		if ok {
			GinkgoWriter.Printf("推理服务探针 HTTP 200 验证通过（镜像 %s）\n", image)
			return
		}
		GinkgoWriter.Printf("推理探针 Job 使用镜像 %s 未成功（疑似镜像拉取问题=%v），诊断信息：\n%s\n",
			image, imageProblem, describeMetricsInferenceJob(clientset, ctx, namespace, jobName))
	}
	Fail(fmt.Sprintf("推理探针 Job 应成功（HTTP 200），依次尝试镜像 %v 后仍失败（最后使用 %s），诊断信息：\n%s",
		esMetricsHTTPClientImages, lastImage, describeMetricsInferenceJob(clientset, ctx, namespace, jobName)))
}

// StartMetricsInferenceLoadTest 启动持续推理压测 Job。
func StartMetricsInferenceLoadTest(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName, MetricsInferenceLoadTestJobName)
}

// StartMetricsInferenceLoadTest010 ES-E2E-010 推理压测（制造 CPU 使用）。
func StartMetricsInferenceLoadTest010(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	startMetricsInferenceLoadTest(clientset, ctx, namespace, MetricsInferenceSvcName010, MetricsInferenceLoadTestJobName010)
}

func startMetricsInferenceLoadTest(clientset *kubernetes.Clientset, ctx context.Context, namespace, svcName, jobName string) {
	WaitForServiceEndpointsReady(clientset, ctx, namespace, svcName, 2*time.Minute)

	probeURL := metricsInferenceServiceURLFor(namespace, svcName) + "/v1/chat/completions"
	script := httpPostShell + fmt.Sprintf(`while true; do http_post %q >/dev/null 2>&1; sleep 0.2; done`, probeURL)

	lastImage := ""
	for _, image := range esMetricsHTTPClientImages {
		lastImage = image
		createMetricsInferenceJob(clientset, ctx, namespace, jobName, "load-test", image, script)
		ok, imageProblem := waitForMetricsInferenceJobOutcome(clientset, ctx, namespace, jobName, false, 90*time.Second)
		if ok {
			GinkgoWriter.Printf("推理压测已启动（镜像 %s）\n", image)
			return
		}
		GinkgoWriter.Printf("推理压测 Job 使用镜像 %s 未 Running（疑似镜像拉取问题=%v），诊断信息：\n%s\n",
			image, imageProblem, describeMetricsInferenceJob(clientset, ctx, namespace, jobName))
	}
	Fail(fmt.Sprintf("推理压测 Job Pod 应 Running，依次尝试镜像 %v 后仍失败（最后使用 %s），诊断信息：\n%s",
		esMetricsHTTPClientImages, lastImage, describeMetricsInferenceJob(clientset, ctx, namespace, jobName)))
}

// StopMetricsInferenceLoadTest 停止推理压测 Job。
func StopMetricsInferenceLoadTest(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName)
}

// StopMetricsInferenceLoadTest010 停止 ES-E2E-010 推理压测。
func StopMetricsInferenceLoadTest010(clientset *kubernetes.Clientset, ctx context.Context, namespace string) {
	deleteMetricsInferenceJob(clientset, ctx, namespace, MetricsInferenceLoadTestJobName010)
}

// WaitForESApaResourceMetricScaling ES-E2E-010：等待 Resource/cpu 采集成功且 apa 输出与 Deployment 对齐。
func WaitForESApaResourceMetricScaling(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, minReplicas, maxReplicas int32, recovery *ResourceMetricsWaitRecovery, timeout time.Duration) {
	VerifyESResourceMetricsWithPodListSuccess(dynamicClient, ctx, esName, namespace, recovery)
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
		"ES %s/%s Resource/cpu 路径 apa 输出应与 Deployment %s 对齐", namespace, esName, deployName)
}

func apaDesiredReplicasAligned(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, expectedDesired int32) bool {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
	if err != nil {
		return false
	}
	desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas")
	if !found || desired != int64(expectedDesired) {
		_, _, msg, _, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
		GinkgoWriter.Printf("ES desiredReplicas=%v，等待 = %d（ScalingCalculated: %s）\n", desired, expectedDesired, msg)
		return false
	}
	status, reason, msg, ok, err := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
	if err != nil || !ok || status != "True" || reason != MetricsReasonScalingSucceeded {
		GinkgoWriter.Printf("ScalingCalculated 未就绪: status=%s reason=%s msg=%s\n", status, reason, msg)
		return false
	}
	dep, err := clientset.AppsV1().Deployments(namespace).Get(ctx, deployName, metav1.GetOptions{})
	if err != nil {
		return false
	}
	if dep.Spec.Replicas == nil || *dep.Spec.Replicas != expectedDesired {
		GinkgoWriter.Printf("Deployment spec.replicas=%v != desired=%d\n", dep.Spec.Replicas, expectedDesired)
		return false
	}
	if dep.Status.ReadyReplicas < expectedDesired {
		GinkgoWriter.Printf("Deployment readyReplicas=%d < desired=%d\n", dep.Status.ReadyReplicas, expectedDesired)
		return false
	}
	return true
}

// VerifyESApaDesiredReplicasImmediate 单次校验 apa desiredReplicas 与 Deployment 已对齐（有序套件紧接上一用例扩容后使用，避免重复 5 分钟 Eventually）。
func VerifyESApaDesiredReplicasImmediate(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, expectedDesired int32) {
	Expect(apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, expectedDesired)).To(BeTrue(),
		"ES %s/%s desiredReplicas 应为 %d 且 Deployment %s 对齐", namespace, esName, expectedDesired, deployName)
}

// WaitForESApaDesiredReplicas 等待 apa 路径达到指定 desiredReplicas 且 Deployment 对齐。
func WaitForESApaDesiredReplicas(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, expectedDesired int32, verifyMetricsPipeline bool, timeout time.Duration) {
	if verifyMetricsPipeline {
		VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)
	}
	Eventually(func() bool {
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, expectedDesired)
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s desiredReplicas 应为 %d 且 Deployment %s 对齐", namespace, esName, expectedDesired, deployName)
}

// WaitForESApaScaleUp 等待 apa 路径扩容：desiredReplicas 在 (minReplicas, maxReplicas] 且 Deployment 对齐。
func WaitForESApaScaleUp(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, minReplicas, maxReplicas int32, timeout time.Duration) {
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)
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
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, int32(desired))
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 应完成 apa 扩容且 Deployment %s 副本对齐", namespace, esName, deployName)
}

// WaitForESApaPromQLPatchObserved 等待 Patch PromQL 后 MetricsPipeline / ScalingCalculated generation 对齐（确认 controller 已用新 query Reconcile）。
func WaitForESApaPromQLPatchObserved(dynamicClient dynamic.Interface, ctx context.Context, esName, namespace string) {
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err != nil {
			return false
		}
		gen, _, _ := unstructured.NestedInt64(obj.Object, "metadata", "generation")
		mp := getESMetricsPipelineDetail(obj)
		if !mp.Found || mp.ObservedGeneration < gen {
			GinkgoWriter.Printf("ES %s/%s Patch 后 MetricsPipeline observedGeneration=%d < generation=%d，等待 Reconcile\n",
				namespace, esName, mp.ObservedGeneration, gen)
			return false
		}
		for _, condType := range []string{"ScalingCalculated", "MetricsPipeline"} {
			_, _, _, ok, err := ESCondition(dynamicClient, ctx, esName, namespace, condType)
			if err != nil || !ok {
				GinkgoWriter.Printf("ES %s/%s Patch 后 %s 条件未就绪\n", namespace, esName, condType)
				return false
			}
		}
		return true
	}, 3*time.Minute, 5*time.Second).Should(BeTrue(),
		"ES %s/%s Patch PromQL 后应完成至少一轮 Reconcile", namespace, esName)
}

// WaitForESApaScaleDown 等待 apa 路径缩容至 minReplicas，并校验 MetricsPipeline 与 lastScaleTime。
// 依赖 controller 默认行为：scaleDown.stabilizationWindowSeconds=180、maxPodsPerStep=1（分步缩容，非一步到 min）。
// timeout<=0 时使用 MetricsApaScaleDownWaitTimeout()（默认 10m，覆盖 180s 稳定窗口 + 3→2→1 分步缩容）。
func WaitForESApaScaleDown(dynamicClient dynamic.Interface, clientset *kubernetes.Clientset, ctx context.Context, esName, namespace, deployName string, minReplicas int32, timeout time.Duration) {
	if timeout <= 0 {
		timeout = MetricsApaScaleDownWaitTimeout()
	}
	VerifyESGetMetricsFullSuccess(dynamicClient, ctx, esName, namespace, 1, true)

	var lastDesired int64 = -1
	Eventually(func() bool {
		obj, err := GetElasticScalerCR(dynamicClient, ctx, esName, namespace)
		if err == nil {
			if desired, found, _ := unstructured.NestedInt64(obj.Object, "status", "desiredReplicas"); found && desired != lastDesired {
				_, _, msg, _, _ := ESCondition(dynamicClient, ctx, esName, namespace, "ScalingCalculated")
				GinkgoWriter.Printf("ES desiredReplicas 变化 %d → %d（ScalingCalculated: %s）\n", lastDesired, desired, msg)
				lastDesired = desired
			}
		}
		return apaDesiredReplicasAligned(dynamicClient, clientset, ctx, esName, namespace, deployName, minReplicas)
	}, timeout, 10*time.Second).Should(BeTrue(),
		"ES %s/%s 应缩容至 minReplicas=%d（controller 默认 scaleDown 稳定窗口 180s、maxPodsPerStep=1）", namespace, esName, minReplicas)
	VerifyESLastScaleTimePresent(dynamicClient, ctx, esName, namespace)
}

// VerifyESLastScaleTimePresent 确认 ES status.lastScaleTime 已设置（发生实际扩缩容）。
func VerifyESLastScaleTimePresent(dynamicClient dynamic.Interface, ctx context.Context, name, namespace string) {
	obj, err := GetElasticScalerCR(dynamicClient, ctx, name, namespace)
	Expect(err).NotTo(HaveOccurred())
	lastScaleTime, found, _ := unstructured.NestedString(obj.Object, "status", "lastScaleTime")
	Expect(found).To(BeTrue(), "ES %s/%s status.lastScaleTime 应存在", namespace, name)
	Expect(strings.TrimSpace(lastScaleTime)).NotTo(BeEmpty(), "ES %s/%s status.lastScaleTime 不应为空", namespace, name)
}
