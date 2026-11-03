package utils

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

// PrometheusQueryResult Prometheus API 响应结构
type PrometheusQueryResult struct {
	Status string `json:"status"`
	Data   struct {
		ResultType string                   `json:"resultType"`
		Result     []map[string]interface{} `json:"result"`
	} `json:"data"`
	ErrorType string `json:"errorType,omitempty"`
	Error     string `json:"error,omitempty"`
}

// PrometheusTargetsResult Prometheus /api/v1/targets 响应结构
type PrometheusTargetsResult struct {
	Status string `json:"status"`
	Data   struct {
		ActiveTargets []PrometheusTarget `json:"activeTargets"`
	} `json:"data"`
}

type PrometheusTarget struct {
	DiscoveredLabels map[string]string `json:"discoveredLabels"`
	Labels           map[string]string `json:"labels"`
	ScrapePool       string            `json:"scrapePool"`
	ScrapeURL        string            `json:"scrapeUrl"`
	Health           string            `json:"health"` // "up" or "down"
	LastError        string            `json:"lastError"`
}

// GetPrometheusURL 获取Prometheus访问地址
func GetPrometheusURL(clientset kubernetes.Interface, ctx context.Context, promNamespace string) (string, error) {
	// 1. 优先环境变量
	if envURL := os.Getenv("PROMETHEUS_URL"); envURL != "" {
		return strings.TrimSuffix(envURL, "/"), nil
	}

	// 2. 从K8s Service发现（扫描所有namespace）
	svcs, err := clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list services: %w", err)
	}

	// 查找Prometheus Service（参考eagle-eye的严格过滤逻辑）
	for i := range svcs.Items {
		svc := &svcs.Items[i]

		// 必须包含"prometheus"但排除operator相关的service
		if !strings.Contains(svc.Name, "prometheus") ||
			strings.Contains(svc.Name, "operator") ||
			strings.Contains(svc.Name, "operated") ||
			svc.Spec.ClusterIP == "" ||
			svc.Spec.ClusterIP == "None" ||
			len(svc.Spec.Ports) == 0 {
			continue
		}

		// 如果指定了namespace，只查找该namespace
		if promNamespace != "" && svc.Namespace != promNamespace {
			continue
		}

		url := buildPrometheusURL(svc)
		if url != "" {
			return url, nil
		}
	}

	if promNamespace != "" {
		return "", fmt.Errorf("no prometheus service found in namespace %s", promNamespace)
	}
	return "", fmt.Errorf("no prometheus service found in cluster")
}

func buildPrometheusURL(svc *corev1.Service) string {
	if svc.Spec.ClusterIP == "" || svc.Spec.ClusterIP == "None" || len(svc.Spec.Ports) == 0 {
		return ""
	}
	// 直接使用第一个端口（参考eagle-eye实现）
	return fmt.Sprintf("http://%s:%d", svc.Spec.ClusterIP, svc.Spec.Ports[0].Port)
}

// QueryPrometheus 执行PromQL instant query（通过临时Pod在集群内查询）
func QueryPrometheus(clientset kubernetes.Interface, ctx context.Context, promURL, query string, timeout time.Duration) (*PrometheusQueryResult, error) {
	apiURL := fmt.Sprintf("%s/api/v1/query?query=%s&time=%d",
		promURL, url.QueryEscape(query), time.Now().Unix())

	// 使用临时Pod在集群内执行HTTP请求
	bodyStr := infernexHTTPGetInCluster(clientset, ctx, apiURL)
	if bodyStr == "" {
		return nil, fmt.Errorf("empty response from prometheus")
	}

	var result PrometheusQueryResult
	if err := json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	if result.Status != "success" {
		return nil, fmt.Errorf("prometheus query failed: %s - %s", result.ErrorType, result.Error)
	}

	return &result, nil
}

// VerifyMetricExists 验证指标是否存在
func VerifyMetricExists(clientset kubernetes.Interface, ctx context.Context, promURL, metricName string, labels map[string]string) error {
	var labelParts []string
	for k, v := range labels {
		labelParts = append(labelParts, fmt.Sprintf(`%s="%s"`, k, v))
	}

	query := metricName
	if len(labelParts) > 0 {
		query = fmt.Sprintf("%s{%s}", metricName, strings.Join(labelParts, ","))
	}

	result, err := QueryPrometheus(clientset, ctx, promURL, query, 30*time.Second)
	if err != nil {
		return err
	}

	if len(result.Data.Result) == 0 {
		return fmt.Errorf("metric %s with labels %v not found", metricName, labels)
	}

	return nil
}

// GetPrometheusTargets 获取Prometheus targets（通过临时Pod在集群内查询）
func GetPrometheusTargets(clientset kubernetes.Interface, ctx context.Context, promURL string) (*PrometheusTargetsResult, error) {
	apiURL := fmt.Sprintf("%s/api/v1/targets", promURL)

	// 使用临时Pod在集群内执行HTTP请求
	bodyStr := infernexHTTPGetInCluster(clientset, ctx, apiURL)
	if bodyStr == "" {
		return nil, fmt.Errorf("empty response from prometheus targets")
	}

	var result PrometheusTargetsResult
	if err := json.Unmarshal([]byte(bodyStr), &result); err != nil {
		return nil, fmt.Errorf("parse response: %w", err)
	}

	return &result, nil
}

// VerifyPrometheusTargetsUp 验证指定namespace和角色的targets状态为UP
func VerifyPrometheusTargetsUp(clientset kubernetes.Interface, ctx context.Context, promURL, namespace, role string) error {
	targets, err := GetPrometheusTargets(clientset, ctx, promURL)
	if err != nil {
		return err
	}

	found := false
	for _, target := range targets.Data.ActiveTargets {
		// 检查namespace和role标签
		if target.Labels["namespace"] == namespace &&
			strings.Contains(target.Labels["pod"], role) {
			found = true
			if target.Health != "up" {
				// 区分"还没up"和"查询失败"，使错误信息更友好
				if target.LastError != "" {
					return fmt.Errorf("target %s health=%s, last_error: %s",
						target.ScrapeURL, target.Health, target.LastError)
				}
				return fmt.Errorf("target %s health=%s (waiting for up)",
					target.ScrapeURL, target.Health)
			}
		}
	}

	if !found {
		return fmt.Errorf("no targets found for namespace=%s, role=%s", namespace, role)
	}

	return nil
}

// infernexHTTPGetInCluster performs an HTTP GET request from inside the cluster by launching a temporary
// alpine pod. This avoids ClusterIP routing issues when tests run outside the cluster.
// Returns the response body as a string, or empty string on error.
func infernexHTTPGetInCluster(clientset kubernetes.Interface, ctx context.Context, url string) string {
	podName := fmt.Sprintf("infernex-http-%d", time.Now().UnixNano())
	script := fmt.Sprintf(`wget -q -O- --timeout=30 '%s' 2>/dev/null || curl -s --max-time 30 '%s'`, url, url)

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: "default"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyNever,
			Containers: []corev1.Container{{
				Name:    "wget",
				Image:   "alpine:latest",
				Command: []string{"/bin/sh", "-c", script},
			}},
			Tolerations: []corev1.Toleration{{Operator: corev1.TolerationOpExists}},
		},
	}

	if _, err := clientset.CoreV1().Pods("default").Create(ctx, pod, metav1.CreateOptions{}); err != nil {
		return ""
	}
	defer func() {
		_ = clientset.CoreV1().Pods("default").Delete(ctx, podName, metav1.DeleteOptions{})
	}()

	// 等待Pod完成
	for i := 0; i < 120; i++ {
		p, err := clientset.CoreV1().Pods("default").Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			time.Sleep(time.Second)
			continue
		}
		if p.Status.Phase == corev1.PodSucceeded || p.Status.Phase == corev1.PodFailed {
			break
		}
		time.Sleep(time.Second)
	}

	// 获取Pod日志
	req := clientset.CoreV1().Pods("default").GetLogs(podName, &corev1.PodLogOptions{})
	logBytes, err := req.DoRaw(ctx)
	if err != nil {
		return ""
	}

	return string(logBytes)
}
