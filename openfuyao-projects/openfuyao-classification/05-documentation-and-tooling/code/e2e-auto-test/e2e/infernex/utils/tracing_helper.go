package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Jaeger API 数据结构定义

// JaegerTrace 表示 Jaeger API 返回的 trace 结构
type JaegerTrace struct {
	TraceID   string                   `json:"traceID"`
	Spans     []JaegerSpan             `json:"spans"`
	Processes map[string]JaegerProcess `json:"processes"`
}

type JaegerSpan struct {
	TraceID       string            `json:"traceID"`
	SpanID        string            `json:"spanID"`
	OperationName string            `json:"operationName"`
	References    []JaegerReference `json:"references"`
	StartTime     int64             `json:"startTime"`
	Duration      int64             `json:"duration"`
	Tags          []JaegerTag       `json:"tags"`
	ProcessID     string            `json:"processID"`
	Process       JaegerProcess     `json:"-"` // 不从 JSON 解析，后续填充
}

type JaegerReference struct {
	RefType string `json:"refType"`
	TraceID string `json:"traceID"`
	SpanID  string `json:"spanID"`
}

type JaegerTag struct {
	Key   string      `json:"key"`
	Type  string      `json:"type"`
	Value interface{} `json:"value"`
}

type JaegerProcess struct {
	ServiceName string      `json:"serviceName"`
	Tags        []JaegerTag `json:"tags"`
}

type JaegerAPIResponse struct {
	Data   []JaegerTrace `json:"data"`
	Errors []interface{} `json:"errors,omitempty"`
}

// Jaeger 相关常量
const (
	JaegerQueryNodePort = 30686
)

// ResolveJaegerQueryURL 解析 Jaeger Query Service 的访问 URL（使用固定 NodePort 30686）
func ResolveJaegerQueryURL(ctx context.Context, k8sClient K8SClientInterface, namespace string) (string, error) {
	// Jaeger Query Service 使用固定 NodePort 30686
	nodeIP, err := ChooseNodeIPForNodePort(ctx, k8sClient)
	if err != nil {
		return "", fmt.Errorf("获取节点 IP 失败: %w", err)
	}
	if nodeIP == "" {
		return "", fmt.Errorf("未找到可用的节点 IP")
	}

	return fmt.Sprintf("http://%s:%d", nodeIP, JaegerQueryNodePort), nil
}

// QueryJaegerTraces 查询 Jaeger traces
func QueryJaegerTraces(ctx context.Context, k8sClient K8SClientInterface, namespace, serviceName string, startTime, endTime time.Time, limit int) ([]JaegerTrace, error) {
	baseURL, err := ResolveJaegerQueryURL(ctx, k8sClient, namespace)
	if err != nil {
		return nil, fmt.Errorf("解析 Jaeger Query URL 失败: %w", err)
	}

	apiURL := fmt.Sprintf("%s/api/traces", strings.TrimSuffix(baseURL, "/"))

	// 构建查询参数
	params := url.Values{}
	params.Set("service", serviceName)
	params.Set("start", fmt.Sprintf("%d", startTime.UnixMicro())) // Jaeger 使用微秒时间戳
	params.Set("end", fmt.Sprintf("%d", endTime.UnixMicro()))
	params.Set("limit", fmt.Sprintf("%d", limit))

	fullURL := fmt.Sprintf("%s?%s", apiURL, params.Encode())

	req, err := http.NewRequestWithContext(ctx, "GET", fullURL, nil)
	if err != nil {
		return nil, fmt.Errorf("创建请求失败: %w", err)
	}

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("查询 Jaeger API 失败: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Jaeger API 返回状态码 %d: %s", resp.StatusCode, string(body))
	}

	var apiResp JaegerAPIResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("解析响应 JSON 失败: %w, 响应: %s", err, string(body))
	}

	if len(apiResp.Errors) > 0 {
		return nil, fmt.Errorf("Jaeger API 返回错误: %v", apiResp.Errors)
	}

	// 填充每个 span 的 Process 字段（通过 processID 查找）
	for i := range apiResp.Data {
		trace := &apiResp.Data[i]
		for j := range trace.Spans {
			span := &trace.Spans[j]
			if process, ok := trace.Processes[span.ProcessID]; ok {
				span.Process = process
			}
		}
	}

	return apiResp.Data, nil
}

// FindSpanByOperationName 在 trace 中查找指定 operation name 的 span
func FindSpanByOperationName(trace *JaegerTrace, operationName string) *JaegerSpan {
	for i := range trace.Spans {
		if trace.Spans[i].OperationName == operationName {
			return &trace.Spans[i]
		}
	}
	return nil
}

// GetSpanTag 获取 span 的指定 tag
func GetSpanTag(span *JaegerSpan, key string) (interface{}, bool) {
	for _, tag := range span.Tags {
		if tag.Key == key {
			return tag.Value, true
		}
	}
	return nil, false
}

// GetSpanTagFloat 获取 span 的 float 类型 tag
func GetSpanTagFloat(span *JaegerSpan, key string) (float64, bool) {
	val, ok := GetSpanTag(span, key)
	if !ok {
		return 0, false
	}

	// 处理不同的数值类型
	switch v := val.(type) {
	case float64:
		return v, true
	case float32:
		return float64(v), true
	case int:
		return float64(v), true
	case int64:
		return float64(v), true
	default:
		return 0, false
	}
}

// GetSpanTagString 获取 span 的 string 类型 tag
func GetSpanTagString(span *JaegerSpan, key string) (string, bool) {
	val, ok := GetSpanTag(span, key)
	if !ok {
		return "", false
	}

	if str, ok := val.(string); ok {
		return str, true
	}
	return "", false
}

// IsChildSpan 检查 childSpan 是否是 parentSpan 的子 span
func IsChildSpan(childSpan, parentSpan *JaegerSpan) bool {
	for _, ref := range childSpan.References {
		if ref.RefType == "CHILD_OF" && ref.SpanID == parentSpan.SpanID {
			return true
		}
	}
	return false
}

// ============================================================================
// 推理请求验证相关函数
// ============================================================================

// VerifyInferenceRequestWithOptions 发送推理请求，支持自定义 prompt 和 max_tokens
func VerifyInferenceRequestWithOptions(ctx context.Context, k8sClient K8SClientInterface, targetNS, model, prompt string, maxTokens int) error {
	baseURL, err := ResolveInferenceURL(ctx, k8sClient, targetNS)
	if err != nil {
		return fmt.Errorf("解析推理服务 URL 失败: %w", err)
	}

	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"

	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": prompt},
		},
		"stream": false,
	}
	if maxTokens > 0 {
		reqBody["max_tokens"] = maxTokens
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求体失败: %w", err)
	}

	client := &http.Client{Timeout: 300 * time.Second} // 增加到 5 分钟，适应长 prompt
	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送请求失败: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("期望状态码 200，实际 %d，响应: %s", resp.StatusCode, string(respBody))
	}

	var respJSON map[string]any
	if err := json.Unmarshal(respBody, &respJSON); err != nil {
		return fmt.Errorf("解析响应 JSON 失败: %w，响应: %s", err, string(respBody))
	}
	if _, ok := respJSON["choices"]; !ok {
		return fmt.Errorf("响应不包含 'choices' 字段: %s", string(respBody))
	}

	return nil
}

// VerifyInferenceRequestWithPrompt 发送推理请求，支持自定义 prompt
func VerifyInferenceRequestWithPrompt(ctx context.Context, k8sClient K8SClientInterface, targetNS, model, prompt string) error {
	return VerifyInferenceRequestWithOptions(ctx, k8sClient, targetNS, model, prompt, 0)
}

// VerifyStreamingInferenceRequest 发送流式推理请求并验证响应
func VerifyStreamingInferenceRequest(ctx context.Context, k8sClient K8SClientInterface, targetNS, model string) error {
	baseURL, err := ResolveInferenceURL(ctx, k8sClient, targetNS)
	if err != nil {
		return fmt.Errorf("解析推理服务 URL 失败: %w", err)
	}

	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"

	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": "写一段100字的故事"},
		},
		"max_tokens": 100,
		"stream":     true,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求体失败: %w", err)
	}

	client := &http.Client{Timeout: 300 * time.Second} // 增加到 5 分钟，适应长 prompt
	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送请求失败: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("期望状态码 200，实际 %d，响应: %s", resp.StatusCode, string(body))
	}

	// 读取完整个 SSE 流，确保 decode 阶段也完成并记录到 trace
	buf := make([]byte, 4096)
	dataChunkCount := 0
	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			chunk := string(buf[:n])
			// 统计包含 "data:" 的 chunk 数量
			dataChunkCount += strings.Count(chunk, "data:")
		}
		// 只在读取错误（EOF）时才退出
		if readErr != nil {
			break
		}
	}

	if dataChunkCount == 0 {
		return fmt.Errorf("流式响应未收到任何 'data:' chunk")
	}

	return nil
}

// ResolveInferenceURL 解析推理服务的访问 URL
func ResolveInferenceURL(ctx context.Context, k8sClient K8SClientInterface, namespace string) (string, error) {
	// 从 K8s Service 中自动探测
	svcList, err := k8sClient.ListServices(ctx, namespace, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list services in %s: %w", namespace, err)
	}

	// 优先查找 inference-gateway-istio Service (通过名称)
	for i := range svcList.Items {
		svc := &svcList.Items[i]
		if svc.Name == "inference-gateway-istio" {
			if url := buildServiceURL(ctx, k8sClient, svc); url != "" {
				return url, nil
			}
		}
	}

	// 回退查找 inferenceGateway Service (通过 label)
	for i := range svcList.Items {
		svc := &svcList.Items[i]
		if svc.Labels["app.kubernetes.io/name"] == "inferenceGateway" {
			if url := buildServiceURL(ctx, k8sClient, svc); url != "" {
				return url, nil
			}
		}
	}

	return "", fmt.Errorf("在命名空间 %s 中未找到推理网关 Service (inference-gateway-istio 或 app.kubernetes.io/name=inferenceGateway)", namespace)
}

// buildServiceURL 根据 Service 类型构建访问 URL
// 优先选择业务端口（排除 status-port、health 等非业务端口）
func buildServiceURL(ctx context.Context, k8sClient K8SClientInterface, svc *corev1.Service) string {
	skipPortNames := map[string]bool{"status-port": true, "health": true, "healthz": true, "readiness": true, "metrics": true, "monitor": true}

	// 第一轮：跳过非业务端口
	for _, sp := range svc.Spec.Ports {
		if skipPortNames[strings.ToLower(sp.Name)] {
			continue
		}
		if url := buildServiceURLForPort(ctx, k8sClient, svc, sp); url != "" {
			return url
		}
	}
	// 第二轮：兜底，返回第一个能构建的端口
	for _, sp := range svc.Spec.Ports {
		if url := buildServiceURLForPort(ctx, k8sClient, svc, sp); url != "" {
			return url
		}
	}
	return ""
}

// buildServiceURLForPort 为指定的 Service 端口构建访问 URL
func buildServiceURLForPort(ctx context.Context, k8sClient K8SClientInterface, svc *corev1.Service, sp corev1.ServicePort) string {
	scheme := "http"
	if sp.Port == 443 || strings.Contains(strings.ToLower(sp.Name), "https") {
		scheme = "https"
	}

	switch svc.Spec.Type {
	case corev1.ServiceTypeNodePort:
		if sp.NodePort != 0 {
			nodeIP, err := ChooseNodeIPForNodePort(ctx, k8sClient)
			if err == nil && nodeIP != "" {
				return fmt.Sprintf("%s://%s:%d", scheme, nodeIP, sp.NodePort)
			}
		}
	case corev1.ServiceTypeLoadBalancer:
		if len(svc.Status.LoadBalancer.Ingress) > 0 {
			host := svc.Status.LoadBalancer.Ingress[0].IP
			if host == "" {
				host = svc.Status.LoadBalancer.Ingress[0].Hostname
			}
			if host != "" {
				return fmt.Sprintf("%s://%s:%d", scheme, host, sp.Port)
			}
		}
	default:
		if svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != "None" {
			return fmt.Sprintf("%s://%s:%d", scheme, svc.Spec.ClusterIP, sp.Port)
		}
	}
	return ""
}
