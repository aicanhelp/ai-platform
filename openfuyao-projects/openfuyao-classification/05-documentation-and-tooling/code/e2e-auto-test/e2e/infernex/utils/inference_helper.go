package utils

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "github.com/onsi/ginkgo/v2"
)

type TokenizeResponse struct {
	Tokens []int32 `json:"tokens"`
	Count  int     `json:"count"`
}

const defaultHermesInferencePath = "/v1/chat/completions"

// BuildInferenceRequestBody 构建 /v1/chat/completions 请求体（messages 格式）。
// stream 为 true 表示流式请求。message 为 user 消息正文；若为空则从 HERMES_INFERENCE_MESSAGE 读取，仍为空则使用默认短句。
func BuildInferenceRequestBody(stream bool, message string) map[string]any {
	model := os.Getenv("HERMES_INFERENCE_MODEL")
	if model == "" {
		model = "Qwen/Qwen3-8B"
	}
	content := strings.TrimSpace(message)
	if content == "" {
		content = os.Getenv("HERMES_INFERENCE_MESSAGE")
	}
	if content == "" {
		content = "Hello! How are you today"
	}
	return map[string]any{
		"model": model,
		"messages": []map[string]string{
			{
				"role":    "user",
				"content": content,
			},
		},
		"stream": stream,
	}
}

// BuildCompletionsRequestBody 从环境变量构建 /v1/completions 请求体（prompt 格式），VLLM 要求提供 prompt 或 prompt_embeds
func BuildCompletionsRequestBody(stream bool) map[string]any {
	model := os.Getenv("HERMES_INFERENCE_MODEL")
	if model == "" {
		model = "Qwen/Qwen3-8B"
	}
	prompt := os.Getenv("HERMES_INFERENCE_MESSAGE")
	if prompt == "" {
		prompt = "Hello! How are you today"
	}
	return map[string]any{
		"model":  model,
		"prompt": prompt,
		"stream": stream,
	}
}

// BuildTokenizeRequestBody 构建 vLLM /tokenize 请求体。
func BuildTokenizeRequestBody(prompt string) map[string]any {
	model := os.Getenv("HERMES_INFERENCE_MODEL")
	if model == "" {
		model = "Qwen/Qwen3-8B"
	}
	return map[string]any{
		"model":  model,
		"prompt": prompt,
	}
}

// ParseTokenizeResponse 解析 vLLM /tokenize 响应体。
func ParseTokenizeResponse(body []byte) (*TokenizeResponse, error) {
	var resp TokenizeResponse
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, fmt.Errorf("parse tokenize response failed: %v, body: %s", err, string(body))
	}
	if len(resp.Tokens) == 0 {
		return nil, fmt.Errorf("tokenize response does not contain tokens: %s", string(body))
	}
	return &resp, nil
}

// DoInferenceRequestWithTimeout 发送推理请求；timeout 含连接与读完整响应体。timeout<=0 时默认 30s。
func DoInferenceRequestWithTimeout(gatewayURL string, reqBody map[string]any, timeout time.Duration) (statusCode int, body []byte, err error) {
	if gatewayURL == "" {
		return 0, nil, fmt.Errorf("gateway url is empty")
	}
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return 0, nil, fmt.Errorf("marshal request body failed: %v", err)
	}
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequest(http.MethodPost, gatewayURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return 0, nil, fmt.Errorf("create http request failed: %v", err)
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return 0, nil, fmt.Errorf("send http request failed: %v", err)
	}
	defer resp.Body.Close()
	body, err = io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, nil, fmt.Errorf("read response body failed: %v", err)
	}
	return resp.StatusCode, body, nil
}

// DoInferenceRequest 仅发送推理请求，返回状态码与原始响应体（不校验内容）；HTTP 超时 30s。
func DoInferenceRequest(gatewayURL string, reqBody map[string]any) (statusCode int, body []byte, err error) {
	return DoInferenceRequestWithTimeout(gatewayURL, reqBody, 30*time.Second)
}

// ResolveInferenceURL resolves and caches the full Hermes inference endpoint URL.
func (env *HermesTestEnv) ResolveInferenceURL() (string, error) {
	if env == nil {
		return "", fmt.Errorf("hermes test env is nil")
	}
	if strings.TrimSpace(env.GatewayURL) != "" {
		return env.GatewayURL, nil
	}
	baseURL, err := ResolveGatewayURL(env.Ctx, env.Client, []string{env.Namespace, "ai-inference"})
	if err != nil {
		return "", err
	}
	env.GatewayURL = GatewayURLForPath(baseURL, defaultHermesInferencePath)
	return env.GatewayURL, nil
}

// DoInferenceRequestFromEnv resolves the Hermes inference URL on demand and
// sends a standard chat completions request.
func DoInferenceRequestFromEnv(env *HermesTestEnv, reqBody map[string]any) (statusCode int, body []byte, err error) {
	return DoInferenceRequestFromEnvWithTimeout(env, reqBody, 0)
}

// DoInferenceRequestFromEnvWithTimeout resolves the Hermes inference URL on demand and
// sends a chat completions request with an explicit HTTP timeout. timeout<=0 falls back
// to the 30s default. Use a larger timeout for long-prompt inference that is slow on NPU.
func DoInferenceRequestFromEnvWithTimeout(env *HermesTestEnv, reqBody map[string]any, timeout time.Duration) (statusCode int, body []byte, err error) {
	gatewayURL, err := env.ResolveInferenceURL()
	if err != nil {
		return 0, nil, err
	}
	return DoInferenceRequestWithTimeout(gatewayURL, reqBody, timeout)
}

// ValidateNonStreamResponse 校验非流式响应：状态码 200，JSON 且包含 choices
func ValidateNonStreamResponse(statusCode int, body []byte) error {
	if statusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d, body: %s", statusCode, string(body))
	}
	var respJSON map[string]any
	if err := json.Unmarshal(body, &respJSON); err != nil {
		return fmt.Errorf("parse response json failed: %v, body: %s", err, string(body))
	}
	if _, ok := respJSON["choices"]; !ok {
		return fmt.Errorf("response json does not contain 'choices': %s", string(body))
	}
	return nil
}

// ValidateStreamResponse 校验流式响应（SSE）：状态码 200，且至少有一条 data: {...} 含 choices/delta
func ValidateStreamResponse(statusCode int, body []byte) error {
	if statusCode != http.StatusOK {
		return fmt.Errorf("unexpected status code: %d, body: %s", statusCode, string(body))
	}
	lines := strings.Split(string(body), "\n")
	var gotValidChunk bool
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			continue
		}
		var chunk map[string]any
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}
		if _, ok := chunk["choices"]; ok {
			gotValidChunk = true
			break
		}
	}
	if !gotValidChunk {
		return fmt.Errorf("stream response did not contain any valid data chunk with 'choices': %s", string(body))
	}
	return nil
}

// ValidateResponse 根据 stream 调用对应的校验：stream=false 用 ValidateNonStreamResponse，stream=true 用 ValidateStreamResponse
func ValidateResponse(statusCode int, body []byte, stream bool) error {
	if stream {
		return ValidateStreamResponse(statusCode, body)
	}
	return ValidateNonStreamResponse(statusCode, body)
}

// DefaultStreamInferenceTimeout 流式推理 HTTP 超时（连接 + 读完整 SSE 响应体）。
const DefaultStreamInferenceTimeout = 120 * time.Second

// VerifyStreamChatCompletionsInference 解析网关地址，发送流式 /v1/chat/completions 并校验 SSE 响应。
// message 为 user 消息正文；model 非空时覆盖请求体中的 model，否则使用 HERMES_INFERENCE_MODEL 或默认 Qwen3-8B；
// timeout<=0 时使用 DefaultStreamInferenceTimeout。
func VerifyStreamChatCompletionsInference(ctx context.Context, k8sClient K8SClientInterface, namespace, message, model string, timeout time.Duration) error {
	if timeout <= 0 {
		timeout = DefaultStreamInferenceTimeout
	}
	baseURL, err := ResolveGatewayURL(ctx, k8sClient, []string{namespace})
	if err != nil {
		return fmt.Errorf("resolve gateway url: %w", err)
	}
	gatewayURL := GatewayURLForPath(baseURL, "/v1/chat/completions")
	reqBody := BuildInferenceRequestBody(true, message)
	if strings.TrimSpace(model) != "" {
		reqBody["model"] = model
	}
	// CSV 只要求收到分片并收到 done；不限制 max_tokens 时引擎可能长生成，120s 读 body 会超时。
	if _, ok := reqBody["max_tokens"]; !ok {
		reqBody["max_tokens"] = 32
	}
	statusCode, body, err := DoInferenceRequestWithTimeout(gatewayURL, reqBody, timeout)
	if err != nil {
		return err
	}
	return ValidateResponse(statusCode, body, true)
}

// GatewayURLForPath 用 ResolveGatewayURL 返回的 base URL（仅 scheme+host+port）与 path 拼成完整请求 URL，path 如 /v1/chat/completions 或 /v1/completions
func GatewayURLForPath(baseURL string, path string) string {
	return strings.TrimSuffix(baseURL, "/") + path
}

// ResolveGatewayURL 若未显式设置 HERMES_GATEWAY_URL，则从集群探测网关 Service，返回仅含 scheme+host+port 的 base URL，路径由调用方用 GatewayURLForPath(baseURL, path) 拼装
func ResolveGatewayURL(ctx context.Context, k8sClient K8SClientInterface, namespaces []string) (string, error) {
	// 显式指定则归一化为 base（去掉 /v1/... 路径以便统一用 GatewayURLForPath 拼装）
	if v := os.Getenv("HERMES_GATEWAY_URL"); v != "" {
		if idx := strings.Index(v, "/v1/"); idx != -1 {
			return strings.TrimSuffix(v[:idx], "/"), nil
		}
		return strings.TrimSuffix(v, "/"), nil
	}

	type candidate struct {
		url  string
		hint string
	}

	serviceCandidate := func(ns string, svc *corev1.Service) (*candidate, error) {
		p := PickServicePort(svc)
		if p == nil {
			return nil, nil
		}
		scheme := "http"
		if p.Port == 443 || strings.Contains(strings.ToLower(p.Name), "https") {
			scheme = "https"
		}

		switch svc.Spec.Type {
		case corev1.ServiceTypeLoadBalancer:
			if len(svc.Status.LoadBalancer.Ingress) > 0 {
				host := svc.Status.LoadBalancer.Ingress[0].IP
				if host == "" {
					host = svc.Status.LoadBalancer.Ingress[0].Hostname
				}
				if host != "" {
					return &candidate{
						url:  fmt.Sprintf("%s://%s:%d", scheme, host, p.Port),
						hint: fmt.Sprintf("svc=%s/%s type=LoadBalancer", ns, svc.Name),
					}, nil
				}
			}
		case corev1.ServiceTypeNodePort:
			if p.NodePort != 0 {
				nodeIP, err := ChooseNodeIPForNodePort(ctx, k8sClient)
				if err != nil {
					return nil, err
				}
				return &candidate{
					url:  fmt.Sprintf("http://%s:%d", nodeIP, p.NodePort),
					hint: fmt.Sprintf("svc=%s/%s type=NodePort (nodeIP=%s)", ns, svc.Name, nodeIP),
				}, nil
			}
		default:
			if svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != "None" {
				return &candidate{
					url:  fmt.Sprintf("%s://%s:%d", scheme, svc.Spec.ClusterIP, p.Port),
					hint: fmt.Sprintf("svc=%s/%s type=%s (clusterIP=%s)", ns, svc.Name, svc.Spec.Type, svc.Spec.ClusterIP),
				}, nil
			}
		}

		return nil, nil
	}

	// 1) 优先使用传入命名空间中的 inference-gateway-istio；若未找到，再兜底 ai-inference
	{
		preferNamespaces := append([]string{}, namespaces...)
		foundAIInference := false
		for _, ns := range preferNamespaces {
			if ns == "ai-inference" {
				foundAIInference = true
				break
			}
		}
		if !foundAIInference {
			preferNamespaces = append(preferNamespaces, "ai-inference")
		}

		const preferName = "inference-gateway-istio"
		for _, preferNS := range preferNamespaces {
			svcs, err := k8sClient.ListServices(ctx, preferNS, metav1.ListOptions{})
			if err != nil {
				continue
			}
			for _, svc := range svcs.Items {
				if svc.Name != preferName {
					continue
				}

				cand, err := serviceCandidate(preferNS, &svc)
				if err != nil {
					return "", err
				}
				if cand != nil {
					GinkgoWriter.Printf("use inference-gateway-istio Service: %s (%s)\n", cand.url, cand.hint)
					return cand.url, nil
				}
			}
		}
	}

	var cands []candidate

	for _, ns := range namespaces {
		svcs, err := k8sClient.ListServices(ctx, ns, metav1.ListOptions{})
		if err != nil {
			continue
		}
		for _, svc := range svcs.Items {
			name := svc.Name
			// 尽量避免误选 istio-system 里的基础服务；优先含 gateway/router/infernex/hermes
			lname := strings.ToLower(name)
			if !(strings.Contains(lname, "gateway") || strings.Contains(lname, "router") || strings.Contains(lname, "infernex") || strings.Contains(lname, "hermes")) {
				continue
			}
			if lname == "istiod" || strings.Contains(lname, "prometheus") || strings.Contains(lname, "grafana") {
				continue
			}

			cand, err := serviceCandidate(ns, &svc)
			if err != nil {
				continue
			}
			if cand != nil {
				cands = append(cands, *cand)
			}
		}
	}

	if len(cands) == 0 {
		return "", fmt.Errorf("HERMES_GATEWAY_URL is empty and no gateway Service candidate found; set HERMES_GATEWAY_URL explicitly")
	}

	GinkgoWriter.Printf("gateway base url candidates (first used): %s (%s)\n", cands[0].url, cands[0].hint)
	return cands[0].url, nil
}
