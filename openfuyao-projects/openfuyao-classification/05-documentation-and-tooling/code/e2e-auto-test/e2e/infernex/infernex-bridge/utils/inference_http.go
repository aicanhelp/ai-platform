package utils

import (
	"bytes"
	"fmt"
	"hash/fnv"
	"io"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/gomega"
)

const (
	defaultInferenceModel    = "Qwen/Qwen2.5-0.5B"
	inferenceLogBodyMaxBytes = 65536
	inferencePortForwardBase = 28000
)

// HTTPExchange captures a logged request/response pair.
type HTTPExchange struct {
	StatusCode int
	Headers    http.Header
	Body       string
}

type inferenceCandidate struct {
	Name          string
	URL           string
	Cleanup       func()
	Access        string // port-forward, nodeport, loadbalancer, in-cluster-curl, status-url
	Backend       string // cluster Service target
	ClusterTarget string // in-cluster URL (no port-forward)
	ListenHost    string // client-visible host/IP
	ListenPort    int    // client-visible port
	Path          string
}

type gatewayService struct {
	Namespace string
	Name      string
	Port      int
}

// ChatMessage is one OpenAI chat/completions message.
type ChatMessage struct {
	Role    string
	Content string
}

// AssertInferenceChatCompletionsOK tries several cluster access paths and logs full request/response.
func AssertInferenceChatCompletionsOK(fitID, owner, userContent string, maxTokens int) {
	AssertInferenceChatCompletionsWithModel(fitID, owner, userContent, maxTokens, defaultInferenceModel)
}

// AssertInferenceChatCompletionsWithModel is like AssertInferenceChatCompletionsOK but uses an explicit model name.
func AssertInferenceChatCompletionsWithModel(fitID, owner, userContent string, maxTokens int, model string) {
	assertInferenceChatCompletionsMessages(fitID, owner, []ChatMessage{{Role: "user", Content: userContent}}, maxTokens, "", model)
}

// AssertInferenceChatCompletionsMessagesOK sends a multi-turn payload; logs request/response per candidate URL.
// phaseLabel is shown in phase banners (e.g. "mooncake KV pass 1"); empty uses default chat/completions label.
func AssertInferenceChatCompletionsMessagesOK(fitID, owner string, messages []ChatMessage, maxTokens int, phaseLabel string, httpTimeout ...time.Duration) {
	assertInferenceChatCompletionsMessages(fitID, owner, messages, maxTokens, phaseLabel, defaultInferenceModel, httpTimeout...)
}

func assertInferenceChatCompletionsMessages(fitID, owner string, messages []ChatMessage, maxTokens int, phaseLabel, model string, httpTimeout ...time.Duration) {
	if SkipInferenceE2E() {
		LogStep(fitID, "SKIP_INFERENCE_E2E=true — skip chat/completions")
		return
	}
	if maxTokens <= 0 {
		maxTokens = 32
	}
	if len(messages) == 0 {
		messages = []ChatMessage{{Role: "user", Content: "hello infernex e2e"}}
	}
	reqTimeout := 90 * time.Second
	if len(httpTimeout) > 0 && httpTimeout[0] > 0 {
		reqTimeout = httpTimeout[0]
	}

	ns := E2ENS()
	if strings.TrimSpace(model) == "" {
		model = defaultInferenceModel
	}
	payload, err := buildChatCompletionsPayload(messages, maxTokens, model)
	Expect(err).NotTo(HaveOccurred())

	phase := inferenceChatPhaseLabel(owner, phaseLabel)
	probeStart := time.Now()
	LogPhaseBegin(fitID, phase)
	defer func() {
		LogStep(fitID, "%s finished total_elapsed=%s", phase, time.Since(probeStart).Round(time.Millisecond))
		LogPhaseEnd(fitID, phase)
	}()

	logInferenceMessagesSummary(fitID, messages, maxTokens, len(payload))

	WaitProxyServerPDBackendsRegistered(fitID, ns, owner)

	LogE2EConnectivitySnapshot(fitID, ns, owner)

	candidates := buildInferenceCandidates(fitID, ns, owner)
	Expect(candidates).NotTo(BeEmpty(),
		"no gateway NodePort/LB endpoint for owner=%s (need node IP + gateway exposed port; set INFERENCE_E2E_NODE_IP and/or INFERNEX_GATEWAY_SVC=ns/svc; kubectl get svc -A -l gateway.envoyproxy.io/owning-gateway-name)",
		owner)
	LogStep(fitID, "inference candidates (%d):", len(candidates))
	for i, cand := range candidates {
		logInferenceCandidate(fitID, i+1, cand)
	}

	LogStep(fitID, "POST chat/completions model=%q max_tokens=%d payload_bytes=%d timeout=%s",
		model, maxTokens, len(payload), reqTimeout)

	var lastDetail string
	for i, cand := range candidates {
		LogStep(fitID, ">>> HTTP try %d/%d via %s url=%s", i+1, len(candidates), cand.Name, cand.URL)
		logInferenceHTTPRequest(fitID, cand, []byte(payload))
		reqStart := time.Now()
		ex, err := postJSONExchange(cand.URL, []byte(payload), reqTimeout)
		roundTrip := time.Since(reqStart).Round(time.Millisecond)
		if err != nil {
			lastDetail = fmt.Sprintf("%s: request err=%v", cand.Name, err)
			LogStep(fitID, "<<< HTTP try %d/%d FAILED via %s elapsed=%s err=%v", i+1, len(candidates), cand.Name, roundTrip, err)
			if cand.Cleanup != nil {
				cand.Cleanup()
			}
			continue
		}
		LogStep(fitID, "<<< HTTP RESPONSE via %s elapsed=%s status=%d body_bytes=%d", cand.Name, roundTrip, ex.StatusCode, len(ex.Body))
		logHTTPHeaders(fitID, "RESPONSE", ex.Headers)
		logHTTPMessage(fitID, "RESPONSE", "body: "+ex.Body)
		logInferenceResponsePreview(fitID, ex.Body)

		if ex.StatusCode >= 200 && ex.StatusCode < 300 && strings.TrimSpace(ex.Body) != "" &&
			!strings.Contains(strings.ToLower(ex.Body), `"error"`) {
			LogStep(fitID, "HTTP OK via %s status=%d body_len=%d round_trip=%s host=%s port=%d path=%s",
				cand.Name, ex.StatusCode, len(ex.Body), roundTrip, cand.ListenHost, cand.ListenPort, cand.Path)
			if cand.Cleanup != nil {
				defer cand.Cleanup()
			}
			return
		}
		lastDetail = fmt.Sprintf("%s: status=%d body_len=%d", cand.Name, ex.StatusCode, len(ex.Body))
		LogStep(fitID, "HTTP not acceptable via %s: status=%d (want 2xx, non-empty, no error JSON) elapsed=%s",
			cand.Name, ex.StatusCode, roundTrip)
		if cand.Cleanup != nil {
			cand.Cleanup()
		}
	}
	Expect(false).To(BeTrue(), "all inference paths failed for owner=%s; last=%s; tried %d candidates", owner, lastDetail, len(candidates))
}

func inferenceChatPhaseLabel(owner, phaseLabel string) string {
	if strings.TrimSpace(phaseLabel) != "" {
		return phaseLabel
	}
	return fmt.Sprintf("chat/completions owner=%s", owner)
}

func logInferenceMessagesSummary(fitID string, messages []ChatMessage, maxTokens, payloadBytes int) {
	totalChars := 0
	for i, m := range messages {
		totalChars += len(m.Content)
		preview := m.Content
		if len(preview) > 160 {
			preview = preview[:160] + "..."
		}
		LogStep(fitID, "request message[%d] role=%s content_len=%d preview=%q", i, m.Role, len(m.Content), preview)
	}
	LogStep(fitID, "request summary: messages=%d total_content_chars=%d payload_bytes=%d max_tokens=%d",
		len(messages), totalChars, payloadBytes, maxTokens)
}

func logInferenceResponsePreview(fitID, body string) {
	preview := strings.TrimSpace(body)
	if len(preview) > 400 {
		preview = preview[:400] + "..."
	}
	if preview == "" {
		LogStep(fitID, "response preview: (empty body)")
		return
	}
	LogStep(fitID, "response preview: %s", preview)
}

func buildChatCompletionsPayload(messages []ChatMessage, maxTokens int, model string) (string, error) {
	if strings.TrimSpace(model) == "" {
		model = defaultInferenceModel
	}
	var parts []string
	for _, m := range messages {
		role := strings.TrimSpace(m.Role)
		if role == "" {
			role = "user"
		}
		content, err := jsonStringLiteral(m.Content)
		if err != nil {
			return "", err
		}
		parts = append(parts, fmt.Sprintf(`{"role":%q,"content":%s}`, role, content))
	}
	return fmt.Sprintf(`{"model":%q,"messages":[%s],"max_tokens":%d,"stream":false}`,
		model, strings.Join(parts, ","), maxTokens), nil
}

func jsonStringLiteral(s string) (string, error) {
	var b strings.Builder
	if err := jsonEscape(&b, s); err != nil {
		return "", err
	}
	return `"` + b.String() + `"`, nil
}

func jsonEscape(b *strings.Builder, s string) error {
	for _, r := range s {
		switch r {
		case '"', '\\':
			b.WriteByte('\\')
			b.WriteRune(r)
		case '\n':
			b.WriteString(`\n`)
		case '\r':
			b.WriteString(`\r`)
		case '\t':
			b.WriteString(`\t`)
		default:
			if r < 0x20 {
				return fmt.Errorf("unsupported control character in prompt: U+%04X", r)
			}
			b.WriteRune(r)
		}
	}
	return nil
}

func buildInferenceCandidates(fitID, ns, owner string) []inferenceCandidate {
	if override := strings.TrimSpace(os.Getenv("INFERENCE_E2E_URL")); override != "" {
		url := strings.ReplaceAll(override, "{owner}", owner)
		url = strings.ReplaceAll(url, "{ns}", ns)
		LogStep(fitID, "inference using INFERENCE_E2E_URL override: %s", url)
		return []inferenceCandidate{{Name: "env:INFERENCE_E2E_URL", URL: url}}
	}

	paths := discoverChatCompletionPaths(fitID, ns, owner)
	gatewaySvcs := discoverGatewayServices(fitID, ns, owner)
	gwNS, gwSvc := "", ""
	if len(gatewaySvcs) > 0 {
		gwNS, gwSvc = gatewaySvcs[0].Namespace, gatewaySvcs[0].Name
	}
	LogStep(fitID, "inference discovery (gateway e2e only): chat paths=%v gateway services=%s nodeIP=%s",
		paths, formatGatewayServices(gatewaySvcs), InferenceE2ENodeIP(fitID, gwNS, gwSvc))

	var out []inferenceCandidate
	seen := map[string]struct{}{}

	add := func(c inferenceCandidate) {
		if _, ok := seen[c.URL]; ok {
			return
		}
		seen[c.URL] = struct{}{}
		out = append(out, c)
	}

	// 1) LLMISVC status 上的公网/节点可达 URL（排除 cluster.local）
	for _, u := range discoverLLMInferenceServiceURLs(fitID, ns, owner, paths) {
		if isExternalInferenceURL(u) {
			add(enrichCandidateFromURL(inferenceCandidate{Name: "llmisvc-status-url", URL: u, Access: "llmisvc-status-external"}))
		}
	}

	// 2) 本机/节点 IP + 网关 Service 暴露的 NodePort 或 LoadBalancer（不经 port-forward、不直连 workload）
	for _, gs := range gatewaySvcs {
		for _, path := range paths {
			for _, c := range gatewayExternalAccessURLs(fitID, gs.Namespace, gs.Name, path) {
				add(c)
			}
		}
	}

	// 3) 调试兜底：仅当显式 INFERENCE_ALLOW_WORKLOAD_DIRECT=true
	if AllowWorkloadDirectInference() {
		LogStep(fitID, "INFERENCE_ALLOW_WORKLOAD_DIRECT=true — adding workload direct candidates (not e2e gateway path)")
		workloadSvc := owner + "-kserve-workload-svc"
		const workloadPort = 8000
		directPath := "/v1/chat/completions"
		if _, ok := serviceExists(fitID, ns, workloadSvc); ok {
			if c, ok := portForwardCandidate(fitID, ns, workloadSvc, workloadPort, directPath, "port-forward-workload"); ok {
				add(c)
			}
		}
	}

	return out
}

func enrichCandidateFromURL(c inferenceCandidate) inferenceCandidate {
	if strings.HasPrefix(c.URL, "incluster://") {
		if c.ClusterTarget == "" {
			c.ClusterTarget = strings.TrimPrefix(c.URL, "incluster://")
		}
		if u, err := url.Parse(c.ClusterTarget); err == nil {
			c.ListenHost = u.Hostname()
			c.ListenPort = portFromHost(u.Host)
			c.Path = u.Path
		}
		return c
	}
	u, err := url.Parse(c.URL)
	if err != nil {
		return c
	}
	c.ListenHost = u.Hostname()
	c.ListenPort = portFromHost(u.Host)
	c.Path = u.Path
	return c
}

func portFromHost(host string) int {
	if host == "" {
		return 0
	}
	_, portStr, ok := strings.Cut(host, ":")
	if !ok || portStr == "" {
		if strings.Contains(host, "https") {
			return 443
		}
		return 80
	}
	p, _ := strconv.Atoi(portStr)
	return p
}

func logInferenceCandidate(fitID string, idx int, c inferenceCandidate) {
	LogStep(fitID, "  candidate[%d] %s access=%s", idx, c.Name, c.Access)
	LogStep(fitID, "    request host=%s port=%d path=%s", c.ListenHost, c.ListenPort, c.Path)
	LogStep(fitID, "    url=%s", c.URL)
	if c.Backend != "" {
		LogStep(fitID, "    backend=%s", c.Backend)
	}
	if c.ClusterTarget != "" {
		LogStep(fitID, "    cluster_target=%s", c.ClusterTarget)
	}
}

func formatGatewayServices(svcs []gatewayService) string {
	if len(svcs) == 0 {
		return "(none)"
	}
	var b strings.Builder
	for i, gs := range svcs {
		if i > 0 {
			b.WriteString(", ")
		}
		fmt.Fprintf(&b, "%s/%s:%d", gs.Namespace, gs.Name, gs.Port)
	}
	return b.String()
}

// discoverChatCompletionPaths reads PathPrefix from KServe or Bridge HTTPRoutes.
func discoverChatCompletionPaths(fitID, ns, owner string) []string {
	var paths []string
	for _, routeName := range []string{owner + "-kserve-route", owner + "-infernex-route"} {
		paths = append(paths, pathsFromHTTPRoute(fitID, ns, routeName)...)
	}
	paths = dedupeStrings(paths)
	if len(paths) == 0 {
		paths = []string{
			fmt.Sprintf("/%s/%s/v1/chat/completions", ns, owner),
			fmt.Sprintf("/kserve/%s/v1/chat/completions", owner),
		}
		LogStep(fitID, "no httproute paths for %s; defaults %v", owner, paths)
	}
	return paths
}

func pathsFromHTTPRoute(fitID, ns, routeName string) []string {
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get httproute %q -n %q --ignore-not-found -o jsonpath='{range .spec.rules[*].matches[*]}{.path.value}{\"\\n\"}{end}'",
		localKubeconfig(), routeName, ns)
	out, _, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		return nil
	}
	var paths []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line != "" && strings.Contains(line, "chat/completions") {
			paths = append(paths, ensureLeadingSlash(line))
		}
	}
	if len(paths) > 0 {
		LogStep(fitID, "httproute %s/%s paths: %v", ns, routeName, paths)
	}
	return paths
}

func listGatewayLabeledServices(fitID, gwNs, gwName string) []gatewayService {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get svc -n %q -l gateway.networking.k8s.io/gateway-name=%q -o jsonpath='{range .items[*]}{.metadata.name}{\" \"}{.spec.ports[0].port}{\"\\n\"}{end}'",
		kc, gwNs, gwName)
	out, stderr, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogStep(fitID, "gateway svc list ns=%s gateway=%s: empty (%v %s)", gwNs, gwName, err, stderr)
		return nil
	}
	var outSvcs []gatewayService
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		port, err := strconv.Atoi(fields[1])
		if err != nil || port <= 0 {
			continue
		}
		outSvcs = append(outSvcs, gatewayService{Namespace: gwNs, Name: fields[0], Port: port})
	}
	LogStep(fitID, "gateway services (ns=%s gateway=%s) => %v", gwNs, gwName, outSvcs)
	return outSvcs
}

func discoverLLMInferenceServiceURLs(fitID, ns, llmName string, chatPaths []string) []string {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl get llminferenceservices.serving.kserve.io %q -n %q --ignore-not-found -o jsonpath='{range .status.addresses[*]}{.url}{\"\\n\"}{end}'",
		kc, llmName, ns)
	out, _, _ := ExecLocal(cmd)
	var urls []string
	for _, line := range strings.Split(strings.TrimSpace(out), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if strings.Contains(line, "chat/completions") {
			urls = append(urls, line)
			continue
		}
		base := strings.TrimSuffix(line, "/")
		for _, p := range chatPaths {
			urls = append(urls, base+ensureLeadingSlash(p))
		}
	}
	if len(urls) > 0 {
		LogStep(fitID, "llminferenceservice %s/%s status URLs: %v", ns, llmName, urls)
	}
	return dedupeStrings(urls)
}

func kubectlJSONPathQuiet(kc, kubectlArgs string) string {
	cmd := fmt.Sprintf("KUBECONFIG=%q %s", kc, kubectlArgs)
	out, _, _ := ExecLocal(cmd)
	return out
}

func serviceExists(fitID, ns, svcName string) (int, bool) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get svc %q -n %q --ignore-not-found -o name", kc, svcName, ns)
	out, _, err := ExecLocal(cmd)
	return 0, err == nil && strings.TrimSpace(out) != ""
}

func servicePort(fitID, ns, svcName string) (int, bool) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get svc %q -n %q -o jsonpath='{.spec.ports[0].port}'", kc, svcName, ns)
	out, stderr, err := ExecLocal(cmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogStep(fitID, "svc %s/%s: no port (%v %s)", ns, svcName, err, stderr)
		return 0, false
	}
	p, err := strconv.Atoi(strings.TrimSpace(out))
	if err != nil || p <= 0 {
		return 0, false
	}
	return p, true
}

func firstNodeInternalIP(fitID string) string {
	kc := localKubeconfig()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type==\"InternalIP\")].address}'", kc)
	out, stderr, err := ExecLocal(cmd)
	if err != nil {
		LogStep(fitID, "get node IP: %v stderr=%s", err, stderr)
		return ""
	}
	return strings.TrimSpace(out)
}

func portForwardCandidate(fitID, ns, svcName string, remotePort int, path, name string) (inferenceCandidate, bool) {
	if _, ok := serviceExists(fitID, ns, svcName); !ok {
		return inferenceCandidate{}, false
	}
	localPort := inferenceLocalPort(svcName, ns)
	stop, err := PortForwardService(ns, svcName, localPort, remotePort)
	if err != nil {
		LogStep(fitID, "port-forward %s/%s:%d failed: %v", ns, svcName, remotePort, err)
		return inferenceCandidate{}, false
	}
	url := fmt.Sprintf("http://127.0.0.1:%d%s", localPort, path)
	clusterHost := fmt.Sprintf("%s.%s.svc.cluster.local", svcName, ns)
	return inferenceCandidate{
		Name:          name,
		URL:           url,
		Cleanup:       stop,
		Access:        "port-forward",
		Backend:       fmt.Sprintf("svc/%s/%s:%d", ns, svcName, remotePort),
		ClusterTarget: fmt.Sprintf("http://%s:%d%s", clusterHost, remotePort, path),
		ListenHost:    "127.0.0.1",
		ListenPort:    localPort,
		Path:          path,
	}, true
}

func inferenceLocalPort(svcName, ns string) int {
	h := fnv.New32a()
	_, _ = h.Write([]byte(ns + "/" + svcName))
	return inferencePortForwardBase + int(h.Sum32()%2000)
}

func postJSONExchange(url string, body []byte, timeout time.Duration) (HTTPExchange, error) {
	if strings.HasPrefix(url, "incluster://") {
		return inClusterCurlPost(strings.TrimPrefix(url, "incluster://"), body, timeout)
	}
	client := &http.Client{Timeout: timeout}
	req, err := http.NewRequest(http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return HTTPExchange{}, err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return HTTPExchange{}, err
	}
	defer resp.Body.Close()
	b, readErr := io.ReadAll(resp.Body)
	if readErr != nil {
		return HTTPExchange{StatusCode: resp.StatusCode, Headers: resp.Header},
			fmt.Errorf("read response body (status=%d, headers=%v): %w", resp.StatusCode, resp.Header, readErr)
	}
	return HTTPExchange{StatusCode: resp.StatusCode, Headers: resp.Header, Body: string(b)}, nil
}

func inClusterCurlPost(url string, body []byte, timeout time.Duration) (HTTPExchange, error) {
	kc := localKubeconfig()
	pod := fmt.Sprintf("infernex-e2e-curl-%d", time.Now().UnixNano()%100000)
	escaped := strings.ReplaceAll(string(body), "'", "'\\''")
	cmd := fmt.Sprintf(
		`KUBECONFIG=%q kubectl run %s --rm -i --restart=Never --image=curlimages/curl:8.5.0 -n %q --command -- `+
			`sh -c 'curl -sS -w "\n__HTTP_CODE__:%%{http_code}" -X POST -H "Content-Type: application/json" `+
			`-d %q --max-time %d %q'`,
		kc, pod, E2ENS(), escaped, int(timeout.Seconds()), url)
	stdout, stderr, err := ExecLocal(cmd)
	if err != nil {
		return HTTPExchange{}, fmt.Errorf("in-cluster curl: %w stderr=%s", err, stderr)
	}
	code := 0
	respBody := stdout
	if idx := strings.LastIndex(stdout, "\n__HTTP_CODE__:"); idx >= 0 {
		respBody = stdout[:idx]
		codeStr := strings.TrimSpace(stdout[idx+len("\n__HTTP_CODE__:"):])
		code, _ = strconv.Atoi(codeStr)
	}
	if code == 0 {
		return HTTPExchange{}, fmt.Errorf("in-cluster curl: no http code in output: %s", stdout)
	}
	return HTTPExchange{StatusCode: code, Body: respBody}, nil
}

func ensureLeadingSlash(p string) string {
	if strings.HasPrefix(p, "/") {
		return p
	}
	return "/" + p
}

func dedupeStrings(in []string) []string {
	seen := map[string]struct{}{}
	var out []string
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

func logInferenceHTTPRequest(fitID string, cand inferenceCandidate, payload []byte) {
	LogStep(fitID, "----- HTTP REQUEST -----")
	LogStep(fitID, "  method: POST")
	if cand.URL != "" && !strings.HasPrefix(cand.URL, "(see") {
		LogStep(fitID, "  url: %s", cand.URL)
	}
	if cand.Access != "" {
		LogStep(fitID, "  access: %s", cand.Access)
	}
	if cand.ListenHost != "" && cand.ListenPort > 0 {
		LogStep(fitID, "  client_host: %s  client_port: %d  path: %s", cand.ListenHost, cand.ListenPort, cand.Path)
	}
	LogStep(fitID, "  header Content-Type: application/json")
	logHTTPMessage(fitID, "REQUEST", "body: "+string(payload))
	if curl := buildCurlPOSTLine(cand.URL, payload); curl != "" {
		LogStep(fitID, "  curl: %s", curl)
	}
	LogStep(fitID, "----- end HTTP REQUEST -----")
}

func buildCurlPOSTLine(rawURL string, payload []byte) string {
	if strings.HasPrefix(rawURL, "incluster://") {
		return ""
	}
	escaped := strings.ReplaceAll(string(payload), "'", "'\\''")
	return fmt.Sprintf("curl -sS -X POST %q -H 'Content-Type: application/json' -d '%s'", rawURL, escaped)
}

func logHTTPMessage(fitID, kind, msg string) {
	if len(msg) <= inferenceLogBodyMaxBytes {
		LogStep(fitID, "HTTP %s %s", kind, msg)
		return
	}
	LogStep(fitID, "HTTP %s (%d bytes, truncated):\n%s", kind, len(msg), msg[:inferenceLogBodyMaxBytes])
	LogStep(fitID, "HTTP %s ... [%d bytes omitted]", kind, len(msg)-inferenceLogBodyMaxBytes)
}

func logHTTPHeaders(fitID, kind string, h http.Header) {
	if len(h) == 0 {
		LogStep(fitID, "HTTP %s headers: (none)", kind)
		return
	}
	for k, vals := range h {
		LogStep(fitID, "HTTP %s header %s: %s", kind, k, strings.Join(vals, ", "))
	}
}
