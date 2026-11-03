package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const defaultProxyDiscoveryIntervalSec = "10"

// ProxyBackendKind selects proxy discovery label scheme (KServe LLMISVC vs direct InferNexService).
type ProxyBackendKind int

const (
	ProxyBackendKServe ProxyBackendKind = iota
	ProxyBackendInferNexService
)

func llmisvcProxyPrefillSelector(workloadName string) string {
	return fmt.Sprintf("app.kubernetes.io/component in (llminferenceservice-workload-prefill,llminferenceservice-workload-leader-prefill,llminferenceservice-workload-worker-prefill),app.kubernetes.io/name=%s",
		workloadName)
}

func llmisvcProxyDecodeSelector(workloadName string) string {
	return fmt.Sprintf("app.kubernetes.io/component in (llminferenceservice-workload,llminferenceservice-workload-leader,llminferenceservice-workload-worker),app.kubernetes.io/name=%s",
		workloadName)
}

func openfuyaoPDPrefillSelector(workloadName string) string {
	return fmt.Sprintf("openfuyao.com/pdRole=prefill,openfuyao.com/pdGroupID=%s", workloadName)
}

func openfuyaoPDDecodeSelector(workloadName string) string {
	return fmt.Sprintf("openfuyao.com/pdRole=decode,openfuyao.com/pdGroupID=%s", workloadName)
}

// AssertProxyServerHelmAlignedArgs verifies proxy-server deployment args/env match Bridge Helm assets.
func AssertProxyServerHelmAlignedArgs(ctx context.Context, client kubernetes.Interface, fitID, ns, ownerName string, backend ProxyBackendKind) {
	deployName := DeployName(ownerName, "proxy-server")
	LogPhaseBegin(fitID, fmt.Sprintf("proxy-server helm-aligned args deploy=%s", deployName))
	defer LogPhaseEnd(fitID, "proxy-server helm-aligned args")

	var prefLabel, decLabel string
	switch backend {
	case ProxyBackendKServe:
		prefLabel = llmisvcProxyPrefillSelector(ownerName)
		decLabel = llmisvcProxyDecodeSelector(ownerName)
	case ProxyBackendInferNexService:
		prefLabel = openfuyaoPDPrefillSelector(ownerName)
		decLabel = openfuyaoPDDecodeSelector(ownerName)
	default:
		Expect(backend).To(Equal(ProxyBackendKServe), "unknown proxy backend kind %v", backend)
	}

	d, err := client.AppsV1().Deployments(ns).Get(ctx, deployName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "get deployment %s", deployName)

	var proxy *corev1.Container
	for i := range d.Spec.Template.Spec.Containers {
		if d.Spec.Template.Spec.Containers[i].Name == "proxy-server" {
			proxy = &d.Spec.Template.Spec.Containers[i]
			break
		}
	}
	Expect(proxy).NotTo(BeNil(), "proxy-server container in %s", deployName)

	argsText := strings.Join(proxy.Args, "\n")
	for _, frag := range []string{
		"--host 0.0.0.0",
		"--port 8000",
		"--prefiller-container prefill-engine",
		"--decoder-container decode-engine",
		"--prefiller-port-name prefill-port",
		"--decoder-port-name decode-port",
		"--disable-access-log-for-endpoints=/health,/metrics",
		`"${INFERNEX_PROXY_PREFILLER_LABEL}"`,
		`"${INFERNEX_PROXY_DECODER_LABEL}"`,
		`"${INFERNEX_PROXY_WORKLOAD_NAMESPACE}"`,
		`"${INFERNEX_PROXY_DISCOVERY_INTERVAL_SEC}"`,
	} {
		Expect(argsText).To(ContainSubstring(frag), "proxy-server args missing %q", frag)
	}

	envMap := containerEnvMap(proxy)
	Expect(envMap["INFERNEX_PROXY_PREFILLER_LABEL"]).To(Equal(prefLabel))
	Expect(envMap["INFERNEX_PROXY_DECODER_LABEL"]).To(Equal(decLabel))
	Expect(envMap["INFERNEX_PROXY_WORKLOAD_NAMESPACE"]).To(Equal(ns))
	Expect(envMap["INFERNEX_PROXY_DISCOVERY_INTERVAL_SEC"]).To(Equal(defaultProxyDiscoveryIntervalSec))
}

func containerEnvMap(c *corev1.Container) map[string]string {
	out := make(map[string]string, len(c.Env))
	for _, e := range c.Env {
		if e.Value != "" {
			out[e.Name] = e.Value
		}
	}
	return out
}

const (
	proxyBackendRegistrationTimeout = 5 * time.Minute
	proxyBackendRegistrationPoll    = 10 * time.Second
)

// WaitProxyServerPDBackendsRegistered waits until the PD proxy-server has discovered and
// registered at least one prefiller and one decoder backend.
func WaitProxyServerPDBackendsRegistered(fitID, ns, owner string) {
	deployName := DeployName(owner, "proxy-server")
	kc := localKubeconfig()
	existsCmd := fmt.Sprintf("KUBECONFIG=%q kubectl get deploy -n %q %q -o name --ignore-not-found", kc, ns, deployName)
	out, _, err := ExecLocal(existsCmd)
	if err != nil || strings.TrimSpace(out) == "" {
		LogStep(fitID, "no deploy/%s in ns=%s — skip PD proxy backend registration wait", deployName, ns)
		return
	}

	phase := fmt.Sprintf("proxy-server PD backends registered deploy=%s", deployName)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	Eventually(func(g Gomega) {
		logsCmd := fmt.Sprintf("KUBECONFIG=%q kubectl logs -n %q deploy/%q --tail=200", kc, ns, deployName)
		stdout, stderr, err := ExecLocal(logsCmd)
		g.Expect(err).NotTo(HaveOccurred(), "read logs for deploy/%s: %s", deployName, stderr)
		lower := strings.ToLower(stdout)
		prefillRegistered := strings.Contains(lower, "added prefiller node")
		decodeRegistered := strings.Contains(lower, "added decoder node")
		if !prefillRegistered || !decodeRegistered {
			LogPollThrottled(fitID, "proxy-backends-"+deployName, defaultPollLogInterval,
				"proxy-server backends not fully registered: %s", lastProxyDiscoveryState(stdout))
		}
		g.Expect(prefillRegistered).To(BeTrue(), "deploy/%s: no \"Added prefiller node\" in logs", deployName)
		g.Expect(decodeRegistered).To(BeTrue(), "deploy/%s: no \"Added decoder node\" in logs", deployName)
	}).WithTimeout(proxyBackendRegistrationTimeout).WithPolling(proxyBackendRegistrationPoll).Should(Succeed())
	LogStep(fitID, "proxy-server deploy/%s: prefiller and decoder backends registered", deployName)
}

func lastProxyDiscoveryState(logs string) string {
	state := "no discovery line in log tail"
	for _, line := range strings.Split(logs, "\n") {
		if strings.Contains(line, "Service discovery changed") {
			state = strings.TrimSpace(line)
		}
	}
	return state
}
