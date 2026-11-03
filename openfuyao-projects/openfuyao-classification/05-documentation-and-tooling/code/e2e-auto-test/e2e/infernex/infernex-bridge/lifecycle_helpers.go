package infernexbridge

import (
	"os"
	"time"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

func lifecycleWorkloadTimeout() time.Duration {
	if bridgeutils.SkipNPUWorkloadE2E() {
		return 5 * time.Minute
	}
	return 25 * time.Minute
}

func lifecycleReadyTimeout() time.Duration {
	if bridgeutils.SkipNPUWorkloadE2E() {
		return 2 * time.Minute
	}
	return 10 * time.Minute
}

// lwsWorkloadTimeout covers MoE LWS engine init download + multi-node rollout (FIT 56–59).
func lwsWorkloadTimeout() time.Duration {
	if bridgeutils.SkipNPUWorkloadE2E() {
		return 5 * time.Minute
	}
	return 90 * time.Minute
}

// lwsReadyTimeout allows extra headroom after workloads exist (large MoE model load).
func lwsReadyTimeout() time.Duration {
	if bridgeutils.SkipNPUWorkloadE2E() {
		return 2 * time.Minute
	}
	return 45 * time.Minute
}

func gatewayLifecycleChecksEnabled(fitID string) bool {
	if bridgeutils.SkipNoKubeconfigReason() != "" {
		return false
	}
	if !bridgeutils.CRDExists(fitID, "gateway.networking.k8s.io", "gateways") {
		return false
	}
	if !bridgeutils.CRDExists(fitID, "gateway.networking.k8s.io", "httproutes") {
		return false
	}
	return bridgeutils.CRDExists(fitID, "inference.networking.x-k8s.io", "inferencepools")
}

func finalizeLifecycleLLMISVC(fitID, name, manifestPath string, deploySuffixes []string) {
	ns := bridgeutils.E2ENS()
	timeout := 5 * time.Minute
	bridgeutils.LogStep(fitID, "lifecycle final teardown LLMISVC owner=%s", name)
	if manifestPath != "" {
		stdout, stderr, err := bridgeutils.DeleteManifest(manifestPath)
		bridgeutils.LogStep(fitID, "delete -f %s: stdout=%s stderr=%s err=%v", manifestPath, stdout, stderr, err)
		_ = os.Remove(manifestPath)
	}
	_ = bridgeutils.DeleteLLMInferenceService(suiteCtx, k8sClient.DynamicClient, ns, name)
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, name)
	bridgeutils.WaitLLMISvcAggregateTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, deploySuffixes, timeout)
}

func finalizeLifecycleInsvc(fitID, name, manifestPath, templateEngineConfig string, deploySuffixes []string) {
	ns := bridgeutils.E2ENS()
	timeout := 5 * time.Minute
	bridgeutils.LogStep(fitID, "lifecycle final teardown InferNexService owner=%s", name)
	if manifestPath != "" {
		stdout, stderr, err := bridgeutils.DeleteManifest(manifestPath)
		bridgeutils.LogStep(fitID, "delete -f %s: stdout=%s stderr=%s err=%v", manifestPath, stdout, stderr, err)
		_ = os.Remove(manifestPath)
	}
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, name)
	bridgeutils.WaitInsvcDeployTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, templateEngineConfig, deploySuffixes, timeout)
}

// assertLifecycleAfterDeleteLLMISVC verifies InferNexService and owner workloads are removed after LLMISVC delete.
func assertLifecycleAfterDeleteLLMISVC(fitID, name string, deploySuffixes []string) {
	ns := bridgeutils.E2ENS()
	timeout := 5 * time.Minute
	bridgeutils.WaitInferNexServiceAbsent(suiteCtx, k8sClient.DynamicClient, fitID, ns, name, timeout)
	bridgeutils.WaitLLMISvcAggregateTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, deploySuffixes, timeout)
}

// assertLifecycleAfterDeleteInsvc verifies owner workloads (and managed gateway when applicable) are gone.
func assertLifecycleAfterDeleteInsvc(fitID, name, templateEngineConfig string, deploySuffixes []string) {
	ns := bridgeutils.E2ENS()
	timeout := 5 * time.Minute
	bridgeutils.WaitInsvcDeployTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, templateEngineConfig, deploySuffixes, timeout)
	if gatewayLifecycleChecksEnabled(fitID) {
		bridgeutils.WaitManagedGatewayTripleAbsent(fitID, ns, name, timeout)
	}
}
