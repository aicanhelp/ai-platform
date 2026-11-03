package utils

import (
	"context"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// AggregateLLMISvcDeploySuffixes lists all deployment suffixes for teardown (includes engine-* for direct insvc).
func AggregateLLMISvcDeploySuffixes() []string {
	out := append([]string{"engine-aggregate"}, AggregateLLMISvcBridgeDeploySuffixes()...)
	return out
}

// AggregateLLMISvcComponentKeys are status.components keys expected ready when NPU workloads roll out.
func AggregateLLMISvcComponentKeys() []string {
	return []string{"inferenceEngine", "cacheIndexer", "hermesRouter", "mooncake"}
}

// PDLLMISvcDeploySuffixes lists all deployment suffixes for teardown (includes engine-pd-* for direct insvc).
func PDLLMISvcDeploySuffixes() []string {
	out := append([]string{"engine-pd-prefill", "engine-pd-decode"}, PDLLMISvcBridgeDeploySuffixes()...)
	return out
}

// PDLLMISvcComponentKeys are status.components keys for PD LLMISVC path InferNexService.
func PDLLMISvcComponentKeys() []string {
	return PDInsvcComponentKeys()
}

func WaitAggregateLLMISvcWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	if !SkipNPUWorkloadE2E() {
		WaitKServeLLMInferenceWorkloads(ctx, client, fitID, ns, owner, timeout)
	}
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, AggregateLLMISvcBridgeDeploySuffixes(), timeout)
	WaitOwnerDaemonSetWorkloads(ctx, client, fitID, ns, owner, EagleEyeDaemonSetSuffixes(), timeout)
}

func WaitPDLLMISvcWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	if !SkipNPUWorkloadE2E() {
		WaitKServePDInferenceWorkloads(ctx, client, fitID, ns, owner, timeout)
	}
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, PDLLMISvcBridgeDeploySuffixes(), timeout)
	WaitOwnerDaemonSetWorkloads(ctx, client, fitID, ns, owner, EagleEyeDaemonSetSuffixes(), timeout)
}

func AssertPDLLMISvcLogHealth(fitID, ns, owner string) {
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-master"),
		[]string{"Master Metrics", "Mem Storage"},
		[]string{"panic", "fatal", "connection refused", "segmentation fault"})
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-metadata"),
		nil, []string{"panic", "fatal", "connection refused"})
	CheckDeploymentLogHealth(fitID, ns, DeployName(owner, "proxy-server"),
		nil, []string{"panic", "connection refused", "upstream unavailable"})
	CheckDeploymentLogHealth(fitID, ns, KServeSchedulerDeployName(owner),
		nil, []string{"panic", "fatal", "connection refused"})
}

func AssertAggregateLLMISvcLogHealth(fitID, ns, owner string) {
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-master"),
		[]string{"Master Metrics", "Mem Storage"},
		[]string{"panic", "fatal", "connection refused", "segmentation fault"})
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-metadata"),
		nil, []string{"panic", "fatal", "connection refused"})
	CheckDeploymentLogHealth(fitID, ns, KServeSchedulerDeployName(owner),
		nil, []string{"panic", "fatal", "connection refused"})
}

// AggregateLLMISvcBridgeComponentKeys are InferNexService status.components keys backed by Bridge Deployments in LLMISVC linked mode.
// inferenceEngine and hermesRouter are KServe-managed (see WaitKServeLLMInferenceWorkloads).
func AggregateLLMISvcBridgeComponentKeys() []string {
	return []string{"cacheIndexer", "mooncake"}
}

func WaitAggregateLLMISvcComponentsReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	for _, key := range AggregateLLMISvcBridgeComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
	WaitOptionalBridgeComponentsReady(ctx, dyn, fitID, ns, name, timeout)
}

// PDLLMISvcBridgeComponentKeys are Bridge-backed status.components keys for PD LLMISVC (not KServe engine/router).
func PDLLMISvcBridgeComponentKeys() []string {
	return []string{"proxyServer", "cacheIndexer", "mooncake"}
}

func WaitPDLLMISvcComponentsReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	for _, key := range PDLLMISvcBridgeComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
	WaitOptionalBridgeComponentsReady(ctx, dyn, fitID, ns, name, timeout)
}

// WaitOwnerWorkloadTeardown waits until InferNexService/LLMISVC and owner deployments are removed.
func WaitOwnerWorkloadTeardown(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, owner, templateEngineConfigName string, deploySuffixes []string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		exists, err := InferNexServiceExists(ctx, dyn, ns, owner)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(exists).To(BeFalse())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	Eventually(func(g Gomega) {
		exists, err := LLMInferenceServiceExists(ctx, dyn, ns, owner)
		g.Expect(err).NotTo(HaveOccurred())
		g.Expect(exists).To(BeFalse())
	}).WithTimeout(timeout).WithPolling(5 * time.Second).Should(Succeed())
	if templateEngineConfigName != "" {
		_ = DeleteInferNexServiceConfig(ctx, dyn, TemplateNS(), templateEngineConfigName)
	}
	_ = DeleteLLMInferenceServiceConfig(ctx, dyn, ns, owner+"-config")
	WaitOwnerResourcesGone(ctx, client, fitID, ns, owner, timeout)
	for _, suffix := range deploySuffixes {
		WaitDeploymentAbsent(ctx, client, fitID, ns, DeployName(owner, suffix), timeout)
	}
	for _, suffix := range EagleEyeDaemonSetSuffixes() {
		WaitDaemonSetAbsent(ctx, client, fitID, ns, DeployName(owner, suffix), timeout)
	}
	LogStep(fitID, "owner workload teardown complete for %s", owner)
}

func WaitLLMISvcAggregateTeardown(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, owner string, deploySuffixes []string, timeout time.Duration) {
	WaitOwnerWorkloadTeardown(ctx, dyn, client, fitID, ns, owner, "", deploySuffixes, timeout)
}
