package infernexbridge

import (
	"time"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

func stackFullyReadyOpts() bridgeutils.StackFullyReadyOpts {
	return bridgeutils.StackFullyReadyOpts{
		WorkloadTimeout: lifecycleWorkloadTimeout(),
		ReadyTimeout:    lifecycleReadyTimeout(),
	}
}

func waitStackFullyReady(fitID, owner string, path bridgeutils.StackPath, mode bridgeutils.StackMode, opts bridgeutils.StackFullyReadyOpts) {
	if opts.WorkloadTimeout == 0 {
		opts.WorkloadTimeout = lifecycleWorkloadTimeout()
	}
	if opts.ReadyTimeout == 0 {
		opts.ReadyTimeout = lifecycleReadyTimeout()
	}
	bridgeutils.WaitStackFullyReady(
		suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset,
		fitID, bridgeutils.E2ENS(), owner, path, mode, opts,
	)
}

func waitAggregateLLMISvcFullyReady(fitID, name string) {
	waitStackFullyReady(fitID, name, bridgeutils.StackLLMISvc, bridgeutils.StackAggregate, stackFullyReadyOpts())
	bridgeutils.AssertNoDeploymentSubstring(suiteCtx, k8sClient.Clientset, bridgeutils.E2ENS(), name+"-proxy-server")
}

func waitAggregateInsvcFullyReady(fitID, name string) {
	waitStackFullyReady(fitID, name, bridgeutils.StackInsvc, bridgeutils.StackAggregate, stackFullyReadyOpts())
	bridgeutils.AssertNoDeploymentSubstring(suiteCtx, k8sClient.Clientset, bridgeutils.E2ENS(), name+"-proxy-server")
	if gatewayLifecycleChecksEnabled(fitID) {
		bridgeutils.WaitManagedGatewayTripleExists(fitID, bridgeutils.E2ENS(), name, 5*time.Minute)
	}
}

func waitPDInsvcFullyReady(fitID, name string) {
	waitStackFullyReady(fitID, name, bridgeutils.StackInsvc, bridgeutils.StackPD, stackFullyReadyOpts())
}

func waitPDLLMISvcFullyReady(fitID, name string) {
	waitStackFullyReady(fitID, name, bridgeutils.StackLLMISvc, bridgeutils.StackPD, stackFullyReadyOpts())
}

// waitStackWorkloadsOnly rolls out inference + bridge workloads without CR Ready / components (scaling/orchestrator pre-phases).
func waitStackWorkloadsOnly(fitID, owner string, path bridgeutils.StackPath, mode bridgeutils.StackMode, workloadTimeout time.Duration) {
	if workloadTimeout == 0 {
		workloadTimeout = lifecycleWorkloadTimeout()
	}
	ns := bridgeutils.E2ENS()
	modeStr := string(mode)
	switch path {
	case bridgeutils.StackLLMISvc:
		bridgeutils.WaitInferNexServiceSourceRefAndMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, "LLMInferenceService", modeStr, 5*time.Minute)
	case bridgeutils.StackInsvc:
		bridgeutils.WaitInferNexServiceMode(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, modeStr, 5*time.Minute)
	}
	switch {
	case path == bridgeutils.StackInsvc && mode == bridgeutils.StackAggregate:
		bridgeutils.WaitAggregateInsvcWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, owner, workloadTimeout)
	case path == bridgeutils.StackInsvc && mode == bridgeutils.StackPD:
		bridgeutils.WaitPDInsvcWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, owner, workloadTimeout)
	case path == bridgeutils.StackLLMISvc && mode == bridgeutils.StackAggregate:
		bridgeutils.WaitAggregateLLMISvcWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, owner, workloadTimeout)
	case path == bridgeutils.StackLLMISvc && mode == bridgeutils.StackPD:
		bridgeutils.WaitPDLLMISvcWorkloads(suiteCtx, k8sClient.Clientset, fitID, ns, owner, workloadTimeout)
	}
}

// waitStackReadyWithOwnerFallback waits InferNexService [/ LLMISVC] Ready with engine pod-delete fallback only.
func waitStackReadyWithOwnerFallback(fitID, owner string, path bridgeutils.StackPath, readyTimeout time.Duration) {
	if readyTimeout == 0 {
		readyTimeout = lifecycleReadyTimeout()
	}
	ns := bridgeutils.E2ENS()
	bridgeutils.WaitInferNexServiceReadyTrueWithOwnerFallback(
		suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, owner, owner, readyTimeout,
	)
	if path == bridgeutils.StackLLMISvc {
		bridgeutils.WaitLLMInferenceServiceReadyTrueWithOwnerFallback(
			suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, owner, owner, readyTimeout,
		)
	}
}
