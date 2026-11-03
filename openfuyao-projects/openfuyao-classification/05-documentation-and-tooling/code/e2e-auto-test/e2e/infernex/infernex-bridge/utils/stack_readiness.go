package utils

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// StackPath is how inference engines are deployed: Bridge InferNexService (insvc) or KServe LLMISVC.
type StackPath string

const (
	StackInsvc   StackPath = "insvc"
	StackLLMISvc StackPath = "llmisvc"
)

// StackMode is PD (prefill + decode) or aggregate (single engine).
type StackMode string

const (
	StackPD        StackMode = "pd"
	StackAggregate StackMode = "aggregate"
)

// StackFullyReadyOpts tunes timeouts and optional component/log phases.
type StackFullyReadyOpts struct {
	WorkloadTimeout  time.Duration // 0 → DefaultStackWorkloadTimeout()
	ReadyTimeout     time.Duration // 0 → DefaultStackReadyTimeout()
	ComponentTimeout time.Duration // 0 → 10m
	SkipComponents   bool
	SkipLogHealth    bool
}

// DefaultStackWorkloadTimeout is the standard rollout wait for full inference stacks.
func DefaultStackWorkloadTimeout() time.Duration {
	if SkipNPUWorkloadE2E() {
		return 5 * time.Minute
	}
	return 25 * time.Minute
}

// DefaultStackReadyTimeout is the standard CR Ready wait with engine pod-delete fallback.
func DefaultStackReadyTimeout() time.Duration {
	if SkipNPUWorkloadE2E() {
		return 2 * time.Minute
	}
	return 10 * time.Minute
}

// WaitStackFullyReady is the unified readiness path for infernex stacks.
//
// Order (all paths): mode/source → inference workloads (rollout + KServe owner fallback) →
// InferNexService Ready (WithOwnerFallback) → [LLMISVC Ready WithOwnerFallback] → components/logs.
//
// Inference engine naming:
//   - insvc aggregate:  {owner}-engine-aggregate (1 pod)
//   - insvc pd:         {owner}-engine-pd-prefill, {owner}-engine-pd-decode (2 pods)
//   - llmisvc aggregate: {owner}-kserve (KServe)
//   - llmisvc pd:        {owner}-kserve-prefill, {owner}-kserve (decode)
func WaitStackFullyReady(
	ctx context.Context,
	dyn dynamic.Interface,
	client kubernetes.Interface,
	fitID, ns, owner string,
	path StackPath,
	mode StackMode,
	opts StackFullyReadyOpts,
) {
	modeStr := string(mode)
	Expect(modeStr == "pd" || modeStr == "aggregate").To(BeTrue(), "stack mode must be pd or aggregate, got %q", modeStr)

	workloadTimeout := opts.WorkloadTimeout
	if workloadTimeout == 0 {
		workloadTimeout = DefaultStackWorkloadTimeout()
	}
	readyTimeout := opts.ReadyTimeout
	if readyTimeout == 0 {
		readyTimeout = DefaultStackReadyTimeout()
	}
	componentTimeout := opts.ComponentTimeout
	if componentTimeout == 0 {
		componentTimeout = 10 * time.Minute
	}

	phase := fmt.Sprintf("stack fully ready owner=%s path=%s mode=%s", owner, path, mode)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	switch path {
	case StackLLMISvc:
		Expect(string(mode)).To(BeElementOf("pd", "aggregate"))
		WaitInferNexServiceSourceRefAndMode(ctx, dyn, fitID, ns, owner, "LLMInferenceService", modeStr, 5*time.Minute)
	case StackInsvc:
		WaitInferNexServiceMode(ctx, dyn, fitID, ns, owner, modeStr, 5*time.Minute)
	default:
		Expect(fmt.Errorf("unknown stack path %q", path)).NotTo(HaveOccurred())
	}

	waitStackInferenceWorkloads(ctx, client, fitID, ns, owner, path, mode, workloadTimeout)

	// Final heal pass before CR Ready wait (covers engines that became unhealthy during bridge rollout).
	RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)

	WaitInferNexServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	if path == StackLLMISvc {
		WaitLLMInferenceServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	}

	if opts.SkipComponents || SkipNPUWorkloadE2E() {
		LogStep(fitID, "stack ready (skip components) owner=%s path=%s mode=%s", owner, path, mode)
		return
	}

	waitStackComponentsReady(ctx, dyn, fitID, ns, owner, path, mode, componentTimeout)
	if !opts.SkipLogHealth {
		assertStackLogHealth(fitID, ns, owner, path, mode)
	}
	LogStep(fitID, "stack fully ready owner=%s path=%s mode=%s", owner, path, mode)
}

func waitStackInferenceWorkloads(
	ctx context.Context,
	client kubernetes.Interface,
	fitID, ns, owner string,
	path StackPath,
	mode StackMode,
	timeout time.Duration,
) {
	switch {
	case path == StackInsvc && mode == StackAggregate:
		WaitAggregateInsvcWorkloads(ctx, client, fitID, ns, owner, timeout)
	case path == StackInsvc && mode == StackPD:
		WaitPDInsvcWorkloads(ctx, client, fitID, ns, owner, timeout)
	case path == StackLLMISvc && mode == StackAggregate:
		WaitAggregateLLMISvcWorkloads(ctx, client, fitID, ns, owner, timeout)
	case path == StackLLMISvc && mode == StackPD:
		WaitPDLLMISvcWorkloads(ctx, client, fitID, ns, owner, timeout)
	default:
		Expect(fmt.Errorf("unhandled path=%s mode=%s", path, mode)).NotTo(HaveOccurred())
	}
}

func waitStackComponentsReady(
	ctx context.Context,
	dyn dynamic.Interface,
	fitID, ns, owner string,
	path StackPath,
	mode StackMode,
	timeout time.Duration,
) {
	switch {
	case path == StackInsvc && mode == StackAggregate:
		WaitAggregateInsvcComponentsReady(ctx, dyn, fitID, ns, owner, timeout)
	case path == StackInsvc && mode == StackPD:
		WaitPDInsvcComponentsReady(ctx, dyn, fitID, ns, owner, timeout)
	case path == StackLLMISvc && mode == StackAggregate:
		WaitAggregateLLMISvcComponentsReady(ctx, dyn, fitID, ns, owner, timeout)
	case path == StackLLMISvc && mode == StackPD:
		WaitPDLLMISvcComponentsReady(ctx, dyn, fitID, ns, owner, timeout)
	}
}

func assertStackLogHealth(fitID, ns, owner string, path StackPath, mode StackMode) {
	switch {
	case path == StackInsvc && mode == StackAggregate:
		AssertAggregateInsvcLogHealth(fitID, ns, owner)
	case path == StackInsvc && mode == StackPD:
		AssertPDInsvcLogHealth(fitID, ns, owner)
	case path == StackLLMISvc && mode == StackAggregate:
		AssertAggregateLLMISvcLogHealth(fitID, ns, owner)
	case path == StackLLMISvc && mode == StackPD:
		AssertPDLLMISvcLogHealth(fitID, ns, owner)
	}
}
