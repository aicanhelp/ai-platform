package utils

import (
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// HermesRouterDeployName is the Bridge-managed Hermes/EPP router Deployment (InferNexService direct path).
// LLMISVC path uses KServeSchedulerDeployName instead.
func HermesRouterDeployName(owner string) string {
	return DeployName(owner, "hermes-router")
}

// AggregateInsvcDeploySuffixes lists Bridge-managed deployments for aggregate InferNexService (fit06-like).
func AggregateInsvcDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"engine-aggregate",
		"hermes-router",
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	})
}

// AggregateInsvcBridgeOnlyDeploySuffixes lists Bridge Deployments for aggregate insvc (engine may be LWS).
func AggregateInsvcBridgeOnlyDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"hermes-router",
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	})
}

// AggregateInsvcDeploySuffixesForGatewayTest is the minimal rollout set for gateway FITs (FIT 35+).
func AggregateInsvcDeploySuffixesForGatewayTest() []string {
	return []string{"engine-aggregate", "hermes-router"}
}

// AggregateInsvcComponentKeysForGatewayTest are status.components keys for gateway-only aggregate insvc.
func AggregateInsvcComponentKeysForGatewayTest() []string {
	return []string{"inferenceEngine", "hermesRouter"}
}

// AggregateInsvcDeploySuffixesForPDOrchestratorTest returns fit06-like suffixes; pd-orchestrator deploys optional, eagle-eye always included.
func AggregateInsvcDeploySuffixesForPDOrchestratorTest(pdoOn bool) []string {
	core := []string{
		"engine-aggregate",
		"hermes-router",
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	}
	out := append(append([]string{}, core...), EagleEyeDeploymentSuffixes()...)
	if pdoOn {
		out = append(out, PDOrchestratorDeploySuffixes()...)
	}
	return out
}

// AggregateInsvcComponentKeys are status.components keys for aggregate InferNexService.
func AggregateInsvcComponentKeys() []string {
	return AggregateLLMISvcComponentKeys()
}

// PDInsvcDeploySuffixes lists Bridge-managed deployments for PD InferNexService (fit07-like).
func PDInsvcDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"engine-pd-prefill",
		"engine-pd-decode",
		"proxy-server",
		"cache-indexer",
		"hermes-router",
		"mooncake-master",
		"mooncake-metadata",
	})
}

// PDInsvcComponentKeys are status.components keys for PD InferNexService.
func PDInsvcComponentKeys() []string {
	return []string{"inferenceEngine", "proxyServer", "cacheIndexer", "hermesRouter", "mooncake"}
}

// PDInsvcBridgeOnlyDeploySuffixes lists Bridge Deployments for PD insvc (prefill/decode engines may be LWS).
func PDInsvcBridgeOnlyDeploySuffixes() []string {
	return appendOptionalBridgeDeploySuffixes([]string{
		"proxy-server",
		"cache-indexer",
		"hermes-router",
		"mooncake-master",
		"mooncake-metadata",
	})
}

// PDInsvcDeploySuffixesForGatewayTest is the minimal PD rollout set for gateway FITs (35–42).
func PDInsvcDeploySuffixesForGatewayTest() []string {
	return []string{"engine-pd-prefill", "engine-pd-decode", "proxy-server", "hermes-router"}
}

// PDInsvcDeploySuffixesRouterOff is the PD rollout set when intelligentGatewayRouting.router is disabled (FIT 34).
func PDInsvcDeploySuffixesRouterOff() []string {
	return []string{
		"engine-pd-prefill",
		"engine-pd-decode",
		"proxy-server",
		"cache-indexer",
		"mooncake-master",
		"mooncake-metadata",
	}
}

// PDInsvcComponentKeysRouterOff are status.components keys for router-off PD insvc (no hermesRouter).
func PDInsvcComponentKeysRouterOff() []string {
	return []string{"inferenceEngine", "proxyServer", "cacheIndexer", "mooncake"}
}

// PDInsvcComponentKeysForGatewayTest are status.components keys for gateway PD insvc.
func PDInsvcComponentKeysForGatewayTest() []string {
	return []string{"inferenceEngine", "proxyServer", "hermesRouter"}
}

const lightweightBridgeDeployTimeout = 10 * time.Minute

func IsLightweightBridgeDeploySuffix(suffix string) bool {
	if strings.HasPrefix(suffix, "eagle-eye-") || strings.HasPrefix(suffix, "pd-orchestrator-") {
		return true
	}
	switch suffix {
	case "hermes-router", "cache-indexer", "mooncake-master", "mooncake-metadata", "proxy-server":
		return true
	}
	return false
}

func WorkloadRolloutTimeoutForSuffix(suffix string, timeout time.Duration) time.Duration {
	if IsLightweightBridgeDeploySuffix(suffix) {
		return min(lightweightBridgeDeployTimeout, timeout)
	}
	return timeout
}

func WaitOwnerWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, suffixes []string, timeout time.Duration) {
	LogPhaseBegin(fitID, fmt.Sprintf("owner workloads owner=%s count=%d skipNPU=%v", owner, len(suffixes), SkipNPUWorkloadE2E()))
	defer LogPhaseEnd(fitID, fmt.Sprintf("owner workloads owner=%s", owner))
	for i, suffix := range suffixes {
		deploy := BridgeWorkloadDeployName(owner, suffix)
		wait := WorkloadRolloutTimeoutForSuffix(suffix, timeout)
		LogStep(fitID, "workload step %d/%d: %s (wait %s)", i+1, len(suffixes), deploy, wait)
		if SkipNPUWorkloadE2E() {
			WaitDeploymentExists(ctx, client, fitID, ns, deploy, wait)
			continue
		}
		WaitDeploymentRollout(ctx, client, fitID, ns, deploy, wait)
	}
	LogStep(fitID, "owner=%s workloads: all %d deployment(s) rolled out", owner, len(suffixes))
}

func WaitAggregateInsvcWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, AggregateInsvcDeploySuffixes(), timeout)
	WaitOwnerDaemonSetWorkloads(ctx, client, fitID, ns, owner, EagleEyeDaemonSetSuffixes(), timeout)
}

func WaitAggregateInsvcWorkloadsForPDOrchestratorTest(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, pdoOn bool, timeout time.Duration) {
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, AggregateInsvcDeploySuffixesForPDOrchestratorTest(pdoOn), timeout)
	WaitOwnerDaemonSetWorkloads(ctx, client, fitID, ns, owner, EagleEyeDaemonSetSuffixes(), timeout)
}

func WaitPDInsvcWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, timeout time.Duration) {
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, PDInsvcDeploySuffixes(), timeout)
	WaitOwnerDaemonSetWorkloads(ctx, client, fitID, ns, owner, EagleEyeDaemonSetSuffixes(), timeout)
}

func assertInsvcBridgeComponentLogHealth(fitID, ns, owner string) {
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-master"),
		[]string{"Master Metrics", "Mem Storage"},
		[]string{"panic", "fatal", "connection refused", "segmentation fault"})
	CheckDeploymentLogHealth(fitID, ns, BridgeWorkloadDeployName(owner, "mooncake-metadata"),
		nil, []string{"panic", "fatal", "connection refused"})
	CheckDeploymentLogHealth(fitID, ns, HermesRouterDeployName(owner),
		nil, []string{"panic", "fatal", "connection refused"})
}

func AssertAggregateInsvcLogHealth(fitID, ns, owner string) {
	assertInsvcBridgeComponentLogHealth(fitID, ns, owner)
}

func AssertPDInsvcLogHealth(fitID, ns, owner string) {
	assertInsvcBridgeComponentLogHealth(fitID, ns, owner)
	CheckDeploymentLogHealth(fitID, ns, DeployName(owner, "proxy-server"),
		nil, []string{"panic", "connection refused", "upstream unavailable"})
}

func WaitAggregateInsvcComponentsReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	for _, key := range AggregateInsvcComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
	WaitOptionalBridgeComponentsReady(ctx, dyn, fitID, ns, name, timeout)
}

func WaitAggregateInsvcComponentsReadyForPDOrchestratorTest(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, pdoOn bool, timeout time.Duration) {
	for _, key := range AggregateInsvcComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
	WaitComponentReady(ctx, dyn, fitID, ns, name, "eagleEye", timeout)
	if pdoOn {
		WaitComponentReady(ctx, dyn, fitID, ns, name, "pdOrchestrator", timeout)
	}
}

func WaitPDInsvcComponentsReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	for _, key := range PDInsvcComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
	WaitOptionalBridgeComponentsReady(ctx, dyn, fitID, ns, name, timeout)
}

func WaitInsvcDeployTeardown(ctx context.Context, dyn dynamic.Interface, client kubernetes.Interface, fitID, ns, owner string, templateEngineConfigName string, deploySuffixes []string, timeout time.Duration) {
	WaitOwnerWorkloadTeardown(ctx, dyn, client, fitID, ns, owner, templateEngineConfigName, deploySuffixes, timeout)
}
