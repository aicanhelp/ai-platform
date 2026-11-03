package utils

import (
	"context"
	"fmt"
	"time"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// PDOrchestratorDeploySuffixes are Bridge-managed pd-orchestrator Deployments (default-on in platform templates).
func PDOrchestratorDeploySuffixes() []string {
	return []string{
		"pd-orchestrator-elastic-scaler",
		"pd-orchestrator-tidal",
		"pd-orchestrator-rsg",
	}
}

// EagleEyeDeploymentSuffixes are Bridge-managed eagle-eye Deployments.
func EagleEyeDeploymentSuffixes() []string {
	return []string{"eagle-eye-hardware-diagnosis"}
}

// EagleEyeDaemonSetSuffixes are Bridge-managed eagle-eye DaemonSets.
func EagleEyeDaemonSetSuffixes() []string {
	return []string{
		"eagle-eye-hardware-monitor",
		"eagle-eye-network-performance-exporter",
	}
}

// OptionalBridgeDeploySuffixes lists default-on Bridge Deployments beyond core inference sidecars.
func OptionalBridgeDeploySuffixes() []string {
	out := append([]string{}, PDOrchestratorDeploySuffixes()...)
	return append(out, EagleEyeDeploymentSuffixes()...)
}

// OptionalBridgeComponentKeys are InferNexService status.components keys for optional Bridge components.
func OptionalBridgeComponentKeys() []string {
	return []string{"pdOrchestrator", "eagleEye"}
}

func appendOptionalBridgeDeploySuffixes(base []string) []string {
	out := make([]string, len(base), len(base)+len(OptionalBridgeDeploySuffixes()))
	copy(out, base)
	return append(out, OptionalBridgeDeploySuffixes()...)
}

func WaitOptionalBridgeComponentsReady(ctx context.Context, dyn dynamic.Interface, fitID, ns, name string, timeout time.Duration) {
	for _, key := range OptionalBridgeComponentKeys() {
		WaitComponentReady(ctx, dyn, fitID, ns, name, key, timeout)
	}
}

func WaitOwnerDaemonSetWorkloads(ctx context.Context, client kubernetes.Interface, fitID, ns, owner string, suffixes []string, timeout time.Duration) {
	if len(suffixes) == 0 {
		return
	}
	LogPhaseBegin(fitID, fmt.Sprintf("owner daemonsets owner=%s count=%d skipNPU=%v", owner, len(suffixes), SkipNPUWorkloadE2E()))
	defer LogPhaseEnd(fitID, fmt.Sprintf("owner daemonsets owner=%s", owner))
	for i, suffix := range suffixes {
		dsName := DeployName(owner, suffix)
		wait := WorkloadRolloutTimeoutForSuffix(suffix, timeout)
		LogStep(fitID, "daemonset step %d/%d: %s (wait %s)", i+1, len(suffixes), dsName, wait)
		if SkipNPUWorkloadE2E() {
			WaitDaemonSetExists(ctx, client, fitID, ns, dsName, wait)
			continue
		}
		WaitDaemonSetRollout(ctx, client, fitID, ns, dsName, wait)
	}
	LogStep(fitID, "owner=%s daemonsets: all %d rolled out", owner, len(suffixes))
}
