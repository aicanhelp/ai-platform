package utils

import (
	"fmt"
	"strings"
	"time"
)

// GatewayCleanupRefs names external Gateway API objects created outside Bridge-managed triples.
type GatewayCleanupRefs struct {
	ExternalGW    string
	ExternalRoute string
	ExternalPool  string
}

// CleanupGatewayE2EResources deletes Gateway/HTTPRoute/InferencePool for gateway FITs (35–42).
// Called on SIGINT, DeferCleanup, pre-test cleanup, and after InferNexService delete.
func CleanupGatewayE2EResources(fitID, ns, owner string, refs GatewayCleanupRefs, timeout time.Duration) {
	if SkipNoKubeconfigReason() != "" {
		return
	}
	if !GatewayAPIInstalled(fitID) {
		LogStep(fitID, "gateway cleanup skipped: Gateway API CRDs not installed")
		return
	}
	managedGW, managedRoute, managedPool := ManagedResourceNames(owner)
	LogStep(fitID, "gateway cleanup start ns=%s owner=%s managed=(gw=%s route=%s pool=%s) external=(gw=%q route=%q pool=%q)",
		ns, owner, managedGW, managedRoute, managedPool, refs.ExternalGW, refs.ExternalRoute, refs.ExternalPool)

	// HTTPRoute before Gateway (parent refs); pool before gateway routing chain.
	for _, item := range []struct {
		resource string
		name     string
	}{
		{"httproute", managedRoute},
		{"inferencepool", managedPool},
		{"gateway", managedGW},
		{"httproute", refs.ExternalRoute},
		{"inferencepool", refs.ExternalPool},
		{"gateway", refs.ExternalGW},
	} {
		if strings.TrimSpace(item.name) == "" {
			continue
		}
		kubectlDeleteResource(fitID, item.resource, ns, item.name)
	}
	if gw := strings.TrimSpace(refs.ExternalGW); gw != "" {
		kubectlDeleteResource(fitID, "envoyproxy", ns, gw+"-infernex-nodeport")
	}

	if timeout > 0 {
		waitGatewayNamesAbsent(fitID, ns, owner, refs, timeout)
	}
	LogStep(fitID, "gateway cleanup done owner=%s", owner)
}

func waitGatewayNamesAbsent(fitID, ns, owner string, refs GatewayCleanupRefs, timeout time.Duration) {
	managedGW, managedRoute, managedPool := ManagedResourceNames(owner)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		pending := pendingGatewayDeletes(fitID, ns, managedGW, managedRoute, managedPool, refs)
		if len(pending) == 0 {
			LogStep(fitID, "gateway cleanup: all target resources absent")
			return
		}
		LogPollThrottled(fitID, "gw-cleanup-absent-"+owner, defaultPollLogInterval,
			"still waiting for delete: %v", pending)
		time.Sleep(3 * time.Second)
	}
	LogStep(fitID, "gateway cleanup: timeout waiting absent (continuing) pending=%v",
		pendingGatewayDeletes(fitID, ns, managedGW, managedRoute, managedPool, refs))
}

func pendingGatewayDeletes(fitID, ns, managedGW, managedRoute, managedPool string, refs GatewayCleanupRefs) []string {
	var pending []string
	for _, item := range []struct {
		resource string
		name     string
	}{
		{"gateway", managedGW},
		{"httproute", managedRoute},
		{"inferencepool", managedPool},
		{"gateway", refs.ExternalGW},
		{"httproute", refs.ExternalRoute},
		{"inferencepool", refs.ExternalPool},
	} {
		if strings.TrimSpace(item.name) == "" {
			continue
		}
		if KubectlExists(fitID, item.resource, ns, item.name) {
			pending = append(pending, fmt.Sprintf("%s/%s", item.resource, item.name))
		}
	}
	if gw := strings.TrimSpace(refs.ExternalGW); gw != "" {
		if KubectlExists(fitID, "envoyproxy", ns, gw+"-infernex-nodeport") {
			pending = append(pending, "envoyproxy/"+gw+"-infernex-nodeport")
		}
	}
	return pending
}

func kubectlDeleteResource(fitID, resource, ns, name string) {
	kc := localKubeconfig()
	cmd := fmt.Sprintf(
		"KUBECONFIG=%q kubectl delete %s %q -n %q --ignore-not-found --wait=false",
		kc, resource, name, ns)
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "kubectl delete %s/%s ns=%s: stdout=%s stderr=%s err=%v",
		resource, name, ns, strings.TrimSpace(stdout), strings.TrimSpace(stderr), err)
}
