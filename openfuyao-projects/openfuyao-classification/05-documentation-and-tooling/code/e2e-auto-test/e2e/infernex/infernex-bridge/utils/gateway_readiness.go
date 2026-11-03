package utils

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

// AssertManagedGatewayTripleNotCreated expects Bridge did not create managed Gateway/HTTPRoute/InferencePool.
func AssertManagedGatewayTripleNotCreated(fitID, ns, owner string) {
	gw, route, pool := ManagedResourceNames(owner)
	Expect(KubectlExists(fitID, "gateway", ns, gw)).To(BeFalse(), "managed gateway %s should not exist when router disabled", gw)
	Expect(KubectlExists(fitID, "httproute", ns, route)).To(BeFalse(), "managed httproute %s should not exist", route)
	Expect(KubectlExists(fitID, "inferencepool", ns, pool)).To(BeFalse(), "managed inferencepool %s should not exist", pool)
	LogStep(fitID, "managed gateway triple absent as expected owner=%s", owner)
}

// AssertManagedGatewayTripleWiring checks HTTPRoute parent Gateway and InferencePool endpoint picker.
func AssertManagedGatewayTripleWiring(fitID, ns, owner string) {
	gw, route, pool := ManagedResourceNames(owner)
	parent := KubectlJSONPath(fitID, "httproute", ns, route, "{.spec.parentRefs[0].name}")
	Expect(parent).To(Equal(gw), "HTTPRoute should reference managed Gateway")
	picker := KubectlJSONPath(fitID, "inferencepool", ns, pool, "{.spec.endpointPickerRef.name}")
	Expect(picker).To(Equal(DeployName(owner, "hermes-router")), "InferencePool should point at hermes-router")
	LogStep(fitID, "managed gateway triple wiring OK gw=%s route=%s pool=%s", gw, route, pool)
}

// WaitGatewayManagedAggregateInsvcReady waits for insvc aggregate engine + hermes-router, managed gateway triple, and CR ready.
// Does not require pre-existing Gateway objects in the cluster — Bridge creates {owner}-infernex-{gateway,route,pool}.
func WaitGatewayManagedAggregateInsvcReady(
	ctx context.Context,
	dyn dynamic.Interface,
	client kubernetes.Interface,
	fitID, ns, owner string,
	workloadTimeout, readyTimeout time.Duration,
) {
	phase := fmt.Sprintf("gateway-managed aggregate insvc ready owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	WaitInferNexServiceMode(ctx, dyn, fitID, ns, owner, "aggregate", 5*time.Minute)
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, AggregateInsvcDeploySuffixesForGatewayTest(), workloadTimeout)
	RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)

	WaitManagedGatewayTripleExists(fitID, ns, owner, 5*time.Minute)
	AssertManagedGatewayTripleWiring(fitID, ns, owner)

	WaitInferNexServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	if !SkipNPUWorkloadE2E() {
		for _, key := range AggregateInsvcComponentKeysForGatewayTest() {
			WaitComponentReady(ctx, dyn, fitID, ns, owner, key, 10*time.Minute)
		}
	}
	if !SkipInferenceE2E() {
		WaitInferNexEnvoyIngressServiceReady(fitID, ns, owner, 10*time.Minute)
	}
	LogStep(fitID, "gateway-managed aggregate insvc ready owner=%s", owner)
}

// WaitGatewayManagedPDInsvcReady waits for PD insvc (fit07-like), managed gateway triple, and CR ready.
func WaitGatewayManagedPDInsvcReady(
	ctx context.Context,
	dyn dynamic.Interface,
	client kubernetes.Interface,
	fitID, ns, owner string,
	workloadTimeout, readyTimeout time.Duration,
) {
	phase := fmt.Sprintf("gateway-managed PD insvc ready owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	WaitInferNexServiceMode(ctx, dyn, fitID, ns, owner, "pd", 5*time.Minute)
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, PDInsvcDeploySuffixesForGatewayTest(), workloadTimeout)
	RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)

	WaitManagedGatewayTripleExists(fitID, ns, owner, 5*time.Minute)
	AssertManagedGatewayTripleWiring(fitID, ns, owner)

	WaitInferNexServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	if !SkipNPUWorkloadE2E() {
		for _, key := range PDInsvcComponentKeysForGatewayTest() {
			WaitComponentReady(ctx, dyn, fitID, ns, owner, key, 10*time.Minute)
		}
	}
	if !SkipInferenceE2E() {
		WaitInferNexEnvoyIngressServiceReady(fitID, ns, owner, 10*time.Minute)
	}
	LogStep(fitID, "gateway-managed PD insvc ready owner=%s", owner)
}

// WaitGatewayPDInsvcWorkloadsAndReady waits PD engine/router without requiring managed gateway triple (ref/spec FITs).
func WaitGatewayPDInsvcWorkloadsAndReady(
	ctx context.Context,
	dyn dynamic.Interface,
	client kubernetes.Interface,
	fitID, ns, owner string,
	workloadTimeout, readyTimeout time.Duration,
) {
	phase := fmt.Sprintf("gateway PD insvc workloads+ready owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	WaitInferNexServiceMode(ctx, dyn, fitID, ns, owner, "pd", 5*time.Minute)
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, PDInsvcDeploySuffixesForGatewayTest(), workloadTimeout)
	RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)
	WaitInferNexServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	if !SkipNPUWorkloadE2E() {
		for _, key := range PDInsvcComponentKeysForGatewayTest() {
			WaitComponentReady(ctx, dyn, fitID, ns, owner, key, 10*time.Minute)
		}
	}
	if !SkipInferenceE2E() {
		WaitInferNexEnvoyIngressServiceReady(fitID, ns, owner, 10*time.Minute)
	}
	LogStep(fitID, "gateway PD insvc workloads+ready owner=%s", owner)
}

// WaitPDInsvcRouterOffFullyReady waits full PD insvc (fit07-like) with router disabled — no hermes/gateway triple.
func WaitPDInsvcRouterOffFullyReady(
	ctx context.Context,
	dyn dynamic.Interface,
	client kubernetes.Interface,
	fitID, ns, owner string,
	workloadTimeout, readyTimeout time.Duration,
) {
	phase := fmt.Sprintf("PD insvc router-off ready owner=%s", owner)
	LogPhaseBegin(fitID, phase)
	defer LogPhaseEnd(fitID, phase)

	WaitInferNexServiceMode(ctx, dyn, fitID, ns, owner, "pd", 5*time.Minute)
	WaitOwnerWorkloads(ctx, client, fitID, ns, owner, PDInsvcDeploySuffixesRouterOff(), workloadTimeout)
	RestartFlappingInferenceEnginePodsByOwner(ctx, client, fitID, ns, owner)
	WaitInferNexServiceReadyTrueWithOwnerFallback(ctx, dyn, client, fitID, ns, owner, owner, readyTimeout)
	if !SkipNPUWorkloadE2E() {
		for _, key := range PDInsvcComponentKeysRouterOff() {
			WaitComponentReady(ctx, dyn, fitID, ns, owner, key, 10*time.Minute)
		}
	}
	LogStep(fitID, "PD insvc router-off ready owner=%s", owner)
}
