package infernexbridge

import (
	"time"

	. "github.com/onsi/gomega"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

const (
	fit34RouterDisabledPDFixture = "fit34_router_disabled.yaml"
	fit35GatewayManagedPDFixture = "fit35_gateway_managed.yaml"
	fit36GatewayRefFixture       = "fit36_gateway_ref.yaml"
	fit37GatewaySpecFixture      = "fit37_gateway_custom_spec.yaml"
	fit38PoolRefFixture          = "fit38_external_pool.yaml"
	fit39PoolSpecFixture         = "fit39_pool_custom_spec.yaml"
	fit40HTTPRouteRefFixture     = "fit40_external_httproute.yaml"
	fit41HTTPRouteSpecFixture    = "fit41_httproute_spec_parentrefs.yaml"
	fit42MixedGatewayRefFixture  = "fit42_mixed_gateway_ref.yaml"
)

func gatewayPDTimeouts() (workload, ready time.Duration) {
	return lifecycleWorkloadTimeout(), lifecycleReadyTimeout()
}

func gatewayCleanupRefsFromRender(renderOpt ...bridgeutils.RenderOptions) bridgeutils.GatewayCleanupRefs {
	if len(renderOpt) == 0 {
		return bridgeutils.GatewayCleanupRefs{}
	}
	o := renderOpt[0]
	return bridgeutils.GatewayCleanupRefs{
		ExternalGW:    o.ExternalGW,
		ExternalRoute: o.ExternalRoute,
		ExternalPool:  o.ExternalPool,
	}
}

func beginGatewayPDInsvcTest(fitID, name, fixtureFile string, renderOpt ...bridgeutils.RenderOptions) *workloadCleanupHandle {
	engineCfg := name + "-vllm-pd-engine-config"
	refs := gatewayCleanupRefsFromRender(renderOpt...)
	h := beginInsvcWorkloadCleanupWithGateway(fitID, name, engineCfg, bridgeutils.PDInsvcDeploySuffixesForGatewayTest(), true, refs)
	var path string
	if len(renderOpt) > 0 {
		path = applyDeployFixtureNoDefer(fitID, fixtureFile, name, renderOpt[0])
	} else {
		path = applyDeployFixtureNoDefer(fitID, fixtureFile, name)
	}
	h.SetManifest(path, true)
	return h
}

func waitGatewayManagedPDInsvcFullyReady(fitID, name string) {
	wt, rt := gatewayPDTimeouts()
	bridgeutils.WaitGatewayManagedPDInsvcReady(
		suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset,
		fitID, bridgeutils.E2ENS(), name, wt, rt,
	)
}

func waitGatewayPDInsvcWorkloadsAndReady(fitID, name string) {
	wt, rt := gatewayPDTimeouts()
	bridgeutils.WaitGatewayPDInsvcWorkloadsAndReady(
		suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset,
		fitID, bridgeutils.E2ENS(), name, wt, rt,
	)
}

func assertGatewayManagedInference(fitID, owner string) {
	if bridgeutils.SkipInferenceE2E() {
		bridgeutils.LogStep(fitID, "SKIP_INFERENCE_E2E=true — skip gateway inference HTTP for owner=%s", owner)
		return
	}
	bridgeutils.AssertInferenceChatCompletionsOK(fitID, owner, "hello infernex "+fitID, 32)
}

func assertGatewayPDInference(fitID, owner string) {
	assertGatewayManagedInference(fitID, owner)
}

func applyExternalManifest(fitID, fixtureFile string, opts bridgeutils.RenderOptions, h *workloadCleanupHandle) string {
	path, err := bridgeutils.WriteRenderedFixtureOpts([]string{"deploy", fixtureFile}, opts)
	Expect(err).NotTo(HaveOccurred())
	stdout, stderr, err := bridgeutils.ApplyManifest(path)
	bridgeutils.LogStep(fitID, "apply external %s: stdout=%s stderr=%s err=%v", fixtureFile, stdout, stderr, err)
	Expect(err).NotTo(HaveOccurred())
	if h != nil {
		h.RegisterExternalManifest(path, true)
	}
	return path
}

func beginPDInsvcRouterOffTest(fitID, name string) *workloadCleanupHandle {
	engineCfg := name + "-vllm-pd-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, bridgeutils.PDInsvcDeploySuffixesRouterOff())
	path := applyDeployFixtureNoDefer(fitID, fit34RouterDisabledPDFixture, name)
	h.SetManifest(path, true)
	return h
}

func waitPDInsvcRouterOffFullyReady(fitID, name string) {
	wt, rt := gatewayPDTimeouts()
	bridgeutils.WaitPDInsvcRouterOffFullyReady(
		suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset,
		fitID, bridgeutils.E2ENS(), name, wt, rt,
	)
}

func assertFit34RouterDisabledStatus(fitID, name string) {
	ns := bridgeutils.E2ENS()
	waitPDInsvcRouterOffFullyReady(fitID, name)
	bridgeutils.WaitInferNexCondition(suiteCtx, k8sClient.DynamicClient, fitID, ns, name,
		"GatewayRoutingReady", "True", "RouterDisabled", 3*time.Minute)
	bridgeutils.AssertManagedGatewayTripleNotCreated(fitID, ns, name)
	bridgeutils.LogStep(fitID, "FIT_34 PD insvc ready, GatewayRoutingReady=RouterDisabled, no managed gateway triple")
}
