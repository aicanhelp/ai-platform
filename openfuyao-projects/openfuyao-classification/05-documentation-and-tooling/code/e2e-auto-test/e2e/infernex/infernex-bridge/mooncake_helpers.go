package infernexbridge

import (
	"fmt"
	"time"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

func assertMooncakeDeploymentsReady(fitID, owner string) {
	ns := bridgeutils.E2ENS()
	timeout := 10 * time.Minute
	if bridgeutils.SkipNPUWorkloadE2E() {
		timeout = 5 * time.Minute
	}
	master := bridgeutils.BridgeWorkloadDeployName(owner, "mooncake-master")
	meta := bridgeutils.BridgeWorkloadDeployName(owner, "mooncake-metadata")
	bridgeutils.WaitDeploymentExists(suiteCtx, k8sClient.Clientset, fitID, ns, master, timeout)
	bridgeutils.WaitDeploymentExists(suiteCtx, k8sClient.Clientset, fitID, ns, meta, timeout)
	if !bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitDeploymentRollout(suiteCtx, k8sClient.Clientset, fitID, ns, master, timeout)
		bridgeutils.WaitDeploymentRollout(suiteCtx, k8sClient.Clientset, fitID, ns, meta, timeout)
		bridgeutils.WaitComponentReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, "mooncake", timeout)
		bridgeutils.CheckDeploymentLogHealth(fitID, ns, master,
			[]string{"Master Metrics", "Mem Storage"},
			[]string{"panic", "fatal", "connection refused", "segmentation fault"})
	}
	bridgeutils.LogStep(fitID, "mooncake deployments ready owner=%s", owner)
}

// assertMooncakeInferenceAndKV runs long-prefix chat/completions twice and verifies mooncake-master Get/cache hit counters.
func assertMooncakeInferenceAndKV(fitID, owner, promptTag string) {
	assertMooncakeInferenceAndKVWithEngine(fitID, owner, promptTag, "")
}

func assertMooncakeInferenceAndKVWithEngine(fitID, owner, promptTag, engineDeploy string) {
	// Caller must use SkipGate(..., inference=true) on Mooncake FITs; inference + KV hit are mandatory.
	ns := bridgeutils.E2ENS()
	baseline := bridgeutils.AssertMooncakeMasterMetricsBaseline(fitID, ns, owner)

	sharedPrefix := bridgeutils.MooncakeE2ELongPrompt(promptTag)
	msgs1 := []bridgeutils.ChatMessage{
		{Role: "system", Content: sharedPrefix},
		{Role: "user", Content: "mooncake e2e first pass " + promptTag},
	}
	bridgeutils.LogStep(fitID, "mooncake inference pass 1: shared system prefix len=%d", len(sharedPrefix))
	bridgeutils.AssertInferenceChatCompletionsMessagesOK(fitID, owner, msgs1, 64,
		fmt.Sprintf("mooncake KV pass 1 (write KV) owner=%s", owner), 3*time.Minute)

	msgs2 := []bridgeutils.ChatMessage{
		{Role: "system", Content: sharedPrefix},
		{Role: "user", Content: "mooncake e2e second pass " + promptTag},
	}
	bridgeutils.AssertInferenceChatCompletionsMessagesOK(fitID, owner, msgs2, 64,
		fmt.Sprintf("mooncake KV pass 2 (expect cache hit) owner=%s", owner), 3*time.Minute)

	bridgeutils.LogStep(fitID, "mooncake-master KV metrics check after pass 1+2")
	bridgeutils.AssertMooncakeKVCacheHitAfterWarmup(fitID, ns, owner, baseline)
	if engineDeploy != "" {
		bridgeutils.AssertInferenceEngineKVCacheLogHint(fitID, ns, owner, engineDeploy)
	}
}
