package infernexbridge

import (
	"context"
	"log"
	"os"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
	infutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

func TestInferNexBridge(t *testing.T) {
	if deadline, ok := t.Deadline(); ok {
		remaining := time.Until(deadline)
		// FIT 4–7 each need several minutes (PD LLMISVC often >8m for Ready); running 3456 together needs headroom.
		if remaining < 45*time.Minute {
			t.Logf("WARNING: go test -timeout=%v is likely too short for multi-FIT runs (e.g. FIT_4+5+6). "+
				"Recommend: go test -timeout 120m ./infernex/infernex-bridge/... -ginkgo.label-filter=...", remaining.Round(time.Second))
		}
	}
	RegisterFailHandler(Fail)
	suiteConfig, reporterConfig := GinkgoConfiguration()
	if liveStepLogSuiteDefaults() {
		// go test captures stdout/stderr unless interceptor is disabled; LogStep writes stderr directly.
		if suiteConfig.OutputInterceptorMode == "" {
			suiteConfig.OutputInterceptorMode = "none"
		}
		if suiteConfig.PollProgressAfter == 0 {
			suiteConfig.PollProgressAfter = 30 * time.Second
			suiteConfig.PollProgressInterval = 30 * time.Second
		}
		reporterConfig.ForceNewlines = true
	}
	RunSpecs(t, "InferNex-Bridge FIT Suite", suiteConfig, reporterConfig)
}

func liveStepLogSuiteDefaults() bool {
	switch strings.TrimSpace(os.Getenv("INFERNEX_BRIDGE_LIVE_STDERR")) {
	case "0", "false", "no":
		return false
	default:
		return true
	}
}

var _ = BeforeSuite(func() {
	if err := env.LoadEnv(".env"); err != nil {
		log.Printf("load .env: %v (optional)", err)
	}
	if bridgeutils.SkipInferenceE2E() {
		log.Printf("SKIP_INFERENCE_E2E=true — Mooncake FIT 48–51 and optional FIT 4–7 HTTP probes will Skip")
	} else {
		log.Printf("inference E2E enabled (default); export SKIP_INFERENCE_E2E=true to disable HTTP/KV probes")
	}
	bridgeutils.EnableLiveStepLog()

	suiteCtx, suiteCancel = context.WithTimeout(context.Background(), bridgeutils.SuiteContextTimeout())
	DeferCleanup(func() {
		if suiteCancel != nil {
			suiteCancel()
		}
	})

	if reason := bridgeutils.SkipNoKubeconfigReason(); reason != "" {
		log.Printf("BeforeSuite: %s — cluster tests will Skip", reason)
		return
	}
	var err error
	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "create k8s client")

	Expect(infutils.EnsureNPUOperatorInstalled()).To(Succeed(), "npu-operator 应当已存在或安装成功")

	bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, bridgeutils.E2ENS())
	ensurePrerequisitesInBeforeSuite()
	initWorkloadCleanupInterruptHandler()
	log.Printf("infernex-bridge: SIGINT/SIGTERM workload cleanup handler enabled")
})

var _ = AfterSuite(func() {
	bridgeutils.TeardownSuite("AfterSuite")
})

// ensurePrerequisitesInBeforeSuite installs NATS, kube-prometheus-stack, infernex-bridge (OCI helm by default),
// then optional KServe/Gateway hooks. FIT_1–3 verify readiness; use SKIP_*_INSTALL when components are pre-installed.
func ensurePrerequisitesInBeforeSuite() {
	const fitID = "BeforeSuite"
	bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, bridgeutils.BridgeSystemNS())
	bridgeutils.LogStep(fitID, "BeforeSuite prerequisites: NATS=%v prometheus=%v bridge=%v",
		!bridgeutils.SkipNATSInstall(), !bridgeutils.SkipKubePrometheusInstall(), !bridgeutils.SkipBridgeInstall())
	err := bridgeutils.InstallBeforeSuitePrerequisites(fitID)
	if err != nil && !bridgeutils.SkipBridgeInstall() {
		bridgeutils.DumpBridgeDiagnostics(fitID, k8sClient, "")
	}
	Expect(err).NotTo(HaveOccurred(),
		"BeforeSuite prerequisites (SKIP_NATS_INSTALL / SKIP_KUBE_PROMETHEUS_INSTALL / SKIP_BRIDGE_INSTALL / INFERNEX_BRIDGE_CHART_PATH)")
	if !bridgeutils.SkipBridgeInstall() {
		bridgeutils.WaitBridgeControllerReady(suiteCtx, k8sClient.Clientset, fitID, 12*time.Minute)
		bridgeutils.LogStep(fitID, "infernex-bridge ready")
	}
	if !bridgeutils.SkipKServeInstall() {
		bridgeutils.WaitKServeLLMISVCControllerReady(suiteCtx, k8sClient.Clientset, fitID, 15*time.Minute)
		bridgeutils.LogStep(fitID, "kserve llmisvc controller ready")
	}
}
