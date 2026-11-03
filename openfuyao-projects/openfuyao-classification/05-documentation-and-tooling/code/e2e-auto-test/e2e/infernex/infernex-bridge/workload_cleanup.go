package infernexbridge

import (
	"fmt"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"

	. "github.com/onsi/ginkgo/v2"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

// workloadCleanupSpec describes cluster objects created by apply -f workload fixtures.
type workloadCleanupSpec struct {
	fitID                string
	ownerName            string
	manifestPath         string
	removeManifestFile   bool
	externalManifests    []externalManifestCleanup
	templateEngineConfig string
	deploySuffixes       []string
	llmisvc              bool
	scalingCleanup       bool
	scalingPD            bool
	gatewayCleanup       bool
	gatewayRefs          bridgeutils.GatewayCleanupRefs
}

type externalManifestCleanup struct {
	path       string
	removeFile bool
}

type workloadCleanupHandle struct {
	spec workloadCleanupSpec
	once sync.Once
}

var (
	workloadCleanupMu     sync.Mutex
	activeWorkloadCleanup *workloadCleanupHandle
)

func initWorkloadCleanupInterruptHandler() {
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		msg := fmt.Sprintf("[infernex-bridge] received %v — best-effort workload cleanup (see InferNexBridge_FIT_* lines below)", sig)
		fmt.Fprintln(os.Stderr, msg)
		_, _ = GinkgoWriter.Write([]byte(msg + "\n"))
		runActiveWorkloadCleanup("signal")
		bridgeutils.TeardownSuite("signal")
	}()
}

func registerWorkloadCleanup(spec workloadCleanupSpec) *workloadCleanupHandle {
	h := &workloadCleanupHandle{spec: spec}
	workloadCleanupMu.Lock()
	activeWorkloadCleanup = h
	workloadCleanupMu.Unlock()

	DeferCleanup(func() {
		h.run("ginkgo-defer-cleanup")
		workloadCleanupMu.Lock()
		if activeWorkloadCleanup == h {
			activeWorkloadCleanup = nil
		}
		workloadCleanupMu.Unlock()
	})
	bridgeutils.LogStep(spec.fitID, "workload cleanup registered (DeferCleanup+AfterEach+SIGINT) owner=%s llmisvc=%v deploySuffixes=%d",
		spec.ownerName, spec.llmisvc, len(spec.deploySuffixes))
	return h
}

func runActiveWorkloadCleanup(reason string) {
	workloadCleanupMu.Lock()
	h := activeWorkloadCleanup
	workloadCleanupMu.Unlock()
	if h != nil {
		h.run(reason)
	}
}

func afterEachWorkloadCleanupFallback() {
	workloadCleanupMu.Lock()
	h := activeWorkloadCleanup
	workloadCleanupMu.Unlock()
	if h != nil {
		h.run("after-each-fallback")
	}
}

func (h *workloadCleanupHandle) run(reason string) {
	h.once.Do(func() {
		if k8sClient == nil || bridgeutils.SkipNoKubeconfigReason() != "" {
			return
		}
		spec := h.spec
		bridgeutils.LogStep(spec.fitID, "workload cleanup (%s) owner=%s llmisvc=%v scaling=%v", reason, spec.ownerName, spec.llmisvc, spec.scalingCleanup)
		timeout := 5 * time.Minute
		if reason == "signal" {
			timeout = 3 * time.Minute
		}
		if spec.scalingCleanup {
			ns := bridgeutils.E2ENS()
			bridgeutils.LogStep(spec.fitID, "scaling cleanup first (ES/Tidal/RSG) owner=%s pd=%v", spec.ownerName, spec.scalingPD)
			bridgeutils.CleanupBridgeScalingChain(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, spec.fitID, ns, spec.ownerName, spec.scalingPD)
		}
		if spec.llmisvc {
			execLLMISvcWorkloadCleanup(spec, timeout)
		} else {
			execInsvcWorkloadCleanup(spec, timeout)
		}
		if spec.removeManifestFile && spec.manifestPath != "" {
			_ = os.Remove(spec.manifestPath)
		}
	})
}

func execLLMISvcWorkloadCleanup(spec workloadCleanupSpec, timeout time.Duration) {
	ns := bridgeutils.E2ENS()
	if spec.manifestPath != "" {
		stdout, stderr, err := bridgeutils.DeleteManifest(spec.manifestPath)
		bridgeutils.LogStep(spec.fitID, "delete -f %s: stdout=%s stderr=%s err=%v", spec.manifestPath, stdout, stderr, err)
	}
	_ = bridgeutils.DeleteLLMInferenceService(suiteCtx, k8sClient.DynamicClient, ns, spec.ownerName)
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, spec.ownerName)
	bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, spec.fitID, ns, spec.ownerName, timeout)
	bridgeutils.WaitOwnerWorkloadTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, spec.fitID, ns, spec.ownerName, "", spec.deploySuffixes, timeout)
	bridgeutils.LogStep(spec.fitID, "workload cleanup finished (%s) owner=%s", "llmisvc", spec.ownerName)
}

func execInsvcWorkloadCleanup(spec workloadCleanupSpec, timeout time.Duration) {
	ns := bridgeutils.E2ENS()
	if spec.manifestPath != "" {
		stdout, stderr, err := bridgeutils.DeleteManifest(spec.manifestPath)
		bridgeutils.LogStep(spec.fitID, "delete -f %s: stdout=%s stderr=%s err=%v", spec.manifestPath, stdout, stderr, err)
	}
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, spec.ownerName)
	for _, ext := range spec.externalManifests {
		if ext.path != "" {
			stdout, stderr, err := bridgeutils.DeleteManifest(ext.path)
			bridgeutils.LogStep(spec.fitID, "delete external -f %s: stdout=%s stderr=%s err=%v", ext.path, stdout, stderr, err)
		}
		if ext.removeFile && ext.path != "" {
			_ = os.Remove(ext.path)
		}
	}
	if spec.gatewayCleanup {
		gwTimeout := timeout
		if gwTimeout > 2*time.Minute {
			gwTimeout = 2 * time.Minute
		}
		bridgeutils.CleanupGatewayE2EResources(spec.fitID, ns, spec.ownerName, spec.gatewayRefs, gwTimeout)
	}
	bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, spec.fitID, ns, spec.ownerName, timeout)
	bridgeutils.WaitOwnerWorkloadTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, spec.fitID, ns, spec.ownerName, spec.templateEngineConfig, spec.deploySuffixes, timeout)
	bridgeutils.LogStep(spec.fitID, "workload cleanup finished (%s) owner=%s", "insvc", spec.ownerName)
}

func cleanupLLMISvcEntry(fitID, name string) {
	ns := bridgeutils.E2ENS()
	_ = bridgeutils.DeleteLLMInferenceService(suiteCtx, k8sClient.DynamicClient, ns, name)
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, name)
	bridgeutils.WaitOwnerResourcesGone(suiteCtx, k8sClient.Clientset, fitID, ns, name, 3*time.Minute)
}

func cleanupInsvcEntry(fitID, name, templateEngineConfigName string, deploySuffixes []string, gatewayRefs bridgeutils.GatewayCleanupRefs, gatewayCleanup bool) {
	ns := bridgeutils.E2ENS()
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, name)
	if gatewayCleanup {
		bridgeutils.CleanupGatewayE2EResources(fitID, ns, name, gatewayRefs, 90*time.Second)
	}
	if templateEngineConfigName != "" {
		_ = bridgeutils.DeleteInferNexServiceConfig(suiteCtx, k8sClient.DynamicClient, bridgeutils.TemplateNS(), templateEngineConfigName)
	}
	bridgeutils.WaitOwnerWorkloadTeardown(suiteCtx, k8sClient.DynamicClient, k8sClient.Clientset, fitID, ns, name, templateEngineConfigName, deploySuffixes, 3*time.Minute)
}

func beginLLMISvcWorkloadCleanup(fitID, name string, deploySuffixes []string) *workloadCleanupHandle {
	bridgeutils.LogStep(fitID, "workload pre-cleanup start owner=%s (llmisvc)", name)
	cleanupLLMISvcEntry(fitID, name)
	bridgeutils.LogStep(fitID, "workload pre-cleanup done owner=%s", name)
	return registerWorkloadCleanup(workloadCleanupSpec{
		fitID: fitID, ownerName: name, llmisvc: true, deploySuffixes: deploySuffixes,
	})
}

func beginInsvcWorkloadCleanup(fitID, name, templateEngineConfig string, deploySuffixes []string) *workloadCleanupHandle {
	return beginInsvcWorkloadCleanupWithGateway(fitID, name, templateEngineConfig, deploySuffixes, false, bridgeutils.GatewayCleanupRefs{})
}

func beginInsvcWorkloadCleanupWithGateway(fitID, name, templateEngineConfig string, deploySuffixes []string, gatewayCleanup bool, gatewayRefs bridgeutils.GatewayCleanupRefs) *workloadCleanupHandle {
	bridgeutils.LogStep(fitID, "workload pre-cleanup start owner=%s (insvc) engineConfig=%s gatewayCleanup=%v", name, templateEngineConfig, gatewayCleanup)
	cleanupInsvcEntry(fitID, name, templateEngineConfig, deploySuffixes, gatewayRefs, gatewayCleanup)
	bridgeutils.LogStep(fitID, "workload pre-cleanup done owner=%s", name)
	return registerWorkloadCleanup(workloadCleanupSpec{
		fitID: fitID, ownerName: name, templateEngineConfig: templateEngineConfig,
		llmisvc: false, deploySuffixes: deploySuffixes,
		gatewayCleanup: gatewayCleanup, gatewayRefs: gatewayRefs,
	})
}

func (h *workloadCleanupHandle) SetManifest(path string, removeFile bool) {
	h.spec.manifestPath = path
	h.spec.removeManifestFile = removeFile
	bridgeutils.LogStep(h.spec.fitID, "workload cleanup will delete -f %s removeTemp=%v", path, removeFile)
}

// RegisterExternalManifest adds a separate kubectl delete -f target (FIT 36/42 external Gateway YAML).
func (h *workloadCleanupHandle) RegisterExternalManifest(path string, removeFile bool) {
	if path == "" {
		return
	}
	h.spec.externalManifests = append(h.spec.externalManifests, externalManifestCleanup{path: path, removeFile: removeFile})
	bridgeutils.LogStep(h.spec.fitID, "gateway cleanup will delete external -f %s removeTemp=%v", path, removeFile)
}

// EnableScalingCleanup ensures ES/Tidal/(RSG) are deleted before insvc/llmisvc on teardown.
func (h *workloadCleanupHandle) EnableScalingCleanup(pd bool) {
	h.spec.scalingCleanup = true
	h.spec.scalingPD = pd
	bridgeutils.LogStep(h.spec.fitID, "scaling cleanup registered before workload delete owner=%s pd=%v", h.spec.ownerName, pd)
}
