package infernexbridge

import (
	"os"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	bridgeutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/infernex-bridge/utils"
)

func applyDeployFixture(fitID string, fixtureFile, insvcName string, opt ...bridgeutils.RenderOptions) string {
	ns := bridgeutils.E2ENS()
	bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, ns)
	opts := bridgeutils.RenderOptions{Namespace: ns, Name: insvcName}
	if len(opt) > 0 {
		o := opt[0]
		if o.ExternalGW != "" {
			opts.ExternalGW = o.ExternalGW
		}
		if o.ExternalRoute != "" {
			opts.ExternalRoute = o.ExternalRoute
		}
		if o.ExternalPool != "" {
			opts.ExternalPool = o.ExternalPool
		}
		if o.KServeNS != "" {
			opts.KServeNS = o.KServeNS
		}
	}
	path, err := bridgeutils.WriteRenderedFixtureOpts([]string{"deploy", fixtureFile}, opts)
	Expect(err).NotTo(HaveOccurred())
	stdout, stderr, err := bridgeutils.ApplyManifest(path)
	bridgeutils.LogStep(fitID, "apply %s: stdout=%s stderr=%s err=%v", fixtureFile, stdout, stderr, err)
	Expect(err).NotTo(HaveOccurred())
	return path
}

func applyDeployFixtureNoDefer(fitID string, fixtureFile, insvcName string, opt ...bridgeutils.RenderOptions) string {
	return applyDeployFixture(fitID, fixtureFile, insvcName, opt...)
}

func applyDeployFixtureAndTrack(fitID string, fixtureFile, insvcName string, opt ...bridgeutils.RenderOptions) string {
	return applyMinimalInsvcFixtureAndTrack(fitID, fixtureFile, insvcName, nil, opt...)
}

func applyMinimalInsvcFixtureAndTrack(fitID, fixtureFile, insvcName string, deploySuffixes []string, opt ...bridgeutils.RenderOptions) string {
	path := applyDeployFixtureNoDefer(fitID, fixtureFile, insvcName, opt...)
	h := registerWorkloadCleanup(workloadCleanupSpec{
		fitID: fitID, ownerName: insvcName, manifestPath: path, removeManifestFile: true,
		llmisvc: false, deploySuffixes: deploySuffixes,
	})
	_ = h
	return path
}

func pdOrchestratorInsvcDeploySuffixes(pdoOn bool) []string {
	return bridgeutils.AggregateInsvcDeploySuffixesForPDOrchestratorTest(pdoOn)
}

func beginPDOrchestratorInsvcTest(fitID, name, fixtureFile string, pdoOn bool) {
	engineCfg := name + "-vllm-aggregate-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, pdOrchestratorInsvcDeploySuffixes(pdoOn))
	path := applyDeployFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
}

func waitEngineAggregateRollout(fitID, name string, timeout time.Duration) {
	deploy := bridgeutils.DeployName(name, "engine-aggregate")
	if bridgeutils.SkipNPUWorkloadE2E() {
		bridgeutils.WaitDeploymentExists(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), deploy, timeout)
		return
	}
	bridgeutils.WaitDeploymentRollout(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), deploy, timeout)
}

func beginAggregateInsvcScalingTest(fitID, name, fixtureFile string) {
	engineCfg := name + "-vllm-aggregate-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, bridgeutils.AggregateInsvcDeploySuffixes())
	h.EnableScalingCleanup(false)
	path := applyDeployFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
}

func beginPDInsvcScalingTest(fitID, name, fixtureFile string) {
	engineCfg := name + "-vllm-pd-engine-config"
	h := beginInsvcWorkloadCleanup(fitID, name, engineCfg, bridgeutils.PDInsvcDeploySuffixes())
	bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, 2*time.Minute)
	h.EnableScalingCleanup(true)
	path := applyDeployFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
}

func beginAggregateLLMISvcScalingTest(fitID, name, fixtureFile string) {
	deploySuffixes := bridgeutils.AggregateLLMISvcDeploySuffixes()
	h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
	h.EnableScalingCleanup(false)
	path := applyLLMISvcFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
}

func beginPDLLMISvcScalingTest(fitID, name, fixtureFile string) {
	deploySuffixes := bridgeutils.PDLLMISvcDeploySuffixes()
	h := beginLLMISvcWorkloadCleanup(fitID, name, deploySuffixes)
	bridgeutils.CleanupPDRSGCloneDeploymentsFallback(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), name, 2*time.Minute)
	h.EnableScalingCleanup(true)
	path := applyLLMISvcFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
}

func requireGatewayAPI(fitID string) {
	bridgeutils.LogStep(fitID, "require Gateway API CRDs")
	if reason := bridgeutils.SkipNoKubeconfigReason(); reason != "" {
		Skip(reason)
	}
	if !bridgeutils.GatewayAPIInstalled(fitID) {
		Skip("Gateway API CRDs not installed (need gateways/httproutes/inferencepools; install Gateway API + inference extension before FIT 35+)")
	}
	bridgeutils.LogStep(fitID, "gateway API CRDs present")
}

func applyLLMISvcFixture(fitID, fixtureFile, name string) string {
	h := beginLLMISvcWorkloadCleanup(fitID, name, nil)
	path := applyLLMISvcFixtureNoDefer(fitID, fixtureFile, name)
	h.SetManifest(path, true)
	return path
}

// applyLLMISvcFixtureNoDefer applies fixture YAML without DeferCleanup (caller deletes manifest explicitly).
func applyLLMISvcFixtureNoDefer(fitID, fixtureFile, name string) string {
	ns := bridgeutils.E2ENS()
	bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, ns)
	opts := bridgeutils.RenderOptions{Namespace: ns, Name: name, KServeNS: bridgeutils.KServeNS()}
	path, err := bridgeutils.WriteRenderedFixtureOpts([]string{"llmisvc", fixtureFile}, opts)
	Expect(err).NotTo(HaveOccurred())
	stdout, stderr, err := bridgeutils.ApplyManifest(path)
	bridgeutils.LogStep(fitID, "apply llmisvc %s: stdout=%s stderr=%s err=%v", fixtureFile, stdout, stderr, err)
	Expect(err).NotTo(HaveOccurred())
	return path
}

func requireLLMISVC(fitID string) {
	bridgeutils.LogStep(fitID, "require LLMISVC (KServe CRD)")
	requireBridgeInstalled(fitID)
	if bridgeutils.SkipLLMISVCE2E() {
		Skip("SKIP_LLMISVC_E2E=true")
	}
	if !bridgeutils.CRDExists(fitID, "serving.kserve.io", "llminferenceservices") {
		Skip("LLMInferenceService CRD not found (KServe 0.17?)")
	}
	bridgeutils.LogStep(fitID, "LLMISVC CRD ok")
}

// requireKServeMutateTargetConfigs fails (no Skip) when KServe namespace lacks a webhook target config.
func requireKServeMutateTargetConfigs(fitID string) string {
	kserveNS := bridgeutils.KServeNS()
	missing := bridgeutils.MissingLLMInferenceServiceConfigs(suiteCtx, k8sClient.DynamicClient, kserveNS, bridgeutils.KServeConfigNames())
	Expect(missing).To(BeEmpty(),
		"%s: install KServe LLMInferenceServiceConfig templates in namespace %q; missing: %v",
		fitID, kserveNS, missing)
	return kserveNS
}

func validatingEngineConfigName(insvcName string) string {
	return insvcName + "-vllm-aggregate-engine-config"
}

// ValidatingFixtureCleanup: pre-delete by name, then DeferCleanup with delete -f on tracked manifests + name fallback.
type ValidatingFixtureCleanup struct {
	fitID            string
	insvcName        string
	withEngineConfig bool
	llmName          string
	manifestPaths    []string
}

func beginValidatingFixtureClean(fitID, insvcName string, withEngineConfig bool, llmName string) *ValidatingFixtureCleanup {
	c := &ValidatingFixtureCleanup{
		fitID: fitID, insvcName: insvcName, withEngineConfig: withEngineConfig, llmName: llmName,
	}
	c.deleteByName()
	DeferCleanup(c.cleanupAll)
	return c
}

// TrackManifest registers a rendered fixture path for kubectl delete -f during defer cleanup.
func (c *ValidatingFixtureCleanup) TrackManifest(path string) {
	if path != "" {
		c.manifestPaths = append(c.manifestPaths, path)
	}
}

func (c *ValidatingFixtureCleanup) deleteByName() {
	deleteValidatingFixtureResources(c.insvcName, c.withEngineConfig, c.llmName)
}

func (c *ValidatingFixtureCleanup) cleanupAll() {
	// Reverse apply order (e.g. FIT22: insvc manifest before llmisvc stub).
	for i := len(c.manifestPaths) - 1; i >= 0; i-- {
		p := c.manifestPaths[i]
		stdout, stderr, err := bridgeutils.DeleteManifest(p)
		bridgeutils.LogStep(c.fitID, "delete -f %s: stdout=%s stderr=%s err=%v", p, stdout, stderr, err)
	}
	c.deleteByName()
	for _, p := range c.manifestPaths {
		_ = os.Remove(p)
	}
	bridgeutils.LogStep(c.fitID, "defer cleanup validating fixture insvc=%s engineConfig=%v llm=%q manifests=%d",
		c.insvcName, c.withEngineConfig, c.llmName, len(c.manifestPaths))
}

// deleteValidatingFixtureResources best-effort removes InferNexService, optional engine Config, optional LLMISVC.
func deleteValidatingFixtureResources(insvcName string, withEngineConfig bool, llmName string) {
	ns := bridgeutils.E2ENS()
	tmpl := bridgeutils.TemplateNS()
	_ = bridgeutils.DeleteInferNexService(suiteCtx, k8sClient.DynamicClient, ns, insvcName)
	if withEngineConfig {
		_ = bridgeutils.DeleteInferNexServiceConfig(suiteCtx, k8sClient.DynamicClient, tmpl, validatingEngineConfigName(insvcName))
	}
	if llmName != "" {
		_ = bridgeutils.DeleteLLMInferenceService(suiteCtx, k8sClient.DynamicClient, ns, llmName)
	}
}

func expectValidatingAdmissionDenied(fitID string, fixtureParts []string, wantSubstr, insvcName string, withEngineConfig bool) {
	requireBridgeInstalled(fitID)
	ns := bridgeutils.E2ENS()
	bridgeutils.EnsureNamespace(suiteCtx, k8sClient.Clientset, ns)
	clean := beginValidatingFixtureClean(fitID, insvcName, withEngineConfig, "")
	opts := bridgeutils.RenderOptions{Namespace: ns, Name: insvcName}
	path, err := bridgeutils.WriteRenderedFixtureOpts(fixtureParts, opts)
	Expect(err).NotTo(HaveOccurred())
	clean.TrackManifest(path)
	bridgeutils.LogStep(fitID, "apply fixture %s ns=%s", path, ns)
	stdout, stderr, err := bridgeutils.ApplyManifest(path)
	bridgeutils.ExpectAdmissionDenied(fitID, stdout, stderr, err, wantSubstr)
	exists, err := bridgeutils.InferNexServiceExists(suiteCtx, k8sClient.DynamicClient, ns, insvcName)
	Expect(err).NotTo(HaveOccurred())
	Expect(exists).To(BeFalse(), "invalid InferNexService should not be persisted")
}

func expectValidatingAdmissionDeniedNoConfig(fitID string, fixtureParts []string, wantSubstr, insvcName string) {
	expectValidatingAdmissionDenied(fitID, fixtureParts, wantSubstr, insvcName, false)
}

type workloadConnectivityMode int

const (
	workloadConnectivityAggregateLLMISVC workloadConnectivityMode = iota
	workloadConnectivityPDLLMISVC
	workloadConnectivityAggregateInsvc
	workloadConnectivityPDInsvc
)

// assertPostWorkloadConnectivity runs component log health + chat/completions for full-stack deploy FITs (4–7, 56–59).
// Optional model overrides the default Qwen2.5-0.5B (FIT 56–59 pass MoEOfflineInferenceModel).
func assertPostWorkloadConnectivity(fitID, owner string, mode workloadConnectivityMode, model ...string) {
	ns := bridgeutils.E2ENS()
	if !bridgeutils.SkipNPUWorkloadE2E() {
		switch mode {
		case workloadConnectivityAggregateLLMISVC:
			bridgeutils.WaitAggregateLLMISvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, 10*time.Minute)
			bridgeutils.AssertAggregateLLMISvcLogHealth(fitID, ns, owner)
		case workloadConnectivityPDLLMISVC:
			bridgeutils.WaitPDLLMISvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, 10*time.Minute)
			bridgeutils.AssertPDLLMISvcLogHealth(fitID, ns, owner)
		case workloadConnectivityAggregateInsvc:
			bridgeutils.WaitAggregateInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, 10*time.Minute)
			bridgeutils.AssertAggregateInsvcLogHealth(fitID, ns, owner)
		case workloadConnectivityPDInsvc:
			bridgeutils.WaitPDInsvcComponentsReady(suiteCtx, k8sClient.DynamicClient, fitID, ns, owner, 10*time.Minute)
			bridgeutils.AssertPDInsvcLogHealth(fitID, ns, owner)
		}
	}
	if mode == workloadConnectivityAggregateInsvc || mode == workloadConnectivityPDInsvc {
		if !bridgeutils.SkipInferenceE2E() {
			bridgeutils.WaitInferNexEnvoyIngressServiceReady(fitID, ns, owner, 10*time.Minute)
		}
	}
	inferenceModel := ""
	if len(model) > 0 {
		inferenceModel = model[0]
	}
	if inferenceModel != "" {
		bridgeutils.AssertInferenceChatCompletionsWithModel(fitID, owner, "hello infernex "+fitID, 32, inferenceModel)
		return
	}
	bridgeutils.AssertInferenceChatCompletionsOK(fitID, owner, "hello infernex "+fitID, 32)
}

func waitEngineRolloutIfEnabled(fitID, owner string, suffixes ...string) {
	if bridgeutils.SkipNPUWorkloadE2E() {
		for _, s := range suffixes {
			bridgeutils.WaitDeploymentExists(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), bridgeutils.DeployName(owner, s), 5*time.Minute)
		}
		return
	}
	for _, s := range suffixes {
		bridgeutils.WaitDeploymentRollout(suiteCtx, k8sClient.Clientset, fitID, bridgeutils.E2ENS(), bridgeutils.DeployName(owner, s), 10*time.Minute)
	}
}
