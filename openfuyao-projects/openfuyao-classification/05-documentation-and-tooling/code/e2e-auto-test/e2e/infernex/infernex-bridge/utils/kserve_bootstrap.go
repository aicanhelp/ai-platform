package utils

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

const (
	defaultKServeGitURL = "https://github.com/kserve/kserve.git"
	defaultKServeGitRef = "v0.18.0"
	kserveInfraRel      = "hack/setup/infra"
)

// kserveInfraStep matches KServe hack/setup/infra manage.*.sh components (LLMISVC profile).
type kserveInfraStep struct {
	component        string
	method           string
	env              []string
	sharedClusterDep bool // cluster-wide (cert-manager/envoy/CRDs); still uninstalled unless SKIP_KSERVE_UNINSTALL_SHARED
}

// llmisvcInfraInstallOrder follows llmisvc-dependency-install.definition + gateway-api-crd + kserve-helm.
var llmisvcInfraInstallOrder = []kserveInfraStep{
	{component: "external-lb"},
	{component: "cert-manager", method: "helm", sharedClusterDep: true},
	{component: "gateway-api-crd", sharedClusterDep: true},
	{component: "gateway-api-extension-crd", sharedClusterDep: true},
	{component: "envoy-gateway", method: "helm", sharedClusterDep: true},
	{component: "envoy-ai-gateway", method: "helm", sharedClusterDep: true},
	{component: "gateway-api-gwclass", sharedClusterDep: true},
	{component: "gateway-api-gw"},
	{
		component: "kserve-helm",
		method:    "helm",
		env: []string{
			"ENABLE_KSERVE=false",
			"ENABLE_LLMISVC=true",
			"INSTALL_RUNTIMES=false",
			"INSTALL_LLMISVC_CONFIGS=true",
		},
	},
}

var kserveBootstrap struct {
	mu                      sync.Mutex
	workDir                 string
	repoRoot                string
	installed               bool
	teardownRun             bool
	bootstrapInstalledOrder []string // install steps run in BeforeSuite (reverse order on teardown)
}

// TeardownKServeBootstrap uninstalls cluster components (--uninstall) and deletes the temp clone.
func TeardownKServeBootstrap(fitID string) {
	kserveBootstrap.mu.Lock()
	if kserveBootstrap.teardownRun {
		kserveBootstrap.mu.Unlock()
		return
	}
	kserveBootstrap.teardownRun = true

	repo := kserveBootstrap.repoRoot
	installed := kserveBootstrap.installed
	workDir := kserveBootstrap.workDir
	order := append([]string(nil), kserveBootstrap.bootstrapInstalledOrder...)
	kserveBootstrap.workDir = ""
	kserveBootstrap.repoRoot = ""
	kserveBootstrap.installed = false
	kserveBootstrap.bootstrapInstalledOrder = nil
	kserveBootstrap.mu.Unlock()

	if installed && !SkipKServeUninstall() && repo != "" {
		LogStep(fitID, "KServe bootstrap teardown: --uninstall all BeforeSuite infra steps (incl. shared deps unless SKIP_KSERVE_UNINSTALL_SHARED)")
		if err := uninstallKServeInfra(fitID, repo, order); err != nil {
			LogStep(fitID, "KServe uninstall warnings: %v", err)
		}
	}
	if workDir != "" {
		LogStep(fitID, "removing temp KServe workspace %s", workDir)
		if err := os.RemoveAll(workDir); err != nil {
			LogStep(fitID, "remove temp workspace: %v", err)
		}
	}
}

func beginKServeWorkspace(fitID string) (repoRoot string, err error) {
	kserveBootstrap.mu.Lock()
	defer kserveBootstrap.mu.Unlock()
	if kserveBootstrap.repoRoot != "" {
		return kserveBootstrap.repoRoot, nil
	}
	// Optional local override for debugging only; default path is always a temp clone.
	if p := strings.TrimSpace(os.Getenv("KSERVE_SOURCE_DIR")); p != "" {
		if st, e := os.Stat(p); e != nil || !st.IsDir() {
			return "", fmt.Errorf("KSERVE_SOURCE_DIR %q: %w", p, e)
		}
		kserveBootstrap.repoRoot = filepath.Clean(p)
		LogStep(fitID, "KServe workspace: KSERVE_SOURCE_DIR=%s (debug override; not deleted on teardown)", kserveBootstrap.repoRoot)
		return kserveBootstrap.repoRoot, nil
	}

	workDir, err := os.MkdirTemp("", "infernex-bridge-kserve-")
	if err != nil {
		return "", err
	}
	kserveBootstrap.workDir = workDir
	dest := filepath.Join(workDir, "kserve")
	url := KServeGitURL()
	ref := KServeGitRef()
	LogStep(fitID, "temp KServe workspace %s — clone %s checkout %s (KSERVE_VERSION / helm SET_KSERVE_VERSION)", workDir, url, ref)
	if err := cloneAndCheckoutKServe(fitID, url, ref, dest); err != nil {
		_ = os.RemoveAll(workDir)
		kserveBootstrap.workDir = ""
		return "", err
	}
	kserveBootstrap.repoRoot = dest
	return kserveBootstrap.repoRoot, nil
}

// cloneAndCheckoutKServe clones upstream into dest then checks out ref (tag or branch).
func cloneAndCheckoutKServe(fitID, url, ref, dest string) error {
	cloneCmd := fmt.Sprintf("git clone --filter=blob:none --no-checkout %q %q", url, dest)
	stdout, stderr, err := ExecLocal(cloneCmd)
	LogStep(fitID, "git clone: stdout=%s stderr=%s err=%v", trimForLog(stdout), trimForLog(stderr), err)
	if err != nil {
		return fmt.Errorf("git clone: %w (%s)", err, strings.TrimSpace(stderr))
	}
	fetchCmd := fmt.Sprintf(
		`cd %q && (git fetch --depth 1 origin tag %q 2>/dev/null || git fetch --depth 1 origin %q)`,
		dest, ref, ref,
	)
	stdout, stderr, err = ExecLocal(fetchCmd)
	LogStep(fitID, "git fetch %s: stdout=%s stderr=%s err=%v", ref, trimForLog(stdout), trimForLog(stderr), err)
	if err != nil {
		return fmt.Errorf("git fetch %s: %w (%s)", ref, err, strings.TrimSpace(stderr))
	}
	checkoutCmd := fmt.Sprintf("git -C %q checkout %q", dest, ref)
	stdout, stderr, err = ExecLocal(checkoutCmd)
	LogStep(fitID, "git checkout %s: stdout=%s stderr=%s err=%v", ref, trimForLog(stdout), trimForLog(stderr), err)
	if err != nil {
		return fmt.Errorf("git checkout %s: %w (%s)", ref, err, strings.TrimSpace(stderr))
	}
	return nil
}

// normalizeKServeGitRef maps bare semver (0.18.0) to release tags (v0.18.0).
func normalizeKServeGitRef(v string) string {
	v = strings.TrimSpace(v)
	if v == "" {
		return defaultKServeGitRef
	}
	if strings.HasPrefix(v, "v") || !looksLikeSemverVersion(v) {
		return v
	}
	return "v" + v
}

func looksLikeSemverVersion(v string) bool {
	base := v
	if i := strings.IndexByte(v, '-'); i >= 0 {
		base = v[:i]
	}
	parts := strings.Split(base, ".")
	if len(parts) < 2 {
		return false
	}
	for _, p := range parts {
		if p == "" {
			return false
		}
		for _, r := range p {
			if r < '0' || r > '9' {
				return false
			}
		}
	}
	return true
}

func kserveInfraEnvExports(extraEnv []string) []string {
	exports := []string{fmt.Sprintf("KUBECONFIG=%q", localKubeconfig())}
	if v := KServeHelmVersion(); v != "" {
		exports = append(exports, fmt.Sprintf("SET_KSERVE_VERSION=%q", v))
	}
	return append(exports, extraEnv...)
}

func findKServeInfraScript(repoRoot, component, method string) (string, error) {
	infraDir := filepath.Join(repoRoot, filepath.FromSlash(kserveInfraRel))
	parts := strings.Split(component, "-")
	var folders []string
	for i := len(parts); i > 0; i-- {
		folders = append(folders, strings.Join(parts[:i], "-"))
	}
	folders = append(folders, "")

	for _, folder := range folders {
		base := infraDir
		if folder != "" {
			base = filepath.Join(infraDir, folder)
		}
		if method != "" {
			p := filepath.Join(base, fmt.Sprintf("manage.%s-%s.sh", component, method))
			if st, err := os.Stat(p); err == nil && !st.IsDir() {
				return p, nil
			}
		}
		p := filepath.Join(base, fmt.Sprintf("manage.%s.sh", component))
		if st, err := os.Stat(p); err == nil && !st.IsDir() {
			return p, nil
		}
	}
	return "", fmt.Errorf("manage script not found for component %q (method=%q) under %s", component, method, infraDir)
}

func runKServeInfraScript(fitID, script string, uninstall bool, extraEnv []string) (stdout string, err error) {
	flags := ""
	if uninstall {
		flags = " --uninstall"
	}
	envPrefix := strings.Join(kserveInfraEnvExports(extraEnv), " ")
	cmd := fmt.Sprintf("%s bash %q%s", envPrefix, script, flags)
	LogStep(fitID, "infra script: %s", filepath.Base(script))
	stdout, stderr, err := ExecLocal(cmd)
	LogStep(fitID, "  stdout: %s", trimForLog(stdout))
	if stderr != "" {
		LogStep(fitID, "  stderr: %s", trimForLog(stderr))
	}
	return stdout + "\n" + stderr, err
}

func scriptOutputSkippedPreexisting(out string) bool {
	lower := strings.ToLower(out)
	return strings.Contains(lower, "already installed") ||
		strings.Contains(lower, "already exists. use --reinstall")
}

func kserveInfraStepsForInstall() []kserveInfraStep {
	steps := llmisvcInfraInstallOrder
	if SkipKServeExternalLB() {
		var filtered []kserveInfraStep
		for _, s := range steps {
			if s.component != "external-lb" {
				filtered = append(filtered, s)
			}
		}
		return filtered
	}
	return steps
}

func lookupKServeInfraStep(component string) (kserveInfraStep, bool) {
	for _, s := range llmisvcInfraInstallOrder {
		if s.component == component {
			return s, true
		}
	}
	return kserveInfraStep{}, false
}

func recordBootstrapInstallStep(component string) {
	kserveBootstrap.mu.Lock()
	defer kserveBootstrap.mu.Unlock()
	for _, c := range kserveBootstrap.bootstrapInstalledOrder {
		if c == component {
			return
		}
	}
	kserveBootstrap.bootstrapInstalledOrder = append(kserveBootstrap.bootstrapInstalledOrder, component)
}

func runKServeInfraInstallStep(fitID, repoRoot string, step kserveInfraStep) error {
	script, err := findKServeInfraScript(repoRoot, step.component, step.method)
	if err != nil {
		return err
	}
	env := append([]string(nil), step.env...)
	flags := ""
	if step.component == "kserve-helm" && KServeHelmReinstall() {
		flags = " --reinstall"
	}
	cmd := fmt.Sprintf("%s bash %q%s", strings.Join(kserveInfraEnvExports(env), " "), script, flags)
	LogStep(fitID, "infra install: %s", filepath.Base(script))
	stdout, stderr, err := ExecLocal(cmd)
	combined := stdout + stderr
	LogStep(fitID, "  output: %s", trimForLog(combined))
	if err != nil {
		return fmt.Errorf("%s install: %w", step.component, err)
	}
	recordBootstrapInstallStep(step.component)
	if scriptOutputSkippedPreexisting(combined) {
		LogStep(fitID, "  %s reported pre-existing — will still --uninstall on teardown", step.component)
	}
	return nil
}

// prepareKServeClusterForInstall fixes common partial-uninstall leftovers on shared clusters.
func prepareKServeClusterForInstall(fitID string) {
	unstuckKServeNamespace(fitID)
	cleanupOrphanLLMISVCManagerRBAC(fitID)
}

func unstuckKServeNamespace(fitID string) {
	ns := KServeNS()
	cmd := fmt.Sprintf("KUBECONFIG=%q kubectl get ns %s -o jsonpath='{.status.phase}' 2>/dev/null", localKubeconfig(), ns)
	phase, _, _ := ExecLocal(cmd)
	if strings.TrimSpace(phase) != "Terminating" {
		return
	}
	LogStep(fitID, "namespace %s is Terminating — removing stale kserve validating webhooks", ns)
	for _, wh := range []string{
		"llminferenceservice.serving.kserve.io",
		"llminferenceserviceconfig.serving.kserve.io",
	} {
		del := fmt.Sprintf("KUBECONFIG=%q kubectl delete validatingwebhookconfiguration %s --ignore-not-found", localKubeconfig(), wh)
		_, _, _ = ExecLocal(del)
	}
	patch := fmt.Sprintf(
		`KUBECONFIG=%q kubectl get ns %s -o json | python3 -c "import json,sys; d=json.load(sys.stdin); d['spec']['finalizers']=[]; print(json.dumps(d))" | kubectl replace --raw /api/v1/namespaces/%s/finalize -f -`,
		localKubeconfig(), ns, ns,
	)
	_, _, _ = ExecLocal(patch)
}

func cleanupOrphanLLMISVCManagerRBAC(fitID string) {
	checkRelease := fmt.Sprintf(
		"KUBECONFIG=%q helm list -n %s -q 2>/dev/null | grep -x kserve-llmisvc-resources",
		localKubeconfig(), KServeNS(),
	)
	out, _, _ := ExecLocal(checkRelease)
	if strings.TrimSpace(out) != "" {
		return
	}
	LogStep(fitID, "no helm release kserve-llmisvc-resources — remove orphan llmisvc ClusterRole if present")
	for _, obj := range []struct{ kind, name string }{
		{"clusterrole", "kserve-llmisvc-manager-role"},
		{"clusterrolebinding", "llmisvc-manager-rolebinding"},
	} {
		del := fmt.Sprintf("KUBECONFIG=%q kubectl delete %s %s --ignore-not-found", localKubeconfig(), obj.kind, obj.name)
		_, _, _ = ExecLocal(del)
	}
}

func installKServeInfra(fitID, repoRoot string) error {
	if err := EnsureHelmRepos(fitID); err != nil {
		return err
	}
	prepareKServeClusterForInstall(fitID)

	steps := kserveInfraStepsForInstall()
	if SkipKServeExternalLB() {
		LogStep(fitID, "KSERVE_SKIP_EXTERNAL_LB=true — skip external-lb")
	}

	for _, step := range steps {
		if err := runKServeInfraInstallStep(fitID, repoRoot, step); err != nil {
			return err
		}
	}
	return nil
}

func uninstallKServeInfra(fitID, repoRoot string, order []string) error {
	if len(order) == 0 {
		LogStep(fitID, "no KServe bootstrap install steps recorded — skip infra uninstall")
		return nil
	}

	var agg []string
	for i := len(order) - 1; i >= 0; i-- {
		component := order[i]
		step, ok := lookupKServeInfraStep(component)
		if !ok {
			agg = append(agg, fmt.Sprintf("unknown bootstrap component %q", component))
			continue
		}
		if SkipKServeUninstallShared() && step.sharedClusterDep {
			LogStep(fitID, "skip uninstall %s (SKIP_KSERVE_UNINSTALL_SHARED=true)", step.component)
			continue
		}
		script, err := findKServeInfraScript(repoRoot, step.component, step.method)
		if err != nil {
			agg = append(agg, err.Error())
			continue
		}
		if _, err := runKServeInfraScript(fitID, script, true, step.env); err != nil {
			agg = append(agg, fmt.Sprintf("%s: %v", step.component, err))
		}
	}
	if len(agg) > 0 {
		return fmt.Errorf("%s", strings.Join(agg, "; "))
	}
	return nil
}

// InstallKServeLLMISVCStack clones KServe into a temp dir and runs hack/setup/infra manage.*.sh scripts.
func InstallKServeLLMISVCStack(fitID string) error {
	if SkipKServeInstall() {
		LogStep(fitID, "SKIP_KSERVE_INSTALL=true")
		return nil
	}
	repoRoot, err := beginKServeWorkspace(fitID)
	if err != nil {
		return err
	}
	if err := installKServeInfra(fitID, repoRoot); err != nil {
		return err
	}
	restartAIGatewayControllerForInferencePoolCRD(fitID)
	kserveBootstrap.mu.Lock()
	kserveBootstrap.installed = true
	kserveBootstrap.mu.Unlock()
	return nil
}

// InstallGatewayAPIStack installs gateway dependencies only (no kserve-helm).
func InstallGatewayAPIStack(fitID string) error {
	if SkipGatewayStackInstall() {
		return nil
	}
	repoRoot, err := beginKServeWorkspace(fitID)
	if err != nil {
		return err
	}
	for _, step := range kserveInfraStepsForInstall() {
		if step.component == "kserve-helm" {
			continue
		}
		if err := runKServeInfraInstallStep(fitID, repoRoot, step); err != nil {
			return fmt.Errorf("%s: %w", step.component, err)
		}
	}
	kserveBootstrap.mu.Lock()
	kserveBootstrap.installed = true
	kserveBootstrap.mu.Unlock()
	return nil
}

// WaitKServeLLMISVCControllerReady waits for llmisvc-controller-manager in KServe namespace.
func WaitKServeLLMISVCControllerReady(ctx context.Context, client kubernetes.Interface, fitID string, timeout time.Duration) {
	ns := KServeNS()
	LogPhaseBegin(fitID, "kserve llmisvc controller Ready")
	defer LogPhaseEnd(fitID, "kserve llmisvc controller Ready")
	Eventually(func(g Gomega) {
		deploys, err := client.AppsV1().Deployments(ns).List(ctx, metav1.ListOptions{})
		g.Expect(err).NotTo(HaveOccurred())
		ready := false
		for _, d := range deploys.Items {
			if strings.Contains(d.Name, "llmisvc-controller") && d.Status.ReadyReplicas >= 1 {
				ready = true
				LogStep(fitID, "llmisvc deployment %s readyReplicas=%d", d.Name, d.Status.ReadyReplicas)
				break
			}
		}
		g.Expect(ready).To(BeTrueBecause("timeout waiting for llmisvc-controller in ns=%s", ns))
	}).WithTimeout(timeout).WithPolling(10 * time.Second).Should(Succeed())
}

func bootstrapKServeStack(fitID string) error {
	return InstallKServeLLMISVCStack(fitID)
}

// restartAIGatewayControllerForInferencePoolCRD restarts the ai-gateway-controller deployment
// once kserve-helm has installed the inferencepools.inference.networking.k8s.io (v1) CRD.
//
// Install order: envoy-ai-gateway (starts the controller) runs BEFORE kserve-helm (installs the
// GIE v1 CRD). The controller probes for the InferencePool CRD only once at startup; when it is
// absent the InferencePool controller is permanently skipped, so InferencePool status.parents is
// never written and KServe LLMInferenceService stays Ready=False / WaitingForGateway forever.
// Restarting the pod after the CRD exists makes the controller register the InferencePool
// controller, which then writes status.parents (Accepted) that KServe waits on.
func restartAIGatewayControllerForInferencePoolCRD(fitID string) {
	kc := localKubeconfig()
	ns := AIGatewaySystemNS()

	crdOut, _, _ := ExecLocal(fmt.Sprintf("KUBECONFIG=%q kubectl get crd inferencepools.inference.networking.k8s.io --ignore-not-found -o name", kc))
	if strings.TrimSpace(crdOut) == "" {
		LogStep(fitID, "inferencepools.inference.networking.k8s.io CRD not installed — skip ai-gateway-controller restart")
		return
	}
	depOut, _, _ := ExecLocal(fmt.Sprintf("KUBECONFIG=%q kubectl get deploy -n %s ai-gateway-controller --ignore-not-found -o name", kc, ns))
	if strings.TrimSpace(depOut) == "" {
		LogStep(fitID, "ai-gateway-controller not found in ns=%s — skip restart", ns)
		return
	}
	LogStep(fitID, "restarting ai-gateway-controller so it registers the InferencePool controller (v1 CRD installed by kserve-helm after controller startup)")
	if _, _, err := ExecLocal(fmt.Sprintf("KUBECONFIG=%q kubectl rollout restart -n %s deploy/ai-gateway-controller", kc, ns)); err != nil {
		LogStep(fitID, "  rollout restart: %v", err)
		return
	}
	if _, _, err := ExecLocal(fmt.Sprintf("KUBECONFIG=%q kubectl rollout status -n %s deploy/ai-gateway-controller --timeout=180s", kc, ns)); err != nil {
		LogStep(fitID, "  rollout status: %v", err)
		return
	}
	LogStep(fitID, "ai-gateway-controller restarted — InferencePool controller registered")
}

func bootstrapGatewayStack(fitID string) error {
	if !SkipKServeInstall() {
		LogStep(fitID, "gateway infra installed with KServe LLMISVC bootstrap")
		return nil
	}
	return InstallGatewayAPIStack(fitID)
}
