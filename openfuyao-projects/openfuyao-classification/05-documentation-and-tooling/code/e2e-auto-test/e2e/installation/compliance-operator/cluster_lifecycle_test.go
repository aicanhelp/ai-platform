package compliance_operator

import (
	"fmt"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	testenv "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	installationutils "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
	. "github.com/onsi/ginkgo/v2"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

const (
	workloadInstallTimeout   = 20 * time.Minute
	workloadUninstallTimeout = 15 * time.Minute
	workloadPollInterval     = 30 * time.Second
)

// workloadClusterLifecycle records every resource as soon as it is created so
// AfterAll can still perform best-effort cleanup when setup fails partway.
type workloadClusterLifecycle struct {
	guideExecutor             *executor.SSHExecutor
	clusterManager            *installationutils.ClusterManager
	managementKubeconfigPath  string
	managementKubeconfigOwned bool
	clusterName               string
	clusterConfigPath         string
	nodeConfigPath            string
	kubeconfigPath            string
	entryKubeconfigPath       string
	entryKubeconfigBackup     string
	entryKubeconfigExisted    bool
	entryKubeconfigPrepared   bool
	vipHost                   string
	vipPort                   string
}

func newWorkloadClusterLifecycle() (*workloadClusterLifecycle, error) {
	if err := loadComplianceEnvironment(); err != nil {
		return nil, err
	}

	guide := config.LoadGuideNodeFromEnv()
	if guide.Host == "" || guide.Password == "" {
		return nil, fmt.Errorf("GUIDE_NODE_HOST and GUIDE_NODE_PASSWORD must be provided by .env or system environment variables")
	}
	guideExecutor, err := executor.NewSSHExecutor(guide.Host, guide.Port, guide.Username, guide.Password)
	if err != nil {
		return nil, fmt.Errorf("connect to management/bootstrap node: %w", err)
	}

	vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv()
	if vipHost == "" || vipPort == "" {
		_ = guideExecutor.Close()
		return nil, fmt.Errorf("WORKLOAD_CLUSTER_VIP_HOST and WORKLOAD_CLUSTER_VIP_PORT must be configured for the 3M1W compliance cluster")
	}

	lifecycle := &workloadClusterLifecycle{
		guideExecutor:  guideExecutor,
		clusterManager: installationutils.NewClusterManager(guideExecutor, nil),
		clusterName:    fmt.Sprintf("compliance-e2e-3m1w-%d", time.Now().Unix()),
		vipHost:        vipHost,
		vipPort:        vipPort,
	}
	if err := lifecycle.prepareManagementKubeconfig(); err != nil {
		_ = guideExecutor.Close()
		return nil, err
	}
	return lifecycle, nil
}

// loadComplianceEnvironment keeps CI configuration isolated from developer
// files. When every mandatory value is already exported by the runner, .env is
// not read at all, so stale local values cannot be mixed into a pipeline run.
// Local execution retains the existing .env fallback when process variables are
// incomplete.
func loadComplianceEnvironment() error {
	required := []string{
		"GUIDE_NODE_HOST",
		"GUIDE_NODE_PASSWORD",
		"WORKLOAD_CLUSTER_VIP_HOST",
	}
	for i := 2; i <= 5; i++ {
		required = append(required,
			fmt.Sprintf("TEST_NODE%d_IP", i),
			fmt.Sprintf("TEST_NODE%d_PASSWORD", i),
		)
	}

	missing := make([]string, 0, len(required))
	for _, name := range required {
		if strings.TrimSpace(os.Getenv(name)) == "" {
			missing = append(missing, name)
		}
	}
	if len(missing) == 0 {
		GinkgoWriter.Printf("using compliance configuration exclusively from process environment variables\n")
		return nil
	}

	GinkgoWriter.Printf("process environment is incomplete (%s); loading e2e/.env for local compatibility\n", strings.Join(missing, ","))
	if err := testenv.LoadEnv(".env"); err != nil {
		return fmt.Errorf("load E2E environment: %w", err)
	}
	return nil
}

// prepareManagementKubeconfig resolves KUBECONFIG without assuming whether its
// path belongs to GUIDE_NODE or the runner. A path already usable on GUIDE_NODE
// wins. Otherwise a runner-local config is flattened and copied to GUIDE_NODE.
// If KUBECONFIG is unset, or neither location is usable, the lifecycle retains
// GUIDE_NODE's default kubectl context; verifyManagementClusterReady validates
// the selected context before it can mutate cluster resources.
func (l *workloadClusterLifecycle) prepareManagementKubeconfig() error {
	kubeconfigEnv := strings.TrimSpace(os.Getenv("KUBECONFIG"))
	if kubeconfigEnv == "" {
		GinkgoWriter.Printf("KUBECONFIG is not provided; using GUIDE_NODE default kubectl context\n")
		return nil
	}

	if usable, reason := l.remoteKubeconfigUsable(kubeconfigEnv); usable {
		l.managementKubeconfigPath = kubeconfigEnv
		GinkgoWriter.Printf("using KUBECONFIG path already available on GUIDE_NODE: %s\n", kubeconfigEnv)
		return nil
	} else {
		GinkgoWriter.Printf("KUBECONFIG path is not usable on GUIDE_NODE (%s); checking runner-local file\n", reason)
	}

	loadingRules := &clientcmd.ClientConfigLoadingRules{Precedence: filepath.SplitList(kubeconfigEnv)}
	rawConfig, err := loadingRules.Load()
	if err != nil {
		GinkgoWriter.Printf("runner KUBECONFIG %q is not readable (%v); using GUIDE_NODE default kubectl context\n", kubeconfigEnv, err)
		return nil
	}
	if err := clientcmdapi.FlattenConfig(rawConfig); err != nil {
		GinkgoWriter.Printf("runner KUBECONFIG %q cannot be flattened (%v); using GUIDE_NODE default kubectl context\n", kubeconfigEnv, err)
		return nil
	}
	data, err := clientcmd.Write(*rawConfig)
	if err != nil {
		GinkgoWriter.Printf("runner KUBECONFIG %q cannot be serialized (%v); using GUIDE_NODE default kubectl context\n", kubeconfigEnv, err)
		return nil
	}

	remotePath := fmt.Sprintf("/tmp/compliance-management-kubeconfig-%s", l.clusterName)
	writeCommand := fmt.Sprintf("umask 077; cat > %s", shellQuote(remotePath))
	result, err := l.guideExecutor.ExecWithInput(writeCommand, string(data))
	if err != nil || result == nil || result.ExitCode != 0 {
		_, _ = l.guideExecutor.Exec(fmt.Sprintf("rm -f %s", shellQuote(remotePath)))
		stderr := ""
		if result != nil {
			stderr = strings.TrimSpace(result.Stderr)
		}
		GinkgoWriter.Printf("copy runner KUBECONFIG to GUIDE_NODE failed (err=%v stderr=%s); using GUIDE_NODE default kubectl context\n", err, stderr)
		return nil
	}

	if usable, reason := l.remoteKubeconfigUsable(remotePath); !usable {
		_, _ = l.guideExecutor.Exec(fmt.Sprintf("rm -f %s", shellQuote(remotePath)))
		GinkgoWriter.Printf("runner KUBECONFIG copied to GUIDE_NODE is not usable (%s); using GUIDE_NODE default kubectl context\n", reason)
		return nil
	}

	l.managementKubeconfigPath = remotePath
	l.managementKubeconfigOwned = true
	GinkgoWriter.Printf("using runner KUBECONFIG copied to GUIDE_NODE: %s\n", remotePath)
	return nil
}

func (l *workloadClusterLifecycle) remoteKubeconfigUsable(kubeconfigPath string) (bool, string) {
	command := fmt.Sprintf(`export KUBECONFIG=%s
kubectl cluster-info --request-timeout=5s >/dev/null &&
kubectl get crd bkeclusters.bke.bocloud.com --no-headers >/dev/null &&
test "$(kubectl -n cluster-system get deployment bke-controller-manager -o jsonpath='{.status.availableReplicas}')" -ge 1`, shellQuote(kubeconfigPath))
	result, err := l.guideExecutor.Exec(command)
	if err == nil && result != nil && result.ExitCode == 0 {
		return true, ""
	}
	stderr := ""
	if result != nil {
		stderr = strings.TrimSpace(result.Stderr)
	}
	return false, fmt.Sprintf("err=%v stderr=%s", err, stderr)
}

func (l *workloadClusterLifecycle) managementKubectl(command string) string {
	if l.managementKubeconfigPath == "" {
		return "kubectl " + command
	}
	return fmt.Sprintf("KUBECONFIG=%s kubectl %s", shellQuote(l.managementKubeconfigPath), command)
}

// verifyManagementClusterReady confirms that the selected management kubeconfig
// points at an initialized management environment. It is deliberately read-only:
// bootstrap deployment and bke init happen before this suite starts.
func (l *workloadClusterLifecycle) verifyManagementClusterReady() error {
	result, err := l.guideExecutor.Exec(l.managementKubectl("cluster-info --request-timeout=5s"))
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("existing management cluster API is not reachable: %v", err)
	}

	result, err = l.guideExecutor.Exec(l.managementKubectl("get crd bkeclusters.bke.bocloud.com --no-headers"))
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("existing management cluster is missing the BKECluster CRD; run bke init before starting this suite: %v", err)
	}

	result, err = l.guideExecutor.Exec(l.managementKubectl("-n cluster-system get deployment bke-controller-manager -o jsonpath='{.status.availableReplicas}'"))
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("existing management cluster bke-controller-manager is unavailable: %v", err)
	}
	availableReplicas, err := strconv.Atoi(strings.TrimSpace(result.Stdout))
	if err != nil || availableReplicas < 1 {
		return fmt.Errorf("existing management cluster bke-controller-manager has no available replicas: %q", strings.TrimSpace(result.Stdout))
	}

	kubeconfigSource := "GUIDE_NODE default context"
	if l.managementKubeconfigPath != "" {
		kubeconfigSource = l.managementKubeconfigPath
	}
	GinkgoWriter.Printf("existing management cluster is ready (kubeconfig=%s, bke-controller-manager availableReplicas=%d)\n", kubeconfigSource, availableReplicas)
	return nil
}

func (l *workloadClusterLifecycle) create() error {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return fmt.Errorf("3Master1Worker compliance cluster requires TEST_NODE2 through TEST_NODE5")
	}
	clusterNodes := append([]config.NodeInfo(nil), nodes[:4]...)
	for i := 0; i < 3; i++ {
		clusterNodes[i].Role = []string{"master/node", "etcd"}
	}
	clusterNodes[3].Role = []string{"node"}

	clusterConfig := config.NewBKEClusterConfigForHA(l.clusterName, clusterNodes, l.vipHost, l.vipPort)
	By(fmt.Sprintf("生成 3Master1Worker workload cluster 配置（controlPlaneEndpoint=%s:%s）", l.vipHost, l.vipPort))
	var err error
	l.clusterConfigPath, l.nodeConfigPath, err = l.clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return fmt.Errorf("generate workload cluster config: %w", err)
	}

	By("通过 management cluster 创建 compliance workload cluster")
	if err = l.clusterManager.CreateClusterWithKubeconfig(l.clusterConfigPath, l.nodeConfigPath, l.managementKubeconfigPath); err != nil {
		return fmt.Errorf("create workload cluster: %w", err)
	}

	By("等待 compliance workload cluster 达到 Ready")
	deadline := time.Now().Add(workloadInstallTimeout)
	for time.Now().Before(deadline) {
		phase, state, clusterStatus, statusErr := l.clusterManager.GetClusterFullStatusWithKubeconfig(l.clusterName, l.managementKubeconfigPath)
		GinkgoWriter.Printf("workload cluster status: phase=%s state=%s clusterStatus=%s err=%v\n", phase, state, clusterStatus, statusErr)
		if statusErr == nil && clusterStatus == "Ready" {
			break
		}
		if state == "DeployFailed" || clusterStatus == "InitializationFailed" {
			return fmt.Errorf("workload cluster entered failure state: phase=%s state=%s clusterStatus=%s", phase, state, clusterStatus)
		}
		time.Sleep(workloadPollInterval)
	}
	phase, state, clusterStatus, err := l.clusterManager.GetClusterFullStatusWithKubeconfig(l.clusterName, l.managementKubeconfigPath)
	if err != nil || clusterStatus != "Ready" {
		return fmt.Errorf("workload cluster did not become Ready: phase=%s state=%s clusterStatus=%s err=%v", phase, state, clusterStatus, err)
	}

	By("从 management cluster 获取 workload kubeconfig 并校验 API endpoint")
	checker := installationutils.NewClusterCheckerWithParentKubeconfig(l.guideExecutor, l.clusterName, l.managementKubeconfigPath)
	l.kubeconfigPath, err = checker.SaveKubeconfigToFile()
	if err != nil {
		return fmt.Errorf("save workload kubeconfig: %w", err)
	}
	result, err := l.guideExecutor.Exec(fmt.Sprintf("KUBECONFIG=%s kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'", l.kubeconfigPath))
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("read workload kubeconfig endpoint: err=%v", err)
	}
	if err := requireEndpoint(strings.TrimSpace(result.Stdout), l.vipHost, l.vipPort); err != nil {
		return fmt.Errorf("workload kubeconfig does not use configured VIP: %w", err)
	}
	return nil
}

// prepareEntryContext installs the workload kubeconfig fetched from the
// management cluster as the entry node's default kubectl context. Existing
// content is backed up and restored by AfterAll.
func (l *workloadClusterLifecycle) prepareEntryContext(entryExecutor *executor.SSHExecutor) error {
	encodedResult, err := l.guideExecutor.Exec(fmt.Sprintf("base64 -w0 %s", shellQuote(l.kubeconfigPath)))
	if err != nil || encodedResult == nil || encodedResult.ExitCode != 0 || strings.TrimSpace(encodedResult.Stdout) == "" {
		return fmt.Errorf("read workload kubeconfig from guide node: %v", err)
	}

	pathResult, err := entryExecutor.Exec(`printf '%s' "${KUBECONFIG:-$HOME/.kube/config}"`)
	if err != nil || pathResult == nil || pathResult.ExitCode != 0 || strings.TrimSpace(pathResult.Stdout) == "" {
		return fmt.Errorf("resolve entry node default kubeconfig path: %v", err)
	}
	l.entryKubeconfigPath = strings.TrimSpace(pathResult.Stdout)
	l.entryKubeconfigBackup = fmt.Sprintf("/tmp/compliance-entry-kubeconfig-%s.bak", l.clusterName)

	existsResult, err := entryExecutor.Exec(fmt.Sprintf("test -f %s && echo yes || echo no", shellQuote(l.entryKubeconfigPath)))
	if err != nil || existsResult == nil || existsResult.ExitCode != 0 {
		return fmt.Errorf("inspect entry node kubeconfig: %v", err)
	}
	l.entryKubeconfigExisted = strings.TrimSpace(existsResult.Stdout) == "yes"
	if l.entryKubeconfigExisted {
		result, backupErr := entryExecutor.Exec(fmt.Sprintf("cp -p %s %s", shellQuote(l.entryKubeconfigPath), shellQuote(l.entryKubeconfigBackup)))
		if backupErr != nil || result == nil || result.ExitCode != 0 {
			return fmt.Errorf("backup entry node kubeconfig: %v", backupErr)
		}
	}
	l.entryKubeconfigPrepared = true

	encoded := strings.TrimSpace(encodedResult.Stdout)
	temporaryPath := l.entryKubeconfigPath + ".compliance-e2e.tmp"
	writeCommand := fmt.Sprintf(
		"mkdir -p %s && printf '%%s' %s | base64 -d > %s && chmod 600 %s && mv %s %s",
		shellQuote(filepath.Dir(l.entryKubeconfigPath)), shellQuote(encoded),
		shellQuote(temporaryPath), shellQuote(temporaryPath),
		shellQuote(temporaryPath), shellQuote(l.entryKubeconfigPath),
	)
	result, err := entryExecutor.Exec(writeCommand)
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("install workload kubeconfig on entry node: %v", err)
	}
	return l.verifyEntryContext(entryExecutor)
}

func (l *workloadClusterLifecycle) verifyEntryContext(entryExecutor *executor.SSHExecutor) error {
	result, err := entryExecutor.Exec("kubectl config view --minify -o jsonpath='{.clusters[0].cluster.server}'")
	if err != nil || result == nil || result.ExitCode != 0 {
		return fmt.Errorf("read entry node default kubeconfig endpoint: %w", err)
	}
	return requireEndpoint(strings.TrimSpace(result.Stdout), l.vipHost, l.vipPort)
}

func (l *workloadClusterLifecycle) restoreEntryContext(entryExecutor *executor.SSHExecutor) error {
	if l == nil || !l.entryKubeconfigPrepared || entryExecutor == nil {
		return nil
	}
	if l.entryKubeconfigExisted {
		result, err := entryExecutor.Exec(fmt.Sprintf("mv %s %s", shellQuote(l.entryKubeconfigBackup), shellQuote(l.entryKubeconfigPath)))
		if err != nil || result == nil || result.ExitCode != 0 {
			return fmt.Errorf("restore entry node kubeconfig: %v", err)
		}
	} else {
		result, err := entryExecutor.Exec(fmt.Sprintf("rm -f %s %s", shellQuote(l.entryKubeconfigPath), shellQuote(l.entryKubeconfigBackup)))
		if err != nil || result == nil || result.ExitCode != 0 {
			return fmt.Errorf("remove temporary entry node kubeconfig: %v", err)
		}
	}
	l.entryKubeconfigPrepared = false
	return nil
}

func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func (l *workloadClusterLifecycle) cleanup() {
	if l == nil {
		return
	}
	if l.guideExecutor != nil {
		defer l.guideExecutor.Close()
	}
	defer l.cleanupManagementKubeconfig()
	if l.clusterManager != nil && l.clusterName != "" {
		By("删除本轮 compliance workload cluster")
		if err := l.clusterManager.DeleteClusterWithKubeconfig(l.clusterName, l.managementKubeconfigPath); err != nil {
			GinkgoWriter.Printf("trigger workload cluster deletion failed: %v\n", err)
		}

		deadline := time.Now().Add(workloadUninstallTimeout)
		for l.clusterManager.ClusterExistsWithKubeconfig(l.clusterName, l.managementKubeconfigPath) && time.Now().Before(deadline) {
			phase, state, clusterStatus, err := l.clusterManager.GetClusterFullStatusWithKubeconfig(l.clusterName, l.managementKubeconfigPath)
			GinkgoWriter.Printf("waiting for workload cluster deletion: phase=%s state=%s clusterStatus=%s err=%v\n", phase, state, clusterStatus, err)
			time.Sleep(workloadPollInterval)
		}

		if err := l.clusterManager.CleanupClusterNamespaceWithKubeconfig(l.clusterName, l.managementKubeconfigPath); err != nil {
			GinkgoWriter.Printf("cleanup workload cluster namespace failed: %v\n", err)
		}
		for _, path := range []string{l.clusterConfigPath, l.nodeConfigPath, l.kubeconfigPath} {
			if path == "" {
				continue
			}
			if err := l.clusterManager.CleanupConfig(path); err != nil {
				GinkgoWriter.Printf("cleanup generated file %s failed: %v\n", path, err)
			}
		}
		stillExists := l.clusterManager.ClusterExistsWithKubeconfig(l.clusterName, l.managementKubeconfigPath)
		if stillExists {
			GinkgoWriter.Printf("workload cluster %s still exists after cleanup timeout; manual cleanup is required\n", l.clusterName)
		}
		if stillExists {
			Fail(fmt.Sprintf("workload cluster %s still exists after cleanup timeout", l.clusterName))
		}
	}
}

func (l *workloadClusterLifecycle) cleanupManagementKubeconfig() {
	if !l.managementKubeconfigOwned || l.managementKubeconfigPath == "" || l.guideExecutor == nil {
		return
	}
	if result, err := l.guideExecutor.Exec(fmt.Sprintf("rm -f %s", shellQuote(l.managementKubeconfigPath))); err != nil || result == nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("cleanup management kubeconfig %s failed: %v\n", l.managementKubeconfigPath, err)
	}
	l.managementKubeconfigOwned = false
}

func requireEndpoint(server, expectedHost, expectedPort string) error {
	parsed, err := url.Parse(server)
	if err != nil {
		return fmt.Errorf("parse API server %q: %w", server, err)
	}
	host := parsed.Hostname()
	port := parsed.Port()
	if host == "" {
		return fmt.Errorf("API server %q has no host", server)
	}
	if port == "" {
		return fmt.Errorf("API server %q has no explicit port", server)
	}
	if host != expectedHost || port != expectedPort {
		return fmt.Errorf("API server is %s:%s, want VIP %s:%s", host, port, expectedHost, expectedPort)
	}
	return nil
}
