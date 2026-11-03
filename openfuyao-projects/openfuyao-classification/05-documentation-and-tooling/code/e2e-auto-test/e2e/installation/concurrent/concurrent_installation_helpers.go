package concurrent

import (
	"context"
	"fmt"
	"path/filepath"
	"strings"
	"sync"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

const (
	installationItTimeout = 45 * time.Minute
	concurrentItTimeout   = 90 * time.Minute
	uninstallTimeout      = 15 * time.Minute
	// scaleUninstallTimeout 扩缩容后节点回收较慢，清理等待长于普通删除场景。
	scaleUninstallTimeout = 30 * time.Minute
	pollInterval          = 30 * time.Second
	failedStatesMsgTmpl   = "集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy"

	// mgmtKubeconfigPath 管理集群与引导节点同机共部署时，引导节点默认 kubeconfig 即管理集群 API
	//（与 multi_image_repo_test.go「管理集群作为安装起点」一致，传空字符串）
	mgmtKubeconfigPath = ""
)

// WorkloadClusterJob 业务集群创建任务（含配置路径，供清理使用）。
type WorkloadClusterJob struct {
	ClusterName    string
	ConfigPath     string
	NodeConfigPath string
}

var (
	sshExecutor    *executor.SSHExecutor
	clusterManager *utils.ClusterManager
	guideNodeIP    string
)

// setupConcurrentTestFixtures 每条用例前初始化 SSH 与 ClusterManager（对齐 with-management-cluster 老用例）。
func setupConcurrentTestFixtures() {
	guideConfig := config.LoadGuideNodeFromEnv()
	Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
	Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")
	guideNodeIP = guideConfig.Host

	var err error
	sshExecutor, err = executor.NewSSHExecutor(
		guideConfig.Host,
		guideConfig.Port,
		guideConfig.Username,
		guideConfig.Password,
	)
	Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

	dynamicClient, err := dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

	clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
}

// PartitionWorkloadNodes 从 TEST_NODE2 起切分业务节点，排除引导节点 IP（附录 B A1）。
func PartitionWorkloadNodes(
	total []config.NodeInfo,
	guideNodeIP string,
	mastersPerCluster, workersPerCluster, clusterCount int,
) ([][]config.NodeInfo, error) {
	nodesPerCluster := mastersPerCluster + workersPerCluster
	needed := nodesPerCluster * clusterCount

	filtered := make([]config.NodeInfo, 0, len(total))
	for _, n := range total {
		if n.IP == guideNodeIP {
			continue
		}
		filtered = append(filtered, n)
	}
	if len(filtered) < needed {
		return nil, fmt.Errorf("业务节点不足: 需要 %d 台，可用 %d 台（已排除引导节点 %s）",
			needed, len(filtered), guideNodeIP)
	}

	groups := make([][]config.NodeInfo, clusterCount)
	idx := 0
	for c := 0; c < clusterCount; c++ {
		group := make([]config.NodeInfo, 0, nodesPerCluster)
		for m := 0; m < mastersPerCluster; m++ {
			node := filtered[idx]
			node.Role = []string{"master/node", "etcd"}
			group = append(group, node)
			idx++
		}
		for w := 0; w < workersPerCluster; w++ {
			node := filtered[idx]
			node.Role = []string{"node"}
			group = append(group, node)
			idx++
		}
		groups[c] = group
	}
	return groups, nil
}

func uniqueClusterName(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}

func uploadWorkloadConfig(
	clusterName string,
	nodes []config.NodeInfo,
) (configPath, nodeConfigPath string, err error) {
	cfg := config.NewDefaultBKEClusterConfig(clusterName, nodes)
	return clusterManager.GetConfigGenerator().GenerateAndUpload(cfg)
}

func runConcurrent(wg *sync.WaitGroup, fn func()) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer GinkgoRecover()
		fn()
	}()
}

func runConcurrentErr(wg *sync.WaitGroup, errs chan<- error, fn func() error) {
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer GinkgoRecover()
		errs <- fn()
	}()
}

func runConcurrentOperations(ops ...func() error) {
	if len(ops) == 0 {
		return
	}
	var wg sync.WaitGroup
	errs := make(chan error, len(ops))
	for _, op := range ops {
		task := op
		runConcurrentErr(&wg, errs, task)
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		Expect(err).NotTo(HaveOccurred())
	}
}

func waitClusterHealthyReady(clusterName string, timeout time.Duration) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] phase=%s, state=%s, clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, timeout, pollInterval).Should(BeTrue(), fmt.Sprintf("集群 %s 应达到 Ready", clusterName))
}

// isClusterScalingActive 扩缩容/删除等中间态，尚未收敛到可断言或删除的空闲态。
func isClusterScalingActive(phase, clusterStatus string) bool {
	switch clusterStatus {
	case "ScalingWorkerNodesUp", "ScalingWorkerNodesDown", "DeployingAddon", "Checking", "Initializing", "Deleting":
		return true
	}
	switch phase {
	case "EnsureBKEAgent", "EnsureNodesEnv", "EnsureCerts", "EnsureMasterInit",
		"EnsureWorkerJoin", "EnsureWorkerDelete", "EnsureAddonDeploy", "EnsureDeleteOrReset":
		return true
	default:
		return false
	}
}

func clusterIPsMatch(ips, wantIPs, excludeIPs []string) bool {
	if len(ips) != len(wantIPs) {
		return false
	}
	seen := make(map[string]struct{}, len(ips))
	for _, ip := range ips {
		seen[ip] = struct{}{}
	}
	for _, ip := range wantIPs {
		if _, ok := seen[ip]; !ok {
			return false
		}
	}
	for _, ip := range excludeIPs {
		if _, ok := seen[ip]; ok {
			return false
		}
	}
	return true
}

// waitClusterQuiescent 删除前等待集群离开扩缩容中间态并达到 Healthy/Ready。
func waitClusterQuiescent(clusterName string, timeout time.Duration) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] quiescent: phase=%s, state=%s, clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		if isClusterScalingActive(phase, clusterStatus) {
			return false
		}
		return clusterStatus == "Ready"
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 应收敛到可删除的空闲 Ready", clusterName))
}

// eventuallyClusterScaleSettled 扩缩容后联合校验：非中间态 + Healthy/Ready + BKENode IP 集合符合预期。
func eventuallyClusterScaleSettled(clusterName string, wantIPs, excludeIPs []string, timeout time.Duration) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取状态失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] phase=%s, state=%s, clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		if isClusterScalingActive(phase, clusterStatus) {
			return false
		}
		if clusterStatus != "Ready" {
			return false
		}
		ips, err := listBKENodeIPs(clusterName)
		if err != nil {
			GinkgoWriter.Printf("[%s] 获取 BKENode IP 失败: %v\n", clusterName, err)
			return false
		}
		GinkgoWriter.Printf("[%s] BKENode IPs=%v\n", clusterName, ips)
		return clusterIPsMatch(ips, wantIPs, excludeIPs)
	}, timeout, pollInterval).Should(BeTrue(),
		fmt.Sprintf("集群 %s 扩缩容应收敛: want=%v exclude=%v", clusterName, wantIPs, excludeIPs))
}

func waitClustersHealthyReady(names []string, timeout time.Duration) {
	stop := startConcurrentStatusLog(names, pollInterval)
	defer stop()

	// 断言保持主协程串行执行，避免 goroutine 里跑 Eventually/Expect 导致超时泄漏。
	for _, name := range names {
		waitClusterHealthyReady(name, timeout)
	}
}

func startConcurrentStatusLog(names []string, interval time.Duration) func() {
	ctx, cancel := context.WithCancel(context.Background())
	var wg sync.WaitGroup

	logOne := func(clusterName string) {
		defer wg.Done()

		ticker := time.NewTicker(interval)
		defer ticker.Stop()

		// 先打一条即时状态，避免首轮日志等待一个 poll 周期。
		logStatus(clusterName)

		for {
			select {
			case <-ctx.Done():
				return
			case <-ticker.C:
				logStatus(clusterName)
			}
		}
	}

	for _, name := range names {
		n := name
		wg.Add(1)
		go logOne(n)
	}

	return func() {
		cancel()
		wg.Wait()
	}
}

func logStatus(clusterName string) {
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
	if err != nil {
		GinkgoWriter.Printf("[%s] 并发日志采集失败: %v\n", clusterName, err)
		return
	}
	GinkgoWriter.Printf("[%s] phase=%s, state=%s, clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
}

func waitClusterDeleted(clusterName string) {
	Eventually(func() bool {
		return !clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath)
	}, uninstallTimeout, 30*time.Second).Should(BeTrue(),
		fmt.Sprintf("集群 %s 应被完全删除", clusterName))
}

func waitClustersDeleted(names []string, timeout time.Duration) {
	Eventually(func() bool {
		allDeleted := true
		for _, name := range names {
			if clusterManager.ClusterExistsWithKubeconfig(name, mgmtKubeconfigPath) {
				allDeleted = false
				GinkgoWriter.Printf("[%s] 仍在删除中，等待资源回收完成\n", name)
			}
		}
		return allDeleted
	}, timeout, 30*time.Second).Should(BeTrue(),
		fmt.Sprintf("集群应被完全删除: %s", strings.Join(names, ", ")))
}

func deleteClustersConcurrent(names []string) {
	ops := make([]func() error, 0, len(names))
	for _, name := range names {
		n := name
		ops = append(ops, func() error {
			return clusterManager.DeleteClusterWithKubeconfig(n, mgmtKubeconfigPath)
		})
	}
	runConcurrentOperations(ops...)
}

func workloadClustersStillExisting(names []string) []string {
	remaining := make([]string, 0, len(names))
	for _, name := range names {
		if clusterManager.ClusterExistsWithKubeconfig(name, mgmtKubeconfigPath) {
			remaining = append(remaining, name)
		}
	}
	return remaining
}

func cleanupWorkloadClusters(jobs []WorkloadClusterJob) {
	cleanupWorkloadClustersWithTimeout(jobs, uninstallTimeout)
}

// cleanupScaleWorkloadClusters 供 CONC-08~11 等扩缩容场景使用更长删除等待。
func cleanupScaleWorkloadClusters(jobs []WorkloadClusterJob) {
	for _, j := range jobs {
		waitClusterQuiescent(j.ClusterName, installationItTimeout)
	}
	cleanupWorkloadClustersWithTimeout(jobs, scaleUninstallTimeout)
}

func cleanupWorkloadClustersWithTimeout(jobs []WorkloadClusterJob, deleteTimeout time.Duration) {
	if len(jobs) == 0 {
		return
	}
	names := make([]string, len(jobs))
	for i, j := range jobs {
		names[i] = j.ClusterName
	}
	deleteClustersConcurrent(names)
	remaining := workloadClustersStillExisting(names)
	if len(remaining) > 0 {
		waitClustersDeleted(remaining, deleteTimeout)
	}
	for _, j := range jobs {
		if j.ConfigPath != "" {
			_ = clusterManager.CleanupConfig(j.ConfigPath)
		}
		if j.NodeConfigPath != "" {
			_ = clusterManager.CleanupConfig(j.NodeConfigPath)
		}
	}
}

// registerWorkloadClusterCleanup registers teardown via Ginkgo DeferCleanup so it runs
// after JustAfterEach failure artifact collection. Go defer in It runs earlier and would
// delete BC/BN before BKE failure artifact capture.
func registerWorkloadClusterCleanup(jobs *[]WorkloadClusterJob) {
	DeferCleanup(func() {
		cleanupWorkloadClusters(*jobs)
	})
}

func registerWorkloadClusterCleanupJobs(jobs []WorkloadClusterJob) {
	DeferCleanup(func() {
		cleanupWorkloadClusters(jobs)
	})
}

func registerScaleWorkloadClusterCleanup(jobs *[]WorkloadClusterJob) {
	DeferCleanup(func() {
		cleanupScaleWorkloadClusters(*jobs)
	})
}

func registerScaleWorkloadClusterCleanupJobs(jobs []WorkloadClusterJob) {
	DeferCleanup(func() {
		cleanupScaleWorkloadClusters(jobs)
	})
}

func listBKENodeIPs(clusterName string) ([]string, error) {
	cmd := fmt.Sprintf("kubectl get bkenode -n bke-%s -o custom-columns=IP:.spec.ip --no-headers", clusterName)
	result, err := sshExecutor.Exec(cmd)
	if err != nil {
		return nil, fmt.Errorf("获取 BKENode IP 列表失败: %w", err)
	}
	ips := []string{}
	for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
		ip := strings.TrimSpace(line)
		if ip != "" {
			ips = append(ips, ip)
		}
	}
	return ips, nil
}

func getBKENodeIPs(clusterName string) []string {
	ips, err := listBKENodeIPs(clusterName)
	if err != nil {
		Fail(err.Error())
	}
	return ips
}

func countBKENodes(clusterName string) int {
	return len(getBKENodeIPs(clusterName))
}

func assertNodeIPsDisjoint(ipSets ...[]string) {
	seen := map[string]string{}
	for i, set := range ipSets {
		for _, ip := range set {
			if prev, ok := seen[ip]; ok {
				Fail(fmt.Sprintf("节点 IP %s 在组 %s 与组 %d 间重复", ip, prev, i))
			}
			seen[ip] = fmt.Sprintf("group-%d", i)
		}
	}
}

func assertIPsNotGuideNode(ipSets ...[]string) {
	for _, set := range ipSets {
		for _, ip := range set {
			Expect(ip).NotTo(Equal(guideNodeIP),
				"业务节点 IP 不得占用引导节点 %s", guideNodeIP)
		}
	}
}

func assertClusterBCExists(clusterName string) {
	Expect(clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath)).To(BeTrue(),
		"管理集群上应存在业务 BC %s", clusterName)
}

func createAndWait1MasterCluster(clusterName string, masterNode config.NodeInfo) WorkloadClusterJob {
	masterNode.Role = []string{"master/node", "etcd"}
	configPath, nodeConfigPath, err := uploadWorkloadConfig(clusterName, []config.NodeInfo{masterNode})
	Expect(err).NotTo(HaveOccurred(), "生成 1Master 配置应成功")

	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, mgmtKubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "创建业务集群应成功")

	waitClusterHealthyReady(clusterName, installationItTimeout)
	return WorkloadClusterJob{ClusterName: clusterName, ConfigPath: configPath, NodeConfigPath: nodeConfigPath}
}

func createAndWait1M1WCluster(clusterName string, nodes []config.NodeInfo) WorkloadClusterJob {
	Expect(len(nodes)).To(Equal(2), "1M1W 需要 2 个节点")
	master := nodes[0]
	master.Role = []string{"master/node", "etcd"}
	worker := nodes[1]
	worker.Role = []string{"node"}

	configPath, nodeConfigPath, err := uploadWorkloadConfig(clusterName, []config.NodeInfo{master, worker})
	Expect(err).NotTo(HaveOccurred(), "生成 1M1W 配置应成功")

	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, mgmtKubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "创建 1M1W 业务集群应成功")

	waitClusterHealthyReady(clusterName, installationItTimeout)
	return WorkloadClusterJob{ClusterName: clusterName, ConfigPath: configPath, NodeConfigPath: nodeConfigPath}
}

func isClusterFailedState(state, clusterStatus string) bool {
	return state == "DeployFailed" || clusterStatus == "InitializationFailed"
}

func failOnClusterFailure(state, clusterStatus string) {
	if isClusterFailedState(state, clusterStatus) {
		Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
	}
}

func loadWorkloadNodes() []config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置 TEST_NODE2 业务节点")
	return nodes
}

func requireMinWorkloadNodes(min int) []config.NodeInfo {
	nodes := loadWorkloadNodes()
	Expect(len(nodes)).To(BeNumerically(">=", min),
		"当前环境业务节点不足，需要至少 %d 台（从 TEST_NODE2 起）", min)
	return nodes
}
