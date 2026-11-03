package installation

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

// Cluster1M1WConfig 1Master1Worker 集群配置
type Cluster1M1WConfig struct {
	ClusterName    string
	ClusterConfig  *config.BKEClusterConfig
	ConfigPath     string
	NodeConfigPath string
	KubeconfigPath string
}

// Create1M1WManagementCluster 创建 1Master1Worker 管理集群
// 使用节点 1-2（nodes[0:2]），第1个作为 Master，第2个作为 Worker
func Create1M1WManagementCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterNamePrefix string,
	includeClusterAPI bool,
) (*Cluster1M1WConfig, error) {
	clusterName := fmt.Sprintf("%s-%d", clusterNamePrefix, time.Now().Unix())
	GinkgoWriter.Printf("=== Management Cluster (1M1W) Configuration ===\n")
	GinkgoWriter.Printf("Cluster Name: %s\n", clusterName)

	nodes := config.LoadTestNodesFromEnv()
	for _, node := range nodes {
		GinkgoWriter.Printf("node: %s\n", node.IP)
	}
	Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Worker management cluster requires at least 2 nodes (nodes 1-2)")

	// Management cluster uses nodes 1-2: nodes[0:2]
	mgmtNodes := make([]config.NodeInfo, 2)
	copy(mgmtNodes, nodes[:2])
	// First node as Master
	mgmtNodes[0].Hostname = "mm1"
	mgmtNodes[0].Role = []string{"master/node", "etcd"}
	// Second node as Worker
	mgmtNodes[1].Hostname = "mw1"
	mgmtNodes[1].Role = []string{"node"}

	// Print node configuration
	GinkgoWriter.Printf("Nodes Configuration:\n")
	for i, node := range mgmtNodes {
		GinkgoWriter.Printf("  Node %d: IP=%s, Hostname=%s, Roles=%v\n", i+1, node.IP, node.Hostname, node.Role)
	}

	// Create cluster config
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
	if includeClusterAPI {
		// Management cluster includes cluster-api but excludes openfuyao-system-controller
		clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
	}

	partialCluster := &Cluster1M1WConfig{
		ClusterName:   clusterName,
		ClusterConfig: clusterConfig,
	}

	// Generate cluster config file
	By("Generating cluster config file")
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to generate config file: %w", err)
	}
	partialCluster.ConfigPath = configPath
	partialCluster.NodeConfigPath = nodeConfigPath
	GinkgoWriter.Printf("Config file path: %s\n", configPath)
	GinkgoWriter.Printf("Node config file path: %s\n", nodeConfigPath)

	// Create cluster
	By("Executing bke cluster create command")
	err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
	if err != nil {
		return partialCluster, fmt.Errorf("failed to create cluster: %w", err)
	}

	// WaitAndVerifyManagementCluster 已在引导节点经 SSH 落盘 kubeconfig，勿再用 localExecutor 重复获取。
	By("Waiting for cluster status to become Healthy and Ready")
	kubeconfigPath, err := WaitAndVerifyManagementCluster(clusterManager, clusterName)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to wait and verify cluster: %w", err)
	}
	partialCluster.KubeconfigPath = kubeconfigPath
	GinkgoWriter.Printf("Cluster kubeconfig saved to: %s\n", kubeconfigPath)

	return partialCluster, nil
}

// Create1M1WWorkloadCluster 创建 1Master1Worker 业务集群
// 使用节点 3-4（nodes[2:4]），第1个作为 Master，第2个作为 Worker
// 在指定的管理集群上创建
func Create1M1WWorkloadCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	mgmtKubeconfigPath string,
	clusterNamePrefix string,
) (*Cluster1M1WConfig, error) {
	clusterName := fmt.Sprintf("%s-%d", clusterNamePrefix, time.Now().Unix())
	GinkgoWriter.Printf("=== Workload Cluster (1M1W) Configuration ===\n")
	GinkgoWriter.Printf("Cluster Name: %s\n", clusterName)

	nodes := config.LoadTestNodesFromEnv()
	Expect(len(nodes)).To(BeNumerically(">=", 4), "1Master1Worker workload cluster requires at least 4 nodes (uses nodes 3-4)")

	// Workload cluster uses nodes 3-4: nodes[2:4]
	workloadNodes := make([]config.NodeInfo, 2)
	copy(workloadNodes, nodes[2:4])
	// First node as Master
	workloadNodes[0].Hostname = "wm1"
	workloadNodes[0].Role = []string{"master/node", "etcd"}
	// Second node as Worker
	workloadNodes[1].Hostname = "ww1"
	workloadNodes[1].Role = []string{"node"}

	// Print node configuration
	GinkgoWriter.Printf("Nodes Configuration:\n")
	for i, node := range workloadNodes {
		GinkgoWriter.Printf("  Node %d: IP=%s, Hostname=%s, Roles=%v\n", i+3, node.IP, node.Hostname, node.Role)
	}

	// Create workload cluster config (default excludes cluster-api and openfuyao-system-controller)
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, workloadNodes)

	// Generate workload cluster config file
	By("Generating workload cluster config file")
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate workload cluster config file: %w", err)
	}

	partialCluster := &Cluster1M1WConfig{
		ClusterName:    clusterName,
		ClusterConfig:  clusterConfig,
		ConfigPath:     configPath,
		NodeConfigPath: nodeConfigPath,
	}

	GinkgoWriter.Printf("Workload cluster config file path: %s\n", configPath)
	GinkgoWriter.Printf("Workload cluster node config file path: %s\n", nodeConfigPath)

	// Create workload cluster on management cluster
	By("Executing bke cluster create command on management cluster")
	err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, mgmtKubeconfigPath)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to create workload cluster: %w", err)
	}

	// Wait for workload cluster to become Healthy and verify
	err = WaitAndVerify1M1WCluster(clusterManager, localExecutor, clusterName, mgmtKubeconfigPath)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to wait and verify workload cluster: %w", err)
	}

	return partialCluster, nil
}

// Delete1M1WManagementCluster 删除 1Master1Worker 管理集群。
func Delete1M1WManagementCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) {
	nodes := config.LoadTestNodesFromEnv()
	var mgmtNodes []config.NodeInfo
	if len(nodes) >= 2 {
		mgmtNodes = make([]config.NodeInfo, 2)
		copy(mgmtNodes, nodes[:2])
	}
	CleanupManagementClusterUnified(clusterManager, localExecutor, clusterName, kubeconfigPath, mgmtNodes)
}

// Verify1M1WCluster verifies a 1Master1Worker cluster status without waiting
func Verify1M1WCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) error {
	if kubeconfigPath == "" {
		checker := utils.NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, "")
		mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
		if err != nil {
			return fmt.Errorf("failed to get management cluster kubeconfig: %w", err)
		}
		return verifyManagementClusterHealth(clusterManager, clusterName, mgmtKubeconfigPath)
	}

	By("Verifying node count")
	// kubeconfig 落在引导节点，须经 SSH 执行 kubectl（执行机常无 kubectl / 无该文件）
	checker := utils.NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, kubeconfigPath)
	count, err := checker.GetNodeCountWithKubeconfig(kubeconfigPath)
	if err != nil {
		return fmt.Errorf("failed to get node count: %w", err)
	}
	if count == 0 {
		nodes, _ := checker.GetNodesWithKubeconfig(kubeconfigPath)
		return fmt.Errorf("expected 2 nodes (1 Master + 1 Worker), got 0 (nodes list: %v). This may indicate kubeconfig issue or nodes not yet joined", nodes)
	}
	if count != 2 {
		return fmt.Errorf("expected 2 nodes (1 Master + 1 Worker), got %d", count)
	}

	By("Verifying all nodes are Ready")
	var readyCount int
	readyCount, err = checker.GetReadyNodeCountWithKubeconfig(kubeconfigPath)
	if err != nil {
		return fmt.Errorf("failed to get ready node count: %w", err)
	}
	if readyCount != 2 {
		return fmt.Errorf("expected 2 ready nodes, got %d", readyCount)
	}

	return nil
}

// WaitAndVerify1M1WCluster waits for a 1Master1Worker cluster to become Healthy and Ready, then verifies its status
func WaitAndVerify1M1WCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) error {
	By("Waiting for cluster status to become Healthy and Ready")
	if kubeconfigPath != "" {
		e := eventuallyShouldBeTrue(func() bool {
			phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
			if err != nil {
				GinkgoWriter.Printf("Failed to get cluster full status: %v\n", err)
				GinkgoWriter.Printf("Current cluster status: state=%s (phase and clusterStatus unavailable)\n", state)
				return false
			} else {
				GinkgoWriter.Printf("Current cluster status: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			}
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus == "Ready"
		}, installTimeout, pollInterval, "Cluster should become Ready")
		if e != nil {
			return e
		}
	} else {
		_, err := WaitAndVerifyManagementCluster(clusterManager, clusterName)
		return err
	}

	// Verify cluster status (reuse Verify1M1WCluster)
	return Verify1M1WCluster(clusterManager, localExecutor, clusterName, kubeconfigPath)
}

// Delete1M1WWorkloadCluster 删除 1Master1Worker 业务集群
func Delete1M1WWorkloadCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	mgmtKubeconfigPath string,
) {
	By("Cleaning up workload cluster (1M1W)")
	if mgmtKubeconfigPath != "" && clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath) {
		By("Triggering workload cluster deletion on management cluster")
		err := clusterManager.DeleteClusterWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("Failed to trigger workload cluster deletion: %v\n", err)
		}

		By("Waiting for workload cluster to be completely deleted")
		Eventually(func() bool {
			exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath)
			if exists {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
				GinkgoWriter.Printf("[DEBUG] Workload cluster still exists: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				bcCheckCmd := fmt.Sprintf("kubectl --kubeconfig %s get bc bke-%s -n bke-%s --no-headers 2>&1", mgmtKubeconfigPath, clusterName, clusterName)
				result, _ := clusterManager.SSHExecutor().Exec(bcCheckCmd)
				GinkgoWriter.Printf("[DEBUG] Workload BC resource check: exit_code=%d, stdout=%s, stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
			} else {
				GinkgoWriter.Printf("[DEBUG] Workload cluster BC resource deleted\n")

				nodes := config.LoadTestNodesFromEnv()
				if len(nodes) >= 4 {
					result, _ := utils.ExecuteCommandOnNode(nodes[2], "systemctl is-active kubelet 2>&1 || echo inactive")
					GinkgoWriter.Printf("[DEBUG] Workload cluster node %s kubelet status: %s\n", nodes[2].IP, strings.TrimSpace(result.Stdout))
				}
			}
			return !exists
		}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "Workload cluster should be completely deleted")
	} else {
		GinkgoWriter.Printf("[DEBUG] Workload cluster does not exist or mgmtKubeconfigPath is empty, skipping deletion\n")
	}
}
