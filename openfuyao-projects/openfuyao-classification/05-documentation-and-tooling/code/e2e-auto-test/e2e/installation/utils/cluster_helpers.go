package utils

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

const (
	installTimeout      = 20 * time.Minute
	mgmtInstallTimeout  = 25 * time.Minute
	uninstallTimeout    = 15 * time.Minute
	pollInterval        = 30 * time.Second
	failedStatesMsgTmpl = "集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy"
)

// Cluster3M1WConfig 3Master1Worker 集群配置
type Cluster3M1WConfig struct {
	ClusterName    string
	ClusterConfig  *config.BKEClusterConfig
	ConfigPath     string
	NodeConfigPath string
	KubeconfigPath string
}

// Create3M1WManagementCluster 创建 3Master1Worker 管理集群
// 使用节点 1-4（nodes[0:4]），前3个作为 Master，第4个作为 Worker
func Create3M1WManagementCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterNamePrefix string,
	includeClusterAPI bool,
) (*Cluster3M1WConfig, error) {
	clusterName := fmt.Sprintf("%s-%d", clusterNamePrefix, time.Now().Unix())
	GinkgoWriter.Printf("=== Management Cluster Configuration ===\n")
	GinkgoWriter.Printf("Cluster Name: %s\n", clusterName)

	nodes := config.LoadTestNodesFromEnv()
	Expect(len(nodes)).To(BeNumerically(">=", 4), "3Master1Worker management cluster requires at least 4 nodes (nodes 1-4)")

	// Management cluster uses nodes 1-4: nodes[0:4]
	mgmtNodes := make([]config.NodeInfo, 4)
	copy(mgmtNodes, nodes[:4])
	// First 3 nodes (1-3) as Master
	for i := 0; i < 3; i++ {
		mgmtNodes[i].Role = []string{"master/node", "etcd"}
	}
	// 4th node as Worker
	mgmtNodes[3].Role = []string{"node"}

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
	// else: default ExcludeAddons already excludes cluster-api and openfuyao-system-controller
	partialCluster := &Cluster3M1WConfig{
		ClusterName:   clusterName,
		ClusterConfig: clusterConfig,
	}

	// Load VIP configuration from environment variables (if provided)
	vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
	if vipHost != "" {
		clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
			Host: vipHost,
			Port: vipPort,
		}
		GinkgoWriter.Printf("VIP Configuration: host=%s, port=%s\n", vipHost, vipPort)
	} else {
		GinkgoWriter.Printf("VIP Configuration: Not configured (will be set to empty object {} in YAML)\n")
	}

	// Print addons configuration
	GinkgoWriter.Printf("Addons Configuration:\n")
	for _, addon := range clusterConfig.Addons {
		GinkgoWriter.Printf("  - %s (version: %s)\n", addon.Name, addon.Version)
	}
	GinkgoWriter.Printf("========================================\n")

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

// Create3M1WWorkloadCluster 创建 3Master1Worker 业务集群
// 使用节点 5-8（nodes[4:8]），前3个作为 Master，第4个作为 Worker
// 在指定的管理集群上创建
func Create3M1WWorkloadCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	mgmtKubeconfigPath string,
	clusterNamePrefix string,
) (*Cluster3M1WConfig, error) {
	clusterName := fmt.Sprintf("%s-%d", clusterNamePrefix, time.Now().Unix())
	GinkgoWriter.Printf("=== Workload Cluster Configuration ===\n")
	GinkgoWriter.Printf("Cluster Name: %s\n", clusterName)

	nodes := config.LoadTestNodesFromEnv()
	Expect(len(nodes)).To(BeNumerically(">=", 8), "3Master1Worker workload cluster requires at least 8 nodes (uses nodes 5-8)")

	// Workload cluster uses nodes 5-8: nodes[4:8]
	workloadNodes := make([]config.NodeInfo, 4)
	copy(workloadNodes, nodes[4:8])
	// First 3 nodes (5-7) as Master
	for i := 0; i < 3; i++ {
		workloadNodes[i].Role = []string{"master/node", "etcd"}
	}
	// 4th node (8) as Worker
	workloadNodes[3].Role = []string{"node"}

	// Print node configuration
	GinkgoWriter.Printf("Nodes Configuration:\n")
	for i, node := range workloadNodes {
		GinkgoWriter.Printf("  Node %d: IP=%s, Hostname=%s, Roles=%v\n", i+5, node.IP, node.Hostname, node.Role)
	}

	// Create workload cluster config (default excludes cluster-api and openfuyao-system-controller)
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, workloadNodes)

	// Load VIP configuration from environment variables (if provided)
	vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv()
	if vipHost != "" {
		clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
			Host: vipHost,
			Port: vipPort,
		}
		GinkgoWriter.Printf("VIP Configuration: host=%s, port=%s\n", vipHost, vipPort)
	} else {
		GinkgoWriter.Printf("VIP Configuration: Not configured (will be set to empty object {} in YAML)\n")
	}

	// Print addons configuration
	GinkgoWriter.Printf("Addons Configuration:\n")
	for _, addon := range clusterConfig.Addons {
		GinkgoWriter.Printf("  - %s (version: %s)\n", addon.Name, addon.Version)
	}
	GinkgoWriter.Printf("========================================\n")

	// Generate workload cluster config file
	By("Generating workload cluster config file")
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate workload cluster config file: %w", err)
	}

	partialCluster := &Cluster3M1WConfig{
		ClusterName:    clusterName,
		ClusterConfig:  clusterConfig,
		ConfigPath:     configPath,
		NodeConfigPath: nodeConfigPath,
	}

	GinkgoWriter.Printf("Workload cluster config file path: %s\n", configPath)
	GinkgoWriter.Printf("Workload cluster node config file path: %s\n", nodeConfigPath)

	// Create workload cluster on management cluster
	By("Creating namespace and applying workload cluster config on management cluster")
	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, mgmtKubeconfigPath)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to create workload cluster: %w", err)
	}

	// Wait for workload cluster to become Healthy and verify
	err = WaitAndVerify3M1WCluster(clusterManager, localExecutor, clusterName, mgmtKubeconfigPath)
	if err != nil {
		return partialCluster, fmt.Errorf("failed to wait and verify workload cluster: %w", err)
	}

	return partialCluster, nil
}

// CleanupManagementClusterUnified 统一管理集群卸载（自管 BC 删除 → 节点 bke reset → 引导面强制删 BC/ns）。
func CleanupManagementClusterUnified(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName, mgmtKubeconfigPath string,
	mgmtNodes []config.NodeInfo,
	configPaths ...string,
) {
	By("Cleaning up management cluster")
	clusterManager.CleanupManagementCluster(ManagementClusterCleanupOptions{
		ClusterName:         clusterName,
		MgmtKubeconfigPath:  mgmtKubeconfigPath,
		MgmtNodes:           mgmtNodes,
		LocalExecutor:       localExecutor,
		ConfigPaths:         configPaths,
		SavedKubeconfigPath: mgmtKubeconfigPath,
	})
}

// Delete3M1WManagementCluster 删除 3Master1Worker 管理集群。
func Delete3M1WManagementCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) {
	mgmtNodes := loadMgmtNodesFromEnv(4)
	CleanupManagementClusterUnified(clusterManager, localExecutor, clusterName, kubeconfigPath, mgmtNodes)
}

func loadMgmtNodesFromEnv(count int) []config.NodeInfo {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < count {
		return nil
	}
	mgmtNodes := make([]config.NodeInfo, count)
	copy(mgmtNodes, nodes[:count])
	return mgmtNodes
}

// WaitAndVerifyManagementCluster 等待管理集群健康（引导节点 BC + 管理集群自管 BC，仅校验 clusterStatus Ready）。
// kubeconfig 必须经 ClusterManager 的 SSH 执行器在引导节点上读取并落盘，路径才能被后续自管 BC 查询复用。
func WaitAndVerifyManagementCluster(
	clusterManager *ClusterManager,
	clusterName string,
) (mgmtKubeconfigPath string, err error) {
	By("等待引导节点上管理集群 BC 变为 Ready")
	err = EventuallyShouldBeTrue(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
		if e != nil {
			GinkgoWriter.Printf("[bootstrap] 获取集群状态失败: %v\n", e)
			return false
		}
		GinkgoWriter.Printf("[bootstrap] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval, "引导节点上管理集群 BC 应该变为 Ready")
	if err != nil {
		return
	}

	By("获取管理集群 kubeconfig（引导节点）")
	checker := NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, "")
	err = EventuallyShouldBeTrue(func() bool {
		path, e := checker.SaveKubeconfigToFile()
		if e != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", e)
			return false
		}
		mgmtKubeconfigPath = path
		return true
	}, 5*time.Minute, 10*time.Second, "应该成功获取管理集群 kubeconfig")
	if err != nil {
		return
	}

	By("等待管理集群自管 BC 变为 Ready")
	err = EventuallyShouldBeTrue(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
		if e != nil {
			GinkgoWriter.Printf("[mgmt-self] 获取自管 BC 状态失败: %v\n", e)
			return false
		}
		GinkgoWriter.Printf("[mgmt-self] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval, "管理集群自管 BC 应该变为 Ready")
	if err != nil {
		return
	}

	return mgmtKubeconfigPath, nil
}

func VerifyManagementClusterHealth(
	clusterManager *ClusterManager,
	clusterName string,
	mgmtKubeconfigPath string,
) error {
	By("验证引导节点上管理集群 BC 状态")
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
	if err != nil {
		return fmt.Errorf("failed to get bootstrap cluster full status: %w", err)
	}
	if clusterStatus != "Ready" {
		return fmt.Errorf("bootstrap cluster clusterStatus is not Ready: %s", clusterStatus)
	}
	GinkgoWriter.Printf("[bootstrap] 最终状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)

	By("验证管理集群自管 BC 状态")
	phase, state, clusterStatus, err = clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
	if err != nil {
		return fmt.Errorf("failed to get management cluster self BC status: %w", err)
	}
	if clusterStatus != "Ready" {
		return fmt.Errorf("management cluster self BC clusterStatus is not Ready: %s", clusterStatus)
	}
	GinkgoWriter.Printf("[mgmt-self] 最终状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
	return nil
}

// Verify3M1WCluster verifies a 3Master1Worker cluster status without waiting
// clusterName: name of the cluster
// kubeconfigPath: kubeconfig path for workload cluster (empty string for management cluster on bootstrap)
func Verify3M1WCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) error {
	if kubeconfigPath == "" {
		checker := NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, "")
		mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
		if err != nil {
			return fmt.Errorf("failed to get management cluster kubeconfig: %w", err)
		}
		return VerifyManagementClusterHealth(clusterManager, clusterName, mgmtKubeconfigPath)
	}

	// Verify node count
	By("Verifying node count")
	// kubeconfig 落在引导节点，须经 SSH 执行 kubectl（执行机常无 kubectl / 无该文件）
	checker := NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, kubeconfigPath)
	count, err := checker.GetNodeCountWithKubeconfig(kubeconfigPath)
	if err != nil {
		return fmt.Errorf("failed to get node count: %w", err)
	}
	if count == 0 {
		nodes, _ := checker.GetNodesWithKubeconfig(kubeconfigPath)
		return fmt.Errorf("expected 4 nodes (3 Master + 1 Worker), got 0 (nodes list: %v). This may indicate kubeconfig issue or nodes not yet joined", nodes)
	}
	if count != 4 {
		return fmt.Errorf("expected 4 nodes (3 Master + 1 Worker), got %d", count)
	}

	// Verify all nodes are Ready
	By("Verifying all nodes are Ready")
	var readyCount int
	readyCount, err = checker.GetReadyNodeCountWithKubeconfig(kubeconfigPath)
	if err != nil {
		return fmt.Errorf("failed to get ready node count: %w", err)
	}
	if readyCount != 4 {
		return fmt.Errorf("expected 4 ready nodes, got %d", readyCount)
	}

	return nil
}

// WaitAndVerify3M1WCluster waits for a 3Master1Worker cluster to become Healthy and Ready, then verifies its status
// clusterName: name of the cluster
// kubeconfigPath: kubeconfig path for workload cluster (empty string for management cluster on bootstrap)
func WaitAndVerify3M1WCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	kubeconfigPath string,
) error {
	// Wait for cluster to become Healthy and Ready
	By("Waiting for cluster status to become Healthy and Ready")
	if kubeconfigPath != "" {
		// Workload cluster
		e := EventuallyShouldBeTrue(func() bool {
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

	// Verify cluster status (reuse Verify3M1WCluster)
	return Verify3M1WCluster(clusterManager, localExecutor, clusterName, kubeconfigPath)
}

// Delete3M1WWorkloadCluster 删除 3Master1Worker 业务集群
// 在指定的管理集群上删除
func Delete3M1WWorkloadCluster(
	clusterManager *ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName string,
	mgmtKubeconfigPath string,
) {
	By("Cleaning up workload cluster")
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
				// Check if BC resource still exists on management cluster
				bcCheckCmd := fmt.Sprintf("kubectl --kubeconfig %s get bc bke-%s -n bke-%s --no-headers 2>&1", mgmtKubeconfigPath, clusterName, clusterName)
				result, _ := clusterManager.SSHExecutor().Exec(bcCheckCmd)
				GinkgoWriter.Printf("[DEBUG] Workload BC resource check: exit_code=%d, stdout=%s, stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
			} else {
				GinkgoWriter.Printf("[DEBUG] Workload cluster BC resource deleted\n")

				// Check if k8s components are still running on workload cluster nodes (for debugging)
				nodes := config.LoadTestNodesFromEnv()
				if len(nodes) >= 8 {
					result, _ := ExecuteCommandOnNode(nodes[4], "systemctl is-active kubelet 2>&1 || echo inactive")
					GinkgoWriter.Printf("[DEBUG] Workload cluster node %s kubelet status: %s\n", nodes[4].IP, strings.TrimSpace(result.Stdout))
				}
			}
			return !exists
		}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "Workload cluster should be completely deleted")
	} else {
		GinkgoWriter.Printf("[DEBUG] Workload cluster does not exist or mgmtKubeconfigPath is empty, skipping deletion\n")
	}
}

// DisconnectNetwork 断开指定节点的网络（通过注释 /etc/resolv.conf）
// 通过 SSH 连接到目标节点执行命令
// node: 目标节点信息
// localExecutor: 本地执行器（保留参数用于兼容）
// 返回备份文件路径和错误
func DisconnectNetwork(node config.NodeInfo, localExecutor *executor.LocalExecutor) (string, error) {
	By(fmt.Sprintf("Disconnecting network on node %s (%s)", node.Hostname, node.IP))

	// 生成备份文件名
	timestamp := time.Now().Unix()
	backupFile := fmt.Sprintf("/etc/resolv.conf.backup.%d", timestamp)

	_ = localExecutor

	// Step 1: 备份 /etc/resolv.conf
	backupCmd := fmt.Sprintf("cp /etc/resolv.conf %s", backupFile)
	result, err := ExecuteCommandOnNode(node, backupCmd)
	if err != nil {
		return "", fmt.Errorf("failed to backup resolv.conf: %w", err)
	}
	if result.ExitCode != 0 {
		return "", fmt.Errorf("backup command failed (exit code %d): %s", result.ExitCode, result.Stderr)
	}

	// Step 2: 注释 /etc/resolv.conf 的所有行
	commentCmd := "sed -i \"s/^/#/\" /etc/resolv.conf"
	result, err = ExecuteCommandOnNode(node, commentCmd)
	if err != nil {
		return backupFile, fmt.Errorf("failed to comment resolv.conf: %w", err)
	}
	if result.ExitCode != 0 {
		return backupFile, fmt.Errorf("comment command failed (exit code %d): %s", result.ExitCode, result.Stderr)
	}

	// Step 3: 验证 resolv.conf 已被注释
	verifyCmd := "cat /etc/resolv.conf"
	result, err = ExecuteCommandOnNode(node, verifyCmd)
	if err == nil && result.ExitCode == 0 {
		GinkgoWriter.Printf("Resolv.conf after commenting on node %s:\n%s\n", node.IP, result.Stdout)
	}

	return backupFile, nil
}

// RestoreNetwork 恢复指定节点的网络（通过恢复 /etc/resolv.conf）
// 通过 SSH 连接到目标节点执行命令
// node: 目标节点信息
// localExecutor: 本地执行器（保留参数用于兼容）
// backupFile: 可选的备份文件路径，如果为空则自动查找最新的备份文件
func RestoreNetwork(node config.NodeInfo, localExecutor *executor.LocalExecutor, backupFile string) error {
	By(fmt.Sprintf("Restoring network on node %s (%s)", node.Hostname, node.IP))
	_ = localExecutor

	// 如果没有指定备份文件，查找最新的备份文件
	if backupFile == "" {
		findBackupCmd := "ls -t /etc/resolv.conf.backup.* 2>/dev/null | head -1"
		result, err := ExecuteCommandOnNode(node, findBackupCmd)
		if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
			backupFile = strings.TrimSpace(result.Stdout)
		}
	}

	// 如果找到了备份文件，尝试恢复
	if backupFile != "" {
		restoreCmd := fmt.Sprintf("cp %s /etc/resolv.conf", backupFile)
		result, err := ExecuteCommandOnNode(node, restoreCmd)
		if err == nil && result.ExitCode == 0 {
			GinkgoWriter.Printf("Restored resolv.conf from backup: %s\n", backupFile)
			return nil
		}
		// 如果恢复失败，继续尝试取消注释
		GinkgoWriter.Printf("Failed to restore from backup %s, trying to uncomment instead\n", backupFile)
	}

	// 如果备份恢复失败或没有找到备份文件，尝试取消注释
	uncommentCmd := "sed -i \"s/^#//\" /etc/resolv.conf"
	result, err := ExecuteCommandOnNode(node, uncommentCmd)
	if err != nil {
		return fmt.Errorf("failed to uncomment resolv.conf: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("uncomment command failed (exit code %d): %s", result.ExitCode, result.Stderr)
	}

	if backupFile == "" {
		GinkgoWriter.Printf("Uncommented resolv.conf (no backup found)\n")
	} else {
		GinkgoWriter.Printf("Uncommented resolv.conf (backup restore failed)\n")
	}

	return nil
}

// getTestPatchYAMLContent 返回 testpatch.yaml 文件内容
func GetTestPatchYAMLContent() []byte {
	return []byte(`registry:
  imageAddress: cr.openfuyao.cn/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
openFuyaoVersion: v25.09.999
kubernetesVersion: v1.33.1-of.2
etcdVersion: v3.5.21-of.1
containerdVersion: v2.1.1
repos:
  - architecture:
      - amd64
      - arm64
    isKubernetes: false
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao/coredns
        targetRepo: kubernetes/kubernetes
        images:
          - name: coredns
            usedPodInfo:
              - podPrefix: coredns
                namespace: kube-system
            tag:
              - v1.10.2
      - sourceRepo: cr.openfuyao.cn/openfuyao/ingress-nginx
        targetRepo: kubernetes/ingress-nginx
        images:
          - name: controller
            usedPodInfo:
              - podPrefix: ingress-nginx-controller
                namespace: ingress-nginx
            tag:
              - v1.9.5
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: user-management-operator
            usedPodInfo:
              - podPrefix: user-management-operator
                namespace: openfuyao-system
            tag:
              - 1.2.3
files:
  - address: https://openfuyao.obs.cn-north-4.myhuaweicloud.com/containerd/containerd/releases/download/v2.1.1-origin/
    files:
      - fileName: containerd-v2.1.1-linux-amd64.tar.gz
      - fileName: containerd-v2.1.1-linux-arm64.tar.gz
`)
}

func GetTestPatchYAMLContentLargeScale() []byte {
	return []byte(`registry:
  imageAddress: hub.oepkgs.net/openfuyao/registry:2.8.1
  architecture:
    - amd64
    - arm64
repos:
  - architecture:
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        images:
          - name: registry
            tag: ["2.8.1"]
          - name: nginx
            tag: ["1.23.0-alpine"]
          - name: helm/chartmuseum
            tag: ["v0.16.2"]
          - name: openebs/nfs-server-alpine
            tag: ["0.9.0"]
          - name: rancher/k3s
            tag: ["v1.25.16-k3s4"]
          - name: rancher/mirrored-pause
            tag: ["3.6"]

  - architecture:
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: coredns
            tag: ["1.12.2-of.1"]
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: kube-apiserver
            tag: ["1.34.3-of.1"]
          - name: kube-controller-manager
            tag: ["1.34.3-of.1"]
          - name: kube-scheduler
            tag: ["1.34.3-of.1"]
          - name: kube-proxy
            tag: ["1.34.3-of.1"]
          - name: etcd
            tag: ["3.6.7-of.1"]
      - sourceRepo: cr.openfuyao.cn/openfuyao/keepalived
        targetRepo: kubernetes/keepalived
        images:
          - name: keepalived
            tag: ["1.3.5"]
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        images:
          - name: haproxy
            tag: ["2.1.4"]
      - sourceRepo: hub.oepkgs.net/openfuyao/calico
        targetRepo: kubernetes/calico
        images:
          - name: cni
            tag: ["v3.31.3"]
          - name: node
            tag: ["v3.31.3"]
          - name: kube-controllers
            tag: ["v3.31.3"]
  - architecture:
      - amd64
      - arm64
    needDownload: true
    subImages:
      - sourceRepo: cr.openfuyao.cn/openfuyao
        targetRepo: kubernetes
        images:
          - name: bke-console-website
            tag: ["latest"]
          - name: bke-console-service
            tag: ["latest"]
          - name: installer-website
            tag: ["latest"]
          - name: installer-service
            tag: ["latest"]
          - name: application-management-service
            tag: ["latest"]
          - name: cluster-api-provider-bke
            tag: ["latest"]
          - name: bke-manifests
            tag: ["latest"]
          - name: bkeagent-deployer
            tag: ["latest"]
          - name: bkeagent-launcher
            tag: ["latest"]
          - name: console-service
            tag: ["latest"]
          - name: console-website
            tag: ["latest"]
          - name: marketplace-service
            tag: ["latest"]
          - name: monitoring-service
            tag: ["latest"]
          - name: oauth-proxy
            tag: ["latest"]
          - name: oauth-server
            tag: ["latest"]
          - name: oauth-webhook
            tag: ["latest"]
          - name: openfuyao-system-controller
            tag: ["latest"]
          - name: plugin-management-service
            tag: ["latest"]
          - name: user-management-operator
            tag: ["latest"]
          - name: web-terminal-service
            tag: ["latest"]
          - name: kubectl-openfuyao
            tag: ["latest"]
      - sourceRepo: hub.oepkgs.net/openfuyao/brancz
        targetRepo: kubernetes/brancz
        images:
          - name: kube-rbac-proxy
            tag: ["v0.18.0"]
      - sourceRepo: hub.oepkgs.net/openfuyao
        targetRepo: kubernetes
        images:
          - name: busybox
            tag: ["1.36.1"]
      - sourceRepo: hub.oepkgs.net/openfuyao/cluster-api
        targetRepo: kubernetes/cluster-api
        images:
          - name: cluster-api-controller
            tag: ["v1.4.3"]
          - name: kubeadm-bootstrap-controller
            tag: ["v1.4.3"]
          - name: kubeadm-control-plane-controller
            tag: ["v1.4.3"]
      - sourceRepo: hub.oepkgs.net/openfuyao/ingress-nginx
        targetRepo: kubernetes/ingress-nginx
        images:
          - name: controller
            tag: ["v1.9.4"]
          - name: kube-webhook-certgen
            tag: ["v20231011-8b53cabe0"]
      - sourceRepo: hub.oepkgs.net/openfuyao/jimmidyson
        targetRepo: kubernetes/jimmidyson
        images:
          - name: configmap-reload
            tag: ["v0.13.0"]
`)
}

func isClusterFailedState(state, clusterStatus string) bool {
	return state == "DeployFailed" || clusterStatus == "InitializationFailed"
}

func failOnClusterFailure(state, clusterStatus string) {
	if isClusterFailedState(state, clusterStatus) {
		Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
	}
}

// eventuallyShouldBeTrue 等价于 Eventually(condition, timeout, interval).Should(BeTrue(), msg)，
// 但在超时或失败时返回 error 而非 panic，便于调用方在失败后仍能返回已创建的资源信息以供 AfterAll 清理。
// 注意：condition 闭包内的任意 panic（含 Ginkgo Fail 与真实 bug）均会被转为 error；
// 详细失败信息与堆栈由 Ginkgo Fail 在 panic 前记录到测试输出。
func EventuallyShouldBeTrue(condition func() bool, timeout, interval time.Duration, msg string) error {
	err := fmt.Errorf("%s", msg)
	func() {
		defer func() {
			if r := recover(); r != nil {
				err = fmt.Errorf("%s: %v", msg, r)
			}
		}()
		Eventually(condition, timeout, interval).Should(BeTrue(), msg)
		err = nil
	}()
	return err
}
