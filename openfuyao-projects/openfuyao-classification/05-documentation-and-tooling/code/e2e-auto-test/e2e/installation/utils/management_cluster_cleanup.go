package utils

import (
	"fmt"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"

	. "github.com/onsi/ginkgo/v2"
)

const (
	// DefaultMgmtClusterSelfBCWait 触发自管 BC 删除后，等待对象消失的短窗口。
	DefaultMgmtClusterSelfBCWait = 2 * time.Minute
	// DefaultMgmtClusterFinalWait 引导面强制删除 BC/命名空间后的最终确认窗口。
	DefaultMgmtClusterFinalWait = 2 * time.Minute
	mgmtClusterCleanupPoll      = 15 * time.Second
)

// ManagementClusterCleanupOptions 管理集群统一卸载选项。
type ManagementClusterCleanupOptions struct {
	ClusterName         string
	BootstrapKubeconfig string // 空字符串表示引导节点默认 kubeconfig
	MgmtKubeconfigPath  string // 已知路径；为空时尝试从引导 Secret/文件拉取
	MgmtNodes           []config.NodeInfo
	LocalExecutor       *executor.LocalExecutor // 节点 bke reset
	ConfigPaths         []string
	SavedKubeconfigPath string // 清理结束后删除的 kubeconfig 文件路径（通常在引导节点）
	SelfBCWaitTimeout   time.Duration
	FinalWaitTimeout    time.Duration
}

// CleanupManagementCluster 统一管理集群卸载（FIT/e2e 快速 teardown）：
//  1. 删除管理集群自管 BC（name/ns = bke-{ClusterName}，继承引导侧逻辑名）；
//  2. 各管理节点 bke reset --all --mount；
//  3. 引导面强制删除 BC 与命名空间（含 BKENode 残留）。
//
// 自管 BC 消失后，引导 BC 无需再等 EnsureDeleteOrReset 优雅收尾，直接强制清理即可。
func (m *ClusterManager) CleanupManagementCluster(opts ManagementClusterCleanupOptions) {
	if opts.ClusterName == "" {
		return
	}

	bootstrapKC := opts.BootstrapKubeconfig
	selfBCWait := opts.SelfBCWaitTimeout
	if selfBCWait == 0 {
		selfBCWait = DefaultMgmtClusterSelfBCWait
	}
	finalWait := opts.FinalWaitTimeout
	if finalWait == 0 {
		finalWait = DefaultMgmtClusterFinalWait
	}

	mgmtPath := opts.MgmtKubeconfigPath
	tempMgmtPath := false
	if mgmtPath == "" {
		checker := NewClusterCheckerWithParentKubeconfig(m.executor, opts.ClusterName, bootstrapKC)
		if path, err := checker.SaveKubeconfigToFile(); err != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", err)
		} else {
			mgmtPath = path
			tempMgmtPath = true
		}
	}
	defer func() {
		if tempMgmtPath && mgmtPath != "" {
			_, _ = m.executor.Exec(fmt.Sprintf("rm -f %s", mgmtPath))
		}
	}()

	// Step 1: 删除自管 BC（管理面 API 仍可达时触发）。
	if mgmtPath != "" && m.managementClusterSelfBCExists(mgmtPath, opts.ClusterName) {
		By(fmt.Sprintf("触发管理集群自管 BC 删除（%s）", ManagementClusterSelfBCName(opts.ClusterName)))
		if err := m.DeleteManagementClusterSelfBC(mgmtPath, opts.ClusterName); err != nil {
			GinkgoWriter.Printf("触发管理集群自管 BC 删除失败: %v\n", err)
		}
		By(fmt.Sprintf("等待自管 BC 删除（最多 %s）", selfBCWait))
		m.waitForSelfBCDeleted(mgmtPath, opts.ClusterName, selfBCWait, mgmtClusterCleanupPoll)
	} else if mgmtPath == "" {
		GinkgoWriter.Printf("无管理集群 kubeconfig，跳过自管 BC 删除\n")
	} else {
		GinkgoWriter.Printf("自管 BC 已不存在，跳过\n")
	}

	// Step 2: 节点 reset，回收 kubelet/containerd 等残留。
	if len(opts.MgmtNodes) > 0 && opts.LocalExecutor != nil {
		By("对各管理节点执行 bke reset --all --mount，回收节点残留")
		if err := m.ForceResetManagementCluster(opts.MgmtNodes, opts.LocalExecutor); err != nil {
			GinkgoWriter.Printf("管理节点 bke reset 失败: %v\n", err)
		}
	}

	// Step 3: 引导面强制删除 BC 与命名空间（BN 随 ns 清理）。
	if m.ClusterExistsWithKubeconfig(opts.ClusterName, bootstrapKC) {
		By("引导面强制删除管理集群 BC")
		if err := m.CleanupBKEClusterWithKubeconfig(opts.ClusterName, bootstrapKC, false); err != nil {
			GinkgoWriter.Printf("引导面强制删除 BC 失败: %v\n", err)
		}
	}
	By("引导面强制删除管理集群命名空间（含 BKENode 残留）")
	if err := m.CleanupClusterNamespaceWithKubeconfig(opts.ClusterName, bootstrapKC); err != nil {
		GinkgoWriter.Printf("引导面强制删除命名空间失败: %v\n", err)
	}

	By(fmt.Sprintf("确认引导面 BC/命名空间已清理（最多 %s）", finalWait))
	m.waitForBootstrapClusterRemoved(opts.ClusterName, bootstrapKC, finalWait, mgmtClusterCleanupPoll)

	for _, path := range opts.ConfigPaths {
		if path == "" {
			continue
		}
		_ = m.CleanupConfig(path)
	}
	if opts.SavedKubeconfigPath != "" {
		_, _ = m.executor.Exec(fmt.Sprintf("rm -f %s", opts.SavedKubeconfigPath))
	}
}

func (m *ClusterManager) waitForSelfBCDeleted(mgmtKubeconfigPath, bootstrapClusterName string, timeout, interval time.Duration) {
	selfBCName := ManagementClusterSelfBCName(bootstrapClusterName)
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if !m.managementClusterSelfBCExists(mgmtKubeconfigPath, bootstrapClusterName) {
			GinkgoWriter.Printf("[mgmt-self] 自管 BC (%s) 已删除\n", selfBCName)
			return
		}
		phase, state, clusterStatus, _ := m.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName)
		GinkgoWriter.Printf("[mgmt-self] 自管 BC 仍存在: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		time.Sleep(interval)
	}
	if m.managementClusterSelfBCExists(mgmtKubeconfigPath, bootstrapClusterName) {
		GinkgoWriter.Printf("[mgmt-self] 等待自管 BC 删除超时，继续节点 reset 与引导面强制清理\n")
	}
}

func (m *ClusterManager) waitForBootstrapClusterRemoved(clusterName, bootstrapKubeconfig string, timeout, interval time.Duration) {
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if bootstrapKubeconfig != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", bootstrapKubeconfig)
	}
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		bcExists := m.ClusterExistsWithKubeconfig(clusterName, bootstrapKubeconfig)
		nsExists := m.bootstrapNamespaceExists(kubeconfigArg, namespace)
		if !bcExists && !nsExists {
			GinkgoWriter.Printf("[bootstrap] BC 与命名空间 %s 均已清理\n", namespace)
			return
		}
		if bcExists {
			phase, state, clusterStatus, _ := m.GetClusterFullStatusWithKubeconfig(clusterName, bootstrapKubeconfig)
			GinkgoWriter.Printf("[bootstrap] BC 仍存在: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		}
		if nsExists {
			GinkgoWriter.Printf("[bootstrap] 命名空间 %s 仍存在\n", namespace)
		}
		time.Sleep(interval)
	}
}

func (m *ClusterManager) bootstrapNamespaceExists(kubeconfigArg, namespace string) bool {
	checkCmd := fmt.Sprintf(`%skubectl get namespace %s --no-headers 2>/dev/null`, kubeconfigArg, namespace)
	result, _ := m.executor.Exec(checkCmd)
	return result != nil && result.ExitCode == 0 && result.Stdout != ""
}

func (m *ClusterManager) managementClusterSelfBCExists(mgmtKubeconfigPath, bootstrapClusterName string) bool {
	_, _, _, err := m.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName)
	return err == nil
}

// CleanupBKEClusterWithKubeconfig 删除指定 kubeconfig 上下文中的 BKECluster BC。
func (m *ClusterManager) CleanupBKEClusterWithKubeconfig(clusterName, kubeconfigPath string, wait bool) error {
	if !m.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
		return nil
	}

	bcName := "bke-" + clusterName
	namespace := "bke-" + clusterName
	kubeconfigArg := ""
	if kubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
	}

	waitFlag := ""
	if !wait {
		waitFlag = " --wait=false"
	}
	cmd := fmt.Sprintf(`%skubectl delete bc %s -n %s%s`, kubeconfigArg, bcName, namespace, waitFlag)
	result, err := m.executor.Exec(cmd)
	if kubectlCommandFailed(result, err) {
		if kubectlOutputHasNotFound(result) {
			return nil
		}
		return formatKubectlCommandFailure("删除 BC 资源失败", result, err)
	}
	return nil
}
