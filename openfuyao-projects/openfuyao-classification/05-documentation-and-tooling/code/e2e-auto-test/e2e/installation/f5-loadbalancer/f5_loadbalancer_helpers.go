package f5_loadbalancer

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

const (
	InstallationItTimeout = 45 * time.Minute
	installTimeout        = 20 * time.Minute
	mgmtInstallTimeout    = 25 * time.Minute
	uninstallTimeout      = 15 * time.Minute
	pollInterval          = 30 * time.Second
)

func cleanupManagementCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName, mgmtKubeconfigPath string,
	mgmtNodes []config.NodeInfo,
	configPaths ...string,
) {
	By("Cleaning up management cluster")
	clusterManager.CleanupManagementCluster(utils.ManagementClusterCleanupOptions{
		ClusterName:         clusterName,
		MgmtKubeconfigPath:  mgmtKubeconfigPath,
		MgmtNodes:           mgmtNodes,
		LocalExecutor:       localExecutor,
		ConfigPaths:         configPaths,
		SavedKubeconfigPath: mgmtKubeconfigPath,
	})
}

func delete3M1WManagementCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
	clusterName, kubeconfigPath string,
) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) > 4 {
		nodes = nodes[:4]
	}
	cleanupManagementCluster(clusterManager, localExecutor, clusterName, kubeconfigPath, nodes)
}

func waitAndVerifyManagementCluster(
	clusterManager *utils.ClusterManager,
	clusterName string,
) (mgmtKubeconfigPath string, err error) {
	By("等待引导节点上管理集群 BC 变为 Ready")
	err = eventuallyShouldBeTrue(func() bool {
		phase, state, clusterStatus, statusErr := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
		if statusErr != nil {
			GinkgoWriter.Printf("[bootstrap] 获取集群状态失败: %v\n", statusErr)
			return false
		}
		GinkgoWriter.Printf("[bootstrap] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval, "引导节点上管理集群 BC 应该变为 Ready")
	if err != nil {
		return "", err
	}

	By("获取管理集群 kubeconfig（引导节点）")
	checker := utils.NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), clusterName, "")
	err = eventuallyShouldBeTrue(func() bool {
		path, saveErr := checker.SaveKubeconfigToFile()
		if saveErr != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", saveErr)
			return false
		}
		mgmtKubeconfigPath = path
		return true
	}, 5*time.Minute, 10*time.Second, "应该成功获取管理集群 kubeconfig")
	if err != nil {
		return "", err
	}

	By("等待管理集群自管 BC 变为 Ready")
	err = eventuallyShouldBeTrue(func() bool {
		phase, state, clusterStatus, statusErr := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
		if statusErr != nil {
			GinkgoWriter.Printf("[mgmt-self] 获取自管 BC 状态失败: %v\n", statusErr)
			return false
		}
		GinkgoWriter.Printf("[mgmt-self] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval, "管理集群自管 BC 应该变为 Ready")
	return mgmtKubeconfigPath, err
}

func verifyManagementClusterHealth(clusterManager *utils.ClusterManager, clusterName, mgmtKubeconfigPath string) error {
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
	if err != nil {
		return fmt.Errorf("failed to get bootstrap cluster full status: %w", err)
	}
	if clusterStatus != "Ready" {
		return fmt.Errorf("bootstrap cluster clusterStatus is not Ready: %s", clusterStatus)
	}
	GinkgoWriter.Printf("[bootstrap] 最终状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)

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

func delete3M1WWorkloadCluster(clusterManager *utils.ClusterManager, _ *executor.LocalExecutor, clusterName, mgmtKubeconfigPath string) {
	By("Cleaning up workload cluster")
	if mgmtKubeconfigPath == "" || !clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath) {
		GinkgoWriter.Printf("[DEBUG] Workload cluster does not exist or mgmtKubeconfigPath is empty, skipping deletion\n")
		return
	}
	if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, mgmtKubeconfigPath); err != nil {
		GinkgoWriter.Printf("Failed to trigger workload cluster deletion: %v\n", err)
	}
	Eventually(func() bool {
		exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, mgmtKubeconfigPath)
		if exists {
			phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, mgmtKubeconfigPath)
			GinkgoWriter.Printf("[DEBUG] Workload cluster still exists: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		}
		return !exists
	}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "Workload cluster should be completely deleted")
}

func failOnClusterFailure(state, clusterStatus string) {
	if state == "DeployFailed" || clusterStatus == "InitializationFailed" {
		Fail(fmt.Sprintf("集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy", state, clusterStatus))
	}
}

func eventuallyShouldBeTrue(condition func() bool, timeout, interval time.Duration, msg string) error {
	err := fmt.Errorf("%s", msg)
	func() {
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("%s: %v", msg, recovered)
			}
		}()
		Eventually(condition, timeout, interval).Should(BeTrue(), msg)
		err = nil
	}()
	return err
}
