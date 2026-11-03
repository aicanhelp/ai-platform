package declarative_upgrade

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

// runDeclarativeUpgradeFITGateClosedInstallAndVerify 安装 Gate 关闭 FIT 业务集群（无 helm-component 注解）。
func runDeclarativeUpgradeFITGateClosedInstallAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName, kubeconfigPath string,
) (configPath, nodeConfigPath string) {
	sourceVersion := utils.DeclarativeUpgradeFITSourceVersion()
	clusterConfig.OpenFuyaoVersion = sourceVersion
	if clusterConfig.Annotations != nil {
		delete(clusterConfig.Annotations, utils.HelmComponentAnnotation)
	}
	return runDeclarativeUpgradeInstallAndVerify(clusterManager, duHelper, clusterConfig, clusterName, kubeconfigPath)
}

// runDeclarativeUpgradeFITGateClosedMgmtInstallAndVerify 安装 Gate 关闭 FIT 管理集群（无 helm-component 注解）。
func runDeclarativeUpgradeFITGateClosedMgmtInstallAndVerify(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName string,
) (configPath, nodeConfigPath, mgmtKubeconfigPath string) {
	sourceVersion := utils.DeclarativeUpgradeFITSourceVersion()
	clusterConfig.OpenFuyaoVersion = sourceVersion
	if clusterConfig.Annotations != nil {
		delete(clusterConfig.Annotations, utils.HelmComponentAnnotation)
	}
	return runDeclarativeUpgradeMgmtInstallAndVerify(clusterManager, sshExecutor, duHelper, clusterConfig, clusterName)
}

// runDeclarativeUpgradeFITGateClosedMgmtUpgradeAndVerify Gate 关闭管理集群 Legacy 升级：
// 校验 DU-IT-005~006；CCS 不写 Installed；declarativeUpgrade.completed 含 fit-yaml-gate。
func runDeclarativeUpgradeFITGateClosedMgmtUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	bootstrapClusterName string,
	mgmtKubeconfigPath string,
	sourceVersion, targetVersion string,
) {
	runDeclarativeUpgradeMgmtUpgradeAndVerify(clusterManager, duHelper, bootstrapClusterName, "", nil, mgmtKubeconfigPath, sourceVersion, targetVersion, false)
	verifyFITYamlComponentsLegacyPath(
		duHelper, utils.ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName), mgmtKubeconfigPath, utils.FITYamlBaselineComponents,
	)
}

// runDeclarativeUpgradeFITGateClosedUpgradeAndVerify Gate 关闭 Legacy 升级：
// 校验 DU-IT-005~006；CCS 不写 Installed；declarativeUpgrade.completed 含 fit-yaml-gate。
func runDeclarativeUpgradeFITGateClosedUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	sourceVersion, targetVersion string,
) {
	runDeclarativeUpgradeWorkloadUpgradeAndVerify(
		clusterManager, duHelper, clusterName, kubeconfigPath, sourceVersion, targetVersion, false,
	)
	verifyFITYamlComponentsLegacyPath(duHelper, clusterName, kubeconfigPath, utils.FITYamlBaselineComponents)
}

// verifyFITYamlComponentsLegacyPath Gate 关闭时 yaml 走 Legacy：
// clusterComponentStatuses 不写 Installed（未走 YamlInstaller）；
// status.declarativeUpgrade.completed 含 fit-yaml-gate（旧 Manifest 链路写回成功）。
func verifyFITYamlComponentsLegacyPath(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	components []string,
) {
	if len(components) == 0 {
		components = utils.FITYamlBaselineComponents
	}
	By("FIT Gate 关闭: 校验 clusterComponentStatuses 中 yaml 组件未写 Installed（未走新路径）")
	for _, component := range components {
		verifyFITYamlComponentNotNewPathInstalled(duHelper, clusterName, kubeconfigPath, component)
	}
	verifyFITYamlComponentRecordedInLegacyCompleted(duHelper, clusterName, kubeconfigPath, utils.FITYamlGateComponent)
}

// verifyFITYamlComponentNotNewPathInstalled 组件不存在或未写 Installed 均视为 Legacy 路径符合预期。
func verifyFITYamlComponentNotNewPathInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName string,
) {
	componentName = strings.TrimSpace(componentName)
	Expect(componentName).NotTo(BeEmpty())
	status, err := duHelper.GetClusterComponentStatus(clusterName, componentName, kubeconfigPath)
	if err != nil {
		GinkgoWriter.Printf("Gate 关闭 Legacy: clusterComponentStatuses[%s] 不存在（符合预期）: %v\n", componentName, err)
		return
	}
	Expect(status.Phase).NotTo(Equal(utils.ClusterComponentPhaseInstalled),
		"Gate 关闭时 clusterComponentStatuses[%s] 不应为 Installed，实际 phase=%s message=%q",
		componentName, status.Phase, status.Message)
	GinkgoWriter.Printf("Gate 关闭 Legacy: %s phase=%s（非 Installed，符合预期）\n", componentName, status.Phase)
}

// verifyFITYamlComponentRecordedInLegacyCompleted Gate 关闭正向证据：yaml 组件写入 declarativeUpgrade.completed。
// CCS 为空只能说明没走新路径；completed 有记录才能证明 Legacy 升级成功，而不是组件被跳过。
func verifyFITYamlComponentRecordedInLegacyCompleted(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName string,
) {
	componentName = strings.TrimSpace(componentName)
	Expect(componentName).NotTo(BeEmpty())
	By(fmt.Sprintf("FIT Gate 关闭: 校验 declarativeUpgrade.completed 含 %s（Legacy 写回）", componentName))
	Eventually(func() bool {
		item, err := duHelper.FindDeclarativeUpgradeCompleted(clusterName, kubeconfigPath, componentName)
		if err != nil {
			GinkgoWriter.Printf("Gate 关闭 Legacy: 读取 declarativeUpgrade.completed[%s] 失败: %v\n", componentName, err)
			return false
		}
		if item.Version == "" || item.CompletedAt == "" {
			GinkgoWriter.Printf("Gate 关闭 Legacy: declarativeUpgrade.completed[%s] version=%q completedAt=%q 不完整\n",
				componentName, item.Version, item.CompletedAt)
			return false
		}
		GinkgoWriter.Printf("Gate 关闭 Legacy: declarativeUpgrade.completed[%s] version=%s completedAt=%s\n",
			componentName, item.Version, item.CompletedAt)
		return true
	}, 3*time.Minute, fitComponentPollInterval).Should(BeTrue(),
		"Gate 关闭时 %s 应出现在 status.declarativeUpgrade.completed（Legacy 成功写回）；"+
			"clusterComponentStatuses 无 Installed 只说明未走 YamlInstaller",
		componentName)
}

// verifyFITYamlComponentNewPathInstalled Gate 开启（注解覆盖全局）时组件应写 Installed。
func verifyFITYamlComponentNewPathInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName, targetVersion string,
) {
	verifyFITYamlComponentsInstalled(duHelper, clusterName, kubeconfigPath, targetVersion, []string{componentName})
}

// runDeclarativeUpgradeFIT7Dual1M1WAnnotationOverride DAG_EXEC_FIT_7：
// 全局 Gate 关闭时，串行触发、并行部署/卸载两个 1M1W 业务集群——A 有 helm-component 注解走新路径，
// B 无注解走 Legacy——再对比 clusterComponentStatuses / declarativeUpgrade 写回差异。
func runDeclarativeUpgradeFIT7Dual1M1WAnnotationOverride(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	kubeconfigPath string,
	sourceVersion, targetVersion string,
) {
	nodesA, nodesB, err := utils.LoadDeclarativeUpgradeDual1M1WNodes()
	Expect(err).NotTo(HaveOccurred(), "FIT_7 双集群需要至少 4 个测试节点（2×1M1W）")

	nameA := fmt.Sprintf("du-fit-gate-anno-a-1m1w-%d", time.Now().UnixNano())
	nameB := fmt.Sprintf("du-fit-gate-anno-b-1m1w-%d", time.Now().UnixNano())
	cfgA := utils.NewDeclarativeUpgradeFITClusterConfig(nameA, nodesA, sourceVersion, true)
	cfgB := utils.NewDeclarativeUpgradeFITClusterConfig(nameB, nodesB, sourceVersion, false)

	var pathA, nodePathA, pathB, nodePathB string
	defer cleanupDeclarativeUpgradeClustersSerialTriggerParallelWait(
		clusterManager, kubeconfigPath,
		declarativeUpgradeClusterCleanupJob{ClusterName: nameA, ConfigPath: pathA, NodeConfigPath: nodePathA},
		declarativeUpgradeClusterCleanupJob{ClusterName: nameB, ConfigPath: pathB, NodeConfigPath: nodePathB},
	)

	By("串行触发双集群安装（apply BC/BN），并行等待 Healthy/Ready")
	By("触发安装带 helm-component 注解的 1M1W 业务集群 A（注解覆盖 → 新路径）")
	pathA, nodePathA = triggerDeclarativeUpgradeClusterInstall(clusterManager, cfgA, nameA, kubeconfigPath)
	By("触发安装无 helm-component 注解的 1M1W 业务集群 B（全局 Gate 关闭 → Legacy）")
	pathB, nodePathB = triggerDeclarativeUpgradeClusterInstall(clusterManager, cfgB, nameB, kubeconfigPath)
	waitDeclarativeUpgradeClustersInstallReady(clusterManager, kubeconfigPath, nameA, nameB)

	verifyDeclarativeUpgradeResources(duHelper, nameA, kubeconfigPath, sourceVersion)
	verifyDeclarativeUpgradeResources(duHelper, nameB, kubeconfigPath, sourceVersion)

	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, nameA, kubeconfigPath, false)
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, nameB, kubeconfigPath, false)
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, targetVersion)

	By("同时触发两集群升级（先后 patch desiredVersion，再并行等待收敛）")
	Expect(duHelper.PatchClusterVersionDesiredVersion(nameA, kubeconfigPath, targetVersion)).To(Succeed())
	Expect(duHelper.PatchClusterVersionDesiredVersion(nameB, kubeconfigPath, targetVersion)).To(Succeed())

	Eventually(func() bool {
		doneA := isDeclarativeUpgradeWorkloadReady(clusterManager, duHelper, nameA, kubeconfigPath, targetVersion)
		doneB := isDeclarativeUpgradeWorkloadReady(clusterManager, duHelper, nameB, kubeconfigPath, targetVersion)
		GinkgoWriter.Printf("FIT_7 双集群升级等待: A(%s)=%v B(%s)=%v\n", nameA, doneA, nameB, doneB)
		return doneA && doneB
	}, upgradeTimeout, pollInterval).Should(BeTrue(),
		"集群 A（注解新路径）与集群 B（Legacy）应同时升级收敛到 %s", targetVersion)

	verifyDeclarativeUpgradeUpgradeComplete(
		duHelper, clusterManager, nameA, kubeconfigPath, targetVersion,
		[][]string{{sourceVersion, targetVersion}},
	)
	verifyDeclarativeUpgradeUpgradeComplete(
		duHelper, clusterManager, nameB, kubeconfigPath, targetVersion,
		[][]string{{sourceVersion, targetVersion}},
	)

	By("校验集群 A：注解覆盖全局 Gate，fit-yaml-gate 写 clusterComponentStatuses Installed")
	verifyFITYamlComponentNewPathInstalled(
		duHelper, nameA, kubeconfigPath, utils.FITYamlGateComponent, targetVersion,
	)

	By("校验集群 B：无注解走 Legacy，CCS 不写 Installed，completed 含 fit-yaml-gate")
	verifyFITYamlComponentsLegacyPath(duHelper, nameB, kubeconfigPath, utils.FITYamlBaselineComponents)

	By("对比两集群写回差异：仅带注解集群 A 出现新路径 Installed")
	statusA, errA := duHelper.GetClusterComponentStatus(nameA, utils.FITYamlGateComponent, kubeconfigPath)
	Expect(errA).NotTo(HaveOccurred(), "集群 A 应有 clusterComponentStatuses[%s]", utils.FITYamlGateComponent)
	Expect(statusA.Phase).To(Equal(utils.ClusterComponentPhaseInstalled))
	verifyFITYamlComponentNotNewPathInstalled(duHelper, nameB, kubeconfigPath, utils.FITYamlGateComponent)
	GinkgoWriter.Printf("FIT_7 对比完成: A phase=%s（新路径）; B 无 Installed（Legacy）\n", statusA.Phase)
}

// isDeclarativeUpgradeWorkloadReady 业务集群 CV 已达目标版本且 BC Healthy/Ready。
func isDeclarativeUpgradeWorkloadReady(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) bool {
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
	if err != nil {
		GinkgoWriter.Printf("获取 BC %s 状态失败: %v\n", clusterName, err)
		return false
	}
	if state == "DeployFailed" {
		Fail(fmt.Sprintf("集群 %s 升级失败: state=%s phase=%s clusterStatus=%s",
			clusterName, state, phase, clusterStatus))
	}
	cv, cvErr := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	if cvErr != nil {
		return false
	}
	return cv.CurrentVersion == targetVersion &&
		(cv.Phase == "Ready" || cv.Phase == "Installed") &&
		clusterStatus == "Ready"
}

// snapshotFITComponentPhaseBeforeUpgrade 记录升级前 probe 组件 phase（可为空）。
func snapshotFITComponentPhaseBeforeUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName string,
) string {
	status, err := duHelper.GetClusterComponentStatus(clusterName, componentName, kubeconfigPath)
	if err != nil {
		GinkgoWriter.Printf("升级前 clusterComponentStatuses[%s] 不存在\n", componentName)
		return ""
	}
	GinkgoWriter.Printf("升级前 clusterComponentStatuses[%s] phase=%s\n", componentName, status.Phase)
	return status.Phase
}

// expectPreUpgradePhaseNotInstalled 升级前不应已有 Installed（非本轮新路径写入）。
func expectPreUpgradePhaseNotInstalled(componentName, phaseBefore string) {
	if phaseBefore == "" {
		return
	}
	Expect(phaseBefore).NotTo(Equal(utils.ClusterComponentPhaseInstalled),
		"升级前 clusterComponentStatuses[%s] 不应已为 Installed", componentName)
}
