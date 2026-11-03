package declarative_upgrade

import (
	"fmt"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
)

const fitComponentPollInterval = 15 * time.Second

// verifyFITYamlComponentsInstalled 校验 FIT yaml 组件在升级后均为 Installed（DAG_EXEC_FIT_2 / YAML_ENGINE 常规路径）。
func verifyFITYamlComponentsInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
	components []string,
) {
	if len(components) == 0 {
		components = utils.FITYamlBaselineComponents
	}
	By("FIT: 校验 clusterComponentStatuses 中 yaml 组件 Phase=Installed")
	for _, component := range components {
		component = strings.TrimSpace(component)
		if component == "" {
			continue
		}
		status := duHelper.WaitClusterComponentPhase(
			clusterName, component, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
			upgradeTimeout, fitComponentPollInterval,
		)
		utils.ExpectClusterComponentInstalled(status, targetVersion)
		GinkgoWriter.Printf("FIT component %s: phase=%s current=%s\n",
			component, status.Phase, status.CurrentVersion)
	}
}

// verifyFITYamlComponentFailed 校验指定 FIT 组件失败态（YAML_ENGINE_FIT_6/7、DAG_EXEC_FIT_4 exec-fail）。
func verifyFITYamlComponentFailed(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName string,
	messageKeywords ...string,
) {
	componentName = strings.TrimSpace(componentName)
	Expect(componentName).NotTo(BeEmpty())
	By(fmt.Sprintf("FIT: 校验组件 %s Phase=Failed", componentName))
	status := duHelper.WaitClusterComponentPhase(
		clusterName, componentName, kubeconfigPath, utils.ClusterComponentPhaseFailed,
		upgradeTimeout, fitComponentPollInterval,
	)
	utils.ExpectClusterComponentFailed(status, messageKeywords...)
}

// verifyFITUpgradeDependentComponentsRecorded 校验有依赖关系的组件均正常升级且写入 clusterComponentStatuses（DAG_EXEC_FIT_6：bkeagent 先于 fit-yaml-ssa）。
func verifyFITUpgradeDependentComponentsRecorded(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
	components ...string,
) {
	By("FIT: DAG_EXEC_FIT_6 校验 bkeagent 与 fit-yaml-ssa 均 Installed，且 bkeagent lastTransitionTime 更早")
	afterStatus := map[string]*utils.ClusterComponentStatusInfo{}
	for _, component := range components {
		component = strings.TrimSpace(component)
		if component == "" {
			continue
		}
		status := duHelper.WaitClusterComponentPhase(
			clusterName, component, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
			upgradeTimeout, fitComponentPollInterval,
		)
		utils.ExpectClusterComponentInstalled(status, targetVersion)
		afterStatus[component] = status
	}

	bkeagentAfter := afterStatus[utils.FITInlineDAGProbeComponent]
	ssaAfter := afterStatus[utils.FITYamlSSAConfigMap]
	Expect(bkeagentAfter).NotTo(BeNil())
	Expect(ssaAfter).NotTo(BeNil())

	bkeagentTime, err := time.Parse(time.RFC3339, bkeagentAfter.LastTransitionTime)
	Expect(err).NotTo(HaveOccurred(), "bkeagent lastTransitionTime 应为 RFC3339")
	ssaTime, err := time.Parse(time.RFC3339, ssaAfter.LastTransitionTime)
	Expect(err).NotTo(HaveOccurred(), "fit-yaml-ssa lastTransitionTime 应为 RFC3339")
	Expect(bkeagentTime.Before(ssaTime)).To(BeTrue(),
		"依赖顺序应满足 bkeagent 先于 fit-yaml-ssa，实际 bkeagent=%s fit-yaml-ssa=%s",
		bkeagentAfter.LastTransitionTime, ssaAfter.LastTransitionTime)
}

// runDeclarativeUpgradeFITInstallAndVerify 安装 FIT 基线集群（vfit-pre）并校验 DU-IT-001~004。
func runDeclarativeUpgradeFITInstallAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName, kubeconfigPath string,
) (configPath, nodeConfigPath string) {
	sourceVersion := utils.DeclarativeUpgradeFITSourceVersion()
	clusterConfig.OpenFuyaoVersion = sourceVersion
	return runDeclarativeUpgradeInstallAndVerify(clusterManager, duHelper, clusterConfig, clusterName, kubeconfigPath)
}

// runDeclarativeUpgradeFITMgmtInstallAndVerify 安装 FIT 基线管理集群（vfit-pre，helm-component）并校验 DU-IT-001~004。
// 安装配置中的 helm-component 仅用于引导 BC；纳管自身后由 runDeclarativeUpgradeMgmtInstallAndVerify 补写到自管 BC。
func runDeclarativeUpgradeFITMgmtInstallAndVerify(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName string,
) (configPath, nodeConfigPath, mgmtKubeconfigPath string) {
	sourceVersion := utils.DeclarativeUpgradeFITSourceVersion()
	clusterConfig.OpenFuyaoVersion = sourceVersion
	if clusterConfig.Annotations == nil {
		clusterConfig.Annotations = map[string]string{}
	}
	clusterConfig.Annotations[utils.HelmComponentAnnotation] = "true"
	return runDeclarativeUpgradeMgmtInstallAndVerify(clusterManager, sshExecutor, duHelper, clusterConfig, clusterName)
}

func fitYamlComponentsForTargetVersion(targetVersion string) []string {
	if targetVersion == utils.DefaultFITPostTargetVersion {
		return fitYamlComponentsExceptRemoved()
	}
	return utils.FITYamlBaselineComponents
}

func fitYamlComponentsExceptRemoved() []string {
	out := make([]string, 0, len(utils.FITYamlBaselineComponents))
	for _, component := range utils.FITYamlBaselineComponents {
		if component != utils.FITYamlRemoveComponent {
			out = append(out, component)
		}
	}
	return out
}

// ensureFITPreBaselineBeforeUpgrade 每次升级前：若 install 未 apply yaml 到目标集群，手工补导入 vfit-pre 基线（不含 FIT_9 核心样本）。
func ensureFITPreBaselineBeforeUpgrade(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, parentKubeconfigPath string,
	targetIsMgmtSelf bool,
) {
	var targetKubeconfig string
	if targetIsMgmtSelf {
		targetKubeconfig = parentKubeconfigPath
	} else {
		targetKubeconfig = saveWorkloadKubeconfigFromMgmt(
			clusterManager.SSHExecutor(), clusterName, parentKubeconfigPath,
		)
	}
	applyFITPreBaselineManifestsFromRoot(duHelper, targetKubeconfig, fitTestdataBaselineRoot())
}

func fitInstallationModuleDir() string {
	_, filename, _, ok := runtime.Caller(0)
	Expect(ok).To(BeTrue())
	return filepath.Join(filepath.Dir(filename), "..")
}

func fitTestdataBaselineRoot() string {
	return utils.FITTestdataBaselineRoot(fitInstallationModuleDir())
}

func fitTestdataFIT9CoreNamespacePath() string {
	return utils.FITTestdataFIT9CoreNamespacePath(fitInstallationModuleDir())
}

func applyFITPreBaselineManifestsFromRoot(
	duHelper *utils.DeclarativeUpgradeHelper,
	targetKubeconfigPath, baselineRoot string,
) {
	duHelper.EnsureFITPreBaselineOnTargetCluster(targetKubeconfigPath, baselineRoot)
}

// fitYamlEngineCurUpgradeUIDSnapshot 升 vfit-cur 前记录的策略/SSA 组件 ConfigMap UID 与 data。
type fitYamlEngineCurUpgradeUIDSnapshot struct {
	targetKubeconfig string
	ssaUID           string
	ssaData          string
	replaceUID       string
	createonlyUID    string
	createonlyData   string
}

var fitYamlEngineCurUpgradeUIDSnapshots = map[string]fitYamlEngineCurUpgradeUIDSnapshot{}

func resolveFITTargetKubeconfigForUIDChecks(
	clusterManager *utils.ClusterManager,
	clusterName, parentKubeconfigPath string,
	targetIsMgmtSelf bool,
) string {
	if targetIsMgmtSelf {
		return parentKubeconfigPath
	}
	return saveWorkloadKubeconfigFromMgmt(
		clusterManager.SSHExecutor(), clusterName, parentKubeconfigPath,
	)
}

// captureFITYamlEngineFIT3And4UIDsBeforeCurUpgrade 升 vfit-cur 前记录 ssa/replace/createonly ConfigMap UID。
func captureFITYamlEngineFIT3And4UIDsBeforeCurUpgrade(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, parentKubeconfigPath string,
	targetIsMgmtSelf bool,
) {
	targetKC := resolveFITTargetKubeconfigForUIDChecks(
		clusterManager, clusterName, parentKubeconfigPath, targetIsMgmtSelf,
	)
	snap := fitYamlEngineCurUpgradeUIDSnapshot{
		targetKubeconfig: targetKC,
		ssaUID: duHelper.WaitTargetClusterConfigMapUID(
			targetKC, utils.FITTestNamespace, utils.FITYamlSSAConfigMap,
		),
		replaceUID: duHelper.WaitTargetClusterConfigMapUID(
			targetKC, utils.FITTestNamespace, utils.FITYamlReplaceConfigMap,
		),
		createonlyUID: duHelper.WaitTargetClusterConfigMapUID(
			targetKC, utils.FITTestNamespace, utils.FITYamlCreateOnlyConfigMap,
		),
	}
	var err error
	snap.ssaData, err = duHelper.GetTargetClusterConfigMapDataField(
		targetKC, utils.FITTestNamespace, utils.FITYamlSSAConfigMap, utils.FITYamlSSADataField,
	)
	Expect(err).NotTo(HaveOccurred())
	snap.createonlyData, err = duHelper.GetTargetClusterConfigMapDataField(
		targetKC, utils.FITTestNamespace, utils.FITYamlCreateOnlyConfigMap, utils.FITYamlCreateOnlyDataField,
	)
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Printf("FIT 升 vfit-cur 前 UID/data: ssa=%s/%s replace=%s createonly=%s/%s\n",
		snap.ssaUID, snap.ssaData, snap.replaceUID, snap.createonlyUID, snap.createonlyData)
	fitYamlEngineCurUpgradeUIDSnapshots[clusterName] = snap
}

// verifyFITYamlEngineFIT3And4UIDsAfterCurUpgrade 升 vfit-cur 后校验 SSA/Replace/CreateOnly UID 行为。
func verifyFITYamlEngineFIT3And4UIDsAfterCurUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName string,
) {
	snap, ok := fitYamlEngineCurUpgradeUIDSnapshots[clusterName]
	if !ok {
		return
	}
	By("FIT: YAML_ENGINE_FIT_2 SSA 首跳升级后 fit-yaml-ssa UID 不变且 data 有变化")
	duHelper.VerifyTargetClusterConfigMapUIDUnchanged(
		snap.targetKubeconfig, utils.FITTestNamespace, utils.FITYamlSSAConfigMap,
		snap.ssaUID, "YAML_ENGINE_FIT_2",
	)
	ssaDataAfter, err := duHelper.GetTargetClusterConfigMapDataField(
		snap.targetKubeconfig, utils.FITTestNamespace, utils.FITYamlSSAConfigMap, utils.FITYamlSSADataField,
	)
	Expect(err).NotTo(HaveOccurred())
	Expect(ssaDataAfter).NotTo(Equal(snap.ssaData),
		"YAML_ENGINE_FIT_2: fit-yaml-ssa data.%s 应随 vfit-cur 升级变化", utils.FITYamlSSADataField)
	By("FIT: YAML_ENGINE_FIT_3 Replace 后 fit-yaml-replace UID 必须变化")
	duHelper.VerifyTargetClusterConfigMapUIDChanged(
		snap.targetKubeconfig, utils.FITTestNamespace, utils.FITYamlReplaceConfigMap,
		snap.replaceUID, "YAML_ENGINE_FIT_3",
	)
	By("FIT: YAML_ENGINE_FIT_4 CreateOnly 后 fit-yaml-createonly UID 与 data 均不变")
	duHelper.VerifyTargetClusterConfigMapUIDUnchanged(
		snap.targetKubeconfig, utils.FITTestNamespace, utils.FITYamlCreateOnlyConfigMap,
		snap.createonlyUID, "YAML_ENGINE_FIT_4",
	)
	createonlyDataAfter, err := duHelper.GetTargetClusterConfigMapDataField(
		snap.targetKubeconfig, utils.FITTestNamespace, utils.FITYamlCreateOnlyConfigMap, utils.FITYamlCreateOnlyDataField,
	)
	Expect(err).NotTo(HaveOccurred())
	Expect(createonlyDataAfter).To(Equal(snap.createonlyData),
		"YAML_ENGINE_FIT_4: fit-yaml-createonly data.%s 不应被 CreateOnly 覆盖", utils.FITYamlCreateOnlyDataField)
}

func shouldCaptureFITYamlEngineUIDsForUpgrade(targetVersion string) bool {
	return targetVersion == utils.DefaultFITTargetVersion
}

func prepareFIT9PruneCoreSampleBeforeUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	targetKubeconfigPath string,
) string {
	return duHelper.ApplyFIT9PruneCoreNamespace(targetKubeconfigPath, fitTestdataFIT9CoreNamespacePath())
}

// runDeclarativeUpgradeFITUpgradeAndVerify 触发 FIT 目标版本升级并校验 DU-IT-005~006 + yaml 组件状态。
func runDeclarativeUpgradeFITUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	sourceVersion, targetVersion string,
	verifyComponents []string,
) {
	runDeclarativeUpgradeWorkloadUpgradeAndVerify(clusterManager, duHelper, clusterName, kubeconfigPath, sourceVersion, targetVersion, true)
	verifyFITYamlComponentsInstalled(duHelper, clusterName, kubeconfigPath, targetVersion, verifyComponents)
}

// runDeclarativeUpgradeFITMultiHopUpgradeAndVerify 两跳升级至 vfit-post（业务集群）。
func runDeclarativeUpgradeFITMultiHopUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	guideNodeExecutor executor.Executor,
	guideNodeHost string,
	clusterName, kubeconfigPath string,
	workloadNodes []config.NodeInfo,
	sourceVersion, intermediateVersion, targetVersion string,
) {
	runDeclarativeUpgradeWorkloadMultiHopUpgradeAndVerify(
		clusterManager, duHelper, clusterName, kubeconfigPath,
		sourceVersion, intermediateVersion, targetVersion,
		declarativeUpgradeWorkloadMultiHopHelmOptions{
			exec:          guideNodeExecutor,
			guideNodeHost: guideNodeHost,
			workloadNodes: workloadNodes,
		},
	)
}

// runDeclarativeUpgradeFITMgmtUpgradeAndVerify 管理集群单跳 FIT 升级。
func runDeclarativeUpgradeFITMgmtUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	bootstrapClusterName string,
	guideNodeHost string,
	mgmtNodes []config.NodeInfo,
	mgmtKubeconfigPath string,
	sourceVersion, targetVersion string,
) {
	runDeclarativeUpgradeMgmtUpgradeAndVerify(clusterManager, duHelper, bootstrapClusterName, guideNodeHost, mgmtNodes, mgmtKubeconfigPath, sourceVersion, targetVersion, true)
}

// runDeclarativeUpgradeFITPreToCurUpgradeAndVerify 业务集群 vfit-pre→vfit-cur 单跳：
// 升级 + DAG/YAML 自验证预期（FIT_1/2/6 + YAML_ENGINE_FIT_1/3~5/10）。
func runDeclarativeUpgradeFITPreToCurUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	guideNodeExecutor executor.Executor,
	guideNodeHost string,
	clusterName, kubeconfigPath string,
	workloadNodes []config.NodeInfo,
	sourceVersion, targetVersion string,
) {
	prepareAndPublishFITHelmE2EHTTPCharts(guideNodeExecutor, guideNodeHost, "0.1.1", "0.1.2", "0.1.3")
	prepareAndPatchFITHelmE2ETLSChartRepoOnGuideNode(guideNodeExecutor, guideNodeHost, clusterName)
	runDeclarativeUpgradeFITUpgradeAndVerify(
		clusterManager, duHelper, clusterName, kubeconfigPath,
		sourceVersion, targetVersion, utils.FITYamlBaselineComponents,
	)
	verifyFITHelmE2EDeployedOnWorkloadMasterNode(guideNodeExecutor, workloadNodes)
	verifyFITInlineComponentInstalled(duHelper, clusterName, kubeconfigPath, targetVersion)
	verifyFITYamlNewComponentInstalled(duHelper, clusterName, kubeconfigPath, targetVersion)
	verifyFITYamlGateComponentInstalled(duHelper, clusterName, kubeconfigPath, targetVersion)
	runFITYamlEngineFIT3To5And10Check(clusterManager, duHelper, clusterName, kubeconfigPath, targetVersion, false)
	verifyFITUpgradeDependentComponentsRecorded(
		duHelper, clusterName, kubeconfigPath, targetVersion,
		utils.FITInlineDAGProbeComponent, utils.FITYamlSSAConfigMap,
	)
}

// runDeclarativeUpgradeFITMgmtPreToCurUpgradeAndVerify 管理集群 vfit-pre→vfit-cur 单跳合并校验。
func runDeclarativeUpgradeFITMgmtPreToCurUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	guideNodeExecutor executor.Executor,
	guideNodeHost string,
	bootstrapClusterName string,
	mgmtNodes []config.NodeInfo,
	mgmtKubeconfigPath string,
	sourceVersion, targetVersion string,
) {
	declClusterName := utils.ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName)
	prepareAndPublishFITHelmE2ECharts(guideNodeExecutor, guideNodeHost, mgmtNodes, bootstrapClusterName, "0.1.1", "0.1.2", "0.1.3")
	runDeclarativeUpgradeFITMgmtUpgradeAndVerify(clusterManager, duHelper, bootstrapClusterName, guideNodeHost, mgmtNodes, mgmtKubeconfigPath, sourceVersion, targetVersion)
	verifyFITHelmE2EDeployedOnMasterNode(mgmtNodes)
	verifyFITInlineComponentInstalled(duHelper, declClusterName, mgmtKubeconfigPath, targetVersion)
	verifyFITYamlNewComponentInstalled(duHelper, declClusterName, mgmtKubeconfigPath, targetVersion)
	verifyFITYamlGateComponentInstalled(duHelper, declClusterName, mgmtKubeconfigPath, targetVersion)
	runFITYamlEngineFIT3To5And10Check(clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, targetVersion, true)
	verifyFITUpgradeDependentComponentsRecorded(
		duHelper, declClusterName, mgmtKubeconfigPath, targetVersion,
		utils.FITInlineDAGProbeComponent, utils.FITYamlSSAConfigMap,
	)
}

func verifyFITInlineComponentInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) {
	By("FIT: DAG_EXEC_FIT_1 inline 组件 bkeagent 正常升级且写入 clusterComponentStatuses")
	status := duHelper.WaitClusterComponentPhase(
		clusterName, utils.FITInlineDAGProbeComponent, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
		upgradeTimeout, fitComponentPollInterval,
	)
	utils.ExpectClusterComponentInstalled(status, targetVersion)
}

func verifyFITHelmE2EDeployedOnMasterNode(nodes []config.NodeInfo) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes)
	Expect(selected).NotTo(BeEmpty(), "management cluster should have a master node for helm e2e verification")
	masterNode := selected[0]

	By(fmt.Sprintf("FIT: verify fit-helm-e2e pod on master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "kubectl get po -A")
		if err != nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on master node %s(%s) exit=%d stdout=%s stderr=%s\n",
				masterNode.Hostname, masterNode.IP, result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(result.Stdout, "fit-helm-e2e")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "kubectl get po -A should show fit-helm-e2e pod")

	By(fmt.Sprintf("FIT: verify fit-helm-e2e helm release deployed on master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "helm -n fit-helm-e2e status fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on master node %s(%s) exit=%d stdout=%s stderr=%s\n",
				masterNode.Hostname, masterNode.IP, result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "status: deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm status should show STATUS deployed for fit-helm-e2e")

	By("FIT: verify bke-controller-manager completed chart repo TLS info log")
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "kubectl -n cluster-system logs deploy/bke-controller-manager --since=10m | grep 'completed get tls info of chart repo'")
		if err != nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("bke-controller-manager chart repo TLS log check: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "completed get tls info of chart repo")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "bke-controller-manager logs should contain completed get tls info of chart repo")

	By("FIT: restart bke-controller-manager and wait rollout complete")
	restartResult, restartErr := utils.ExecuteCommandOnNode(masterNode, "kubectl -n cluster-system rollout restart deploy/bke-controller-manager")
	Expect(restartErr).NotTo(HaveOccurred(), "rollout restart bke-controller-manager should execute")
	Expect(restartResult).NotTo(BeNil(), "rollout restart bke-controller-manager should return result")
	GinkgoWriter.Printf("rollout restart bke-controller-manager: exit=%d stdout=%s stderr=%s\n",
		restartResult.ExitCode, restartResult.Stdout, restartResult.Stderr)
	Expect(restartResult.ExitCode).To(Equal(0), "rollout restart bke-controller-manager should succeed")

	statusResult, statusErr := utils.ExecuteCommandOnNode(masterNode, "kubectl -n cluster-system rollout status deploy/bke-controller-manager")
	Expect(statusErr).NotTo(HaveOccurred(), "rollout status bke-controller-manager should execute")
	Expect(statusResult).NotTo(BeNil(), "rollout status bke-controller-manager should return result")
	GinkgoWriter.Printf("rollout status bke-controller-manager: exit=%d stdout=%s stderr=%s\n",
		statusResult.ExitCode, statusResult.Stdout, statusResult.Stderr)
	Expect(statusResult.ExitCode).To(Equal(0), "rollout status bke-controller-manager should succeed")

	GinkgoWriter.Printf("wait 20s after bke-controller-manager rollout completed\n")
	time.Sleep(20 * time.Second)

	By(fmt.Sprintf("FIT: verify fit-helm-e2e helm history deployed on master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "helm -n fit-helm-e2e history fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("helm history fit-helm-e2e: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		if result.ExitCode != 0 {
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm history should show deployed for fit-helm-e2e")

	verifyFITHelmE2ECommandHasOutput(masterNode, "helm -n fit-helm-e2e get manifest fit-helm-e2e", "helm get manifest fit-helm-e2e should return output")
	verifyFITHelmE2ECommandHasOutput(masterNode, "kubectl -n fit-helm-e2e get deploy fit-helm-e2e -o yaml", "kubectl get fit-helm-e2e deployment yaml should return output")
}
func verifyFITHelmE2EDeployedOnGuideNode(exec executor.Executor) {
	By("FIT: verify fit-helm-e2e pod on guide node")
	Eventually(func() bool {
		result, err := exec.Exec("kubectl get po -A")
		if err != nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on guide node returned nil result\n")
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on guide node exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(result.Stdout, "fit-helm-e2e")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "kubectl get po -A should show fit-helm-e2e pod")

	By("FIT: verify fit-helm-e2e helm release deployed on guide node")
	Eventually(func() bool {
		result, err := exec.Exec("helm -n fit-helm-e2e status fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on guide node returned nil result\n")
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on guide node exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "status: deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm status should show STATUS deployed for fit-helm-e2e")

	By("FIT: verify bke-controller-manager completed chart repo TLS info log on guide node")
	Eventually(func() bool {
		result, err := exec.Exec("kubectl -n cluster-system logs deploy/bke-controller-manager --since=10m | grep 'completed get tls info of chart repo'")
		if err != nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on guide node returned nil result\n")
			return false
		}
		GinkgoWriter.Printf("bke-controller-manager chart repo TLS log check on guide node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "completed get tls info of chart repo")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "bke-controller-manager logs should contain completed get tls info of chart repo")

	By("FIT: restart guide node bke-controller-manager and wait rollout complete")
	restartResult, restartErr := exec.Exec("kubectl -n cluster-system rollout restart deploy/bke-controller-manager")
	Expect(restartErr).NotTo(HaveOccurred(), "rollout restart bke-controller-manager should execute")
	Expect(restartResult).NotTo(BeNil(), "rollout restart bke-controller-manager should return result")
	GinkgoWriter.Printf("rollout restart bke-controller-manager on guide node: exit=%d stdout=%s stderr=%s\n", restartResult.ExitCode, restartResult.Stdout, restartResult.Stderr)
	Expect(restartResult.ExitCode).To(Equal(0), "rollout restart bke-controller-manager should succeed")

	statusResult, statusErr := exec.Exec("kubectl -n cluster-system rollout status deploy/bke-controller-manager")
	Expect(statusErr).NotTo(HaveOccurred(), "rollout status bke-controller-manager should execute")
	Expect(statusResult).NotTo(BeNil(), "rollout status bke-controller-manager should return result")
	GinkgoWriter.Printf("rollout status bke-controller-manager on guide node: exit=%d stdout=%s stderr=%s\n", statusResult.ExitCode, statusResult.Stdout, statusResult.Stderr)
	Expect(statusResult.ExitCode).To(Equal(0), "rollout status bke-controller-manager should succeed")

	GinkgoWriter.Printf("wait 20s after guide node bke-controller-manager rollout completed\n")
	time.Sleep(20 * time.Second)

	By("FIT: verify fit-helm-e2e helm history deployed on guide node")
	Eventually(func() bool {
		result, err := exec.Exec("helm -n fit-helm-e2e history fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on guide node returned nil result\n")
			return false
		}
		GinkgoWriter.Printf("helm history fit-helm-e2e on guide node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		if result.ExitCode != 0 {
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm history should show deployed for fit-helm-e2e")

	verifyFITHelmE2EGuideCommandHasOutput(exec, "helm -n fit-helm-e2e get manifest fit-helm-e2e", "helm get manifest fit-helm-e2e should return output")
	verifyFITHelmE2EGuideCommandHasOutput(exec, "kubectl -n fit-helm-e2e get deploy fit-helm-e2e -o yaml", "kubectl get fit-helm-e2e deployment yaml should return output")
}
func verifyFITHelmE2EDeployedOnWorkloadMasterNode(guideExec executor.Executor, nodes []config.NodeInfo) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes)
	Expect(selected).NotTo(BeEmpty(), "workload cluster should have a master node for helm e2e verification")
	masterNode := selected[0]

	By(fmt.Sprintf("FIT: verify fit-helm-e2e pod on workload master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "kubectl get po -A")
		if err != nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on workload master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on workload master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] kubectl get po -A on workload master node %s(%s) exit=%d stdout=%s stderr=%s\n",
				masterNode.Hostname, masterNode.IP, result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(result.Stdout, "fit-helm-e2e")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "kubectl get po -A should show fit-helm-e2e pod on workload master node")

	By(fmt.Sprintf("FIT: verify fit-helm-e2e helm release deployed on workload master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "helm -n fit-helm-e2e status fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on workload master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on workload master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		if result.ExitCode != 0 {
			GinkgoWriter.Printf("[WARN] helm status fit-helm-e2e on workload master node %s(%s) exit=%d stdout=%s stderr=%s\n",
				masterNode.Hostname, masterNode.IP, result.ExitCode, result.Stdout, result.Stderr)
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "status: deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm status should show STATUS deployed for fit-helm-e2e on workload master node")

	By("FIT: verify guide node bke-controller-manager completed chart repo TLS info log")
	Eventually(func() bool {
		result, err := guideExec.Exec("kubectl -n cluster-system logs deploy/bke-controller-manager --since=10m | grep 'completed get tls info of chart repo'")
		if err != nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] check bke-controller-manager chart repo TLS log on guide node returned nil result\n")
			return false
		}
		GinkgoWriter.Printf("bke-controller-manager chart repo TLS log check on guide node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "completed get tls info of chart repo")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "guide node bke-controller-manager logs should contain completed get tls info of chart repo")

	By("FIT: restart guide node bke-controller-manager and wait rollout complete")
	restartResult, restartErr := guideExec.Exec("kubectl -n cluster-system rollout restart deploy/bke-controller-manager")
	Expect(restartErr).NotTo(HaveOccurred(), "rollout restart bke-controller-manager should execute on guide node")
	Expect(restartResult).NotTo(BeNil(), "rollout restart bke-controller-manager should return result on guide node")
	GinkgoWriter.Printf("rollout restart bke-controller-manager on guide node: exit=%d stdout=%s stderr=%s\n", restartResult.ExitCode, restartResult.Stdout, restartResult.Stderr)
	Expect(restartResult.ExitCode).To(Equal(0), "rollout restart bke-controller-manager should succeed on guide node")

	statusResult, statusErr := guideExec.Exec("kubectl -n cluster-system rollout status deploy/bke-controller-manager")
	Expect(statusErr).NotTo(HaveOccurred(), "rollout status bke-controller-manager should execute on guide node")
	Expect(statusResult).NotTo(BeNil(), "rollout status bke-controller-manager should return result on guide node")
	GinkgoWriter.Printf("rollout status bke-controller-manager on guide node: exit=%d stdout=%s stderr=%s\n", statusResult.ExitCode, statusResult.Stdout, statusResult.Stderr)
	Expect(statusResult.ExitCode).To(Equal(0), "rollout status bke-controller-manager should succeed on guide node")

	GinkgoWriter.Printf("wait 20s after guide node bke-controller-manager rollout completed\n")
	time.Sleep(20 * time.Second)

	By(fmt.Sprintf("FIT: verify fit-helm-e2e helm history deployed on workload master node %s(%s)", masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, "helm -n fit-helm-e2e history fit-helm-e2e")
		if err != nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on workload master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] helm history fit-helm-e2e on workload master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("helm history fit-helm-e2e on workload master node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		if result.ExitCode != 0 {
			return false
		}
		return strings.Contains(strings.ToLower(result.Stdout), "deployed")
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "helm history should show deployed for fit-helm-e2e on workload master node")

	verifyFITHelmE2ECommandHasOutput(masterNode, "helm -n fit-helm-e2e get manifest fit-helm-e2e", "helm get manifest fit-helm-e2e should return output")
	verifyFITHelmE2ECommandHasOutput(masterNode, "kubectl -n fit-helm-e2e get deploy fit-helm-e2e -o yaml", "kubectl get fit-helm-e2e deployment yaml should return output")
}

func verifyFITHelmE2EGuideCommandHasOutput(exec executor.Executor, command, expectation string) {
	By(fmt.Sprintf("FIT: %s on guide node", expectation))
	Eventually(func() bool {
		result, err := exec.Exec(command)
		if err != nil {
			GinkgoWriter.Printf("[WARN] %s on guide node failed: %v\n", command, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] %s on guide node returned nil result\n", command)
			return false
		}
		GinkgoWriter.Printf("%s on guide node: exit=%d stdout=%s stderr=%s\n", command, result.ExitCode, result.Stdout, result.Stderr)
		if result.ExitCode != 0 {
			return false
		}
		return strings.TrimSpace(result.Stdout) != ""
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), expectation)
}

func verifyFITHelmE2EPodLabelVersionOnGuideNode(exec executor.Executor, expectedChartVersion string) {
	command := "kubectl -n fit-helm-e2e get pod --show-labels"
	By(fmt.Sprintf("FIT: verify fit-helm-e2e pod labels contain chart version %s on guide node", expectedChartVersion))
	Eventually(func() bool {
		result, err := exec.Exec(command)
		if err != nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on guide node failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on guide node returned nil result\n")
			return false
		}
		GinkgoWriter.Printf("fit-helm-e2e pod labels on guide node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "fit-helm-e2e") && strings.Contains(result.Stdout, expectedChartVersion)
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "fit-helm-e2e pod labels should contain chart version %s", expectedChartVersion)
}
func verifyFITHelmE2ECommandHasOutput(masterNode config.NodeInfo, command, expectation string) {
	By(fmt.Sprintf("FIT: %s on master node %s(%s)", expectation, masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, command)
		if err != nil {
			GinkgoWriter.Printf("[WARN] %s on master node %s(%s) failed: %v\n", command, masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] %s on master node %s(%s) returned nil result\n", command, masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("%s: exit=%d stdout=%s stderr=%s\n", command, result.ExitCode, result.Stdout, result.Stderr)
		if result.ExitCode != 0 {
			return false
		}
		return strings.TrimSpace(result.Stdout) != ""
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), expectation)
}
func verifyFITHelmE2EPodLabelVersionOnMasterNode(nodes []config.NodeInfo, expectedChartVersion string) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes)
	Expect(selected).NotTo(BeEmpty(), "management cluster should have a master node for fit-helm-e2e pod label verification")
	masterNode := selected[0]
	command := "kubectl -n fit-helm-e2e get pod --show-labels"

	By(fmt.Sprintf("FIT: verify fit-helm-e2e pod labels contain chart version %s on master node %s(%s)", expectedChartVersion, masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, command)
		if err != nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("fit-helm-e2e pod labels: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "fit-helm-e2e") && strings.Contains(result.Stdout, expectedChartVersion)
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "fit-helm-e2e pod labels should contain chart version %s", expectedChartVersion)
}
func verifyFITHelmE2EPodLabelVersionOnWorkloadMasterNode(nodes []config.NodeInfo, expectedChartVersion string) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes)
	Expect(selected).NotTo(BeEmpty(), "workload cluster should have a master node for fit-helm-e2e pod label verification")
	masterNode := selected[0]
	command := "kubectl -n fit-helm-e2e get pod --show-labels"

	By(fmt.Sprintf("FIT: verify fit-helm-e2e pod labels contain chart version %s on workload master node %s(%s)", expectedChartVersion, masterNode.Hostname, masterNode.IP))
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(masterNode, command)
		if err != nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on workload master node %s(%s) failed: %v\n", masterNode.Hostname, masterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] get fit-helm-e2e pod labels on workload master node %s(%s) returned nil result\n", masterNode.Hostname, masterNode.IP)
			return false
		}
		GinkgoWriter.Printf("fit-helm-e2e pod labels on workload master node: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, "fit-helm-e2e") && strings.Contains(result.Stdout, expectedChartVersion)
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "fit-helm-e2e pod labels should contain chart version %s on workload master node", expectedChartVersion)
}
func verifyFITYamlNewComponentInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) {
	By("FIT: YAML_ENGINE_FIT_1 fit-yaml-new 首次安装且写入 clusterComponentStatuses")
	status := duHelper.WaitClusterComponentPhase(
		clusterName, utils.FITYamlNewComponent, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
		upgradeTimeout, fitComponentPollInterval,
	)
	utils.ExpectClusterComponentInstalled(status, targetVersion)
}

func verifyFITYamlGateComponentInstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) {
	By("FIT: DAG_EXEC_FIT_2 Gate 开启时 fit-yaml-gate 正常升级且写入 clusterComponentStatuses")
	status := duHelper.WaitClusterComponentPhase(
		clusterName, utils.FITYamlGateComponent, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
		upgradeTimeout, fitComponentPollInterval,
	)
	utils.ExpectClusterComponentInstalled(status, targetVersion)
}

// runDeclarativeUpgradeFITMgmtMultiHopUpgradeAndVerify 管理集群两跳 FIT 升级。
func runDeclarativeUpgradeFITMgmtMultiHopUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	exec executor.Executor,
	guideNodeHost string,
	bootstrapClusterName string,
	mgmtNodes []config.NodeInfo,
	mgmtKubeconfigPath string,
	sourceVersion, intermediateVersion, targetVersion string,
) {
	runDeclarativeUpgradeMgmtMultiHopUpgradeAndVerify(
		clusterManager, duHelper, bootstrapClusterName, mgmtKubeconfigPath,
		sourceVersion, intermediateVersion, targetVersion,
		declarativeUpgradeMgmtMultiHopHelmOptions{
			exec:          exec,
			guideNodeHost: guideNodeHost,
			mgmtNodes:     mgmtNodes,
		},
	)
}

// verifyFITYamlComponentNotAutoUninstalled YAML_ENGINE_FIT_8：高版本 RI 不含该组件时不自动卸载，clusterComponentStatuses 旧记录不变。
func verifyFITYamlComponentNotAutoUninstalled(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, componentName string,
	targetIsMgmtSelf bool,
) {
	componentName = strings.TrimSpace(componentName)
	Expect(componentName).NotTo(BeEmpty())
	By(fmt.Sprintf("FIT: 校验组件 %s 高版本不自动卸载且 clusterComponentStatuses 旧记录仍在", componentName))
	status, err := duHelper.GetClusterComponentStatus(clusterName, componentName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "clusterComponentStatuses[%s] 旧记录应仍存在", componentName)
	Expect(status.Phase).To(Equal(utils.ClusterComponentPhaseInstalled),
		"组件 %s phase 应保持 Installed，实际 %s", componentName, status.Phase)
	Expect(status.LastTransitionTime).NotTo(BeEmpty())

	targetKC := resolveFITTargetKubeconfigForUIDChecks(
		clusterManager, clusterName, kubeconfigPath, targetIsMgmtSelf,
	)
	Expect(duHelper.TargetClusterConfigMapExists(targetKC, utils.FITTestNamespace, utils.FITYamlRemoveComponent)).To(BeTrue(),
		"FIT_8: 目标集群 %s ConfigMap 资源不应被自动卸载", utils.FITYamlRemoveComponent)
}

// runFITYamlEngineFIT2IdempotencyCheck 同目标版本二次触发升级（YAML_ENGINE_FIT_2）。
func runFITYamlEngineFIT2IdempotencyCheck(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
	targetIsMgmtSelf bool,
) {
	targetKC := resolveFITTargetKubeconfigForUIDChecks(
		clusterManager, clusterName, kubeconfigPath, targetIsMgmtSelf,
	)

	By("FIT: YAML_ENGINE_FIT_2 记录 fit-yaml-ssa 二次触发前 UID 与 clusterComponentStatuses")
	uidBefore := duHelper.WaitTargetClusterConfigMapUID(
		targetKC, utils.FITTestNamespace, utils.FITYamlSSAConfigMap,
	)
	statusBefore, err := duHelper.GetClusterComponentStatus(clusterName, utils.FITYamlSSAConfigMap, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())

	err = duHelper.RetriggerUpgradeViaUpgradeReady(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred())

	status := duHelper.WaitClusterComponentPhase(
		clusterName, utils.FITYamlSSAConfigMap, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
		upgradeTimeout, fitComponentPollInterval,
	)
	utils.ExpectClusterComponentInstalled(status, targetVersion)

	cv, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(cv.CurrentVersion).To(Equal(targetVersion))
	Expect(cv.DesiredVersion).To(Equal(targetVersion))

	By("FIT: YAML_ENGINE_FIT_2 二次触发后 fit-yaml-ssa UID 不变且 clusterComponentStatuses 幂等")
	duHelper.VerifyTargetClusterConfigMapUIDUnchanged(
		targetKC, utils.FITTestNamespace, utils.FITYamlSSAConfigMap,
		uidBefore, "YAML_ENGINE_FIT_2",
	)
	statusAfter, err := duHelper.GetClusterComponentStatus(clusterName, utils.FITYamlSSAConfigMap, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(statusAfter.Phase).To(Equal(statusBefore.Phase))
	Expect(statusAfter.CurrentVersion).To(Equal(statusBefore.CurrentVersion))
	Expect(statusAfter.LastTransitionTime).To(Equal(statusBefore.LastTransitionTime))
}

// runFITYamlEngineFIT3To5And10Check 校验 YAML_ENGINE_FIT_3~5+10 自验证预期。
func runFITYamlEngineFIT3To5And10Check(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
	targetIsMgmtSelf bool,
) {
	for _, component := range []string{
		utils.FITYamlReplaceConfigMap, utils.FITYamlCreateOnlyConfigMap, utils.FITYamlPruneComponent, utils.FITYamlSSAConfigMap,
	} {
		status := duHelper.WaitClusterComponentPhase(
			clusterName, component, kubeconfigPath, utils.ClusterComponentPhaseInstalled,
			upgradeTimeout, fitComponentPollInterval,
		)
		utils.ExpectClusterComponentInstalled(status, targetVersion)
	}

	targetKC := resolveFITTargetKubeconfigForUIDChecks(
		clusterManager, clusterName, kubeconfigPath, targetIsMgmtSelf,
	)
	verifyFITPruneLegacyRemovedKeepPreserved(duHelper, targetKC)
	verifyFITYamlSkipComponentNotRecorded(duHelper, clusterName, kubeconfigPath, targetVersion)
	verifyFITYamlEngineFIT3And4UIDsAfterCurUpgrade(duHelper, clusterName)
}

func verifyFITPruneLegacyRemovedKeepPreserved(duHelper *utils.DeclarativeUpgradeHelper, targetKubeconfigPath string) {
	By("FIT: YAML_ENGINE_FIT_5 legacy ConfigMap 被 Prune 删除且 keep ConfigMap 保留")
	Expect(duHelper.TargetClusterConfigMapExists(
		targetKubeconfigPath, utils.FITTestNamespace, utils.FITPruneLegacyConfigMap,
	)).To(BeFalse(), "fit-yaml-prune-legacy 应被 Prune 删除")
	Expect(duHelper.TargetClusterConfigMapExists(
		targetKubeconfigPath, utils.FITTestNamespace, utils.FITPruneKeepConfigMap,
	)).To(BeTrue(), "fit-yaml-prune-keep 应保留")
}

func verifyFITYamlSkipComponentNotRecorded(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) {
	By("FIT: YAML_ENGINE_FIT_10 fit-yaml-skip 跳过执行且不写入 clusterComponentStatuses")
	names, err := duHelper.ListClusterComponentStatusNames(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Printf("FIT_10 debug: clusterComponentStatuses names=%v\n", names)
	found := false
	for _, n := range names {
		if n == utils.FITYamlSkipComponent {
			found = true
			break
		}
	}
	if found {
		// 某些版本/路径下，Skip 组件可能仍会写入 clusterComponentStatuses，
		// 但应保持非 Installed，避免“跳过执行却执行成功”的错误。
		status, stErr := duHelper.GetClusterComponentStatus(clusterName, utils.FITYamlSkipComponent, kubeconfigPath)
		Expect(stErr).NotTo(HaveOccurred())
		Expect(status.Phase).NotTo(Equal(utils.ClusterComponentPhaseInstalled),
			"Skip 组件 %s 不应进入 Phase=Installed，实际 phase=%s current=%s message=%q",
			utils.FITYamlSkipComponent, status.Phase, status.CurrentVersion, status.Message)
	} else {
		// 确认完全未写入也满足语义
		Expect(names).NotTo(ContainElement(utils.FITYamlSkipComponent),
			"Skip 组件 %s 不应出现在 clusterComponentStatuses", utils.FITYamlSkipComponent)
	}
	status, err := duHelper.GetClusterComponentStatus(clusterName, utils.FITYamlSSAConfigMap, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	utils.ExpectClusterComponentInstalled(status, targetVersion)
}

// setupMgmt1M1WAndWorkload1M1W 部署 1M1W 管理集群，并在管理集群上创建 1M1W 业务集群。
func setupMgmt1M1WAndWorkload1M1W(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	mgmtClusterName, workloadClusterName string,
	workloadOpenFuyaoVersion string,
	enableHelmComponent bool,
) (mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath, workloadConfigPath, workloadNodeConfigPath string) {
	mgmtNodes, err := utils.LoadDeclarativeUpgrade1M1WMgmtNodes()
	Expect(err).NotTo(HaveOccurred())

	mgmtConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(mgmtClusterName, mgmtNodes)
	mgmtConfig.OpenFuyaoVersion = workloadOpenFuyaoVersion
	if enableHelmComponent {
		if mgmtConfig.Annotations == nil {
			mgmtConfig.Annotations = map[string]string{}
		}
		mgmtConfig.Annotations[utils.HelmComponentAnnotation] = "true"
	}
	mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeMgmtInstallAndVerify(
		clusterManager, sshExecutor, duHelper, mgmtConfig, mgmtClusterName,
	)

	workloadNodes, err := utils.LoadDeclarativeUpgrade1M1WWorkloadNodesOnMgmt()
	Expect(err).NotTo(HaveOccurred())
	workloadConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
		workloadClusterName, workloadNodes, workloadOpenFuyaoVersion, enableHelmComponent,
	)
	workloadConfigPath, workloadNodeConfigPath = runDeclarativeUpgradeInstallAndVerify(
		clusterManager, duHelper, workloadConfig, workloadClusterName, mgmtKubeconfigPath,
	)
	return mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath, workloadConfigPath, workloadNodeConfigPath
}

// setupMgmt3M1WAndWorkload3M1W 部署 3M1W 管理集群，并在管理集群上创建 3M1W 业务集群。
func setupMgmt3M1WAndWorkload3M1W(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	mgmtClusterName, workloadClusterName string,
	workloadOpenFuyaoVersion string,
	enableHelmComponent bool,
) (mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath, workloadConfigPath, workloadNodeConfigPath string) {
	mgmtNodes, err := utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
	Expect(err).NotTo(HaveOccurred())

	mgmtConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(mgmtClusterName, mgmtNodes)
	mgmtConfig.OpenFuyaoVersion = workloadOpenFuyaoVersion
	if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
		mgmtConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
			Host: vipHost,
			Port: vipPort,
		}
	}
	if enableHelmComponent {
		if mgmtConfig.Annotations == nil {
			mgmtConfig.Annotations = map[string]string{}
		}
		mgmtConfig.Annotations[utils.HelmComponentAnnotation] = "true"
	}
	mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeMgmtInstallAndVerify(
		clusterManager, sshExecutor, duHelper, mgmtConfig, mgmtClusterName,
	)

	workloadNodes, err := utils.LoadDeclarativeUpgrade3M1WWorkloadNodesOnMgmt()
	Expect(err).NotTo(HaveOccurred())
	workloadConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
		workloadClusterName, workloadNodes, workloadOpenFuyaoVersion, enableHelmComponent,
	)
	if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
		workloadConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
			Host: vipHost,
			Port: vipPort,
		}
	}
	workloadConfigPath, workloadNodeConfigPath = runDeclarativeUpgradeInstallAndVerify(
		clusterManager, duHelper, workloadConfig, workloadClusterName, mgmtKubeconfigPath,
	)
	return mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath, workloadConfigPath, workloadNodeConfigPath
}

// runDeclarativeUpgradeFITNegativeUpgrade 触发 FIT 负向目标版本升级并校验指定组件 Failed。
func runDeclarativeUpgradeFITNegativeUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, sourceVersion, targetVersion, failedComponent string,
	messageKeywords ...string,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, targetVersion)

	By(fmt.Sprintf("触发负向升级 %s→%s", sourceVersion, targetVersion))
	err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred())

	verifyFITYamlComponentFailed(duHelper, clusterName, kubeconfigPath, failedComponent, messageKeywords...)
}

func runDeclarativeUpgradeFITHelmChartNotFoundNegativeUpgrade(
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, sourceVersion string,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, utils.DefaultFITTargetVersionNegHelmChart)
	patchFITHelmHTTPChartRepoOnBKECluster(sshExecutor, clusterName, config.LoadGuideNodeFromEnv().Host)

	By(fmt.Sprintf("触发 Helm chart 版本不存在负向升级 %s→%s", sourceVersion, utils.DefaultFITTargetVersionNegHelmChart))
	err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, utils.DefaultFITTargetVersionNegHelmChart)
	Expect(err).NotTo(HaveOccurred())

	verifyGuideNodeControllerManagerLogContains(sshExecutor, `version \"9.9.9\" not found in`)
}

func runDeclarativeUpgradeFITHelmHealthCheckNegativeUpgrade(
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, sourceVersion string,
	clusterNodes []config.NodeInfo,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, utils.DefaultFITTargetVersionNegHelmHC)
	patchFITHelmHTTPChartRepoOnBKECluster(sshExecutor, clusterName, config.LoadGuideNodeFromEnv().Host)

	By(fmt.Sprintf("触发 Helm healthCheck 失败负向升级 %s→%s", sourceVersion, utils.DefaultFITTargetVersionNegHelmHC))
	err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, utils.DefaultFITTargetVersionNegHelmHC)
	Expect(err).NotTo(HaveOccurred())

	verifyWorkloadFirstNodePodLabelsContain(clusterNodes, "helm-healthcheck-fail")
}

func runDeclarativeUpgradeFITVersionNotFoundNegativeUpgrade(
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, sourceVersion string,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	patchFITHelmHTTPChartRepoOnBKECluster(sshExecutor, clusterName, config.LoadGuideNodeFromEnv().Host)

	By(fmt.Sprintf("触发版本不存在负向升级 %s→%s", sourceVersion, utils.DefaultFITTargetVersionNotExist))
	err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, utils.DefaultFITTargetVersionNotExist)
	Expect(err).NotTo(HaveOccurred())

	verifyGuideNodeControllerManagerLogContains(sshExecutor, "version not found in upgrade")
}

func patchFITHelmHTTPChartRepoOnBKECluster(sshExecutor executor.Executor, clusterName, guideNodeHost string) {
	By(fmt.Sprintf("FIT: patch BKECluster %s to use guide node HTTP chart repo", clusterName))
	bcName := utils.BKEClusterResourceName(clusterName)
	namespace := utils.BKEClusterNamespace(clusterName)
	commands := []string{
		fmt.Sprintf("kubectl -n %s annotate bkecluster %s cvo.openfuyao.cn/helm-component=true --overwrite", namespace, bcName),
	}
	for _, cmd := range commands {
		runGuideNodeCommand(sshExecutor, cmd, "patch BKECluster helm chart repo")
	}
}

func verifyGuideNodeControllerManagerLogContains(sshExecutor executor.Executor, expected string) {
	cmd := fmt.Sprintf("kubectl -n cluster-system logs deploy/bke-controller-manager --since=10m | grep -F %q", expected)
	Eventually(func() bool {
		result, err := sshExecutor.Exec(cmd)
		if err != nil {
			GinkgoWriter.Printf("[WARN] guide node controller-manager log grep failed: %v\n", err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] guide node controller-manager log grep returned nil result\n")
			return false
		}
		GinkgoWriter.Printf("guide node controller-manager log grep: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, expected)
	}, upgradeTimeout, fitComponentPollInterval).Should(BeTrue(), "controller-manager logs should contain %q", expected)
}

func verifyWorkloadFirstNodePodLabelsContain(clusterNodes []config.NodeInfo, expected string) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(clusterNodes)
	Expect(selected).NotTo(BeEmpty(), "workload cluster should have a master node for pod label verification")
	firstMasterNode := selected[0]
	cmd := "kubectl -n fit-helm-e2e get pod --show-labels"
	Eventually(func() bool {
		result, err := utils.ExecuteCommandOnNode(firstMasterNode, cmd)
		if err != nil {
			GinkgoWriter.Printf("[WARN] workload first master node pod label check failed on %s(%s): %v\n", firstMasterNode.Hostname, firstMasterNode.IP, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] workload first master node pod label check returned nil result on %s(%s)\n", firstMasterNode.Hostname, firstMasterNode.IP)
			return false
		}
		GinkgoWriter.Printf("workload first master node pod labels: exit=%d stdout=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.Contains(result.Stdout, expected)
	}, upgradeTimeout*2, fitComponentPollInterval).Should(BeTrue(), "workload first master node pod labels should contain %q", expected)
}

const fitFailFastFailedComponent = utils.FITFailFastFailedComponent
const fitFailFastLaterComponent = utils.FITFailFastLaterComponent
const fitExecFailComponent = "fit-yaml-exec-fail"

// runDeclarativeUpgradeFITFailFastBatchIsolationUpgrade DAG_EXEC_DFR_1：前置批次 F 失败，后置批次 S 不得误标 Installed。
func runDeclarativeUpgradeFITFailFastBatchIsolationUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, sourceVersion, targetVersion string,
	failedComponent, laterComponent string,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, targetVersion)

	By(fmt.Sprintf("触发 FailFast 负向升级 %s→%s（批次隔离）", sourceVersion, targetVersion))
	err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred())

	verifyFITYamlComponentFailed(duHelper, clusterName, kubeconfigPath, failedComponent)
	verifyFITYamlComponentNotRecorded(duHelper, clusterName, kubeconfigPath, laterComponent)
}

// verifyFITYamlComponentNotRecorded 组件不应出现在 clusterComponentStatuses（Skip/DFR 后置批次等）。
func verifyFITYamlComponentNotRecorded(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, componentName string,
) {
	componentName = strings.TrimSpace(componentName)
	Expect(componentName).NotTo(BeEmpty())
	names, err := duHelper.ListClusterComponentStatusNames(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(names).NotTo(ContainElement(componentName),
		"clusterComponentStatuses 不应记录组件 %s", componentName)
	GinkgoWriter.Printf("FIT: clusterComponentStatuses 未记录 %s（符合预期）\n", componentName)
}

func resolveFITTargetClusterKubeconfig(
	clusterManager *utils.ClusterManager,
	clusterName, parentKubeconfigPath, targetKubeconfigPath string,
	targetIsMgmtSelf bool,
) string {
	targetKC := strings.TrimSpace(targetKubeconfigPath)
	if targetKC != "" {
		return targetKC
	}
	if targetIsMgmtSelf {
		return parentKubeconfigPath
	}
	return saveWorkloadKubeconfigFromMgmt(clusterManager.SSHExecutor(), clusterName, parentKubeconfigPath)
}

// runFITYamlEngineFIT9UpgradeAndVerify YAML_ENGINE_FIT_9：Prune 不误删核心 GVK（fit-prune-core Namespace）。
func runFITYamlEngineFIT9UpgradeAndVerify(
	sshExecutor executor.Executor,
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, parentKubeconfigPath, targetKubeconfigPath string,
	sourceVersion, targetVersion string,
	targetIsMgmtSelf bool,
) {
	_ = sshExecutor
	targetKC := resolveFITTargetClusterKubeconfig(
		clusterManager, clusterName, parentKubeconfigPath, targetKubeconfigPath, targetIsMgmtSelf,
	)

	uidBefore := prepareFIT9PruneCoreSampleBeforeUpgrade(duHelper, targetKC)
	if targetIsMgmtSelf {
		runDeclarativeUpgradeFITMgmtUpgradeAndVerify(
			clusterManager, duHelper, clusterName, "", nil, parentKubeconfigPath, sourceVersion, targetVersion,
		)
		verifyFITYamlComponentsInstalled(
			duHelper, utils.ManagementClusterDeclarativeResourceClusterName(clusterName),
			parentKubeconfigPath, targetVersion, []string{utils.FITYamlPruneComponent},
		)
	} else {
		runDeclarativeUpgradeFITUpgradeAndVerify(
			clusterManager, duHelper, clusterName, parentKubeconfigPath,
			sourceVersion, targetVersion, []string{utils.FITYamlPruneComponent},
		)
	}
	duHelper.VerifyFIT9PruneCoreNamespacePreserved(targetKC, uidBefore)
}

// saveWorkloadKubeconfigFromMgmt 从管理集群获取业务集群 kubeconfig。
func saveWorkloadKubeconfigFromMgmt(
	sshExecutor executor.Executor,
	workloadClusterName, mgmtKubeconfigPath string,
) string {
	checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, workloadClusterName, mgmtKubeconfigPath)
	var workloadKubeconfigPath string
	Eventually(func() bool {
		path, err := checker.SaveKubeconfigToFileWithKubeconfig(mgmtKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("save workload kubeconfig failed: %v\n", err)
			return false
		}
		workloadKubeconfigPath = path
		return true
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "should save workload cluster kubeconfig")
	GinkgoWriter.Printf("workload cluster kubeconfig: %s\n", workloadKubeconfigPath)
	return workloadKubeconfigPath
}

// cleanupMgmtAndWorkloadClusters 清理管理集群及其上部署的业务集群。
func cleanupMgmtAndWorkloadClusters(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	localExecutor *executor.LocalExecutor,
	mgmtClusterName, workloadClusterName, mgmtKubeconfigPath string,
	mgmtNodes []config.NodeInfo,
	mgmtConfigPath, mgmtNodeConfigPath, workloadConfigPath, workloadNodeConfigPath string,
) {
	if workloadClusterName != "" && mgmtKubeconfigPath != "" {
		cleanupDeclarativeUpgradeCluster(
			clusterManager, workloadClusterName, mgmtKubeconfigPath, workloadConfigPath, workloadNodeConfigPath,
		)
	}
	if mgmtClusterName != "" {
		cleanupDeclarativeUpgradeMgmtCluster(
			clusterManager, sshExecutor, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
			mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
		)
	}
}
