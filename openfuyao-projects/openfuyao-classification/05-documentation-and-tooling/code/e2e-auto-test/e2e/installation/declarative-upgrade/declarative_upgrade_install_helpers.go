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

const (
	installTimeout        = 10 * time.Minute
	mgmtInstallTimeout    = 15 * time.Minute
	upgradeTimeout        = 15 * time.Minute
	uninstallTimeout      = 15 * time.Minute
	pollInterval          = 30 * time.Second
	installationItTimeout = 30 * time.Minute

	// declarativeUpgradeMgmtKubeconfigPath 管理集群与引导节点同机共部署时，引导节点默认 kubeconfig 即管理集群 API（与 concurrent 模块一致）。
	declarativeUpgradeMgmtKubeconfigPath = ""
)

// runDeclarativeUpgradeInstallAndVerify 在引导集群安装业务集群并执行 DU-IT-001~004 校验。
func runDeclarativeUpgradeInstallAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName, kubeconfigPath string,
) (configPath, nodeConfigPath string) {
	By(fmt.Sprintf("生成集群配置文件（openFuyaoVersion=%s，不含 cluster-api addon）", utils.DeclarativeUpgradeOpenFuyaoVersion))
	var err error
	configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
	err = duHelper.PatchClusterConfigEtcdVersion(configPath)
	Expect(err).NotTo(HaveOccurred(), "应该成功设置 etcdVersion")
	GinkgoWriter.Printf("配置文件: cluster=%s, nodes=%s\n", configPath, nodeConfigPath)

	By("apply BKENode 与 BKECluster（等价 kubectl apply -f bc.yaml -f bn.yaml）")
	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "apply BC/BN 应该成功")

	By("等待 BKECluster 进入稳定 Ready 状态")
	Eventually(func() bool {
		phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		GinkgoWriter.Printf("BC 状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, installTimeout, pollInterval).Should(BeTrue(), "BKECluster Reconcile 应成功完成安装并达 Ready")

	verifyDeclarativeUpgradeResources(duHelper, clusterName, kubeconfigPath, utils.DeclarativeUpgradeOpenFuyaoVersion)
	return configPath, nodeConfigPath
}

// runDeclarativeUpgradeMgmtInstallAndVerify 在引导节点部署 1M1W 管理集群，并在管理集群 API 上校验 CV/RI/UP。
func runDeclarativeUpgradeMgmtInstallAndVerify(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName string,
) (configPath, nodeConfigPath, mgmtKubeconfigPath string) {
	By(fmt.Sprintf("生成管理集群配置文件（BC 与 cluster-api addon 的 openFuyaoVersion=%s）", utils.DeclarativeUpgradeOpenFuyaoVersion))
	var err error
	configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")
	err = duHelper.PatchClusterConfigEtcdVersion(configPath)
	Expect(err).NotTo(HaveOccurred(), "应该成功设置 etcdVersion")
	GinkgoWriter.Printf("管理集群配置: cluster=%s, nodes=%s\n", configPath, nodeConfigPath)

	waitDeclarativeUpgradeMgmtClusterCreatePreflight(clusterManager, clusterName, clusterConfig.Nodes)

	By("在引导节点执行管理集群 bke cluster create")
	err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
	Expect(err).NotTo(HaveOccurred(), "管理集群创建命令应该成功执行")

	// 创建命令已发出即注册清理：纳管等待 Fail 时 AfterAll 可能不跑，且 It 内「install 返回后再 defer」也不会注册。
	mgmtNodesForCleanup := append([]config.NodeInfo(nil), clusterConfig.Nodes...)
	cfgPath, nodePath := configPath, nodeConfigPath
	mgmtKubeconfigForCleanup := new(string)
	DeferCleanup(func() {
		By(fmt.Sprintf("DeferCleanup: 清理管理集群 %s", clusterName))
		cleanupDeclarativeUpgradeMgmtCluster(
			clusterManager, sshExecutor, executor.NewLocalExecutor(10*time.Minute),
			clusterName, *mgmtKubeconfigForCleanup, mgmtNodesForCleanup, cfgPath, nodePath,
		)
	})

	mgmtKubeconfigPath = waitDeclarativeUpgradeMgmtClusterFullyHealthy(clusterManager, sshExecutor, clusterName)
	*mgmtKubeconfigForCleanup = mgmtKubeconfigPath
	ensureMgmtSelfBCAnnotationsFromInstallConfig(duHelper, clusterConfig, mgmtKubeconfigPath)
	verifyDeclarativeUpgradeResources(
		duHelper,
		utils.ManagementClusterDeclarativeResourceClusterName(clusterName),
		mgmtKubeconfigPath,
		utils.DeclarativeUpgradeOpenFuyaoVersion,
	)
	verifyDeclarativeUpgradeUpgradePath(duHelper, mgmtKubeconfigPath)
	return configPath, nodeConfigPath, mgmtKubeconfigPath
}

func waitDeclarativeUpgradeMgmtClusterCreatePreflight(
	clusterManager *utils.ClusterManager,
	clusterName string,
	mgmtNodes []config.NodeInfo,
) {
	By("创建管理集群前检查引导面残留和管理节点状态")
	if isDeclarativeUpgradeMgmtClusterCreatePreflightReady(clusterManager, clusterName, mgmtNodes) {
		return
	}

	GinkgoWriter.Printf("管理集群创建前检查未通过，先对候选管理节点执行一次 bke reset --all --mount\n")
	if err := clusterManager.ForceResetManagementCluster(mgmtNodes, executor.NewLocalExecutor(10*time.Minute)); err != nil {
		GinkgoWriter.Printf("[WARN] 创建前管理节点 reset 返回错误: %v\n", err)
	}

	Eventually(func() bool {
		return isDeclarativeUpgradeMgmtClusterCreatePreflightReady(clusterManager, clusterName, mgmtNodes)
	}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "创建管理集群前引导面残留应清理且管理节点应可用于新建")
}

func isDeclarativeUpgradeMgmtClusterCreatePreflightReady(
	clusterManager *utils.ClusterManager,
	clusterName string,
	mgmtNodes []config.NodeInfo,
) bool {
	if strings.TrimSpace(clusterName) == "" {
		GinkgoWriter.Printf("[preflight] clusterName is empty\n")
		return false
	}
	if len(mgmtNodes) == 0 {
		GinkgoWriter.Printf("[preflight] management node list is empty\n")
		return false
	}

	namespace := utils.BKEClusterNamespace(clusterName)
	if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
		GinkgoWriter.Printf("[preflight] bootstrap BKECluster %s still exists\n", utils.BKEClusterResourceName(clusterName))
		return false
	}
	if declarativeUpgradeBootstrapNamespaceExists(clusterManager, namespace) {
		GinkgoWriter.Printf("[preflight] bootstrap namespace %s still exists\n", namespace)
		return false
	}

	for _, node := range mgmtNodes {
		if !isDeclarativeUpgradeMgmtNodeReadyForCreate(node) {
			return false
		}
	}
	return true
}

func declarativeUpgradeBootstrapNamespaceExists(clusterManager *utils.ClusterManager, namespace string) bool {
	cmd := fmt.Sprintf("kubectl get namespace %s --no-headers 2>/dev/null", namespace)
	result, err := clusterManager.SSHExecutor().Exec(cmd)
	return err == nil && result != nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != ""
}

func isDeclarativeUpgradeMgmtNodeReadyForCreate(node config.NodeInfo) bool {
	cmd := "test ! -d /bke && (test ! -d /etc/kubernetes/manifests || test -z \"$(ls -A /etc/kubernetes/manifests 2>/dev/null)\") && (test ! -d /var/lib/etcd || test -z \"$(ls -A /var/lib/etcd 2>/dev/null)\")"
	result, err := utils.ExecuteCommandOnNode(node, cmd)
	if err != nil {
		GinkgoWriter.Printf("[preflight] management node %s(%s) is not ready: %v\n", node.Hostname, node.IP, err)
		return false
	}
	if result == nil {
		GinkgoWriter.Printf("[preflight] management node %s(%s) readiness check returned nil result\n", node.Hostname, node.IP)
		return false
	}
	if result.ExitCode != 0 {
		GinkgoWriter.Printf("[preflight] management node %s(%s) has install residue: exit=%d stdout=%s stderr=%s\n",
			node.Hostname, node.IP, result.ExitCode, result.Stdout, result.Stderr)
		return false
	}
	return true
}

// waitDeclarativeUpgradeMgmtClusterFullyHealthy 等待引导节点 BC 与管理集群自管 BC 均达到 Healthy/Ready。
// 注意：安装配置里的 BC 注解（如 helm-component）只作用于引导阶段 BC；纳管自身后自管 BC
// 是新建对象，不会继承引导 BC 注解。需升级自管 BC 且依赖注解的场景，在就绪后另行补写（见 ensureMgmtSelfBCAnnotationsFromInstallConfig）。
func waitDeclarativeUpgradeMgmtClusterFullyHealthy(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	clusterName string,
) string {
	By("等待引导节点上管理集群 BC 变为 Healthy 和 Ready")
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
		if err != nil {
			return false
		}
		GinkgoWriter.Printf("[bootstrap] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("管理集群创建失败，状态为 DeployFailed。phase=%s, clusterStatus=%s", phase, clusterStatus))
		}
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "引导节点上管理集群应变为 Ready")

	By("获取管理集群 kubeconfig（从引导节点 Secret 或 ~/.bke 路径读取）")
	checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
	var mgmtKubeconfigPath string
	Eventually(func() bool {
		path, err := checker.SaveKubeconfigToFile()
		if err != nil {
			GinkgoWriter.Printf("获取管理集群 kubeconfig 失败: %v\n", err)
			return false
		}
		mgmtKubeconfigPath = path
		return true
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "应该成功获取管理集群 kubeconfig")
	GinkgoWriter.Printf("管理集群 kubeconfig: %s\n", mgmtKubeconfigPath)

	By("等待管理集群自管 BC 变为 Ready")
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
		if err != nil {
			return false
		}
		GinkgoWriter.Printf("[mgmt-self] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
		if isClusterFailedState(state, clusterStatus) || clusterStatus == "ManageFailed" {
			Fail(fmt.Sprintf("管理集群纳管失败，不再等待 Ready。phase=%s, state=%s, clusterStatus=%s",
				phase, state, clusterStatus))
		}
		return clusterStatus == "Ready"
	}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "管理集群自管 BC 应变为 Ready")

	return mgmtKubeconfigPath
}

// ensureMgmtSelfBCAnnotationsFromInstallConfig 纳管自身后把安装配置中要求的注解补写到自管 BC。
// 引导 BC 上设置的 helm-component 不会自动带到自管 BC。
func ensureMgmtSelfBCAnnotationsFromInstallConfig(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	mgmtKubeconfigPath string,
) {
	if clusterConfig == nil || clusterConfig.Annotations == nil {
		return
	}
	val, ok := clusterConfig.Annotations[utils.HelmComponentAnnotation]
	if !ok || !strings.EqualFold(strings.TrimSpace(val), "true") {
		return
	}
	declClusterName := utils.ManagementClusterDeclarativeResourceClusterName(clusterConfig.ClusterName)
	By("管理集群纳管自身后补写 helm-component 注解到自管 BC（引导 BC 注解不会继承）")
	Eventually(func() bool {
		err := duHelper.PatchBCAnnotation(declClusterName, mgmtKubeconfigPath, utils.HelmComponentAnnotation, "true")
		if err != nil {
			GinkgoWriter.Printf("补写自管 BC helm-component 注解失败: %v\n", err)
			return false
		}
		annot, err := duHelper.GetBCAnnotation(declClusterName, mgmtKubeconfigPath, utils.HelmComponentAnnotation)
		if err != nil {
			GinkgoWriter.Printf("读取自管 BC helm-component 注解失败: %v\n", err)
			return false
		}
		GinkgoWriter.Printf("自管 BC helm-component 注解当前值: %q\n", annot)
		return strings.EqualFold(strings.TrimSpace(annot), "true")
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(),
		"自管 BC 应在 2 分钟内稳定持有 helm-component=true")
}

// verifyDeclarativeUpgradeResources 在指定集群 API 上校验 DU-IT-001~004（CV/RI）。
func verifyDeclarativeUpgradeResources(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, expectedVersion string,
) {
	desiredVersion := expectedVersion
	GinkgoWriter.Printf("校验目标版本 openFuyaoVersion=%s\n", desiredVersion)

	bcName := utils.BKEClusterResourceName(clusterName)

	By("DU-IT-001: 验证 ClusterVersion 已创建（与 BC 同命名空间关联）且 ownerReferences 指向 BKECluster")
	Eventually(func() bool {
		exists, e := duHelper.ClusterVersionExists(clusterName, kubeconfigPath)
		return e == nil && exists
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "应自动创建 ClusterVersion")

	cv, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Printf("ClusterVersion: desired=%s, current=%s, phase=%s, ownerBC=%s\n",
		cv.DesiredVersion, cv.CurrentVersion, cv.Phase, cv.OwnerBCName)
	Expect(cv.Namespace).To(Equal(utils.BKEClusterNamespace(clusterName)),
		"ClusterVersion 应与 BC 同命名空间")
	Expect(cv.DesiredVersion).To(Equal(desiredVersion), "spec.desiredVersion 应与安装目标 openFuyaoVersion 一致")
	Expect(cv.OwnerBCName).To(Equal(bcName), "ClusterVersion ownerReferences 应指向 BKECluster")

	By("DU-IT-002: 验证 CV 从 OCI 拉取 ri 并创建 ReleaseImage CR（spec.version 正确）")
	expectedRIName := utils.ReleaseImageNameForVersion(desiredVersion)
	Eventually(func() string {
		name, e := duHelper.FindReleaseImageByVersion(clusterName, desiredVersion, kubeconfigPath)
		if e != nil {
			names, _ := duHelper.ListReleaseImageNames(clusterName, kubeconfigPath)
			GinkgoWriter.Printf("当前 ReleaseImage 列表: %v\n", names)
			return ""
		}
		return name
	}, 10*time.Minute, 15*time.Second).ShouldNot(BeEmpty(), "CV 应从 OCI 拉取 ri 并创建 ReleaseImage")

	riName, err := duHelper.FindReleaseImageByVersion(clusterName, desiredVersion, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	GinkgoWriter.Printf("ReleaseImage 名称: %s (期望 %s)\n", riName, expectedRIName)
	Expect(riName).To(Equal(expectedRIName), "ReleaseImage 命名应与版本约定一致")

	ri, err := duHelper.GetReleaseImage(clusterName, riName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(ri.Version).To(Equal(desiredVersion), "ReleaseImage spec.version 应与安装版本一致")

	By("DU-IT-003: 验证 ReleaseImage 调谐完成，status.phase=Valid")
	Eventually(func() string {
		riSnap, e := duHelper.GetReleaseImage(clusterName, riName, kubeconfigPath)
		if e != nil {
			return ""
		}
		GinkgoWriter.Printf("ReleaseImage 状态: phase=%s, digest=%s, componentCount=%d\n",
			riSnap.Phase, riSnap.Digest, riSnap.ComponentCount)
		return riSnap.Phase
	}, 10*time.Minute, 15*time.Second).Should(Equal("Valid"), "ReleaseImage 应协调为 Valid")

	ri, err = duHelper.GetReleaseImage(clusterName, riName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(ri.Phase).To(Equal("Valid"))
	Expect(ri.Digest).NotTo(BeEmpty(), "status.digest 不应为空")
	Expect(ri.ComponentCount).To(BeNumerically(">", 0), "status.componentCount 应大于 0")

	By("DU-IT-004: 验证安装收尾由 BC 调谐器写入 ClusterVersion.status")
	Eventually(func() bool {
		cvSnap, e := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if e != nil {
			return false
		}
		GinkgoWriter.Printf("CV 收尾状态: phase=%s, current=%s, desired=%s\n",
			cvSnap.Phase, cvSnap.CurrentVersion, cvSnap.DesiredVersion)
		if cvSnap.CurrentVersion != cvSnap.DesiredVersion {
			return false
		}
		return cvSnap.Phase == "Ready" || cvSnap.Phase == "Installed"
	}, 10*time.Minute, 15*time.Second).Should(BeTrue(),
		"安装完成后 status.currentVersion 应等于 desiredVersion，且 phase 为 Ready/Installed")

	cv, err = duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(cv.CurrentVersion).To(Equal(desiredVersion), "ClusterVersion.status.currentVersion 应与安装目标一致")
	Expect(cv.CurrentVersion).To(Equal(cv.DesiredVersion))
	Expect(cv.Phase).To(Or(Equal("Ready"), Equal("Installed")),
		"ClusterVersion.status.phase 应为 Ready 或 Installed")
}

// runDeclarativeUpgradeWorkloadUpgradeAndVerify 在已安装的 vfit-pre 业务集群上触发升级到 targetVersion 并校验 DU-IT-005~006。
// captureYamlEngineUIDs 仅 Gate 开启的 YamlInstaller 路径需要：比较 SSA/Replace/CreateOnly UID。Gate 关闭走 Legacy，不比较 UID。
func runDeclarativeUpgradeWorkloadUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	sourceVersion, targetVersion string,
	captureYamlEngineUIDs bool,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	if captureYamlEngineUIDs && shouldCaptureFITYamlEngineUIDsForUpgrade(targetVersion) {
		captureFITYamlEngineFIT3And4UIDsBeforeCurUpgrade(
			clusterManager, duHelper, clusterName, kubeconfigPath, false,
		)
	}
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, targetVersion)

	cvBefore, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "升级前应能读取 ClusterVersion")
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "升级前 currentVersion 应为源版本")
	GinkgoWriter.Printf("升级前 CV: desired=%s, current=%s, phase=%s\n",
		cvBefore.DesiredVersion, cvBefore.CurrentVersion, cvBefore.Phase)

	By("DU-IT-005: Patch ClusterVersion.spec.desiredVersion 触发声明式升级")
	err = duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion 应成功")

	Eventually(func() string {
		cv, e := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if e != nil {
			return ""
		}
		return cv.DesiredVersion
	}, 5*time.Minute, 10*time.Second).Should(Equal(targetVersion), "CV spec.desiredVersion 应更新为目标版本")

	By("DU-IT-005: 等待升级 DAG 执行并完成版本切换")
	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("获取 BC 状态失败: %v\n", e)
			return false
		}
		upgradeReady, _ := duHelper.GetBCAnnotation(clusterName, kubeconfigPath, utils.UpgradeReadyAnnotation)
		cv, cvErr := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if cvErr == nil {
			GinkgoWriter.Printf("升级中 BC: phase=%s, state=%s, clusterStatus=%s, upgrade-ready=%q, CV current=%s desired=%s phase=%s\n",
				phase, state, clusterStatus, upgradeReady, cv.CurrentVersion, cv.DesiredVersion, cv.Phase)
		}
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("升级失败，BC state=%s, phase=%s, clusterStatus=%s", state, phase, clusterStatus))
		}
		if cvErr != nil {
			return false
		}
		return cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed") &&
			clusterStatus == "Ready"
	}, upgradeTimeout, pollInterval).Should(BeTrue(), "升级应完成且 BC Ready、CV currentVersion 达到目标版本")

	verifyDeclarativeUpgradeUpgradeComplete(
		duHelper, clusterManager, clusterName, kubeconfigPath, targetVersion,
		[][]string{{sourceVersion, targetVersion}},
	)
	if captureYamlEngineUIDs && shouldCaptureFITYamlEngineUIDsForUpgrade(targetVersion) {
		verifyFITYamlEngineFIT3And4UIDsAfterCurUpgrade(duHelper, clusterName)
	}
}

// runDeclarativeUpgradeWorkloadMultiHopUpgradeAndVerify 一次 patch desiredVersion 为最终版本，
// 由调谐器按 UpgradePath 分跳（如 vfit-pre→vfit-cur→vfit-post）完成升级并校验 DU-IT-005~006。
type declarativeUpgradeWorkloadMultiHopHelmOptions struct {
	exec          executor.Executor
	guideNodeHost string
	workloadNodes []config.NodeInfo
}

func runDeclarativeUpgradeWorkloadMultiHopUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	sourceVersion, intermediateVersion, targetVersion string,
	helmOptions ...declarativeUpgradeWorkloadMultiHopHelmOptions,
) {
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, clusterName, kubeconfigPath, false)
	verifyDeclarativeUpgradeMultiHopUpgradePathForUpgrade(
		duHelper, kubeconfigPath, sourceVersion, intermediateVersion, targetVersion)

	var helmOption *declarativeUpgradeWorkloadMultiHopHelmOptions
	if len(helmOptions) > 0 {
		option := helmOptions[0]
		if option.exec != nil && strings.TrimSpace(option.guideNodeHost) != "" && len(option.workloadNodes) > 0 {
			helmOption = &option
		} else {
			GinkgoWriter.Printf("[WARN] skip FIT helm e2e workload multi-hop setup: incomplete options\n")
		}
	}
	if helmOption != nil {
		prepareAndPublishFITHelmE2EHTTPCharts(helmOption.exec, helmOption.guideNodeHost, "0.1.1", "0.1.2", "0.1.3")
		prepareAndPatchFITHelmE2ETLSChartRepoOnGuideNode(helmOption.exec, helmOption.guideNodeHost, clusterName)
	}

	cvBefore, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "升级前应能读取 ClusterVersion")
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "升级前 currentVersion 应为源版本")
	GinkgoWriter.Printf("两跳升级前 CV: desired=%s, current=%s, phase=%s\n",
		cvBefore.DesiredVersion, cvBefore.CurrentVersion, cvBefore.Phase)

	By(fmt.Sprintf("DU-IT-005: 一次 Patch desiredVersion=%s，由调谐器分跳完成 %s→%s→%s",
		targetVersion, sourceVersion, intermediateVersion, targetVersion))
	err = duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion 应成功")

	Eventually(func() string {
		cv, e := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if e != nil {
			return ""
		}
		return cv.DesiredVersion
	}, 5*time.Minute, 10*time.Second).Should(Equal(targetVersion), "CV spec.desiredVersion 应直接更新为最终目标版本")

	By("DU-IT-005: 等待调谐器按 UpgradePath 分跳执行并完成版本切换")
	sawIntermediate := false
	refreshedHelmRepoAfterIntermediate := false
	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("获取 BC 状态失败: %v\n", e)
			return false
		}
		upgradeReady, _ := duHelper.GetBCAnnotation(clusterName, kubeconfigPath, utils.UpgradeReadyAnnotation)
		cv, cvErr := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if cvErr == nil {
			if cv.CurrentVersion == intermediateVersion {
				sawIntermediate = true
				if helmOption != nil && !refreshedHelmRepoAfterIntermediate {
					By(fmt.Sprintf("FIT: first hop reached %s, refresh guide node TLS chart repo before next hop", intermediateVersion))
					patchFITHelmE2ETLSChartRepoOnGuideNode(helmOption.exec, helmOption.guideNodeHost, clusterName)
					refreshedHelmRepoAfterIntermediate = true
				}
			}
			GinkgoWriter.Printf("两跳升级中 BC: phase=%s, state=%s, clusterStatus=%s, upgrade-ready=%q, CV current=%s desired=%s phase=%s\n",
				phase, state, clusterStatus, upgradeReady, cv.CurrentVersion, cv.DesiredVersion, cv.Phase)
		}
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("两跳升级失败，BC state=%s, phase=%s, clusterStatus=%s", state, phase, clusterStatus))
		}
		if cvErr != nil {
			return false
		}
		return cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed") &&
			clusterStatus == "Ready"
	}, upgradeTimeout*2, pollInterval).Should(BeTrue(),
		"两跳升级应完成且 BC Ready、CV currentVersion 达到最终目标版本")

	if sawIntermediate {
		GinkgoWriter.Printf("观察到调谐器经中间版本 %s 完成第一跳\n", intermediateVersion)
	} else {
		GinkgoWriter.Printf("未在轮询中捕获中间版本 %s（可能已快速跃迁）\n", intermediateVersion)
	}

	verifyDeclarativeUpgradeUpgradeComplete(
		duHelper, clusterManager, clusterName, kubeconfigPath, targetVersion,
		[][]string{{sourceVersion, intermediateVersion}, {intermediateVersion, targetVersion}},
	)
	if helmOption != nil {
		verifyFITHelmE2EPodLabelVersionOnWorkloadMasterNode(helmOption.workloadNodes, "0.1.2")
	}
}

func prepareAndPublishFITHelmE2ECharts(exec executor.Executor, guideNodeHost string, mgmtNodes []config.NodeInfo, bootstrapClusterName string, versions ...string) {
	prepareAndPublishFITHelmE2EHTTPCharts(exec, guideNodeHost, versions...)
	prepareAndPatchFITHelmE2ETLSChartRepo(exec, guideNodeHost, mgmtNodes, bootstrapClusterName)
}

func prepareAndPublishFITHelmE2EHTTPCharts(exec executor.Executor, guideNodeHost string, versions ...string) {
	if len(versions) == 0 {
		versions = []string{"0.1.1", "0.1.2", "0.1.3"}
	}

	prepareCommands := []string{
		"chmod -R 777 /bke/mount/charts",
		"mkdir -p /opt/fit-helm-e2e/templates",
		"test -f /opt/fit-helm-e2e/Chart.yaml || printf 'apiVersion: v2\nname: fit-helm-e2e\ndescription: FIT helm e2e chart\ntype: application\nversion: 0.1.0\nappVersion: 0.1.0\n' > /opt/fit-helm-e2e/Chart.yaml",
		"test -f /opt/fit-helm-e2e/templates/configmap.yaml || printf 'apiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: {{ .Chart.Name }}\ndata:\n  appVersion: {{ .Chart.AppVersion | quote }}\n' > /opt/fit-helm-e2e/templates/configmap.yaml",
	}
	for _, cmd := range prepareCommands {
		runGuideNodeCommand(exec, cmd, "prepare fit-helm-e2e chart")
	}

	chartUploadURL := fmt.Sprintf("http://%s:38080/api/charts", guideNodeHost)
	for _, version := range versions {
		By(fmt.Sprintf("Package and publish fit-helm-e2e chart version %s", version))
		chartArchive := fmt.Sprintf("fit-helm-e2e-%s.tgz", version)
		chartVersionURL := fmt.Sprintf("%s/fit-helm-e2e/%s", chartUploadURL, version)
		commands := []string{
			fmt.Sprintf("cd /opt && helm package ./fit-helm-e2e --version %s --app-version %s", version, version),
			fmt.Sprintf("cd /opt && if curl -fsS %s >/dev/null; then echo fit-helm-e2e-%s already exists, skip upload; else curl --data-binary \"@%s\" %s; fi", chartVersionURL, version, chartArchive, chartUploadURL),
			"helm repo update",
			fmt.Sprintf("cd /opt && helm pull internal/fit-helm-e2e --version %s", version),
		}
		for _, cmd := range commands {
			runGuideNodeCommand(exec, cmd, "publish fit-helm-e2e chart")
		}
	}
}
func prepareAndPatchFITHelmE2ETLSChartRepo(exec executor.Executor, guideNodeHost string, mgmtNodes []config.NodeInfo, bootstrapClusterName string) {
	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(mgmtNodes)
	Expect(selected).NotTo(BeEmpty(), "management cluster should have a master node for chart repo TLS setup")
	masterNode := selected[0]

	if fitHelmE2ETLSChartRepoAlreadyReady(exec, masterNode, guideNodeHost, bootstrapClusterName) {
		GinkgoWriter.Printf("chart repo TLS artifacts already ready, skip TLS setup\n")
		return
	}

	By("Prepare chart repo TLS proxy on guide node")
	nginxConfig := fmt.Sprintf(`cat > /opt/chartrepo-tls/default.conf <<'EOF'
server {
    listen 443 ssl;

    ssl_certificate     /etc/nginx/certs/tls.crt;
    ssl_certificate_key /etc/nginx/certs/tls.key;

    location / {
        proxy_pass http://%s:38080;
        proxy_set_header Host $host;
    }
}
EOF`, guideNodeHost)
	guideCommands := []string{
		"mkdir -p /opt/chartrepo-tls",
		fmt.Sprintf(`openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout /opt/chartrepo-tls/tls.key -out /opt/chartrepo-tls/tls.crt -subj "/CN=%s" -addext "subjectAltName=IP:%s"`, guideNodeHost, guideNodeHost),
		nginxConfig,
		`nerdctl rm -f chartmuseum-tls-proxy >/dev/null 2>&1 || true`,
		`nerdctl run -d --name chartmuseum-tls-proxy --restart=always -p 38443:443 -v /opt/chartrepo-tls/default.conf:/etc/nginx/conf.d/default.conf:ro -v /opt/chartrepo-tls:/etc/nginx/certs:ro cr.internal.openfuyao.cn/pass/nginx:1.23.0-alpine`,
	}
	for _, cmd := range guideCommands {
		runGuideNodeCommand(exec, cmd, "prepare chart repo TLS proxy")
	}
	waitFITHelmE2ETLSChartRepoReachable(exec, guideNodeHost)

	copyFITHelmE2ETLSDirToMasterNode(exec, masterNode)

	patchFITHelmE2ETLSChartRepoOnMasterNode(masterNode, guideNodeHost, bootstrapClusterName)
}

func waitFITHelmE2ETLSChartRepoReachable(exec executor.Executor, guideNodeHost string) {
	cmd := fmt.Sprintf("curl -sS --cacert /opt/chartrepo-tls/tls.crt -o /dev/null -w '%%{http_code}' https://%s:38443/index.yaml", guideNodeHost)
	Eventually(func() bool {
		result, err := exec.Exec(cmd)
		if err != nil {
			GinkgoWriter.Printf("[WARN] chart repo TLS proxy is not reachable yet: cmd=%s err=%v\n", cmd, err)
			return false
		}
		if result == nil {
			GinkgoWriter.Printf("[WARN] chart repo TLS proxy check returned nil result: cmd=%s\n", cmd)
			return false
		}
		GinkgoWriter.Printf("chart repo TLS proxy check: exit=%d httpCode=%s stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
		return result.ExitCode == 0 && strings.TrimSpace(result.Stdout) == "200"
	}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "chart repo TLS proxy should return 200 for index.yaml")
}
func patchFITHelmE2ETLSChartRepoOnMasterNode(masterNode config.NodeInfo, guideNodeHost, bootstrapClusterName string) {
	selfNS := utils.ManagementClusterSelfBCNamespace(bootstrapClusterName)
	selfName := utils.ManagementClusterSelfBCName(bootstrapClusterName)
	masterCommands := []string{
		fmt.Sprintf("kubectl -n %s annotate bkecluster %s cvo.openfuyao.cn/helm-component=true --overwrite", selfNS, selfName),
		fmt.Sprintf(`kubectl -n %s create secret generic chartrepo-tls --from-file=ca.crt=/opt/chartrepo-tls/tls.crt --from-file=cert.crt=/opt/chartrepo-tls/tls.crt --from-file=key.key=/opt/chartrepo-tls/tls.key --dry-run=client -o yaml | kubectl apply -f -`, selfNS),
		fmt.Sprintf(`kubectl -n %s patch bkecluster %s --type merge -p '{"spec":{"clusterConfig":{"cluster":{"chartRepo":{"domain":"","ip":"%s","port":"38443","prefix":"","tlsSecretRef":{"name":"chartrepo-tls","caKey":"ca.crt","certKey":"cert.crt","keyKey":"key.key"}}}}}}'`, selfNS, selfName, guideNodeHost),
	}
	for _, cmd := range masterCommands {
		result, err := utils.ExecuteCommandOnNode(masterNode, cmd)
		Expect(err).NotTo(HaveOccurred(), "chart repo TLS command should execute on master node: %s", cmd)
		Expect(result).NotTo(BeNil(), "chart repo TLS command should return result on master node: %s", cmd)
		Expect(result.ExitCode).To(Equal(0), "chart repo TLS command should succeed on master node: %s\nstdout: %s\nstderr: %s", cmd, result.Stdout, result.Stderr)
	}
}
func prepareAndPatchFITHelmE2ETLSChartRepoOnGuideNode(exec executor.Executor, guideNodeHost, clusterName string) {
	if fitHelmE2EGuideTLSChartRepoAlreadyReady(exec, guideNodeHost, clusterName) {
		GinkgoWriter.Printf("chart repo TLS artifacts already ready on guide node for cluster %s, skip TLS setup\n", clusterName)
		return
	}

	By("Prepare chart repo TLS proxy on guide node")
	nginxConfig := fmt.Sprintf(`cat > /opt/chartrepo-tls/default.conf <<'EOF'
server {
    listen 443 ssl;

    ssl_certificate     /etc/nginx/certs/tls.crt;
    ssl_certificate_key /etc/nginx/certs/tls.key;

    location / {
        proxy_pass http://%s:38080;
        proxy_set_header Host $host;
    }
}
EOF`, guideNodeHost)
	guideCommands := []string{
		"mkdir -p /opt/chartrepo-tls",
		fmt.Sprintf(`openssl req -x509 -newkey rsa:2048 -nodes -days 365 -keyout /opt/chartrepo-tls/tls.key -out /opt/chartrepo-tls/tls.crt -subj "/CN=%s" -addext "subjectAltName=IP:%s"`, guideNodeHost, guideNodeHost),
		nginxConfig,
		`nerdctl rm -f chartmuseum-tls-proxy >/dev/null 2>&1 || true`,
		`nerdctl run -d --name chartmuseum-tls-proxy --restart=always -p 38443:443 -v /opt/chartrepo-tls/default.conf:/etc/nginx/conf.d/default.conf:ro -v /opt/chartrepo-tls:/etc/nginx/certs:ro cr.internal.openfuyao.cn/pass/nginx:1.23.0-alpine`,
	}
	for _, cmd := range guideCommands {
		runGuideNodeCommand(exec, cmd, "prepare chart repo TLS proxy")
	}
	waitFITHelmE2ETLSChartRepoReachable(exec, guideNodeHost)
	patchFITHelmE2ETLSChartRepoOnGuideNode(exec, guideNodeHost, clusterName)
}

func patchFITHelmE2ETLSChartRepoOnGuideNode(exec executor.Executor, guideNodeHost, clusterName string) {
	bcName := utils.BKEClusterResourceName(clusterName)
	namespace := utils.BKEClusterNamespace(clusterName)
	guideCommands := []string{
		fmt.Sprintf("kubectl -n %s annotate bkecluster %s cvo.openfuyao.cn/helm-component=true --overwrite", namespace, bcName),
		fmt.Sprintf(`kubectl -n %s create secret generic chartrepo-tls --from-file=ca.crt=/opt/chartrepo-tls/tls.crt --from-file=cert.crt=/opt/chartrepo-tls/tls.crt --from-file=key.key=/opt/chartrepo-tls/tls.key --dry-run=client -o yaml | kubectl apply -f -`, namespace),
		fmt.Sprintf(`kubectl -n %s patch bkecluster %s --type merge -p '{"spec":{"clusterConfig":{"cluster":{"chartRepo":{"domain":"","ip":"%s","port":"38443","prefix":"","tlsSecretRef":{"name":"chartrepo-tls","caKey":"ca.crt","certKey":"cert.crt","keyKey":"key.key"}}}}}}'`, namespace, bcName, guideNodeHost),
	}
	for _, cmd := range guideCommands {
		runGuideNodeCommand(exec, cmd, "patch guide node BKECluster chart repo TLS")
	}
}

func fitHelmE2EGuideTLSChartRepoAlreadyReady(exec executor.Executor, guideNodeHost, clusterName string) bool {
	guideCheckCmd := fmt.Sprintf("test -s /opt/chartrepo-tls/tls.key -a -s /opt/chartrepo-tls/tls.crt -a -s /opt/chartrepo-tls/default.conf && nerdctl ps --format '{{.Names}}' | grep -qx chartmuseum-tls-proxy && curl --cacert /opt/chartrepo-tls/tls.crt -I https://%s:38443/index.yaml | grep -q '200'", guideNodeHost)
	if result, err := exec.Exec(guideCheckCmd); err != nil || result == nil || result.ExitCode != 0 {
		return false
	}

	bcName := utils.BKEClusterResourceName(clusterName)
	namespace := utils.BKEClusterNamespace(clusterName)
	clusterCheckCmd := fmt.Sprintf(`kubectl -n %s get secret chartrepo-tls -o jsonpath='{.data.ca\.crt}' | base64 -d | cmp -s - /opt/chartrepo-tls/tls.crt && kubectl -n %s get secret chartrepo-tls -o jsonpath='{.data.key\.key}' | base64 -d | grep -q 'BEGIN .*PRIVATE KEY' && kubectl -n %s get bkecluster %s -o jsonpath='{.spec.clusterConfig.cluster.chartRepo.ip}{":"}{.spec.clusterConfig.cluster.chartRepo.port}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.name}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.caKey}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.certKey}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.keyKey}' | grep -q '^%s:38443:chartrepo-tls:ca.crt:cert.crt:key.key$'`, namespace, namespace, namespace, bcName, guideNodeHost)
	result, err := exec.Exec(clusterCheckCmd)
	return err == nil && result != nil && result.ExitCode == 0
}
func copyFITHelmE2ETLSDirToMasterNode(exec executor.Executor, masterNode config.NodeInfo) {
	files := []string{"tls.crt", "tls.key", "default.conf"}
	mkdirResult, mkdirErr := utils.ExecuteCommandOnNode(masterNode, "mkdir -p /opt/chartrepo-tls")
	Expect(mkdirErr).NotTo(HaveOccurred(), "create /opt/chartrepo-tls on master node should execute")
	Expect(mkdirResult).NotTo(BeNil(), "create /opt/chartrepo-tls on master node should return result")
	Expect(mkdirResult.ExitCode).To(Equal(0), "create /opt/chartrepo-tls on master node should succeed: stdout=%s stderr=%s", mkdirResult.Stdout, mkdirResult.Stderr)

	for _, fileName := range files {
		readCmd := fmt.Sprintf("base64 -w0 /opt/chartrepo-tls/%s", fileName)
		readResult, readErr := exec.Exec(readCmd)
		Expect(readErr).NotTo(HaveOccurred(), "read chart repo TLS file from guide node should execute: %s", fileName)
		Expect(readResult).NotTo(BeNil(), "read chart repo TLS file from guide node should return result: %s", fileName)
		Expect(readResult.ExitCode).To(Equal(0), "read chart repo TLS file from guide node should succeed: %s\nstdout: %s\nstderr: %s", fileName, readResult.Stdout, readResult.Stderr)

		writeCmd := fmt.Sprintf("printf '%s' | base64 -d > /opt/chartrepo-tls/%s", strings.TrimSpace(readResult.Stdout), fileName)
		writeResult, writeErr := utils.ExecuteCommandOnNode(masterNode, writeCmd)
		Expect(writeErr).NotTo(HaveOccurred(), "write chart repo TLS file to master node should execute: %s", fileName)
		Expect(writeResult).NotTo(BeNil(), "write chart repo TLS file to master node should return result: %s", fileName)
		Expect(writeResult.ExitCode).To(Equal(0), "write chart repo TLS file to master node should succeed: %s\nstdout: %s\nstderr: %s", fileName, writeResult.Stdout, writeResult.Stderr)
	}
}
func fitHelmE2ETLSChartRepoAlreadyReady(exec executor.Executor, masterNode config.NodeInfo, guideNodeHost, bootstrapClusterName string) bool {
	guideCheckCmd := fmt.Sprintf("test -s /opt/chartrepo-tls/tls.key -a -s /opt/chartrepo-tls/tls.crt -a -s /opt/chartrepo-tls/default.conf && nerdctl ps --format '{{.Names}}' | grep -qx chartmuseum-tls-proxy && curl --cacert /opt/chartrepo-tls/tls.crt -I https://%s:38443/index.yaml | grep -q '200'", guideNodeHost)
	if result, err := exec.Exec(guideCheckCmd); err != nil || result == nil || result.ExitCode != 0 {
		return false
	}

	selfNS := utils.ManagementClusterSelfBCNamespace(bootstrapClusterName)
	selfName := utils.ManagementClusterSelfBCName(bootstrapClusterName)
	masterCheckCmd := fmt.Sprintf(`test -s /opt/chartrepo-tls/tls.key -a -s /opt/chartrepo-tls/tls.crt && kubectl -n %s get secret chartrepo-tls -o jsonpath='{.data.ca\\.crt}' | base64 -d | cmp -s - /opt/chartrepo-tls/tls.crt && kubectl -n %s get secret chartrepo-tls -o jsonpath='{.data.key\\.key}' | base64 -d | grep -q 'BEGIN .*PRIVATE KEY' && kubectl -n %s get bkecluster %s -o jsonpath='{.spec.clusterConfig.cluster.chartRepo.ip}{":"}{.spec.clusterConfig.cluster.chartRepo.port}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.name}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.caKey}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.certKey}{":"}{.spec.clusterConfig.cluster.chartRepo.tlsSecretRef.keyKey}' | grep -q '^%s:38443:chartrepo-tls:ca.crt:cert.crt:key.key$'`, selfNS, selfNS, selfNS, selfName, guideNodeHost)
	result, err := utils.ExecuteCommandOnNode(masterNode, masterCheckCmd)
	return err == nil && result != nil && result.ExitCode == 0
}

func cleanupFITHelmE2ETLSArtifacts(exec executor.Executor, nodes []config.NodeInfo, guideNodeHost, bootstrapClusterName string) {
	By("Cleanup FIT helm e2e TLS chart repo artifacts")

	guideCommands := []string{
		"nerdctl rm -f chartmuseum-tls-proxy >/dev/null 2>&1 || true",
		"rm -rf /opt/chartrepo-tls",
	}
	for _, cmd := range guideCommands {
		runGuideNodeCleanupCommand(exec, cmd, "cleanup chart repo TLS artifacts on guide node")
	}

	selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes)
	if len(selected) == 0 {
		GinkgoWriter.Printf("[WARN] skip cleanup chart repo TLS artifacts on master node: no management master node found\n")
		return
	}
	masterNode := selected[0]
	selfNS := utils.ManagementClusterSelfBCNamespace(bootstrapClusterName)
	selfName := utils.ManagementClusterSelfBCName(bootstrapClusterName)
	masterCommands := []string{
		fmt.Sprintf(`kubectl -n %s patch bkecluster %s --type merge -p '{"spec":{"clusterConfig":{"cluster":{"chartRepo":{"domain":"","ip":"%s","port":"38080","prefix":"","tlsSecretRef":null}}}}}' || true`, selfNS, selfName, guideNodeHost),
		fmt.Sprintf("kubectl -n %s delete secret chartrepo-tls --ignore-not-found || true", selfNS),
		"rm -rf /opt/chartrepo-tls",
	}
	for _, cmd := range masterCommands {
		runNodeCleanupCommand(masterNode, cmd, "cleanup chart repo TLS artifacts on master node")
	}
}

func runGuideNodeCleanupCommand(exec executor.Executor, cmd, action string) {
	result, err := exec.Exec(cmd)
	if err != nil {
		GinkgoWriter.Printf("[WARN] %s failed: cmd=%s err=%v\n", action, cmd, err)
		return
	}
	if result == nil {
		GinkgoWriter.Printf("[WARN] %s returned nil result: cmd=%s\n", action, cmd)
		return
	}
	if result.ExitCode != 0 {
		GinkgoWriter.Printf("[WARN] %s exit=%d cmd=%s stdout=%s stderr=%s\n", action, result.ExitCode, cmd, result.Stdout, result.Stderr)
	}
}

func runNodeCleanupCommand(node config.NodeInfo, cmd, action string) {
	result, err := utils.ExecuteCommandOnNode(node, cmd)
	if err != nil {
		GinkgoWriter.Printf("[WARN] %s on node %s(%s) failed: cmd=%s err=%v\n", action, node.Hostname, node.IP, cmd, err)
		return
	}
	if result == nil {
		GinkgoWriter.Printf("[WARN] %s on node %s(%s) returned nil result: cmd=%s\n", action, node.Hostname, node.IP, cmd)
		return
	}
	if result.ExitCode != 0 {
		GinkgoWriter.Printf("[WARN] %s on node %s(%s) exit=%d cmd=%s stdout=%s stderr=%s\n", action, node.Hostname, node.IP, result.ExitCode, cmd, result.Stdout, result.Stderr)
	}
}
func runGuideNodeCommand(exec executor.Executor, cmd, action string) {
	result, err := exec.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), "%s command should execute: %s", action, cmd)
	Expect(result).NotTo(BeNil(), "%s command should return result: %s", action, cmd)
	Expect(result.ExitCode).To(Equal(0), "%s command should succeed: %s\nstdout: %s\nstderr: %s", action, cmd, result.Stdout, result.Stderr)
}

func selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(nodes []config.NodeInfo) []config.NodeInfo {
	selected := make([]config.NodeInfo, 0, 2)
	masterIndex := -1
	workerIndex := -1
	for i, node := range nodes {
		if masterIndex == -1 && declarativeUpgradeNodeHasRole(node, "master") {
			masterIndex = i
			continue
		}
		if workerIndex == -1 && declarativeUpgradeNodeHasWorkerRole(node) {
			workerIndex = i
		}
	}
	if masterIndex >= 0 {
		selected = append(selected, nodes[masterIndex])
	}
	if workerIndex >= 0 && workerIndex != masterIndex {
		selected = append(selected, nodes[workerIndex])
	}
	return selected
}

func declarativeUpgradeNodeHasRole(node config.NodeInfo, expected string) bool {
	for _, role := range node.Role {
		if strings.Contains(strings.ToLower(strings.TrimSpace(role)), expected) {
			return true
		}
	}
	return false
}

func declarativeUpgradeNodeHasWorkerRole(node config.NodeInfo) bool {
	for _, role := range node.Role {
		role = strings.ToLower(strings.TrimSpace(role))
		if role == "node" || strings.Contains(role, "worker") {
			return true
		}
	}
	return false
}

// runDeclarativeUpgradeMgmtUpgradeAndVerify 在已安装的 vfit-pre 管理集群上触发升级到 targetVersion 并校验 DU-IT-005~006。
// bootstrapClusterName 为引导侧管理集群逻辑名（自管 BC 与之同名）。
// captureYamlEngineUIDs 仅 Gate 开启的 YamlInstaller 路径需要；Gate 关闭 Legacy 升级传 false。
func runDeclarativeUpgradeMgmtUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	bootstrapClusterName string,
	guideNodeHost string,
	mgmtNodes []config.NodeInfo,
	mgmtKubeconfigPath string,
	sourceVersion, targetVersion string,
	captureYamlEngineUIDs bool,
) {
	declClusterName := utils.ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName)
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, true)
	if captureYamlEngineUIDs && shouldCaptureFITYamlEngineUIDsForUpgrade(targetVersion) {
		captureFITYamlEngineFIT3And4UIDsBeforeCurUpgrade(
			clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, true,
		)
	}
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, mgmtKubeconfigPath, sourceVersion, targetVersion)

	cvBefore, err := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "升级前应能读取 ClusterVersion")
	GinkgoWriter.Printf("升级前 CV: desired=%s, current=%s, phase=%s\n",
		cvBefore.DesiredVersion, cvBefore.CurrentVersion, cvBefore.Phase)
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "升级前 currentVersion 应为源版本")

	By("DU-IT-005: Patch ClusterVersion.spec.desiredVersion 触发声明式升级")
	err = duHelper.PatchClusterVersionDesiredVersion(declClusterName, mgmtKubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion 应成功")

	Eventually(func() string {
		cv, e := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
		if e != nil {
			return ""
		}
		return cv.DesiredVersion
	}, 5*time.Minute, 10*time.Second).Should(Equal(targetVersion), "CV spec.desiredVersion 应更新为目标版本")

	By("DU-IT-005: 等待升级 DAG 执行并完成版本切换")
	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, declClusterName)
		if e != nil {
			return false
		}
		upgradeReady, _ := duHelper.GetBCAnnotation(declClusterName, mgmtKubeconfigPath, utils.UpgradeReadyAnnotation)
		cv, cvErr := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
		if cvErr == nil {
			GinkgoWriter.Printf("升级中自管 BC: phase=%s, state=%s, clusterStatus=%s, upgrade-ready=%q, CV current=%s desired=%s phase=%s\n",
				phase, state, clusterStatus, upgradeReady, cv.CurrentVersion, cv.DesiredVersion, cv.Phase)
		}
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("管理集群升级失败，BC state=%s, phase=%s, clusterStatus=%s", state, phase, clusterStatus))
		}
		if cvErr != nil {
			return false
		}
		return cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed") &&
			clusterStatus == "Ready"
	}, upgradeTimeout, pollInterval).Should(BeTrue(), "升级应完成且自管 BC Ready、CV currentVersion 达到目标版本")

	verifyDeclarativeUpgradeMgmtUpgradeComplete(
		duHelper, clusterManager, bootstrapClusterName, mgmtKubeconfigPath, targetVersion,
		[][]string{{sourceVersion, targetVersion}},
	)
	if captureYamlEngineUIDs && shouldCaptureFITYamlEngineUIDsForUpgrade(targetVersion) {
		verifyFITYamlEngineFIT3And4UIDsAfterCurUpgrade(duHelper, declClusterName)
	}
}

// runDeclarativeUpgradeMgmtMultiHopUpgradeAndVerify 一次 patch desiredVersion 为最终版本，
// 由调谐器在管理集群上分跳完成升级并校验 DU-IT-005~006。
type declarativeUpgradeMgmtMultiHopHelmOptions struct {
	exec          executor.Executor
	guideNodeHost string
	mgmtNodes     []config.NodeInfo
}

func runDeclarativeUpgradeMgmtMultiHopUpgradeAndVerify(
	clusterManager *utils.ClusterManager,
	duHelper *utils.DeclarativeUpgradeHelper,
	bootstrapClusterName string,
	mgmtKubeconfigPath string,
	sourceVersion, intermediateVersion, targetVersion string,
	helmOptions ...declarativeUpgradeMgmtMultiHopHelmOptions,
) {
	declClusterName := utils.ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName)
	ensureFITPreBaselineBeforeUpgrade(clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, true)
	verifyDeclarativeUpgradeMultiHopUpgradePathForUpgrade(
		duHelper, mgmtKubeconfigPath, sourceVersion, intermediateVersion, targetVersion)

	cvBefore, err := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "升级前应能读取 ClusterVersion")
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "升级前 currentVersion 应为源版本")
	GinkgoWriter.Printf("管理集群两跳升级前 CV: desired=%s, current=%s, phase=%s\n",
		cvBefore.DesiredVersion, cvBefore.CurrentVersion, cvBefore.Phase)

	helmOption := resolveDeclarativeUpgradeMgmtMultiHopHelmOptions(helmOptions)
	if helmOption != nil {
		prepareAndPublishFITHelmE2ECharts(helmOption.exec, helmOption.guideNodeHost, helmOption.mgmtNodes, bootstrapClusterName, "0.1.1", "0.1.2", "0.1.3")
	}

	By(fmt.Sprintf("DU-IT-005: 一次 Patch desiredVersion=%s，由调谐器分跳完成 %s→%s→%s",
		targetVersion, sourceVersion, intermediateVersion, targetVersion))
	err = duHelper.PatchClusterVersionDesiredVersion(declClusterName, mgmtKubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion 应成功")

	Eventually(func() string {
		cv, e := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
		if e != nil {
			return ""
		}
		return cv.DesiredVersion
	}, 5*time.Minute, 10*time.Second).Should(Equal(targetVersion), "CV spec.desiredVersion 应直接更新为最终目标版本")

	By("DU-IT-005: 等待调谐器按 UpgradePath 分跳执行并完成版本切换")
	sawIntermediate := false
	refreshedHelmTLSAfterIntermediate := false
	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, declClusterName)
		if e != nil {
			return false
		}
		upgradeReady, _ := duHelper.GetBCAnnotation(declClusterName, mgmtKubeconfigPath, utils.UpgradeReadyAnnotation)
		cv, cvErr := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
		if cvErr == nil {
			if cv.CurrentVersion == intermediateVersion {
				sawIntermediate = true
				if helmOption != nil && !refreshedHelmTLSAfterIntermediate {
					By(fmt.Sprintf("FIT: first hop reached %s, refresh chart repo TLS before next hop", intermediateVersion))
					selected := selectDeclarativeUpgradeMgmtMasterAndWorkerNodes(helmOption.mgmtNodes)
					Expect(selected).NotTo(BeEmpty(), "management cluster should have a master node for chart repo TLS patch after first hop")
					patchFITHelmE2ETLSChartRepoOnMasterNode(selected[0], helmOption.guideNodeHost, bootstrapClusterName)
					refreshedHelmTLSAfterIntermediate = true
				}
			}
			GinkgoWriter.Printf("管理集群两跳升级中自管 BC: phase=%s, state=%s, clusterStatus=%s, upgrade-ready=%q, CV current=%s desired=%s phase=%s\n",
				phase, state, clusterStatus, upgradeReady, cv.CurrentVersion, cv.DesiredVersion, cv.Phase)
		}
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("管理集群两跳升级失败，BC state=%s, phase=%s, clusterStatus=%s", state, phase, clusterStatus))
		}
		if cvErr != nil {
			return false
		}
		return cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed") &&
			clusterStatus == "Ready"
	}, upgradeTimeout*2, pollInterval).Should(BeTrue(),
		"两跳升级应完成且自管 BC Ready、CV currentVersion 达到最终目标版本")

	if sawIntermediate {
		GinkgoWriter.Printf("观察到调谐器经中间版本 %s 完成第一跳\n", intermediateVersion)
	} else {
		GinkgoWriter.Printf("未在轮询中捕获中间版本 %s（可能已快速跃迁）\n", intermediateVersion)
	}

	verifyDeclarativeUpgradeMgmtUpgradeComplete(
		duHelper, clusterManager, bootstrapClusterName, mgmtKubeconfigPath, targetVersion,
		[][]string{{sourceVersion, intermediateVersion}, {intermediateVersion, targetVersion}},
	)
	if helmOption != nil {
		verifyFITHelmE2EPodLabelVersionOnMasterNode(helmOption.mgmtNodes, "0.1.2")
	}
}

func resolveDeclarativeUpgradeMgmtMultiHopHelmOptions(options []declarativeUpgradeMgmtMultiHopHelmOptions) *declarativeUpgradeMgmtMultiHopHelmOptions {
	if len(options) == 0 {
		return nil
	}
	option := options[0]
	if option.exec == nil || strings.TrimSpace(option.guideNodeHost) == "" || len(option.mgmtNodes) == 0 {
		GinkgoWriter.Printf("[WARN] skip FIT helm e2e multi-hop TLS setup: incomplete options\n")
		return nil
	}
	return &option
}

// verifyDeclarativeUpgradeMgmtUpgradeComplete 校验管理集群 DU-IT-005~006 升级收尾状态。
func verifyDeclarativeUpgradeMgmtUpgradeComplete(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	bootstrapClusterName string,
	mgmtKubeconfigPath, targetVersion string,
	expectedHops [][]string,
) {
	declClusterName := utils.ManagementClusterDeclarativeResourceClusterName(bootstrapClusterName)

	By("DU-IT-006: 验证升级完成后 ClusterVersion.status 与管理集群自管 BC 状态稳定")
	Eventually(func() bool {
		cv, e := duHelper.GetClusterVersion(declClusterName, mgmtKubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 读取 CV 失败: %v\n", e)
			return false
		}
		if cv.DesiredVersion != targetVersion || cv.CurrentVersion != targetVersion {
			GinkgoWriter.Printf("升级收尾轮询: CV version 未就绪 desired=%s current=%s\n", cv.DesiredVersion, cv.CurrentVersion)
			return false
		}
		if cv.Phase != "Ready" && cv.Phase != "Installed" {
			GinkgoWriter.Printf("升级收尾轮询: CV phase=%s\n", cv.Phase)
			return false
		}
		upgradeReady, e := duHelper.GetBCAnnotation(declClusterName, mgmtKubeconfigPath, utils.UpgradeReadyAnnotation)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 读取 upgrade-ready 失败: %v\n", e)
			return false
		}
		if upgradeReady != "" && upgradeReady != "<none>" {
			GinkgoWriter.Printf("升级收尾轮询: upgrade-ready=%q 尚未清除\n", upgradeReady)
			return false
		}
		phase, state, clusterStatus, e := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, declClusterName)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 获取自管 BC 状态失败: %v\n", e)
			return false
		}
		GinkgoWriter.Printf("升级收尾轮询: BC phase=%s, state=%s, clusterStatus=%s, CV phase=%s\n",
			phase, state, clusterStatus, cv.Phase)
		return clusterStatus == "Ready"
	}, 10*time.Minute, 10*time.Second).Should(BeTrue(), "升级完成后 CV 与自管 BC 应稳定为 Ready")

	By("验证目标版本 ReleaseImage 已创建且 status.phase=Valid")
	expectedRIName := utils.ReleaseImageNameForVersion(targetVersion)
	Eventually(func() string {
		name, e := duHelper.FindReleaseImageByVersion(declClusterName, targetVersion, mgmtKubeconfigPath)
		if e != nil {
			return ""
		}
		return name
	}, 10*time.Minute, 15*time.Second).Should(Equal(expectedRIName), "应存在目标版本 ReleaseImage")

	Eventually(func() string {
		ri, e := duHelper.GetReleaseImage(declClusterName, expectedRIName, mgmtKubeconfigPath)
		if e != nil {
			return ""
		}
		return ri.Phase
	}, 10*time.Minute, 15*time.Second).Should(Equal("Valid"), "目标版本 ReleaseImage 应协调为 Valid")

	verifyClusterVersionUpgradeHistory(duHelper, declClusterName, mgmtKubeconfigPath, expectedHops)

	if shouldVerifyFITYamlComponentStatuses(duHelper, declClusterName, mgmtKubeconfigPath, targetVersion) {
		verifyFITYamlComponentsInstalled(duHelper, declClusterName, mgmtKubeconfigPath, targetVersion, fitYamlComponentsForTargetVersion(targetVersion))
	}
}

// verifyDeclarativeUpgradeUpgradeComplete 校验 DU-IT-005~006 升级收尾状态。
func verifyDeclarativeUpgradeUpgradeComplete(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath, targetVersion string,
	expectedHops [][]string,
) {
	By("DU-IT-006: 验证升级完成后 ClusterVersion.status 与 BKECluster 状态稳定")
	Eventually(func() bool {
		cv, e := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 读取 CV 失败: %v\n", e)
			return false
		}
		if cv.DesiredVersion != targetVersion || cv.CurrentVersion != targetVersion {
			GinkgoWriter.Printf("升级收尾轮询: CV version 未就绪 desired=%s current=%s\n", cv.DesiredVersion, cv.CurrentVersion)
			return false
		}
		if cv.Phase != "Ready" && cv.Phase != "Installed" {
			GinkgoWriter.Printf("升级收尾轮询: CV phase=%s\n", cv.Phase)
			return false
		}
		upgradeReady, e := duHelper.GetBCAnnotation(clusterName, kubeconfigPath, utils.UpgradeReadyAnnotation)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 读取 upgrade-ready 失败: %v\n", e)
			return false
		}
		if upgradeReady != "" && upgradeReady != "<none>" {
			GinkgoWriter.Printf("升级收尾轮询: upgrade-ready=%q 尚未清除\n", upgradeReady)
			return false
		}
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("升级收尾轮询: 获取 BC 状态失败: %v\n", e)
			return false
		}
		GinkgoWriter.Printf("升级收尾轮询: BC phase=%s, state=%s, clusterStatus=%s, CV phase=%s\n",
			phase, state, clusterStatus, cv.Phase)
		return clusterStatus == "Ready"
	}, 10*time.Minute, 10*time.Second).Should(BeTrue(), "升级完成后 CV 与 BC 应稳定为 Ready")

	By("验证目标版本 ReleaseImage 已创建且 status.phase=Valid")
	expectedRIName := utils.ReleaseImageNameForVersion(targetVersion)
	Eventually(func() string {
		name, e := duHelper.FindReleaseImageByVersion(clusterName, targetVersion, kubeconfigPath)
		if e != nil {
			return ""
		}
		return name
	}, 10*time.Minute, 15*time.Second).Should(Equal(expectedRIName), "应存在目标版本 ReleaseImage")

	Eventually(func() string {
		ri, e := duHelper.GetReleaseImage(clusterName, expectedRIName, kubeconfigPath)
		if e != nil {
			return ""
		}
		return ri.Phase
	}, 10*time.Minute, 15*time.Second).Should(Equal("Valid"), "目标版本 ReleaseImage 应协调为 Valid")

	verifyClusterVersionUpgradeHistory(duHelper, clusterName, kubeconfigPath, expectedHops)

	if shouldVerifyFITYamlComponentStatuses(duHelper, clusterName, kubeconfigPath, targetVersion) {
		verifyFITYamlComponentsInstalled(duHelper, clusterName, kubeconfigPath, targetVersion, fitYamlComponentsForTargetVersion(targetVersion))
	}
}

func shouldVerifyFITYamlComponentStatuses(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, targetVersion string,
) bool {
	if !strings.HasPrefix(targetVersion, "vfit-") {
		return false
	}
	annot, err := duHelper.GetBCAnnotation(clusterName, kubeconfigPath, utils.HelmComponentAnnotation)
	if err != nil {
		return false
	}
	return strings.EqualFold(strings.TrimSpace(annot), "true")
}

// verifyClusterVersionUpgradeHistory 校验 CV.status.upgradeHistory 包含期望的每一跳且 status=Succeeded。
func verifyClusterVersionUpgradeHistory(
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath string,
	expectedHops [][]string,
) {
	By("DU-IT-006: 验证 ClusterVersion.status.upgradeHistory 记录完整且每跳 status=Succeeded")
	Eventually(func() bool {
		history, err := duHelper.GetClusterVersionUpgradeHistory(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("读取 upgradeHistory 失败: %v\n", err)
			return false
		}
		if len(history) != len(expectedHops) {
			GinkgoWriter.Printf("upgradeHistory 条数=%d, 期望=%d\n", len(history), len(expectedHops))
			return false
		}
		for i, hop := range history {
			if hop.From != expectedHops[i][0] || hop.To != expectedHops[i][1] ||
				hop.Status != utils.ClusterVersionUpgradeStatusSucceeded {
				GinkgoWriter.Printf("upgradeHistory[%d]=%+v, 期望 %s→%s Succeeded\n",
					i, hop, expectedHops[i][0], expectedHops[i][1])
				return false
			}
		}
		return true
	}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
		"upgradeHistory 应包含所有升级跳次且每跳 status=Succeeded")

	history, err := duHelper.GetClusterVersionUpgradeHistory(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	for i, hop := range history {
		GinkgoWriter.Printf("upgradeHistory[%d]: %s→%s status=%s\n", i, hop.From, hop.To, hop.Status)
	}
}

// verifyDeclarativeUpgradeUpgradePath 校验 UpgradePath CR 存在且 status.phase=Active。
func verifyDeclarativeUpgradeUpgradePath(
	duHelper *utils.DeclarativeUpgradeHelper,
	kubeconfigPath string,
) {
	By("验证 UpgradePath CR 存在且 status.phase=Active")
	Eventually(func() bool {
		exists, err := duHelper.UpgradePathExists(kubeconfigPath)
		return err == nil && exists
	}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "应存在 UpgradePath CR")

	Eventually(func() string {
		up, err := duHelper.GetUpgradePath(kubeconfigPath)
		if err != nil {
			return ""
		}
		GinkgoWriter.Printf("UpgradePath: name=%s, phase=%s, pathCount=%d\n", up.Name, up.Phase, up.PathCount)
		return up.Phase
	}, 10*time.Minute, 15*time.Second).Should(Equal("Active"), "UpgradePath 应协调为 Active")

	up, err := duHelper.GetUpgradePath(kubeconfigPath)
	Expect(err).NotTo(HaveOccurred())
	Expect(up.Phase).To(Equal("Active"))
	Expect(up.PathCount).To(BeNumerically(">", 0), "UpgradePath status.pathCount 应大于 0")
}

// verifyDeclarativeUpgradeMultiHopUpgradePathForUpgrade 升级前校验 UP 包含每一跳升级路径。
func verifyDeclarativeUpgradeMultiHopUpgradePathForUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	kubeconfigPath, sourceVersion, intermediateVersion, targetVersion string,
) {
	By(fmt.Sprintf("DU-IT-005 前置: 验证 UpgradePath 允许 %s→%s→%s 两跳升级",
		sourceVersion, intermediateVersion, targetVersion))
	verifyDeclarativeUpgradeUpgradePath(duHelper, kubeconfigPath)

	for _, hop := range [][2]string{{sourceVersion, intermediateVersion}, {intermediateVersion, targetVersion}} {
		fromVersion, toVersion := hop[0], hop[1]
		Eventually(func() bool {
			allowed, err := duHelper.HasUpgradePathBetween(kubeconfigPath, fromVersion, toVersion)
			if err != nil {
				GinkgoWriter.Printf("检查 UpgradePath 规则失败: %v\n", err)
				return false
			}
			return allowed
		}, 10*time.Minute, 15*time.Second).Should(BeTrue(),
			fmt.Sprintf("UpgradePath 应包含未 blocked 的 %s→%s 升级路径", fromVersion, toVersion))
	}
}

// verifyDeclarativeUpgradeUpgradePathForUpgrade 升级前校验 UP 资源就绪且包含源版本到目标版本的升级路径。
func verifyDeclarativeUpgradeUpgradePathForUpgrade(
	duHelper *utils.DeclarativeUpgradeHelper,
	kubeconfigPath, sourceVersion, targetVersion string,
) {
	By(fmt.Sprintf("DU-IT-005 前置: 验证 UpgradePath 存在且允许 %s→%s 升级", sourceVersion, targetVersion))
	verifyDeclarativeUpgradeUpgradePath(duHelper, kubeconfigPath)

	Eventually(func() bool {
		allowed, err := duHelper.HasUpgradePathBetween(kubeconfigPath, sourceVersion, targetVersion)
		if err != nil {
			GinkgoWriter.Printf("检查 UpgradePath 规则失败: %v\n", err)
			return false
		}
		return allowed
	}, 10*time.Minute, 15*time.Second).Should(BeTrue(),
		fmt.Sprintf("UpgradePath 应包含未 blocked 的 %s→%s 升级路径", sourceVersion, targetVersion))
}

// cleanupDeclarativeUpgradeMgmtCluster 删除管理集群：引导 BC 先 reset、短窗等待、节点 reset、BC 收尾。
func cleanupDeclarativeUpgradeMgmtCluster(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	localExecutor *executor.LocalExecutor,
	clusterName, mgmtKubeconfigPath string,
	mgmtNodes []config.NodeInfo,
	configPath, nodeConfigPath string,
) {
	_ = sshExecutor
	By("清理管理集群")
	var configPaths []string
	if configPath != "" {
		configPaths = append(configPaths, configPath)
	}
	if nodeConfigPath != "" {
		configPaths = append(configPaths, nodeConfigPath)
	}
	clusterManager.CleanupManagementCluster(utils.ManagementClusterCleanupOptions{
		ClusterName:         clusterName,
		MgmtKubeconfigPath:  mgmtKubeconfigPath,
		MgmtNodes:           mgmtNodes,
		LocalExecutor:       localExecutor,
		ConfigPaths:         configPaths,
		SavedKubeconfigPath: mgmtKubeconfigPath,
	})
}

// Copied from e2e/installation/cluster_test_helpers.go — kept as local copy because this
// package (declarative_upgrade) is a separate Go package and cannot import the parent.
const failedStatesMsgTmpl = "集群进入失败态: state=%s, clusterStatus=%s，不再等待 Healthy"

// isClusterFailedState reports whether the cluster is in a terminal failure state.
// Source: e2e/installation/cluster_test_helpers.go
func isClusterFailedState(state, clusterStatus string) bool {
	return state == "DeployFailed" || clusterStatus == "InitializationFailed"
}

// failOnClusterFailure calls Fail immediately if the cluster has entered a terminal failure state.
// Source: e2e/installation/cluster_test_helpers.go
func failOnClusterFailure(state, clusterStatus string) {
	if isClusterFailedState(state, clusterStatus) {
		Fail(fmt.Sprintf(failedStatesMsgTmpl, state, clusterStatus))
	}
}

// triggerDeclarativeUpgradeClusterInstall 生成配置并 apply BC/BN，不等待 Healthy/Ready。
func triggerDeclarativeUpgradeClusterInstall(
	clusterManager *utils.ClusterManager,
	clusterConfig *config.BKEClusterConfig,
	clusterName, kubeconfigPath string,
) (configPath, nodeConfigPath string) {
	var err error
	configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
	GinkgoWriter.Printf("配置文件: cluster=%s, nodes=%s\n", configPath, nodeConfigPath)

	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "apply BC/BN 应该成功")
	return configPath, nodeConfigPath
}

func isDeclarativeUpgradeClusterInstallReady(
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath string,
) bool {
	phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
	if err != nil {
		GinkgoWriter.Printf("BC %s 状态查询失败: %v\n", clusterName, err)
		return false
	}
	GinkgoWriter.Printf("BC %s 状态: phase=%s, state=%s, clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
	failOnClusterFailure(state, clusterStatus)
	return clusterStatus == "Ready"
}

// waitDeclarativeUpgradeClustersInstallReady 并行等待多个业务集群安装收敛。
func waitDeclarativeUpgradeClustersInstallReady(
	clusterManager *utils.ClusterManager,
	kubeconfigPath string,
	clusterNames ...string,
) {
	By("并行等待集群安装收敛至 Healthy/Ready")
	Eventually(func() bool {
		for _, name := range clusterNames {
			if !isDeclarativeUpgradeClusterInstallReady(clusterManager, name, kubeconfigPath) {
				return false
			}
		}
		return true
	}, installTimeout, pollInterval).Should(BeTrue(), "所有集群应达 Healthy/Ready")
}

// declarativeUpgradeClusterCleanupJob 声明式升级业务集群清理任务。
type declarativeUpgradeClusterCleanupJob struct {
	ClusterName    string
	ConfigPath     string
	NodeConfigPath string
}

func cleanupDeclarativeUpgradeCluster(
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath string,
	configPath, nodeConfigPath string,
) {
	By("清理测试集群")
	if clusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
		By("触发集群删除")
		err := clusterManager.DeleteClusterWithKubeconfig(clusterName, kubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
		}

		By("等待集群完全删除")
		Eventually(func() bool {
			exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath)
			if exists {
				phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(clusterName, kubeconfigPath)
				GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
			}
			return !exists
		}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "集群应该被完全删除")
	}
	if err := clusterManager.CleanupClusterNamespaceWithKubeconfig(clusterName, kubeconfigPath); err != nil {
		GinkgoWriter.Printf("清理集群命名空间及 BKENode 残留失败: %v\n", err)
	}
	if configPath != "" {
		_ = clusterManager.CleanupConfig(configPath)
	}
	if nodeConfigPath != "" {
		_ = clusterManager.CleanupConfig(nodeConfigPath)
	}
}

// cleanupDeclarativeUpgradeClustersSerialTriggerParallelWait 串行触发删除、并行等待 BC 消失，再清理 ns 与配置文件。
func cleanupDeclarativeUpgradeClustersSerialTriggerParallelWait(
	clusterManager *utils.ClusterManager,
	kubeconfigPath string,
	jobs ...declarativeUpgradeClusterCleanupJob,
) {
	if len(jobs) == 0 {
		return
	}
	By("清理测试集群（串行触发删除，并行等待回收）")
	for _, job := range jobs {
		if !clusterManager.ClusterExistsWithKubeconfig(job.ClusterName, kubeconfigPath) {
			continue
		}
		By(fmt.Sprintf("触发集群 %s 删除", job.ClusterName))
		if err := clusterManager.DeleteClusterWithKubeconfig(job.ClusterName, kubeconfigPath); err != nil {
			GinkgoWriter.Printf("触发集群 %s 删除失败: %v\n", job.ClusterName, err)
		}
	}

	clusterNames := make([]string, len(jobs))
	for i, job := range jobs {
		clusterNames[i] = job.ClusterName
	}
	Eventually(func() bool {
		for _, name := range clusterNames {
			if clusterManager.ClusterExistsWithKubeconfig(name, kubeconfigPath) {
				phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(name, kubeconfigPath)
				GinkgoWriter.Printf("集群 %s 仍在删除中: phase=%s\n", name, phase)
				return false
			}
		}
		return true
	}, uninstallTimeout, pollInterval).Should(BeTrue(), "所有集群应被完全删除")

	for _, job := range jobs {
		if err := clusterManager.CleanupClusterNamespaceWithKubeconfig(job.ClusterName, kubeconfigPath); err != nil {
			GinkgoWriter.Printf("清理集群 %s 命名空间及 BKENode 残留失败: %v\n", job.ClusterName, err)
		}
		if job.ConfigPath != "" {
			_ = clusterManager.CleanupConfig(job.ConfigPath)
		}
		if job.NodeConfigPath != "" {
			_ = clusterManager.CleanupConfig(job.NodeConfigPath)
		}
	}
}
