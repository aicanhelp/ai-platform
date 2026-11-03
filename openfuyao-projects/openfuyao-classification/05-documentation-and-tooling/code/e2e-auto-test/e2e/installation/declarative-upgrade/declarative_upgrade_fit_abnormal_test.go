package declarative_upgrade

import (
	"fmt"
	"path/filepath"
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

const helmNegativeCleanupUpgradeControlPlaneTimeout = time.Minute

/*
  FIT 业务集群自定义仓和离线场景负向升级（负向路径，与 fit_upgrade 常规成功路径互补）。

  覆盖范围：仅 1M1W / 3M1W 业务集群（管理集群负向不在本文件）。

  ── 本文件覆盖的 FIT/DFR 编号含义 ──
    DAG_EXEC_FIT_4 — fit-yaml-exec-fail 执行失败，clusterComponentStatuses 记录 Failed 与错误信息（vfit-cur-neg-exec）
    DAG_EXEC_DFR_1 — fit-yaml-failfast-f Failed 后 fit-yaml-failfast-s 不得写入 clusterComponentStatuses（vfit-cur-neg-failfast）
    YAML_ENGINE_FIT_6 — prune=true 但未配置 pruneLabelSelector 时必须 Failed（fit-yaml-prune-bad，vfit-cur-neg-prune）
    YAML_ENGINE_FIT_7 — Apply 成功但 healthCheck 失败时不得 Installed，写 Failed（fit-yaml-hc-fail，vfit-cur-neg-hc）
*/

var _ = Describe("DAG/YamlInstaller/HelmInstaller FIT - 业务集群自定义仓和离线场景负向升级",
	Label("declarative-upgrade-fit-abnormal", "declarative-upgrade-fit", "dag-refactor", "yaml-installer", "abnormal-upgrade", "with-management-cluster", "offline", "custom-repo"), func() {

		var (
			sshExecutor    *executor.SSHExecutor
			guideConfig    *config.GuideNodeConfig
			dynamicClient  dynamic.Interface
			clusterManager *utils.ClusterManager
			duHelper       *utils.DeclarativeUpgradeHelper
		)

		BeforeEach(func() {
			guideConfig = config.LoadGuideNodeFromEnv()
			Expect(guideConfig.Host).NotTo(BeEmpty())
			Expect(guideConfig.Password).NotTo(BeEmpty())

			var err error
			sshExecutor, err = executor.NewSSHExecutor(
				guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password,
			)
			Expect(err).NotTo(HaveOccurred())

			kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
			restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
			Expect(err).NotTo(HaveOccurred())

			dynamicClient, err = dynamic.NewForConfig(restConfig)
			Expect(err).NotTo(HaveOccurred())

			clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
			duHelper = utils.NewDeclarativeUpgradeHelper(sshExecutor)
		})

		Describe("1M1W 业务集群 FIT 负向升级", Label("1m1w", "1m1w-workload-on-mgmt"), func() {
			var (
				workloadClusterName    string
				workloadConfigPath     string
				workloadNodeConfigPath string
				clusterNodes           []config.NodeInfo
				sourceVersion          string
				helmNegativeCleanup    bool
			)

			BeforeEach(func() {
				helmNegativeCleanup = false
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				workloadClusterName = fmt.Sprintf("du-fit-abnormal-wl-1m1w-%d", time.Now().UnixNano())

				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				workloadConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
					workloadClusterName, clusterNodes, sourceVersion, true,
				)

				workloadConfigPath, workloadNodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, workloadConfig, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterEach(func() {
				if helmNegativeCleanup {
					cleanupDeclarativeUpgradeClusterWithUpgradeControlPlaneUnblock(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						workloadConfigPath, workloadNodeConfigPath, sourceVersion,
					)
					return
				}
				cleanupDeclarativeUpgradeCluster(
					clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
					workloadConfigPath, workloadNodeConfigPath,
				)
			})

			/*
			 用例名称：YAML_ENGINE_FIT_6-1M1W业务集群 缺少 pruneLabelSelector 升级失败
			 用例含义：YAML_ENGINE_FIT_6 — prune 开启但缺少 pruneLabelSelector 时引擎必须 Failed，不得误标 Installed；
			         负向版本 vfit-cur-neg-prune，组件 fit-yaml-prune-bad
			 用例步骤：1）BeforeEach 安装 vfit-pre 基线 1M1W 业务集群（helm-component 注解开启），预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若 install 未落到目标集群），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-prune，预期结果 2
			         4）WaitClusterComponentPhase 等待 fit-yaml-prune-bad Phase=Failed，预期结果 2、3
			         5）校验 Message 含 pruneLabelSelector 关键字，预期结果 3
			 预期结果：1）vfit-pre 业务集群基线就绪
			         2）负向升级被阻断，fit-yaml-prune-bad 不写 Installed
			         3）Phase=Failed，Message 非空且含 pruneLabelSelector
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_6-1M1W业务集群 缺少 pruneLabelSelector 升级失败",
				Label("required_nodes:2", "YAML_ENGINE_FIT_6", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegPrune,
						"fit-yaml-prune-bad", "pruneLabelSelector",
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_7-1M1W业务集群 健康检查失败阻断 Installed
			 用例含义：YAML_ENGINE_FIT_7 — healthCheck 失败时组件 Phase 必须为 Failed，禁止写 Installed；
			         负向版本 vfit-cur-neg-hc，组件 fit-yaml-hc-fail
			 用例步骤：1）BeforeEach 安装 vfit-pre 基线 1M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-hc，预期结果 2
			         4）等待 fit-yaml-hc-fail Phase=Failed，预期结果 3
			         5）校验 Message 含 health 相关关键字，预期结果 3
			 预期结果：1）业务集群基线就绪
			         2）健康检查失败时不写 Installed
			         3）fit-yaml-hc-fail Phase=Failed，Message 含 health
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_7-1M1W业务集群 健康检查失败阻断 Installed",
				Label("required_nodes:2", "YAML_ENGINE_FIT_7", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegHC,
						"fit-yaml-hc-fail", "health",
					)
				})

			/*
			 用例名称：DAG_EXEC_FIT_4-1M1W业务集群 yaml 组件执行失败写回 Failed
			 用例含义：DAG_EXEC_FIT_4 — yaml 执行阶段稳定失败时 FailFast，clusterComponentStatuses 写 Phase=Failed 并保留 message；
			         负向版本 vfit-cur-neg-exec，组件 fit-yaml-exec-fail
			 用例步骤：1）BeforeEach 安装 vfit-pre 基线 1M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-exec，预期结果 2
			         4）等待 fit-yaml-exec-fail Phase=Failed，预期结果 3
			         5）校验 Message 非空，预期结果 3
			 预期结果：1）业务集群基线就绪
			         2）执行阶段稳定失败，FailFast 写回 Failed
			         3）fit-yaml-exec-fail Phase=Failed，Message 非空
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_4-1M1W业务集群 yaml 组件执行失败写回 Failed",
				Label("required_nodes:2", "DAG_EXEC_FIT_4", "P1"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegExec,
						fitExecFailComponent,
					)
				})

			/*
			 用例名称：DAG_EXEC_DFR_1-1M1W业务集群 FailFast 下批次失败隔离后置组件不误标 Installed
			 用例含义：DAG_EXEC_DFR_1 — 前置批次 fit-yaml-failfast-f Failed 后，后置 fit-yaml-failfast-s
			         不得写入 clusterComponentStatuses；负向版本 vfit-cur-neg-failfast
			 用例步骤：1）BeforeEach 安装 vfit-pre 基线 1M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-failfast 并等待收敛/FailFast，预期结果 2
			         4）读取 F=fit-yaml-failfast-f 与 S=fit-yaml-failfast-s 的 clusterComponentStatuses，预期结果 2、3
			 预期结果：1）业务集群基线就绪
			         2）fit-yaml-failfast-f Phase=Failed，Message 非空
			         3）fit-yaml-failfast-s 不得出现在 clusterComponentStatuses
			*/
			// Author: wangyikai
			It("DAG_EXEC_DFR_1-1M1W业务集群 FailFast 下批次失败隔离后置组件不误标 Installed",
				Label("required_nodes:2", "DAG_EXEC_DFR_1", "P2", "dfr"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITFailFastBatchIsolationUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegFailFast,
						fitFailFastFailedComponent, fitFailFastLaterComponent,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_DFX_3+Helm_FIT_6-1M1W业务集群 chart 版本不存在升级失败",
				Label("required_nodes:2", "Helm_DFX_3", "Helm_FIT_6", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITHelmChartNotFoundNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_DFX_2-1M1W业务集群 healthCheck 失败升级失败",
				Label("required_nodes:2", "Helm_DFX_2", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITHelmHealthCheckNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion, clusterNodes,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_FIT_9-1M1W业务集群 目标版本不存在升级失败",
				Label("required_nodes:2", "Helm_FIT_9", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITVersionNotFoundNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion,
					)
				})
		})

		Describe("3Master1Worker 业务集群 FIT 负向升级", Label("3m1w", "3m1w-workload-on-mgmt"), func() {
			var (
				workloadClusterName    string
				workloadConfigPath     string
				workloadNodeConfigPath string
				clusterNodes           []config.NodeInfo
				sourceVersion          string
				helmNegativeCleanup    bool
			)

			BeforeEach(func() {
				helmNegativeCleanup = false
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				workloadClusterName = fmt.Sprintf("du-fit-abnormal-wl-3m1w-%d", time.Now().UnixNano())

				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				workloadConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
					workloadClusterName, clusterNodes, sourceVersion, true,
				)
				if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
					workloadConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				workloadConfigPath, workloadNodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, workloadConfig, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterEach(func() {
				if helmNegativeCleanup {
					cleanupDeclarativeUpgradeClusterWithUpgradeControlPlaneUnblock(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						workloadConfigPath, workloadNodeConfigPath, sourceVersion,
					)
					return
				}
				cleanupDeclarativeUpgradeCluster(
					clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
					workloadConfigPath, workloadNodeConfigPath,
				)
			})

			/*
			 用例名称：YAML_ENGINE_FIT_6-3M1W业务集群 缺少 pruneLabelSelector 升级失败
			 用例含义：YAML_ENGINE_FIT_6 — 3M1W 高可用业务集群复验 prune 缺 selector 负向（同 1M1W FIT_6 语义）
			 用例步骤：1）BeforeEach 安装 vfit-pre 基线 3M1W 业务集群（含 VIP 配置），预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-prune，预期结果 2
			         4）fit-yaml-prune-bad Phase=Failed，Message 含 pruneLabelSelector，预期结果 3
			 预期结果：1）3M1W 业务集群基线就绪  2）负向升级被阻断  3）Phase=Failed
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_6-3M1W业务集群 缺少 pruneLabelSelector 升级失败",
				Label("required_nodes:4", "YAML_ENGINE_FIT_6", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegPrune,
						"fit-yaml-prune-bad", "pruneLabelSelector",
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_7-3M1W业务集群 健康检查失败阻断 Installed
			 用例含义：YAML_ENGINE_FIT_7 — 3M1W 业务集群 healthCheck 失败负向（同 1M1W FIT_7 语义）
			 用例步骤：1）BeforeEach 安装 vfit-pre 3M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-hc，预期结果 2
			         4）fit-yaml-hc-fail Phase=Failed，Message 含 health，预期结果 3
			 预期结果：1）业务集群基线就绪  2）不写 Installed  3）Phase=Failed
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_7-3M1W业务集群 健康检查失败阻断 Installed",
				Label("required_nodes:4", "YAML_ENGINE_FIT_7", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegHC,
						"fit-yaml-hc-fail", "health",
					)
				})

			/*
			 用例名称：DAG_EXEC_FIT_4-3M1W业务集群 yaml 组件执行失败写回 Failed
			 用例含义：DAG_EXEC_FIT_4 — 3M1W 业务集群 yaml 执行失败 FailFast 写 Failed（同 1M1W FIT_4 语义）
			 用例步骤：1）BeforeEach 安装 vfit-pre 3M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-exec，预期结果 2
			         4）fit-yaml-exec-fail Phase=Failed，Message 非空，预期结果 3
			 预期结果：1）业务集群基线就绪  2）FailFast 写 Failed  3）Message 非空
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_4-3M1W业务集群 yaml 组件执行失败写回 Failed",
				Label("required_nodes:4", "DAG_EXEC_FIT_4", "P1"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITNegativeUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegExec,
						fitExecFailComponent,
					)
				})

			/*
			 用例名称：DAG_EXEC_DFR_1-3M1W业务集群 FailFast 下批次失败隔离后置组件不误标 Installed
			 用例含义：DAG_EXEC_DFR_1 — 3M1W 业务集群复验 FailFast 批次隔离（同 1M1W DFR_1 语义）
			 用例步骤：1）BeforeEach 安装 vfit-pre 3M1W 业务集群，预期结果 1
			         2）升级前补导入 testdata 基线 fit-yaml manifest（若需要），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur-neg-failfast，预期结果 2
			         4）校验 fit-yaml-failfast-f=Failed 且 fit-yaml-failfast-s 未写入 clusterComponentStatuses，预期结果 3
			 预期结果：1）业务集群基线就绪  2）F Failed  3）S 不得写入 clusterComponentStatuses
			*/
			// Author: wangyikai
			It("DAG_EXEC_DFR_1-3M1W业务集群 FailFast 下批次失败隔离后置组件不误标 Installed",
				Label("required_nodes:4", "DAG_EXEC_DFR_1", "P2", "dfr"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITFailFastBatchIsolationUpgrade(
						duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, utils.DefaultFITTargetVersionNegFailFast,
						fitFailFastFailedComponent, fitFailFastLaterComponent,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_DFX_3+Helm_FIT_6-3M1W业务集群 chart 版本不存在升级失败",
				Label("required_nodes:4", "Helm_DFX_3", "Helm_FIT_6", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITHelmChartNotFoundNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_DFX_2-3M1W业务集群 healthCheck 失败升级失败",
				Label("required_nodes:4", "Helm_DFX_2", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITHelmHealthCheckNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion, clusterNodes,
					)
				})

			// Author: wangyikai
			It("升级适配Helm_FIT_9-3M1W业务集群 目标版本不存在升级失败",
				Label("required_nodes:4", "Helm_FIT_9", "P0"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					helmNegativeCleanup = true
					runDeclarativeUpgradeFITVersionNotFoundNegativeUpgrade(
						sshExecutor, duHelper, clusterManager, workloadClusterName, declarativeUpgradeMgmtKubeconfigPath, sourceVersion,
					)
				})
		})
	})

func cleanupDeclarativeUpgradeClusterWithUpgradeControlPlaneUnblock(
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterManager *utils.ClusterManager,
	clusterName, kubeconfigPath string,
	configPath, nodeConfigPath string,
	fallbackVersion string,
) {
	By("清理 Helm 负向升级测试集群")
	if clusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath) {
		By("触发集群删除")
		if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, kubeconfigPath); err != nil {
			GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
		}

		By("等待集群完全删除")
		var upgradeControlPlaneSince time.Time
		unblocked := false
		Eventually(func() bool {
			exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, kubeconfigPath)
			if exists {
				phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(clusterName, kubeconfigPath)
				GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
				if phase == "UpgradeControlPlane" && !unblocked {
					if upgradeControlPlaneSince.IsZero() {
						upgradeControlPlaneSince = time.Now()
					} else if time.Since(upgradeControlPlaneSince) >= helmNegativeCleanupUpgradeControlPlaneTimeout {
						unblockDeclarativeUpgradeBeforeCleanup(sshExecutor, duHelper, clusterName, kubeconfigPath, fallbackVersion)
						unblocked = true
					}
				} else if phase != "UpgradeControlPlane" {
					upgradeControlPlaneSince = time.Time{}
				}
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

func unblockDeclarativeUpgradeBeforeCleanup(
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterName, kubeconfigPath, fallbackVersion string,
) {
	By("Helm 负向升级清理: 回退 ClusterVersion 并清除 upgrade-ready 注解")
	targetVersion := fallbackVersion
	if cv, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath); err != nil {
		GinkgoWriter.Printf("读取 ClusterVersion 失败，使用 fallbackVersion=%q: %v\n", fallbackVersion, err)
	} else if cv.CurrentVersion != "" {
		targetVersion = cv.CurrentVersion
	}
	if targetVersion != "" {
		if err := duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion); err != nil {
			GinkgoWriter.Printf("回退 ClusterVersion desiredVersion 到 %s 失败: %v\n", targetVersion, err)
		}
	}

	bcName := utils.BKEClusterResourceName(clusterName)
	namespace := utils.BKEClusterNamespace(clusterName)
	cmd := fmt.Sprintf("%skubectl annotate bc %s -n %s cvo.openfuyao.cn/upgrade-ready- cvo.openfuyao.cn/upgrade-path- cvo.openfuyao.cn/cluster-version- --overwrite",
		fitKubeconfigPrefix(kubeconfigPath), bcName, namespace)
	result, err := sshExecutor.Exec(cmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		stdout, stderr, exitCode := "", "", -1
		if result != nil {
			stdout, stderr, exitCode = result.Stdout, result.Stderr, result.ExitCode
		}
		GinkgoWriter.Printf("清除 upgrade-ready 注解失败: err=%v exit=%d stdout=%s stderr=%s\n", err, exitCode, stdout, stderr)
	}
}

func fitKubeconfigPrefix(kubeconfigPath string) string {
	if kubeconfigPath == "" {
		return ""
	}
	return fmt.Sprintf("KUBECONFIG=%s ", kubeconfigPath)
}
