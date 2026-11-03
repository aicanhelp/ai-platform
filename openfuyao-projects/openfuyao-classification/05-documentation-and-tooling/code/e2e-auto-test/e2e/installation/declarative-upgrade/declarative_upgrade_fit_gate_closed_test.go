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

/*
  特性：DAG 重构 — Gate 关闭 / Legacy 路径 FIT 用例（自定义仓和离线场景）

  前置：
    1）管理集群已就绪（with-management-cluster）或本文件内安装 FIT 管理集群（1M1W/3M1W 管理场景）
    2）离线仓含 FIT 制品 vfit-pre/vfit-cur
    3）管理面 capbke 全局 --helm-component-support=false（Gate 关闭）

  覆盖：
    FIT_3 Legacy — 四种拓扑（1M1W/3M1W 业务与管理）
    FIT_7 注解覆盖 — 仅 1M1W 业务双集群并存同时升级（有注解新路径 / 无注解 Legacy）
*/

var _ = Describe("DAG/YamlInstaller FIT - 集群自定义仓和离线场景 Gate 关闭 Legacy 升级",
	Label("declarative-upgrade-fit-gate-closed", "declarative-upgrade-fit", "dag-refactor", "gate-closed", "with-management-cluster", "offline", "custom-repo"), func() {

		var (
			sshExecutor     *executor.SSHExecutor
			localExecutor   *executor.LocalExecutor
			guideConfig     *config.GuideNodeConfig
			dynamicClient   dynamic.Interface
			clusterManager  *utils.ClusterManager
			duHelper        *utils.DeclarativeUpgradeHelper
			declClusterName string
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

			localExecutor = executor.NewLocalExecutor(10 * time.Minute)

			kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
			restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
			Expect(err).NotTo(HaveOccurred())

			dynamicClient, err = dynamic.NewForConfig(restConfig)
			Expect(err).NotTo(HaveOccurred())

			clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
			duHelper = utils.NewDeclarativeUpgradeHelper(sshExecutor)
		})

		Describe("1Master1Worker 业务集群 Gate 关闭单跳升级", Label("1m1w", "single-upgrade", "P0", "26.09"), func() {
			var (
				clusterName    string
				clusterConfig  *config.BKEClusterConfig
				configPath     string
				nodeConfigPath string
				sourceVersion  string
				targetVersion  string
			)

			BeforeEach(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-gate-closed-1m1w-%d", time.Now().UnixNano())
				clusterNodes, err := utils.LoadDeclarativeUpgrade1M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeClusterConfig(clusterName, clusterNodes)

				configPath, nodeConfigPath = runDeclarativeUpgradeFITGateClosedInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterEach(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_3-1M1W Gate 关闭时 fit-yaml-gate Legacy 升级成功且无 clusterComponentStatuses 记录
			 用例含义：DAG_EXEC_FIT_3 — Gate 关闭时 fit-yaml-gate 走 Legacy：CCS 不写 Installed，但 declarativeUpgrade.completed 有记录
			 用例步骤：1）BeforeEach 安装无 helm-component 注解的 vfit-pre 1M1W 业务集群，记录 fit-yaml-gate phase，预期结果 1
			         2）升级前补导入 testdata 基线 manifest 到目标集群（若 install 未 apply yaml），预期结果 1
			         3）Patch CV desiredVersion=vfit-cur，等待 BC/CV 收敛，预期结果 2
			         4）校验 CCS 无 Installed，且 declarativeUpgrade.completed 含 fit-yaml-gate，预期结果 3
			 预期结果：1）vfit-pre 基线安装成功，升级前 fit-yaml-gate 非 Installed
			         2）CV desired/current=vfit-cur，BC Healthy/Ready，目标集群 yaml 数据已更新
			         3）CCS 中 fit-yaml-gate 无 Installed（未走 YamlInstaller）；completed 含 fit-yaml-gate（Legacy 写回成功）
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_3-1M1W Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed",
				Label("required_nodes:2", "DAG_EXEC_FIT_3", "P0", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					phaseBefore := snapshotFITComponentPhaseBeforeUpgrade(
						duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath, utils.FITYamlGateComponent,
					)
					expectPreUpgradePhaseNotInstalled(utils.FITYamlGateComponent, phaseBefore)

					runDeclarativeUpgradeFITGateClosedUpgradeAndVerify(
						clusterManager, duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, targetVersion,
					)
				})
		})

		Describe("3Master1Worker 业务集群 Gate 关闭单跳升级", Label("3m1w", "single-upgrade", "P0", "26.09"), func() {
			var (
				clusterName    string
				clusterConfig  *config.BKEClusterConfig
				configPath     string
				nodeConfigPath string
				sourceVersion  string
				targetVersion  string
			)

			BeforeEach(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-gate-closed-3m1w-%d", time.Now().UnixNano())
				clusterNodes, err := utils.LoadDeclarativeUpgrade3M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeClusterConfig(clusterName, clusterNodes)
				if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
					clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				configPath, nodeConfigPath = runDeclarativeUpgradeFITGateClosedInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterEach(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_3-3M1W Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed
			 用例含义：DAG_EXEC_FIT_3 — 3M1W 业务集群 Gate 关闭 Legacy 路径复验（同 1M1W FIT_3 语义）
			 用例步骤：1）BeforeEach 安装无注解 vfit-pre 3M1W 业务集群（含 VIP），预期结果 1
			         2）升级前补导入 testdata 基线 manifest（若需要），预期结果 1
			         3）升 vfit-cur 并等待收敛，预期结果 2
			         4）校验 CCS 无 Installed，且 declarativeUpgrade.completed 含 fit-yaml-gate，预期结果 3
			 预期结果：1）3M1W 基线就绪  2）升级成功  3）CCS 不写 Installed；completed 含 fit-yaml-gate
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_3-3M1W Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed",
				Label("required_nodes:4", "DAG_EXEC_FIT_3", "P0", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					phaseBefore := snapshotFITComponentPhaseBeforeUpgrade(
						duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath, utils.FITYamlGateComponent,
					)
					expectPreUpgradePhaseNotInstalled(utils.FITYamlGateComponent, phaseBefore)

					runDeclarativeUpgradeFITGateClosedUpgradeAndVerify(
						clusterManager, duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath,
						sourceVersion, targetVersion,
					)
				})
		})

		Describe("1Master1Worker 管理集群 Gate 关闭单跳升级", Label("1m1w-mgmt", "single-upgrade", "P0", "26.09"), func() {
			var (
				mgmtClusterName    string
				mgmtNodes          []config.NodeInfo
				mgmtConfigPath     string
				mgmtNodeConfigPath string
				mgmtKubeconfigPath string
				sourceVersion      string
				targetVersion      string
			)

			BeforeEach(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				mgmtClusterName = fmt.Sprintf("du-fit-gate-closed-mgmt-1m1w-%d", time.Now().UnixNano())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(mgmtClusterName)
				var err error
				mgmtNodes, err = utils.LoadDeclarativeUpgrade1M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				mgmtConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(mgmtClusterName, mgmtNodes)

				mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITGateClosedMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, mgmtConfig, mgmtClusterName,
				)
			})

			AfterEach(func() {
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
					mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_3-1M1W管理集群 Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed
			 用例含义：DAG_EXEC_FIT_3 — 1M1W 管理集群 Gate 关闭 Legacy 路径（自管 BC）
			 用例步骤：1）BeforeEach 安装无 helm-component 注解 vfit-pre 1M1W 管理集群，预期结果 1
			         2）升级前补导入 testdata 基线 manifest 到管理集群目标面，预期结果 1
			         3）自管 BC 升 vfit-cur，预期结果 2
			         4）校验 CCS 无 Installed，且 declarativeUpgrade.completed 含 fit-yaml-gate，预期结果 3
			 预期结果：1）管理集群基线就绪  2）升级成功  3）CCS 不写 Installed；completed 含 fit-yaml-gate
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_3-1M1W管理集群 Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed",
				Label("required_nodes:2", "DAG_EXEC_FIT_3", "P0", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					phaseBefore := snapshotFITComponentPhaseBeforeUpgrade(
						duHelper, declClusterName, mgmtKubeconfigPath, utils.FITYamlGateComponent,
					)
					expectPreUpgradePhaseNotInstalled(utils.FITYamlGateComponent, phaseBefore)

					runDeclarativeUpgradeFITGateClosedMgmtUpgradeAndVerify(
						clusterManager, duHelper, mgmtClusterName, mgmtKubeconfigPath, sourceVersion, targetVersion,
					)
				})
		})

		Describe("3Master1Worker 管理集群 Gate 关闭单跳升级", Label("3m1w-mgmt", "single-upgrade", "P0", "26.09"), func() {
			var (
				mgmtClusterName    string
				mgmtNodes          []config.NodeInfo
				mgmtConfigPath     string
				mgmtNodeConfigPath string
				mgmtKubeconfigPath string
				sourceVersion      string
				targetVersion      string
			)

			BeforeEach(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				mgmtClusterName = fmt.Sprintf("du-fit-gate-closed-mgmt-3m1w-%d", time.Now().UnixNano())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(mgmtClusterName)
				var err error
				mgmtNodes, err = utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				mgmtConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(mgmtClusterName, mgmtNodes)
				if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
					mgmtConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				mgmtConfigPath, mgmtNodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITGateClosedMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, mgmtConfig, mgmtClusterName,
				)
			})

			AfterEach(func() {
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
					mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_3-3M1W管理集群 Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed
			 用例含义：DAG_EXEC_FIT_3 — 3M1W 管理集群 Gate 关闭 Legacy 路径复验
			 用例步骤：1）BeforeEach 安装无注解 vfit-pre 3M1W 管理集群（含 VIP），预期结果 1
			         2）升级前补导入 testdata 基线 manifest（若需要），预期结果 1
			         3）自管 BC 升 vfit-cur，预期结果 2
			         4）校验 CCS 无 Installed，且 declarativeUpgrade.completed 含 fit-yaml-gate，预期结果 3
			 预期结果：1）3M1W 管理集群基线就绪  2）升级成功  3）CCS 不写 Installed；completed 含 fit-yaml-gate
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_3-3M1W管理集群 Gate 关闭时 yaml Legacy 升级成功且 clusterComponentStatuses 不写 Installed",
				Label("required_nodes:4", "DAG_EXEC_FIT_3", "P0", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					phaseBefore := snapshotFITComponentPhaseBeforeUpgrade(
						duHelper, declClusterName, mgmtKubeconfigPath, utils.FITYamlGateComponent,
					)
					expectPreUpgradePhaseNotInstalled(utils.FITYamlGateComponent, phaseBefore)

					runDeclarativeUpgradeFITGateClosedMgmtUpgradeAndVerify(
						clusterManager, duHelper, mgmtClusterName, mgmtKubeconfigPath, sourceVersion, targetVersion,
					)
				})
		})

		Describe("1Master1Worker 业务集群注解覆盖全局 Gate（双集群并存）",
			Label("1m1w", "gate-annotation-override", "P2", "26.09"), func() {
				var sourceVersion, targetVersion string

				BeforeEach(func() {
					sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
					targetVersion = utils.DeclarativeUpgradeFITTargetVersion()
				})

				/*
				 用例名称：DAG_EXEC_FIT_7-1M1W 双集群注解覆盖全局 Gate（有注解新路径 / 无注解 Legacy）
				 用例含义：DAG_EXEC_FIT_7 — 全局 Gate 关闭时，同时部署并升级两个 1M1W 业务集群：
				          A 带 helm-component 注解走新路径写 Installed；B 无注解走 Legacy 不写 Installed
				 用例步骤：1）串行 apply 带注解集群 A 与无注解集群 B（互斥节点，各 1M1W），并行等待均就绪，预期结果 1
				         2）同时 patch 两集群 desiredVersion=vfit-cur 并等待均收敛，预期结果 2
				         3）对比 A/B 的 clusterComponentStatuses 与 declarativeUpgrade 写回差异，预期结果 3
				 预期结果：1）两集群基线就绪
				         2）两集群 CV/BC 均收敛到 vfit-cur
				         3）仅 A 的 fit-yaml-gate Phase=Installed；B 的 CCS 无 Installed 且 completed 含 fit-yaml-gate
				*/
				// Author: xuyouhong
				It("DAG_EXEC_FIT_7-1M1W 双集群同时升级：注解开启新路径与无注解 Legacy",
					Label("required_nodes:4", "DAG_EXEC_FIT_7", "P2", "vfit-cur"),
					SpecTimeout(2*installationItTimeout+upgradeTimeout),
					func(ctx SpecContext) {
						runDeclarativeUpgradeFIT7Dual1M1WAnnotationOverride(
							clusterManager, duHelper, declarativeUpgradeMgmtKubeconfigPath,
							sourceVersion, targetVersion,
						)
					})
			})
	})
