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
  特性：DAG 重构 + YamlInstaller 引擎 — FIT 管理集群自定义仓和离线场景升级用例（DU-IT-005~006）

  运行前确保离线仓含 FIT 预制组件。
  BeforeAll 安装 vfit-pre FIT 管理集群（helm-component），AfterAll 清理管理集群。
  自管 BC 逻辑名与引导侧 ClusterName 一致（ManagementClusterDeclarativeResourceClusterName）。
  引导 BC 上的 helm-component 不会继承到自管 BC，安装就绪后会补写到自管 BC。

  场景：SC-FIT-MGMT 1M1W/3M1W 单跳与两跳（FIT 编号含义同 declarative_upgrade_fit_upgrade_test.go 文件头）。

  覆盖：DAG_EXEC_FIT_1/2/5/6，YAML_ENGINE_FIT_1~5/8~10（观测 clusterComponentStatuses[组件名]）
*/

var _ = Describe("DAG/YamlInstaller/HelmInstaller  FIT - 管理集群自定义仓和离线场景升级",
	Label("declarative-upgrade-fit", "declarative-upgrade-mgmt-upgrade", "dag-refactor", "yaml-installer", "post-init", "offline", "custom-repo"), func() {

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
			Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
			Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

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

		Describe("1Master1Worker 管理集群单跳升级 vfit-pre→vfit-cur", Label("1m1w-mgmt", "single-upgrade", "P0", "26.09"), Ordered, func() {
			var (
				clusterName        string
				clusterConfig      *config.BKEClusterConfig
				clusterNodes       []config.NodeInfo
				configPath         string
				nodeConfigPath     string
				mgmtKubeconfigPath string
				sourceVersion      string
				targetVersion      string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-mgmt-upgrade-1m1w-%d", time.Now().Unix())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)

				By("前置：安装 vfit-pre FIT 管理集群并校验 DU-IT-001~004")
				configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
				)
			})

			AfterAll(func() {
				cleanupFITHelmE2ETLSArtifacts(sshExecutor, clusterNodes, guideConfig.Host, clusterName)
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
					clusterNodes, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_1+6+YAML_ENGINE_FIT_1~5+10-1M1W管理集群 vfit-pre 升级到 vfit-cur
			 用例含义：DAG_EXEC_FIT_1/2/6 + YAML_ENGINE_FIT_1/3~5/10 — 管理集群自管 BC 单跳升级合并校验
			 用例步骤：1）BeforeAll 安装 vfit-pre 1M1W 管理集群（helm-component），预期结果 1
			         2）Patch 自管 CV desiredVersion=vfit-cur，校验 baseline/策略/批次/UID，预期结果 2~9
			 预期结果：同业务集群 1M1W 单跳合并 It
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_1+2+6+YAML_ENGINE_FIT_1+3~5+10+Helm_DFX_1+4~6+Helm_FIT_1~3+5+7~8+10-1M1W管理集群 vfit-pre 升级到 vfit-cur 并校验 yaml 组件",
				Label("required_nodes:2", "DAG_EXEC_FIT_1", "DAG_EXEC_FIT_2", "DAG_EXEC_FIT_6", "YAML_ENGINE_FIT_1",
					"YAML_ENGINE_FIT_3", "YAML_ENGINE_FIT_4", "YAML_ENGINE_FIT_5", "YAML_ENGINE_FIT_10",
					"Helm_DFX_1", "Helm_DFX_4", "Helm_DFX_5", "Helm_DFX_6", "Helm_FIT_1", "Helm_FIT_2", "Helm_FIT_3", "Helm_FIT_5", "Helm_FIT_7", "Helm_FIT_8", "Helm_FIT_10",
					"P0", "P1", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMgmtPreToCurUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, clusterNodes, mgmtKubeconfigPath, sourceVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_2-1M1W管理集群 同目标版本二次触发升级保持 SSA 幂等
			 用例含义：YAML_ENGINE_FIT_2 — 管理集群同目标版本 upgrade-ready 二次触发，fit-yaml-ssa SSA 幂等
			 用例步骤：1）记录 fit-yaml-ssa ConfigMap UID，预期结果 1
			         2）RetriggerUpgradeViaUpgradeReady 二次触发自管 BC，预期结果 2
			         3）fit-yaml-ssa 仍为 Installed，CV 版本保持 vfit-cur，预期结果 2
			         4）校验 fit-yaml-ssa UID 与步骤 1 一致，预期结果 3
			 预期结果：1）UID 已记录
			         2）二次触发不导致 Failed，fit-yaml-ssa Phase=Installed，currentVersion=vfit-cur
			         3）fit-yaml-ssa ConfigMap UID 不变（SSA 幂等）
			*/
			// Author: xuyouhong
			It("YAML_ENGINE_FIT_2-1M1W管理集群 同目标版本二次触发升级保持 SSA 幂等",
				Label("required_nodes:2", "YAML_ENGINE_FIT_2", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runFITYamlEngineFIT2IdempotencyCheck(clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, targetVersion, true)
				})
		})

		Describe("1Master1Worker 管理集群两跳升级 vfit-pre→vfit-post", Label("1m1w-mgmt", "multi-upgrade"), Ordered, func() {
			var (
				clusterName         string
				clusterConfig       *config.BKEClusterConfig
				clusterNodes        []config.NodeInfo
				configPath          string
				nodeConfigPath      string
				mgmtKubeconfigPath  string
				sourceVersion       string
				intermediateVersion string
				targetVersion       string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				intermediateVersion = utils.DeclarativeUpgradeFITTargetVersion()
				targetVersion = utils.DefaultFITPostTargetVersion

				clusterName = fmt.Sprintf("du-fit-mgmt-multi-upgrade-1m1w-%d", time.Now().Unix())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)

				configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
				)
			})

			AfterAll(func() {
				cleanupFITHelmE2ETLSArtifacts(sshExecutor, clusterNodes, guideConfig.Host, clusterName)
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
					clusterNodes, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1-1M1W管理集群 两跳升级 vfit-pre→vfit-post
			 用例含义：DAG_EXEC_FIT_5 + YAML_ENGINE_FIT_1 — 管理集群两跳 UpgradePath 至 vfit-post 收敛
			 用例步骤：1）一次 Patch 自管 CV desiredVersion=vfit-post，由调谐器分跳升级，预期结果 1
			         2）Eventually CV/BC 收敛，校验保留 yaml 组件 Installed，预期结果 2
			 预期结果：1）upgradeHistory 两跳 Succeeded
			         2）CV desired/current=vfit-post，除 fit-yaml-remove 外组件 Installed
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1+Helm_FIT_4-1M1W管理集群 一次 patch desiredVersion=vfit-post 由调谐器两跳升级",
				Label("required_nodes:2", "DAG_EXEC_FIT_5", "YAML_ENGINE_FIT_1", "Helm_FIT_4", "P0", "vfit-post"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMgmtMultiHopUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, clusterNodes, mgmtKubeconfigPath,
						sourceVersion, intermediateVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_8-1M1W管理集群 vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变
			 用例含义：YAML_ENGINE_FIT_8 — 高版本 RI 不含 fit-yaml-remove，clusterComponentStatuses 旧记录不变，资源不删除
			 用例步骤：1）读取 fit-yaml-remove clusterComponentStatus 与目标集群 ConfigMap，预期结果 1、2
			 预期结果：1）fit-yaml-removed Phase=Installed，clusterComponentStatuses 旧记录仍在
			         2）fit-yaml-removed ConfigMap 资源仍存在
			*/
			// Author: xuyouhong
			It("YAML_ENGINE_FIT_8-1M1W管理集群 vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变",
				Label("required_nodes:2", "YAML_ENGINE_FIT_8", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					verifyFITYamlComponentNotAutoUninstalled(
						duHelper, clusterManager, declClusterName, mgmtKubeconfigPath,
						utils.FITYamlRemoveComponent, true,
					)
				})
		})

		Describe("3Master1Worker 管理集群单跳升级 vfit-pre→vfit-cur", Label("3m1w-mgmt", "single-upgrade", "P0", "26.09"), Ordered, func() {
			var (
				clusterName        string
				clusterConfig      *config.BKEClusterConfig
				clusterNodes       []config.NodeInfo
				configPath         string
				nodeConfigPath     string
				mgmtKubeconfigPath string
				sourceVersion      string
				targetVersion      string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-mgmt-upgrade-3m1w-%d", time.Now().Unix())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)
				if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
					clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
				)
			})

			AfterAll(func() {
				cleanupFITHelmE2ETLSArtifacts(sshExecutor, clusterNodes, guideConfig.Host, clusterName)
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
					clusterNodes, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_1+6+YAML_ENGINE_FIT_1~5+10-3M1W管理集群 vfit-pre 升级到 vfit-cur
			 用例含义：同 1M1W 管理集群单跳合并 It — FIT_1/2/6 + YAML_ENGINE_FIT_1/3~5/10（3M1W 高可用复验）
			 用例步骤：1）BeforeAll 安装 vfit-pre 3M1W 管理集群（含 VIP），预期结果 1
			         2）Patch 自管 CV desiredVersion=vfit-cur 并校验 baseline/策略/批次/UID，预期结果 2~9
			 预期结果：同 1M1W 管理集群单跳合并 It
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_1+2+6+YAML_ENGINE_FIT_1+3~5+10+Helm_DFX_1+4~6+Helm_FIT_1~3+5+7~8+10-3M1W管理集群 vfit-pre 升级到 vfit-cur 并校验 yaml 组件",
				Label("required_nodes:4", "DAG_EXEC_FIT_1", "DAG_EXEC_FIT_2", "DAG_EXEC_FIT_6", "YAML_ENGINE_FIT_1",
					"YAML_ENGINE_FIT_3", "YAML_ENGINE_FIT_4", "YAML_ENGINE_FIT_5", "YAML_ENGINE_FIT_10",
					"Helm_DFX_1", "Helm_DFX_4", "Helm_DFX_5", "Helm_DFX_6", "Helm_FIT_1", "Helm_FIT_2", "Helm_FIT_3", "Helm_FIT_5", "Helm_FIT_7", "Helm_FIT_8", "Helm_FIT_10",
					"P0", "P1", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMgmtPreToCurUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, clusterNodes, mgmtKubeconfigPath, sourceVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_2-3M1W管理集群 同目标版本二次触发升级保持 SSA 幂等
			 用例含义：YAML_ENGINE_FIT_2 — 3M1W 管理集群 SSA 幂等复验（同 1M1W 管理 FIT_2）
			 用例步骤：1）记录 fit-yaml-ssa ConfigMap UID，预期结果 1
			         2）RetriggerUpgradeViaUpgradeReady 二次触发自管 BC，预期结果 2
			         3）fit-yaml-ssa 仍为 Installed，CV 版本保持 vfit-cur，预期结果 2
			         4）校验 fit-yaml-ssa UID 与步骤 1 一致，预期结果 3
			 预期结果：1）UID 已记录
			         2）二次触发不导致 Failed，fit-yaml-ssa Phase=Installed，currentVersion=vfit-cur
			         3）fit-yaml-ssa ConfigMap UID 不变（SSA 幂等）
			*/
			// Author: xuyouhong
			It("YAML_ENGINE_FIT_2-3M1W管理集群 同目标版本二次触发升级保持 SSA 幂等",
				Label("required_nodes:4", "YAML_ENGINE_FIT_2", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runFITYamlEngineFIT2IdempotencyCheck(clusterManager, duHelper, declClusterName, mgmtKubeconfigPath, targetVersion, true)
				})
		})

		Describe("3Master1Worker 管理集群两跳升级 vfit-pre→vfit-post", Label("3m1w-mgmt", "multi-upgrade"), Ordered, func() {
			var (
				clusterName         string
				clusterConfig       *config.BKEClusterConfig
				clusterNodes        []config.NodeInfo
				configPath          string
				nodeConfigPath      string
				mgmtKubeconfigPath  string
				sourceVersion       string
				intermediateVersion string
				targetVersion       string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				intermediateVersion = utils.DeclarativeUpgradeFITTargetVersion()
				targetVersion = utils.DefaultFITPostTargetVersion

				clusterName = fmt.Sprintf("du-fit-mgmt-multi-upgrade-3m1w-%d", time.Now().Unix())
				declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)
				if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
					clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
					clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
				)
			})

			AfterAll(func() {
				cleanupFITHelmE2ETLSArtifacts(sshExecutor, clusterNodes, guideConfig.Host, clusterName)
				cleanupDeclarativeUpgradeMgmtCluster(
					clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
					clusterNodes, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1-3M1W管理集群 两跳升级 vfit-pre→vfit-post
			 用例含义：DAG_EXEC_FIT_5 + YAML_ENGINE_FIT_1 — 3M1W 管理集群两跳至 vfit-post（同 1M1W 管理两跳）
			 用例步骤：1）一次 Patch 自管 CV desiredVersion=vfit-post，由调谐器分跳升级，预期结果 1
			         2）Eventually CV/BC 收敛，校验保留 yaml 组件 Installed，预期结果 2
			 预期结果：1）upgradeHistory 两跳 Succeeded
			         2）CV desired/current=vfit-post，除 fit-yaml-remove 外组件 Installed
			*/
			// Author: xuyouhong
			It("DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1+Helm_FIT_4-3M1W管理集群 一次 patch desiredVersion=vfit-post 由调谐器两跳升级",
				Label("required_nodes:4", "DAG_EXEC_FIT_5", "YAML_ENGINE_FIT_1", "Helm_FIT_4", "P0", "vfit-post"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMgmtMultiHopUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, clusterNodes, mgmtKubeconfigPath,
						sourceVersion, intermediateVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_8-3M1W管理集群 vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变
			 用例含义：YAML_ENGINE_FIT_8 — 3M1W 管理集群复验 fit-yaml-remove 不自动卸载（同 1M1W 管理 FIT_8）
			 用例步骤：1）读取 fit-yaml-remove clusterComponentStatus 与目标集群 ConfigMap，预期结果 1、2
			 预期结果：1）fit-yaml-removed Phase=Installed，clusterComponentStatuses 旧记录仍在
			         2）fit-yaml-removed ConfigMap 资源仍存在
			*/
			// Author: xuyouhong
			It("YAML_ENGINE_FIT_8-3M1W管理集群 vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变",
				Label("required_nodes:4", "YAML_ENGINE_FIT_8", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					verifyFITYamlComponentNotAutoUninstalled(
						duHelper, clusterManager, declClusterName, mgmtKubeconfigPath,
						utils.FITYamlRemoveComponent, true,
					)
				})
		})

		Describe("1Master1Worker 管理集群 YAML_ENGINE_FIT_9 Prune 不误删核心资源",
			Label("1m1w-mgmt", "single-upgrade", "P1", "26.09"), func() {
				var (
					clusterName        string
					clusterNodes       []config.NodeInfo
					configPath         string
					nodeConfigPath     string
					mgmtKubeconfigPath string
					sourceVersion      string
					targetVersion      string
				)

				BeforeEach(func() {
					sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
					targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

					clusterName = fmt.Sprintf("du-fit-fit9-mgmt-1m1w-%d", time.Now().UnixNano())
					declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
					var err error
					clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WMgmtNodes()
					Expect(err).NotTo(HaveOccurred())
					clusterConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)

					configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
						clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
					)
				})

				AfterEach(func() {
					cleanupDeclarativeUpgradeMgmtCluster(
						clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
						clusterNodes, configPath, nodeConfigPath,
					)
				})

				/*
				 用例名称：YAML_ENGINE_FIT_9-1M1W管理集群 升级 Prune 不误删 fit-prune-core Namespace
				 用例含义：YAML_ENGINE_FIT_9 — 管理集群自管 BC（cluster）复验 Prune 不误删核心 Namespace
				 用例步骤：1）BeforeEach 安装 vfit-pre 1M1W 管理集群（helm-component），预期结果 1
				         2）升级前补导入 testdata 基线 manifest 到管理集群目标面，预期结果 1
				         3）apply fit-prune-core Namespace 并记录 UID，预期结果 2
				         4）自管 BC 升 vfit-cur，预期结果 3
				         5）校验 UID 不变且 fit-yaml-prune Installed，预期结果 3、4
				 预期结果：1）管理集群基线就绪  2）fit-prune-core 已预置  3）fit-yaml-prune Installed
				         4）fit-prune-core UID 不变
				*/
				// Author: xuyouhong
				It("YAML_ENGINE_FIT_9-1M1W管理集群 升级 Prune 不误删 fit-prune-core Namespace",
					Label("required_nodes:2", "YAML_ENGINE_FIT_9", "P1", "vfit-cur"),
					SpecTimeout(installationItTimeout+upgradeTimeout),
					func(ctx SpecContext) {
						runFITYamlEngineFIT9UpgradeAndVerify(
							sshExecutor, clusterManager, duHelper,
							declClusterName, mgmtKubeconfigPath, mgmtKubeconfigPath,
							sourceVersion, targetVersion, true,
						)
					})
			})

		Describe("3Master1Worker 管理集群 YAML_ENGINE_FIT_9 Prune 不误删核心资源",
			Label("3m1w-mgmt", "single-upgrade", "P1", "26.09"), func() {
				var (
					clusterName        string
					clusterNodes       []config.NodeInfo
					configPath         string
					nodeConfigPath     string
					mgmtKubeconfigPath string
					sourceVersion      string
					targetVersion      string
				)

				BeforeEach(func() {
					sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
					targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

					clusterName = fmt.Sprintf("du-fit-fit9-mgmt-3m1w-%d", time.Now().UnixNano())
					declClusterName = utils.ManagementClusterDeclarativeResourceClusterName(clusterName)
					var err error
					clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
					Expect(err).NotTo(HaveOccurred())
					clusterConfig := utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)
					if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
						clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
							Host: vipHost,
							Port: vipPort,
						}
					}

					configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeFITMgmtInstallAndVerify(
						clusterManager, sshExecutor, duHelper, clusterConfig, clusterName,
					)
				})

				AfterEach(func() {
					cleanupDeclarativeUpgradeMgmtCluster(
						clusterManager, sshExecutor, localExecutor, clusterName, mgmtKubeconfigPath,
						clusterNodes, configPath, nodeConfigPath,
					)
				})

				/*
				 用例名称：YAML_ENGINE_FIT_9-3M1W管理集群 升级 Prune 不误删 fit-prune-core Namespace
				 用例含义：YAML_ENGINE_FIT_9 — 3M1W 管理集群复验 Prune 不误删核心 Namespace（同 1M1W 管理 FIT_9）
				 用例步骤：1）BeforeEach 安装 vfit-pre 3M1W 管理集群（含 VIP），预期结果 1
				         2）升级前补导入 testdata 基线 manifest（若需要），预期结果 1
				         3）apply fit-prune-core 并记录 UID，预期结果 2
				         4）自管 BC 升 vfit-cur，预期结果 3
				         5）校验 UID 不变且 fit-yaml-prune Installed，预期结果 3、4
				 预期结果：1）3M1W 管理集群基线就绪  2）fit-prune-core 已预置  3）fit-yaml-prune Installed
				         4）fit-prune-core UID 不变
				*/
				// Author: xuyouhong
				It("YAML_ENGINE_FIT_9-3M1W管理集群 升级 Prune 不误删 fit-prune-core Namespace",
					Label("required_nodes:4", "YAML_ENGINE_FIT_9", "P1", "vfit-cur"),
					SpecTimeout(installationItTimeout+upgradeTimeout),
					func(ctx SpecContext) {
						runFITYamlEngineFIT9UpgradeAndVerify(
							sshExecutor, clusterManager, duHelper,
							declClusterName, mgmtKubeconfigPath, mgmtKubeconfigPath,
							sourceVersion, targetVersion, true,
						)
					})
			})
	})
