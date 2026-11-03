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
  特性：DAG 重构 + YamlInstaller 引擎 — FIT 业务集群自定义仓和离线场景升级用例（DU-IT-005~006）

  前置：管理集群已就绪（标签 with-management-cluster），FIT 业务集群在管理集群 API 上创建。
  运行前确保离线仓含 FIT 预制组件（release-image FIT-TEST-NOTES）。
  Gate 开启：BKECluster 注解 cvo.openfuyao.cn/helm-component=true，yaml 走 YamlComponentExecutor。
  AfterAll 仅清理业务集群，不清理管理集群。

  场景：
    SC-FIT-BIZ-1M1W / 3M1W — 单跳 vfit-pre→vfit-cur + FIT 组件断言
    SC-FIT-BIZ-1M1W-MULTI / 3M1W-MULTI — 两跳 vfit-pre→vfit-post + 组件移除

  ── FIT 编号含义（对齐自验证文档 v26.09）────────────────────────────────
  【DAG 调度层 — DAG_EXEC_FIT_*】
    FIT_1 — bkeagent inline 组件正常升级且写入 clusterComponentStatuses
    FIT_2 — fit-yaml-gate 在 Gate 开启时正常升级且写入 clusterComponentStatuses
    FIT_5 — fit-yaml-hc-ok 启用 healthCheck 的 yaml 组件正常升级（两跳场景验收）
    FIT_6 — 有前后依赖关系的组件均正常升级且写入 clusterComponentStatuses（bkeagent 先于 fit-yaml-ssa）
  【Yaml 引擎层 — YAML_ENGINE_FIT_*】
    FIT_1 — fit-yaml-new 首次安装且写入 clusterComponentStatuses
    FIT_2 — fit-yaml-ssa SSA：UID 不变、data 变化；二次触发 clusterComponentStatuses 幂等
    FIT_3 — fit-yaml-replace Replace：UID 与 data 均变化
    FIT_4 — fit-yaml-createonly CreateOnly：UID 与 data 均不变
    FIT_5 — fit-yaml-prune Prune：legacy ConfigMap 删除、keep 保留
    FIT_8 — fit-yaml-remove 高版本 RI 不含该组件时不自动卸载，clusterComponentStatuses 旧记录不变
    FIT_10 — fit-yaml-skip 跳过执行不写 clusterComponentStatuses；fit-yaml-ssa 正常记录
    FIT_9 — fit-yaml-prune + fit-prune-core Namespace：legacy cm 删除、NS 核心资源保留
  【负向/DFR 见 declarative_upgrade_fit_abnormal_test.go】FIT_4 exec-fail、FIT_6 prune-bad、FIT_7 hc-fail、DFR_1 failfast 批次隔离
  【Gate 关闭见 declarative_upgrade_fit_gate_closed_test.go】FIT_3 Legacy 无 clusterComponentStatuses；FIT_7 1M1W 双集群注解覆盖
  ────────────────────────────────────────────────────────────────────
*/

var _ = Describe("DAG/YamlInstaller/HelmInstaller FIT - 业务集群自定义仓和离线场景升级",
	Label("declarative-upgrade-fit", "declarative-upgrade-upgrade", "dag-refactor", "yaml-installer", "with-management-cluster", "offline", "custom-repo"), func() {

		var (
			sshExecutor    *executor.SSHExecutor
			guideConfig    *config.GuideNodeConfig
			dynamicClient  dynamic.Interface
			clusterManager *utils.ClusterManager
			duHelper       *utils.DeclarativeUpgradeHelper
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

			kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
			restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
			Expect(err).NotTo(HaveOccurred())

			dynamicClient, err = dynamic.NewForConfig(restConfig)
			Expect(err).NotTo(HaveOccurred())

			clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
			duHelper = utils.NewDeclarativeUpgradeHelper(sshExecutor)
		})

		Describe("1Master1Worker 业务集群单跳升级 vfit-pre→vfit-cur", Label("1m1w", "single-upgrade", "P0", "26.09"), Ordered, func() {
			var (
				clusterName    string
				clusterConfig  *config.BKEClusterConfig
				clusterNodes   []config.NodeInfo
				configPath     string
				nodeConfigPath string
				sourceVersion  string
				targetVersion  string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-upgrade-1m1w-%d", time.Now().Unix())
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeFITClusterConfig(
					clusterName, clusterNodes, sourceVersion, true,
				)

				By("前置：在管理集群上安装 vfit-pre FIT 业务集群并校验 DU-IT-001~004")
				configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterAll(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_1+2+6+YAML_ENGINE_FIT_1+3~5+10-1M1W vfit-pre 升级到 vfit-cur
			 用例含义：自验证预期合并验收 — bkeagent/fit-yaml-gate/fit-yaml-new 写入 clusterComponentStatuses；
			         fit-yaml-ssa/replace/createonly/prune/skip 策略行为；有依赖组件均含升级记录
			 用例步骤：1）BeforeAll 安装 vfit-pre FIT 业务集群，预期结果 1
			         2）Patch CV desiredVersion=vfit-cur，校验 DU-IT-005~006 与 clusterComponentStatuses，预期结果 2~9
			 预期结果：1）vfit-pre 业务集群基线就绪
			         2）BC Healthy/Ready，CV desired/current=vfit-cur
			         3）bkeagent、fit-yaml-gate、fit-yaml-new Installed 且写入 clusterComponentStatuses
			         4）fit-yaml-ssa UID 不变、data 变化；fit-yaml-replace UID/data 变化
			         5）fit-yaml-createonly UID/data 不变；fit-yaml-prune legacy 删除、keep 保留
			         6）fit-yaml-skip 不写入 clusterComponentStatuses；fit-yaml-ssa 正常记录
			         7）有依赖组件（bkeagent 先于 fit-yaml-ssa）均含 Installed 升级记录
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_1+2+6+YAML_ENGINE_FIT_1+3~5+10+Helm_DFX_1+4~6+Helm_FIT_1~3+5+7~8+10-1M1W vfit-pre 升级到 vfit-cur 并校验 yaml 组件",
				Label("required_nodes:2", "DAG_EXEC_FIT_1", "DAG_EXEC_FIT_2", "DAG_EXEC_FIT_6", "YAML_ENGINE_FIT_1",
					"YAML_ENGINE_FIT_3", "YAML_ENGINE_FIT_4", "YAML_ENGINE_FIT_5", "YAML_ENGINE_FIT_10",
					"Helm_DFX_1", "Helm_DFX_4", "Helm_DFX_5", "Helm_DFX_6", "Helm_FIT_1", "Helm_FIT_2", "Helm_FIT_3", "Helm_FIT_5", "Helm_FIT_7", "Helm_FIT_8", "Helm_FIT_10",
					"P0", "P1", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITPreToCurUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, declarativeUpgradeMgmtKubeconfigPath, clusterNodes,
						sourceVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_2-1M1W 同目标版本二次触发升级保持 SSA 幂等
			 用例含义：YAML_ENGINE_FIT_2 — 二次触发后 fit-yaml-ssa UID 不变，clusterComponentStatuses 记录幂等不变
			 用例步骤：1）记录 fit-yaml-ssa UID 与 clusterComponentStatuses，预期结果 1
			         2）RetriggerUpgradeViaUpgradeReady 二次触发，预期结果 2
			         3）fit-yaml-ssa 仍为 Installed，CV 版本保持 vfit-cur，预期结果 2
			         4）校验 UID 与 clusterComponentStatuses 与步骤 1 一致，预期结果 3
			 预期结果：1）UID 与 clusterComponentStatuses 已记录
			         2）二次触发不导致 Failed，fit-yaml-ssa Phase=Installed，currentVersion=vfit-cur
			         3）fit-yaml-ssa UID 不变，clusterComponentStatuses 字段不变（幂等）
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_2-1M1W 同目标版本二次触发升级保持 SSA 幂等",
				Label("required_nodes:2", "YAML_ENGINE_FIT_2", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runFITYamlEngineFIT2IdempotencyCheck(
						clusterManager, duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath, targetVersion, false,
					)
				})
		})

		Describe("1Master1Worker 业务集群两跳升级 vfit-pre→vfit-post", Label("1m1w", "multi-upgrade"), Ordered, func() {
			var (
				clusterName         string
				clusterConfig       *config.BKEClusterConfig
				clusterNodes        []config.NodeInfo
				configPath          string
				nodeConfigPath      string
				sourceVersion       string
				intermediateVersion string
				targetVersion       string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				intermediateVersion = utils.DeclarativeUpgradeFITTargetVersion()
				targetVersion = utils.DefaultFITPostTargetVersion

				clusterName = fmt.Sprintf("du-fit-multi-upgrade-1m1w-%d", time.Now().Unix())
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade1M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeFITClusterConfig(
					clusterName, clusterNodes, sourceVersion, true,
				)

				By("前置：在管理集群上安装 vfit-pre FIT 业务集群并校验 DU-IT-001~004")
				configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterAll(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1-1M1W 两跳升级 vfit-pre→vfit-post
			 用例含义：DAG_EXEC_FIT_5 — 多跳 UpgradePath 上 yaml 组件（含 HealthCheck）可收敛 Installed；
			         YAML_ENGINE_FIT_1 — 两跳升级完成后保留组件仍为 Installed、currentVersion=vfit-post
			 用例步骤：1）一次 Patch CV desiredVersion=vfit-post，由调谐器分跳 vfit-pre→vfit-cur→vfit-post，预期结果 1
			         2）Eventually CV/BC 收敛，校验保留 yaml 组件 Installed，预期结果 2
			 预期结果：1）upgradeHistory 两跳 Succeeded
			         2）CV desired/current=vfit-post，除 fit-yaml-remove 外组件 Installed
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1+Helm_FIT_4-1M1W 一次 patch desiredVersion=vfit-post 由调谐器两跳升级",
				Label("required_nodes:2", "DAG_EXEC_FIT_5", "YAML_ENGINE_FIT_1", "Helm_FIT_4", "P0", "vfit-post"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMultiHopUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, declarativeUpgradeMgmtKubeconfigPath, clusterNodes,
						sourceVersion, intermediateVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_8-1M1W vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变
			 用例含义：YAML_ENGINE_FIT_8 — 高版本 RI 不含 fit-yaml-remove，升级到 vfit-post 后集群资源不删除，
			         clusterComponentStatuses 保留 vfit-cur 时的旧记录不变
			 用例步骤：1）两跳升级完成后读取 fit-yaml-remove 的 clusterComponentStatus 与目标集群 ConfigMap，预期结果 1、2
			 预期结果：1）fit-yaml-removed Phase=Installed，clusterComponentStatuses 旧记录仍在
			         2）目标集群 fit-yaml-removed ConfigMap 资源仍存在
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_8-1M1W vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变",
				Label("required_nodes:2", "YAML_ENGINE_FIT_8", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					verifyFITYamlComponentNotAutoUninstalled(
						duHelper, clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath,
						utils.FITYamlRemoveComponent, false,
					)
				})
		})

		Describe("3Master1Worker 业务集群单跳升级 vfit-pre→vfit-cur", Label("3m1w", "single-upgrade", "P0", "26.09"), Ordered, func() {
			var (
				clusterName    string
				clusterConfig  *config.BKEClusterConfig
				clusterNodes   []config.NodeInfo
				configPath     string
				nodeConfigPath string
				sourceVersion  string
				targetVersion  string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

				clusterName = fmt.Sprintf("du-fit-upgrade-3m1w-%d", time.Now().Unix())
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeFITClusterConfig(
					clusterName, clusterNodes, sourceVersion, true,
				)
				if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
					clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				By("前置：在管理集群上安装 vfit-pre FIT 3M1W 业务集群并校验 DU-IT-001~004")
				configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterAll(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_1+6+YAML_ENGINE_FIT_1~5+10-3M1W vfit-pre 升级到 vfit-cur
			 用例含义：同 1M1W 单跳合并 It — FIT_1/2/6 + YAML_ENGINE_FIT_1/3~5/10（3M1W 高可用业务集群复验）
			 用例步骤：1）BeforeAll 安装 vfit-pre 3M1W FIT 业务集群，预期结果 1
			         2）Patch CV desiredVersion=vfit-cur 并校验 baseline/策略/批次/UID，预期结果 2~9
			 预期结果：同 1M1W 单跳合并 It
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_1+2+6+YAML_ENGINE_FIT_1+3~5+10+Helm_DFX_1+4~6+Helm_FIT_1~3+5+7~8+10-3M1W vfit-pre 升级到 vfit-cur 并校验 yaml 组件",
				Label("required_nodes:4", "DAG_EXEC_FIT_1", "DAG_EXEC_FIT_2", "DAG_EXEC_FIT_6", "YAML_ENGINE_FIT_1",
					"YAML_ENGINE_FIT_3", "YAML_ENGINE_FIT_4", "YAML_ENGINE_FIT_5", "YAML_ENGINE_FIT_10",
					"Helm_DFX_1", "Helm_DFX_4", "Helm_DFX_5", "Helm_DFX_6", "Helm_FIT_1", "Helm_FIT_2", "Helm_FIT_3", "Helm_FIT_5", "Helm_FIT_7", "Helm_FIT_8", "Helm_FIT_10",
					"P0", "P1", "vfit-cur"),
				SpecTimeout(installationItTimeout+upgradeTimeout),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITPreToCurUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, declarativeUpgradeMgmtKubeconfigPath, clusterNodes,
						sourceVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_2-3M1W 同目标版本二次触发升级保持 SSA 幂等
			 用例含义：YAML_ENGINE_FIT_2 — 3M1W 高可用业务集群复验 SSA 幂等（同 1M1W FIT_2 语义）
			 用例步骤：1）记录 fit-yaml-ssa ConfigMap UID，预期结果 1
			         2）RetriggerUpgradeViaUpgradeReady 二次触发，预期结果 2
			         3）fit-yaml-ssa 仍为 Installed，CV 版本保持 vfit-cur，预期结果 2
			         4）校验 fit-yaml-ssa UID 与步骤 1 一致，预期结果 3
			 预期结果：1）UID 已记录
			         2）二次触发不导致 Failed，fit-yaml-ssa Phase=Installed，currentVersion=vfit-cur
			         3）fit-yaml-ssa ConfigMap UID 不变（SSA 幂等）
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_2-3M1W 同目标版本二次触发升级保持 SSA 幂等",
				Label("required_nodes:4", "YAML_ENGINE_FIT_2", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					runFITYamlEngineFIT2IdempotencyCheck(
						clusterManager, duHelper, clusterName, declarativeUpgradeMgmtKubeconfigPath, targetVersion, false,
					)
				})
		})

		Describe("3Master1Worker 业务集群两跳升级 vfit-pre→vfit-post", Label("3m1w", "multi-upgrade"), Ordered, func() {
			var (
				clusterName         string
				clusterConfig       *config.BKEClusterConfig
				clusterNodes        []config.NodeInfo
				configPath          string
				nodeConfigPath      string
				sourceVersion       string
				intermediateVersion string
				targetVersion       string
			)

			BeforeAll(func() {
				sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
				intermediateVersion = utils.DeclarativeUpgradeFITTargetVersion()
				targetVersion = utils.DefaultFITPostTargetVersion

				clusterName = fmt.Sprintf("du-fit-multi-upgrade-3m1w-%d", time.Now().Unix())
				var err error
				clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WNodes()
				Expect(err).NotTo(HaveOccurred())
				clusterConfig = utils.NewDeclarativeUpgradeFITClusterConfig(
					clusterName, clusterNodes, sourceVersion, true,
				)
				if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
					clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
						Host: vipHost,
						Port: vipPort,
					}
				}

				By("前置：在管理集群上安装 vfit-pre FIT 3M1W 业务集群并校验 DU-IT-001~004")
				configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
					clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
				)
			})

			AfterAll(func() {
				cleanupDeclarativeUpgradeCluster(
					clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
				)
			})

			/*
			 用例名称：DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1-3M1W 两跳升级 vfit-pre→vfit-post
			 用例含义：DAG_EXEC_FIT_5 + YAML_ENGINE_FIT_1 — 3M1W 业务集群两跳 UpgradePath（同 1M1W 两跳首 It）
			 用例步骤：1）一次 Patch CV desiredVersion=vfit-post，由调谐器分跳 vfit-pre→vfit-cur→vfit-post，预期结果 1
			         2）Eventually CV/BC 收敛，校验保留 yaml 组件 Installed，预期结果 2
			 预期结果：1）upgradeHistory 两跳 Succeeded
			         2）CV desired/current=vfit-post，除 fit-yaml-remove 外组件 Installed
			*/
			// Author: wangyikai
			It("DAG_EXEC_FIT_5+YAML_ENGINE_FIT_1+Helm_FIT_4-3M1W 一次 patch desiredVersion=vfit-post 由调谐器两跳升级",
				Label("required_nodes:4", "DAG_EXEC_FIT_5", "YAML_ENGINE_FIT_1", "Helm_FIT_4", "P0", "vfit-post"),
				SpecTimeout(upgradeTimeout*2),
				func(ctx SpecContext) {
					runDeclarativeUpgradeFITMultiHopUpgradeAndVerify(
						clusterManager, duHelper, sshExecutor, guideConfig.Host, clusterName, declarativeUpgradeMgmtKubeconfigPath, clusterNodes,
						sourceVersion, intermediateVersion, targetVersion,
					)
				})

			/*
			 用例名称：YAML_ENGINE_FIT_8-3M1W vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变
			 用例含义：YAML_ENGINE_FIT_8 — 3M1W 业务集群复验 fit-yaml-remove 不自动卸载（同 1M1W FIT_8）
			 用例步骤：1）读取 fit-yaml-remove clusterComponentStatus 与目标集群 ConfigMap，预期结果 1、2
			 预期结果：1）fit-yaml-removed Phase=Installed，clusterComponentStatuses 旧记录仍在
			         2）fit-yaml-removed ConfigMap 资源仍存在
			*/
			// Author: wangyikai
			It("YAML_ENGINE_FIT_8-3M1W vfit-post 升级后 fit-yaml-remove 不自动卸载且旧记录不变",
				Label("required_nodes:4", "YAML_ENGINE_FIT_8", "P0"),
				SpecTimeout(upgradeTimeout),
				func(ctx SpecContext) {
					verifyFITYamlComponentNotAutoUninstalled(
						duHelper, clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath,
						utils.FITYamlRemoveComponent, false,
					)
				})
		})

		Describe("1Master1Worker 业务集群 YAML_ENGINE_FIT_9 Prune 不误删核心资源",
			Label("1m1w", "single-upgrade", "P1", "26.09"), func() {
				var (
					clusterName    string
					configPath     string
					nodeConfigPath string
					sourceVersion  string
					targetVersion  string
				)

				BeforeEach(func() {
					sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
					targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

					clusterName = fmt.Sprintf("du-fit-fit9-1m1w-%d", time.Now().UnixNano())
					var err error
					clusterNodes, err := utils.LoadDeclarativeUpgrade1M1WNodes()
					Expect(err).NotTo(HaveOccurred())
					clusterConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
						clusterName, clusterNodes, sourceVersion, true,
					)

					configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
						clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
					)
				})

				AfterEach(func() {
					cleanupDeclarativeUpgradeCluster(
						clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
					)
				})

				/*
				 用例名称：YAML_ENGINE_FIT_9-1M1W业务集群 升级 Prune 不误删 fit-prune-core Namespace
				 用例含义：YAML_ENGINE_FIT_9 — Prune 不误删不在 PruneableGVKs 内的核心资源；预置 fit-prune-core Namespace
				         （带 openfuyao.cn/fit-prune=fit-yaml-prune label），升 vfit-pre→vfit-cur 后 UID 不变
				 用例步骤：1）BeforeEach 安装 vfit-pre 1M1W 业务集群（helm-component），预期结果 1
				         2）升级前补导入 testdata/fit/vfit-pre-baseline（若 install 未落到目标集群），预期结果 1
				         3）在目标集群 apply yaml-engine-fit-9/fit-prune-core-namespace.yaml，记录 UID，预期结果 2
				         4）Patch CV desiredVersion=vfit-cur 并等待收敛，预期结果 3
				         5）校验 fit-prune-core UID 不变且 fit-yaml-prune Installed，预期结果 3、4
				 预期结果：1）vfit-pre 业务集群基线就绪
				         2）fit-prune-core Namespace 存在且 UID 已记录
				         3）fit-yaml-prune Phase=Installed，currentVersion=vfit-cur
				         4）fit-prune-core UID 与升级前一致（未被 Prune 误删）
				*/
				// Author: wangyikai
				It("YAML_ENGINE_FIT_9-1M1W业务集群 升级 Prune 不误删 fit-prune-core Namespace",
					Label("required_nodes:2", "YAML_ENGINE_FIT_9", "P1", "vfit-cur"),
					SpecTimeout(installationItTimeout+upgradeTimeout),
					func(ctx SpecContext) {
						runFITYamlEngineFIT9UpgradeAndVerify(
							sshExecutor, clusterManager, duHelper,
							clusterName, declarativeUpgradeMgmtKubeconfigPath, "",
							sourceVersion, targetVersion, false,
						)
					})
			})

		Describe("3Master1Worker 业务集群 YAML_ENGINE_FIT_9 Prune 不误删核心资源",
			Label("3m1w", "single-upgrade", "P1", "26.09"), func() {
				var (
					clusterName    string
					configPath     string
					nodeConfigPath string
					sourceVersion  string
					targetVersion  string
				)

				BeforeEach(func() {
					sourceVersion = utils.DeclarativeUpgradeFITSourceVersion()
					targetVersion = utils.DeclarativeUpgradeFITTargetVersion()

					clusterName = fmt.Sprintf("du-fit-fit9-3m1w-%d", time.Now().UnixNano())
					var err error
					clusterNodes, err := utils.LoadDeclarativeUpgrade3M1WNodes()
					Expect(err).NotTo(HaveOccurred())
					clusterConfig := utils.NewDeclarativeUpgradeFITClusterConfig(
						clusterName, clusterNodes, sourceVersion, true,
					)
					if vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv(); vipHost != "" {
						clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
							Host: vipHost,
							Port: vipPort,
						}
					}

					configPath, nodeConfigPath = runDeclarativeUpgradeFITInstallAndVerify(
						clusterManager, duHelper, clusterConfig, clusterName, declarativeUpgradeMgmtKubeconfigPath,
					)
				})

				AfterEach(func() {
					cleanupDeclarativeUpgradeCluster(
						clusterManager, clusterName, declarativeUpgradeMgmtKubeconfigPath, configPath, nodeConfigPath,
					)
				})

				/*
				 用例名称：YAML_ENGINE_FIT_9-3M1W业务集群 升级 Prune 不误删 fit-prune-core Namespace
				 用例含义：YAML_ENGINE_FIT_9 — 3M1W 高可用业务集群复验 Prune 不误删核心 Namespace（同 1M1W FIT_9 语义）
				 用例步骤：1）BeforeEach 安装 vfit-pre 3M1W 业务集群（含 VIP），预期结果 1
				         2）升级前补导入 testdata 基线 manifest（若需要），预期结果 1
				         3）apply fit-prune-core Namespace 并记录 UID，预期结果 2
				         4）升 vfit-cur 并等待收敛，预期结果 3
				         5）校验 UID 不变且 fit-yaml-prune Installed，预期结果 3、4
				 预期结果：1）3M1W 业务集群基线就绪  2）fit-prune-core 已预置  3）fit-yaml-prune Installed
				         4）fit-prune-core UID 不变
				*/
				// Author: wangyikai
				It("YAML_ENGINE_FIT_9-3M1W业务集群 升级 Prune 不误删 fit-prune-core Namespace",
					Label("required_nodes:4", "YAML_ENGINE_FIT_9", "P1", "vfit-cur"),
					SpecTimeout(installationItTimeout+upgradeTimeout),
					func(ctx SpecContext) {
						runFITYamlEngineFIT9UpgradeAndVerify(
							sshExecutor, clusterManager, duHelper,
							clusterName, declarativeUpgradeMgmtKubeconfigPath, "",
							sourceVersion, targetVersion, false,
						)
					})
			})
	})
