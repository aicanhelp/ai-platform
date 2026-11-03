package installation

import (
	"fmt"
	"path/filepath"
	"strings"
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

const InstallationItTimeout = 45 * time.Minute

var _ = SIGDescribe("BKE Cluster Installation", func() {
	// InstallationItTimeout is the default Spec timeout for tests under e2e/installatio

	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
		// 加载引导节点配置
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		// 创建 SSH 执行器连接引导节点
		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

		// 创建 K8s 动态客户端（连接引导集群）
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		// 创建集群管理器
		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)

		// 创建本地执行器用于证书验证（通过引导节点连接到目标节点）
		localExecutor = executor.NewLocalExecutor(30 * time.Second)
	})

	Describe("创建 1Master 集群并断 1Master节点的网络", Label("1master", "network-disconnect", "post-init", "skip-temporarily", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			backupFile     string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m-disconnect-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "1Master 集群需要至少 1 个节点")

			// 使用第一个节点作为 Master
			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 创建集群配置
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			// 使用辅助函数断开 master 节点的网络
			var err error
			backupFile, err = DisconnectNetwork(masterNode, localExecutor)
			Expect(err).NotTo(HaveOccurred(), "应该成功断开 master 节点的网络")
		})

		AfterEach(func() {
			// 使用辅助函数恢复 master 节点的网络
			err := RestoreNetwork(masterNode, localExecutor, backupFile)
			if err != nil {
				GinkgoWriter.Printf("恢复 master 节点网络失败: %v\n", err)
			}

			// 清理集群
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				By("触发集群删除")
				err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
				}

				// 等待集群完全删除
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
					if exists {
						phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(clusterName, "")
						GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
					}
					return !exists
				}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}

			// 清理配置文件
			if configPath != "" {
				if err := clusterManager.CleanupConfig(configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：当 master 节点网络断开时应该阻止集群创建
		// 用例步骤：1) 通过引导节点生成并上传集群配置文件；2) 在 master 节点网络断开场景下执行 `bke cluster create`；3) 轮询集群状态直到确认创建失败（DeployFailed/InitializationFailed）。
		// 预期结果：最终集群状态不为 Healthy，且集群创建被正确阻止。
		It("当 master 节点网络断开时应该阻止集群创建", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待并验证集群创建被阻止")
			// 等待集群创建失败，状态为 DeployFailed 和 InitializationFailed
			checkInterval := 30 * time.Second
			checkTimeout := 10 * time.Minute // 检查 10 分钟以验证被阻止
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)

				if clusterStatus == "Ready" {
					// 如果集群变为 Ready，说明没有被阻止 - 这是意外的
					Fail(fmt.Sprintf("集群意外变为 Ready。预期集群创建被阻止，但集群状态为: %s", clusterStatus))
				}

				// 检查状态是否为 DeployFailed 且 clusterStatus 为 InitializationFailed
				// 由于现在当 clusterStatus 长时间为 InitializationFailed，state也不会变为 DeployFailed，修改退出逻辑
				return state == "DeployFailed" || clusterStatus == "InitializationFailed"
			}, checkTimeout, checkInterval).Should(BeTrue(), "集群创建应该被阻止，状态为 DeployFailed 和 InitializationFailed")

			// 最终状态检查
			_, _, finalClusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(finalClusterStatus).NotTo(Equal("Ready"), "当 master 节点网络断开时，集群不应该是 Ready")
			GinkgoWriter.Printf("最终集群状态: %s (预期被阻止，不是 Ready)\n", finalClusterStatus)
		})
	})

	Describe("创建 1Master1Worker 业务集群并断开 Worker 节点网络", Label("1master1worker", "network-disconnect", "post-init", "skip-temporarily", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			workerNode     config.NodeInfo
			backupFile     string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m1w-disconnect-worker-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Worker 集群需要至少 2 个节点")

			// 使用第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 使用第二个节点作为 Worker
			workerNode = nodes[1]
			workerNode.Role = []string{"node"}

			// 创建集群配置（业务集群不带 cluster-api，默认排除 cluster-api）
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode, workerNode})

			// 使用辅助函数断开 worker 节点的网络
			var err error
			backupFile, err = DisconnectNetwork(workerNode, localExecutor)
			Expect(err).NotTo(HaveOccurred(), "应该成功断开 worker 节点的网络")
		})

		AfterEach(func() {
			// 使用辅助函数恢复 worker 节点的网络
			err := RestoreNetwork(workerNode, localExecutor, backupFile)
			if err != nil {
				GinkgoWriter.Printf("恢复 worker 节点网络失败: %v\n", err)
			}

			// 清理集群
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				By("触发集群删除")
				err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
				}

				// 等待集群完全删除
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
					if exists {
						phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(clusterName, "")
						GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
					}
					return !exists
				}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}

			// 清理配置文件
			if configPath != "" {
				if err := clusterManager.CleanupConfig(configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：当 worker 节点网络断开时应该成功创建集群并保持 Healthy
		// 用例步骤：1) 通过引导节点生成并上传集群配置并校验配置已在引导节点落地；2) 执行 `bke cluster create` 创建集群；3) 等待集群达到 Healthy 且 Ready。
		// 预期结果：即使 worker 网络断开，集群仍可完成初始化并保持 `state=Healthy`、`clusterStatus=Ready`。
		It("当 worker 节点网络断开时应该成功创建集群并保持 Healthy", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "即使 worker 节点网络断开，集群也应该变为 Ready 状态")
		})
	})

	Describe("创建 1Master1Worker 集群并断开 Worker 节点网络后缩容", Label("1m1w-delete-disconnect-1w", "network-disconnect", "post-init", "skip-temporarily", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			workerNode     config.NodeInfo
			backupFile     string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m1w-disconnect-worker-scalein-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Worker 集群需要至少 2 个节点")

			// 使用第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 使用第二个节点作为 Worker
			workerNode = nodes[1]
			workerNode.Role = []string{"node"}

			// 创建集群配置（业务集群不带 cluster-api，默认排除 cluster-api）
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode, workerNode})

			// 使用辅助函数断开 worker 节点的网络
			var err error
			backupFile, err = DisconnectNetwork(workerNode, localExecutor)
			Expect(err).NotTo(HaveOccurred(), "应该成功断开 worker 节点的网络")
		})

		AfterEach(func() {
			// 使用辅助函数恢复 worker 节点的网络
			err := RestoreNetwork(workerNode, localExecutor, backupFile)
			if err != nil {
				GinkgoWriter.Printf("恢复 worker 节点网络失败: %v\n", err)
			}

			// 清理集群
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				By("触发集群删除")
				err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
				}

				// 等待集群完全删除
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
					if exists {
						phase, _ := clusterManager.GetClusterPhaseWithKubeconfig(clusterName, "")
						GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
					}
					return !exists
				}, uninstallTimeout, 60*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}

			// 清理配置文件
			if configPath != "" {
				if err := clusterManager.CleanupConfig(configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：当 worker 节点网络断开时应该成功创建集群并保持 Healthy，然后缩容 worker 节点
		// 用例步骤：1) 通过引导节点创建集群并等待 Healthy/Ready；2) 在缩容前再次确认 Healthy/Ready；3) 执行缩容移除 worker 并等待最终 Healthy/Ready，必要时校验节点数量回落。
		// 预期结果：缩容后集群仍保持可用，`state` 为 Healthy，`clusterStatus` 为 Ready（且节点数量符合预期）。
		It("当 worker 节点网络断开时应该成功创建集群并保持 Healthy，然后缩容 worker 节点", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "即使 worker 节点网络断开，集群也应该变为 Ready 状态")

			// 使用现有的 ScaleInNode 函数缩容 worker 节点
			By(fmt.Sprintf("缩容 worker 节点: %s", workerNode.IP))
			err = clusterManager.ScaleInNode(clusterName, workerNode.IP)
			Expect(err).NotTo(HaveOccurred(), "应该成功缩容 worker 节点")

			// 步骤 4: 等待缩容后集群变为 Ready
			By("等待缩容后集群变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群完整状态失败: %v\n", err)
					return false
				}
				GinkgoWriter.Printf("缩容后当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "缩容后集群应该变为 Ready")

			// 验证节点数量减少到 1（只有 master 节点）
			By("验证缩容后节点数量减少到 1")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			count, err := checker.GetNodeCountWithKubeconfig("")
			Expect(err).NotTo(HaveOccurred(), "应该成功获取节点数量")
			Expect(count).To(Equal(1), "缩容后应该有 1 个节点（只有 master 节点）")
			GinkgoWriter.Printf("缩容后节点数量: %d (预期: 1)\n", count)
		})
	})

	Describe("创建 1Master 管理集群和 1Master1Worker 业务集群并断开 Worker 节点网络", Label("1m-mgmt-1m1w-workload", "network-disconnect", "post-init", "skip-temporarily", "online"), Ordered, func() {
		var (
			mgmtClusterName        string
			mgmtClusterConfig      *config.BKEClusterConfig
			mgmtConfigPath         string
			mgmtNodeConfigPath     string
			mgmtKubeconfigPath     string
			workloadClusterName    string
			workloadClusterConfig  *config.BKEClusterConfig
			workloadConfigPath     string
			workloadNodeConfigPath string
			workerNode             config.NodeInfo
		)

		BeforeAll(func() {
			By("在引导集群上创建 1Master 管理集群")
			mgmtClusterName = fmt.Sprintf("test-mgmt-1m-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "1Master 管理集群需要至少 1 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 创建管理集群配置（管理集群需要 cluster-api 来管理业务集群）
			// 包含 cluster-api 但排除 openfuyao-system-controller（与证书测试用例相同）
			mgmtClusterConfig = config.NewBKEClusterConfigForMgmt(mgmtClusterName, []config.NodeInfo{masterNode})

			By("生成管理集群配置文件")
			var err error
			mgmtConfigPath, mgmtNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(mgmtClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			By("执行管理集群的 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(mgmtConfigPath, mgmtNodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "管理集群创建命令应该成功执行")

			By("等待管理集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(mgmtClusterName, "")
				GinkgoWriter.Printf("当前管理集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("管理集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "管理集群应该变为 Ready 状态")

			By("获取管理集群 kubeconfig")
			mgmtChecker := utils.NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), mgmtClusterName, "")
			mgmtKubeconfigPath, err = mgmtChecker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "应该成功获取管理集群 kubeconfig")
			GinkgoWriter.Printf("管理集群 kubeconfig 保存到: %s\n", mgmtKubeconfigPath)

			By("准备业务集群配置并断开 worker 节点网络")
			workloadClusterName = fmt.Sprintf("test-workload-1m1w-disconnect-worker-%d", time.Now().Unix())

			// 管理集群使用 nodes[0]，所以业务集群需要至少 nodes[1] 和 nodes[2]
			Expect(len(nodes)).To(BeNumerically(">=", 3), "1Master1Worker 业务集群需要至少 3 个节点（管理集群使用节点 1，业务集群需要节点 2-3）")

			// 使用节点 2 作为 Master (nodes[1])
			workloadMasterNode := nodes[1]
			workloadMasterNode.Role = []string{"master/node", "etcd"}

			// 使用节点 3 作为 Worker (nodes[2])
			workerNode = nodes[2]
			workerNode.Role = []string{"node"}

			// 创建业务集群配置（业务集群不带 cluster-api，默认排除 cluster-api）
			workloadClusterConfig = config.NewDefaultBKEClusterConfig(workloadClusterName, []config.NodeInfo{workloadMasterNode, workerNode})

			// 使用辅助函数断开 worker 节点网络
			_, err = DisconnectNetwork(workerNode, localExecutor)
			Expect(err).NotTo(HaveOccurred(), "应该成功断开 worker 节点的网络")
		})

		AfterAll(func() {
			// 使用辅助函数恢复 worker 节点的网络
			if workerNode.IP != "" {
				err := RestoreNetwork(workerNode, localExecutor, "")
				if err != nil {
					GinkgoWriter.Printf("恢复 worker 节点网络失败: %v\n", err)
				}
			}

			// 清理业务集群
			By("清理业务集群")
			if workloadClusterName != "" && mgmtKubeconfigPath != "" {
				if clusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath) {
					if err := clusterManager.DeleteClusterWithKubeconfig(workloadClusterName, mgmtKubeconfigPath); err != nil {
						GinkgoWriter.Printf("删除业务集群失败: %v\n", err)
					}
					Eventually(func() bool {
						return !clusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
					}, uninstallTimeout, 10*time.Second).Should(BeTrue())
				}
				if workloadConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadConfigPath); err != nil {
						GinkgoWriter.Printf("清理业务集群配置文件失败: %v\n", err)
					}
				}
				if workloadNodeConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadNodeConfigPath); err != nil {
						GinkgoWriter.Printf("Failed to cleanup workload node config file: %v\n", err)
					}
				}
			}

			nodes := config.LoadTestNodesFromEnv()
			var mgmtNodes []config.NodeInfo
			if len(nodes) >= 1 {
				mgmtNodes = []config.NodeInfo{nodes[0]}
			}
			CleanupManagementClusterUnified(
				clusterManager, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
				mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
			)
		})

		// 用例名称：当 worker 节点网络断开时应该成功创建 1Master1Worker 业务集群并保持 Healthy
		// 用例步骤：1) 使用引导集群/管理集群（mgmtKubeconfig）生成并在管理集群上创建业务集群（BC）；2) 在 worker 网络断开场景下等待 BC 达到 Healthy/Ready；3) 校验业务集群最终状态。
		// 预期结果：业务集群 `state` 为 Healthy 且 `clusterStatus` 为 Ready，即使 worker 网络断开也能完成就绪。
		It("当 worker 节点网络断开时应该成功创建 1Master1Worker 业务集群并保持 Healthy", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成业务集群配置文件")
			var err error
			workloadConfigPath, workloadNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(workloadClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成业务集群配置文件")

			By("在管理集群上创建业务集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(workloadConfigPath, workloadNodeConfigPath, mgmtKubeconfigPath)
			if err != nil {
				// 失败时尝试获取更多关于 BC 资源的调试信息
				namespace := ""
				getNsCmd := fmt.Sprintf("yq eval '.metadata.namespace' %s", workloadConfigPath)
				result, _ := sshExecutor.Exec(getNsCmd)
				if result.ExitCode == 0 {
					namespace = strings.TrimSpace(result.Stdout)
					bcName := "bke-" + workloadClusterName
					getBCCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bc %s -n %s -o yaml 2>&1", mgmtKubeconfigPath, bcName, namespace)
					result, _ := sshExecutor.Exec(getBCCmd)
					GinkgoWriter.Printf("[DEBUG] 应用失败后的 BC 资源状态: exit_code=%d, stdout=%s, stderr=%s\n", result.ExitCode, result.Stdout, result.Stderr)
				}
			}
			Expect(err).NotTo(HaveOccurred(), "业务集群创建命令应该成功执行")

			By("等待业务集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
				if err != nil {
					GinkgoWriter.Printf("获取业务集群完整状态失败: %v, 当前状态: %s\n", err, state)
					return false
				} else {
					GinkgoWriter.Printf("当前业务集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				}
				if state == "DeployFailed" {
					// 失败时获取完整的 BC 资源和事件用于调试
					bcName := "bke-" + workloadClusterName
					namespace := "bke-" + workloadClusterName
					getBCCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bc %s -n %s -o yaml 2>&1", mgmtKubeconfigPath, bcName, namespace)
					result, _ := sshExecutor.Exec(getBCCmd)
					if result.ExitCode == 0 {
						GinkgoWriter.Printf("[DEBUG] === DeployFailed 时的 BC 资源 ===\n%s\n", result.Stdout)
					}
					getEventsCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get events -n %s --sort-by='.lastTimestamp' 2>&1 | tail -20", mgmtKubeconfigPath, namespace)
					result, _ = sshExecutor.Exec(getEventsCmd)
					if result.ExitCode == 0 {
						GinkgoWriter.Printf("[DEBUG] === 最近的事件 ===\n%s\n", result.Stdout)
					}
					Fail(fmt.Sprintf("业务集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "即使 worker 节点网络断开，业务集群也应该变为 Ready 状态")
		})
	})

	Describe("创建 1Master 集群并使用无效的 Master 节点用户名密码", Label("1master", "invalid-credentials", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m-invalid-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "1Master 集群需要至少 1 个节点")

			// 第一个节点作为 Master，使用错误的凭据
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			// 设置错误的密码
			masterNode.Password = "invalid-password"

			// 创建带有错误 master 节点凭据的集群配置
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
		})

		AfterEach(func() {
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, ""); err != nil {
					GinkgoWriter.Printf("删除集群失败: %v\n", err)
				}
				Eventually(func() bool {
					return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
				}, uninstallTimeout, 10*time.Second).Should(BeTrue())
			}

			if configPath != "" {
				if err := clusterManager.CleanupConfig(configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
			time.Sleep(60 * time.Second) // 等待ssh的封禁结束，避免对后续测试造成影响
		})

		// 用例名称：使用无效的 master 节点凭据应该无法创建集群
		// 用例步骤：1) 修改 master 节点用户名/密码为无效值并生成错误凭据的集群配置；2) 通过引导节点生成并上传配置并校验配置存在；3) 执行 `bke cluster create`，等待集群状态进入 DeployFailed/InitializationFailed。
		// 预期结果：最终集群状态不为 Healthy，集群创建应失败且状态符合预期。
		It("使用无效的 master 节点凭据应该无法创建集群", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成带有无效 master 节点凭据的集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待并验证集群创建失败")
			// 等待集群创建失败，状态为 DeployFailed 和 InitializationFailed
			checkInterval := 30 * time.Second
			checkTimeout := 10 * time.Minute // 检查 10 分钟以验证失败
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)

				if clusterStatus == "Ready" {
					// 如果集群变为 Ready，说明没有被阻止 - 这是意外的
					Fail(fmt.Sprintf("集群意外变为 Ready。预期集群创建失败，但集群状态为: %s", clusterStatus))
				}

				// 检查状态是否为 DeployFailed 且 clusterStatus 为 InitializationFailed
				// 由于现在当 clusterStatus 长时间为 InitializationFailed，state也不会变为 DeployFailed，修改退出逻辑
				return state == "DeployFailed" || clusterStatus == "InitializationFailed"
			}, checkTimeout, checkInterval).Should(BeTrue(), "集群创建应该失败，状态为 DeployFailed 和 InitializationFailed")

			// 最终状态检查
			_, _, finalClusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			Expect(err).NotTo(HaveOccurred())
			Expect(finalClusterStatus).NotTo(Equal("Ready"), "当 master 节点凭据无效时，集群不应该是 Ready")
			GinkgoWriter.Printf("最终集群状态: %s (预期失败，不是 Ready)\n", finalClusterStatus)
		})
	})

	Describe("创建 1Master1Worker 集群并使用无效的 Worker 节点用户名密码", Label("1master1worker", "invalid-credentials", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m1w-invalid-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Worker 集群需要至少 2 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 第二个节点作为 Worker，使用错误的凭据
			workerNode := nodes[1]
			workerNode.Role = []string{"node"}
			// 设置错误的用户名和密码
			workerNode.Username = "invalid-user"
			workerNode.Password = "invalid-password"

			// 创建带有错误 worker 节点凭据的集群配置
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode, workerNode})
		})

		AfterEach(func() {
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, ""); err != nil {
					GinkgoWriter.Printf("Failed to delete cluster: %v\n", err)
				}
				Eventually(func() bool {
					return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
				}, uninstallTimeout, 10*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				if err := clusterManager.CleanupConfig(configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：即使使用无效的 worker 节点凭据也应该成功创建集群
		// 用例步骤：1) 在集群配置中将 worker 节点用户名/密码设置为无效值；2) 通过引导节点生成并上传配置并校验配置存在；3) 执行 `bke cluster create`，等待集群达到 Healthy/Ready。
		// 预期结果：即使 worker 凭据无效，集群仍能完成创建并保持 `state=Healthy`、`clusterStatus=Ready`。
		It("即使使用无效的 worker 节点凭据也应该成功创建集群", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成带有无效 worker 节点凭据的集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "集群应该变为 Ready 状态")
		})
	})

	Describe("创建 1Master 管理集群和 1Master1Worker 业务集群并使用无效的 Worker 节点用户名密码", Label("1m-mgmt-1m1w-workload", "invalid-credentials", "post-init", "online"), Ordered, func() {
		var (
			mgmtClusterName        string
			mgmtClusterConfig      *config.BKEClusterConfig
			mgmtConfigPath         string
			mgmtNodeConfigPath     string
			mgmtKubeconfigPath     string
			workloadClusterName    string
			workloadClusterConfig  *config.BKEClusterConfig
			workloadConfigPath     string
			workloadNodeConfigPath string
		)

		BeforeAll(func() {
			By("在引导集群上创建 1Master 管理集群")
			mgmtClusterName = fmt.Sprintf("test-mgmt-1m-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "1Master 管理集群需要至少 1 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 创建管理集群配置（管理集群需要 cluster-api 来管理业务集群）
			// 包含 cluster-api 但排除 openfuyao-system-controller（与证书测试用例相同）
			mgmtClusterConfig = config.NewBKEClusterConfigForMgmt(mgmtClusterName, []config.NodeInfo{masterNode})

			By("生成管理集群配置文件")
			var err error
			mgmtConfigPath, mgmtNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(mgmtClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			By("执行管理集群的 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(mgmtConfigPath, mgmtNodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "管理集群创建命令应该成功执行")

			By("等待管理集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(mgmtClusterName, "")
				GinkgoWriter.Printf("当前管理集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("管理集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "管理集群应该变为 Ready 状态")

			By("获取管理集群 kubeconfig")
			mgmtChecker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, mgmtClusterName, "")
			mgmtKubeconfigPath, err = mgmtChecker.SaveKubeconfigToFileWithKubeconfig("")
			Expect(err).NotTo(HaveOccurred(), "应该成功获取管理集群 kubeconfig")
			GinkgoWriter.Printf("管理集群 kubeconfig 保存到: %s\n", mgmtKubeconfigPath)
		})

		AfterAll(func() {
			By("清理业务集群")
			if workloadClusterName != "" && mgmtKubeconfigPath != "" {
				if clusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath) {
					if err := clusterManager.DeleteClusterWithKubeconfig(workloadClusterName, mgmtKubeconfigPath); err != nil {
						GinkgoWriter.Printf("删除业务集群失败: %v\n", err)
					}
					Eventually(func() bool {
						return !clusterManager.ClusterExistsWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
					}, uninstallTimeout, 10*time.Second).Should(BeTrue())
				}
				if workloadConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadConfigPath); err != nil {
						GinkgoWriter.Printf("清理业务集群配置文件失败: %v\n", err)
					}
				}
				if workloadNodeConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadNodeConfigPath); err != nil {
						GinkgoWriter.Printf("Failed to cleanup workload node config file: %v\n", err)
					}
				}
			}

			nodes := config.LoadTestNodesFromEnv()
			var mgmtNodes []config.NodeInfo
			if len(nodes) >= 1 {
				mgmtNodes = []config.NodeInfo{nodes[0]}
			}
			CleanupManagementClusterUnified(
				clusterManager, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
				mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
			)
		})

		// 用例名称：在管理集群上使用无效的 worker 节点凭据应该成功创建 1Master1Worker 业务集群
		// 用例步骤：1) 在业务集群配置中把 worker 凭据设置为无效；2) 生成并在引导节点侧完成配置上传/落地；3) 使用管理集群的 kubeconfig 在管理集群上创建 BC；4) 等待业务集群达到 Healthy/Ready。
		// 预期结果：即使业务集群 worker 使用无效凭据，业务集群也应仍能完成初始化并保持 `state=Healthy`、`clusterStatus=Ready`。
		It("在管理集群上使用无效的 worker 节点凭据应该成功创建 1Master1Worker 业务集群", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("创建带有无效 worker 节点凭据的 1Master1Worker 业务集群配置")
			workloadClusterName = fmt.Sprintf("test-workload-1m1w-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			// 管理集群使用 nodes[0]，所以业务集群需要至少 nodes[1] 和 nodes[2]
			Expect(len(nodes)).To(BeNumerically(">=", 3), "1Master1Worker 业务集群需要至少 3 个节点（管理集群使用节点 1，业务集群需要节点 2-3）")

			// 使用节点 2 作为 Master (nodes[1])
			masterNode := nodes[1]
			masterNode.Role = []string{"master/node", "etcd"}

			// 使用节点 3 作为 Worker，使用错误的凭据 (nodes[2])
			workerNode := nodes[2]
			workerNode.Role = []string{"node"}
			// 设置错误的用户名和密码
			workerNode.Username = "invalid-user"
			workerNode.Password = "invalid-password"

			// 创建业务集群配置
			workloadClusterConfig = config.NewDefaultBKEClusterConfig(workloadClusterName, []config.NodeInfo{masterNode, workerNode})

			By("生成业务集群配置文件")
			var err error
			workloadConfigPath, workloadNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(workloadClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成业务集群配置文件")

			By("在管理集群上创建业务集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(workloadConfigPath, workloadNodeConfigPath, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "业务集群创建命令应该成功执行")

			By("等待业务集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
				GinkgoWriter.Printf("当前业务集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("业务集群创建失败，状态为 DeployFailed。Phase: %s, clusterStatus: %s", phase, clusterStatus))
				}
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "业务集群应该变为 Ready 状态")
		})
	})

})
