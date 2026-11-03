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

// 诊断与处理相关的扩容异常用例
var _ = Describe("Diagnosis and Handle - ScaleOut Abnormal Nodes", Ordered, func() {
	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
	)

	BeforeAll(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())
		Expect(guideConfig.Password).NotTo(BeEmpty())

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred())

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)

		localExecutor = executor.NewLocalExecutor(30 * time.Second)
	})

	// 扩容时新节点异常 -> 不应影响已有集群的 Healthy
	Describe("扩容异常节点", Label("diagnosis_handle", "scaleout-abnormal", "post-init", "online"), Ordered, func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeAll(func() {
			clusterName = fmt.Sprintf("test-diagnosis-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "需要至少两个节点用于扩容测试")

			master := nodes[0]
			master.Role = []string{"master/node", "etcd"}

			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{master})

			// 节点选择将在每个 It 中进行，以便 BeforeAll 只做创建集群的必要操作

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("[BeforeAll] 当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))
		})

		AfterAll(func() {
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				_ = clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				Eventually(func() bool {
					return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
				}, uninstallTimeout, pollInterval).Should(BeTrue())
			}
			if configPath != "" {
				_ = clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				_ = clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		// 场景1：扩容到一个凭据错误的节点 -> 该节点不应加入集群，集群应保持 Healthy
		It("扩容节点密码错误", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			By("scale-out to a node with invalid credentials")
			nodes := config.LoadTestNodesFromEnv()
			invalidNode := nodes[1]
			invalidNode.Role = []string{"node"}
			invalidNode.Username = "invalid-user"
			invalidNode.Password = "invalid-pass"

			err := clusterManager.ScaleOutNode(clusterName, invalidNode)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() bool {
				c, _ := checker.GetNodeCountWithKubeconfig("")
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				bnState, _ := clusterManager.GetBKENodeState(clusterName, invalidNode.IP)
				GinkgoWriter.Printf("[cred-it] 当前集群状态: phase=%s, state=%s, clusterStatus=%s, nodeCount=%d, bnState=%s\n", phase, state, clusterStatus, c, bnState)
				return c == 1 && clusterStatus == "Ready" && bnState == "InitFailed"
			}, installTimeout, pollInterval).Should(BeTrue(), "凭据错误的节点不应被加入集群，且 BKENode 状态应为 InitFailed")

			// 使用现有的 ScaleInNode 函数缩容 worker 节点
			By(fmt.Sprintf("缩容 worker 节点: %s", invalidNode.IP))
			err = clusterManager.ScaleInNode(clusterName, invalidNode.IP)
			Expect(err).NotTo(HaveOccurred(), "应该成功缩容 worker 节点")

			By("等待缩容后集群变为 Ready")
			Eventually(func() bool {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群完整状态失败: %v\n", err)
					return false
				}
				GinkgoWriter.Printf("缩容后当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "缩容后集群应该变为 Ready")

			By("等待 1 分钟，确保使用错误密码连接节点后 SSH 恢复可达")
			// 使用错误密码连接节点会触发 SSH 连接失败，需要等待一段时间让 SSH 服务恢复可达
			time.Sleep(1 * time.Minute)
		})

		// 场景2：扩容到一个网络断开的节点 -> 该节点不应加入集群，集群应保持 Healthy
		It("扩容节点断网", Label("required_nodes:2", "network-disconnect"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			By("scale-out to a network-disconnected node")
			nodes := config.LoadTestNodesFromEnv()
			netNode := nodes[1]
			netNode.Role = []string{"node"}
			netNode.Hostname = "w1"
			var err error
			// 断网并记录备份
			backup, err := DisconnectNetwork(netNode, localExecutor)
			Expect(err).NotTo(HaveOccurred())

			// 无论后续步骤是否失败，都会恢复网络
			DeferCleanup(func() error {
				if backup != "" {
					err := RestoreNetwork(netNode, localExecutor, backup)
					if err != nil {
						GinkgoWriter.Printf("恢复节点网络失败: %v\n", err)
					}
					time.Sleep(5 * time.Second)
				}
				return nil
			})

			err = clusterManager.ScaleOutNode(clusterName, netNode)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() bool {
				c, _ := checker.GetNodeCountWithKubeconfig("")
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s, nodeCount=%d\n", phase, state, clusterStatus, c)
				return c == 1 && strings.Contains(clusterStatus, "Ready")
			}, installTimeout, pollInterval).Should(BeTrue(), "断网节点不应被加入集群")
		})

	})

	// 使用2个无效的 Worker 节点凭据创建集群 -> 集群应该成功创建，Worker 节点不应加入集群
	Describe("创建 1Master2Worker 集群并使用无效的 Worker 节点用户名密码", Label("1master2worker", "post-init", "invalid-credentials", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m2w-invalid-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "1Master2Worker 集群需要至少 3 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 第二个和第三个节点作为 Worker，使用错误的凭据
			workerNode1 := nodes[1]
			workerNode1.Role = []string{"node"}
			workerNode1.Username = "invalid-user"
			workerNode1.Password = "invalid-password"

			workerNode2 := nodes[2]
			workerNode2.Role = []string{"node"}
			workerNode2.Username = "invalid-user"
			workerNode2.Password = "invalid-password"

			// 创建带有两个错误 worker 节点凭据的集群配置 (1 master + 2 workers)
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode, workerNode1, workerNode2})
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
		})

		It("即使使用无效的 worker 节点凭据也应该成功创建集群", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
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

})
