package installation

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

var _ = SIGDescribe("BKE Cluster Installation Basic", func() {
	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

		localExecutor = executor.NewLocalExecutor(10 * time.Minute)

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
	})

	Describe("创建 1Master1Node 集群", Label("1master1node", "for-test", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m1n-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "1Master1Node 集群需要至少 2 个节点")

			nodes[0].Role = []string{"master/node", "etcd"}
			nodes[1].Role = []string{"node"}

			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, nodes[:2])
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
					GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：应该成功创建 1Master1Node 集群
		// 用例步骤：1) 通过引导节点生成集群与节点配置文件；2) 在引导节点侧触发 `bke cluster create` 创建集群；3) 轮询集群状态直到达到 Healthy。
		// 预期结果：集群创建完成后 `state` 为 Healthy，且最终状态查询结果与预期一致。
		It("应该成功创建 1Master1Node 集群", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))
		})
	})

	Describe("集群扩缩容", Label("scale", "for-test", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			nodes          []config.NodeInfo
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-scale-%d", time.Now().Unix())

			nodes = config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "扩缩容测试需要至少 2 个节点")

			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("创建1master集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))
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
					GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：应该成功扩缩容节点
		// 用例步骤：1) 通过引导节点先创建 1master 基础集群；2) 执行扩容将新节点加入集群，并验证新节点 Ready；3) 执行缩容移除节点，并验证节点数量与 Ready 状态变化。
		// 预期结果：扩容后节点可用且集群保持就绪，缩容后节点数量正确且处于就绪状态。
		It("应该成功扩缩容节点", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			newNode := nodes[1]
			newNode.Role = []string{"node"}

			By("执行扩容命令")
			err := clusterManager.ScaleOutNodeWithKubeconfig(clusterName, newNode, "")
			Expect(err).NotTo(HaveOccurred())

			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			By("验证新节点 Ready")
			Eventually(func() bool {
				count, _ := checker.GetReadyNodeCountWithKubeconfig("")
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s, nodeCount=%d\n", phase, state, clusterStatus, count)
				return count == 2 && clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue())

			By("执行缩容命令")
			err = clusterManager.ScaleInNodeWithKubeconfig(clusterName, newNode.IP, "")
			Expect(err).NotTo(HaveOccurred())

			By("验证节点数量减少到 1")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				count, _ := checker.GetNodeCountWithKubeconfig("")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s, nodeCount=%d\n", phase, state, clusterStatus, count)
				return count == 1 && clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue())
		})
	})

	Describe("创建 3Master 高可用集群", Label("3master", "ha", "for-test", "with-management-cluster", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-3m-ha-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "3Master 高可用集群需要至少 3 个节点")

			nodes[0].Role = []string{"master/node", "etcd"}
			nodes[1].Role = []string{"master/node", "etcd"}
			nodes[2].Role = []string{"master/node", "etcd"}

			vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv()
			Expect(vipHost).NotTo(BeEmpty(), "WORKLOAD_CLUSTER_VIP_HOST 环境变量必须设置")

			clusterConfig = config.NewBKEClusterConfigForHA(clusterName, nodes[:3], vipHost, vipPort)
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
					GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：应该成功创建 3Master 高可用集群
		// 用例步骤：1) 通过引导节点生成集群与节点配置文件（包含 VIP）；2) 在引导节点侧触发 `bke cluster create` 创建集群；3) 轮询集群状态直到达到 Healthy。
		// 预期结果：集群创建完成后 `state` 为 Healthy，且最终状态查询结果与预期一致。
		It("应该成功创建 3Master 高可用集群", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))
		})
	})

	Describe("创建 3Master 高可用管理集群", Label("3master-mgmt", "ha", "management", "for-test", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			mgmtNodes      []config.NodeInfo
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-3m-mgmt-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "3Master 高可用管理集群需要至少 3 个节点")

			mgmtNodes = make([]config.NodeInfo, 3)
			copy(mgmtNodes, nodes[:3])
			for i := 0; i < 3; i++ {
				mgmtNodes[i].Role = []string{"master/node", "etcd"}
			}

			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			Expect(vipHost).NotTo(BeEmpty(), "MGMT_CLUSTER_VIP_HOST 环境变量必须设置")

			clusterConfig = config.NewBKEClusterConfigForHA(clusterName, mgmtNodes, vipHost, vipPort)
			clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
		})

		AfterEach(func() {
			CleanupManagementClusterUnified(
				clusterManager, localExecutor, clusterName, "",
				mgmtNodes, configPath, nodeConfigPath,
			)
		})

		// 用例名称：应该成功创建 3Master 高可用管理集群
		// 用例步骤：1) 通过引导节点生成集群与节点配置文件（包含 VIP 和 cluster-api）；2) 在引导节点侧触发 `bke cluster create` 创建管理集群；3) 轮询集群状态直到达到 Healthy。
		// 预期结果：管理集群创建完成后 `state` 为 Healthy，且最终状态查询结果与预期一致。
		It("应该成功创建 3Master 高可用管理集群", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("创建管理集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			By("等待并验证管理集群健康（引导节点 + 管理集群自身）")
			_, err = WaitAndVerifyManagementCluster(clusterManager, clusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")
		})
	})
})
