package installation

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"path/filepath"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var _ = Describe("Management Cluster Installation", func() {
	var (
		sshExecutor    *executor.SSHExecutor
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

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
	})

	Describe("创建 1Master 管理集群", Label("mgmt-1master", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("mgmt-1m-%d", time.Now().Unix())

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置一个测试节点")

			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			clusterConfig = config.NewBKEClusterConfigWithAllAddons(clusterName, []config.NodeInfo{masterNode})
		})

		AfterEach(func() {
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				By("触发集群销毁")
				err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				By("等待集群完全删除")
				Eventually(func() bool {
					return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
				}, uninstallTimeout, 30*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}
		})

		// 用例名称：应该成功创建包含 Cluster API 的管理集群
		// 用例步骤：1) 通过引导节点准备 1Master 管理集群的配置文件并包含 Cluster API addon；2) 执行 `bke cluster create` 创建管理集群；3) 轮询集群状态直到 Healthy。
		// 预期结果：管理集群成功创建，集群 `state` 为 Healthy，且完成后集群处于可用状态。
		It("应该成功创建包含 Cluster API 的管理集群", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "生成配置文件应该成功")
			GinkgoWriter.Printf("配置文件已生成: %s\n", configPath)

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "执行创建命令应该成功")

			By("等待并验证管理集群健康（引导节点 + 管理集群自身）")
			_, err = WaitAndVerifyManagementCluster(clusterManager, clusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")

			By("验证管理集群创建成功")
		})
	})

	Describe("创建 3Master 管理集群", Label("mgmt-3master", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("mgmt-3m-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "至少需要配置三个测试节点")

			masterNodes := []config.NodeInfo{}
			for i := 0; i < 3; i++ {
				node := nodes[i]
				node.Role = []string{"master/node", "etcd"}
				masterNodes = append(masterNodes, node)
			}

			clusterConfig = config.NewBKEClusterConfigWithAllAddons(clusterName, masterNodes)
		})

		AfterEach(func() {
			By("清理测试集群")
			if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
				By("触发集群销毁")
				err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				By("等待集群完全删除")
				Eventually(func() bool {
					return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
				}, uninstallTimeout, 30*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}
		})

		// 用例名称：应该成功创建 3Master 管理集群
		// 用例步骤：1) 在引导节点侧准备 3master 管理集群配置（3 个 master 都包含 master/node 与 etcd 角色）；2) 执行 `bke cluster create` 创建管理集群；3) 轮询集群状态直到 Healthy。
		// 预期结果：3master 管理集群创建成功，集群 `state` 为 Healthy，且管理集群面向业务/集群管理能力正常就绪。
		It("应该成功创建 3Master 管理集群", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "生成配置文件应该成功")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "执行创建命令应该成功")

			By("等待并验证管理集群健康（引导节点 + 管理集群自身）")
			_, err = WaitAndVerifyManagementCluster(clusterManager, clusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")
		})
	})
})
