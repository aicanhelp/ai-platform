package addons

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

var _ = Describe("Addon 节点标签与调度", Label("addons", "post-init", "online"), func() {
	var (
		sshExecutor    *executor.SSHExecutor
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")
		dynamicClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")
		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
	})

	Describe("节点标签功能", Label("node-labels", "post-init", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-labels-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "节点标签亲和性测试至少需要 2 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			masterNode.Labels = []config.NodeLabel{
				{Key: "test-label", Value: "e2e-test"},
				{Key: "environment", Value: "testing"},
			}

			// 第二个节点作为 Worker
			workerNode := nodes[1]
			workerNode.Role = []string{"node"}
			workerNode.Labels = []config.NodeLabel{
				{Key: "addon-node", Value: "redis"},
			}

			// 创建集群配置
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
					GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
				}
			}
			if nodeConfigPath != "" {
				if err := clusterManager.CleanupConfig(nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		})

		// 用例名称：应该成功创建集群并验证节点标签
		// 用例步骤：1) 在引导节点侧生成并上传带有 master/worker 节点 labels 的集群配置；2) 执行 `bke cluster create` 创建集群；3) 等待集群 Healthy；4) 通过集群检查器验证 master/worker 节点标签是否生效。
		// 预期结果：集群创建成功，master 节点包含 `test-label`/`environment`，worker 节点包含 `addon-node=redis`。
		// Author: leoliu600
		It("应该成功创建集群并验证节点标签", Label("required_nodes:2", "node-labels-basic"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("执行集群创建命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
					return ""
				}
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证节点标签")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 Master 节点的标签
			Eventually(func() bool {
				hasLabel, _ := checker.CheckNodeHasLabel("test-label", "e2e-test")
				return hasLabel
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "Master 节点应该有 test-label 标签")

			// 检查 Worker 节点的标签
			Eventually(func() bool {
				hasLabel, _ := checker.CheckNodeHasLabel("addon-node", "redis")
				return hasLabel
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "Worker 节点应该有 addon-node 标签")
		})

		// 用例名称：应该支持多标签并验证 addon 亲和性部署
		// 用例步骤：1) 在集群配置中为 redis addon 配置 NodeSelector（例如 `addon-node=redis`）；2) 通过引导节点创建集群并等待 Healthy；3) 获取 redis Pod 所在节点并校验其节点标签满足亲和性。
		// 预期结果：redis Pod 只调度到带有 `addon-node=redis` 标签的节点，且集群总体保持 Healthy。
		// Author: leoliu600
		It("应该支持多标签并验证 addon 亲和性部署", Label("required_nodes:2", "node-labels-addon-affinity"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			// 使用基础 addon 列表并添加 redis addon 配置 nodeSelector 亲和性
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "redis",
				Version: "6.2.12",
				Block:   false,
				NodeSelector: map[string]string{
					"addon-node": "redis",
				},
			})

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("执行集群创建命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
					return ""
				}
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证 redis addon Pod 运行在带有正确标签的节点上")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			Eventually(func() bool {
				nodeName, err := checker.GetPodNodeName("", "app.kubernetes.io/name=redis")
				if err != nil || nodeName == "" {
					return false
				}
				GinkgoWriter.Printf("redis Pod 运行在节点: %s\n", nodeName)

				// 验证该节点确实是带有 redis 标签的节点，只有 Worker 节点有这个标签
				hasLabel, _ := checker.CheckNodeHasLabel("addon-node", "redis")
				return hasLabel
			}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "redis 应该部署在带有 addon-node=redis 标签的节点上")
		})

		// 用例名称：应该支持 addon 未设置亲和性时的默认部署
		// 用例步骤：1) 在引导节点侧创建集群并安装 redis addon，但不设置 NodeSelector；2) 等待集群 Healthy；3) 通过检查器找到 redis Pod 节点，验证 Pod 可以正常部署到任意节点。
		// 预期结果：redis Pod 正常存在且运行节点非空，说明未设置亲和性时走默认调度策略。
		// Author: leoliu600
		It("应该支持 addon 未设置亲和性时的默认部署", Label("required_nodes:2", "node-labels-no-affinity"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			// 添加 redis 但不设置 NodeSelector
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "redis",
				Version: "6.2.12",
			})

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("执行集群创建命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
					return ""
				}
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证 redis addon Pod 正常部署")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			Eventually(func() bool {
				nodeName, _ := checker.GetPodNodeName("", "app.kubernetes.io/name=redis")
				return nodeName != ""
			}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "未设置亲和性时，redis 应该能正常部署在任意节点")
		})

		// 用例名称：应该在 addon 标签不匹配时不部署至节点
		// 用例步骤：1) 配置 redis addon 的 NodeSelector 为不存在的标签键值；2) 通过引导节点创建集群并等待 Healthy；3) 查询 redis Pod 所在节点，断言不存在调度（nodeName 为空）。
		// 预期结果：当 addon 亲和性标签不匹配时，redis Pod 不会被调度到任何节点，集群仍保持 Healthy。
		// Author: leoliu600
		It("应该在 addon 标签不匹配时不部署至节点", Label("required_nodes:2", "node-labels-mismatch"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			// 添加 redis 并设置一个不存在的标签亲和性
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "redis",
				Version: "6.2.12",
				NodeSelector: map[string]string{
					"non-existent-key": "some-value",
				},
			})

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("执行集群创建命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
					return ""
				}
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证 redis addon Pod 未被调度")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			Consistently(func() bool {
				nodeName, _ := checker.GetPodNodeName("", "app.kubernetes.io/name=redis")
				return nodeName == ""
			}, 1*time.Minute, 10*time.Second).Should(BeTrue(), "标签不匹配时，redis Pod 不应该被调度到任何节点")
		})

		// 用例名称：应该忽略包含非法字符的节点标签
		// 用例步骤：1) 在集群配置中向节点添加包含非法字符的 label（例如 `invalid-label` / `!!!#@$`）；2) 通过引导节点创建集群；3) 等待集群 Healthy；4) 检查非法标签是否没有被应用到节点。
		// 预期结果：非法字符的节点标签不会出现在节点标签集合中，集群仍可正常就绪。
		// Author: luolong817
		It("应该忽略包含非法字符的节点标签", Label("required_nodes:2", "node-labels-invalid"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			// 为 Master 节点添加非法标签 (例如包含空格或特殊字符)
			clusterConfig.Nodes[1].Labels = append(clusterConfig.Nodes[1].Labels,
				config.NodeLabel{Key: "invalid-label", Value: "!!!#@$"},
			)

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			GinkgoWriter.Printf("生成的集群配置文件: %s, 节点配置文件: %s\n", configPath, nodeConfigPath)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("执行集群创建命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				if err != nil {
					GinkgoWriter.Printf("获取集群状态失败: %v\n", err)
					return ""
				}
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证非法标签未被应用")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			Eventually(func() bool {
				hasLabel, _ := checker.CheckNodeHasLabel("invalid-label", "!!!#@$")
				return hasLabel
			}, 1*time.Minute, 10*time.Second).Should(BeFalse(), "非法字符的标签不应该存在于节点中")
		})
	})
})
