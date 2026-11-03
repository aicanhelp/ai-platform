package installation

import (
	"fmt"
	"os"
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

// ==================================================================================
// BKE 前置脚本 E2E 测试
//
// 基于《前置后置处理框架设计》文档实现，核心机制：
//   - user-system 命名空间下创建 n 个脚本 ConfigMap（label: bke.preprocess.script="true"）
//   - user-system 命名空间下创建 1 个全局配置 ConfigMap（name: preprocess-all-config）
//   - 集群创建/扩容时 BKEAgent 自动拉取脚本并在节点上执行
//   - 脚本落盘路径: /etc/openFuyao/bkeagent/scripts
//   - 参数通过 ${PARAM_NAME} 模板渲染，NODE_IP 自动注入
//
// 运行方式:
//   ginkgo -v --label-filter="prescript" ./e2e/installation/...
//   ginkgo -v --label-filter="prescript && P0" ./e2e/installation/...
// ==================================================================================

var _ = SIGDescribe("BKE Pre-Script Installation", func() {
	var (
		sshExecutor      *executor.SSHExecutor
		localExecutor    *executor.LocalExecutor
		guideConfig      *config.GuideNodeConfig
		dynamicClient    dynamic.Interface
		clusterManager   *utils.ClusterManager
		preScriptManager *utils.PreScriptManager
	)

	BeforeEach(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 环境变量必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 环境变量必须设置")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host, guideConfig.Port,
			guideConfig.Username, guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "应该成功连接到引导节点")

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)
		preScriptManager = utils.NewPreScriptManager(sshExecutor, localExecutor)
	})

	// ==================== P0 测试用例 ====================

	// 用例1: 单master在线安装1个脚本1个全局参数
	Describe("单master在线安装1个脚本1个全局参数", Label("prescript", "1master", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-1m1s-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要 1 个测试节点")

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("在引导集群创建 user-system 命名空间 + 脚本 ConfigMap + 全局配置")
			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred(), "应该成功部署前置脚本资源")
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			preScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool {
					return !clusterManager.ClusterExists(clusterName)
				}, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("应该成功创建集群并在节点上部署和执行前置脚本", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			By("检查日志中是否有前置处理的日志（非阻塞）")
			found, log, _ := preScriptManager.VerifyPreProcessingLogs(clusterName, "")
			if found {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", log)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			By("验证 /etc/openFuyao/bkeagent/scripts 中有对应的脚本")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptLanded(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证前置脚本在节点上正确执行")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// 用例2: 单master在线安装多个脚本1个全局参数
	Describe("单master在线安装多个脚本1个全局参数", Label("prescript", "1master", "multi-script", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh", "setup-env.sh", "install-agent.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-1mms-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署 3 个脚本 + 全局配置")
			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{
					utils.TestScriptInitOS(),
					utils.TestScriptSetupEnv(),
					utils.TestScriptInstallAgent(),
				},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "setup-env.sh", Order: 20, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "install-agent.sh", Order: 30, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			preScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("应该成功创建集群并部署执行所有前置脚本", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			By("检查日志（非阻塞）")
			if f, l, _ := preScriptManager.VerifyPreProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			By("验证所有脚本落盘")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptLanded(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证所有脚本执行")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// 用例3: 多master(3M)在线安装1个脚本1个全局参数
	Describe("多master(3M)在线安装1个脚本1个全局参数", Label("prescript", "3master", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-3m1s-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "至少需要 3 个测试节点")

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				preScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
			}
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("应该在所有3个master节点上部署和执行前置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			if f, l, _ := preScriptManager.VerifyPreProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})
	})

	// 用例4: 多master(3M)在线安装多个脚本1个全局参数
	Describe("多master(3M)在线安装多个脚本1个全局参数", Label("prescript", "3master", "multi-script", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"init-os.sh", "setup-env.sh", "install-agent.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-3mms-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3))

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{
					utils.TestScriptInitOS(),
					utils.TestScriptSetupEnv(),
					utils.TestScriptInstallAgent(),
				},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "setup-env.sh", Order: 20, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "install-agent.sh", Order: 30, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				preScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
			}
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("应该在所有3个master节点上部署和执行所有前置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			if f, l, _ := preScriptManager.VerifyPreProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})
	})

	// ==================== P1 测试用例 ====================

	// 用例5: 单master在线安装1脚本1全局参数，参数校验失败
	Describe("单master在线安装1脚本1全局参数-参数校验失败", Label("prescript", "1master", "param-validation", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-1mpf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署脚本 + 包含非法参数的全局配置（分号不在白名单内）")
			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{
						"HTTP_REPO": "http://repo.example.com;rm -rf /",
					}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			preScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("脚本参数校验不通过时脚本无法正常执行", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群安装完成（集群本身应该可以正常创建）")
			Eventually(func() bool {
				_, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: state=%s, clusterStatus=%s\n", state, cs)
				return cs == "Ready" || state == "DeployFailed"
			}, installTimeout, pollInterval).Should(BeTrue())

			By("验证脚本标记文件不存在（参数校验失败导致脚本被跳过）")
			notExecuted, err := preScriptManager.VerifyScriptNotExecuted(
				masterNode.IP, masterNode.Username, masterNode.Password, "init-os.sh",
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(notExecuted).To(BeTrue(), "参数校验失败时脚本不应该成功执行")
		})
	})

	// 用例6: 单master在线安装1脚本1全局参数，脚本安装失败不阻塞集群
	Describe("单master在线安装1脚本1全局参数-脚本失败不阻塞集群", Label("prescript", "1master", "script-failure", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"fail-script.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-1msf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署一个会故意失败的脚本（exit 1）")
			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptFail()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "fail-script.sh", Order: 10, Params: map[string]string{"PKG_VERSION": "99.99.99"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			preScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("脚本安装失败但集群应该正常创建", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready（脚本失败不应阻塞集群创建）")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"), "即使脚本失败，集群也应该正常创建")
		})
	})

	// 用例7: 多集群(3M管理集群)在线安装1脚本1全局参数，脚本安装失败不阻塞集群
	Describe("多集群(3M)在线安装1脚本1全局参数-脚本失败不阻塞集群", Label("prescript", "3master", "script-failure", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"fail-script.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-3msf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3))

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptFail()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "fail-script.sh", Order: 10, Params: map[string]string{"PKG_VERSION": "99.99.99"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				preScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
			}
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("脚本安装失败但3master管理集群应该正常创建", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready（脚本失败不应阻塞集群创建）")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"), "即使脚本失败，集群也应该正常创建")
		})
	})

	// 用例10: 扩容时执行前置脚本安装
	Describe("扩容时执行前置脚本安装", Label("prescript", "scale-out", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scaleOutNode   config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("pre-scale-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "扩容测试至少需要 2 个节点")

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			scaleOutNode = nodes[1]
			scaleOutNode.Role = []string{"node"}

			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())

			By("创建初始 1Master 集群")
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
			var createErr error
			configPath, nodeConfigPath, createErr = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(createErr).NotTo(HaveOccurred())
			createErr = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(createErr).NotTo(HaveOccurred())

			By("等待初始集群变为 Ready")
			Eventually(func() string {
				_, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))
		})

		AfterEach(func() {
			preScriptManager.Cleanup(scriptNames, "")
			preScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			preScriptManager.CleanupScriptsOnNode(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password)
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})

		It("扩容节点时应该在新节点上执行前置脚本", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("验证 Master 节点上脚本已执行")
			Eventually(func() bool {
				ok, _, _ := preScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("执行扩容操作")
			err := clusterManager.ScaleOutNode(clusterName, scaleOutNode)
			Expect(err).NotTo(HaveOccurred())

			By("等待扩容后集群恢复 Ready")
			Eventually(func() bool {
				_, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("扩容后: state=%s, clusterStatus=%s\n", state, cs)
				return cs == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue())

			By("验证扩容节点上脚本已落盘")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptLanded(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password, scriptNames)
				GinkgoWriter.Printf("扩容节点落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证扩容节点上脚本已执行")
			Eventually(func() bool {
				ok, detail, _ := preScriptManager.VerifyScriptExecuted(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password, scriptNames)
				GinkgoWriter.Printf("扩容节点执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// ==================== P2 测试用例 ====================

	// 用例8: 管理集群多master在线安装多个脚本创建业务集群
	Describe("管理集群多master在线安装多个脚本创建业务集群", Label("prescript", "3master", "mgmt-workload", "P2", "post-init", "skip-temporarily", "online"), Ordered, func() {
		var (
			mgmtClusterName        string
			mgmtConfigPath         string
			mgmtNodeConfigPath     string
			mgmtKubeconfigPath     string
			workloadClusterName    string
			workloadConfigPath     string
			workloadNodeConfigPath string
			workloadNodes          []config.NodeInfo
			scriptNames            = []string{"init-os.sh", "setup-env.sh"}
		)

		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			if len(nodes) < 8 {
				Skip(fmt.Sprintf("跳过: 管理集群+业务集群场景需要至少 8 个节点，当前只有 %d 个", len(nodes)))
			}

			By("创建管理集群 (3Master1Worker)")
			mgmtCluster, err := Create3M1WManagementCluster(clusterManager, localExecutor, "pre-mgmt", true)
			if mgmtCluster != nil {
				mgmtClusterName = mgmtCluster.ClusterName
				mgmtConfigPath = mgmtCluster.ConfigPath
				mgmtNodeConfigPath = mgmtCluster.NodeConfigPath
				mgmtKubeconfigPath = mgmtCluster.KubeconfigPath
			}
			Expect(err).NotTo(HaveOccurred())

			By("在管理集群上部署前置脚本资源")
			err = preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS(), utils.TestScriptSetupEnv()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "setup-env.sh", Order: 20, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				mgmtKubeconfigPath,
			)
			Expect(err).NotTo(HaveOccurred())

			workloadNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				workloadNodes[i] = nodes[4+i]
				workloadNodes[i].Role = []string{"master/node", "etcd"}
			}
			workloadClusterName = fmt.Sprintf("pre-wl-%d", time.Now().Unix())
		})

		It("应该在业务集群所有节点上部署和执行前置脚本", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			workloadConfig := config.NewDefaultBKEClusterConfig(workloadClusterName, workloadNodes)
			vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv()
			if vipHost != "" {
				workloadConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			workloadConfigPath, workloadNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(workloadConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(workloadConfigPath, workloadNodeConfigPath, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred())

			By("等待业务集群状态变为 Ready")
			Eventually(func() bool {
				phase, state, cs, _ := clusterManager.GetClusterFullStatusWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
				GinkgoWriter.Printf("业务集群: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				if state == "DeployFailed" {
					Fail(fmt.Sprintf("业务集群创建失败: phase=%s", phase))
				}
				return cs == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue())

			By("检查日志（非阻塞）")
			if f, l, _ := preScriptManager.VerifyPreProcessingLogs(workloadClusterName, mgmtKubeconfigPath); f {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			for i, node := range workloadNodes {
				By(fmt.Sprintf("验证业务集群 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})

		AfterAll(func() {
			if mgmtKubeconfigPath != "" {
				preScriptManager.Cleanup(scriptNames, mgmtKubeconfigPath)
			}
			for _, n := range workloadNodes {
				preScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
			}
			if workloadClusterName != "" && mgmtKubeconfigPath != "" {
				Delete3M1WWorkloadCluster(clusterManager, localExecutor, workloadClusterName, mgmtKubeconfigPath)
			}
			if workloadConfigPath != "" {
				clusterManager.CleanupConfig(workloadConfigPath)
			}
			if workloadNodeConfigPath != "" {
				clusterManager.CleanupConfig(workloadNodeConfigPath)
			}
			if mgmtClusterName != "" {
				Delete3M1WManagementCluster(clusterManager, localExecutor, mgmtClusterName, mgmtKubeconfigPath)
			}
			if mgmtConfigPath != "" {
				clusterManager.CleanupConfig(mgmtConfigPath)
			}
			if mgmtNodeConfigPath != "" {
				clusterManager.CleanupConfig(mgmtNodeConfigPath)
			}
		})
	})

	// 用例9: 离线多master在线安装多个脚本创建业务集群
	Describe("离线多master安装多个脚本创建业务集群", Label("prescript", "3master", "offline", "P2", "post-init"), Ordered, func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"init-os.sh", "setup-env.sh"}
		)

		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3))

			clusterName = fmt.Sprintf("pre-off3m-%d", time.Now().Unix())
			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := preScriptManager.SetupPreScripts(
				[]utils.ScriptContentDef{utils.TestScriptInitOS(), utils.TestScriptSetupEnv()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "setup-env.sh", Order: 20, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		It("离线模式下应该在所有节点上部署和执行前置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			imageRepoDomain := os.Getenv("IMAGE_REPO_DOMAIN")
			imageRepoIP := os.Getenv("IMAGE_REPO_IP")

			var clusterConfig *config.BKEClusterConfig
			if imageRepoDomain != "" {
				clusterConfig = config.NewBKEClusterConfigOffline(clusterName, masterNodes, imageRepoDomain, imageRepoIP)
			} else {
				GinkgoWriter.Printf("未配置离线镜像仓库，使用在线模式\n")
				clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			}
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			if f, l, _ := preScriptManager.VerifyPreProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("前置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到前置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := preScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})

		AfterAll(func() {
			preScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				preScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
			}
			if clusterManager.ClusterExists(clusterName) {
				clusterManager.DeleteCluster(clusterName)
				Eventually(func() bool { return !clusterManager.ClusterExists(clusterName) }, uninstallTimeout, 60*time.Second).Should(BeTrue())
			}
			if configPath != "" {
				clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				clusterManager.CleanupConfig(nodeConfigPath)
			}
		})
	})
})
