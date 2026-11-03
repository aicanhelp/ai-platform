package installation

import (
	"fmt"
	"os"
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

// ==================================================================================
// BKE 后置脚本 E2E 测试
//
// 后置处理与前置处理机制对称，核心差异：
//   - ConfigMap label: bke.postprocess.script="true"
//   - 全局配置 ConfigMap name: postprocess-all-config
//   - 脚本落盘路径: /etc/openFuyao/bkeagent/scripts/postprocess
//   - 脚本在集群创建/扩容完成后执行
//
// 运行方式:
//   ginkgo -v --label-filter="postscript" ./e2e/installation/...
//   ginkgo -v --label-filter="postscript && P0" ./e2e/installation/...
// ==================================================================================

var _ = SIGDescribe("BKE Post-Script Installation", func() {
	var (
		sshExecutor       *executor.SSHExecutor
		localExecutor     *executor.LocalExecutor
		guideConfig       *config.GuideNodeConfig
		dynamicClient     dynamic.Interface
		clusterManager    *utils.ClusterManager
		postScriptManager *utils.PostScriptManager
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
		postScriptManager = utils.NewPostScriptManager(sshExecutor, localExecutor)
	})

	// ==================== P0 测试用例 ====================

	// 用例1: 单master业务集群后置处理1个脚本1个全局参数
	Describe("单master业务集群后置处理1个脚本1个全局参数", Label("postscript", "1master", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-1m1s-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要 1 个测试节点")

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("在引导集群创建 user-system 命名空间 + 后置脚本 ConfigMap + 全局配置")
			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred(), "应该成功部署后置脚本资源")
		})

		AfterEach(func() {
			postScriptManager.Cleanup(scriptNames, "")
			postScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
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

		It("应该成功创建集群并在节点上部署和执行后置脚本", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			By("生成集群配置文件")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			By("检查日志中是否有后置处理的日志（非阻塞）")
			found, log, _ := postScriptManager.VerifyPostProcessingLogs(clusterName, "")
			if found {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", log)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			By("验证 /etc/openFuyao/bkeagent/scripts/postprocess 中有对应的脚本")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptLanded(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证后置脚本在节点上正确执行")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// 用例2: 单master业务集群后置处理多个脚本1个全局参数
	Describe("单master业务集群后置处理多个脚本1个全局参数", Label("postscript", "1master", "multi-script", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh", "setup-env.sh", "install-agent.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-1mms-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署 3 个后置脚本 + 全局配置")
			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{
					utils.PostTestScriptInitOS(),
					utils.PostTestScriptSetupEnv(),
					utils.PostTestScriptInstallAgent(),
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
			postScriptManager.Cleanup(scriptNames, "")
			postScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
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

		It("应该成功创建集群并部署执行所有后置脚本", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			By("检查日志（非阻塞）")
			if f, l, _ := postScriptManager.VerifyPostProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			By("验证所有脚本落盘")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptLanded(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证所有脚本执行")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
				GinkgoWriter.Printf("脚本执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// 用例3: 多master(3M)业务集群后置处理1个脚本1个全局参数
	Describe("多master(3M)业务集群后置处理1个脚本1个全局参数", Label("postscript", "3master", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-3m1s-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "至少需要 3 个测试节点")

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			postScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				postScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
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

		It("应该在所有3个master节点上部署和执行后置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
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

			if f, l, _ := postScriptManager.VerifyPostProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})
	})

	// 用例4: 多master(3M)业务集群后置处理多个脚本1个全局参数
	Describe("多master(3M)业务集群后置处理多个脚本1个全局参数", Label("postscript", "3master", "multi-script", "P0", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"init-os.sh", "setup-env.sh", "install-agent.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-3mms-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3))

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{
					utils.PostTestScriptInitOS(),
					utils.PostTestScriptSetupEnv(),
					utils.PostTestScriptInstallAgent(),
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
			postScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				postScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
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

		It("应该在所有3个master节点上部署和执行所有后置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			if f, l, _ := postScriptManager.VerifyPostProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})
	})

	// ==================== P1 测试用例 ====================

	// 用例5: 后置处理参数校验失败
	Describe("单master业务集群后置处理-参数校验失败", Label("postscript", "1master", "param-validation", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-1mpf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署后置脚本 + 包含非法参数的全局配置（分号不在白名单内）")
			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS()},
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
			postScriptManager.Cleanup(scriptNames, "")
			postScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
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

		It("后置脚本参数校验不通过时脚本无法正常执行", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("检查节点是否干净（无残留集群）")
			bcListResult, _ := sshExecutor.Exec("kubectl get bc -A --no-headers 2>/dev/null | head -10")
			if bcListResult != nil && bcListResult.Stdout != "" {
				GinkgoWriter.Printf("当前集群列表:\n%s\n", bcListResult.Stdout)
			}

			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			By("后台创建集群")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群进入 EnsureNodesPostProcess 或最终状态")
			Eventually(func() bool {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				return cs == "Ready" || state == "DeployFailed" || phase == "EnsureNodesPostProcess"
			}, installTimeout, pollInterval).Should(BeTrue())

			By("验证后置脚本标记文件不存在（参数校验失败导致脚本被跳过）")
			notExecuted, err := postScriptManager.VerifyScriptNotExecuted(
				masterNode.IP, masterNode.Username, masterNode.Password, "init-os.sh",
			)
			Expect(err).NotTo(HaveOccurred())
			Expect(notExecuted).To(BeTrue(), "参数校验失败时后置脚本不应该成功执行")
		})
	})

	// 用例6: 后置脚本安装失败不阻塞集群（单master）
	// 注意: 后置脚本失败可能导致 bke cluster create 命令阻塞不返回，
	// 因此使用 nohup 后台执行，然后独立轮询集群状态。
	Describe("单master业务集群后置处理-脚本失败不阻塞集群", Label("postscript", "1master", "script-failure", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scriptNames    = []string{"fail-script.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-1msf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1))

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			By("部署一个会故意失败的后置脚本（exit 1）")
			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptFail()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "fail-script.sh", Order: 10, Params: map[string]string{"PKG_VERSION": "99.99.99"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			postScriptManager.Cleanup(scriptNames, "")
			postScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
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

		It("后置脚本失败时单master集群仍可进入最终 Ready", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群最终进入 Ready（当前实现下脚本失败不阻塞集群）")
			Eventually(func() bool {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				return cs == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(),
				"当前实现下后置脚本失败不应阻塞单master集群进入最终 Ready")

			By("验证集群虽然卡在后置处理阶段但 Kubernetes 组件已运行")
			nodeExec, err := executor.NewSSHExecutor(masterNode.IP, 22, masterNode.Username, masterNode.Password)
			Expect(err).NotTo(HaveOccurred())
			kubeletResult, _ := nodeExec.Exec("systemctl is-active kubelet")
			if kubeletResult != nil {
				GinkgoWriter.Printf("kubelet 状态: %s\n", kubeletResult.Stdout)
				Expect(strings.TrimSpace(kubeletResult.Stdout)).To(Equal("active"),
					"即使后置脚本失败，kubelet 也应该在运行")
			}
		})
	})

	// 用例7: 后置脚本安装失败不阻塞集群（3master管理集群，部分节点失败）
	// 后置脚本失败会导致 bke cluster create 阻塞在 EnsureNodesPostProcess 阶段，
	// 因此使用 nohup 后台执行，然后独立轮询集群阶段。
	Describe("多集群(3M)管理集群后置处理-脚本失败卡在后置处理阶段", Label("postscript", "3master", "script-failure", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNodes    []config.NodeInfo
			scriptNames    = []string{"fail-script.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-3msf-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3))

			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptFail()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "fail-script.sh", Order: 10, Params: map[string]string{"PKG_VERSION": "99.99.99"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterEach(func() {
			postScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				postScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
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

		It("后置脚本失败时3master集群仍可进入最终 Ready", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, masterNodes)
			vipHost, vipPort := config.LoadManagementClusterVIPFromEnv()
			if vipHost != "" {
				clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
			}

			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群最终进入 Ready（当前实现下脚本失败不阻塞集群）")
			Eventually(func() bool {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				return cs == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(),
				"当前实现下后置脚本失败不应阻塞3master集群进入最终 Ready")

			By("验证所有节点上 kubelet 仍在运行")
			for i, node := range masterNodes {
				nodeExec, err := executor.NewSSHExecutor(node.IP, 22, node.Username, node.Password)
				Expect(err).NotTo(HaveOccurred())
				kubeletResult, _ := nodeExec.Exec("systemctl is-active kubelet")
				if kubeletResult != nil {
					GinkgoWriter.Printf("节点 %d (%s) kubelet 状态: %s\n", i+1, node.IP, kubeletResult.Stdout)
					Expect(strings.TrimSpace(kubeletResult.Stdout)).To(Equal("active"),
						fmt.Sprintf("节点 %s: 即使后置脚本失败，kubelet 也应该在运行", node.IP))
				}
			}
		})
	})

	// 用例8: 扩容时执行后置脚本安装
	Describe("扩容时执行后置脚本安装", Label("postscript", "scale-out", "P1", "post-init", "online"), func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			masterNode     config.NodeInfo
			scaleOutNode   config.NodeInfo
			scriptNames    = []string{"init-os.sh"}
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("post-scale-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "扩容测试至少需要 2 个节点")

			masterNode = nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			scaleOutNode = nodes[1]
			scaleOutNode.Role = []string{"node"}

			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS()},
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
			createErr = clusterManager.CreateCluster(configPath, nodeConfigPath)
			Expect(createErr).NotTo(HaveOccurred())

			By("等待初始集群变为 Ready")
			Eventually(func() string {
				_, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))
		})

		AfterEach(func() {
			postScriptManager.Cleanup(scriptNames, "")
			postScriptManager.CleanupScriptsOnNode(masterNode.IP, masterNode.Username, masterNode.Password)
			postScriptManager.CleanupScriptsOnNode(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password)
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

		It("扩容节点时应该在新节点上执行后置脚本", Label("required_nodes:2"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("验证 Master 节点上后置脚本已执行")
			Eventually(func() bool {
				ok, _, _ := postScriptManager.VerifyScriptExecuted(masterNode.IP, masterNode.Username, masterNode.Password, scriptNames)
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

			By("验证扩容节点上后置脚本已落盘")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptLanded(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password, scriptNames)
				GinkgoWriter.Printf("扩容节点落盘: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())

			By("验证扩容节点上后置脚本已执行")
			Eventually(func() bool {
				ok, detail, _ := postScriptManager.VerifyScriptExecuted(scaleOutNode.IP, scaleOutNode.Username, scaleOutNode.Password, scriptNames)
				GinkgoWriter.Printf("扩容节点执行: %s\n", detail)
				return ok
			}, 5*time.Minute, 15*time.Second).Should(BeTrue())
		})
	})

	// ==================== P2 测试用例 ====================

	// 用例9: 管理集群多master后置处理创建业务集群
	Describe("管理集群多master后置处理创建业务集群", Label("postscript", "3master", "mgmt-workload", "P2", "post-init", "skip-temporarily", "online"), Ordered, func() {
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
			mgmtCluster, err := Create3M1WManagementCluster(clusterManager, localExecutor, "post-mgmt", true)
			if mgmtCluster != nil {
				mgmtClusterName = mgmtCluster.ClusterName
				mgmtConfigPath = mgmtCluster.ConfigPath
				mgmtNodeConfigPath = mgmtCluster.NodeConfigPath
				mgmtKubeconfigPath = mgmtCluster.KubeconfigPath
			}
			Expect(err).NotTo(HaveOccurred())

			By("在管理集群上部署后置脚本资源")
			err = postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS(), utils.PostTestScriptSetupEnv()},
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
			workloadClusterName = fmt.Sprintf("post-wl-%d", time.Now().Unix())
		})

		It("应该在业务集群所有节点上部署和执行后置脚本", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
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
			if f, l, _ := postScriptManager.VerifyPostProcessingLogs(workloadClusterName, mgmtKubeconfigPath); f {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			for i, node := range workloadNodes {
				By(fmt.Sprintf("验证业务集群 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})

		AfterAll(func() {
			if mgmtKubeconfigPath != "" {
				postScriptManager.Cleanup(scriptNames, mgmtKubeconfigPath)
			}
			for _, n := range workloadNodes {
				postScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
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

	// 用例10: 离线多master后置处理创建业务集群
	Describe("离线多master后置处理创建业务集群", Label("postscript", "3master", "offline", "P2", "post-init"), Ordered, func() {
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

			clusterName = fmt.Sprintf("post-off3m-%d", time.Now().Unix())
			masterNodes = make([]config.NodeInfo, 3)
			for i := 0; i < 3; i++ {
				masterNodes[i] = nodes[i]
				masterNodes[i].Role = []string{"master/node", "etcd"}
			}

			err := postScriptManager.SetupPostScripts(
				[]utils.ScriptContentDef{utils.PostTestScriptInitOS(), utils.PostTestScriptSetupEnv()},
				[]utils.ScriptConfigEntry{
					{ScriptName: "init-os.sh", Order: 10, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
					{ScriptName: "setup-env.sh", Order: 20, Params: map[string]string{"HTTP_REPO": "http://repo.example.com"}},
				},
				"",
			)
			Expect(err).NotTo(HaveOccurred())
		})

		It("离线模式下应该在所有节点上部署和执行后置脚本", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
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

			By("后台创建集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, utils.DefaultBootstraoClusterKubeconfig)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群资源出现")
			Eventually(func() bool {
				return clusterManager.ClusterExists(clusterName)
			}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "集群资源应该被创建")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, cs, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, cs)
				failOnClusterFailure(state, cs)
				return cs
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			if f, l, _ := postScriptManager.VerifyPostProcessingLogs(clusterName, ""); f {
				GinkgoWriter.Printf("后置处理日志:\n%s\n", l)
			} else {
				GinkgoWriter.Printf("未找到后置处理日志（不影响测试结果）\n")
			}

			for i, node := range masterNodes {
				By(fmt.Sprintf("验证 Master 节点 %d (%s)", i+1, node.IP))

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptLanded(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 落盘: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())

				Eventually(func() bool {
					ok, detail, _ := postScriptManager.VerifyScriptExecuted(node.IP, node.Username, node.Password, scriptNames)
					GinkgoWriter.Printf("节点 %s 执行: %s\n", node.IP, detail)
					return ok
				}, 5*time.Minute, 15*time.Second).Should(BeTrue())
			}
		})

		AfterAll(func() {
			postScriptManager.Cleanup(scriptNames, "")
			for _, n := range masterNodes {
				postScriptManager.CleanupScriptsOnNode(n.IP, n.Username, n.Password)
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
