package custom_installation

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

/*
  特性：支持二进制扩展和自定义（kubelet和containerd）
  用例：
      1）支持二进制配置扩展和自定义-自定义正确kubelet参数
      2）支持二进制配置扩展和自定义-自定义正确containerd参数
      3）支持二进制配置扩展和自定义-自定义kubelet参数为空
      4）支持二进制配置扩展和自定义-自定义containerd参数为空
      5）支持二进制配置扩展和自定义-自定义错误kubelet参数
      6）支持二进制配置扩展和自定义-自定义错误containerd参数
*/
// 由于后续有删除bc 和 bn操作未指定kubeconfig，所以创建集群时使用默认的配置（共部署时不是引导节点）
var _ = Describe("BKE 二进制配置扩展和自定义", Label("binary-config-install", "post-init", "online"), func() {
	var (
		sshExecutor          *executor.SSHExecutor
		guideConfig          *config.GuideNodeConfig
		dynamicClient        dynamic.Interface
		clusterManager       *utils.ClusterManager
		clusterName          string
		clusterConfig        *config.BKEClusterConfig
		configPath           string
		nodeConfigPath       string
		kubeletConfigPath    string
		containerdConfigPath string
		masterNode           config.NodeInfo
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

		clusterName = fmt.Sprintf("test-%d", time.Now().Unix())

		// 加载节点配置
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置一个测试节点")

		// 只使用第一个节点作为 Master
		masterNode = nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}

		// 创建集群配置（使用基础 Addons 以加快测试速度）
		clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
	})

	AfterEach(func() {
		// 清理：删除集群
		By("清理测试集群")
		if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
			// 触发集群销毁（设置 spec.reset=true）
			By("触发集群销毁")
			err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
			if err != nil {
				GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
			}

			// 等待集群完全删除（BC 资源不存在）
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
				GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
			}
		}

		// 清理二进制配置文件（kubelet/containerd）
		if kubeletConfigPath != "" {
			if err := clusterManager.CleanupConfig(kubeletConfigPath); err != nil {
				GinkgoWriter.Printf("Failed to cleanup kubelet config file: %v\n", err)
			}
		}
		if containerdConfigPath != "" {
			if err := clusterManager.CleanupConfig(containerdConfigPath); err != nil {
				GinkgoWriter.Printf("Failed to cleanup containerd config file: %v\n", err)
			}
		}

		// 清理对应的自定义资源（忽略不存在）
		// 说明：kubelet 使用 kct，containerd 使用 cct
		cleanupCRCmd := strings.Join([]string{
			"kubectl delete kct -n bke-kubelet bke-kubelet-2 --ignore-not-found=true",
			"kubectl delete kct -n bke-kubelet bke-kubelet-dynamic --ignore-not-found=true",
			"kubectl delete cct -n bke-containerd bke-containerd-2 --ignore-not-found=true",
		}, " && ")
		if result, err := sshExecutor.Exec(cleanupCRCmd); err != nil {
			GinkgoWriter.Printf("Failed to cleanup kct/cct resources: %v, stderr: %s\n", err, result.Stderr)
		}
	})

	/*
	 用例名称：1）支持二进制配置扩展和自定义-自定义正确kubelet参数 2）支持二进制配置扩展和自定义-自定义正确containerd参数
	 用例步骤：1）在BKECluster中配置kubelet和containerd引用
	         2）部署集群，预期结果1
	         3）检查节点上kubelet和containerd的配置，预期结果2
	 预期结果：1）集群部署成功
	         2）配置和自定义的一致
	*/
	Describe("使用containerd和kubelet的默认自定义配置", Label("use-self-cfg"), func() {
		// Author: xuyouhong
		It("应该成功创建 1Master 集群并验证自定义配置参数", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成kubelet和containerd配置文件")
			kubeletConfigPath, containerdConfigPath, err = clusterManager.GetConfigGenerator().GenerateBinaryConfig(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("新增containerd和kubelet自定义配置")
			err = clusterManager.CreateBinaryConfig(kubeletConfigPath, containerdConfigPath)
			Expect(err).NotTo(HaveOccurred(), "新增containerd和kubelet自定义配置应该成功执行")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("检查节点上kubelet和containerd的配置")
			// 从节点上实际生效的 systemd service 中读取 StartLimitBurst
			checkCmd := "systemctl show containerd -p StartLimitBurst --value; systemctl show kubelet -p StartLimitBurst --value"
			result, err := utils.ExecuteCommandOnNode(masterNode, checkCmd)
			Expect(err).NotTo(HaveOccurred(), "检查节点的 kubelet/containerd StartLimitBurst 应该成功执行")

			values := strings.Fields(result.Stdout)
			Expect(values).To(HaveLen(2), "应分别返回 containerd 和 kubelet 的 StartLimitBurst 值")
			Expect(values[0]).To(Equal("10"), "containerd StartLimitBurst 应为 10")
			Expect(values[1]).To(Equal("10"), "kubelet StartLimitBurst 应为 10")
		})

		// Author: zqw1
		It("应该支持 kubelet 自定义 CR 通过脚本动态渲染变量", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成kubelet和containerd配置文件")
			kubeletConfigPath, containerdConfigPath, err = clusterManager.GetConfigGenerator().GenerateBinaryConfig(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("配置 kubelet 自定义 CR 的脚本动态变量")
			err = clusterManager.GetConfigGenerator().EnableKubeletDynamicVariableSubstitution(kubeletConfigPath)
			Expect(err).NotTo(HaveOccurred(), "设置 kubelet 动态变量配置应该成功执行")

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("新增containerd和kubelet自定义配置")
			err = clusterManager.CreateBinaryConfig(kubeletConfigPath, containerdConfigPath)
			Expect(err).NotTo(HaveOccurred(), "新增containerd和kubelet自定义配置应该成功执行")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("验证 kubelet.service 中脚本变量已在节点侧渲染")
			hostnameResult, err := utils.ExecuteCommandOnNode(masterNode, "hostname")
			Expect(err).NotTo(HaveOccurred(), "获取节点 hostname 应该成功执行")
			renderedHostname := strings.TrimSpace(hostnameResult.Stdout)
			Expect(renderedHostname).NotTo(BeEmpty(), "节点 hostname 不应为空")

			serviceResult, err := utils.ExecuteCommandOnNode(masterNode, "systemctl cat kubelet")
			Expect(err).NotTo(HaveOccurred(), "读取 kubelet.service 应该成功执行")
			serviceContent := serviceResult.Stdout
			GinkgoWriter.Printf("kubelet.service 内容:\n%s\n", serviceContent)

			Expect(serviceContent).To(ContainSubstring(fmt.Sprintf(`Environment="BKE_DYNAMIC_HOSTNAME=%s"`, renderedHostname)),
				"kubelet.service 应包含脚本动态渲染后的 hostname")
			Expect(serviceContent).To(ContainSubstring(`Environment="BKE_DYNAMIC_MARK=kubelet-dynamic"`),
				"kubelet.service 应包含脚本动态渲染后的固定标记")
			Expect(serviceContent).NotTo(ContainSubstring("${EXPR|hostname|END}"),
				"kubelet.service 不应残留 hostname 动态表达式")
			Expect(serviceContent).NotTo(ContainSubstring("${EXPR|printf kubelet-dynamic|END}"),
				"kubelet.service 不应残留固定标记动态表达式")
		})
	})

	/*
	 用例名称：3）支持二进制配置扩展和自定义-自定义kubelet参数为空 4）支持二进制配置扩展和自定义-自定义containerd参数为空
	 用例步骤：1）在BKECluster中不配置kubelet和containerd引用
	         2）部署集群，预期结果1
	         3）检查节点上kubelet和containerd的配置，预期结果2
	 预期结果：1）集群部署成功
	         2）配置和默认的一致
	*/
	Describe("不配置containerd和kubelet的自定义配置，使用默认的配置", Label("not-self-cfg"), func() {
		// Author: xuyouhong
		It("应该成功创建 1Master 集群并验证默认配置参数", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("从 BKECluster 配置中删除 kubelet/containerd 的自定义配置引用")
			err = clusterManager.GetConfigGenerator().RemoveBinaryConfigRefs(configPath)
			Expect(err).NotTo(HaveOccurred(), "删除 kubelet/containerd 引用应该成功执行")

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("检查节点上kubelet和containerd的配置")
			// 从节点上实际生效的 systemd service 中读取 StartLimitBurst
			checkCmd := "systemctl show containerd -p StartLimitBurst --value; systemctl show kubelet -p StartLimitBurst --value"
			result, err := utils.ExecuteCommandOnNode(masterNode, checkCmd)
			Expect(err).NotTo(HaveOccurred(), "检查节点的 kubelet/containerd StartLimitBurst 应该成功执行")

			values := strings.Fields(result.Stdout)
			Expect(values).To(HaveLen(2), "应分别返回 containerd 和 kubelet 的 StartLimitBurst 值")
			Expect(values[0]).To(Equal("5"), "containerd StartLimitBurst 应为 5")
			Expect(values[1]).To(Equal("5"), "kubelet StartLimitBurst 应为 5")
		})
	})

	/*
	 用例名称：5）支持二进制配置扩展和自定义-自定义错误kubelet参数
	 用例步骤：1）在BKECluster中配置kubelet引用
	         2）部署集群，预期结果1
	 预期结果：1）集群部署失败，且kubelet服务异常
	*/
	Describe("配置错误的kubelet自定义配置，集群部署失败", Label("error-kubelet-self-cfg"), func() {
		// Author: xuyouhong
		It("当配置错误的 kubelet 参数时，集群部署应该失败，且 kubelet 服务异常", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成kubelet和containerd配置文件")
			kubeletConfigPath, containerdConfigPath, err = clusterManager.GetConfigGenerator().GenerateBinaryConfig(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("将 kubelet 自定义配置修改为错误参数")
			err = clusterManager.GetConfigGenerator().MakeKubeletConfigInvalid(kubeletConfigPath)
			Expect(err).NotTo(HaveOccurred(), "设置错误的 kubelet 配置应该成功执行")

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("新增containerd和kubelet自定义配置（其中 kubelet 配置为错误参数）")
			err = clusterManager.CreateBinaryConfig(kubeletConfigPath, containerdConfigPath)
			Expect(err).NotTo(HaveOccurred(), "新增containerd和kubelet自定义配置应该成功执行")

			By("在后台执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "后台创建集群命令应该成功启动")

			By("等待集群状态变为 InitializationFailed")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("InitializationFailed"), "集群应该变为 InitializationFailed 状态")

			By("检查 kubelet 服务状态，应为异常状态")
			checkCmd := "systemctl is-active kubelet || echo inactive"
			result, err := utils.ExecuteCommandOnNode(masterNode, checkCmd)
			Expect(err).NotTo(HaveOccurred(), "检查节点的 kubelet 服务状态应该成功执行")

			status := strings.TrimSpace(result.Stdout)
			Expect(status).NotTo(Equal("active"), "kubelet 服务不应为 active 状态")

			// 额外清理：查询并删除当前集群命名空间下的所有 BKECluster（bc）和 BKENode（bn）资源
			bcNamespace := fmt.Sprintf("bke-%s", clusterName)
			By("删除集群命名空间下的所有 bc 和 bn 资源")
			cleanupCRCmd := strings.Join([]string{
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bc -n %s --all --ignore-not-found=true", "", bcNamespace),
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bn -n %s --all --ignore-not-found=true", "", bcNamespace),
			}, " && ")
			if result, err := sshExecutor.Exec(cleanupCRCmd); err != nil {
				GinkgoWriter.Printf("Failed to cleanup bc/bn resources: %v, stderr: %s\n", err, result.Stderr)
			}

			By("等待并验证 bc 和 bn 资源已被删除")
			checkTimeout := 5 * time.Minute
			checkInterval := 5 * time.Second
			Eventually(func() bool {
				// 检查 bc 资源：使用 --ignore-not-found，资源不存在时 stdout 为空
				checkBCCmd := fmt.Sprintf("kubectl get bc -n %s --no-headers --ignore-not-found 2>&1", bcNamespace)
				bcResult, _ := sshExecutor.Exec(checkBCCmd)
				bcOutput := strings.TrimSpace(bcResult.Stdout)
				// 资源存在：退出码为 0 且输出不为空（且不包含错误信息）
				bcExists := bcResult.ExitCode == 0 && bcOutput != "" && !strings.Contains(bcOutput, "NotFound") && !strings.Contains(bcOutput, "No resources found")

				// 检查 bn 资源：使用 --ignore-not-found，资源不存在时 stdout 为空
				checkBNCmd := fmt.Sprintf("kubectl get bn -n %s --no-headers --ignore-not-found 2>&1", bcNamespace)
				bnResult, _ := sshExecutor.Exec(checkBNCmd)
				bnOutput := strings.TrimSpace(bnResult.Stdout)
				// 资源存在：退出码为 0 且输出不为空（且不包含错误信息）
				bnExists := bnResult.ExitCode == 0 && bnOutput != "" && !strings.Contains(bnOutput, "NotFound") && !strings.Contains(bnOutput, "No resources found")

				if bcExists || bnExists {
					GinkgoWriter.Printf("资源仍存在: bc=%v (exitCode=%d, output=%q), bn=%v (exitCode=%d, output=%q)\n",
						bcExists, bcResult.ExitCode, bcOutput, bnExists, bnResult.ExitCode, bnOutput)
					return false
				}

				return true
			}, checkTimeout, checkInterval).Should(BeTrue(), "bc 和 bn 资源应该在 5 分钟内被完全删除")
		})
	})

	/*
	 用例名称：6）支持二进制配置扩展和自定义-自定义错误containerd参数
	 用例步骤：1）在BKECluster中配置containerd引用
	         2）部署集群，预期结果1
	 预期结果：1）集群部署失败，且containerd服务异常
	*/
	Describe("配置错误的containerd自定义配置，集群部署失败", Label("error-containerd-self-cfg"), func() {
		// Author: xuyouhong
		It("当配置错误的 containerd 参数时，集群部署应该失败，且 containerd 服务异常", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成kubelet和containerd配置文件")
			kubeletConfigPath, containerdConfigPath, err = clusterManager.GetConfigGenerator().GenerateBinaryConfig(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

			By("将 containerd 自定义配置修改为错误参数")
			err = clusterManager.GetConfigGenerator().MakeContainerdConfigInvalid(containerdConfigPath)
			Expect(err).NotTo(HaveOccurred(), "设置错误的 containerd 配置应该成功执行")

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("新增containerd和kubelet自定义配置（其中 containerd 配置为错误参数）")
			err = clusterManager.CreateBinaryConfig(kubeletConfigPath, containerdConfigPath)
			Expect(err).NotTo(HaveOccurred(), "新增containerd和kubelet自定义配置应该成功执行")

			By("在后台执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "后台创建集群命令应该成功启动")

			By("等待集群状态变为 InitializationFailed")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("InitializationFailed"), "集群应该变为 clusterStatus 状态")

			By("检查 containerd 服务状态，应为异常状态")
			checkCmd := "systemctl is-active containerd || echo inactive"
			result, err := utils.ExecuteCommandOnNode(masterNode, checkCmd)
			Expect(err).NotTo(HaveOccurred(), "检查节点的 containerd 服务状态应该成功执行")

			status := strings.TrimSpace(result.Stdout)
			Expect(status).NotTo(Equal("active"), "containerd 服务不应为 active 状态")

			// 额外清理：查询并删除当前集群命名空间下的所有 BKECluster（bc）和 BKENode（bn）资源
			bcNamespace := fmt.Sprintf("bke-%s", clusterName)
			By("删除集群命名空间下的所有 bc 和 bn 资源")
			cleanupCRCmd := strings.Join([]string{
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bc -n %s --all --ignore-not-found=true", "", bcNamespace),
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bn -n %s --all --ignore-not-found=true", "", bcNamespace),
			}, " && ")
			if result, err := sshExecutor.Exec(cleanupCRCmd); err != nil {
				GinkgoWriter.Printf("Failed to cleanup bc/bn resources: %v, stderr: %s\n", err, result.Stderr)
			}

			By("等待并验证 bc 和 bn 资源已被删除")
			checkTimeout := 5 * time.Minute
			checkInterval := 5 * time.Second
			Eventually(func() bool {
				// 检查 bc 资源：使用 --ignore-not-found，资源不存在时 stdout 为空
				checkBCCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bc -n %s --no-headers --ignore-not-found 2>&1", "", bcNamespace)
				bcResult, _ := sshExecutor.Exec(checkBCCmd)
				bcOutput := strings.TrimSpace(bcResult.Stdout)
				// 资源存在：退出码为 0 且输出不为空（且不包含错误信息）
				bcExists := bcResult.ExitCode == 0 && bcOutput != "" && !strings.Contains(bcOutput, "NotFound") && !strings.Contains(bcOutput, "No resources found")

				// 检查 bn 资源：使用 --ignore-not-found，资源不存在时 stdout 为空
				checkBNCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bn -n %s --no-headers --ignore-not-found 2>&1", "", bcNamespace)
				bnResult, _ := sshExecutor.Exec(checkBNCmd)
				bnOutput := strings.TrimSpace(bnResult.Stdout)
				// 资源存在：退出码为 0 且输出不为空（且不包含错误信息）
				bnExists := bnResult.ExitCode == 0 && bnOutput != "" && !strings.Contains(bnOutput, "NotFound") && !strings.Contains(bnOutput, "No resources found")

				if bcExists || bnExists {
					GinkgoWriter.Printf("资源仍存在: bc=%v (exitCode=%d, output=%q), bn=%v (exitCode=%d, output=%q)\n",
						bcExists, bcResult.ExitCode, bcOutput, bnExists, bnResult.ExitCode, bnOutput)
					return false
				}

				return true
			}, checkTimeout, checkInterval).Should(BeTrue(), "bc 和 bn 资源应该在 5 分钟内被完全删除")
		})
	})
})
