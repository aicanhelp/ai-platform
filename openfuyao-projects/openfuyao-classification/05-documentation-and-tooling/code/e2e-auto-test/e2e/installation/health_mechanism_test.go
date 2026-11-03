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

/*
  特性：健康检查机制优化（bkeagent、bke-controller-manager）
  用例：
      1）后端安装业务集群（默认端口）
      2）后端安装业务集群（自定义端口）
      3）k8s组件健康状态模拟检查
      4）bkeagent监听端口（58080）占用导致部署失败并恢复
      5）后端安装管理集群（自定义端口）
      6）管理集群安装（默认端口）
*/

/*
用例名称：1）后端安装业务集群（默认端口） 3）k8s组件健康状态模拟检查 6）管理集群安装（默认端口）
用例步骤：1）检查管理集群的监听端口，预期结果1和6

	2）安装业务集群，预期结果1
	3）k8s组件健康状态模拟检测，预期结果2、3、4和5

预期结果：1）bkeagent的状态正常，默认使用58080健康监听端口

	2）kube-scheduler的状态正常，默认使用HTTPS协议监听10259端口
	3）kube-apiserver的状态正常，默认使用HTTPS协议监听6443端口
	4）kube-controller-manager的状态正常，默认使用HTTPS协议监听10257端口
	5）etcd的状态正常，默认使用HTTPS协议监听2379端口
	6）bke-manager-controller的状态正常，默认使用HTTPS协议监听9444端口
*/
var _ = Describe("管理集群下默认端口健康检查与业务集群安装", Label("bke-health-mechanism", "with-management-cluster", "online", "use-default-port"), func() {
	var (
		sshExecutor    *executor.SSHExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
		clusterName    string
		clusterConfig  *config.BKEClusterConfig
		configPath     string
		nodeConfigPath string
		masterNode     config.NodeInfo
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
		By("清理测试集群")
		if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
			By("触发集群销毁")
			err := clusterManager.DeleteClusterWithKubeconfig(clusterName, "")
			if err != nil {
				GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
			}

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

		if configPath != "" {
			if err := clusterManager.CleanupConfig(configPath); err != nil {
				GinkgoWriter.Printf("Failed to cleanup config file: %v\n", err)
			}
		}
	})

	It("应该成功创建 1Master 集群并验证默认健康监听端口", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		var err error

		// 预检查：管理集群上的关键组件应使用期望的监听端口
		By("预检查管理集群关键组件端口：bkeagent(58080)、bke-controller-manager(9444)")
		// bkeagent - http://127.0.0.1:58080/healthz -> ok
		By("验证 bkeagent 状态正常，默认使用 58080 健康监听端口")
		// 使用 curl 检查 bkeagent 的 /healthz 端点，应该返回 ok
		selfCheckBkeagentHealthCmd := "curl -s http://127.0.0.1:58080/healthz"
		selfBkeagentHealthResult, err := sshExecutor.Exec(selfCheckBkeagentHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 bkeagent /healthz 端点应该成功执行")
		selfBkeagentHealthOutput := strings.TrimSpace(selfBkeagentHealthResult.Stdout)
		Expect(selfBkeagentHealthOutput).To(Equal("ok"), "bkeagent /healthz 端点应该返回 ok，实际返回: %s", selfBkeagentHealthOutput)
		GinkgoWriter.Printf("bkeagent /healthz 响应: %s\n", selfBkeagentHealthOutput)

		// bke-controller-manager - 先获取 Pod IP，再通过 https://<podIP>:9444/healthz 校验
		By("验证 bke-controller-manager 使用 9444 端口（通过 Pod IP 检查 /healthz）")
		getPodIPCmd := "kubectl get pod -n cluster-system -l control-plane=controller-manager -o jsonpath='{.items[0].status.podIP}' 2>&1"
		ipResult, err := sshExecutor.Exec(getPodIPCmd)
		Expect(err).NotTo(HaveOccurred(), "获取 bke-controller-manager Pod IP 应该成功执行")
		podIP := strings.TrimSpace(ipResult.Stdout)
		Expect(podIP).NotTo(BeEmpty(), "bke-controller-manager Pod IP 不应为空")

		checkHealthCmd := fmt.Sprintf("curl -s -k https://%s:9444/healthz", podIP)
		healthResult, err := sshExecutor.Exec(checkHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 bke-controller-manager /healthz 应该成功执行")
		output := strings.TrimSpace(healthResult.Stdout)
		Expect(output).To(Equal("ok"), "bke-controller-manager /healthz 端点应该返回 ok，实际返回: %s", output)
		GinkgoWriter.Printf("bke-controller-manager /healthz 响应: %s (podIP=%s)\n", output, podIP)

		By("生成集群配置文件")
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
		Eventually(func() string {
			phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

		// 预期结果1：验证 bkeagent 的状态正常，默认使用 58080 健康监听端口
		By("验证 bkeagent 状态正常，默认使用 58080 健康监听端口")
		// 使用 curl 检查 bkeagent 的 /healthz 端点，应该返回 ok
		checkBkeagentHealthCmd := "curl -s http://127.0.0.1:58080/healthz"
		bkeagentHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkBkeagentHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 bkeagent /healthz 端点应该成功执行")
		bkeagentHealthOutput := strings.TrimSpace(bkeagentHealthResult.Stdout)
		Expect(bkeagentHealthOutput).To(Equal("ok"), "bkeagent /healthz 端点应该返回 ok，实际返回: %s", bkeagentHealthOutput)
		GinkgoWriter.Printf("bkeagent /healthz 响应: %s\n", bkeagentHealthOutput)

		// 预期结果2： kube-scheduler的状态正常，默认使用HTTPS协议监听10259端口
		checkSchedulerHealthCmd := fmt.Sprintf("curl -s --cacert /etc/kubernetes/pki/ca.crt https://%s:10259/healthz", masterNode.IP)
		schedulerHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkSchedulerHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 kube-scheduler /healthz 端点应该成功执行")
		schedulerHealthOutput := strings.TrimSpace(schedulerHealthResult.Stdout)
		Expect(schedulerHealthOutput).To(Equal("ok"), "kube-scheduler /healthz 端点应该返回 ok，实际返回: %s", schedulerHealthOutput)
		GinkgoWriter.Printf("kube-scheduler /healthz 响应: %s\n", schedulerHealthOutput)

		// 预期结果3：kube-apiserver 状态正常，默认使用 HTTPS 协议监听 6443 端口
		By("验证 kube-apiserver 状态正常，默认使用 HTTPS 协议监听 6443 端口")
		checkAPIServerLivezCmd := fmt.Sprintf("curl -s --cacert /etc/kubernetes/pki/ca.crt https://%s:6443/livez", masterNode.IP)
		apiServerLivezResult, err := utils.ExecuteCommandOnNode(masterNode, checkAPIServerLivezCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 kube-apiserver /livez 端点应该成功执行")
		apiServerLivezOutput := strings.TrimSpace(apiServerLivezResult.Stdout)
		Expect(apiServerLivezOutput).To(Equal("ok"), "kube-apiserver /livez 端点应该返回 ok，实际返回: %s", apiServerLivezOutput)
		GinkgoWriter.Printf("kube-apiserver /livez 响应: %s\n", apiServerLivezOutput)

		// 预期结果4：kube-controller-manager 状态正常，默认使用 HTTPS 协议监听 10257 端口
		By("验证 kube-controller-manager 状态正常，默认使用 HTTPS 协议监听 10257 端口")
		checkControllerManagerHealthCmd := fmt.Sprintf("curl -s --cacert /etc/kubernetes/pki/ca.crt https://%s:10257/healthz", masterNode.IP)
		controllerManagerHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkControllerManagerHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 kube-controller-manager /healthz 端点应该成功执行")
		controllerManagerHealthOutput := strings.TrimSpace(controllerManagerHealthResult.Stdout)
		Expect(controllerManagerHealthOutput).To(Equal("ok"), "kube-controller-manager /healthz 端点应该返回 ok，实际返回: %s", controllerManagerHealthOutput)
		GinkgoWriter.Printf("kube-controller-manager /healthz 响应: %s\n", controllerManagerHealthOutput)

		// 预期结果5：etcd 状态正常，默认使用 HTTPS 协议监听 2379 端口（客户端接口）
		By("验证 etcd 状态正常，默认使用 HTTPS 协议监听 2379 端口")
		checkEtcdHealthCmd := "curl -s --cacert /etc/kubernetes/pki/etcd/ca.crt --cert /etc/kubernetes/pki/etcd/server.crt --key /etc/kubernetes/pki/etcd/server.key https://127.0.0.1:2379/health"
		etcdHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkEtcdHealthCmd)
		Expect(err).NotTo(HaveOccurred(), "检查 etcd /health 端点应该成功执行")
		etcdHealthOutput := strings.TrimSpace(etcdHealthResult.Stdout)
		Expect(etcdHealthOutput).To(ContainSubstring("\"health\":\"true\""), "etcd /health 应返回健康 JSON，实际返回: %s", etcdHealthOutput)
		GinkgoWriter.Printf("etcd /health 响应: %s\n", etcdHealthOutput)
	})
})

var _ = Describe("引导集群下健康检查机制优化", Label("bke-health-mechanism", "post-init", "online"), func() {
	var (
		sshExecutor    *executor.SSHExecutor
		guideConfig    *config.GuideNodeConfig
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
		clusterName    string
		clusterConfig  *config.BKEClusterConfig
		configPath     string
		nodeConfigPath string
		masterNode     config.NodeInfo
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
	})

	/*
	 用例名称：4）bkeagent监听端口（58080）占用导致部署失败并恢复
	 用例步骤：
	         1）在 master 节点占用 58080 端口（nc -l 58080），导致 bkeagent 无法监听，从而集群部署失败，预期结果1
	         2）清理 bc 和 bn 资源，并等待其在 5 分钟内删除完成
	         3）释放 58080 端口（kill nc）
	         4）重新部署集群，期望部署成功（Healthy）且 bkeagent /healthz 返回 ok，预期结果2
	 预期结果：
	         1）端口占用阶段集群不会进入 Healthy，且状态进入失败态（DeployFailed / InitializationFailed）
	         2）清理并释放端口后集群可成功部署且 bkeagent 健康端点正常
	*/
	Describe("bkeagent监听端口（58080）占用导致部署失败并恢复", Label("bkeagent-port-occupied"), func() {
		It("节点 58080 被占用时部署失败，清理 bc/bn 并释放端口后部署成功", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("在 master 节点占用 58080 端口（模拟端口被占用）")
			// 预检查：先停止节点上的 bkeagent，避免已有进程占用/影响预期失败路径
			By("预检查：停止 master 节点上的 bkeagent（避免抢占 58080）")
			stopBkeagentCmd := "systemctl stop bkeagent 2>/dev/null || true; pkill -9 -f bkeagent >/dev/null 2>&1 || true"
			_, _ = utils.ExecuteCommandOnNode(masterNode, stopBkeagentCmd)

			By("等待确认 58080 端口已未被监听")
			Eventually(func() bool {
				checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
				res, _ := utils.ExecuteCommandOnNode(masterNode, checkPortCmd)
				return strings.TrimSpace(res.Stdout) == "not"
			}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 1 分钟内释放/停止监听")

			// 使用 nohup + 后台，确保 ssh session 结束后端口占用仍然存在
			// 说明：不同 nc 实现可能在首次连接后退出；使用 while 循环确保端口持续被占用直到 kill。
			occupyCmd := "nohup sh -c 'while true; do nc -l 58080 >/dev/null 2>&1; done' >/dev/null 2>&1 & echo $!"
			occupyResult, occErr := utils.ExecuteCommandOnNode(masterNode, occupyCmd)
			Expect(occErr).NotTo(HaveOccurred(), "占用 58080 端口的命令执行失败")
			occupyPid := strings.TrimSpace(occupyResult.Stdout)
			Expect(occupyPid).NotTo(BeEmpty(), "未获取到占用进程 PID")

			DeferCleanup(func() {
				By("DeferCleanup: 检测 58080 端口是否仍被 nc 占用（非 nc 占用即通过，nc 残留则兜底 kill）")
				Eventually(func() bool {
					checkNcCmd := "((ss -ltnp 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' | grep -i nc >/dev/null 2>&1) && echo occupied || echo free"
					res, _ := utils.ExecuteCommandOnNode(masterNode, checkNcCmd)
					if strings.TrimSpace(res.Stdout) == "free" {
						return true
					}
					cleanupNcCmd := fmt.Sprintf(
						"kill -9 %s >/dev/null 2>&1 || true; pkill -9 -f 'nc.*58080' >/dev/null 2>&1 || true; pkill -9 -f 'nc -l 58080' >/dev/null 2>&1 || true",
						occupyPid,
					)
					_, _ = utils.ExecuteCommandOnNode(masterNode, cleanupNcCmd)
					return false
				}, 2*time.Minute, 5*time.Second).Should(BeTrue(), "DeferCleanup: 58080 端口不应被 nc 占用")
			})

			By("等待确认 58080 端口已被占用")
			Eventually(func() bool {
				// 兼容 ss/netstat，判断端口监听是否存在
				checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
				res, _ := utils.ExecuteCommandOnNode(masterNode, checkPortCmd)
				return strings.TrimSpace(res.Stdout) == "listening"
			}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 1 分钟内开始被监听")

			By("在后台执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "后台创建集群命令应该成功启动")

			By("等待集群进入失败状态（端口占用导致 bkeagent 无法监听）")
			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				if clusterStatus == "Ready" {
					return false
				}
				return state == "DeployFailed" || state == "InitializationFailed" || clusterStatus == "InitializationFailed"
			}, installTimeout*2, pollInterval).Should(BeTrue(), "集群应在端口占用情况下进入失败态")

			// 失败后额外清理：删除当前集群命名空间下的 bc 和 bn
			bcNamespace := fmt.Sprintf("bke-%s", clusterName)
			By("删除集群命名空间下的所有 bc 和 bn 资源")
			cleanupCRCmd := strings.Join([]string{
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bc -n %s --all --ignore-not-found=true", "", bcNamespace),
				fmt.Sprintf("KUBECONFIG=%s kubectl delete bn -n %s --all --ignore-not-found=true", "", bcNamespace),
			}, " && ")
			if result, cleanupErr := sshExecutor.Exec(cleanupCRCmd); cleanupErr != nil {
				GinkgoWriter.Printf("Failed to cleanup bc/bn resources: %v, stderr: %s\n", cleanupErr, result.Stderr)
			}

			By("等待并验证 bc 和 bn 资源已被删除（5 分钟超时）")
			checkTimeout := 5 * time.Minute
			checkInterval := 5 * time.Second
			Eventually(func() bool {
				checkBCCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bc -n %s --no-headers 2>&1 || true", "", bcNamespace)
				bcResult, _ := sshExecutor.Exec(checkBCCmd)
				bcOutput := strings.TrimSpace(bcResult.Stdout)
				bcExists := bcOutput != "" &&
					!strings.Contains(bcOutput, "NotFound") &&
					!strings.Contains(bcOutput, "No resources found")

				checkBNCmd := fmt.Sprintf("KUBECONFIG=%s kubectl get bn -n %s --no-headers 2>&1 || true", "", bcNamespace)
				bnResult, _ := sshExecutor.Exec(checkBNCmd)
				bnOutput := strings.TrimSpace(bnResult.Stdout)
				bnExists := bnOutput != "" &&
					!strings.Contains(bnOutput, "NotFound") &&
					!strings.Contains(bnOutput, "No resources found")

				return !(bcExists || bnExists)
			}, checkTimeout, checkInterval).Should(BeTrue(), "bc 和 bn 资源应该在 5 分钟内被完全删除")

			By(fmt.Sprintf("停止 58080 端口占用并释放端口（kill pid=%s + 清理 nc 监听）", occupyPid))
			// 关键点：占用时是 while 循环持续拉起 nc，因此释放端口要同时停止 while 进程，并且清理所有 nc 对 58080 的监听。
			stopOccupyCmd := fmt.Sprintf(
				"kill -9 %s >/dev/null 2>&1 || true; pkill -9 -f 'nc.*58080' >/dev/null 2>&1 || true; pkill -9 -f 'nc -l 58080' >/dev/null 2>&1 || true",
				occupyPid,
			)
			_, _ = utils.ExecuteCommandOnNode(masterNode, stopOccupyCmd)

			By("等待确认 58080 端口已释放（监听不存在）")
			Eventually(func() bool {
				checkPortCmd := "((ss -ltn 2>/dev/null || netstat -lntp 2>/dev/null) | grep -E '(:|\\.)58080\\b' >/dev/null 2>&1) && echo listening || echo not"
				res, _ := utils.ExecuteCommandOnNode(masterNode, checkPortCmd)
				return strings.TrimSpace(res.Stdout) == "not"
			}, 1*time.Minute, 5*time.Second).Should(BeTrue(), "58080 端口应在 1 分钟内释放")

			By("重新部署集群")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "重新创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			By("单次验证 bkeagent 在 58080 端口上健康")
			checkCmd := "curl -s http://127.0.0.1:58080/healthz"
			result, cmdErr := utils.ExecuteCommandOnNode(masterNode, checkCmd)
			Expect(cmdErr).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(result.Stdout)).To(Equal("ok"), "bkeagent /healthz 端点应该返回 ok")
		})
	})

	/*
	 用例名称：3）后端安装业务集群（自定义端口）
	 用例步骤：1）安装业务集群，预期结果1
	 预期结果：1）bkeagent的状态正常，使用自定义的健康监听端口
	*/
	Describe("业务集群使用自定义的监听端口", Label("use-selfdefine-port"), func() {
		It("应该成功创建业务集群并验证自定义 bkeagent 监听端口", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("从 BKECluster 配置bkeagent的自定义监听端口")
			err = clusterManager.GetConfigGenerator().ModBkeagentHealthPort(configPath, "58090")
			Expect(err).NotTo(HaveOccurred(), "配置 bkeagent 自定义监听端口应该成功执行")

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

			// 预期结果1：验证 bkeagent 的状态正常，使用自定义的 58090 健康监听端口
			By("验证 bkeagent 状态正常，使用自定义的 58090 健康监听端口")
			// 使用 curl 检查 bkeagent 的 /healthz 端点，应该返回 ok
			checkBkeagentHealthCmd := "curl -s http://127.0.0.1:58090/healthz"
			bkeagentHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkBkeagentHealthCmd)
			Expect(err).NotTo(HaveOccurred(), "检查 bkeagent /healthz 端点应该成功执行")
			bkeagentHealthOutput := strings.TrimSpace(bkeagentHealthResult.Stdout)
			Expect(bkeagentHealthOutput).To(Equal("ok"), "bkeagent /healthz 端点应该返回 ok，实际返回: %s", bkeagentHealthOutput)
			GinkgoWriter.Printf("bkeagent /healthz 响应: %s\n", bkeagentHealthOutput)
		})
	})

	/*
	 用例名称：5）后端安装管理集群（自定义端口）
	 用例步骤：1）安装管理集群，预期结果1和2
	 预期结果：1）bkeagent的状态正常，使用自定义的健康监听端口
	         2）bke-controller-manager的状态正常，使用HTTPS协议监听自定义的端口
	*/
	Describe("管理集群使用自定义的监听端口", Label("manager-use-selfdefine-port"), func() {
		BeforeEach(func() {
			By("修改 bke-controller-manager 的监听端口")
			err := clusterManager.GetConfigGenerator().ModBkeControllerManagerPort("9444", "9449")
			if err != nil {
				GinkgoWriter.Printf("修改 bke-controller-manager 监听端口失败: %v\n", err)
			}
		})

		AfterEach(func() {
			// 删除管理集群自身：在管理集群 master 上执行（自管 BC 继承引导侧名，见 #224）
			By("在管理集群 master 节点上触发管理集群自身删除")
			selfBC := utils.ManagementClusterSelfBCName(clusterName)
			selfNS := utils.ManagementClusterSelfBCNamespace(clusterName)
			deleteSelfCmd := strings.Join([]string{
				fmt.Sprintf(`kubectl annotate bc %s -n %s bke.bocloud.com/ignore-target-cluster-delete="false" --overwrite`, selfBC, selfNS),
				fmt.Sprintf(`kubectl annotate bc %s -n %s bke.bocloud.com/ignore-namespace-delete="false" --overwrite`, selfBC, selfNS),
				fmt.Sprintf(`kubectl patch bc %s -n %s --type=merge -p '{"spec":{"reset":true}}'`, selfBC, selfNS),
			}, " && ")
			deleteSelfResult, err := utils.ExecuteCommandOnNode(masterNode, deleteSelfCmd)
			if err != nil || deleteSelfResult.ExitCode != 0 {
				GinkgoWriter.Printf("管理集群删除自身失败: %v, exit=%d\n", err, deleteSelfResult.ExitCode)
			}

			// 恢复 bke-manager-controller 的监听端口为默认的 9444
			By("恢复 bke-manager-controller 的监听端口为默认的 9444")
			err = clusterManager.GetConfigGenerator().ModBkeControllerManagerPort("9449", "9444")
			if err != nil {
				GinkgoWriter.Printf("恢复 bke-controller-manager 监听端口失败: %v\n", err)
			}
		})

		It("应该成功创建管理集群并验证自定义健康监听端口", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error
			// 预处理，加上cluster-api的addon
			clusterConfig = config.NewBKEClusterConfigForMgmt(clusterName, []config.NodeInfo{masterNode})

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("从 BKECluster 配置bkeagent的自定义监听端口")
			err = clusterManager.GetConfigGenerator().ModBkeagentHealthPort(configPath, "58090")
			Expect(err).NotTo(HaveOccurred(), "配置 bkeagent 自定义监听端口应该成功执行")

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
			}, mgmtInstallTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			// 预期结果1：验证 bkeagent 的状态正常，使用自定义的 58090 健康监听端口
			By("验证 bkeagent 状态正常，使用自定义的 58090 健康监听端口")
			// 使用 curl 检查 bkeagent 的 /healthz 端点，应该返回 ok
			checkBkeagentHealthCmd := "curl -s http://127.0.0.1:58090/healthz"
			bkeagentHealthResult, err := utils.ExecuteCommandOnNode(masterNode, checkBkeagentHealthCmd)
			Expect(err).NotTo(HaveOccurred(), "检查 bkeagent /healthz 端点应该成功执行")
			bkeagentHealthOutput := strings.TrimSpace(bkeagentHealthResult.Stdout)
			Expect(bkeagentHealthOutput).To(Equal("ok"), "bkeagent /healthz 端点应该返回 ok，实际返回: %s", bkeagentHealthOutput)
			GinkgoWriter.Printf("bkeagent /healthz 响应: %s\n", bkeagentHealthOutput)

			By("循环检测管理集群自管 BC 状态，直到 Ready")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			mgmtKubeconfigPath, err := checker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "应该成功获取管理集群 kubeconfig")
			Eventually(func() bool {
				_, state, clusterStatus, e := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, clusterName)
				if e != nil {
					GinkgoWriter.Printf("[mgmt-self] 获取自管 BC 状态失败: %v\n", e)
					return false
				}
				GinkgoWriter.Printf("[mgmt-self] state=%s, clusterStatus=%s\n", state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus == "Ready"
			}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "管理集群自管 BC 应该变为 Ready")

			// bke-controller-manager - 先获取 Pod IP，再通过 https://<podIP>:9449/healthz 校验
			By("验证 bke-controller-manager 使用 9449 端口（通过 Pod IP 检查 /healthz）")
			By("循环检测 bke-controller-manager Pod，直到 Running 且 1/1 Ready")
			Eventually(func() bool {
				checkControllerManagerPhaseCmd := "kubectl get pod -n cluster-system -l control-plane=controller-manager -o jsonpath='{.items[0].status.phase}' 2>&1"
				phaseResult, e := utils.ExecuteCommandOnNode(masterNode, checkControllerManagerPhaseCmd)
				if e != nil || phaseResult.ExitCode != 0 {
					return false
				}
				podPhase := strings.TrimSpace(phaseResult.Stdout)

				checkControllerManagerReadyCmd := "kubectl get pod -n cluster-system -l control-plane=controller-manager -o jsonpath='{.items[0].status.containerStatuses[0].ready}' 2>&1"
				readyResult, e := utils.ExecuteCommandOnNode(masterNode, checkControllerManagerReadyCmd)
				if e != nil || readyResult.ExitCode != 0 {
					return false
				}
				podReady := strings.TrimSpace(readyResult.Stdout)
				GinkgoWriter.Printf("bke-controller-manager Pod 状态: phase=%s, ready=%s\n", podPhase, podReady)
				return podPhase == "Running" && podReady == "true"
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "bke-controller-manager Pod 应该在 5 分钟内变为 Running 且 1/1 Ready")

			getPodIPCmd := "kubectl get pod -n cluster-system -l control-plane=controller-manager -o jsonpath='{.items[0].status.podIP}' 2>&1"
			ipResult, err := utils.ExecuteCommandOnNode(masterNode, getPodIPCmd)
			Expect(err).NotTo(HaveOccurred(), "获取 bke-controller-manager Pod IP 应该成功执行")
			podIP := strings.TrimSpace(ipResult.Stdout)
			Expect(podIP).NotTo(BeEmpty(), "bke-controller-manager Pod IP 不应为空")

			checkHealthCmd := fmt.Sprintf("curl -s -k https://%s:9449/healthz", podIP)
			healthResult, err := utils.ExecuteCommandOnNode(masterNode, checkHealthCmd)
			Expect(err).NotTo(HaveOccurred(), "检查 bke-controller-manager /healthz 应该成功执行")
			output := strings.TrimSpace(healthResult.Stdout)
			Expect(output).To(Equal("ok"), "bke-controller-manager /healthz 端点应该返回 ok，实际返回: %s", output)
			GinkgoWriter.Printf("bke-controller-manager /healthz 响应: %s (podIP=%s)\n", output, podIP)
		})
	})
})
