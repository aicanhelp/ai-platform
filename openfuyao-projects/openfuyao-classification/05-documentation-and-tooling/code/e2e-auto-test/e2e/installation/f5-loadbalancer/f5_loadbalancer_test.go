package f5_loadbalancer

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

var _ = SIGDescribe("BKE Cluster Installation With F5 LoadBalancer Mock", func() {
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

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功加载 kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功创建动态客户端")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)
	})

	Describe("引导节点拉起 3Master1Worker 管理集群（haproxy 模拟 F5）", Label("f5-lb", "3m1w-mgmt", "post-init", "skip-temporarily", "online"), Ordered, func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			kubeconfigPath string
			lbNode         config.NodeInfo
		)

		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			if len(nodes) < 5 {
				Skip(fmt.Sprintf("需要至少 5 个节点（3m1w 管理集群 4 个 + 1 个 LB 节点），当前 %d 个，跳过", len(nodes)))
			}

			mgmtNodes := make([]config.NodeInfo, 4)
			copy(mgmtNodes, nodes[:4])
			for i := 0; i < 3; i++ {
				mgmtNodes[i].Role = []string{"master/node", "etcd"}
			}
			mgmtNodes[3].Role = []string{"node"}
			lbNode = nodes[4]
			Expect(strings.TrimSpace(lbNode.IP)).NotTo(BeEmpty(), "LB 节点 IP 不能为空")
			Expect(strings.TrimSpace(lbNode.Password)).NotTo(BeEmpty(), "LB 节点密码不能为空，请检查对应 TEST_NODE 的 *_PASSWORD 配置")

			masterIPs := []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}
			By(fmt.Sprintf("在 LB 节点 %s 部署 haproxy，后端 master=%s", lbNode.IP, strings.Join(masterIPs, ",")))
			err := utils.SetupHAProxyTCPProxyOnNode(lbNode, masterIPs, "36443")
			Expect(err).NotTo(HaveOccurred(), "应该成功在 LB 节点部署 haproxy")

			clusterName = fmt.Sprintf("test-f5-mgmt-3m1w-%d", time.Now().Unix())
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
			clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
			clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
				Host: lbNode.IP,
				Port: "36443",
			}
			clusterConfig.CustomExtra = &config.ClusterCustomExtra{
				ExtraLoadBalanceIP: lbNode.IP,
			}

			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "应该成功触发管理集群创建")

			kubeconfigPath, err = waitAndVerifyManagementCluster(clusterManager, clusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")
		})

		// Author: xinhongchen
		It("应该验证 3Master1Worker 管理集群健康", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			Expect(kubeconfigPath).NotTo(BeEmpty(), "管理集群 kubeconfig 应该已获取")
			err := verifyManagementClusterHealth(clusterManager, clusterName, kubeconfigPath)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if clusterName != "" {
				delete3M1WManagementCluster(clusterManager, localExecutor, clusterName, kubeconfigPath)
			}
			if configPath != "" {
				_ = clusterManager.CleanupConfig(configPath)
			}
			if nodeConfigPath != "" {
				_ = clusterManager.CleanupConfig(nodeConfigPath)
			}
			if kubeconfigPath != "" {
				_ = clusterManager.CleanupConfig(kubeconfigPath)
			}
			if err := utils.CleanupHAProxyOnNode(lbNode); err != nil {
				GinkgoWriter.Printf("清理 haproxy 配置失败: %v\n", err)
			}
		})
	})

	Describe("引导节点拉起 3Master 管理集群（haproxy 模拟 F5，无 worker）", Label("f5-lb", "3m-mgmt", "post-init", "online"), Ordered, func() {
		var (
			clusterName    string
			configPath     string
			nodeConfigPath string
			kubeconfigPath string
			lbNode         config.NodeInfo
		)

		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			if len(nodes) < 4 {
				Skip(fmt.Sprintf("需要至少 4 个节点（3master 管理集群 3 个 + 1 个 LB 节点），当前 %d 个，跳过", len(nodes)))
			}

			mgmtNodes := make([]config.NodeInfo, 3)
			copy(mgmtNodes, nodes[:3])
			for i := 0; i < 3; i++ {
				mgmtNodes[i].Role = []string{"master/node", "etcd"}
			}
			lbNode = nodes[3]
			Expect(strings.TrimSpace(lbNode.IP)).NotTo(BeEmpty(), "LB 节点 IP 不能为空")
			Expect(strings.TrimSpace(lbNode.Password)).NotTo(BeEmpty(), "LB 节点密码不能为空，请检查对应 TEST_NODE 的 *_PASSWORD 配置")

			masterIPs := []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}
			By(fmt.Sprintf("在 LB 节点 %s 部署 haproxy，后端 master=%s", lbNode.IP, strings.Join(masterIPs, ",")))
			err := utils.SetupHAProxyTCPProxyOnNode(lbNode, masterIPs, "36443")
			Expect(err).NotTo(HaveOccurred(), "应该成功在 LB 节点部署 haproxy")

			clusterName = fmt.Sprintf("test-f5-mgmt-3m-%d", time.Now().Unix())
			clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
			clusterConfig.ExcludeAddons = []string{"openfuyao-system-controller"}
			clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
				Host: lbNode.IP,
				Port: "36443",
			}
			clusterConfig.CustomExtra = &config.ClusterCustomExtra{
				ExtraLoadBalanceIP: lbNode.IP,
			}

			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "应该成功触发管理集群创建")

			kubeconfigPath, err = waitAndVerifyManagementCluster(clusterManager, clusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")
		})

		// Author: luolong817
		It("应该验证 3Master 管理集群健康", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			Expect(kubeconfigPath).NotTo(BeEmpty(), "管理集群 kubeconfig 应该已获取")
			err := verifyManagementClusterHealth(clusterManager, clusterName, kubeconfigPath)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if clusterName != "" {
				nodes := config.LoadTestNodesFromEnv()
				var mgmtNodes []config.NodeInfo
				if len(nodes) >= 3 {
					mgmtNodes = make([]config.NodeInfo, 3)
					copy(mgmtNodes, nodes[:3])
				}
				cleanupManagementCluster(
					clusterManager, localExecutor, clusterName, kubeconfigPath,
					mgmtNodes, configPath, nodeConfigPath,
				)
			}
			if kubeconfigPath != "" {
				_ = clusterManager.CleanupConfig(kubeconfigPath)
			}
			if err := utils.CleanupHAProxyOnNode(lbNode); err != nil {
				GinkgoWriter.Printf("清理 haproxy 配置失败: %v\n", err)
			}
		})
	})

	Describe("引导节点拉起 1Master 管理集群，再拉起 3Master1Worker 业务集群（haproxy 模拟 F5）", Label("f5-lb", "1m-mgmt-3m1w-workload", "post-init", "skip-temporarily", "online"), Ordered, func() {
		var (
			mgmtClusterName        string
			mgmtConfigPath         string
			mgmtNodeConfigPath     string
			mgmtKubeconfigPath     string
			workloadClusterName    string
			workloadConfigPath     string
			workloadNodeConfigPath string
			lbNode                 config.NodeInfo
		)

		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			// 1 管理 + 4 业务(3m1w) + 1 独立 LB（与 F5 模拟一致，不在管理/业务节点上混跑 haproxy）
			if len(nodes) < 6 {
				Skip(fmt.Sprintf("需要至少 6 个节点（管理集群 1 + 业务集群 4 + 独立 LB 1），当前 %d 个，跳过", len(nodes)))
			}

			By("创建 1Master 管理集群")
			mgmtClusterName = fmt.Sprintf("test-f5-mgmt-1m-%d", time.Now().Unix())
			mgmtMaster := nodes[0]
			mgmtMaster.Role = []string{"master/node", "etcd"}
			mgmtClusterConfig := config.NewBKEClusterConfigForMgmt(mgmtClusterName, []config.NodeInfo{mgmtMaster})

			var err error
			mgmtConfigPath, mgmtNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(mgmtClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(mgmtConfigPath, mgmtNodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "应该成功触发管理集群创建")

			mgmtKubeconfigPath, err = waitAndVerifyManagementCluster(clusterManager, mgmtClusterName)
			Expect(err).NotTo(HaveOccurred(), "管理集群应该在引导节点与管理集群自身均达到 Healthy/Ready")

			workloadNodes := make([]config.NodeInfo, 4)
			copy(workloadNodes, nodes[1:5])
			for i := 0; i < 3; i++ {
				workloadNodes[i].Role = []string{"master/node", "etcd"}
			}
			workloadNodes[3].Role = []string{"node"}
			lbNode = nodes[5]
			Expect(strings.TrimSpace(lbNode.IP)).NotTo(BeEmpty(), "LB 节点 IP 不能为空")
			Expect(strings.TrimSpace(lbNode.Password)).NotTo(BeEmpty(), "LB 节点密码不能为空，请检查对应 TEST_NODE 的 *_PASSWORD 配置")

			masterIPs := []string{workloadNodes[0].IP, workloadNodes[1].IP, workloadNodes[2].IP}
			By(fmt.Sprintf("在 LB 节点 %s 部署 haproxy，后端 workload master=%s", lbNode.IP, strings.Join(masterIPs, ",")))
			err = utils.SetupHAProxyTCPProxyOnNode(lbNode, masterIPs, "36443")
			Expect(err).NotTo(HaveOccurred(), "应该成功在 LB 节点部署 haproxy")

			workloadClusterName = fmt.Sprintf("test-f5-workload-3m1w-%d", time.Now().Unix())
			workloadConfig := config.NewDefaultBKEClusterConfig(workloadClusterName, workloadNodes)
			workloadConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
				Host: lbNode.IP,
				Port: "36443",
			}
			workloadConfig.CustomExtra = &config.ClusterCustomExtra{
				ExtraLoadBalanceIP: lbNode.IP,
			}

			workloadConfigPath, workloadNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(workloadConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成业务集群配置文件")

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(workloadConfigPath, workloadNodeConfigPath, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "应该成功触发业务集群创建")

			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
				GinkgoWriter.Printf("当前业务集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "业务集群应该变为 Ready")
		})

		// Author: xinhongchen
		It("应该验证 1Master 管理集群与 3Master1Worker 业务集群健康", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			Expect(mgmtKubeconfigPath).NotTo(BeEmpty(), "管理集群 kubeconfig 应该已获取")
			err := verifyManagementClusterHealth(clusterManager, mgmtClusterName, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred())
		})

		AfterAll(func() {
			if workloadClusterName != "" && mgmtKubeconfigPath != "" {
				delete3M1WWorkloadCluster(clusterManager, localExecutor, workloadClusterName, mgmtKubeconfigPath)
			}
			if workloadConfigPath != "" {
				_ = clusterManager.CleanupConfig(workloadConfigPath)
			}
			if workloadNodeConfigPath != "" {
				_ = clusterManager.CleanupConfig(workloadNodeConfigPath)
			}

			if mgmtClusterName != "" {
				nodes := config.LoadTestNodesFromEnv()
				var mgmtNodes []config.NodeInfo
				if len(nodes) >= 1 {
					mgmtNodes = []config.NodeInfo{nodes[0]}
				}
				cleanupManagementCluster(
					clusterManager, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
					mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
				)
			}
			if mgmtKubeconfigPath != "" {
				_ = clusterManager.CleanupConfig(mgmtKubeconfigPath)
			}
			if err := utils.CleanupHAProxyOnNode(lbNode); err != nil {
				GinkgoWriter.Printf("清理 haproxy 配置失败: %v\n", err)
			}
		})
	})
})
