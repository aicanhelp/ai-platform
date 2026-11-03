package custom_installation

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/installation"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var _ = InstallationSIGDescribe("BKE Cluster Installation", func() {
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

	Describe("创建 1Master1Worker 管理集群和业务集群并使用自定义证书签名", Label("1m1w-custom", "certificate", "post-init", "online"), Ordered, func() {
		var (
			mgmtCluster     *Cluster1M1WConfig
			workloadCluster *Cluster1M1WConfig
			certGenerator   *utils.CertificateConfigGenerator
		)

		BeforeAll(func() {
			// 步骤 1: 在引导节点上生成证书链文件
			By("在引导节点上生成证书链文件")
			err := utils.GenerateCertificateChainOnRemote(sshExecutor, "/etc/openFuyao/certs")
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书链文件")
			GinkgoWriter.Printf("证书链文件在引导节点上生成于: %s\n", "/etc/openFuyao/certs")

			// 步骤 2: 在引导节点上生成证书配置
			By("在引导节点上生成证书配置")
			certConfigDir := "/etc/openFuyao/certs/cert_config"
			certGenerator = utils.NewCertificateConfigGenerator(certConfigDir)
			err = certGenerator.GenerateAllCertificateConfigs()
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书配置")
			GinkgoWriter.Printf("证书配置在引导节点上生成于: %s\n", certConfigDir)

			// 步骤 3: 使用辅助函数创建管理集群
			var err2 error
			mgmtCluster, err2 = Create1M1WManagementCluster(
				clusterManager,
				localExecutor,
				"test-mgmt-1m1w-cert",
				true, // includeClusterAPI
			)
			Expect(err2).NotTo(HaveOccurred(), "应该成功创建管理集群")

			// 步骤 4: 使用辅助函数创建业务集群
			workloadCluster, err2 = Create1M1WWorkloadCluster(
				clusterManager,
				localExecutor,
				mgmtCluster.KubeconfigPath,
				"test-workload-1m1w-cert",
			)
			Expect(err2).NotTo(HaveOccurred(), "应该成功创建业务集群")
		})
		/*
			用例名称：验证1master-1worker集群自定义证书导入
			前置条件：
				1.在引导节点上生成证书链文件
				2.在引导节点上生成证书配置
				3.创建1master1worker的管理集群
				4.创建1master1worker的业务集群
			用例步骤：
				1.检查管理集群的master节点的证书配置
				2.检查业务集群的master节点的证书配置
			预期结果：
		*/
		// Author: luolong817
		It("应该验证所有 master 节点的证书签名", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 4), "需要至少 4 个节点（管理集群 2 个 + 业务集群 2 个）")

			// 验证管理集群的 master 节点（节点 1, 索引 0）
			By("验证管理集群 master 节点的证书签名")
			managerMasterNode := nodes[0]
			By(fmt.Sprintf("验证管理集群 master 节点 1 (%s) 的证书签名", managerMasterNode.IP))
			verifier := utils.NewCertificateVerifierWithSSH("/etc/kubernetes/pki", localExecutor, managerMasterNode.IP, managerMasterNode.Username, managerMasterNode.Password)
			results, err := verifier.VerifyCertificateChain()
			Expect(err).NotTo(HaveOccurred(), "应该成功验证管理集群 master 节点 1 的证书链")
			GinkgoWriter.Printf("=== 管理集群 Master 节点 %d (%s) 证书验证结果 ===\n", 1, managerMasterNode.IP)
			verifier.PrintResults(results)
			// 检查所有验证项是否成功
			for _, item := range results.Items {
				if !item.Success {
					GinkgoWriter.Printf("管理集群 master 节点 1 证书验证失败: %s - %v\n", item.Name, item.Error)
				}
				Expect(item.Success).To(BeTrue(), fmt.Sprintf("管理集群 master 节点 1 证书验证应该成功: %s", item.Name))
			}

			// 验证业务集群的 master 节点（节点 3，索引 2）
			By("验证业务集群 master 节点的证书签名")
			workloadMasterNode := nodes[2]
			By(fmt.Sprintf("验证业务集群 master 节点 3 (%s) 的证书签名", workloadMasterNode.IP))
			verifier2 := utils.NewCertificateVerifierWithSSH("/etc/kubernetes/pki", localExecutor, workloadMasterNode.IP, workloadMasterNode.Username, workloadMasterNode.Password)
			results2, err2 := verifier2.VerifyCertificateChain()
			Expect(err2).NotTo(HaveOccurred(), "应该成功验证业务集群 master 节点 3 的证书链")
			GinkgoWriter.Printf("=== 业务集群 Master 节点 3 (%s) 证书验证结果 ===\n", workloadMasterNode.IP)
			verifier.PrintResults(results2)
			// 检查所有验证项是否成功
			for _, item := range results2.Items {
				if !item.Success {
					GinkgoWriter.Printf("业务集群 master 节点 3 证书验证失败: %s - %v\n", item.Name, item.Error)
				}
				Expect(item.Success).To(BeTrue(), fmt.Sprintf("业务集群 master 节点 3 证书验证应该成功: %s", item.Name))
			}
		})

		AfterAll(func() {
			// 清理业务集群
			if workloadCluster != nil && mgmtCluster != nil {
				Delete1M1WWorkloadCluster(
					clusterManager,
					localExecutor,
					workloadCluster.ClusterName,
					mgmtCluster.KubeconfigPath,
				)
				// 清理配置文件
				if workloadCluster.ConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadCluster.ConfigPath); err != nil {
						GinkgoWriter.Printf("清理业务集群配置文件失败: %v\n", err)
					}
				}
				if workloadCluster.NodeConfigPath != "" {
					if err := clusterManager.CleanupConfig(workloadCluster.NodeConfigPath); err != nil {
						GinkgoWriter.Printf("Failed to cleanup workload node config file: %v\n", err)
					}
				}
			}

			// 清理管理集群
			if mgmtCluster != nil {
				Delete1M1WManagementCluster(
					clusterManager,
					localExecutor,
					mgmtCluster.ClusterName,
					mgmtCluster.KubeconfigPath,
				)
				// 清理配置文件
				if mgmtCluster.ConfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.ConfigPath); err != nil {
						GinkgoWriter.Printf("清理管理集群配置文件失败: %v\n", err)
					}
				}
				if mgmtCluster.NodeConfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.NodeConfigPath); err != nil {
						GinkgoWriter.Printf("Failed to cleanup mgmt node config file: %v\n", err)
					}
				}
				// 清理 kubeconfig 文件
				if mgmtCluster.KubeconfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.KubeconfigPath); err != nil {
						GinkgoWriter.Printf("清理管理集群 kubeconfig 文件失败: %v\n", err)
					}
				}
			}

			// 清理证书目录
			By("从引导节点清理证书目录")
			cmd := "rm -rf /etc/openFuyao/certs/*"
			localExecutor.Exec(cmd)
			if _, err := localExecutor.Exec(cmd); err != nil {
				GinkgoWriter.Printf("清理证书目录失败: %v\n", err)
			}
		})
	})

	Describe("创建 3Master1Worker 管理集群并使用自定义证书签名", Label("3m1w-mgmt-custom", "certificate", "post-init", "online"), Ordered, func() {
		var (
			mgmtCluster   *utils.Cluster3M1WConfig
			certGenerator *utils.CertificateConfigGenerator
		)

		BeforeAll(func() {
			// 步骤 1: 在引导节点上生成证书链文件
			By("在引导节点上生成证书链文件")
			err := utils.GenerateCertificateChainOnRemote(sshExecutor, "/etc/openFuyao/certs")
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书链文件")
			GinkgoWriter.Printf("证书链文件在引导节点上生成于: %s\n", "/etc/openFuyao/certs")

			// 步骤 2: 在引导节点上生成证书配置
			By("在引导节点上生成证书配置")
			certConfigDir := "/etc/openFuyao/certs/cert_config"
			certGenerator = utils.NewCertificateConfigGenerator(certConfigDir)
			err = certGenerator.GenerateAllCertificateConfigs()
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书配置")
			GinkgoWriter.Printf("证书配置在引导节点上生成于: %s\n", certConfigDir)

			// 步骤 3: 使用辅助函数创建 3Master1Worker 管理集群
			var err2 error
			mgmtCluster, err2 = utils.Create3M1WManagementCluster(
				clusterManager,
				localExecutor,
				"test-mgmt-3m1w-cert",
				true, // includeClusterAPI
			)
			Expect(err2).NotTo(HaveOccurred(), "应该成功创建 3Master1Worker 管理集群")
		})

		/*
			用例名称：验证 3master1worker 管理集群自定义证书导入
			前置条件：
				1.在引导节点上生成证书链文件
				2.在引导节点上生成证书配置
				3.创建 3master1worker 的管理集群
			用例步骤：
				1.检查管理集群 3 个 master 节点的证书配置
			预期结果：管理集群所有 master 节点证书验证项全部通过
		*/
		// Author: xinhongchen
		It("应该验证 3Master1Worker 管理集群所有 master 节点的证书签名", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 4), "需要至少 4 个节点（管理集群 3 个 master + 1 个 worker）")

			By("验证管理集群所有 master 节点的证书签名")
			for i := 0; i < 3; i++ {
				masterNode := nodes[i]
				By(fmt.Sprintf("验证管理集群 master 节点 %d (%s) 的证书签名", i+1, masterNode.IP))
				verifier := utils.NewCertificateVerifierWithSSH("/etc/kubernetes/pki", localExecutor, masterNode.IP, masterNode.Username, masterNode.Password)
				results, err := verifier.VerifyCertificateChain()
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("应该成功验证管理集群 master 节点 %d 的证书链", i+1))
				GinkgoWriter.Printf("=== 管理集群 Master 节点 %d (%s) 证书验证结果 ===\n", i+1, masterNode.IP)
				verifier.PrintResults(results)
				for _, item := range results.Items {
					if !item.Success {
						GinkgoWriter.Printf("管理集群 master 节点 %d 证书验证失败: %s - %v\n", i+1, item.Name, item.Error)
					}
					Expect(item.Success).To(BeTrue(), fmt.Sprintf("管理集群 master 节点 %d 证书验证应该成功: %s", i+1, item.Name))
				}
			}
		})

		AfterAll(func() {
			// 清理管理集群
			if mgmtCluster != nil {
				utils.Delete3M1WManagementCluster(
					clusterManager,
					localExecutor,
					mgmtCluster.ClusterName,
					mgmtCluster.KubeconfigPath,
				)
				if mgmtCluster.ConfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.ConfigPath); err != nil {
						GinkgoWriter.Printf("清理管理集群配置文件失败: %v\n", err)
					}
				}
				if mgmtCluster.NodeConfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.NodeConfigPath); err != nil {
						GinkgoWriter.Printf("Failed to cleanup mgmt node config file: %v\n", err)
					}
				}
				if mgmtCluster.KubeconfigPath != "" {
					if err := clusterManager.CleanupConfig(mgmtCluster.KubeconfigPath); err != nil {
						GinkgoWriter.Printf("清理管理集群 kubeconfig 文件失败: %v\n", err)
					}
				}
			}

			// 清理证书目录
			By("从引导节点清理证书目录")
			cmd := "rm -rf /etc/openFuyao/certs/*"
			localExecutor.Exec(cmd)
			if _, err := localExecutor.Exec(cmd); err != nil {
				GinkgoWriter.Printf("清理证书目录失败: %v\n", err)
			}
		})
	})

	Describe("创建 1Master 管理集群和 3Master1Worker 业务集群并使用自定义证书签名", Label("1m-mgmt-3m1w-workload-custom", "certificate", "post-init", "skip-temporarily", "online"), Ordered, func() {
		var (
			mgmtClusterName        string
			mgmtConfigPath         string
			mgmtNodeConfigPath     string
			mgmtKubeconfigPath     string
			workloadClusterName    string
			workloadConfigPath     string
			workloadNodeConfigPath string
			certGenerator          *utils.CertificateConfigGenerator
		)

		BeforeAll(func() {
			By("在引导节点上生成证书链文件")
			err := utils.GenerateCertificateChainOnRemote(sshExecutor, "/etc/openFuyao/certs")
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书链文件")

			By("在引导节点上生成证书配置")
			certConfigDir := "/etc/openFuyao/certs/cert_config"
			certGenerator = utils.NewCertificateConfigGenerator(certConfigDir)
			err = certGenerator.GenerateAllCertificateConfigs()
			Expect(err).NotTo(HaveOccurred(), "应该成功生成证书配置")

			nodes := config.LoadTestNodesFromEnv()
			if len(nodes) < 5 {
				Skip(fmt.Sprintf("跳过: 管理集群+业务集群场景需要至少 5 个节点，当前只有 %d 个", len(nodes)))
			}

			By("创建 1Master 管理集群")
			mgmtClusterName = fmt.Sprintf("test-mgmt-1m-cert-%d", time.Now().Unix())
			mgmtMaster := nodes[0]
			mgmtMaster.Role = []string{"master/node", "etcd"}
			mgmtClusterConfig := config.NewBKEClusterConfigForMgmt(mgmtClusterName, []config.NodeInfo{mgmtMaster})

			mgmtConfigPath, mgmtNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(mgmtClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成管理集群配置文件")

			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(mgmtConfigPath, mgmtNodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred(), "应该成功触发管理集群创建")

			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(mgmtClusterName, "")
				GinkgoWriter.Printf("当前管理集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus == "Ready"
			}, mgmtInstallTimeout, pollInterval).Should(BeTrue(), "管理集群应该变为 Ready")

			mgmtChecker := utils.NewClusterCheckerWithParentKubeconfig(clusterManager.SSHExecutor(), mgmtClusterName, "")
			mgmtKubeconfigPath, err = mgmtChecker.SaveKubeconfigToFile()
			Expect(err).NotTo(HaveOccurred(), "应该成功获取管理集群 kubeconfig")

			By("在管理集群上创建 3Master1Worker 业务集群")
			workloadClusterName = fmt.Sprintf("test-workload-3m1w-cert-%d", time.Now().Unix())
			workloadNodes := make([]config.NodeInfo, 4)
			copy(workloadNodes, nodes[1:5])
			for i := 0; i < 3; i++ {
				workloadNodes[i].Role = []string{"master/node", "etcd"}
			}
			workloadNodes[3].Role = []string{"node"}

			workloadClusterConfig := config.NewDefaultBKEClusterConfig(workloadClusterName, workloadNodes)
			workloadConfigPath, workloadNodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(workloadClusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成业务集群配置文件")

			err = clusterManager.CreateClusterWithConfig(workloadConfigPath, workloadNodeConfigPath, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "应该成功创建业务集群")

			Eventually(func() bool {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(workloadClusterName, mgmtKubeconfigPath)
				GinkgoWriter.Printf("当前业务集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus == "Ready"
			}, installTimeout, pollInterval).Should(BeTrue(), "业务集群应该变为 Ready")
		})

		// Author: xinhongchen
		It("应该验证管理和业务集群所有节点的证书签名", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 5), "需要至少 5 个节点（管理集群 1 个 + 业务集群 4 个）")

			By(fmt.Sprintf("验证管理集群 master 节点 (%s) 的证书签名", nodes[0].IP))
			verifier := utils.NewCertificateVerifierWithSSH("/etc/kubernetes/pki", localExecutor, nodes[0].IP, nodes[0].Username, nodes[0].Password)
			results, err := verifier.VerifyCertificateChain()
			Expect(err).NotTo(HaveOccurred(), "应该成功验证管理集群 master 节点证书链")
			verifier.PrintResults(results)
			for _, item := range results.Items {
				Expect(item.Success).To(BeTrue(), fmt.Sprintf("管理集群 master 节点证书验证应该成功: %s", item.Name))
			}

			for i := 1; i < 5; i++ {
				node := nodes[i]
				isWorker := i == 4
				roleDesc := "master"
				if isWorker {
					roleDesc = "worker"
				}

				By(fmt.Sprintf("验证业务集群 %s 节点 %d (%s) 的证书签名", roleDesc, i+1, node.IP))
				verifier := utils.NewCertificateVerifierWithSSH("/etc/kubernetes/pki", localExecutor, node.IP, node.Username, node.Password)
				var results *utils.VerificationResult
				var err error
				if isWorker {
					results, err = verifier.VerifyWorkerCertificateChain()
				} else {
					results, err = verifier.VerifyCertificateChain()
				}
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("应该成功执行业务集群 %s 节点 %d 证书校验", roleDesc, i+1))
				verifier.PrintResults(results)

				for _, item := range results.Items {
					Expect(item.Success).To(BeTrue(), fmt.Sprintf("业务集群 %s 节点 %d 证书验证应该成功: %s", roleDesc, i+1, item.Name))
				}
			}
		})

		AfterAll(func() {
			if workloadClusterName != "" && mgmtKubeconfigPath != "" {
				utils.Delete3M1WWorkloadCluster(clusterManager, localExecutor, workloadClusterName, mgmtKubeconfigPath)
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

			if mgmtClusterName != "" {
				mgmtNodes := loadMgmtNodesFromEnv(4)
				utils.CleanupManagementClusterUnified(
					clusterManager, localExecutor, mgmtClusterName, mgmtKubeconfigPath,
					mgmtNodes, mgmtConfigPath, mgmtNodeConfigPath,
				)
			}
			if mgmtKubeconfigPath != "" {
				if err := clusterManager.CleanupConfig(mgmtKubeconfigPath); err != nil {
					GinkgoWriter.Printf("清理管理集群 kubeconfig 文件失败: %v\n", err)
				}
			}

			By("从引导节点清理证书目录")
			cmd := "rm -rf /etc/openFuyao/certs/*"
			localExecutor.Exec(cmd)
			if _, err := localExecutor.Exec(cmd); err != nil {
				GinkgoWriter.Printf("清理证书目录失败: %v\n", err)
			}
		})
	})

})

var _ = InstallationSIGDescribe("证书异常场景", Label("certificate", "cert-abnormal", "post-init", "online"), Ordered, func() {
	var (
		sshExecutor        *executor.SSHExecutor
		localExecutor      *executor.LocalExecutor
		clusterManager     *utils.ClusterManager
		mgmtCluster        *Cluster1M1WConfig
		workloadCluster    *Cluster1M1WConfig
		workloadMasterNode config.NodeInfo
		workloadMasterTime string
		apiserverKeyBacked bool
		certConfigDir      = "/etc/openFuyao/certs/cert_config"
	)

	execOnWorkloadMaster := func(cmd string) *executor.ExecResult {
		result, err := utils.ExecuteCommandOnNode(workloadMasterNode, cmd)
		Expect(err).NotTo(HaveOccurred(), "command should execute on workload master: %s", cmd)
		Expect(result).NotTo(BeNil())
		return result
	}

	waitBusinessClusterHealthy := func(clusterName, kubeconfigPath string) {
		Eventually(func() bool {
			phase, state, clusterStatus, err := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
			if err != nil {
				GinkgoWriter.Printf("get business cluster %s status failed: %v\n", clusterName, err)
				return false
			}
			GinkgoWriter.Printf("[cert-abnormal %s] phase=%s state=%s clusterStatus=%s\n", clusterName, phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus == "Ready"
		}, installTimeout, pollInterval).Should(BeTrue(), "business cluster should become Ready")
	}

	BeforeAll(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST must be set")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD must be set")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "guide node SSH should connect")

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "kube client config should load")
		kubeClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "kube client should initialize")
		clusterManager = utils.NewClusterManager(sshExecutor, kubeClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)

		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 4), "certificate abnormal cases require at least 4 nodes")
		workloadMasterNode = nodes[2]

		By("准备自定义证书链和完整 CSR/签发策略配置")
		_, _ = sshExecutor.Exec("rm -rf /etc/openFuyao/certs/* && mkdir -p /etc/openFuyao/certs/cert_config")
		err = utils.GenerateCertificateChainOnRemote(sshExecutor, "/etc/openFuyao/certs")
		Expect(err).NotTo(HaveOccurred(), "should generate global CA certificate chain on guide node")
		certGenerator := utils.NewCertificateConfigGenerator(certConfigDir)
		err = certGenerator.GenerateAllCertificateConfigs()
		Expect(err).NotTo(HaveOccurred(), "should generate all certificate CSR and policy configs")

		By("创建 1M1W 管理集群作为业务集群父集群")
		mgmtCluster, err = Create1M1WManagementCluster(clusterManager, localExecutor, "cert-abnormal-mgmt", true)
		Expect(err).NotTo(HaveOccurred(), "management cluster should be created")
	})

	AfterAll(func() {
		if workloadMasterTime != "" {
			_, _ = utils.ExecuteCommandOnNode(workloadMasterNode, fmt.Sprintf("date -s @%s >/dev/null 2>&1 || true; systemctl restart chronyd >/dev/null 2>&1 || true", workloadMasterTime))
		}
		if apiserverKeyBacked {
			_, _ = utils.ExecuteCommandOnNode(workloadMasterNode, "if [ -f /etc/kubernetes/pki/apiserver.key.e2e-bak ]; then mv -f /etc/kubernetes/pki/apiserver.key.e2e-bak /etc/kubernetes/pki/apiserver.key; fi")
		}
		if workloadCluster != nil && mgmtCluster != nil {
			Delete1M1WWorkloadCluster(clusterManager, localExecutor, workloadCluster.ClusterName, mgmtCluster.KubeconfigPath)
			if workloadCluster.ConfigPath != "" {
				_ = clusterManager.CleanupConfig(workloadCluster.ConfigPath)
			}
			if workloadCluster.NodeConfigPath != "" {
				_ = clusterManager.CleanupConfig(workloadCluster.NodeConfigPath)
			}
		}
		if mgmtCluster != nil {
			Delete1M1WManagementCluster(clusterManager, localExecutor, mgmtCluster.ClusterName, mgmtCluster.KubeconfigPath)
			if mgmtCluster.ConfigPath != "" {
				_ = clusterManager.CleanupConfig(mgmtCluster.ConfigPath)
			}
			if mgmtCluster.NodeConfigPath != "" {
				_ = clusterManager.CleanupConfig(mgmtCluster.NodeConfigPath)
			}
			if mgmtCluster.KubeconfigPath != "" {
				_ = clusterManager.CleanupConfig(mgmtCluster.KubeconfigPath)
			}
		}
		if sshExecutor != nil {
			_, _ = sshExecutor.Exec("rm -rf /etc/openFuyao/certs/*")
			_ = sshExecutor.Close()
		}
	})

	/*
		用例名称： 证书异常场景-无apiserver-csr.json验证
		前置条件：1.准备引导节点
		2.准备证书配置（参考用户指南）
		用例步骤： 1.删除apiserver-csr.json
		2.创建业务集群
		预期结果：使用默认配置，集群可以成功创建
	*/
	// Author: ZYH-ang
	It("无 apiserver-csr.json 时使用默认配置创建业务集群成功", Label("required_nodes:4", "P0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		By("删除 apiserver-csr.json，验证缺少该可选 CSR 时回退内置默认配置")
		result, err := sshExecutor.Exec(fmt.Sprintf("rm -f %s/apiserver-csr.json && test ! -f %s/apiserver-csr.json", certConfigDir, certConfigDir))
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0), result.Stderr)

		By("创建 1M1W 业务集群")
		workload, err := Create1M1WWorkloadCluster(clusterManager, localExecutor, mgmtCluster.KubeconfigPath, "cert-abnormal-workload")
		Expect(err).NotTo(HaveOccurred(), "business cluster should be created without apiserver-csr.json")
		workloadCluster = workload
		waitBusinessClusterHealthy(workloadCluster.ClusterName, mgmtCluster.KubeconfigPath)
	})

	/*
		用例名称： 证书异常场景-证书过期测试
		前置条件：1.准备引导节点
		2.准备证书配置（参考用户指南）
		用例步骤：1.修改系统时间至证书到期日之后
		2.测试组件通信
		预期结果：证书已过期，无法访问集群，报错信息：tls: failed to verify certificate: x509: certificate has expired or is not yet valid
	*/
	// Author: ZYH-ang
	It("证书过期后组件通信返回 x509 过期错误", Label("required_nodes:4", "P0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		Expect(workloadCluster).NotTo(BeNil(), "business cluster should be created by the previous case")

		By("记录业务 master 当前时间并调整到 apiserver 证书过期时间之后")
		result := execOnWorkloadMaster("date +%s")
		Expect(result.ExitCode).To(Equal(0), result.Stderr)
		workloadMasterTime = strings.TrimSpace(result.Stdout)

		cmd := `set -e
EXPIRED_EPOCH=$(openssl x509 -in /etc/kubernetes/pki/apiserver.crt -noout -enddate | cut -d= -f2 | xargs -I{} date -d '{} + 1 day' +%s)
date -s "@$EXPIRED_EPOCH" >/dev/null
kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes 2>&1 || true`
		result = execOnWorkloadMaster(cmd)
		out := strings.ToLower(result.Stdout + result.Stderr)
		GinkgoWriter.Printf("kubectl output after certificate expiry time shift:\n%s\n", result.Stdout+result.Stderr)
		Expect(out).To(SatisfyAny(
			ContainSubstring("certificate has expired"),
			ContainSubstring("not yet valid"),
			ContainSubstring("x509"),
		), "kubectl should report an x509 certificate validity error")

		By("恢复业务 master 系统时间")
		result = execOnWorkloadMaster(fmt.Sprintf("date -s @%s >/dev/null 2>&1 || true; systemctl restart chronyd >/dev/null 2>&1 || true; kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes 2>&1 || true", workloadMasterTime))
		GinkgoWriter.Printf("kubectl output after restoring time:\n%s\n", result.Stdout+result.Stderr)
		workloadMasterTime = ""
	})

	/*
		用例名称： 证书异常场景-使用不匹配的tls.key/tls.crt
		前置条件：1.准备引导节点
		2.准备证书配置（参考用户指南）
		用例步骤：1.修改apiserver.key导致不匹配
		2.连接集群
		预期结果：对集群没有任何影响
	*/
	// Author: ZYH-ang
	It("磁盘上的 apiserver.key 与 apiserver.crt 不匹配时不影响已运行集群连接", Label("required_nodes:4", "P0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		Expect(workloadCluster).NotTo(BeNil(), "business cluster should be created by the previous case")

		By("备份 apiserver.key 并替换为不匹配的私钥")
		cmd := `set -e
cp -f /etc/kubernetes/pki/apiserver.key /etc/kubernetes/pki/apiserver.key.e2e-bak
openssl genrsa -out /etc/kubernetes/pki/apiserver.key 2048 >/dev/null 2>&1
CRT_MD5=$(openssl x509 -noout -modulus -in /etc/kubernetes/pki/apiserver.crt | openssl md5 | awk '{print $2}')
KEY_MD5=$(openssl rsa -noout -modulus -in /etc/kubernetes/pki/apiserver.key 2>/dev/null | openssl md5 | awk '{print $2}')
echo "crt=$CRT_MD5 key=$KEY_MD5"
[ "$CRT_MD5" != "$KEY_MD5" ]
kubectl --kubeconfig /etc/kubernetes/admin.conf get nodes`
		result := execOnWorkloadMaster(cmd)
		apiserverKeyBacked = true
		Expect(result.ExitCode).To(Equal(0), result.Stderr)
		Expect(result.Stdout).To(ContainSubstring("crt="))
		Expect(result.Stdout).To(ContainSubstring("key="))
		Expect(result.Stdout).To(ContainSubstring(" Ready"), "running apiserver should keep serving with the already loaded key pair")

		By("恢复 apiserver.key")
		result = execOnWorkloadMaster("mv -f /etc/kubernetes/pki/apiserver.key.e2e-bak /etc/kubernetes/pki/apiserver.key")
		Expect(result.ExitCode).To(Equal(0), result.Stderr)
		apiserverKeyBacked = false
	})
})
