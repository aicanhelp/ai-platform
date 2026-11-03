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

var _ = SIGDescribe("Large Scale Cluster Installation", func() {
	// InstallationItTimeout is the default Spec timeout for tests under e2e/installatio

	var (
		sshExecutor    *executor.SSHExecutor
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

	})
	Describe("calico-typha 组件", Label("calico-typha", "with-management-cluster", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-addon-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点")

			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			clusterConfig = config.NewBKEClusterConfigForCalicoTest(clusterName, []config.NodeInfo{masterNode})
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
		// 用例名称：支持安装 calico-typha 组件
		// 用例步骤：1) 在引导节点创建集群配置时开启 calico addon 并设置 allowTypha=true；2) 执行 `bke cluster create` 创建集群并等待 Healthy；3) 检查 kube-system 命名空间 Pod 就绪；4) 验证 calico-typha 相关 Pod 存在。
		// 预期结果：集群网络组件部署成功，且 typha 组件能够被正确安装与发现。
		It("支持安装 calico-typha 组件", Label("required_nodes:1", "calico-typha", "skip-temporarily"), func() {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "calico",
				Version: "v3.31.3",
				Param: map[string]string{
					"allowTypha":            "true",
					"typhaReplicas":         "1",
					"calicoMode":            "vxlan",
					"ipAutoDetectionMethod": "skip-interface=nerdctl*",
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

			By("验证 calico-typha 组件已安装")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 kube-system 命名空间下的 Pod 状态
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("kube-system", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 kube-system Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "kube-system 命名空间下的 Pod 应该全部就绪")

			// 进一步验证 typha 组件
			Eventually(func() bool {
				for _, labelKey := range []string{"k8s-app", "app.kubernetes.io/name"} {
					nodeName, _ := checker.GetPodNodeName("kube-system", fmt.Sprintf("%s=calico-typha", labelKey))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "应该能找到 calico-typha 组件")
		})

		// 用例名称：未配置allowTypha，默认安装 calico 但不安装 typha 组件
		// 用例步骤：1) 在 calico addon 中不显式配置 allowTypha；2) 通过引导节点创建集群并等待 Healthy；3) 检查 kube-system Pod 中 calico 相关 Pod 存在；4) 断言 typha 相关 Pod 不存在。
		// 预期结果：calico 正常安装运行，但 typha 未被部署。
		It("未配置allowTypha，默认安装 calico 但不安装 typha 组件", Label("required_nodes:1", "calico-typha-no-typha"), func() {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "calico",
				Version: "v3.31.3",
				Param: map[string]string{
					"calicoMode":            "vxlan",
					"ipAutoDetectionMethod": "skip-interface=nerdctl*",
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
			By("验证 calico 组件已安装但 typha 组件未安装")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 kube-system 命名空间下的 Pod 状态，应该有 calico 相关的 Pod 但没有 typha 相关的 Pod
			Eventually(func() bool {
				podStatuses, err := checker.GetNamespacePodStatuses("kube-system")
				if err != nil {
					return false
				}
				hasCalico := false
				hasTypha := false
				for podName := range podStatuses {
					if strings.Contains(podName, "calico") {
						hasCalico = true
					}
					if strings.Contains(podName, "typha") {
						hasTypha = true
					}
				}
				if !hasCalico {
					GinkgoWriter.Printf("未找到 calico 相关 Pod\n")
				}
				if hasTypha {
					GinkgoWriter.Printf("意外找到 typha 相关 Pod\n")
				}
				return hasCalico && !hasTypha
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "应该安装 calico 但不安装 typha 组件")
		})
	})

	Describe("nodelocaldns 组件", Label("nodelocaldns", "with-management-cluster", "online"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-addon-%d", time.Now().Unix())
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点")

			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
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
		// 用例名称：支持安装 nodelocaldns 组件
		// 用例步骤：1) 配置 nodelocaldns addon 并设置 localdns IP；2) 通过引导节点创建集群并等待 Healthy；3) 校验 nodelocaldns 相关 Pod 就绪/部署到集群。
		// 预期结果：nodelocaldns 组件安装成功，Pod 运行正常。
		It("支持安装 nodelocaldns 组件", Label("required_nodes:1", "nodelocaldns"), func() {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "nodelocaldns",
				Version: "v1.26.4",
				Param: map[string]string{
					"localdns": "10.96.0.20",
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

			By("验证 nodelocaldns 组件已安装")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 kube-system 命名空间下的 Pod 状态
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("kube-system", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 nodelocaldns Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "kube-system 命名空间下的 Pod 应该全部就绪")
		})

		// 用例名称：未配置localdns安装失败
		// 用例步骤：1) 配置 nodelocaldns addon，但不提供 localdns 参数（空 Param）；2) 通过引导节点创建集群并等待 Healthy；3) 检查 kube-system 命名空间 Pod 列表，断言不包含 nodelocaldns 相关 Pod。
		// 预期结果：nodelocaldns 因缺少必填参数而未成功部署，kube-system 中不出现 nodelocaldns 相关 Pod。
		It("未配置localdns安装失败", Label("required_nodes:1", "nodelocaldns-failure"), func() {
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "nodelocaldns",
				Version: "v1.26.4",
				Param:   map[string]string{}, // 不提供 localdns 参数
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

			By("验证 nodelocaldns 组件安装失败")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 kube-system 命名空间下的 Pod 状态，应该没有 nodelocaldns 相关的 Pod
			Eventually(func() bool {
				podStatuses, err := checker.GetNamespacePodStatuses("kube-system")
				if err != nil {
					return false
				}
				for podName := range podStatuses {
					if strings.Contains(podName, "nodelocaldns") {
						GinkgoWriter.Printf("意外找到 nodelocaldns 相关 Pod: %s\n", podName)
						return false
					}
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "kube-system 命名空间下不应该有 nodelocaldns 相关的 Pod")
		})
	})

	Describe("victoriametrics 组件", Label("victoriametrics-controller", "with-management-cluster", "online"), func() {
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
			Expect(len(nodes)).To(BeNumerically(">=", 2), "victoriametrics组件安装测试至少需要 2 个节点")

			// 第一个节点作为 Master
			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			// 第二个节点作为 Worker
			workerNode := nodes[1]
			workerNode.Role = []string{"node"}
			workerNode.Labels = []config.NodeLabel{
				{Key: "monitoring.victoria.com/vmstorage", Value: "vmstorage"},
				{Key: "monitoring.victoria.com/metrics", Value: "kube-state-metrics"},
				{Key: "monitoring.victoria.com/vminsert", Value: "vminsert"},
				{Key: "monitoring.victoria.com/vmalert", Value: "vmalert"},
				{Key: "monitoring.victoria.com/vmselect", Value: "vmselect"},
				{Key: "monitoring.victoria.com/vmagent", Value: "vmagent"},
				{Key: "monitoring.victoria.com/vmalertmanager", Value: "vmalertmanager"},
				{Key: "monitoring.victoria.com/vm-grafana", Value: "vm-grafana"},
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

		// 用例名称：支持自定义参数安装 victoriametrics-controller 组件
		// 用例步骤：1) 为 victoriametrics-controller addon 提供自定义参数（vmAgent/vmSelect/vmInsert/vmStorage 等副本与资源）；2) 通过引导节点创建集群并等待 Healthy；3) 校验 `vmks` 命名空间 Pod 就绪；4) 校验关键组件副本数与自定义参数一致。
		// 预期结果：自定义参数生效，vmks 命名空间中各组件副本数符合预期。
		It("支持自定义参数安装 victoriametrics-controller 组件", Label("required_nodes:2", "victoriametrics-controller-free"), func() {
			// 定义要测试的参数
			testParams := map[string]string{
				"useVMSingle":                  "false",
				"vmSingleStorageSize":          "50Gi",
				"vmAgentAllowStatefulSet":      "true",
				"vmAgentCpuCount":              "0.5",
				"vmAgentMemorySize":            "2Gi",
				"vmAgentStorageSize":           "10Gi",
				"vmAgentReplicaCount":          "1",
				"vmAgentShareCount":            "1",
				"vmAgentScrapeInterval":        "20s",
				"vmSelectCpuCount":             "0.5",
				"vmSelectMemorySize":           "2Gi",
				"vmSelectStorageSize":          "10Gi",
				"vmSelectReplicaCount":         "1",
				"vmStorageCPUCount":            "0.5",
				"vmStorageMemorySize":          "2Gi",
				"vmStorageStorageSize":         "20Gi",
				"vmStorageReplicaCount":        "1",
				"vmInsertReplicaCount":         "1",
				"vmInsertCpuCount":             "0.5",
				"vmInsertMemorySize":           "2Gi",
				"vmAlertReplicaCount":          "1",
				"vmAlertCpuCount":              "0.5",
				"vmAlertMemorySize":            "1Gi",
				"vmAlertManagerReplicaCount":   "1",
				"vmAlertManagerCpuCount":       "0.5",
				"vmAlertManagerMemorySize":     "1Gi",
				"vmClusterRetentionPeriod":     "15d",
				"vmClusterReplicationFactor":   "1",
				"grafanaNodePort":              "30010",
				"kubeStateMetricsAutoSharding": "true",
				"kubeStateMetricsReplicaCount": "1",
				"kubeStateMetricsCpuCount":     "0.5",
				"kubeStateMetricsMemorySize":   "2Gi",
			}

			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "victoriametrics-controller",
				Version: "latest",
				Param:   testParams,
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

			By("验证 victoriametrics-controller 组件已安装")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 victoriametrics-controller 命名空间下的 Pod 状态
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("vmks", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 victoriametrics-controller Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "vmks 命名空间下的 Pod 应该全部就绪")

			// 验证自定义参数
			By("验证特定组件的副本数配置")
			Eventually(func() bool {
				podStatuses, err := checker.GetNamespacePodStatuses("vmks")
				if err != nil {
					return false
				}
				// 统计各组件的 Pod 数量
				componentCounts := map[string]int{
					"vmstorage": 0,
					"vminsert":  0,
					"vmselect":  0,
					"vmagent":   0,
				}
				for podName := range podStatuses {
					for component := range componentCounts {
						if strings.Contains(podName, component) {
							componentCounts[component]++
						}
					}
				}

				// 验证 vmstorage 副本数
				expectedStoragecount := utils.GetIntValue(testParams["vmStorageReplicaCount"], 1)
				if componentCounts["vmstorage"] != expectedStoragecount {
					GinkgoWriter.Printf("当前 vmstorage Pod 数: %d, 期望: %d\n", componentCounts["vmstorage"], expectedStoragecount)
					return false
				}

				// 验证 vminsert 副本数
				expectedInsertRecount := utils.GetIntValue(testParams["vmInsertReplicaCount"], 1)
				if componentCounts["vminsert"] != expectedInsertRecount {
					GinkgoWriter.Printf("当前 vminsert Pod 数: %d, 期望: %d\n", componentCounts["vminsert"], expectedInsertRecount)
					return false
				}

				// 验证 vmselect 副本数
				expectedSelectcount := utils.GetIntValue(testParams["vmSelectReplicaCount"], 1) * utils.GetIntValue(testParams["vmAgentShareCount"], 1)
				if componentCounts["vmselect"] != expectedSelectcount {
					GinkgoWriter.Printf("当前 vmselect Pod 数: %d, 期望: %d\n", componentCounts["vmselect"], expectedSelectcount)
					return false
				}

				// 验证 vmagent 副本数
				expectedAgentcount := utils.GetIntValue(testParams["vmAgentReplicaCount"], 1)
				if componentCounts["vmagent"] != expectedAgentcount {
					GinkgoWriter.Printf("当前 vmagent Pod 数: %d, 期望: %d\n", componentCounts["vmagent"], expectedAgentcount)
					return false
				}

				return true
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "组件的副本数应该符合自定义参数配置")
		})

		// 用例名称：使用默认参数安装 victoriametrics-controller 组件
		// 用例步骤：1) 为 victoriametrics-controller addon 不提供全部自定义参数（仅保留少量必要参数让环境可测）；2) 通过引导节点创建集群并等待 Healthy；3) 校验 vmks 命名空间 Pod 就绪；4) 校验关键组件副本数/配置按默认逻辑落地。
		// 预期结果：默认参数安装成功，vmks 命名空间 Pod 运行正常，且关键配置符合默认/调整后的预期。
		It("使用默认参数安装 victoriametrics-controller 组件", Label("victoriametrics-controller-default", "large-scale-cluster", "skip-temporarily"), func() {
			// 说明：默认参数适配超大规模场景，测试需要机器多，故测试中各组件分片副本数设置为 1，超大参数适度调整，仅验证其余参数是否生效。
			// 说明：若环境允许，可删除限制，完全使用默认参数进行安装并验证。
			clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
				Name:    "victoriametrics-controller",
				Version: "latest",
				Param: map[string]string{
					"vmAgentStorageSize":           "10Gi",
					"vmAgentReplicaCount":          "1",
					"vmAgentShareCount":            "1",
					"vmAgentScrapeInterval":        "20s",
					"vmSelectCpuCount":             "0.5",
					"vmSelectMemorySize":           "2Gi",
					"vmSelectReplicaCount":         "1",
					"vmStorageCpuCount":            "0.5",
					"vmStorageMemorySize":          "2Gi",
					"vmStorageStorageSize":         "10Gi",
					"vmStorageReplicaCount":        "1",
					"vmInsertReplicaCount":         "1",
					"vmInsertCpuCount":             "0.5",
					"vmInsertMemorySize":           "2Gi",
					"vmAlertReplicaCount":          "1",
					"vmAlertCpuCount":              "0.5",
					"vmAlertMemorySize":            "1Gi",
					"vmAlertManagerReplicaCount":   "1",
					"vmAlertManagerCpuCount":       "0.5",
					"vmAlertManagerMemorySize":     "1Gi",
					"vmClusterRetentionPeriod":     "15d",
					"vmClusterReplicationFactor":   "1",
					"kubeStateMetricsReplicaCount": "1",
					"kubeStateMetricsCpuCount":     "0.5",
					"kubeStateMetricsMemorySize":   "2Gi",
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

			By("验证 victoriametrics-controller 组件已安装")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查 victoriametrics-controller 命名空间下的 Pod 状态
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("vmks", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 victoriametrics-controller Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "vmks 命名空间下的 Pod 应该全部就绪")
			By("验证 victoriametrics-controller 组件的默认参数生效")
			Eventually(func() bool {
				// 验证 vmAgentMemorySize 参数默认值为 8Gi
				agentMemorySize, err := checker.GetPodResourceRequest("vmks", "app.kubernetes.io/name=vmagent", "memory")
				if err != nil {
					return false
				}
				expectedAgentMemory := "8Gi"
				if agentMemorySize != expectedAgentMemory {
					GinkgoWriter.Printf("当前 vmagent 内存请求: %s, 期望: %s\n", agentMemorySize, expectedAgentMemory)
					return false
				}

				// 验证 vmAgentCpuCount 参数默认值为 4
				agentCpuCount, err := checker.GetPodResourceRequest("vmks", "app.kubernetes.io/name=vmagent", "cpu")
				if err != nil {
					return false
				}
				expectedAgentCpu := "4"
				if agentCpuCount != expectedAgentCpu {
					GinkgoWriter.Printf("当前 vmagent CPU 请求: %s, 期望: %s\n", agentCpuCount, expectedAgentCpu)
					return false
				}

				return true
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "victoriametrics-controller 组件的默认参数应该生效")
		})

	})
})
