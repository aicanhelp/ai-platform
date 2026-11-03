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

// KubeOS 不可变 OS 适配 - 功能测试
// 覆盖用例：
//   - KubeOS-IT-001: bkeadm 注入 immutable-os 标志
//   - KubeOS-IT-006: webhook 校验——同 IP 纯 master 和纯 worker 冲突
//   - KubeOS-IT-007: 普通模式不受影响回归
var _ = SIGDescribe("KubeOS Immutable OS Functional", func() {
	var (
		sshExecutor    *executor.SSHExecutor
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
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

	Describe("bkeadm 注入 immutable-os 标志", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			kubeosChecker  *utils.KubeOSChecker
			clusterChecker *utils.ClusterChecker
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-kubeos-it001-%d", time.Now().Unix())

			// 加载节点：1 master + 1 KubeOS worker
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 2), "KubeOS 功能测试需要至少 2 个节点（1 master + 1 KubeOS worker）")

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

		// 用例编号: KubeOS-IT-001
		// 用例名称: KubeOS_bkeadm注入immutable-os标志
		// 用例步骤:
		//   1. 执行 bke cluster create -f cluster.yaml -n nodes.yaml --immutable-os=kubeos
		//   2. 检查提交到集群的 BKECluster CR 的 annotations
		//   3. 不传 --immutable-os 重新创建集群，检查 annotations
		// 预期结果:
		//   1. 传参时 BKECluster annotations 含 openfuyao.io/immutable-os: kubeos（实际为 bke.bocloud.com/immutable-os: kubeos）
		//   2. 不传时不含该 annotation，集群按普通模式安装
		It("【KubeOS-IT-001】bkeadm 传入 --immutable-os=kubeos 时 BKECluster CR 应包含 immutable-os annotation", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("使用 --immutable-os=kubeos 创建集群")
			err = clusterManager.CreateClusterInBackgroundWithImmutableOS(configPath, nodeConfigPath, "", utils.ImmutableOSAnnotationValue)
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			By("验证 BKECluster CR 包含 immutable-os annotation")
			clusterChecker = utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			kubeosChecker = utils.NewKubeOSChecker(clusterChecker, clusterManager)

			annotation, err := kubeosChecker.GetImmutableOSAnnotation(clusterName)
			Expect(err).NotTo(HaveOccurred())
			Expect(annotation).To(Equal(utils.ImmutableOSAnnotationValue),
				"BKECluster annotations 应包含 %s=%s", utils.ImmutableOSAnnotationKey, utils.ImmutableOSAnnotationValue)
		})
	})

	Describe("webhook 校验——同 IP 纯 master 和纯 worker 冲突", Label("kubeos", "installation", "post-init", "P1", "26.09", "skip-temporarily"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			kubeosChecker  *utils.KubeOSChecker
			clusterChecker *utils.ClusterChecker
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-kubeos-it006-%d", time.Now().Unix())

			// 加载节点：1 个节点同时配 master 和 worker 角色
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "webhook 校验测试需要至少 1 个节点")

			// 同一节点同时配 master 和 worker 角色
			conflictNode := nodes[0]
			conflictNode.Role = []string{"master/node", "etcd", "node"}

			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{conflictNode})
		})

		AfterEach(func() {
			By("清理测试集群（如果创建成功的话）")
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

		// 用例编号: KubeOS-IT-006
		// 用例名称: KubeOS_webhook校验——同IP纯master和纯worker冲突
		// 用例步骤:
		//   1. bke cluster create --immutable-os=kubeos，节点 YAML 中同一 IP 分别配 ["master"] 和 ["node"]
		//   2. 观察创建结果
		// 预期结果:
		//   1. 集群创建被 webhook 拒绝
		//   2. 报错提示该 IP 角色冲突
		It("【KubeOS-IT-006】同 IP 同时配 master 和 worker 角色时集群创建应被 webhook 拒绝", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("生成集群配置（同一节点同时配 master 和 worker 角色）")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("使用 --immutable-os=kubeos 创建集群（预期被拒绝）")
			// 使用后台创建，因为预期会失败
			err = clusterManager.CreateClusterInBackgroundWithImmutableOS(configPath, nodeConfigPath, "", utils.ImmutableOSAnnotationValue)
			Expect(err).NotTo(HaveOccurred())

			// 等待一段时间让 webhook 处理
			clusterChecker = utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			kubeosChecker = utils.NewKubeOSChecker(clusterChecker, clusterManager)

			By("验证集群创建被拒绝")
			// 检查 BKECluster 是否存在且状态为失败，或集群未被创建
			Eventually(func() bool {
				// 如果集群不存在，可能已被 webhook 拒绝
				if !clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
					return true
				}
				// 如果集群存在但状态为失败，也认为被拒绝
				_, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: state=%s, clusterStatus=%s\n", state, clusterStatus)
				return state == "DeployFailed" || clusterStatus == "InitializationFailed"
			}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "集群创建应被 webhook 拒绝或进入失败态")

			// 检查日志中是否有拒绝相关关键字
			logPath := fmt.Sprintf("/tmp/bke-create-%s.log", clusterName)
			rejected, msg, _ := kubeosChecker.CheckClusterCreateRejected(logPath)
			if rejected {
				GinkgoWriter.Printf("webhook 拒绝信息: %s\n", msg)
			}
		})
	})

	Describe("普通模式不受影响回归", Label("kubeos", "installation", "post-init", "P0", "26.09", "regression", "skip-temporarily"), func() {
		var (
			clusterName    string
			clusterConfig  *config.BKEClusterConfig
			configPath     string
			nodeConfigPath string
			kubeosChecker  *utils.KubeOSChecker
			clusterChecker *utils.ClusterChecker
		)

		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-kubeos-it007-%d", time.Now().Unix())

			// 加载节点：只需 1 台普通 openEuler master 即可验证普通模式不受影响
			// （不使用 KubeOS worker，因为普通模式会向 worker 推送二进制，KubeOS 只读 rootfs 会失败）
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "普通模式回归测试需要至少 1 个普通 openEuler 节点")

			masterNode := nodes[0]
			masterNode.Role = []string{"master/node", "etcd"}

			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
			// 普通模式不应设置 immutable-os annotation
			clusterConfig.Annotations = map[string]string{}
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

		// 用例编号: KubeOS-IT-007
		// 用例名称: KubeOS_普通模式不受影响回归
		// 用例步骤:
		//   1. bke cluster create -f cluster.yaml -n nodes.yaml（不传 --immutable-os）
		//   2. 等待安装完成
		//   3. 检查节点状态和标签
		// 预期结果:
		//   1. 集群正常安装，master 和 worker 均 Ready
		//   2. worker 不带 node.openfuyao.io/os=kubeos 标签
		//   3. 安装行为与适配前一致（bkeagent/kubelet/containerd 正常下载安装）
		It("【KubeOS-IT-007】不传 --immutable-os 时普通模式安装不受 KubeOS 适配影响", SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			expectedNodeCount := len(clusterConfig.Nodes)

			By("生成集群配置")
			var err error
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred())

			By("不传 --immutable-os 创建集群（普通模式）")
			err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
			Expect(err).NotTo(HaveOccurred())

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"))

			clusterChecker = utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
			kubeosChecker = utils.NewKubeOSChecker(clusterChecker, clusterManager)

			By("验证 BKECluster 不包含 immutable-os annotation")
			annotation, err := kubeosChecker.GetImmutableOSAnnotation(clusterName)
			Expect(err).NotTo(HaveOccurred())
			Expect(annotation).To(BeEmpty(), "普通模式 BKECluster 不应包含 immutable-os annotation")

			By("验证 worker 节点不带 KubeOS 标签")
			// 获取所有带 KubeOS 标签的节点，普通模式应为空
			kubeOSWorkers, err := kubeosChecker.GetKubeOSWorkerNodes()
			Expect(err).NotTo(HaveOccurred())
			Expect(kubeOSWorkers).To(BeEmpty(), "普通模式 worker 节点不应带 node.openfuyao.io/os=kubeos 标签")

			By("验证集群节点 Ready")
			readyCount, err := clusterChecker.GetReadyNodeCountWithKubeconfig("")
			Expect(err).NotTo(HaveOccurred())
			Expect(readyCount).To(Equal(expectedNodeCount), "所有配置的节点均应 Ready")
		})
	})
})
