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

var _ = Describe("OCI Chart Addon 生命周期", Label("addon-installation", "post-init", "online"), func() {
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

	Describe("Addon 扩展组件安装", Label("addon-installation", "post-init", "online"), func() {
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

		// 用例名称：应该支持从oci格式chart仓库安装、升级、卸载 chart
		// 用例步骤：1) 在引导节点侧创建集群并保证集群 Healthy；2) 将 `logging-package` 以 OCI chart 形式安装为 addon（logging-operator 等组件）；3) 校验相关命名空间 Pod 就绪；4) 升级 chart 版本并验证 Deployment 镜像 tag；5) 卸载 chart addon 并校验 Pod 被删除。
		// 预期结果：安装/升级/卸载三阶段均生效，卸载后对应命名空间不再存在相关组件 Pod。
		// Author: shizilong1
		It("应该支持从oci格式chart仓库安装、升级、卸载 chart", Label("required_nodes:1", "install-upgrade-delete-chart-from-oci-repo"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			// 1. 创建集群

			hasPluginManagementService := false
			for _, addon := range clusterConfig.Addons {
				if addon.Name == "plugin-management-service" || addon.Name == "openfuyao-system-controller" {
					hasPluginManagementService = true
				}
			}

			// 不安装plugin-management-service直接安装logging-package会失败，所以安装plugin-management-service
			if !hasPluginManagementService {
				clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
					Name:        "plugin-management-service",
					Version:     "0.0.0-latest",
					ReleaseName: "plugin-management",
					Namespace:   "openfuyao-system",
					Type:        "chart",
					Block:       true,
				})
			}
			clusterConfig.ChartRepo = config.OpenFuyaoChartRepoConfig()

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

			// 2. 修改bc cr，安装日志组件
			By("安装日志组件")
			logAddon := config.AddonConfig{
				Name:        "logging-package",
				Version:     "1.0.2",
				ReleaseName: "logging",
				Type:        "chart",
				Namespace:   "logging",
			}
			err = clusterManager.AddNewAddons(clusterName, clusterConfig.Namespace, logAddon)
			Expect(err).NotTo(HaveOccurred(), "已经添加日志组件")

			By("验证日志组件")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

			// 检查logging-website状态 (在 logging 命名空间)
			By("验证logging-package组件")
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("logging", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 logging-package Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 10*time.Minute, 10*time.Second).Should(BeTrue(), "logging 命名空间下的 Pod 应该全部就绪")

			// 检查 openfuyao-system 命名空间下的 Pod 状态
			By("验证openfuyao-system命名空间下的组件")
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("openfuyao-system", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 openfuyao-system Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "openfuyao-system 命名空间下的 Pod 应该全部就绪")

			// 检查 loki 命名空间下的 Pod 状态
			By("验证loki组件")
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("loki", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 loki Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "loki 命名空间下的 Pod 应该全部就绪")

			// 进一步验证具体的 logging-package 组件
			for _, component := range []string{"proxy-logging"} {
				Eventually(func() bool {
					for _, labelKey := range []string{"app"} {
						nodeName, _ := checker.GetPodNodeName("logging", fmt.Sprintf("%s=%s", labelKey, component))
						if nodeName != "" {
							return true
						}
					}
					return false
				}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
			}

			// 2. 升级chart包

			By("修改日志组件版本号")
			err = clusterManager.PatchBkeClusterAddonVersion(clusterName, clusterConfig.Namespace,
				"logging-package", "1.0.3")
			Expect(err).NotTo(HaveOccurred(), "日志组件版本号修改成功")

			// 检查 openfuyao-system 命名空间下的 Pod 状态
			Eventually(func() bool {
				allReady, notReady, err := checker.CheckNamespacePodsReadyWithKubeconfig("openfuyao-system", "")
				if err != nil {
					return false
				}
				if !allReady {
					GinkgoWriter.Printf("等待 openfuyao-system Pod 就绪: %s\n", notReady)
					return false
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "openfuyao-system 命名空间下的 pod 应该全部就绪")

			By("获取日志组件版本号")
			Eventually(func() bool {
				images, _ := checker.GetDeploymentImages("openfuyao-system", "logging-operator")
				for _, image := range images {
					tag := utils.GetImageTag(image)
					return tag == "1.0.3"
				}
				return false
			}, 3*time.Minute, 10*time.Second).Should(BeTrue(), "日志组件版本号应该为1.0.3")

			// 3. 卸载chart包

			By("卸载日志组件")
			err = clusterManager.RemoveBkeClusterAddonByName(clusterName, clusterConfig.Namespace, "logging-package")
			Expect(err).NotTo(HaveOccurred(), "从bc中删除日志组成功")

			// 检查 logging 命名空间下是否有Pod
			By("验证logging命名空间下的Pod被删除")
			Eventually(func() bool {
				hasPod, _ := checker.IsNamespaceHasPodsWithKubeconfig("logging", "")
				return hasPod
			}, 3*time.Minute, 10*time.Second).Should(BeFalse(), "logging命名空间下应该没有pod")

			// 检查 openfuyao-system 命名空间下是否有 logging-operator Pod
			Eventually(func() bool {
				hasPod, _ := checker.IsNamespaceHasPodByNameWithKubeconfig("openfuyao-system", "logging-operator", "")
				return hasPod
			}, 3*time.Minute, 10*time.Second).Should(BeFalse(), "openfuyao-system命名空间下应该没有pod")
		})
	})
})
