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

var _ = Describe("默认 Chart 仓库 Addon 安装", Label("addon-installation", "post-init", "online"), func() {
	var (
		sshExecutor    *executor.SSHExecutor
		clusterManager *utils.ClusterManager
		clusterName    string
		clusterConfig  *config.BKEClusterConfig
		configPath     string
		nodeConfigPath string
	)

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())
		Expect(guideConfig.Password).NotTo(BeEmpty())

		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
		Expect(err).NotTo(HaveOccurred())
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())
		dynamicClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())
		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)

		clusterName = fmt.Sprintf("test-addon-%d", time.Now().Unix())
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点")
		masterNode := nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}
		clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})
	})

	AfterEach(func() {
		if clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
			if err := clusterManager.DeleteClusterWithKubeconfig(clusterName, ""); err != nil {
				GinkgoWriter.Printf("Failed to delete cluster: %v\n", err)
			}
			Eventually(func() bool {
				return !clusterManager.ClusterExistsWithKubeconfig(clusterName, "")
			}, uninstallTimeout, 10*time.Second).Should(BeTrue())
		}
		if configPath != "" {
			_ = clusterManager.CleanupConfig(configPath)
		}
		if nodeConfigPath != "" {
			_ = clusterManager.CleanupConfig(nodeConfigPath)
		}
	})

	// Author: luolong817
	It("应该支持从https格式仓库安装chart", Label("required_nodes:1", "install-chart-from-https-repo"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
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

		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons, config.AddonConfig{
			Name:        "logging-package",
			Version:     "1.0.3",
			ReleaseName: "logging",
			Type:        "chart",
			Namespace:   "logging",
		})

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

		By("验证日志组件")
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

		// 验证 logging-website 组件已经安装
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

		// 验证 loki 组件已经安装
		for _, component := range []string{"loki"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app", "name"} {
					nodeName, _ := checker.GetPodNodeName("loki", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}

		// 验证 logging-operator 组件已经安装
		for _, component := range []string{"logging-operator"} {
			Eventually(func() bool {
				for _, labelKey := range []string{"app"} {
					nodeName, _ := checker.GetPodNodeName("openfuyao-system", fmt.Sprintf("%s=%s", labelKey, component))
					if nodeName != "" {
						return true
					}
				}
				return false
			}, 2*time.Minute, 10*time.Second).Should(BeTrue(), fmt.Sprintf("应该能找到 %s 组件", component))
		}
	})
})
