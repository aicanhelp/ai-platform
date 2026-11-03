package addons

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

var _ = Describe("Addon tests", func() {
	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager

		configPath     string
		nodeConfigPath string
	)
	clusterName := fmt.Sprintf("test-addon-order-%d", time.Now().Unix())

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())
		Expect(guideConfig.Password).NotTo(BeEmpty())

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred())

		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)
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

	// 用例: addon组件部署能力增强-addon顺序安装
	// 步骤:
	// 1. 在引导节点打开 bke cluster
	// 2. 依照顺序在 addons 中添加组件内容，并将 block 设置为 true
	// 3. 部署集群
	// 4. 查看集群节点上每个 addon 的部署创建时间和 running 时间
	// 预期结果: addons 顺序中靠前的组件 pod 先被创建，且在组件安装完毕（pod 均 running）后再进行下一个组件安装
	// 用例: addon组件部署能力增强-通过bke cluster删除组件
	// 步骤:
	// 1. 在引导节点打开 bke cluster
	// 2. 删除 addons 中对应的组件字段（例如删除 redis）
	// 3. 在集群节点查看组件情况
	// 预期结果 (删除): 集群节点组件 pod 均删除，组件已卸载
	// Author: luolong817
	It("addon组件部署能力增强-addon顺序安装", Label("required_nodes:1", "installation", "addon", "25.12", "post-init", "online"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {

		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要一个节点用于创建集群")

		masterNode := nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}
		clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode})

		// 在 addons 中按顺序添加组件，并将 block 设置为 true
		clusterConfig.ExcludeAddons = append(clusterConfig.ExcludeAddons, "bkeagent-deployer")
		clusterConfig.ExtraAddons = append(clusterConfig.ExtraAddons,
			config.AddonConfig{
				Name:    "bkeagent-deployer",
				Version: "latest",
				Block:   true,
				Param: map[string]string{
					"tagVersion": "latest",
				},
			},
			config.AddonConfig{Name: "redis", Version: "6.2.12", Block: true},
		)

		By("生成集群配置文件")
		var err error
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")

		By("执行集群创建命令")
		err = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
		Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

		By("等待集群状态变为 Ready")
		Eventually(func() string {
			phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus
		}, installTimeout, pollInterval).Should(Equal("Ready"))

		// 验证按顺序创建：bkeagent-deployer 应先于 redis 相关组件创建，并在其创建前已 Running
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")

		// 等待两个 Pod 都已被调度到节点
		Eventually(func() bool {
			n1, _ := checker.GetPodNodeName("", "app.kubernetes.io/name=redis")
			n2, _ := checker.GetPodNodeName("", "app=bkeagent-deployer")
			return n1 != "" && n2 != ""
		}, 10*time.Minute, 15*time.Second).Should(BeTrue(), "bkeagent-deployer 和 redis 的 Pod 应该被创建并运行")

		// 获取 bootstrap 上保存的业务集群 kubeconfig 文件路径（SaveKubeconfigToFile 会在 bootstrap 上写入文件）
		kubeconfigPath, err := checker.SaveKubeconfigToFile()
		Expect(err).NotTo(HaveOccurred())

		By("验证addon部署顺序")
		// helper: 通过在引导节点执行 kubectl（使用上面返回的 kubeconfigPath）获取 jsonpath 输出并解析为 time.Time
		getTime := func(namespace, selector, jsonpath string) time.Time {
			nsArg := "-A"
			if namespace != "" {
				nsArg = fmt.Sprintf("-n %s", namespace)
			}
			sel := ""
			if selector != "" {
				sel = fmt.Sprintf("-l %s", selector)
			}
			cmd := fmt.Sprintf("KUBECONFIG=%s kubectl get pods %s %s -o jsonpath=\"%s\"", kubeconfigPath, nsArg, sel, jsonpath)
			result, err := localExecutor.Exec(cmd)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.ExitCode).To(Equal(0))
			ts := strings.TrimSpace(strings.Trim(result.Stdout, "'\""))
			t, perr := time.Parse(time.RFC3339, ts)
			Expect(perr).NotTo(HaveOccurred(), "解析时间戳失败: %s", ts)
			return t
		}

		// 获取 creationTimestamp
		redisCreate := getTime("", "app.kubernetes.io/name=redis", "{.items[0].metadata.creationTimestamp}")
		bkeagentCreate := getTime("", "app=bkeagent-deployer", "{.items[0].metadata.creationTimestamp}")

		// 断言创建顺序：bkeagent-deployer 先于 redis 的组件
		Expect(bkeagentCreate.Before(redisCreate) || bkeagentCreate.Equal(redisCreate)).To(BeTrue(), "bkeagent-deployer 应先于 redis 组件创建")

		// 接下来验证通过更新 BKECluster 配置删除已部署的 redis addon
		By("通过更新 BKECluster 配置删除 redis addon")
		// 生成新的集群配置（移除 redis）并上传
		// 由于 clusterConfig 目前包含 redis
		newCfg := clusterConfig
		// 过滤掉 name == "redis" 的 addon
		filtered := []config.AddonConfig{}
		for _, a := range newCfg.ExtraAddons {
			if a.Name == "redis" {
				continue
			}
			filtered = append(filtered, a)
		}
		newCfg.ExtraAddons = filtered

		By("生成新的集群配置文件")
		newConfigPath, newNodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(newCfg)
		Expect(err).NotTo(HaveOccurred(), "应该成功生成用于更新的配置文件")
		// 覆盖 configPath/nodeConfigPath 以便 AfterEach 清理
		configPath = newConfigPath
		nodeConfigPath = newNodeConfigPath

		// 使用 kubectl apply 更新 BKECluster（在引导集群）
		GinkgoWriter.Printf("newConfigPath: %s\n", newConfigPath)
		applyCmd := fmt.Sprintf("kubectl apply -f %s", newConfigPath)
		result, err := sshExecutor.Exec(applyCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0), "应该成功应用更新的 BKECluster 配置")

		// 等待 redis Pod 被删除
		By("等待 redis Pod 被删除")
		Eventually(func() bool {
			nodeName, _ := checker.GetPodNodeName("", "app.kubernetes.io/name=redis")
			return nodeName == ""
		}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "redis Pod 应该在移除 addon 后被删除")
	})
})
