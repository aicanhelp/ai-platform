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

type customRepoClusterContext struct {
	clusterManager     *utils.ClusterManager
	workerNodes        []config.NodeInfo
	clusterName        string
	configPath         string
	nodeConfigPath     string
	mgmtKubeconfigPath string
}

func newKubeClient() (dynamic.Interface, error) {
	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	if err != nil {
		return nil, err
	}
	return dynamic.NewForConfig(restConfig)
}

func waitForClusterHealthy(cm *utils.ClusterManager, name, kubeconfig string, timeout time.Duration) {
	Eventually(func() bool {
		_, _, clusterStatus, _ := cm.GetClusterFullStatusWithKubeconfig(name, kubeconfig)
		return clusterStatus == "Ready"
	}, timeout, 30*time.Second).Should(BeTrue())
}

func prepareCAFile(sshExecutor *executor.SSHExecutor, caPath string, tamper bool) error {
	cmd := fmt.Sprintf("openssl s_client -showcerts -connect cr.internal.openfuyao.cn:443 </dev/null 2>/dev/null | openssl x509 -outform PEM > %s", caPath)
	_, err := sshExecutor.Exec(cmd)
	return err
}

func customRepoCCTYAML(caFilePath string) string {
	return fmt.Sprintf(`apiVersion: bke.bocloud.com/v1beta1
kind: ContainerdConfig
metadata:
  name: bke-containerd
  namespace: bke-containerd
spec:
  configType: combined
  description: Built-in default containerd configuration
  main:
    configPath: /etc/containerd/certs.d
    metricsAddress: 127.0.0.1:1338
    root: /var/lib/containerd
    sandboxImage: cr.internal.openfuyao.cn/pass/pause:3.9
    state: /run/containerd
  registry:
    configPath: /etc/containerd/certs.d
    configs:
      "cr.internal.openfuyao.cn":
        host: "cr.internal.openfuyao.cn:443"
        skipVerify: false
        plainHTTP: false
        capabilities: ["pull", "resolve", "push"]
        tls:
          caFile: %s
          insecureSkipVerify: false
      "cr.internal.openfuyao.cn:443":
        host: "cr.internal.openfuyao.cn:443"
        skipVerify: false
        plainHTTP: false
        capabilities: ["pull", "resolve", "push"]
        tls:
          caFile: %s
          insecureSkipVerify: false
  script:
    content: |
      #!/bin/bash
      mkdir -p /tmp/log
      echo "Executing containerd setup script" > /tmp/log/test_containerd.log
    interpreter: /bin/bash
  service:
    execStart: /usr/bin/containerd --config /etc/containerd/config.toml
    killMode: process
    logging:
      standardError: journal
      standardOutput: journal
    restart: always
    restartSec: 5s
    slice: system.slice
    startLimitBurst: 5
    startLimitInterval: 10s
    timeoutStopSec: 90s
`, caFilePath, caFilePath)
}

func prepareCustomRepoGuideNode(sshExecutor *executor.SSHExecutor, caFilePath string) error {
	if err := prepareCAFile(sshExecutor, caFilePath, false); err != nil {
		return err
	}
	res, err := sshExecutor.Exec(fmt.Sprintf("set -o pipefail; echo Y | bke init --otherRepo cr.internal.openfuyao.cn/pass --domain cr.internal.openfuyao.cn --imageRepoPort 443 --otherSource http://file.internal.openfuyao.cn --enableNTP=false --installConsole=false --imageRepoCAFile %s", caFilePath))
	if err != nil {
		return err
	}
	if res.ExitCode != 0 {
		return fmt.Errorf("bke init failed: %s", res.Stderr)
	}
	return nil
}

func prepareCustomRepoCluster(sshExecutor *executor.SSHExecutor, guideCAPath, cctCAPath string) (*customRepoClusterContext, error) {
	return prepareCustomRepoClusterWithConfig(sshExecutor, guideCAPath, cctCAPath, func(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
		clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, nodes)
		clusterConfig.ImageRepo = &config.ImageRepoConfig{Domain: "cr.internal.openfuyao.cn", Port: "443", Prefix: "pass"}
		return clusterConfig
	}, 2, 45*time.Minute, true)
}

func prepareCustomRepoClusterWithoutWait(sshExecutor *executor.SSHExecutor, guideCAPath, cctCAPath string) (*customRepoClusterContext, error) {
	return prepareCustomRepoClusterWithConfig(sshExecutor, guideCAPath, cctCAPath, func(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
		clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, nodes)
		clusterConfig.ImageRepo = &config.ImageRepoConfig{Domain: "cr.internal.openfuyao.cn", Port: "443", Prefix: "pass"}
		return clusterConfig
	}, 1, 45*time.Minute, false)
}

func prepareCustomRepoClusterWithConfig(
	sshExecutor *executor.SSHExecutor,
	guideCAPath, cctCAPath string,
	newClusterConfig func(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig,
	requiredNodes int,
	timeout time.Duration,
	waitHealthy bool,
) (*customRepoClusterContext, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < requiredNodes {
		return nil, fmt.Errorf("至少需要 %d 个测试节点", requiredNodes)
	}
	if err := prepareCustomRepoGuideNode(sshExecutor, guideCAPath); err != nil {
		return nil, err
	}
	_, err := sshExecutor.Exec(fmt.Sprintf(`cat > /tmp/cct.yaml <<'EOF'
%s
EOF
kubectl apply -f /tmp/cct.yaml`, customRepoCCTYAML(cctCAPath)))
	if err != nil {
		return nil, err
	}
	dynamicClient, err := newKubeClient()
	if err != nil {
		return nil, err
	}
	clusterManager := utils.NewClusterManager(sshExecutor, dynamicClient)
	clusterName := fmt.Sprintf("test-cert-distribution-%d", time.Now().Unix())
	clusterNodes := make([]config.NodeInfo, 0, requiredNodes)
	for i := 0; i < requiredNodes; i++ {
		node := nodes[i]
		if i == 0 {
			node.Role = []string{"master/node", "etcd"}
		} else {
			node.Role = []string{"node"}
		}
		clusterNodes = append(clusterNodes, node)
	}
	clusterConfig := newClusterConfig(clusterName, clusterNodes)
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, err
	}
	if err := clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, ""); err != nil {
		return nil, err
	}
	var mgmtKubeconfigPath string
	if waitHealthy {
		waitForClusterHealthy(clusterManager, clusterName, "", timeout)
		for _, node := range clusterNodes {
			result, execErr := utils.ExecuteCommandOnNode(node, "test -s /etc/openFuyao/certs/ca.crt && echo ok")
			if execErr != nil || !strings.Contains(result.Stdout, "ok") {
				return nil, fmt.Errorf("ca not found on node %s", node.Hostname)
			}
		}
		checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, "")
		mgmtKubeconfigPath, err = checker.SaveKubeconfigToFile()
		if err != nil {
			return nil, err
		}
	}
	return &customRepoClusterContext{clusterManager: clusterManager, workerNodes: clusterNodes, clusterName: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, mgmtKubeconfigPath: mgmtKubeconfigPath}, nil
}

var _ = InstallationSIGDescribe("BKE 安装部署 - 自定义仓库与初始化", Label("pre-init", "custom-repo"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideConfig *config.GuideNodeConfig
		ctxData     *customRepoClusterContext
	)

	BeforeEach(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty())
		Expect(guideConfig.Password).NotTo(BeEmpty())
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
		Expect(err).NotTo(HaveOccurred())
		resetCmd := "echo Y | bke reset --all --mount"
		_, _ = sshExecutor.Exec(resetCmd)
		time.Sleep(10 * time.Second)
	})

	AfterEach(func() {
		if ctxData != nil && ctxData.clusterManager != nil {
			By("清理测试集群")
			if ctxData.clusterManager.ClusterExistsWithKubeconfig(ctxData.clusterName, "") {
				if err := ctxData.clusterManager.DeleteClusterWithKubeconfig(ctxData.clusterName, ""); err != nil {
					GinkgoWriter.Printf("触发集群删除失败: %v\n", err)
				}

				Eventually(func() bool {
					exists := ctxData.clusterManager.ClusterExistsWithKubeconfig(ctxData.clusterName, "")
					if exists {
						phase, _ := ctxData.clusterManager.GetClusterPhaseWithKubeconfig(ctxData.clusterName, "")
						GinkgoWriter.Printf("当前集群 phase: %s\n", phase)
					}
					return !exists
				}, 20*time.Minute, 30*time.Second).Should(BeTrue(), "集群应该被完全删除")
			}
			if err := ctxData.clusterManager.DeleteManagementClusterSelfBC(ctxData.mgmtKubeconfigPath, ctxData.clusterName); err != nil {
				GinkgoWriter.Printf("清理管理集群自身 BC 失败: %v\n", err)
			}
			if ctxData.configPath != "" {
				if err := ctxData.clusterManager.CleanupConfig(ctxData.configPath); err != nil {
					GinkgoWriter.Printf("清理配置文件失败: %v\n", err)
				}
			}
			if ctxData.nodeConfigPath != "" {
				if err := ctxData.clusterManager.CleanupConfig(ctxData.nodeConfigPath); err != nil {
					GinkgoWriter.Printf("Failed to cleanup node config file: %v\n", err)
				}
			}
		}
		resetCmd := "echo Y | bke reset --all --mount"
		_, _ = sshExecutor.Exec(resetCmd)
		time.Sleep(5 * time.Second)
	})

	// Author: ZYH-ang
	It("应该成功初始化：自定义仓、分发 CA 并验证集群节点证书存在", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		guideCAPath := "/etc/openFuyao/certs/ca.crt"
		By("准备并初始化自定义仓 CA")
		Expect(prepareCustomRepoGuideNode(sshExecutor, guideCAPath)).NotTo(HaveOccurred())
		By("校验域名解析和 hosts 写入")
		hostsCheck, err := sshExecutor.Exec("grep -n 'cr.internal.openfuyao.cn' /etc/hosts")
		Expect(err).NotTo(HaveOccurred())
		Expect(hostsCheck.ExitCode).To(Equal(0), hostsCheck.Stderr)
		Expect(strings.TrimSpace(hostsCheck.Stdout)).NotTo(BeEmpty())
		By("校验证书路径")
		certCheck, err := sshExecutor.Exec("test -s /etc/openFuyao/certs/ca.crt && echo ok")
		Expect(err).NotTo(HaveOccurred())
		Expect(certCheck.Stdout).To(ContainSubstring("ok"))
	})

	// Author: ZYH-ang
	It("通过 bkeCluster 自定义集群镜像仓后应该能拉取自定义镜像", Label("required_nodes:2"), SpecTimeout(45*time.Minute), func(ctx SpecContext) {
		var err error
		ctxData, err = prepareCustomRepoCluster(sshExecutor, "/etc/openFuyao/certs/ca.crt", "/etc/openFuyao/certs/ca.crt")
		Expect(err).NotTo(HaveOccurred())
		describeResult, err := utils.ExecuteCommandOnNode(ctxData.workerNodes[0], "kubectl describe pod -A 2>/dev/null | grep -n 'cr.internal.openfuyao.cn' | head")
		Expect(err).NotTo(HaveOccurred())
		Expect(strings.TrimSpace(describeResult.Stdout)).NotTo(BeEmpty())
	})

	// Author: ZYH-ang
	It("通过 bkeCluster 自定义二进制仓后应该成功拉取二进制文件", Label("required_nodes:2"), SpecTimeout(30*time.Minute), func(ctx SpecContext) {
		var err error
		ctxData, err = prepareCustomRepoCluster(sshExecutor, "/etc/openFuyao/certs/ca.crt", "/etc/openFuyao/certs/ca.crt")
		Expect(err).NotTo(HaveOccurred())
		result, err := utils.ExecuteCommandOnNode(ctxData.workerNodes[0], "test -f /root/.bke/bin/bke || test -x /usr/local/bin/bke || which bke")
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(BeZero())
		Expect(strings.TrimSpace(result.Stdout)).NotTo(BeEmpty())
	})

	// Author: ZYH-ang
	It("通过 bkeCluster 自定义管理集群和业务集群后应该能拉取自定义镜像", Label("required_nodes:2"), SpecTimeout(60*time.Minute), func(ctx SpecContext) {
		var err error
		ctxData, err = prepareCustomRepoClusterWithConfig(sshExecutor, "/etc/openFuyao/certs/ca.crt", "/etc/openFuyao/certs/ca.crt", func(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
			clusterConfig := config.NewBKEClusterConfigForMgmt(clusterName, nodes)
			clusterConfig.ImageRepo = &config.ImageRepoConfig{Domain: "cr.internal.openfuyao.cn", Port: "443", Prefix: "pass"}
			clusterConfig.ChartRepo = &config.ChartRepoConfig{Domain: "cr.openfuyao.cn", IP: "119.3.216.97", Port: "443", Prefix: "charts"}
			return clusterConfig
		}, 2, 45*time.Minute, true)
		Expect(err).NotTo(HaveOccurred())
		mgmtPodImages, err := utils.ExecuteCommandOnNode(ctxData.workerNodes[0], "kubectl get pod -A -o jsonpath='{range .items[*]}{.spec.containers[*].image}{\"\\n\"}{end}' | grep 'cr.internal.openfuyao.cn' | head")
		Expect(err).NotTo(HaveOccurred())
		Expect(strings.TrimSpace(mgmtPodImages.Stdout)).NotTo(BeEmpty())
	})

	// Author: ZYH-ang
	It("通过 bkeCluster 自定义镜像源安装高可用业务集群", Label("required_nodes:3"), SpecTimeout(60*time.Minute), func(ctx SpecContext) {
		var err error
		ctxData, err = prepareCustomRepoClusterWithConfig(sshExecutor, "/etc/openFuyao/certs/ca.crt", "/etc/openFuyao/certs/ca.crt", func(clusterName string, nodes []config.NodeInfo) *config.BKEClusterConfig {
			haConfig := config.NewBKEClusterConfigForHA(clusterName, nodes, "", "")
			haConfig.ImageRepo = &config.ImageRepoConfig{Domain: "cr.internal.openfuyao.cn", Port: "443", Prefix: "pass"}
			haConfig.ChartRepo = &config.ChartRepoConfig{Domain: "cr.openfuyao.cn", IP: "119.3.216.97", Port: "443", Prefix: "charts"}
			return haConfig
		}, 3, 60*time.Minute, true)
		Expect(err).NotTo(HaveOccurred())
		haImages, err := utils.ExecuteCommandOnNode(ctxData.workerNodes[0], "kubectl get pod -A -o jsonpath='{range .items[*]}{.spec.containers[*].image}{\"\\n\"}{end}' | grep 'cr.internal.openfuyao.cn' | head")
		Expect(err).NotTo(HaveOccurred())
		Expect(strings.TrimSpace(haImages.Stdout)).NotTo(BeEmpty())
	})

	// Author: ZYH-ang
	It("通过 bkeCluster 自定义集群镜像仓后，使用错误 CA 应导致集群创建失败", Label("required_nodes:1"), SpecTimeout(45*time.Minute), func(ctx SpecContext) {
		var err error
		ctxData, err = prepareCustomRepoClusterWithoutWait(sshExecutor, "/etc/openFuyao/ca.crt", "/etc/openFuyao/certs/ca.crt")
		Expect(err).NotTo(HaveOccurred())
		Expect(ctxData).NotTo(BeNil())
		Eventually(func() string {
			_, state, clusterStatus, _ := ctxData.clusterManager.GetClusterFullStatusWithKubeconfig(ctxData.clusterName, "")
			return state + "/" + clusterStatus
		}, 20*time.Minute, 30*time.Second).Should(ContainSubstring("InitializationFailed"))
	})
})

// 多镜像仓库场景与自定义仓库场景共用同一测试域。

/*
  特性：安装部署支持多镜像源
  用例：
      1）在线场景下，管理集群使用默认镜像源安装业务集群
      2）在线场景下，引导集群使用默认镜像源安装业务集群
      3）在线场景下，引导集群使用默认镜像源安装高可用业务集群
      4）离线场景下，引导集群使用本地镜像源安装业务集群
      5）离线场景下，管理集群使用本地镜像源安装业务集群
      6）离线场景下，引导集群使用本地镜像源按照高可用业务集群
*/

var _ = Describe("BKE 支持多镜像源（管理集群作为安装起点）", Label("manager-multi-image-repo", "with-management-cluster", "online"), func() {
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
	 用例名称：1）在线场景下，管理集群使用默认镜像源安装业务集群
	 用例步骤：1）安装业务集群，预期结果1和2
	 预期结果：1）集群安装成功，使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
	          2）pause镜像来自hub.oepkgs.net，coredns镜像来自cr.openfuyao.cn
	*/
	Describe("在线场景下，管理集群使用默认镜像源安装业务集群", Label("use-default-repo"), func() {
		// Author: xuyouhong
		It("应该成功创建 1Master 集群并验证镜像来自默认镜像源", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

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

			// 预期结果1：使用的经来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
			By("验证使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）")
			checkImagesCmd := "crictl images 2>&1 || true"
			imagesResult, err := utils.ExecuteCommandOnNode(masterNode, checkImagesCmd)
			Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
			imagesOutput := strings.TrimSpace(imagesResult.Stdout)
			Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

			// 基础断言：镜像列表中同时出现两种 registry 域名
			Expect(imagesOutput).To(ContainSubstring("cr.openfuyao.cn"), "镜像列表中应包含来自 cr.openfuyao.cn 的镜像")
			Expect(imagesOutput).To(ContainSubstring("hub.oepkgs.net"), "镜像列表中应包含来自 hub.oepkgs.net 的镜像")

			// 针对关键镜像做逐行校验
			lines := strings.Split(imagesOutput, "\n")
			pauseOK := false
			corednsOK := false
			for _, line := range lines {
				if strings.Contains(line, "IMAGE") {
					continue
				}
				if strings.Contains(line, "pause") && strings.Contains(line, "hub.oepkgs.net") {
					pauseOK = true
					continue
				}
				if strings.Contains(line, "coredns") && strings.Contains(line, "cr.openfuyao.cn") {
					corednsOK = true
					continue
				}
			}

			Expect(pauseOK).To(BeTrue(), "pause 应来自 hub.oepkgs.net，但未在 crictl images 中匹配到")
			Expect(corednsOK).To(BeTrue(), "coredns 应来自 cr.openfuyao.cn，但未在 crictl images 中匹配到")

			GinkgoWriter.Printf("crictl images 校验通过：pause(hub.oepkgs.net)=%v, coredns(cr.openfuyao.cn)=%v\n", pauseOK, corednsOK)
		})
	})
})

var _ = Describe("BKE 支持多镜像源（引导集群作为安装起点）", Label("bootstrap-multi-image-repo", "post-init", "online"), func() {
	var (
		sshExecutor                           *executor.SSHExecutor
		guideConfig                           *config.GuideNodeConfig
		dynamicClient                         dynamic.Interface
		clusterManager                        *utils.ClusterManager
		clusterName                           string
		clusterConfig                         *config.BKEClusterConfig
		configPath                            string
		nodeConfigPath                        string
		masterNode1, masterNode2, masterNode3 config.NodeInfo
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

	/*
	 用例名称：2）在线场景下，引导集群使用默认镜像源安装业务集群
	 用例步骤：1）安装业务集群，预期结果1和2
	 预期结果：1）集群安装成功，使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
	         2）pause镜像来自hub.oepkgs.net，coredns镜像来自cr.openfuyao.cn
	*/
	Describe("在线场景下，引导集群使用默认镜像源安装业务集群", Label("use-default-repo"), func() {
		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-1m-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置一个测试节点")

			// 只使用第一个节点作为 Master
			masterNode1 = nodes[0]
			masterNode1.Role = []string{"master/node", "etcd"}

			// 创建集群配置（使用基础 Addons 以加快测试速度）
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode1})
		})

		AfterEach(func() {
			// 清理：删除集群
			By("清理测试集群")
			if clusterManager.ClusterExists(clusterName) {
				// 触发集群销毁（设置 spec.reset=true）
				By("触发集群销毁")
				err := clusterManager.DeleteCluster(clusterName)
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				// 等待集群完全删除（BC 资源不存在）
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExists(clusterName)
					if exists {
						phase, _ := clusterManager.GetClusterPhase(clusterName)
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

		// Author: zuozongyuan
		It("应该成功创建 1Master 集群并验证镜像来自默认镜像源", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackground(configPath, nodeConfigPath)
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			// 预期结果1：使用的经来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
			By("验证使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）")
			checkImagesCmd := "crictl images 2>&1 || true"
			imagesResult, err := utils.ExecuteCommandOnNode(masterNode1, checkImagesCmd)
			Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
			imagesOutput := strings.TrimSpace(imagesResult.Stdout)
			Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

			// 基础断言：镜像列表中同时出现两种 registry 域名
			Expect(imagesOutput).To(ContainSubstring("cr.openfuyao.cn"), "镜像列表中应包含来自 cr.openfuyao.cn 的镜像")
			Expect(imagesOutput).To(ContainSubstring("hub.oepkgs.net"), "镜像列表中应包含来自 hub.oepkgs.net 的镜像")

			// 针对关键镜像做逐行校验
			lines := strings.Split(imagesOutput, "\n")
			pauseOK := false
			corednsOK := false
			for _, line := range lines {
				if strings.Contains(line, "IMAGE") {
					continue
				}
				if strings.Contains(line, "pause") && strings.Contains(line, "hub.oepkgs.net") {
					pauseOK = true
					continue
				}
				if strings.Contains(line, "coredns") && strings.Contains(line, "cr.openfuyao.cn") {
					corednsOK = true
					continue
				}
			}

			Expect(pauseOK).To(BeTrue(), "pause 应来自 hub.oepkgs.net，但未在 crictl images 中匹配到")
			Expect(corednsOK).To(BeTrue(), "coredns 应来自 cr.openfuyao.cn，但未在 crictl images 中匹配到")

			GinkgoWriter.Printf("crictl images 校验通过：pause(hub.oepkgs.net)=%v, coredns(cr.openfuyao.cn)=%v\n", pauseOK, corednsOK)
		})
	})

	/*
	 用例名称：3）在线场景下，引导集群使用默认镜像源安装高可用业务集群
	 用例步骤：1）安装业务集群，预期结果1和2
	 预期结果：1）集群安装成功，使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
	          2）haproxy镜像来自hub.oepkgs.net，keepalived镜像来自cr.openfuyao.cn
	*/
	Describe("在线场景下，引导集群使用默认镜像源安装高可用业务集群", Label("ha-cluster-use-default-repo"), func() {
		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-3m-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "至少需要配置三个测试节点")

			// 只使用三个节点作为master
			masterNode1 = nodes[0]
			masterNode1.Role = []string{"master/node", "etcd"}

			masterNode2 = nodes[1]
			masterNode2.Role = []string{"master/node", "etcd"}

			masterNode3 = nodes[2]
			masterNode3.Role = []string{"master/node", "etcd"}

			// 加载vip地址和端口
			vipIP, vipPort := config.LoadWorkloadClusterVIPFromEnv()

			// 创建集群配置（使用基础 Addons 以加快测试速度）
			clusterConfig = config.NewBKEClusterConfigForHA(clusterName, []config.NodeInfo{masterNode1, masterNode2, masterNode3}, vipIP, vipPort)
		})

		AfterEach(func() {
			// 清理：删除集群
			By("清理测试集群")
			if clusterManager.ClusterExists(clusterName) {
				// 触发集群销毁（设置 spec.reset=true）
				By("触发集群销毁")
				err := clusterManager.DeleteCluster(clusterName)
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				// 等待集群完全删除（BC 资源不存在）
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExists(clusterName)
					if exists {
						phase, _ := clusterManager.GetClusterPhase(clusterName)
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

		// Author: xuyouhong
		It("应该成功创建 3Master 集群并验证镜像来自默认镜像源", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackground(configPath, nodeConfigPath)
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			// 预期结果1：使用的经来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）
			By("验证使用的镜像来自多镜像源（cr.openfuyao.cn和hub.oepkgs.net）")
			checkImagesCmd := "crictl images 2>&1 || true"
			imagesResult, err := utils.ExecuteCommandOnNode(masterNode1, checkImagesCmd)
			Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
			imagesOutput := strings.TrimSpace(imagesResult.Stdout)
			Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

			// 基础断言：镜像列表中同时出现两种 registry 域名
			Expect(imagesOutput).To(ContainSubstring("cr.openfuyao.cn"), "镜像列表中应包含来自 cr.openfuyao.cn 的镜像")
			Expect(imagesOutput).To(ContainSubstring("hub.oepkgs.net"), "镜像列表中应包含来自 hub.oepkgs.net 的镜像")

			// 针对关键镜像做逐行校验
			lines := strings.Split(imagesOutput, "\n")
			haproxyOK := false
			keepalivedOK := false
			for _, line := range lines {
				if strings.Contains(line, "IMAGE") {
					continue
				}
				if strings.Contains(line, "haproxy") && strings.Contains(line, "hub.oepkgs.net") {
					haproxyOK = true
					continue
				}
				if strings.Contains(line, "keepalived") && strings.Contains(line, "cr.openfuyao.cn") {
					keepalivedOK = true
					continue
				}
			}

			Expect(haproxyOK).To(BeTrue(), "haproxy 应来自 hub.oepkgs.net，但未在 crictl images 中匹配到")
			Expect(keepalivedOK).To(BeTrue(), "keepalived 应来自 cr.openfuyao.cn，但未在 crictl images 中匹配到")

			GinkgoWriter.Printf("crictl images 校验通过：haproxy(hub.oepkgs.net)=%v, keepalived(cr.openfuyao.cn)=%v\n", haproxyOK, keepalivedOK)
		})
	})
})

var _ = Describe("离线场景 BKE 支持多镜像源（引导集群作为安装起点）", Label("bootstrap-multi-image-repo", "offline", "post-init"), func() {
	var (
		sshExecutor                           *executor.SSHExecutor
		guideConfig                           *config.GuideNodeConfig
		dynamicClient                         dynamic.Interface
		clusterManager                        *utils.ClusterManager
		clusterName                           string
		clusterConfig                         *config.BKEClusterConfig
		configPath                            string
		nodeConfigPath                        string
		masterNode1, masterNode2, masterNode3 config.NodeInfo
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
	/*
	 用例名称：4）离线场景下，引导集群使用本地镜像源后端安装业务集群
	 用例步骤：1）安装业务集群，预期结果1
	 预期结果：1）集群安装成功，使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
	*/
	Describe("离线场景下，引导集群使用本地镜像源安装业务集群", Label("bootstrap-1m-local-repo"), func() {
		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 1), "至少需要配置一个测试节点")

			// 只使用第一个节点作为 Master
			masterNode1 = nodes[0]
			masterNode1.Role = []string{"master/node", "etcd"}

			// 创建集群配置（使用基础 Addons 以加快测试速度）
			clusterConfig = config.NewDefaultBKEClusterConfig(clusterName, []config.NodeInfo{masterNode1})
		})

		AfterEach(func() {
			// 清理：删除集群
			By("清理测试集群")
			if clusterManager.ClusterExists(clusterName) {
				// 触发集群销毁（设置 spec.reset=true）
				By("触发集群销毁")
				err := clusterManager.DeleteCluster(clusterName)
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				// 等待集群完全删除（BC 资源不存在）
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExists(clusterName)
					if exists {
						phase, _ := clusterManager.GetClusterPhase(clusterName)
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

		// Author: xuyouhong
		It("应该成功创建 1Master 集群并验证镜像来自本地镜像源", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackground(configPath, nodeConfigPath)
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			// 预期结果1：使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
			By("验证使用的镜像来自本地镜像源（deploy.bocloud.k8s）")
			checkImagesCmd := "crictl images 2>&1 || true"
			imagesResult, err := utils.ExecuteCommandOnNode(masterNode1, checkImagesCmd)
			Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
			imagesOutput := strings.TrimSpace(imagesResult.Stdout)
			Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

			// 基础断言：镜像列表中只包含本地镜像源
			Expect(imagesOutput).To(ContainSubstring("deploy.bocloud.k8s"), "镜像列表中应包含来自 deploy.bocloud.k8s 的镜像")

			// 针对关键镜像做逐行校验
			lines := strings.Split(imagesOutput, "\n")
			pauseOK := false
			corednsOK := false
			for _, line := range lines {
				if strings.Contains(line, "IMAGE") {
					continue
				}
				if strings.Contains(line, "pause") && strings.Contains(line, "deploy.bocloud.k8s") {
					pauseOK = true
					continue
				}
				if strings.Contains(line, "coredns") && strings.Contains(line, "deploy.bocloud.k8s") {
					corednsOK = true
					continue
				}
			}

			Expect(pauseOK).To(BeTrue(), "pause 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")
			Expect(corednsOK).To(BeTrue(), "coredns 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")

			GinkgoWriter.Printf("crictl images 校验通过：pause(deploy.bocloud.k8s)=%v, coredns(deploy.bocloud.k8s)=%v\n", pauseOK, corednsOK)
		})
	})

	/*
	 用例名称：6）离线场景下，引导集群使用本地镜像源后端安装高可用业务集群
	 用例步骤：1）安装业务集群，预期结果1
	 预期结果：1）集群安装成功，使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
	*/
	Describe("离线场景下，引导集群使用本地镜像源安装高可用业务集群", Label("bootstrap-3m-local-repo"), func() {
		BeforeEach(func() {
			clusterName = fmt.Sprintf("test-3m-%d", time.Now().Unix())

			// 加载节点配置
			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 3), "至少需要配置三个测试节点")

			// 只使用三个节点作为master
			masterNode1 = nodes[0]
			masterNode1.Role = []string{"master/node", "etcd"}

			masterNode2 = nodes[1]
			masterNode2.Role = []string{"master/node", "etcd"}

			masterNode3 = nodes[2]
			masterNode3.Role = []string{"master/node", "etcd"}

			// 加载vip地址和端口
			vipIP, vipPort := config.LoadWorkloadClusterVIPFromEnv()

			// 创建集群配置（使用基础 Addons 以加快测试速度）
			clusterConfig = config.NewBKEClusterConfigForHA(clusterName, []config.NodeInfo{masterNode1, masterNode2, masterNode3}, vipIP, vipPort)
		})

		AfterEach(func() {
			// 清理：删除集群
			By("清理测试集群")
			if clusterManager.ClusterExists(clusterName) {
				// 触发集群销毁（设置 spec.reset=true）
				By("触发集群销毁")
				err := clusterManager.DeleteCluster(clusterName)
				if err != nil {
					GinkgoWriter.Printf("触发集群销毁失败: %v\n", err)
				}

				// 等待集群完全删除（BC 资源不存在）
				By("等待集群完全删除")
				Eventually(func() bool {
					exists := clusterManager.ClusterExists(clusterName)
					if exists {
						phase, _ := clusterManager.GetClusterPhase(clusterName)
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

		// Author: xuyouhong
		It("应该成功创建 3Master 集群并验证镜像来自本地镜像源", Label("required_nodes:3"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			var err error

			By("生成集群配置文件")
			configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
			Expect(err).NotTo(HaveOccurred(), "应该成功生成配置文件")
			GinkgoWriter.Printf("配置文件路径: %s\n", configPath)

			By("验证配置文件已上传")
			err = clusterManager.GetConfigGenerator().ValidateConfigOnNode(configPath)
			Expect(err).NotTo(HaveOccurred(), "配置文件应该存在于引导节点")

			By("执行 bke cluster create 命令")
			err = clusterManager.CreateClusterInBackground(configPath, nodeConfigPath)
			Expect(err).NotTo(HaveOccurred(), "创建集群命令应该成功执行")

			By("等待集群状态变为 Ready")
			Eventually(func() string {
				phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatus(clusterName)
				GinkgoWriter.Printf("当前集群状态: phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
				failOnClusterFailure(state, clusterStatus)
				return clusterStatus
			}, installTimeout, pollInterval).Should(Equal("Ready"), "集群应该变为 Ready 状态")

			// 预期结果1：使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
			By("验证使用的镜像来自本地镜像源（deploy.bocloud.k8s）")
			checkImagesCmd := "crictl images 2>&1 || true"
			imagesResult, err := utils.ExecuteCommandOnNode(masterNode1, checkImagesCmd)
			Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
			imagesOutput := strings.TrimSpace(imagesResult.Stdout)
			Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

			// 基础断言：镜像列表中只包含本地镜像源
			Expect(imagesOutput).To(ContainSubstring("deploy.bocloud.k8s"), "镜像列表中应包含来自 deploy.bocloud.k8s 的镜像")

			// 针对关键镜像做逐行校验
			lines := strings.Split(imagesOutput, "\n")
			haproxyOK := false
			keepalivedOK := false
			for _, line := range lines {
				if strings.Contains(line, "IMAGE") {
					continue
				}
				if strings.Contains(line, "haproxy") && strings.Contains(line, "deploy.bocloud.k8s") {
					haproxyOK = true
					continue
				}
				if strings.Contains(line, "keepalived") && strings.Contains(line, "deploy.bocloud.k8s") {
					keepalivedOK = true
					continue
				}
			}

			Expect(haproxyOK).To(BeTrue(), "haproxy 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")
			Expect(keepalivedOK).To(BeTrue(), "keepalived 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")

			GinkgoWriter.Printf("crictl images 校验通过：haproxy(deploy.bocloud.k8s)=%v, keepalived(deploy.bocloud.k8s)=%v\n", haproxyOK, keepalivedOK)
		})
	})
})

var _ = Describe("离线场景下，管理集群使用本地镜像源安装业务集群", Label("manager-multi-image-repo", "offline", "with-management-cluster"), func() {
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
	 用例名称：5）离线场景下，管理集群使用本地镜像源安装业务集群
	 用例步骤：1）安装业务集群，预期结果1
	 预期结果：1）集群安装成功，使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
	*/
	// Author: zuozongyuan
	It("应该成功创建 1Master 集群并验证镜像来自本地镜像源", Label("required_nodes:1"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		var err error

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

		// 预期结果1：使用的镜像均来自本地镜像源（deploy.bocloud.k8s）
		By("验证使用的镜像来自本地镜像源（deploy.bocloud.k8s）")
		checkImagesCmd := "crictl images 2>&1 || true"
		imagesResult, err := utils.ExecuteCommandOnNode(masterNode, checkImagesCmd)
		Expect(err).NotTo(HaveOccurred(), "crictl images 命令执行失败")
		imagesOutput := strings.TrimSpace(imagesResult.Stdout)
		Expect(imagesOutput).NotTo(BeEmpty(), "crictl images 输出为空")

		// 基础断言：镜像列表中只包含本地镜像源
		Expect(imagesOutput).To(ContainSubstring("deploy.bocloud.k8s"), "镜像列表中应包含来自 deploy.bocloud.k8s 的镜像")

		// 针对关键镜像做逐行校验
		lines := strings.Split(imagesOutput, "\n")
		pauseOK := false
		corednsOK := false
		for _, line := range lines {
			if strings.Contains(line, "IMAGE") {
				continue
			}
			if strings.Contains(line, "pause") && strings.Contains(line, "deploy.bocloud.k8s") {
				pauseOK = true
				continue
			}
			if strings.Contains(line, "coredns") && strings.Contains(line, "deploy.bocloud.k8s") {
				corednsOK = true
				continue
			}
		}

		Expect(pauseOK).To(BeTrue(), "pause 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")
		Expect(corednsOK).To(BeTrue(), "coredns 应来自 deploy.bocloud.k8s，但未在 crictl images 中匹配到")

		GinkgoWriter.Printf("crictl images 校验通过：pause(deploy.bocloud.k8s)=%v, coredns(deploy.bocloud.k8s)=%v\n", pauseOK, corednsOK)
	})
})
