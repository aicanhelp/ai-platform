package declarative_upgrade

import (
	"fmt"
	"os"
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

const (
	businessAbnormalSourceVersion          = "v26.06-rc.2"
	businessAbnormalTargetRc3              = "v26.06-rc.3"
	businessAbnormalTargetStable           = "v26.06"
	businessInvalidTargetVersion           = "v26.06-does-not-exist"
	businessAbnormalRc3AgentVersion        = "26.6.0-rc.3"
	businessAbnormalStableAgentVersion     = "26.6.0"
	businessAbnormalInstallTimeout         = 20 * time.Minute
	businessAbnormalMonitoringReadyTimeout = 20 * time.Minute
)

var _ = Describe("集群状态正常但存在 pod 异常时触发升级", Label("online", "pre-init"), func() {
	var (
		sshExecutor     *executor.SSHExecutor
		localExecutor   *executor.LocalExecutor
		guideConfig     *config.GuideNodeConfig
		dynamicClient   dynamic.Interface
		clusterManager  *utils.ClusterManager
		duHelper        *utils.DeclarativeUpgradeHelper
		originalBKELink string
		mainBKEVersion  string
	)

	BeforeEach(func() {
		guideConfig = config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST must be set")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD must be set")

		var err error
		sshExecutor, err = executor.NewSSHExecutor(
			guideConfig.Host,
			guideConfig.Port,
			guideConfig.Username,
			guideConfig.Password,
		)
		Expect(err).NotTo(HaveOccurred(), "should connect to guide node")

		localExecutor = executor.NewLocalExecutor(10 * time.Minute)
		mainBKEVersion = os.Getenv("BKE_VERSION")
		if mainBKEVersion == "" {
			mainBKEVersion = "latest"
		}
		originalBKELink = initGuideNodeForRc2()
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred(), "should load local kubeconfig")

		dynamicClient, err = dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred(), "should create dynamic client")

		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		duHelper = utils.NewDeclarativeUpgradeHelper(sshExecutor)
	})

	Describe("3m1w业务集群 pod 异常时触发升级", Label("online", "pre-init", "abnormal-pod"), func() {
		var (
			clusterName             string
			clusterConfig           *config.BKEClusterConfig
			clusterNodes            []config.NodeInfo
			configPath              string
			nodeConfigPath          string
			bootstrapKubeconfigPath string
			workloadKubeconfigPath  string
		)

		BeforeEach(func() {
			configPath = ""
			nodeConfigPath = ""
			bootstrapKubeconfigPath = "/etc/rancher/k3s/k3s.yaml"
			workloadKubeconfigPath = ""
			clusterName = fmt.Sprintf("business-abnormal-pod-3m1w-%d", time.Now().UnixNano())
			var err error
			clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WNodes()
			Expect(err).NotTo(HaveOccurred())
			clusterConfig = utils.NewDeclarativeUpgradeAbnormalPodClusterConfig(clusterName, clusterNodes)

			By("前置条件：安装 v26.06-rc.2 的 3m1w 业务集群并验证基线")
			configPath, nodeConfigPath = runBusinessAbnormalInstallAndVerify(
				clusterManager,
				sshExecutor,
				duHelper,
				clusterConfig,
				clusterName,
				bootstrapKubeconfigPath,
			)

			By("前置条件：保存业务集群 kubeconfig，用于注入 Pod 异常")
			checker := utils.NewClusterCheckerWithParentKubeconfig(sshExecutor, clusterName, bootstrapKubeconfigPath)
			Eventually(func() bool {
				path, e := checker.SaveKubeconfigToFileWithKubeconfig(bootstrapKubeconfigPath)
				if e != nil {
					GinkgoWriter.Printf("save business cluster kubeconfig failed: %v\n", e)
					return false
				}
				workloadKubeconfigPath = path
				GinkgoWriter.Printf("business cluster kubeconfig: %s\n", workloadKubeconfigPath)
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(), "should save business cluster kubeconfig")
		})

		AfterEach(func() {
			defer resetGuideNodeBKE(sshExecutor, mainBKEVersion, originalBKELink)
			cleanupDeclarativeUpgradeCluster(
				clusterManager,
				clusterName,
				bootstrapKubeconfigPath,
				configPath,
				nodeConfigPath,
			)
			if workloadKubeconfigPath != "" {
				_, _ = sshExecutor.Exec(fmt.Sprintf("rm -f %s", shQuote(workloadKubeconfigPath)))
			}
		})

		// Author: ZYH-ang
		It("pod状态异常时触发升级-后端-高可用-业务集群-3m1w", Label("required_nodes:4", "P2", "26.06-rc.3"), SpecTimeout(installationItTimeout+upgradeTimeout), func(ctx SpecContext) {
			By("1. 在引导节点准备 rc3 版本 bkeagent")
			prepareGuideNodeBkeagent(businessAbnormalRc3AgentVersion)

			By("2. 在业务集群 monitoring 命名空间注入镜像拉取异常 Pod")
			injectMonitoringPodImagePullError(workloadKubeconfigPath)

			By("3. 将业务集群 ClusterVersion 目标版本更新为 rc3")
			patchBusinessClusterVersionAndWaitReady(clusterManager, duHelper, clusterName, bootstrapKubeconfigPath, businessAbnormalSourceVersion, businessAbnormalTargetRc3)

			By("4. 验证业务集群升级完成")
			verifyDeclarativeUpgradeResources(duHelper, clusterName, bootstrapKubeconfigPath, businessAbnormalTargetRc3)
		})

		// Author: ZYH-ang
		It("pod状态异常时触发跨版本升级-后端-高可用-业务集群-3m1w", Label("required_nodes:4", "P2", "26.06"), SpecTimeout(installationItTimeout+upgradeTimeout*2), func(ctx SpecContext) {
			By("1. 在引导节点准备 rc3 和正式版本 bkeagent")
			prepareGuideNodeBkeagent(businessAbnormalRc3AgentVersion)
			prepareGuideNodeBkeagent(businessAbnormalStableAgentVersion)

			By("2. Inject image pull error pod in business cluster monitoring namespace")
			injectMonitoringPodImagePullError(workloadKubeconfigPath)

			By("3. 将业务集群 ClusterVersion 目标版本更新为正式版本")
			patchBusinessClusterMultiHopVersionAndWaitReady(
				clusterManager,
				duHelper,
				clusterName,
				bootstrapKubeconfigPath,
				businessAbnormalSourceVersion,
				businessAbnormalTargetRc3,
				businessAbnormalTargetStable,
			)

			By("4. Verify business cluster upgrade completed")
			verifyDeclarativeUpgradeResources(duHelper, clusterName, bootstrapKubeconfigPath, businessAbnormalTargetStable)
		})

		// Author: ZYH-ang
		It("后端升级时配置不存在版本号", Label("required_nodes:4", "P2", "invalid-version"), SpecTimeout(installationItTimeout), func(ctx SpecContext) {
			By("1. 在业务集群 monitoring 命名空间注入镜像拉取异常 Pod")
			injectMonitoringPodImagePullError(workloadKubeconfigPath)

			By("2. 将业务集群 ClusterVersion 目标版本更新为不存在的版本")
			err := duHelper.PatchClusterVersionDesiredVersion(
				clusterName,
				bootstrapKubeconfigPath,
				businessInvalidTargetVersion,
			)
			Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion should succeed")

			By("3. 验证目标版本已写入，但当前版本未推进")
			Eventually(func() string {
				cv, e := duHelper.GetClusterVersion(clusterName, bootstrapKubeconfigPath)
				if e != nil {
					return ""
				}
				GinkgoWriter.Printf("invalid version CV: desired=%s current=%s phase=%s\n", cv.DesiredVersion, cv.CurrentVersion, cv.Phase)
				return cv.DesiredVersion
			}, 5*time.Minute, pollInterval).Should(Equal(businessInvalidTargetVersion), "CV desiredVersion should be updated")
			Consistently(func() string {
				cv, e := duHelper.GetClusterVersion(clusterName, bootstrapKubeconfigPath)
				if e != nil {
					return ""
				}
				return cv.CurrentVersion
			}, 3*time.Minute, pollInterval).Should(Equal(businessAbnormalSourceVersion), "currentVersion should not advance to invalid version")

			By("4. 检查 bke-controller-manager 的版本相关错误日志")
			ssh, err := executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
			Expect(err).NotTo(HaveOccurred())
			defer ssh.Close()
			result, err := ssh.Exec(`KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl logs -n cluster-system deploy/bke-controller-manager --tail=300 | grep -Ei "(not found|releaseimage|version|failed|error)" | tail -n 20`)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.ExitCode).To(BeZero(), "should find error keywords in bke-controller-manager logs")
		})
	})
	Describe("3m1w管理集群升级后扩容缩容卸载", Label("online", "pre-init", "skip-temporarily"), func() {
		var (
			clusterName        string
			clusterConfig      *config.BKEClusterConfig
			clusterNodes       []config.NodeInfo
			cleanupNodes       []config.NodeInfo
			scaleNode          config.NodeInfo
			configPath         string
			nodeConfigPath     string
			mgmtKubeconfigPath string
		)

		BeforeEach(func() {
			configPath = ""
			nodeConfigPath = ""
			mgmtKubeconfigPath = ""
			cleanupNodes = nil
			clusterName = fmt.Sprintf("mgmt-upgrade-scale-3m1w-%d", time.Now().UnixNano())

			var err error
			clusterNodes, err = utils.LoadDeclarativeUpgrade3M1WMgmtNodes()
			Expect(err).NotTo(HaveOccurred())
			cleanupNodes = append([]config.NodeInfo{}, clusterNodes...)
			clusterConfig = utils.NewDeclarativeUpgradeMgmtClusterConfig(clusterName, clusterNodes)

			nodes := config.LoadTestNodesFromEnv()
			Expect(len(nodes)).To(BeNumerically(">=", 5), "3m1w 管理集群扩缩容用例需要至少配置 5 个测试节点")
			scaleNode = nodes[4]
			scaleNode.Hostname = "mw2"
			scaleNode.Role = []string{"node"}
			cleanupNodes = append(cleanupNodes, scaleNode)

			By("1. 创建高可用 3m1w 管理集群")
			configPath, nodeConfigPath, mgmtKubeconfigPath = runDeclarativeUpgradeMgmtInstallAndVerify(
				clusterManager,
				sshExecutor,
				duHelper,
				clusterConfig,
				clusterName,
			)
		})

		AfterEach(func() {
			defer resetGuideNodeBKE(sshExecutor, mainBKEVersion, originalBKELink)
			if clusterName == "" {
				return
			}
			cleanupDeclarativeUpgradeMgmtCluster(
				clusterManager,
				sshExecutor,
				localExecutor,
				clusterName,
				mgmtKubeconfigPath,
				cleanupNodes,
				configPath,
				nodeConfigPath,
			)
		})

		// Author: ZYH-ang
		It("高可用管理集群-3m1w升级后-扩容-缩容-卸载", SpecTimeout(installationItTimeout+upgradeTimeout*2+installationItTimeout), func(ctx SpecContext) {
			By("2. 编辑 ClusterVersion desiredVersion 为跨 2 个可升级版本")
			runDeclarativeUpgradeMgmtMultiHopUpgradeAndVerify(
				clusterManager,
				duHelper,
				clusterName,
				mgmtKubeconfigPath,
				utils.DeclarativeUpgradeOpenFuyaoVersion,
				utils.DeclarativeUpgradeTargetOpenFuyaoVersion,
				utils.DeclarativeUpgradeSecondTargetOpenFuyaoVersion,
			)

			By("3. 查看并验证管理集群状态和 Pod 状态")
			verifyManagementClusterSelfHealthyAndPods(clusterManager, sshExecutor, mgmtKubeconfigPath, clusterName)

			By("4. 触发扩容")
			err := clusterManager.ScaleOutNodeWithKubeconfig(utils.ManagementClusterDeclarativeResourceClusterName(clusterName), scaleNode, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "管理集群扩容应该成功触发")
			waitManagementClusterNodeReadyCount(clusterManager, sshExecutor, mgmtKubeconfigPath, clusterName, 5)

			By("5. 触发缩容")
			err = clusterManager.ScaleInNodeWithKubeconfig(utils.ManagementClusterDeclarativeResourceClusterName(clusterName), scaleNode.IP, mgmtKubeconfigPath)
			Expect(err).NotTo(HaveOccurred(), "管理集群缩容应该成功触发")
			waitManagementClusterNodeReadyCount(clusterManager, sshExecutor, mgmtKubeconfigPath, clusterName, 4)

			By("6. 缩容后检查管理集群状态正常")
			verifyManagementClusterSelfHealthyAndPods(clusterManager, sshExecutor, mgmtKubeconfigPath, clusterName)

			By("7. 卸载管理集群")
			cleanupDeclarativeUpgradeMgmtCluster(
				clusterManager,
				sshExecutor,
				localExecutor,
				clusterName,
				mgmtKubeconfigPath,
				cleanupNodes,
				configPath,
				nodeConfigPath,
			)
			clusterName = ""
			mgmtKubeconfigPath = ""
			configPath = ""
			nodeConfigPath = ""
			cleanupNodes = nil
		})
	})
})

func resetGuideNodeBKE(sshExecutor executor.Executor, mainBKEVersion, originalBKELink string) {
	if sshExecutor == nil {
		return
	}

	By("清理引导节点 bke reset --all --mount 并恢复主版本 bke 软链接")
	cmd := fmt.Sprintf(`set +e
if command -v bke >/dev/null 2>&1; then
  echo "y" | bke reset --all --mount
  reset_code=$?
else
  echo "bke command not found, skip guide reset"
  reset_code=0
fi
MAIN_BKE_LINK="bke-%s"
if [ -x "/usr/local/bin/${MAIN_BKE_LINK}" ]; then
  ln -sfn "${MAIN_BKE_LINK}" /usr/local/bin/bke
elif [ -n %s ]; then
  ln -sfn %s /usr/local/bin/bke
fi
bke version || true
exit $reset_code`, shQuote(mainBKEVersion), shQuote(originalBKELink), shQuote(originalBKELink))
	result, err := sshExecutor.Exec(cmd)
	if err != nil || result == nil || result.ExitCode != 0 {
		stdout := ""
		stderr := ""
		exitCode := -1
		if result != nil {
			stdout = result.Stdout
			stderr = result.Stderr
			exitCode = result.ExitCode
		}
		GinkgoWriter.Printf("清理引导节点 bke reset --all --mount 或恢复主版本 bke 软链接失败: err=%v exitCode=%d stdout=%s stderr=%s\n", err, exitCode, stdout, stderr)
	}
}
func waitManagementClusterNodeReadyCount(clusterManager *utils.ClusterManager, sshExecutor executor.Executor, mgmtKubeconfigPath, bootstrapClusterName string, expected int) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName)
		if err != nil {
			GinkgoWriter.Printf("获取管理集群自管 BC 状态失败: %v\n", err)
			return false
		}

		result, err := sshExecutor.Exec(fmt.Sprintf(
			`KUBECONFIG=%s kubectl get nodes --no-headers | grep ' Ready' | wc -l`,
			shQuote(mgmtKubeconfigPath),
		))
		if err != nil || result.ExitCode != 0 {
			GinkgoWriter.Printf("获取管理集群 Ready 节点数失败: err=%v stderr=%s\n", err, result.Stderr)
			return false
		}
		readyCount := strings.TrimSpace(result.Stdout)
		GinkgoWriter.Printf("管理集群扩缩容状态: phase=%s state=%s clusterStatus=%s readyNodes=%s expected=%d\n",
			phase, state, clusterStatus, readyCount, expected)
		return readyCount == fmt.Sprintf("%d", expected) && clusterStatus == "Ready"
	}, installationItTimeout, pollInterval).Should(BeTrue(), "管理集群节点数和自管 BC 状态应该就绪")
}

func verifyManagementClusterSelfHealthyAndPods(clusterManager *utils.ClusterManager, sshExecutor executor.Executor, mgmtKubeconfigPath, bootstrapClusterName string) {
	Eventually(func() bool {
		phase, state, clusterStatus, err := clusterManager.GetManagementClusterSelfFullStatus(mgmtKubeconfigPath, bootstrapClusterName)
		if err != nil {
			GinkgoWriter.Printf("获取管理集群自管 BC 状态失败: %v\n", err)
			return false
		}
		result, err := sshExecutor.Exec(fmt.Sprintf(
			`KUBECONFIG=%s kubectl get pod -A --no-headers | awk '$4!="Completed" && $4!="Running" {print; bad=1} END {exit bad}'`,
			shQuote(mgmtKubeconfigPath),
		))
		if err != nil || result.ExitCode != 0 {
			GinkgoWriter.Printf("管理集群存在异常 Pod:\nstdout=%s\nstderr=%s\nerr=%v\n", result.Stdout, result.Stderr, err)
			return false
		}
		GinkgoWriter.Printf("管理集群健康状态: phase=%s state=%s clusterStatus=%s\n", phase, state, clusterStatus)
		return clusterStatus == "Ready"
	}, 20*time.Minute, pollInterval).Should(BeTrue(), "管理集群自管 BC 应该为 Ready，且所有非 Completed Pod 应该为 Running")
}
func runBusinessAbnormalInstallAndVerify(
	clusterManager *utils.ClusterManager,
	sshExecutor executor.Executor,
	duHelper *utils.DeclarativeUpgradeHelper,
	clusterConfig *config.BKEClusterConfig,
	clusterName, kubeconfigPath string,
) (configPath, nodeConfigPath string) {
	By("生成业务集群 Pod 异常场景配置文件")
	var err error
	configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	Expect(err).NotTo(HaveOccurred(), "should generate business cluster config files")

	err = patchBusinessAbnormalEtcdVersion(sshExecutor, configPath)
	Expect(err).NotTo(HaveOccurred(), "should set abnormal scenario etcdVersion")
	GinkgoWriter.Printf("business abnormal config files: cluster=%s, nodes=%s\n", configPath, nodeConfigPath)

	By("应用业务集群的 BKENode 和 BKECluster 资源")
	err = clusterManager.CreateClusterWithKubeconfig(configPath, nodeConfigPath, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "apply BC/BN should succeed")

	By("等待业务 BKECluster 变为 Ready")
	Eventually(func() bool {
		phase, state, clusterStatus, _ := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		GinkgoWriter.Printf("business BC status: phase=%s state=%s clusterStatus=%s\n", phase, state, clusterStatus)
		failOnClusterFailure(state, clusterStatus)
		return clusterStatus == "Ready"
	}, businessAbnormalInstallTimeout, pollInterval).Should(BeTrue(), "业务 BKECluster 应达 Ready")

	verifyDeclarativeUpgradeResources(duHelper, clusterName, kubeconfigPath, businessAbnormalSourceVersion)
	return configPath, nodeConfigPath
}

func patchBusinessAbnormalEtcdVersion(sshExecutor executor.Executor, clusterConfigPath string) error {
	cmd := fmt.Sprintf(
		`yq -i '.spec.clusterConfig.cluster.etcdVersion = "%s"' %s`,
		utils.DeclarativeUpgradeMgmtAbnormalEtcdVersion,
		shQuote(clusterConfigPath),
	)
	result, err := sshExecutor.Exec(cmd)
	if err != nil {
		return fmt.Errorf("set abnormal etcdVersion failed: %w", err)
	}
	if result.ExitCode != 0 {
		return fmt.Errorf("set abnormal etcdVersion failed: %s", result.Stderr)
	}
	return nil
}
func initGuideNodeForRc2() string {
	guideNodeConfig := config.LoadGuideNodeFromEnv()
	guideNodeHost := guideNodeConfig.Host
	guideNodePassword := guideNodeConfig.Password
	port := guideNodeConfig.Port

	By("在引导节点临时切换到 rc2 版本 bkeadm")
	ssh, err := executor.NewSSHExecutor(guideNodeHost, port, "root", guideNodePassword)
	Expect(err).NotTo(HaveOccurred())
	defer ssh.Close()

	cmd := `set -e
ORIGINAL_BKE_LINK=$(readlink /usr/local/bin/bke 2>/dev/null || true)
ARCH=$(uname -m)
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac
TARGET="/usr/local/bin/bke-26.6.0-rc.2"
wget -q -O "$TARGET" "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/bkeadm/releases/download/26.6.0-rc.2/bkeadm_linux_${ARCH}"
chmod +x "$TARGET"
ln -sfn "$(basename "$TARGET")" /usr/local/bin/bke
printf '%s' "$ORIGINAL_BKE_LINK"`
	result, err := ssh.Exec(cmd)
	Expect(err).NotTo(HaveOccurred())
	Expect(result.ExitCode).To(BeZero(), result.Stderr)
	originalBKELink := strings.TrimSpace(result.Stdout)
	GinkgoWriter.Printf("original bke symlink target: %q\n", originalBKELink)

	result, err = ssh.Exec("echo Y | bke init --onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:26.6.0-rc.2")
	Expect(err).NotTo(HaveOccurred())
	Expect(result.ExitCode).To(BeZero(), result.Stderr)
	Expect(result.Stdout).To(ContainSubstring("BKE initialization is complete"), "bke init 日志应包含初始化完成信息")

	By("等待 rc2 引导集群所有 Pod 就绪")
	Eventually(func() bool {
		podsRes, podsErr := ssh.Exec("KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl get pod -A --no-headers")
		if podsErr != nil || podsRes.ExitCode != 0 {
			return false
		}
		lines := strings.Split(strings.TrimSpace(podsRes.Stdout), "\n")
		if len(lines) == 0 {
			return false
		}
		for _, line := range lines {
			if strings.TrimSpace(line) == "" {
				continue
			}
			fields := strings.Fields(line)
			if len(fields) < 4 {
				continue
			}
			status := fields[3]
			if status != "Running" && status != "Completed" {
				GinkgoWriter.Printf("Pod 未就绪: %s (Status: %s)\n", fields[1], status)
				return false
			}
		}
		return true
	}, 20*time.Minute, 10*time.Second).Should(BeTrue(), "rc2 引导集群所有 Pod 应该处于 Running 或 Completed 状态")
	return originalBKELink
}

func prepareGuideNodeBkeagent(version string) {
	guideNodeConfig := config.LoadGuideNodeFromEnv()
	guideNodeHost := guideNodeConfig.Host
	guideNodePassword := guideNodeConfig.Password
	port := guideNodeConfig.Port

	By(fmt.Sprintf("同步引导节点 bkeagent 二进制，版本：%s", version))
	ssh, err := executor.NewSSHExecutor(guideNodeHost, port, "root", guideNodePassword)
	Expect(err).NotTo(HaveOccurred())
	defer ssh.Close()

	cmd := fmt.Sprintf(`set -e
VERSION=%s
ARCH=$(uname -m)
case "$ARCH" in
  x86_64) ARCH="amd64" ;;
  aarch64|arm64) ARCH="arm64" ;;
esac
TARGET="/bke/mount/source_registry/files/bkeagent-${VERSION}-linux-${ARCH}"
mkdir -p /bke/mount/source_registry/files
wget -q -O "$TARGET" "https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/cluster-api-provider-bke/releases/download/${VERSION}/bkeagent_linux_${ARCH}"
chmod 0644 "$TARGET"
ls -l "$TARGET"`, shQuote(version))
	result, err := ssh.Exec(cmd)
	Expect(err).NotTo(HaveOccurred())
	Expect(result.ExitCode).To(BeZero(), result.Stderr)
}

func patchBusinessClusterVersionAndWaitReady(clusterManager *utils.ClusterManager, duHelper *utils.DeclarativeUpgradeHelper, clusterName, kubeconfigPath, sourceVersion, targetVersion string) {
	verifyDeclarativeUpgradeUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, targetVersion)

	cvBefore, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "should read ClusterVersion before upgrade")
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "currentVersion before upgrade should equal sourceVersion")

	By(fmt.Sprintf("更新业务集群 ClusterVersion.spec.desiredVersion=%s", targetVersion))
	err = duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion should succeed")

	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("get business BC status failed: %v\n", e)
			return false
		}
		cv, cvErr := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if cvErr != nil {
			GinkgoWriter.Printf("get business CV failed: %v\n", cvErr)
			return false
		}
		GinkgoWriter.Printf("business upgrade status: phase=%s state=%s clusterStatus=%s desired=%s current=%s cvPhase=%s\n", phase, state, clusterStatus, cv.DesiredVersion, cv.CurrentVersion, cv.Phase)
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("business cluster upgrade failed, BC state=%s phase=%s clusterStatus=%s", state, phase, clusterStatus))
		}
		return cv.DesiredVersion == targetVersion &&
			cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed")
	}, upgradeTimeout, pollInterval).Should(BeTrue(), "business ClusterVersion should finish upgrade")
}

func patchBusinessClusterMultiHopVersionAndWaitReady(clusterManager *utils.ClusterManager, duHelper *utils.DeclarativeUpgradeHelper, clusterName, kubeconfigPath, sourceVersion, intermediateVersion, targetVersion string) {
	verifyDeclarativeUpgradeMultiHopUpgradePathForUpgrade(duHelper, kubeconfigPath, sourceVersion, intermediateVersion, targetVersion)

	cvBefore, err := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "should read ClusterVersion before multi-hop upgrade")
	Expect(cvBefore.CurrentVersion).To(Equal(sourceVersion), "currentVersion before multi-hop upgrade should equal sourceVersion")

	By(fmt.Sprintf("Update business ClusterVersion.spec.desiredVersion=%s for multi-hop upgrade", targetVersion))
	err = duHelper.PatchClusterVersionDesiredVersion(clusterName, kubeconfigPath, targetVersion)
	Expect(err).NotTo(HaveOccurred(), "patch ClusterVersion desiredVersion should succeed")

	sawIntermediate := false
	Eventually(func() bool {
		phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, kubeconfigPath)
		if e != nil {
			GinkgoWriter.Printf("get business BC status failed: %v\n", e)
			return false
		}
		cv, cvErr := duHelper.GetClusterVersion(clusterName, kubeconfigPath)
		if cvErr != nil {
			GinkgoWriter.Printf("get business CV failed: %v\n", cvErr)
			return false
		}
		if cv.CurrentVersion == intermediateVersion {
			sawIntermediate = true
		}
		GinkgoWriter.Printf("business multi-hop upgrade status: phase=%s state=%s clusterStatus=%s desired=%s current=%s cvPhase=%s\n", phase, state, clusterStatus, cv.DesiredVersion, cv.CurrentVersion, cv.Phase)
		if state == "DeployFailed" {
			Fail(fmt.Sprintf("business cluster multi-hop upgrade failed, BC state=%s phase=%s clusterStatus=%s", state, phase, clusterStatus))
		}
		return cv.DesiredVersion == targetVersion &&
			cv.CurrentVersion == targetVersion &&
			(cv.Phase == "Ready" || cv.Phase == "Installed")
	}, upgradeTimeout*2, pollInterval).Should(BeTrue(), "business ClusterVersion should finish multi-hop upgrade")

	if sawIntermediate {
		GinkgoWriter.Printf("observed intermediate version %s during multi-hop upgrade\n", intermediateVersion)
	} else {
		GinkgoWriter.Printf("intermediate version %s was not observed during polling, it may have advanced quickly\n", intermediateVersion)
	}
}

func injectMonitoringPodImagePullError(kubeconfigPath string) {
	Expect(kubeconfigPath).NotTo(BeEmpty(), "business cluster kubeconfig path is required")
	guideNodeConfig := config.LoadGuideNodeFromEnv()
	guideNodeHost := guideNodeConfig.Host
	guideNodePassword := guideNodeConfig.Password
	port := guideNodeConfig.Port
	const deploymentName = "blackbox-exporter"

	ssh, err := executor.NewSSHExecutor(guideNodeHost, port, "root", guideNodePassword)
	Expect(err).NotTo(HaveOccurred())
	defer ssh.Close()

	cmd := fmt.Sprintf(`set -eu
KUBECONFIG=%s
NS=monitoring
DEPLOY=%s
WAIT_SECONDS=%d

for _ in $(seq 1 $((WAIT_SECONDS / 10))); do
  if kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get deploy "$DEPLOY" >/dev/null 2>&1; then
    break
  fi
  sleep 10
done

if ! kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get deploy "$DEPLOY" >/dev/null 2>&1; then
  echo "deployment $NS/$DEPLOY was not created within ${WAIT_SECONDS}s" >&2
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get deploy,pod -o wide >&2 || true
  exit 1
fi

if ! kubectl --kubeconfig "$KUBECONFIG" -n "$NS" wait --for=condition=Available "deployment/$DEPLOY" --timeout="${WAIT_SECONDS}s"; then
  echo "deployment $NS/$DEPLOY did not become Available" >&2
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get deploy,pod -o wide >&2 || true
  kubectl --kubeconfig "$KUBECONFIG" -n "$NS" describe deploy "$DEPLOY" >&2 || true
  exit 1
fi

kubectl --kubeconfig "$KUBECONFIG" -n "$NS" patch deploy "$DEPLOY" --type='json' -p='[{"op":"replace","path":"/spec/template/spec/containers/0/image","value":"cr.openfuyao.cn/library/this-image-does-not-exist:latest"}]'
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" rollout restart deploy "$DEPLOY"

for _ in $(seq 1 $((WAIT_SECONDS / 10))); do
  REASONS=$(for pod in $(kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get pod -o name | sed 's#^pod/##' | grep '^blackbox-exporter-' || true); do
    kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get pod "$pod" -o jsonpath='{range .status.containerStatuses[*]}{.state.waiting.reason}{"\n"}{end}'
  done)
  if echo "$REASONS" | grep -Eq 'ImagePullBackOff|ErrImagePull|InvalidImageName'; then
    kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get pod -o wide | grep '^blackbox-exporter-' || true
    exit 0
  fi
  sleep 10
done

echo "deployment $NS/$DEPLOY did not reach ImagePullBackOff" >&2
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" get pod -o wide >&2 || true
kubectl --kubeconfig "$KUBECONFIG" -n "$NS" describe deploy "$DEPLOY" >&2 || true
exit 1`, shQuote(kubeconfigPath), deploymentName, int(businessAbnormalMonitoringReadyTimeout.Seconds()))
	result, err := ssh.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("inject pod abnormal failed: %s", result.Stderr))
	Expect(result.ExitCode).To(BeZero(), fmt.Sprintf("inject pod abnormal failed: %s", result.Stderr))
}
func shQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\"'\"'") + "'"
}
