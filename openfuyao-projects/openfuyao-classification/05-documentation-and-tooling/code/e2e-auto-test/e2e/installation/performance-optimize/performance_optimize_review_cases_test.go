package performance_optimize

import (
	"context"
	"encoding/base64"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

const (
	defaultBKEClusterNamespace = "bke-cluster"
	defaultBKEClusterName      = "bke-cluster"
	defaultConfigNamespace     = "bke-config"
	defaultHealthConfigName    = "health-check-config"
	defaultControllerNamespace = "bke-system"
	defaultControllerSelector  = "control-plane=controller-manager"

	generatedManagementContext = "__performance_optimize_management__"
	generatedTargetContext     = "__performance_optimize_target__"

	performanceManagementInstallTimeout = 25 * time.Minute
	performanceWorkloadInstallTimeout   = 20 * time.Minute
	performanceDeleteTimeout            = 15 * time.Minute
	performancePollInterval             = 30 * time.Second

	performanceSpecTimeout10Min = 10 * time.Minute
	performanceSpecTimeout15Min = 15 * time.Minute
	performanceSpecTimeout20Min = 20 * time.Minute
	performanceSpecTimeout25Min = 25 * time.Minute
	performanceSpecTimeout30Min = 30 * time.Minute
)

type performanceCluster struct {
	name           string
	configPath     string
	nodeConfigPath string
	kubeconfigPath string
	nodes          []config.NodeInfo
	reused         bool
}

var (
	perfSSHExecutor       *executor.SSHExecutor
	perfLocalExecutor     *executor.LocalExecutor
	perfClusterManager    *utils.ClusterManager
	perfManagementCluster *performanceCluster
	perfWorkloadCluster   *performanceCluster
)

var _ = BeforeSuite(func(ctx SpecContext) {
	By("Preparing performance optimize management and workload clusters")
	manager, localExecutor, sshExecutor := newPerformanceClusterManager()
	perfClusterManager = manager
	perfLocalExecutor = localExecutor
	perfSSHExecutor = sshExecutor

	mgmtCluster, err := createPerformanceManagementCluster(ctx, manager, localExecutor)
	Expect(err).NotTo(HaveOccurred(), "management cluster should be created before review cases")
	perfManagementCluster = mgmtCluster
	_ = os.Setenv("PERFORMANCE_OPTIMIZE_MANAGEMENT_KUBECONFIG", mgmtCluster.kubeconfigPath)
	GinkgoWriter.Printf("[performance-optimize setup] management cluster ready: name=%s kubeconfig=%s nodes=%s\n",
		mgmtCluster.name, mgmtCluster.kubeconfigPath, nodeIPs(mgmtCluster.nodes))
	GinkgoWriter.Printf("[performance-optimize setup] 管理集群已经创建完毕,请确定是否需要修改bke-controller镜像?\n")
	GinkgoWriter.Printf("[performance-optimize setup] 等待五分钟\n")
	select {
	case <-ctx.Done():
		Fail(fmt.Sprintf("context canceled while waiting after management cluster creation: %v", ctx.Err()))
	case <-time.After(5 * time.Minute):
	}

	workloadCluster, err := createPerformanceWorkloadCluster(ctx, manager, localExecutor, mgmtCluster.kubeconfigPath)
	Expect(err).NotTo(HaveOccurred(), "workload cluster should be created before review cases")
	perfWorkloadCluster = workloadCluster
	_ = os.Setenv("PERFORMANCE_OPTIMIZE_TARGET_KUBECONFIG", workloadCluster.kubeconfigPath)
	GinkgoWriter.Printf("[performance-optimize setup] workload cluster ready: name=%s kubeconfig=%s nodes=%s\n",
		workloadCluster.name, workloadCluster.kubeconfigPath, nodeIPs(workloadCluster.nodes))
})

var _ = AfterSuite(func(ctx SpecContext) {
	defer func() {
		if perfSSHExecutor != nil {
			_ = perfSSHExecutor.Close()
		}
	}()

	runAfterSuiteCleanupAttempt(ctx, 1, true)
	runAfterSuiteCleanupAttempt(ctx, 2, false)
})

var _ = Describe("Installation performance optimize FIT cases",
	Label("installation", "performance-optimize", "v26.09"), func() {
		It("【健康检查配置_FIT_1】健康检查配置自动同步到目标集群",
			Label("required_nodes:3", "健康检查配置_FIT_1", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout30Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE2")
				sshLoginNode2(ctx)

				By("检查管理集群存在默认健康检查 ConfigMap")
				managementConfig := getConfigMapData(ctx, managementContext(), defaultConfigNamespace, defaultHealthConfigName, "config.yaml")
				Expect(managementConfig).NotTo(BeEmpty(), "管理集群 health-check-config 的 data.config.yaml 应非空")

				By("检查目标集群同步同名 ConfigMap")
				targetConfig := getConfigMapData(ctx, targetContext(), defaultConfigNamespace, defaultHealthConfigName, "config.yaml")
				Expect(targetConfig).NotTo(BeEmpty(), "目标集群 health-check-config 的 data.config.yaml 应非空")
			})

		It("【健康检查配置_FIT_2】从 ConfigMap 加载自定义健康检查配置并验证修改生效",
			Label("required_nodes:3", "健康检查配置_FIT_2", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout10Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE3 更新目标集群 ConfigMap 为有效自定义配置")
				applyHealthCheckConfigFromNode3(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("SSH 登录 TEST_NODE2 检查 controller-manager 健康检查日志")
				expectCriticalHealthCheckLogOnNode2(ctx)
			})

		It("【健康检查配置_FIT_3】ConfigMap 缺失或非法配置时回退默认健康检查配置",
			Label("required_nodes:3", "健康检查配置_FIT_3", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout10Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE3 删除目标集群健康检查 ConfigMap")
				deleteHealthCheckConfigFromNode3(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("SSH 登录 TEST_NODE2 检查 controller-manager 健康检查配置同步日志")
				expectHealthCheckConfigSyncLogOnNode2(ctx)
			})

		It("【健康检查配置_FIT_4】关键组件失败触发 15 秒重试，可选组件失败触发 30 秒重试",
			Label("required_nodes:3", "健康检查配置_FIT_4", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout25Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE3 备份并修改目标集群健康检查 ConfigMap")
				applyCriticalHealthCheckFaultFromNode3(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("SSH 登录 TEST_NODE2 检查两条 critical 重试间隔日志")
				expectClusterHealthCheckRequeueAfterOnNode2(ctx, 13*time.Second, "5m", "5s")

				By("SSH 登录 TEST_NODE3 将 critical.yaml 中 etcd 优先级改为 important 并重新 apply")
				applyImportantHealthCheckFaultFromNode3(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("SSH 登录 TEST_NODE2 检查两条 important 重试间隔日志")
				expectClusterHealthCheckRequeueAfterOnNode2(ctx, 2*time.Minute, "5m", "15s")

				By("SSH 登录 TEST_NODE3 将 critical.yaml 中 etcd 优先级改为 optional 并重新 apply")
				applyOptionalHealthCheckFaultFromNode3(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("SSH 登录 TEST_NODE2 检查两条 optional 重试间隔日志")
				expectClusterHealthCheckRequeueAfterOnNode2(ctx, 2*time.Minute, "4m", "30s")

				By("SSH 登录 TEST_NODE3 恢复目标集群健康检查 ConfigMap")
				restoreHealthCheckConfigFromNode3(ctx)
			})

		It("【健康检查配置_FIT_5】管理集群-全部健康时使用 5 分钟正常检查间隔",
			Label("required_nodes:3", "健康检查配置_FIT_5", "P0", "post-init", "online"),
			SpecTimeout(performanceSpecTimeout20Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE2 后等待自动健康检查")
				sshLoginNode2(ctx)

				By("等待 6 分钟，等待自动触发新一轮健康检查")
				waitForAutomaticHealthCheck(ctx, 6*time.Minute)

				By("检查节点和核心组件均健康")
				runKubectl(ctx, targetContext(), "wait", "--for=condition=Ready", "nodes", "--all", "--timeout=5m")
				waitForPodsReady(ctx, targetContext(), "kube-system", "k8s-app=kube-dns", 5*time.Minute)

				By("检查下一次调谐间隔为 5m")
				logs := controllerLogs(ctx)
				Expect(logs).To(Or(ContainSubstring("5m"), ContainSubstring("300s")), "健康时下一次调谐间隔应为 5m")
			})

		It("【健康检查配置_FIT_6】oauth-webhook 先就绪再重启 APIServer",
			Label("required_nodes:3", "健康检查配置_FIT_6", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout10Min),
			func(ctx SpecContext) {
				By("SSH 登录 GUIDE_NODE 检查 bke.log 中 webhook 与 k3s 重启顺序")
				assertGuideNodeOauthWebhookBeforeK3sRestart(ctx)
			})
	})

var _ = Describe("Calico deployment optimize FIT cases",
	Label("installation", "performance-optimize", "calico", "v26.09"), func() {
		It("【calico部署方式优化_FIT_1】管理集群-节点环境初始化预拉取 Calico 核心镜像",
			Label("required_nodes:4", "calico部署方式优化_FIT_1", "P0", "post-init", "online"),
			SpecTimeout(performanceSpecTimeout15Min),
			func(ctx SpecContext) {
				By("SSH 登录 GUIDE_NODE 使用错误 Calico 版本创建集群并检查初始化失败")
				createClusterWithBadCalicoImageFromGuideNode(ctx)

				By("SSH 登录 GUIDE_NODE 删除 TEST_NODE5 集群")
				deleteTestNode5ClusterFromGuideNode(ctx)
			})

		It("【calico部署方式优化_FIT_2】管理集群-节点数大于 50 正常启用 Typha",
			Label("required_nodes:3", "calico部署方式优化_FIT_2", "P0", "with-management-cluster", "online", "skip-temporarily"),
			SpecTimeout(performanceSpecTimeout20Min),
			func(ctx SpecContext) {
				By("SSH 登录 TEST_NODE3")
				sshLoginNode3(ctx)

				By("等待 Typha Ready 与 Endpoint 可用")
				waitForPodsReadyOnNode3(ctx, "kube-system", envOrDefault("TYPHA_SELECTOR", "k8s-app=calico-typha"), 10*time.Minute)
				endpoints := runKubectlOnNode3(ctx, "get", "endpoints", "-n", "kube-system", envOrDefault("TYPHA_SERVICE_NAME", "calico-typha"), "-o", "jsonpath={.subsets[*].addresses[*].ip}")
				Expect(strings.Fields(endpoints)).NotTo(BeEmpty(), "Typha Endpoint 应可用")

				By("检查 calico-node Ready")
				waitForPodsReadyOnNode3(ctx, "kube-system", "k8s-app=calico-node", 10*time.Minute)
			})

		It("【calico部署方式优化_FIT_3】Calico Addon 版本为空时部署失败",
			Label("required_nodes:4", "calico部署方式优化_FIT_3", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout10Min),
			func(ctx SpecContext) {
				By("SSH 登录 GUIDE_NODE 使用空 Calico version 创建集群并检查失败日志")
				createClusterWithEmptyCalicoVersionFromGuideNode(ctx)

				By("SSH 登录 GUIDE_NODE 删除 bke-cluster3/bke-cluster3 集群")
				deleteBKECluster3FromGuideNode(ctx)
			})

		It("【calico部署方式优化_FIT_4】多个 Calico Addon 时部署失败",
			Label("required_nodes:4", "calico部署方式优化_FIT_4", "P0", "with-management-cluster", "online"),
			SpecTimeout(performanceSpecTimeout10Min),
			func(ctx SpecContext) {
				By("SSH 登录 GUIDE_NODE 使用重复 Calico addon 创建集群并检查失败日志")
				createClusterWithDuplicateCalicoAddonFromGuideNode(ctx)

				By("SSH 登录 GUIDE_NODE 删除 bke-cluster/bke-cluster 集群")
				deleteTestNode5ClusterFromGuideNode(ctx)
			})
	})

const healthCheckConfigManifest = `apiVersion: v1
kind: Namespace
metadata:
  name: bke-config
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: health-check-config
  namespace: bke-config
data:
  config.yaml: |
    intervals:
      critical: 5s
      important: 15s
      optional: 30s
      normal: 5m
    components:
      - name: etcd
        namespace: kube-system
        prefixes:
          - etcd-
        priority: critical
      - name: kube-apiserver
        namespace: kube-system
        prefixes:
          - kube-apiserver-
        priority: critical
      - name: kube-controller-manager
        namespace: kube-system
        prefixes:
          - kube-controller-manager-
        priority: critical
      - name: kube-scheduler
        namespace: kube-system
        prefixes:
          - kube-scheduler-
        priority: critical
      - name: calico-node
        namespace: kube-system
        prefixes:
          - calico-node
        priority: important
      - name: calico-kube-controllers
        namespace: kube-system
        prefixes:
          - calico-kube-controllers
        priority: important
      - name: kube-proxy
        namespace: kube-system
        prefixes:
          - kube-proxy-
        priority: important
      - name: coredns
        namespace: kube-system
        prefixes:
          - coredns
        priority: critical
`

const healthCheckBackupConfigMapManifest = `apiVersion: v1
kind: ConfigMap
metadata:
  name: health-check-config
  namespace: bke-config
data:
  config.yaml: |
    intervals:
      critical: 5s
      important: 15s
      optional: 30s
      normal: 5m
    components:
      - name: etcd
        namespace: kube-system
        prefixes:
          - etcd-
        priority: critical
      - name: kube-apiserver
        namespace: kube-system
        prefixes:
          - kube-apiserver-
        priority: critical
      - name: kube-controller-manager
        namespace: kube-system
        prefixes:
          - kube-controller-manager-
        priority: critical
      - name: kube-scheduler
        namespace: kube-system
        prefixes:
          - kube-scheduler-
        priority: critical
      - name: calico-node
        namespace: kube-system
        prefixes:
          - calico-node
        priority: important
      - name: calico-kube-controllers
        namespace: kube-system
        prefixes:
          - calico-kube-controllers
        priority: important
      - name: kube-proxy
        namespace: kube-system
        prefixes:
          - kube-proxy-
        priority: important
      - name: coredns
        namespace: kube-system
        prefixes:
          - coredns
        priority: important
`

func newPerformanceClusterManager() (*utils.ClusterManager, *executor.LocalExecutor, *executor.SSHExecutor) {
	guideConfig := config.LoadGuideNodeFromEnv()
	Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST must be set")
	Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD must be set")

	sshExecutor, err := executor.NewSSHExecutor(
		guideConfig.Host,
		guideConfig.Port,
		guideConfig.Username,
		guideConfig.Password,
	)
	Expect(err).NotTo(HaveOccurred(), "should connect to guide node")

	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
	restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred(), "should load bootstrap kubeconfig")
	dynamicClient, err := dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred(), "should create dynamic client")

	return utils.NewClusterManager(sshExecutor, dynamicClient), executor.NewLocalExecutor(30 * time.Minute), sshExecutor
}

func resolveExistingPerformanceManagementCluster(ctx context.Context, manager *utils.ClusterManager, localExecutor *executor.LocalExecutor, nodes []config.NodeInfo) (*performanceCluster, error) {
	clusterName := strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_MANAGEMENT_CLUSTER_NAME"))
	if clusterName == "" {
		clusterName = findPerformanceClusterByPrefix(manager, "", "perf-opt-mgmt-")
	}
	if clusterName == "" {
		return nil, nil
	}
	GinkgoWriter.Printf("[performance-optimize setup] found existing management cluster candidate: name=%s\n", clusterName)
	kubeconfigPath, err := waitAndSavePerformanceClusterKubeconfig(ctx, localExecutor, manager, clusterName, "", "existing management", performanceManagementInstallTimeout)
	if err != nil {
		return nil, err
	}
	if err := waitForPerformanceManagementSelfReady(ctx, manager, kubeconfigPath, clusterName); err != nil {
		return nil, err
	}
	return &performanceCluster{name: clusterName, kubeconfigPath: kubeconfigPath, nodes: nodes, reused: true}, nil
}

func resolveExistingPerformanceWorkloadCluster(ctx context.Context, manager *utils.ClusterManager, localExecutor *executor.LocalExecutor, managementKubeconfigPath string, nodes []config.NodeInfo) (*performanceCluster, error) {
	clusterName := strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_WORKLOAD_CLUSTER_NAME"))
	if clusterName == "" {
		clusterName = findPerformanceClusterByPrefix(manager, managementKubeconfigPath, "perf-opt-workload-")
	}
	if clusterName == "" {
		return nil, nil
	}
	GinkgoWriter.Printf("[performance-optimize setup] found existing workload cluster candidate: name=%s parentKubeconfig=%s\n", clusterName, managementKubeconfigPath)
	if err := waitForPerformanceClusterReady(ctx, manager, localExecutor, clusterName, managementKubeconfigPath, len(nodes)); err != nil {
		return nil, err
	}
	kubeconfigPath, err := waitAndSavePerformanceClusterKubeconfig(ctx, localExecutor, manager, clusterName, managementKubeconfigPath, "existing workload", performanceWorkloadInstallTimeout)
	if err != nil {
		return nil, err
	}
	return &performanceCluster{name: clusterName, kubeconfigPath: kubeconfigPath, nodes: nodes, reused: true}, nil
}

func findPerformanceClusterByPrefix(manager *utils.ClusterManager, parentKubeconfigPath, prefix string) string {
	kubeconfigArg := ""
	if parentKubeconfigPath != "" {
		kubeconfigArg = fmt.Sprintf("KUBECONFIG=%s ", parentKubeconfigPath)
	}
	cmd := fmt.Sprintf(`%skubectl get bc -A --no-headers 2>/dev/null | awk '$2 ~ /^bke-%s/ {print $2}' | sed 's/^bke-//' | sort | tail -n 1`, kubeconfigArg, prefix)
	if perfSSHExecutor == nil {
		GinkgoWriter.Printf("[performance-optimize setup] cannot list existing clusters because guide SSH executor is nil: prefix=%s\n", prefix)
		return ""
	}
	result, err := perfSSHExecutor.Exec(cmd)
	if err != nil || result.ExitCode != 0 {
		GinkgoWriter.Printf("[performance-optimize setup] list existing clusters by prefix failed: prefix=%s err=%v\n", prefix, err)
		return ""
	}
	return strings.TrimSpace(result.Stdout)
}
func createPerformanceManagementCluster(ctx context.Context, manager *utils.ClusterManager, localExecutor *executor.LocalExecutor) (*performanceCluster, error) {
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	node.Role = []string{"master/node", "etcd"}
	if cluster, err := resolveExistingPerformanceManagementCluster(ctx, manager, localExecutor, []config.NodeInfo{node}); err == nil && cluster != nil {
		return cluster, nil
	}

	clusterName := fmt.Sprintf("perf-opt-mgmt-%d", time.Now().Unix())
	clusterConfig := config.NewBKEClusterConfigForMgmt(clusterName, []config.NodeInfo{node})

	GinkgoWriter.Printf("[performance-optimize setup] creating management cluster: name=%s nodes=%s\n", clusterName, nodeIPs(clusterConfig.Nodes))
	configPath, nodeConfigPath, err := manager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, fmt.Errorf("generate management cluster config: %w", err)
	}
	GinkgoWriter.Printf("[performance-optimize setup] management config uploaded: cluster=%s nodes=%s\n", configPath, nodeConfigPath)

	if err := manager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, ""); err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, nodes: clusterConfig.Nodes}, fmt.Errorf("create management cluster: %w", err)
	}
	kubeconfigPath, err := waitAndSavePerformanceClusterKubeconfig(ctx, localExecutor, manager, clusterName, "", "management", performanceManagementInstallTimeout)
	if err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, nodes: clusterConfig.Nodes}, err
	}
	GinkgoWriter.Printf("[performance-optimize setup] management kubeconfig saved: %s\n", kubeconfigPath)
	if err := waitForPerformanceManagementSelfReady(ctx, manager, kubeconfigPath, clusterName); err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, kubeconfigPath: kubeconfigPath, nodes: clusterConfig.Nodes}, err
	}

	return &performanceCluster{
		name:           clusterName,
		configPath:     configPath,
		nodeConfigPath: nodeConfigPath,
		kubeconfigPath: kubeconfigPath,
		nodes:          clusterConfig.Nodes,
	}, nil
}

func createPerformanceWorkloadCluster(ctx context.Context, manager *utils.ClusterManager, localExecutor *executor.LocalExecutor, managementKubeconfigPath string) (*performanceCluster, error) {
	masterNode := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	masterNode.Role = []string{"master/node", "etcd"}
	workerNode := loadPerformanceNodeFromEnv("TEST_NODE4", "pw2")
	workerNode.Role = []string{"node"}
	workloadNodes := []config.NodeInfo{masterNode, workerNode}
	if cluster, err := resolveExistingPerformanceWorkloadCluster(ctx, manager, localExecutor, managementKubeconfigPath, workloadNodes); err == nil && cluster != nil {
		return cluster, nil
	}

	clusterName := fmt.Sprintf("perf-opt-workload-%d", time.Now().Unix())
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, workloadNodes)

	GinkgoWriter.Printf("[performance-optimize setup] creating workload cluster: name=%s parentKubeconfig=%s nodes=%s\n", clusterName, managementKubeconfigPath, nodeIPs(clusterConfig.Nodes))
	configPath, nodeConfigPath, err := manager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, fmt.Errorf("generate workload cluster config: %w", err)
	}
	GinkgoWriter.Printf("[performance-optimize setup] workload config uploaded: cluster=%s nodes=%s\n", configPath, nodeConfigPath)

	if err := manager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, managementKubeconfigPath); err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, nodes: clusterConfig.Nodes}, fmt.Errorf("create workload cluster: %w", err)
	}
	if err := waitForPerformanceClusterReady(ctx, manager, localExecutor, clusterName, managementKubeconfigPath, 2); err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, nodes: clusterConfig.Nodes}, err
	}

	checker := utils.NewClusterCheckerWithParentKubeconfig(localExecutor, clusterName, managementKubeconfigPath)
	kubeconfigPath, err := checker.SaveKubeconfigToFile()
	if err != nil {
		return &performanceCluster{name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath, nodes: clusterConfig.Nodes}, fmt.Errorf("save workload kubeconfig: %w", err)
	}
	GinkgoWriter.Printf("[performance-optimize setup] workload kubeconfig saved: %s\n", kubeconfigPath)

	return &performanceCluster{
		name:           clusterName,
		configPath:     configPath,
		nodeConfigPath: nodeConfigPath,
		kubeconfigPath: kubeconfigPath,
		nodes:          clusterConfig.Nodes,
	}, nil
}

func runAfterSuiteCleanupAttempt(ctx context.Context, attempt int, ignoreFailure bool) {
	if ignoreFailure {
		defer func() {
			if recovered := recover(); recovered != nil {
				GinkgoWriter.Printf("[performance-optimize cleanup] cleanup attempt %d failed and ignored: %v\n", attempt, recovered)
			}
		}()
	}

	By(fmt.Sprintf("Cleaning GUIDE_NODE default-kubeconfig clusters, attempt %d", attempt))
	cleanupGuideNodeDefaultClusters(ctx)

	By(fmt.Sprintf("Cleaning performance optimize workload and management clusters, attempt %d", attempt))
	deletePerformanceClusters(ctx)
}
func cleanupGuideNodeDefaultClusters(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	command := `set +e
for cluster in bke-cluster bke-cluster3; do
  echo "[performance-optimize cleanup] checking default cluster ${cluster}/${cluster}"
  if kubectl get bc "${cluster}" -n "${cluster}" >/dev/null 2>&1; then
    kubectl annotate bc "${cluster}" -n "${cluster}" bke.bocloud.com/ignore-target-cluster-delete="false" --overwrite
    kubectl annotate bc "${cluster}" -n "${cluster}" bke.bocloud.com/ignore-namespace-delete="false" --overwrite
    kubectl patch bc "${cluster}" -n "${cluster}" --type=merge -p '{"spec":{"reset":true}}'
    kubectl delete bc "${cluster}" -n "${cluster}" --ignore-not-found=true
  else
    echo "[performance-optimize cleanup] default cluster ${cluster}/${cluster} not found, skip bc delete"
  fi
  kubectl delete bn -n "${cluster}" "${cluster}-m1" --ignore-not-found=true
  kubectl delete namespace "${cluster}" --ignore-not-found=true
  echo "[performance-optimize cleanup] submitted default cleanup for ${cluster}/${cluster}"
done
kubectl get bc -A | grep -E 'bke-cluster3|bke-cluster' || true`
	result, err := utils.ExecuteCommandOnNode(guideNode, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize cleanup] GUIDE_NODE default cleanup node=%s exit=%d stdout=%s stderr=%s\n", guideNode.IP, result.ExitCode, result.Stdout, result.Stderr)
	}
	if err != nil {
		GinkgoWriter.Printf("[performance-optimize cleanup] GUIDE_NODE default cleanup failed: %v\n", err)
	}
}
func deletePerformanceClusters(ctx context.Context) {
	if perfClusterManager == nil {
		return
	}
	deletePerformanceWorkloadCluster(ctx, perfClusterManager, perfWorkloadCluster, managementKubeconfigPath())
	deletePerformanceManagementCluster(ctx, perfClusterManager, perfManagementCluster)
}

func deletePerformanceWorkloadCluster(ctx context.Context, manager *utils.ClusterManager, cluster *performanceCluster, managementKubeconfigPath string) {
	if cluster == nil || cluster.name == "" || managementKubeconfigPath == "" {
		return
	}
	if cluster.reused {
		GinkgoWriter.Printf("[performance-optimize cleanup] deleting reused workload cluster: name=%s parentKubeconfig=%s\n", cluster.name, managementKubeconfigPath)
	} else {
		GinkgoWriter.Printf("[performance-optimize cleanup] deleting workload cluster: name=%s parentKubeconfig=%s\n", cluster.name, managementKubeconfigPath)
	}
	if manager.ClusterExistsWithKubeconfig(cluster.name, managementKubeconfigPath) {
		if err := manager.DeleteClusterWithKubeconfig(cluster.name, managementKubeconfigPath); err != nil {
			GinkgoWriter.Printf("[performance-optimize cleanup] trigger workload deletion failed: %v\n", err)
		}
	}
	waitForPerformanceClusterDeleted(ctx, manager, cluster.name, managementKubeconfigPath)
	cleanupPerformanceClusterFiles(manager, cluster)
}

func deletePerformanceManagementCluster(ctx context.Context, manager *utils.ClusterManager, cluster *performanceCluster) {
	if cluster == nil || cluster.name == "" {
		return
	}
	if cluster.reused {
		GinkgoWriter.Printf("[performance-optimize cleanup] deleting reused management cluster: name=%s kubeconfig=%s\n", cluster.name, cluster.kubeconfigPath)
	} else {
		GinkgoWriter.Printf("[performance-optimize cleanup] deleting management cluster: name=%s kubeconfig=%s\n", cluster.name, cluster.kubeconfigPath)
	}
	if manager.ClusterExistsWithKubeconfig(cluster.name, "") {
		if err := manager.DeleteClusterWithKubeconfig(cluster.name, ""); err != nil {
			GinkgoWriter.Printf("[performance-optimize cleanup] trigger management parent deletion failed: %v\n", err)
		}
	}
	if cluster.kubeconfigPath != "" {
		if err := manager.DeleteManagementClusterSelfBC(cluster.kubeconfigPath, cluster.name); err != nil {
			GinkgoWriter.Printf("[performance-optimize cleanup] trigger management self deletion failed: %v\n", err)
		}
	}
	waitForPerformanceClusterDeleted(ctx, manager, cluster.name, "")
	cleanupPerformanceClusterFiles(manager, cluster)
}

func waitAndSavePerformanceClusterKubeconfig(ctx context.Context, localExecutor *executor.LocalExecutor, manager *utils.ClusterManager, clusterName, parentKubeconfigPath, label string, installTimeout time.Duration) (string, error) {
	var kubeconfigPath string
	checker := utils.NewClusterCheckerWithParentKubeconfig(localExecutor, clusterName, parentKubeconfigPath)
	GinkgoWriter.Printf("[performance-optimize setup] waiting %s cluster kubeconfig: name=%s parentKubeconfig=%s\n", label, clusterName, parentKubeconfigPath)
	Eventually(func(g Gomega) {
		phase, state, clusterStatus, statusErr := manager.GetClusterFullStatusWithKubeconfig(clusterName, parentKubeconfigPath)
		if statusErr != nil {
			GinkgoWriter.Printf("[performance-optimize setup] %s parent BC status unavailable: name=%s err=%v\n", label, clusterName, statusErr)
		} else {
			GinkgoWriter.Printf("[performance-optimize setup] %s parent BC status: name=%s phase=%s state=%s status=%s\n", label, clusterName, phase, state, clusterStatus)
			g.Expect(state).NotTo(Equal("DeployFailed"), "%s parent BC entered a failure state", label)
			g.Expect(clusterStatus).NotTo(Equal("InitializationFailed"), "%s parent BC initialization failed", label)
		}

		path, err := checker.SaveKubeconfigToFile()
		if err != nil {
			GinkgoWriter.Printf("[performance-optimize setup] %s kubeconfig unavailable: name=%s err=%v\n", label, clusterName, err)
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		kubeconfigPath = path
	}, installTimeout, performancePollInterval).WithContext(ctx).Should(Succeed())
	return kubeconfigPath, nil
}
func waitForPerformanceManagementSelfReady(ctx context.Context, manager *utils.ClusterManager, kubeconfigPath, bootstrapClusterName string) error {
	GinkgoWriter.Printf("[performance-optimize setup] waiting management self BC ready: kubeconfig=%s cluster=%s\n", kubeconfigPath, bootstrapClusterName)
	Eventually(func(g Gomega) {
		phase, state, clusterStatus, err := manager.GetManagementClusterSelfFullStatus(kubeconfigPath, bootstrapClusterName)
		if err != nil {
			GinkgoWriter.Printf("[performance-optimize setup] management self BC status unavailable: kubeconfig=%s err=%v\n", kubeconfigPath, err)
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		GinkgoWriter.Printf("[performance-optimize setup] management self BC status: phase=%s state=%s status=%s\n", phase, state, clusterStatus)
		g.Expect(state).NotTo(Equal("DeployFailed"), "management self BC entered a failure state")
		g.Expect(clusterStatus).NotTo(Equal("InitializationFailed"), "management self BC initialization failed")
		g.Expect(clusterStatus).To(Equal("Ready"))
	}, performanceManagementInstallTimeout, performancePollInterval).WithContext(ctx).Should(Succeed())
	return nil
}
func waitForPerformanceClusterReady(ctx context.Context, manager *utils.ClusterManager, localExecutor *executor.LocalExecutor, clusterName, parentKubeconfigPath string, expectedNodes int) error {
	GinkgoWriter.Printf("[performance-optimize setup] waiting cluster ready: name=%s parentKubeconfig=%s\n", clusterName, parentKubeconfigPath)
	Eventually(func(g Gomega) {
		phase, state, clusterStatus, err := manager.GetClusterFullStatusWithKubeconfig(clusterName, parentKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[performance-optimize setup] cluster status unavailable: name=%s err=%v\n", clusterName, err)
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		GinkgoWriter.Printf("[performance-optimize setup] cluster status: name=%s phase=%s state=%s status=%s\n", clusterName, phase, state, clusterStatus)
		g.Expect(state).NotTo(Or(Equal("DeployFailed"), Equal("Unhealthy")), "cluster entered a failure state")
		g.Expect(clusterStatus).NotTo(Equal("InitializationFailed"), "cluster initialization failed")
		g.Expect(clusterStatus).To(Equal("Ready"))
	}, performanceWorkloadInstallTimeout, performancePollInterval).WithContext(ctx).Should(Succeed())

	checker := utils.NewClusterCheckerWithParentKubeconfig(localExecutor, clusterName, parentKubeconfigPath)
	Eventually(func(g Gomega) {
		count, err := checker.GetNodeCountWithKubeconfig(parentKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[performance-optimize setup] node count unavailable: name=%s err=%v\n", clusterName, err)
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		readyCount, err := checker.GetReadyNodeCountWithKubeconfig(parentKubeconfigPath)
		if err != nil {
			GinkgoWriter.Printf("[performance-optimize setup] ready node count unavailable: name=%s err=%v\n", clusterName, err)
			g.Expect(err).NotTo(HaveOccurred())
			return
		}
		GinkgoWriter.Printf("[performance-optimize setup] node status: name=%s nodeCount=%d readyNodeCount=%d\n", clusterName, count, readyCount)
		g.Expect(count).To(Equal(expectedNodes))
		g.Expect(readyCount).To(Equal(expectedNodes))
	}, 10*time.Minute, performancePollInterval).WithContext(ctx).Should(Succeed())
	return nil
}

func waitForPerformanceClusterDeleted(ctx context.Context, manager *utils.ClusterManager, clusterName, parentKubeconfigPath string) {
	Eventually(func() bool {
		exists := manager.ClusterExistsWithKubeconfig(clusterName, parentKubeconfigPath)
		if exists {
			phase, state, clusterStatus, _ := manager.GetClusterFullStatusWithKubeconfig(clusterName, parentKubeconfigPath)
			GinkgoWriter.Printf("[performance-optimize cleanup] cluster still exists: name=%s phase=%s state=%s status=%s\n", clusterName, phase, state, clusterStatus)
		}
		return !exists
	}, performanceDeleteTimeout, 60*time.Second).WithContext(ctx).Should(BeTrue(), "cluster %s should be deleted", clusterName)
}

func cleanupPerformanceClusterFiles(manager *utils.ClusterManager, cluster *performanceCluster) {
	for _, path := range []string{cluster.configPath, cluster.nodeConfigPath, cluster.kubeconfigPath} {
		if strings.TrimSpace(path) == "" {
			continue
		}
		if err := manager.CleanupConfig(path); err != nil {
			GinkgoWriter.Printf("[performance-optimize cleanup] cleanup file failed: path=%s err=%v\n", path, err)
		}
	}
}

func loadPerformanceNodeFromEnv(prefix, defaultHostname string) config.NodeInfo {
	node := config.NodeInfo{
		Hostname: envOrDefault(prefix+"_HOSTNAME", defaultHostname),
		IP:       strings.TrimSpace(os.Getenv(prefix + "_IP")),
		Port:     envOrDefault(prefix+"_PORT", "22"),
		Username: envOrDefault(prefix+"_USER", "root"),
		Password: os.Getenv(prefix + "_PASSWORD"),
	}
	Expect(node.IP).NotTo(BeEmpty(), "%s_IP must be set", prefix)
	Expect(node.Password).NotTo(BeEmpty(), "%s_PASSWORD must be set", prefix)
	return node
}

func nodeIPs(nodes []config.NodeInfo) string {
	ips := make([]string, 0, len(nodes))
	for _, node := range nodes {
		ips = append(ips, fmt.Sprintf("%s(%s:%s)", node.Hostname, node.IP, strings.Join(node.Role, ",")))
	}
	return strings.Join(ips, ",")
}

func managementKubeconfigPath() string {
	if perfManagementCluster != nil && perfManagementCluster.kubeconfigPath != "" {
		return perfManagementCluster.kubeconfigPath
	}
	return strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_MANAGEMENT_KUBECONFIG"))
}

func targetKubeconfigPath() string {
	if perfWorkloadCluster != nil && perfWorkloadCluster.kubeconfigPath != "" {
		return perfWorkloadCluster.kubeconfigPath
	}
	return strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_TARGET_KUBECONFIG"))
}
func managementContext() string {
	if managementKubeconfigPath() != "" {
		return generatedManagementContext
	}
	return strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_MANAGEMENT_CONTEXT"))
}

func targetContext() string {
	if targetKubeconfigPath() != "" {
		return generatedTargetContext
	}
	return strings.TrimSpace(os.Getenv("PERFORMANCE_OPTIMIZE_TARGET_CONTEXT"))
}

func bkeClusterNamespace() string {
	return envOrDefault("BKE_CLUSTER_NAMESPACE", defaultBKEClusterNamespace)
}

func bkeClusterName() string {
	return envOrDefault("BKE_CLUSTER_NAME", defaultBKEClusterName)
}

func triggerBKEClusterReconcile(ctx context.Context) {
	value := strconv.FormatInt(time.Now().UnixNano(), 10)
	runKubectl(ctx, managementContext(), "annotate", "bkecluster", bkeClusterName(), "-n", bkeClusterNamespace(), "e2e.openfuyao.io/last-trigger="+value, "--overwrite")
}

func triggerBKEClusterReconcileFromNode2(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	value := strconv.FormatInt(time.Now().UnixNano(), 10)
	command := fmt.Sprintf("kubectl annotate bkecluster %s -n %s e2e.openfuyao.io/last-trigger=%s --overwrite", bkeClusterName(), bkeClusterNamespace(), value)
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize trigger] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "trigger BKECluster reconcile should succeed on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "trigger BKECluster reconcile should return a result")
	Expect(result.ExitCode).To(Equal(0), "trigger BKECluster reconcile should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func sshLoginNode2(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	command := "true"
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize ssh] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "SSH login should succeed on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "SSH login should return a result")
	Expect(result.ExitCode).To(Equal(0), "SSH login command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func sshLoginNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := "true"
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize ssh] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "SSH login should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "SSH login should return a result")
	Expect(result.ExitCode).To(Equal(0), "SSH login command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func createClusterWithBadCalicoImageFromGuideNode(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	testNodeIP := strings.TrimSpace(os.Getenv("TEST_NODE5_IP"))
	Expect(testNodeIP).NotTo(BeEmpty(), "TEST_NODE5_IP must be set")
	testNodePassword := os.Getenv("TEST_NODE5_PASSWORD")
	Expect(testNodePassword).NotTo(BeEmpty(), "TEST_NODE5_PASSWORD must be set")
	prepareCommand := fmt.Sprintf(`set -e
cp -f /bke/cluster/1master* /opt/
cd /opt
echo "[FIT_1] update 1master-node.yaml with TEST_NODE5 ip and password"
NODE_IP=%s NODE_PASSWORD=%s perl -0pi -e 's/^(\s*ip:\s*).*?$/${1}$ENV{NODE_IP}/mg; s/^(\s*password:\s*).*?$/${1}$ENV{NODE_PASSWORD}/mg' 1master-node.yaml
echo "[FIT_1] update calico version to v3.31.99"
perl -0pi -e 's/(^([ \t]*)-\s+(?:(?!^\2-\s+).)*?name:\s*calico\b(?:(?!^\2-\s+).)*?version:\s*).*?(\n)/${1}v3.31.99${3}/ms' 1master-cluster.yaml
test "$(grep -c 'v3.31.99' 1master-cluster.yaml)" -ge 1
echo "[FIT_1] start bke cluster create in background"
rm -f /tmp/perf-opt-fit1-bke-create.log
nohup sh -c 'cd /opt && bke cluster create -f1master-cluster.yaml -n 1master-node.yaml' > /tmp/perf-opt-fit1-bke-create.log 2>&1 &
echo "[FIT_1] bke cluster create pid: $!"
echo "[FIT_1] bke cluster create log: /tmp/perf-opt-fit1-bke-create.log"`, shellQuote(testNodeIP), shellQuote(testNodePassword))
	prepareResult, err := utils.ExecuteCommandOnNode(guideNode, prepareCommand)
	if prepareResult != nil {
		GinkgoWriter.Printf("[performance-optimize calico] node=%s command=%s exit=%d stdout=%s stderr=%s\n", guideNode.IP, prepareCommand, prepareResult.ExitCode, prepareResult.Stdout, prepareResult.Stderr)
	}
	Expect(err).NotTo(HaveOccurred(), "bad calico image prepare command should succeed on GUIDE_NODE (%s)", guideNode.IP)
	Expect(prepareResult).NotTo(BeNil(), "bad calico image prepare command should return a result")
	Expect(prepareResult.ExitCode).To(Equal(0), "bad calico image prepare command should exit 0; stdout=%s stderr=%s", prepareResult.Stdout, prepareResult.Stderr)

	GinkgoWriter.Printf("[performance-optimize calico] wait 5 minutes before checking Calico addon failure status\n")
	select {
	case <-ctx.Done():
		Fail(fmt.Sprintf("context canceled while waiting before checking Calico addon failure status: %v", ctx.Err()))
	case <-time.After(5 * time.Minute):
	}

	checkCommand := `set -e
echo "[FIT_1] check background create process"
ps -ef | grep 'bke cluster create -f1master-cluster.yaml -n 1master-node.yaml' | grep -v grep || true
echo "[FIT_1] recent background create log"
tail -n 80 /tmp/perf-opt-fit1-bke-create.log || true
echo "[FIT_1] check BKECluster Calico addon failure status"
kubectl get bc bke-cluster -n bke-cluster -o yaml`
	checkResult, err := utils.ExecuteCommandOnNode(guideNode, checkCommand)
	if checkResult != nil {
		GinkgoWriter.Printf("[performance-optimize calico] node=%s command=%s exit=%d stdout=%s stderr=%s\n", guideNode.IP, checkCommand, checkResult.ExitCode, checkResult.Stdout, checkResult.Stderr)
	}
	Expect(err).NotTo(HaveOccurred(), "bad calico image check command should succeed on GUIDE_NODE (%s)", guideNode.IP)
	Expect(checkResult).NotTo(BeNil(), "bad calico image check command should return a result")
	Expect(checkResult.ExitCode).To(Equal(0), "bad calico image check command should exit 0; stdout=%s stderr=%s", checkResult.Stdout, checkResult.Stderr)
	Expect(checkResult.Stdout).To(ContainSubstring(`create "calico" failed`), "BKECluster status should record the Calico addon creation failure")
	Expect(checkResult.Stdout).To(ContainSubstring("status: Failed"), "BKECluster status should mark the addon deployment phase as failed")
}
func deleteTestNode5ClusterFromGuideNode(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	command := `set -e
if kubectl get bc bke-cluster -n bke-cluster >/dev/null 2>&1; then
  kubectl annotate bc bke-cluster -n bke-cluster bke.bocloud.com/ignore-target-cluster-delete="false" --overwrite
  kubectl annotate bc bke-cluster -n bke-cluster bke.bocloud.com/ignore-namespace-delete="false" --overwrite
  kubectl patch bc bke-cluster -n bke-cluster --type=merge -p '{"spec":{"reset":true}}'
  kubectl delete bc bke-cluster -n bke-cluster --ignore-not-found=true
else
  echo "TEST_NODE5 cluster BC bke-cluster/bke-cluster not found, skip delete"
fi`
	result, err := utils.ExecuteCommandOnNode(guideNode, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize calico cleanup] node=%s command=%s exit=%d\n", guideNode.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "delete TEST_NODE5 cluster should run on GUIDE_NODE (%s)", guideNode.IP)
	Expect(result).NotTo(BeNil(), "delete TEST_NODE5 cluster command should return a result")
	Expect(result.ExitCode).To(Equal(0), "delete TEST_NODE5 cluster command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}
func deleteBKECluster3FromGuideNode(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	command := `set -e
if kubectl get bc bke-cluster3 -n bke-cluster3 >/dev/null 2>&1; then
  kubectl annotate bc bke-cluster3 -n bke-cluster3 bke.bocloud.com/ignore-target-cluster-delete="false" --overwrite
  kubectl annotate bc bke-cluster3 -n bke-cluster3 bke.bocloud.com/ignore-namespace-delete="false" --overwrite
  kubectl patch bc bke-cluster3 -n bke-cluster3 --type=merge -p '{"spec":{"reset":true}}'
  kubectl delete bc bke-cluster3 -n bke-cluster3 --ignore-not-found=true
else
  echo "TEST_NODE5 cluster BC bke-cluster3/bke-cluster3 not found, skip delete"
fi`
	result, err := utils.ExecuteCommandOnNode(guideNode, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize calico cleanup] node=%s command=%s exit=%d\n", guideNode.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "delete bke-cluster3 should run on GUIDE_NODE (%s)", guideNode.IP)
	Expect(result).NotTo(BeNil(), "delete bke-cluster3 command should return a result")
	Expect(result.ExitCode).To(Equal(0), "delete bke-cluster3 command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}
func createClusterWithEmptyCalicoVersionFromGuideNode(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	testNode := loadPerformanceNodeFromEnv("TEST_NODE5", "calico-empty")
	command := fmt.Sprintf(`set -e
cp /bke/cluster/1master* /opt
cd /opt
cleanup() {
  kubectl delete bn -n bke-cluster3 bke-cluster3-m1 --ignore-not-found=true
  kubectl delete bc -n bke-cluster3 bke-cluster3 --ignore-not-found=true
}
trap cleanup EXIT
NODE_IP=%s NODE_PASSWORD=%s perl -0pi -e 's/^(\s*ip:\s*).*?$/${1}$ENV{NODE_IP}/mg; s/^(\s*password:\s*).*?$/${1}$ENV{NODE_PASSWORD}/mg; s/(cluster\.x-k8s\.io\/cluster-name:\s*)bke-cluster\b/${1}bke-cluster3/g; s/(\bnamespace:\s*)bke-cluster\b/${1}bke-cluster3/g' 1master-node.yaml
perl -0pi -e 's/^(\s*name:\s*)bke-cluster$/${1}bke-cluster3/m; s/^(\s*namespace:\s*)bke-cluster$/${1}bke-cluster3/m; s/(\n(\s*)-\s+(?:(?!\n\2-\s+).)*?\n\2\s+name:\s*calico\b(?:(?!\n\2-\s+).)*?\n\2\s+version:\s*)[^\n]*/${1}/s' 1master-cluster.yaml
timeout 5m sh -c 'bke cluster create -f1master-cluster.yaml -n 1master-node.yaml 2>&1 | grep -m1 -E "calico addon version is empty"'
cleanup`, shellQuote(testNode.IP), shellQuote(testNode.Password))
	result, err := utils.ExecuteCommandOnNode(guideNode, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize calico] node=%s command=%s exit=%d\n", guideNode.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "empty calico version create command should succeed on GUIDE_NODE (%s)", guideNode.IP)
	Expect(result).NotTo(BeNil(), "empty calico version create command should return a result")
	Expect(result.ExitCode).To(Equal(0), "empty calico version create command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
	Expect(result.Stdout).To(ContainSubstring("calico addon version is empty"))
}

func createClusterWithDuplicateCalicoAddonFromGuideNode(ctx context.Context) {
	_ = ctx
	guideNode := guideNodeInfoFromEnv()
	testNode := loadPerformanceNodeFromEnv("TEST_NODE5", "calico-duplicate")
	command := fmt.Sprintf(`set -e
cp /bke/cluster/1master* /opt
cd /opt
cleanup() {
  kubectl delete bn -n bke-cluster bke-cluster-m1 --ignore-not-found=true
  kubectl delete bc -n bke-cluster bke-cluster --ignore-not-found=true
}
trap cleanup EXIT
NODE_IP=%s NODE_PASSWORD=%s perl -0pi -e 's/^(\s*ip:\s*).*?$/${1}$ENV{NODE_IP}/mg; s/^(\s*password:\s*).*?$/${1}$ENV{NODE_PASSWORD}/mg' 1master-node.yaml
perl -0pi -e 's/^([ \t]*)-\s+(?:(?!^\1-\s+).)*?name:\s*calico\b(?:(?!^\1-\s+).)*?(?=^\1-\s+|\z)/${1}- block: true\n${1}  name: calico\n${1}  param:\n${1}    allowTypha: "false"\n${1}    calicoMode: vxlan\n${1}    ipAutoDetectionMethod: skip-interface=nerdctl*\n${1}    typhaReplicas: "1"\n${1}  version: v3.31.3\n${1}- block: true\n${1}  name: calico\n${1}  param:\n${1}    allowTypha: "false"\n${1}    calicoMode: vxlan\n${1}    ipAutoDetectionMethod: skip-interface=nerdctl*\n${1}    typhaReplicas: "1"\n${1}  version: v3.31.3\n/ms' 1master-cluster.yaml
test "$(grep -c 'name: calico' 1master-cluster.yaml)" -eq 2
PATTERN='calico'
set +e
bke cluster create -f1master-cluster.yaml -n 1master-node.yaml
CREATE_STATUS=$?
set -e
for i in $(seq 1 60); do
  if grep -F "$PATTERN" /var/log/openFuyao/bke.log | grep -m1 -F "must be unique"; then
    break
  fi
  sleep 5
done
if ! grep -F "$PATTERN" /var/log/openFuyao/bke.log | grep -m1 -F "must be unique"; then
  tail -n 200 /var/log/openFuyao/bke.log | grep -E 'calico|addon' || true
  exit 1
fi
cleanup`, shellQuote(testNode.IP), shellQuote(testNode.Password))
	result, err := utils.ExecuteCommandOnNode(guideNode, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize calico] node=%s command=%s exit=%d\n", guideNode.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "duplicate calico addon create command should succeed on GUIDE_NODE (%s)", guideNode.IP)
	Expect(result).NotTo(BeNil(), "duplicate calico addon create command should return a result")
	Expect(result.ExitCode).To(Equal(0), "duplicate calico addon create command should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
	Expect(result.Stdout).To(And(ContainSubstring("calico"), ContainSubstring("must be unique")))
}
func shellQuote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "'\\''") + "'"
}

func shellJoin(values ...string) string {
	quoted := make([]string, 0, len(values))
	for _, value := range values {
		quoted = append(quoted, shellQuote(value))
	}
	return strings.Join(quoted, " ")
}
func assertGuideNodeOauthWebhookBeforeK3sRestart(ctx context.Context) {
	_ = ctx
	node := guideNodeInfoFromEnv()
	command := `grep -E 'DeployOauthWebhook success|K3sRestart success' /var/log/openFuyao/bke.log`
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize guide-log] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "guide node bke.log grep should succeed on GUIDE_NODE (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "guide node bke.log grep should return a result")
	Expect(result.ExitCode).To(Equal(0), "guide node bke.log grep should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)

	webhookLine, webhookTime := latestLogTimeForMessage(result.Stdout, "DeployOauthWebhook success")
	k3sLine, k3sTime := latestLogTimeForMessage(result.Stdout, "K3sRestart success")
	GinkgoWriter.Printf("[performance-optimize guide-log] DeployOauthWebhook time=%s line=%s\n", webhookTime.Format(time.RFC3339Nano), webhookLine)
	GinkgoWriter.Printf("[performance-optimize guide-log] K3sRestart time=%s line=%s\n", k3sTime.Format(time.RFC3339Nano), k3sLine)
	Expect(webhookTime.Before(k3sTime)).To(BeTrue(), "DeployOauthWebhook success time should be before K3sRestart success time; webhook=%s k3s=%s", webhookTime.Format(time.RFC3339Nano), k3sTime.Format(time.RFC3339Nano))
}

func guideNodeInfoFromEnv() config.NodeInfo {
	guideConfig := config.LoadGuideNodeFromEnv()
	Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST must be set")
	Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD must be set")
	return config.NodeInfo{
		Hostname: "guide-node",
		IP:       guideConfig.Host,
		Port:     strconv.Itoa(guideConfig.Port),
		Username: guideConfig.Username,
		Password: guideConfig.Password,
	}
}

func latestLogTimeForMessage(logs, message string) (string, time.Time) {
	var latestLine string
	for _, line := range strings.Split(logs, "\n") {
		if strings.Contains(line, message) {
			latestLine = strings.TrimSpace(line)
		}
	}
	Expect(latestLine).NotTo(BeEmpty(), "bke.log should contain %q", message)
	return latestLine, parseBKELogTime(latestLine)
}

func parseBKELogTime(line string) time.Time {
	timePattern := regexp.MustCompile(`"time"\s*:\s*"([^"]+)"`)
	matches := timePattern.FindStringSubmatch(line)
	Expect(matches).To(HaveLen(2), "bke.log line should contain a quoted time field: %s", line)

	value := matches[1]
	layouts := []string{
		time.RFC3339Nano,
		time.RFC3339,
		"2006-01-02T15:04:05.999999999Z07:00",
		"2006-01-02 15:04:05.999999999 -0700 MST",
		"2006-01-02 15:04:05.999999999",
		"2006-01-02 15:04:05.999",
		"2006-01-02 15:04:05",
	}
	for _, layout := range layouts {
		parsed, err := time.Parse(layout, value)
		if err == nil {
			return parsed
		}
	}
	Fail(fmt.Sprintf("unsupported bke.log time format %q in line: %s", value, line))
	return time.Time{}
}

func getConfigMapData(ctx context.Context, kubeContext, namespace, name, key string) string {
	return runDefaultKubectl(ctx, "-n", namespace, "get", "cm", name, "-o", fmt.Sprintf("jsonpath={.data.%s}", jsonPathMapKey(key)))
}

func jsonPathMapKey(key string) string {
	return strings.ReplaceAll(key, ".", `\.`)
}
func controllerLogs(ctx context.Context) string {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	command := "kubectl -n cluster-system logs deploy/bke-controller-manager --tail=300"
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize logs] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "controller log command should succeed on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "controller log command should return a result")
	Expect(result.ExitCode).To(Equal(0), "controller log command should exit 0; stderr=%s", result.Stderr)
	return result.Stdout
}

func waitForPodsReady(ctx context.Context, kubeContext, namespace, selector string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		output := runKubectl(ctx, kubeContext, "get", "pods", "-n", namespace, "-l", selector, "-o", "jsonpath={range .items[*]}{.metadata.name} {.status.phase} {.status.containerStatuses[*].ready}{\"\\n\"}{end}")
		assertPodsReadyOutput(g, output, namespace, selector)
	}, timeout, 15*time.Second).Should(Succeed())
}

func waitForPodsReadyOnNode3(ctx context.Context, namespace, selector string, timeout time.Duration) {
	Eventually(func(g Gomega) {
		output := runKubectlOnNode3(ctx, "get", "pods", "-n", namespace, "-l", selector, "-o", "jsonpath={range .items[*]}{.metadata.name} {.status.phase} {.status.containerStatuses[*].ready}{\"\\n\"}{end}")
		assertPodsReadyOutput(g, output, namespace, selector)
	}, timeout, 15*time.Second).Should(Succeed())
}

func assertPodsReadyOutput(g Gomega, output, namespace, selector string) {
	trimmed := strings.TrimSpace(output)
	g.Expect(trimmed).NotTo(BeEmpty(), "selector %q should match at least one pod in namespace %q", selector, namespace)
	lines := strings.Split(trimmed, "\n")
	for _, line := range lines {
		fields := strings.Fields(line)
		g.Expect(len(fields)).To(BeNumerically(">=", 3), "pod status line should include name, phase and ready state: %q", line)
		g.Expect(fields[1]).To(Equal("Running"), "pod %s should be Running", fields[0])
		for _, ready := range fields[2:] {
			g.Expect(ready).To(Equal("true"), "all containers in pod %s should be ready", fields[0])
		}
	}
}

func runKubectlOnNode3(ctx context.Context, args ...string) string {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := "kubectl " + shellJoin(args...)
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize kubectl] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "kubectl on TEST_NODE3 should succeed: %s", command)
	Expect(result).NotTo(BeNil(), "kubectl on TEST_NODE3 should return a result")
	Expect(result.ExitCode).To(Equal(0), "kubectl on TEST_NODE3 should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
	return result.Stdout
}

func applyExpectingFailure(ctx context.Context, manifestPath string) string {
	output, err := kubectl(ctx, managementContext(), "apply", "-f", manifestPath)
	Expect(err).To(HaveOccurred(), "invalid manifest %s should be rejected", manifestPath)
	return output
}

func applyHealthCheckConfigFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	encodedManifest := base64.StdEncoding.EncodeToString([]byte(healthCheckConfigManifest))
	command := fmt.Sprintf("printf '%s' | base64 -d | kubectl apply -f -", encodedManifest)
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "apply health-check-config should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "apply health-check-config should return a result")
	Expect(result.ExitCode).To(Equal(0), "apply health-check-config should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func waitForAutomaticHealthCheck(ctx context.Context, duration time.Duration) {
	select {
	case <-ctx.Done():
		Fail(fmt.Sprintf("context done while waiting for automatic health check: %v", ctx.Err()))
	case <-time.After(duration):
	}
}

func deleteHealthCheckConfigFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := "kubectl delete configmap health-check-config -n bke-config --ignore-not-found=true"
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "delete health-check-config should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "delete health-check-config should return a result")
	Expect(result.ExitCode).To(Equal(0), "delete health-check-config should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func applyCriticalHealthCheckFaultFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	encodedConfig := base64.StdEncoding.EncodeToString([]byte(healthCheckBackupConfigMapManifest))
	command := fmt.Sprintf("printf '%s' | base64 -d > health-check-config-backup.yaml && cp health-check-config-backup.yaml critical.yaml && sed -i '/- etcd-/s/- etcd-/- etcd-not-exist/' critical.yaml && kubectl apply -f critical.yaml", encodedConfig)
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "apply critical health-check fault should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "apply critical health-check fault should return a result")
	Expect(result.ExitCode).To(Equal(0), "apply critical health-check fault should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func restoreHealthCheckConfigFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := "kubectl apply -f health-check-config-backup.yaml"
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "restore health-check-config should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "restore health-check-config should return a result")
	Expect(result.ExitCode).To(Equal(0), "restore health-check-config should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func applyImportantHealthCheckFaultFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := `sed -i '0,/priority: critical/s//priority: important/' critical.yaml && kubectl apply -f critical.yaml`
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "apply important health-check fault should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "apply important health-check fault should return a result")
	Expect(result.ExitCode).To(Equal(0), "apply important health-check fault should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func applyOptionalHealthCheckFaultFromNode3(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE3", "pw1")
	command := `sed -i '0,/priority: important/s//priority: optional/' critical.yaml && kubectl apply -f critical.yaml`
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize config] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "apply optional health-check fault should succeed on TEST_NODE3 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "apply optional health-check fault should return a result")
	Expect(result.ExitCode).To(Equal(0), "apply optional health-check fault should exit 0; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func expectCriticalHealthCheckLogOnNode2(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	command := `timeout 2m sh -c 'kubectl -n cluster-system logs deploy/bke-controller-manager -f | grep -m1 -E "critical component health check start, total=5"'`
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize logs] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "controller health-check log grep should succeed on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "controller health-check log grep should return a result")
	Expect(result.ExitCode).To(Equal(0), "expected controller log to contain critical component health check start, total=5; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func expectHealthCheckConfigSyncLogOnNode2(ctx context.Context) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	command := `timeout 2m sh -c 'kubectl -n cluster-system logs deploy/bke-controller-manager -f | grep -m1 -E "HealthCheckConfigSync"'`
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize logs] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "controller HealthCheckConfigSync log grep should succeed on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "controller HealthCheckConfigSync log grep should return a result")
	Expect(result.ExitCode).To(Equal(0), "expected controller log to contain HealthCheckConfigSync; stdout=%s stderr=%s", result.Stdout, result.Stderr)
}

func expectClusterHealthCheckRequeueAfterOnNode2(ctx context.Context, timeout time.Duration, since, requeueAfter string) {
	_ = ctx
	node := loadPerformanceNodeFromEnv("TEST_NODE2", "pm1")
	seconds := int(timeout.Seconds())
	pattern := fmt.Sprintf("cluster health check failed, requeue after %s", requeueAfter)
	command := fmt.Sprintf(`timeout %ds sh -c 'kubectl -n cluster-system logs deploy/bke-controller-manager -f --since=%s | grep -E "%s" | head -n 2'`, seconds, since, pattern)
	result, err := utils.ExecuteCommandOnNode(node, command)
	if result != nil {
		GinkgoWriter.Printf("[performance-optimize logs] node=%s command=%s exit=%d\n", node.IP, command, result.ExitCode)
	}
	Expect(err).NotTo(HaveOccurred(), "controller requeue log grep should run on TEST_NODE2 (%s)", node.IP)
	Expect(result).NotTo(BeNil(), "controller requeue log grep should return a result")
	lines := strings.FieldsFunc(strings.TrimSpace(result.Stdout), func(r rune) bool { return r == '\n' || r == '\r' })
	Expect(len(lines)).To(BeNumerically(">=", 2), "expected at least two %q log lines within %s; stdout=%s stderr=%s exit=%d", pattern, timeout, result.Stdout, result.Stderr, result.ExitCode)
}
func writeHealthCheckConfigManifest() string {
	file, err := os.CreateTemp("", "health-check-config-*.yaml")
	Expect(err).NotTo(HaveOccurred(), "should create health-check-config manifest")
	_, err = file.WriteString(healthCheckConfigManifest)
	Expect(err).NotTo(HaveOccurred(), "should write health-check-config manifest")
	Expect(file.Close()).To(Succeed(), "should close health-check-config manifest")
	return file.Name()
}
func runDefaultKubectl(ctx context.Context, args ...string) string {
	output, err := runCommand(ctx, "kubectl", args...)
	Expect(err).NotTo(HaveOccurred(), "kubectl %s should succeed\n%s", strings.Join(args, " "), output)
	return output
}
func runKubectl(ctx context.Context, kubeContext string, args ...string) string {
	output, err := kubectl(ctx, kubeContext, args...)
	Expect(err).NotTo(HaveOccurred(), "kubectl %s should succeed\n%s", strings.Join(args, " "), output)
	return output
}

func kubectl(ctx context.Context, kubeContext string, args ...string) (string, error) {
	cmdArgs := make([]string, 0, len(args)+2)
	switch kubeContext {
	case generatedManagementContext, generatedTargetContext, "":
		// Use kubectl's default kubeconfig.
	default:
		cmdArgs = append(cmdArgs, "--context", kubeContext)
	}
	cmdArgs = append(cmdArgs, args...)
	return runCommand(ctx, "kubectl", cmdArgs...)
}

func runRequiredShell(ctx context.Context, envName string) string {
	command := requiredEnv(envName)
	output, err := runShell(ctx, command)
	Expect(err).NotTo(HaveOccurred(), "%s should succeed\n%s", envName, output)
	return output
}

func runOptionalShell(ctx context.Context, envName string) string {
	command := strings.TrimSpace(os.Getenv(envName))
	if command == "" {
		Skip(fmt.Sprintf("%s is required for this environment-specific step", envName))
	}
	output, err := runShell(ctx, command)
	Expect(err).NotTo(HaveOccurred(), "%s should succeed\n%s", envName, output)
	return output
}

func runShell(ctx context.Context, command string) (string, error) {
	return runCommand(ctx, shellName(), shellArgs(command)...)
}

func runCommand(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	output, err := cmd.CombinedOutput()
	return string(output), err
}

func shellName() string {
	if os.Getenv("OS") == "Windows_NT" {
		return "powershell"
	}
	return "sh"
}

func shellArgs(command string) []string {
	if os.Getenv("OS") == "Windows_NT" {
		return []string{"-NoProfile", "-Command", command}
	}
	return []string{"-c", command}
}

func requiredEnv(name string) string {
	value := strings.TrimSpace(os.Getenv(name))
	if value == "" {
		Skip(fmt.Sprintf("%s is required for this environment-specific step", name))
	}
	return value
}

func envOrDefault(name, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(name)); value != "" {
		return value
	}
	return fallback
}
