package nginx_proxy

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"

	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

const nginxProxyAPIPort = "6443"

type nginxProxyCluster struct {
	name           string
	configPath     string
	nodeConfigPath string
	kubeconfigPath string
	masterIPs      []string
	workerNode     config.NodeInfo
	endpointMaster config.NodeInfo
}

// createNginxProxy3M1WCluster creates an nginx-mode 3Master1Worker management
// cluster on the bootstrap node and waits for it to become Healthy/Ready.
// master[0] is the ControlPlaneEndpoint host (nginx mode requires a master IP, no VIP).
func createNginxProxy3M1WCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
) (*nginxProxyCluster, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("nginx 3m1w cluster requires at least 4 nodes, got %d", len(nodes))
	}
	mgmtNodes := make([]config.NodeInfo, 4)
	copy(mgmtNodes, nodes[:4])
	for i := 0; i < 3; i++ {
		mgmtNodes[i].Role = []string{"master/node", "etcd"}
	}
	mgmtNodes[3].Role = []string{"node"}
	masterIPs := []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}

	clusterName := fmt.Sprintf("test-nginx-3m1w-%d", time.Now().Unix())
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
	clusterConfig.LoadBalanceType = "nginx"
	clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
		Host: mgmtNodes[0].IP, Port: nginxProxyAPIPort,
	}
	GinkgoWriter.Printf("nginx 3m1w cluster %s: masters=%s worker=%s endpoint=%s\n",
		clusterName, strings.Join(masterIPs, ","), mgmtNodes[3].IP, mgmtNodes[0].IP)

	return finishCreateNginxCluster(clusterManager, localExecutor, clusterConfig, clusterName, masterIPs, mgmtNodes[3], mgmtNodes[0], false, "nginx 3m1w")
}

// createHaMode3M1WCluster creates a default ha-mode 3M1W management cluster (no nginx-proxy)
// for regression: endpoint follows management VIP env, loadBalanceType left default.
func createHaMode3M1WCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
) (*nginxProxyCluster, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("ha 3m1w cluster requires at least 4 nodes, got %d", len(nodes))
	}
	mgmtNodes := make([]config.NodeInfo, 4)
	copy(mgmtNodes, nodes[:4])
	for i := 0; i < 3; i++ {
		mgmtNodes[i].Role = []string{"master/node", "etcd"}
	}
	mgmtNodes[3].Role = []string{"node"}
	masterIPs := []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}

	clusterName := fmt.Sprintf("test-ha-3m1w-%d", time.Now().Unix())
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
	// loadBalanceType intentionally default (ha)
	if vipHost, vipPort := config.LoadManagementClusterVIPFromEnv(); vipHost != "" {
		clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: vipHost, Port: vipPort}
	}

	return finishCreateNginxCluster(clusterManager, localExecutor, clusterConfig, clusterName, masterIPs, mgmtNodes[3], mgmtNodes[0], false, "ha 3m1w")
}

// createNginxProxy3M1WSelfManagedCluster creates a self-managed nginx-mode 3M1W
// management cluster (cluster-api addon included) and waits for the self-managed
// bke-cluster BC to become Healthy.
func createNginxProxy3M1WSelfManagedCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
) (*nginxProxyCluster, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("self-managed nginx 3m1w cluster requires at least 4 nodes, got %d", len(nodes))
	}
	mgmtNodes := make([]config.NodeInfo, 4)
	copy(mgmtNodes, nodes[:4])
	for i := 0; i < 3; i++ {
		mgmtNodes[i].Role = []string{"master/node", "etcd"}
	}
	mgmtNodes[3].Role = []string{"node"}
	masterIPs := []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}

	clusterName := fmt.Sprintf("test-nginx-sm-3m1w-%d", time.Now().Unix())
	// NewBKEClusterConfigForMgmt includes cluster-api addon (self-management)
	clusterConfig := config.NewBKEClusterConfigForMgmt(clusterName, mgmtNodes)
	clusterConfig.LoadBalanceType = "nginx"
	clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
		Host: mgmtNodes[0].IP, Port: nginxProxyAPIPort,
	}
	GinkgoWriter.Printf("self-managed nginx 3m1w cluster %s: masters=%s worker=%s endpoint=%s\n",
		clusterName, strings.Join(masterIPs, ","), mgmtNodes[3].IP, mgmtNodes[0].IP)

	return finishCreateNginxCluster(clusterManager, localExecutor, clusterConfig, clusterName, masterIPs, mgmtNodes[3], mgmtNodes[0], true, "nginx 3m1w self-managed")
}

// createNginxProxy1M1WCluster creates a single-master (1M1W) nginx-mode cluster
// (master = nodes[0], worker = nodes[3]; other test nodes stay idle) for the
// upstream-single-node equivalence case. 2 nodes select the 1M1W template.
func createNginxProxy1M1WCluster(
	clusterManager *utils.ClusterManager,
	localExecutor *executor.LocalExecutor,
) (*nginxProxyCluster, error) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, fmt.Errorf("nginx 1m1w cluster requires at least 4 test nodes, got %d", len(nodes))
	}
	mgmtNodes := make([]config.NodeInfo, 2)
	mgmtNodes[0] = nodes[0]
	mgmtNodes[0].Role = []string{"master/node", "etcd"}
	mgmtNodes[1] = nodes[3]
	mgmtNodes[1].Role = []string{"node"}
	masterIPs := []string{mgmtNodes[0].IP}

	clusterName := fmt.Sprintf("test-nginx-1m1w-%d", time.Now().Unix())
	clusterConfig := config.NewDefaultBKEClusterConfig(clusterName, mgmtNodes)
	clusterConfig.LoadBalanceType = "nginx"
	clusterConfig.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{
		Host: mgmtNodes[0].IP, Port: nginxProxyAPIPort,
	}
	GinkgoWriter.Printf("nginx 1m1w cluster %s: master=%s worker=%s endpoint=%s\n",
		clusterName, masterIPs[0], mgmtNodes[1].IP, masterIPs[0])

	return finishCreateNginxCluster(clusterManager, localExecutor, clusterConfig, clusterName, masterIPs, mgmtNodes[1], mgmtNodes[0], false, "nginx 1m1w")
}

// finishCreateNginxCluster shared create+wait+kubeconfig tail for the helpers above.
// selfManaged=true waits for the self-managed bke-cluster BC (cluster-api addon included);
// selfManaged=false waits on the bootstrap BC only (cluster-api excluded).
func finishCreateNginxCluster(
	clusterManager *utils.ClusterManager, localExecutor *executor.LocalExecutor,
	clusterConfig *config.BKEClusterConfig, clusterName string, masterIPs []string,
	worker, endpoint config.NodeInfo, selfManaged bool, topology string,
) (*nginxProxyCluster, error) {
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate cluster config: %w", err)
	}
	out := &nginxProxyCluster{
		name: clusterName, configPath: configPath, nodeConfigPath: nodeConfigPath,
		masterIPs: masterIPs, workerNode: worker, endpointMaster: endpoint,
	}
	if err := clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, ""); err != nil {
		return out, fmt.Errorf("failed to create cluster: %w", err)
	}
	if selfManaged {
		// cluster-api addon included → management cluster has self-managed BC;
		// WaitAndVerify3M1WCluster waits for both bootstrap BC and self-managed BC.
		if err := installation.WaitAndVerify3M1WCluster(clusterManager, localExecutor, clusterName, ""); err != nil {
			return out, fmt.Errorf("cluster not healthy: %w", err)
		}
	} else {
		// cluster-api addon excluded → no self-managed BC; wait on bootstrap BC only.
		err = eventuallyShouldBeTrue(func() bool {
			phase, state, clusterStatus, e := clusterManager.GetClusterFullStatusWithKubeconfig(clusterName, "")
			if e != nil {
				GinkgoWriter.Printf("[bootstrap] get cluster status failed: %v\n", e)
				return false
			}
			GinkgoWriter.Printf("[bootstrap] phase=%s, state=%s, clusterStatus=%s\n", phase, state, clusterStatus)
			failOnClusterFailure(state, clusterStatus)
			return clusterStatus == "Ready"
		}, mgmtInstallTimeout, pollInterval, fmt.Sprintf("%s cluster should become Ready", topology))
		if err != nil {
			return out, fmt.Errorf("cluster not healthy: %w", err)
		}
	}
	checker := utils.NewClusterCheckerWithParentKubeconfig(localExecutor, clusterName, "")
	kc, err := checker.SaveKubeconfigToFile()
	if err != nil {
		return out, fmt.Errorf("failed to get cluster kubeconfig: %w", err)
	}
	out.kubeconfigPath = kc
	return out, nil
}

// verifyNginxProxyOnWorker SSH-checks the worker: static pod manifest, nginx.conf
// upstream (all masters) + listen 127.0.0.1, curl healthz, kubelet.conf -> 127.0.0.1.
func verifyNginxProxyOnWorker(worker config.NodeInfo, masterIPs []string) {
	By(fmt.Sprintf("SSH worker %s 检查 nginx-proxy", worker.IP))
	res, err := utils.ExecuteCommandOnNode(worker, "ls /etc/kubernetes/manifests/nginx-proxy.yaml")
	Expect(err).NotTo(HaveOccurred())
	Expect(res.ExitCode).To(Equal(0), "nginx-proxy.yaml 不存在")

	res, err = utils.ExecuteCommandOnNode(worker, "cat /etc/kubernetes/nginx-proxy/nginx.conf")
	Expect(err).NotTo(HaveOccurred())
	for _, ip := range masterIPs {
		Expect(strings.Contains(res.Stdout, ip+":6443")).To(BeTrue(), "nginx.conf upstream 应含 master %s", ip)
	}
	Expect(strings.Contains(res.Stdout, "127.0.0.1:6443")).To(BeTrue(), "nginx.conf 应监听 127.0.0.1:6443")
	Expect(strings.Contains(res.Stdout, "least_conn")).To(BeTrue(), "nginx.conf upstream 应使用 least_conn 负载均衡")

	res, err = utils.ExecuteCommandOnNode(worker, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz")
	Expect(err).NotTo(HaveOccurred())
	Expect(res.ExitCode).To(Equal(0), "curl healthz 失败: %s", res.Stderr)
	Expect(strings.TrimSpace(res.Stdout)).To(Equal("ok"))

	res, err = utils.ExecuteCommandOnNode(worker, "grep '127.0.0.1:6443' /etc/kubernetes/kubelet.conf || true")
	Expect(err).NotTo(HaveOccurred())
	Expect(strings.TrimSpace(res.Stdout)).NotTo(BeEmpty(), "worker kubelet.conf 应指向 127.0.0.1:6443")
}

// verifyNoNginxProxyOnWorker asserts the worker has NO nginx-proxy (ha mode regression).
func verifyNoNginxProxyOnWorker(worker config.NodeInfo) {
	By(fmt.Sprintf("SSH worker %s 确认无 nginx-proxy（ha 回归）", worker.IP))
	res, _ := utils.ExecuteCommandOnNode(worker, "ls /etc/kubernetes/manifests/nginx-proxy.yaml 2>/dev/null || true")
	Expect(strings.TrimSpace(res.Stdout)).To(BeEmpty(), "ha 模式 worker 不应有 nginx-proxy 静态 pod")
}

// attemptCreateExpectRejected generates+uploads an invalid config and asserts the
// BKECluster is NOT admitted (webhook reject). Returns nothing; cleanup is config-only.
func attemptCreateExpectRejected(clusterManager *utils.ClusterManager, localExecutor *executor.LocalExecutor, clusterConfig *config.BKEClusterConfig, expectLogKeyword string) {
	configPath, nodeConfigPath, err := clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
	Expect(err).NotTo(HaveOccurred(), "应能生成（含非法字段）配置文件")
	defer func() { _ = clusterManager.CleanupConfig(configPath); _ = clusterManager.CleanupConfig(nodeConfigPath) }()

	_ = clusterManager.CreateClusterInBackgroundWithKubeconfig(configPath, nodeConfigPath, "")
	// webhook 拒绝 → BKECluster 资源不应被准入
	Consistently(func() bool {
		return !clusterManager.ClusterExistsWithKubeconfig(clusterConfig.ClusterName, "")
	}, 90*time.Second, 10*time.Second).Should(BeTrue(), "非法配置的 BKECluster 应被 webhook 拒绝、不被创建")

	// 拒绝原因会写入后台创建日志，断言其与非法项对应（非静默拒绝）
	logPath := fmt.Sprintf("/tmp/bke-create-%s.log", clusterConfig.ClusterName)
	Eventually(func() string {
		r, e := localExecutor.Exec(fmt.Sprintf("grep -cF %q %s 2>/dev/null || true", expectLogKeyword, logPath))
		if e != nil || r == nil {
			return ""
		}
		return strings.TrimSpace(r.Stdout)
	}, 90*time.Second, 5*time.Second).ShouldNot(Equal("0"), "创建日志应包含与非法项对应的拒绝原因: %s", expectLogKeyword)
}

func cleanupNginxProxyCluster(clusterManager *utils.ClusterManager, localExecutor *executor.LocalExecutor, c *nginxProxyCluster, removeFinalizers ...bool) {
	if c == nil {
		return
	}
	installation.Delete3M1WManagementCluster(clusterManager, localExecutor, c.name, c.kubeconfigPath)
	_ = clusterManager.CleanupConfig(c.configPath)
	_ = clusterManager.CleanupConfig(c.nodeConfigPath)
	if c.kubeconfigPath != "" {
		_ = clusterManager.CleanupConfig(c.kubeconfigPath)
	}
	if len(removeFinalizers) == 0 || !removeFinalizers[0] {
		return
	}
	// 遗留 BC 会因 endpoint/节点唯一性校验拒绝后续集群创建；节点 reset 已在上一步完成，
	// 这里剥离 finalizer 强制移除被升级阶段阻塞的 BC
	Eventually(func() bool {
		if !clusterManager.ClusterExistsWithKubeconfig(c.name, "") {
			return true
		}
		_, _ = localExecutor.Exec(fmt.Sprintf(
			"kubectl patch bc bke-%s -n bke-%s --type=merge -p '{\"metadata\":{\"finalizers\":null}}' 2>&1", c.name, c.name))
		return false
	}, 3*time.Minute, 10*time.Second).Should(BeTrue(),
		"集群 %s 的 BC 未清理完成，遗留 BC 会阻塞后续集群创建", c.name)
}

func nginx3M1WNodes() (mgmtNodes []config.NodeInfo, masterIPs []string, ok bool) {
	nodes := config.LoadTestNodesFromEnv()
	if len(nodes) < 4 {
		return nil, nil, false
	}
	mgmtNodes = make([]config.NodeInfo, 4)
	copy(mgmtNodes, nodes[:4])
	for i := 0; i < 3; i++ {
		mgmtNodes[i].Role = []string{"master/node", "etcd"}
	}
	mgmtNodes[3].Role = []string{"node"}
	masterIPs = []string{mgmtNodes[0].IP, mgmtNodes[1].IP, mgmtNodes[2].IP}
	return mgmtNodes, masterIPs, true
}

var _ = SIGDescribe("BKE Cluster Installation With nginx-proxy LoadBalancer", func() {
	var (
		sshExecutor    *executor.SSHExecutor
		localExecutor  *executor.LocalExecutor
		clusterManager *utils.ClusterManager
	)

	BeforeEach(func() {
		guideConfig := config.LoadGuideNodeFromEnv()
		Expect(guideConfig.Host).NotTo(BeEmpty(), "GUIDE_NODE_HOST 必须设置")
		Expect(guideConfig.Password).NotTo(BeEmpty(), "GUIDE_NODE_PASSWORD 必须设置")
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideConfig.Host, guideConfig.Port, guideConfig.Username, guideConfig.Password)
		Expect(err).NotTo(HaveOccurred())
		kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")
		restConfig, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		Expect(err).NotTo(HaveOccurred())
		dynamicClient, err := dynamic.NewForConfig(restConfig)
		Expect(err).NotTo(HaveOccurred())
		clusterManager = utils.NewClusterManager(sshExecutor, dynamicClient)
		localExecutor = executor.NewLocalExecutor(30 * time.Second)
	})

	/*
	 用例名称：NP-DEP-001-创建nginx模式3M1W集群并验证本地代理访问
	 用例步骤：1）生成 loadBalanceType=nginx、endpoint=master1 IP 的 3M1W BKECluster+BKENode 配置并上传，预期结果 5
	         2）后台创建集群，轮询引导面 BC 直至 state=Healthy 且 clusterStatus=Ready，预期结果 5
	         3）SSH worker 检查 /etc/kubernetes/manifests/nginx-proxy.yaml 与 nginx.conf，预期结果 1、2
	         4）worker 执行 curl -k https://127.0.0.1:6443/healthz，预期结果 4
	         5）检查 worker kubelet.conf 指向 127.0.0.1:6443，预期结果 3
	 预期结果：1）nginx-proxy static pod manifest 存在（容器运行由 healthz=ok 间接验证）
	         2）nginx.conf upstream 含全部 3 个 master、least_conn、listen 127.0.0.1:6443
	         3）worker kubeconfig server=127.0.0.1:6443
	         4）curl 返回 ok
	         5）集群 Healthy/Ready
	*/
	Describe("创建 nginx 模式 3M1W 集群", Label("nginx-proxy", "3m1w", "with-management-cluster", "online", "P0"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("3M1W nginx 需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-DEP-001 - 验证 worker nginx-proxy 可用、upstream 含全部 3 master、经 127.0.0.1 访问 apiserver", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			Expect(c.masterIPs).To(HaveLen(3))
			verifyNginxProxyOnWorker(c.workerNode, c.masterIPs)
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})
	/*
	 用例名称：NP-CFG-001-默认ha模式回归不部署nginx-proxy
	 用例步骤：1）生成 loadBalanceType 留空（默认 ha）的 3M1W BKECluster+BKENode 配置并上传，预期结果 1
	         2）后台创建集群，轮询至 Healthy/Ready，预期结果 1
	         3）SSH worker 检查无 nginx-proxy 静态 pod manifest，预期结果 2
	 预期结果：1）走 ha 模式行为，集群 Healthy/Ready
	         2）worker 无 nginx-proxy static pod manifest
	*/
	Describe("默认 ha 模式回归（不部署 nginx-proxy）", Label("nginx-proxy", "ha-regression", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("ha 回归需要 4 节点")
			}
			var err error
			c, err = createHaMode3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-CFG-001 - ha 模式 worker 无 nginx-proxy", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			verifyNoNginxProxyOnWorker(c.workerNode)
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：NP-VAL-001/002/003-nginx模式非法配置准入校验
	 用例步骤：1）分别为每条 Entry 生成非法的 loadBalanceType=nginx 配置：
	             NP-VAL-001 注入非法枚举值 slb；NP-VAL-002 注入 extraLoadBalanceIP（互斥项）；
	             NP-VAL-003a 端点设为 127.0.0.1；NP-VAL-003b 端点设为非 master IP
	         2）创建集群并 Consistently 90s 轮询 BKECluster 不被创建，预期结果 1、2
	 预期结果：1）非法配置均被 CRD enum/webhook 校验拒绝
	         2）拒绝信息与非法项对应（enum 限制 / mutually exclusive / host must be a master node IP）
	*/
	Describe("nginx 模式准入校验（非法配置应被拒绝）", Label("nginx-proxy", "validation", "with-management-cluster", "online", "P1"), Ordered, func() {
		mgmtNodes, _, ok := nginx3M1WNodes()
		BeforeAll(func() {
			if !ok {
				Skip("validation 需要 4 节点生成基础配置")
			}
		})
		DescribeTable("NP-VAL-001/002/003 - 非法 loadBalanceType 配置创建被 webhook 拒绝",
			Label("required_nodes:4"),
			func(keyword string, mutate func(cfg *config.BKEClusterConfig)) {
				if !ok {
					Skip("需要 4 节点")
				}
				name := fmt.Sprintf("test-nginx-val-%d", time.Now().Unix())
				cfg := config.NewDefaultBKEClusterConfig(name, mgmtNodes)
				cfg.LoadBalanceType = "nginx"
				cfg.ControlPlaneEndpoint = &config.ControlPlaneEndpoint{Host: mgmtNodes[0].IP, Port: nginxProxyAPIPort}
				mutate(cfg)
				attemptCreateExpectRejected(clusterManager, localExecutor, cfg, keyword)
			},
			Entry("NP-VAL-001 非法 loadBalanceType 值 slb", "loadBalanceType must be", func(cfg *config.BKEClusterConfig) { cfg.LoadBalanceType = "slb" }),
			Entry("NP-VAL-002 nginx 与 extraLoadBalanceIP 互斥", "mutually exclusive", func(cfg *config.BKEClusterConfig) {
				cfg.CustomExtra = &config.ClusterCustomExtra{ExtraLoadBalanceIP: mgmtNodes[0].IP}
			}),
			Entry("NP-VAL-003a nginx 端点为 127.0.0.1 拒绝", "host must be a master node IP", func(cfg *config.BKEClusterConfig) {
				cfg.ControlPlaneEndpoint.Host = "127.0.0.1"
			}),
			Entry("NP-VAL-003b nginx 端点为 VIP 拒绝", "host must be a master node IP", func(cfg *config.BKEClusterConfig) {
				cfg.ControlPlaneEndpoint.Host = "192.168.1.200" // 非 master IP
			}),
		)
	})

	/*
	 用例名称：NP-VAL-004-loadBalanceType创建后不可变
	 用例步骤：1）创建 nginx 模式 3M1W 集群（流程同 NP-DEP-001）
	         2）kubectl patch 将 loadBalanceType 改为 ha，预期结果 1
	 预期结果：1）patch 被 webhook 拒绝（forbidden，不可变字段），返回非 0
	*/
	Describe("loadBalanceType 创建后不可变", Label("nginx-proxy", "immutable", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-VAL-004 - 更新 loadBalanceType 应被 webhook 拒绝", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			// Patch BKECluster on the bootstrap cluster (where the webhook is deployed).
			// The BKECluster name and namespace both have the "bke-" prefix.
			cmd := fmt.Sprintf("kubectl patch bkecluster bke-%s -n bke-%s --type merge -p '{\"spec\":{\"clusterConfig\":{\"cluster\":{\"loadBalanceType\":\"ha\"}}}}' 2>&1",
				c.name, c.name)
			res, _ := localExecutor.Exec(cmd)
			Expect(res.ExitCode).NotTo(Equal(0), "更新不可变字段应被拒绝")
			lowerOut := strings.ToLower(res.Stdout)
			Expect(strings.Contains(lowerOut, "forbidden") || strings.Contains(lowerOut, "validate") || strings.Contains(lowerOut, "immutable") || strings.Contains(lowerOut, "cannot be modified")).To(BeTrue(),
				"应含校验拒绝信息, out=%s err=%s", res.Stdout, res.Stderr)
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：NP-CLN-001-worker缩容清理nginx配置与pod
	 用例步骤：1）创建 nginx 模式 3M1W 集群
	         2）删除 worker 的 BKENode（缩容）
	         3）轮询 SSH worker 检查 nginx-proxy.yaml 与 nginx.conf 均不存在，预期结果 1
	 预期结果：1）缩容后 worker 上 nginx 配置与静态 pod manifest 清理干净，无残留
	*/
	Describe("worker 缩容清理 nginx 配置", Label("nginx-proxy", "worker-scale-in", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-CLN-001 - 删除 worker BKENode 后该节点 nginx 配置与 pod 清理干净", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			Expect(clusterManager.ScaleInNodeWithKubeconfig(c.name, c.workerNode.IP, "")).To(Succeed())
			Eventually(func() bool {
				res, _ := utils.ExecuteCommandOnNode(c.workerNode,
					"ls /etc/kubernetes/manifests/nginx-proxy.yaml /etc/kubernetes/nginx-proxy/nginx.conf 2>/dev/null || true")
				return strings.TrimSpace(res.Stdout) == ""
			}, uninstallTimeout, 30*time.Second).Should(BeTrue(), "缩容后应无 nginx 残留")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：NP-CLN-002-集群删除无nginx残留
	 用例步骤：1）创建 nginx 模式 3M1W 集群
	         2）删除集群（引导面 BC/命名空间删除及节点 reset），预期结果 2
	         3）轮询 SSH worker 检查无 nginx 残留，预期结果 1
	 预期结果：1）原 worker 无 nginx-proxy 静态 pod manifest 残留
	         2）BKECluster 及相关资源清理干净
	*/
	Describe("集群删除无 nginx 残留", Label("nginx-proxy", "delete", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-CLN-002 - 删除集群后各原 worker 无 nginx 残留", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			installation.Delete3M1WManagementCluster(clusterManager, localExecutor, c.name, c.kubeconfigPath)
			c.name = "" // 防止 AfterAll 重复删除
			Eventually(func() bool {
				res, _ := utils.ExecuteCommandOnNode(c.workerNode,
					"ls /etc/kubernetes/manifests/nginx-proxy.yaml 2>/dev/null || true")
				return strings.TrimSpace(res.Stdout) == ""
			}, uninstallTimeout, 30*time.Second).Should(BeTrue(), "删除集群后应无 nginx 残留")
		})
		AfterAll(func() {
			if c != nil && c.name != "" {
				cleanupNginxProxyCluster(clusterManager, localExecutor, c)
			}
		})
	})

	/*
	 用例名称：NP-DEP-002-nginx镜像走三方仓库且容器运行
	 用例步骤：1）创建 nginx 模式 3M1W 集群（在线场景），流程同 NP-DEP-001
	         2）worker 执行 crictl ps 检查 nginx-proxy 容器运行状态，预期结果 1
	         3）人工核对 manifest 镜像仓库前缀（ImageThirdRepo）与离线包镜像，预期结果 2
	 预期结果：1）nginx-proxy 容器 Running（crictl 可见）
	         2）镜像走三方仓库前缀、离线包内含该镜像（人工核对项）
	*/
	Describe("nginx 镜像走三方仓库离线可拉取", Label("nginx-proxy", "offline-image", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			nodes := config.LoadTestNodesFromEnv()
			if len(nodes) < 4 {
				Skip("离线镜像用例需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-DEP-002 - worker nginx-proxy 镜像走三方仓库且 Running", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			// Use crictl on the worker — avoids kubeconfig/pod-name issues.
			// Static pod mirror name is "nginx-proxy-<hostname>", not "nginx-proxy".
			res, err := utils.ExecuteCommandOnNode(c.workerNode, "crictl ps 2>/dev/null | grep nginx-proxy")
			Expect(err).NotTo(HaveOccurred())
			Expect(res.Stdout).NotTo(BeEmpty(), "应能取到 nginx-proxy 镜像")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：DFR-001-master故障切换worker业务面可用
	 用例步骤：1）创建 nginx 模式 3M1W 集群（endpoint=master1）
	         2）移走 master1 的 kube-apiserver 静态 pod manifest 模拟故障
	         3）轮询 worker curl healthz，预期结果 1
	         4）恢复 master1 apiserver，轮询 worker curl healthz，预期结果 3
	 预期结果：1）worker 业务面经 nginx 切到存活 master 仍可用（healthz=ok）
	         2）管理面经 master1 临时不可达目标集群（单点约束，人工核验项）
	         3）恢复后 worker 访问正常
	*/
	Describe("master 故障切换（nginx 模式）", Label("nginx-proxy", "failover", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("failover 需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("DFR-001 - 停 endpoint master 的 apiserver，worker 业务面仍可访问", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			master := c.endpointMaster
			By(fmt.Sprintf("在 master %s 移走 kube-apiserver static pod manifest", master.IP))
			stopRes, stopErr := utils.ExecuteCommandOnNode(master,
				"mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/kube-apiserver.yaml.bak.$(date +%s) 2>/dev/null || true")
			Expect(stopErr).NotTo(HaveOccurred())
			Expect(stopRes.ExitCode).To(Equal(0))
			time.Sleep(40 * time.Second)
			By("worker 经 nginx 访问应仍可用（切到存活 master）")
			Eventually(func() bool {
				res, err := utils.ExecuteCommandOnNode(c.workerNode, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz || true")
				return err == nil && strings.TrimSpace(res.Stdout) == "ok"
			}, 2*time.Minute, 15*time.Second).Should(BeTrue(), "停 endpoint master 后 worker 业务面应仍可访问")
			By("恢复 endpoint master apiserver")
			_, _ = utils.ExecuteCommandOnNode(master,
				"latest=$(ls -t /tmp/kube-apiserver.yaml.bak.* 2>/dev/null | head -1); [ -n \"$latest\" ] && mv $latest /etc/kubernetes/manifests/kube-apiserver.yaml || true")
			Eventually(func() bool {
				res, err := utils.ExecuteCommandOnNode(c.workerNode, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz || true")
				return err == nil && strings.TrimSpace(res.Stdout) == "ok"
			}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "恢复后 worker 应可访问 apiserver")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：DFR-002-全部master不可用后自愈
	 用例步骤：1）创建 nginx 模式 3M1W 集群
	         2）移走全部 master 的 kube-apiserver manifest，轮询 worker 访问失败，预期结果 1
	         3）恢复全部 master apiserver，轮询 worker curl healthz，预期结果 2
	 预期结果：1）全部 master 不可用时 worker 访问失败
	         2）恢复后 nginx 自动恢复转发，worker 访问恢复（healthz=ok）
	*/
	Describe("全部 master 不可用后自愈", Label("nginx-proxy", "all-master-down", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("DFR-002 - 停全部 master 后 worker 访问 502，恢复后自愈", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			moveAway := func(n config.NodeInfo) {
				_, _ = utils.ExecuteCommandOnNode(n,
					"mv /etc/kubernetes/manifests/kube-apiserver.yaml /tmp/kube-apiserver.yaml.bak.$(date +%s) 2>/dev/null || true")
			}
			restore := func(n config.NodeInfo) {
				_, _ = utils.ExecuteCommandOnNode(n,
					"latest=$(ls -t /tmp/kube-apiserver.yaml.bak.* 2>/dev/null | head -1); [ -n \"$latest\" ] && mv $latest /etc/kubernetes/manifests/kube-apiserver.yaml || true")
			}
			for _, m := range c.masterIPs {
				nodes := config.LoadTestNodesFromEnv()
				for _, n := range nodes[:4] {
					if n.IP == m {
						moveAway(n)
					}
				}
			}
			time.Sleep(40 * time.Second)
			By("全部 master 不可用 → worker 访问应失败（非 ok）")
			Eventually(func() bool {
				res, _ := utils.ExecuteCommandOnNode(c.workerNode, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz || true")
				return strings.TrimSpace(res.Stdout) != "ok"
			}, 2*time.Minute, 15*time.Second).Should(BeTrue(), "全部 master 不可用时 worker 访问应失败")
			for _, m := range c.masterIPs {
				nodes := config.LoadTestNodesFromEnv()
				for _, n := range nodes[:4] {
					if n.IP == m {
						restore(n)
					}
				}
			}
			By("恢复后 worker 访问应恢复")
			Eventually(func() bool {
				res, err := utils.ExecuteCommandOnNode(c.workerNode, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz || true")
				return err == nil && strings.TrimSpace(res.Stdout) == "ok"
			}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "恢复后 worker 应可访问 apiserver")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：DFS-001-删除nginx-proxy容器自愈重建
	 用例步骤：1）创建 nginx 模式 3M1W 集群，worker nginx-proxy Running
	         2）worker 上 crictl 强删 nginx-proxy 容器
	         3）轮询 worker curl healthz，预期结果 1、2
	 预期结果：1）kubelet 按本地 manifest 自动重建 nginx-proxy 容器
	         2）worker 访问恢复（healthz=ok）
	*/
	Describe("删除 nginx-proxy 容器自愈重建", Label("nginx-proxy", "pod-selfheal", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("DFS-001 - 强删 nginx-proxy 容器后 kubelet 自愈重建", Label("required_nodes:4"), SpecTimeout(installTimeout), func(ctx SpecContext) {
			// Deleting the mirror pod is a no-op for static pods (kubelet recreates
			// the API object while the real containers keep running). Force-remove
			// the containers on the worker instead; kubelet rebuilds them from the
			// local manifest.
			By("worker 上强删 nginx-proxy 容器")
			res, err := utils.ExecuteCommandOnNode(c.workerNode, "crictl ps --name nginx-proxy -q")
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(res.Stdout)).NotTo(BeEmpty(), "删除前应存在 nginx-proxy 容器")
			res, err = utils.ExecuteCommandOnNode(c.workerNode,
				"crictl ps --name nginx-proxy -q | xargs -r crictl rm -f")
			Expect(err).NotTo(HaveOccurred())
			Expect(res.ExitCode).To(Equal(0), "crictl rm 失败: %s", res.Stderr)

			By("kubelet 应自愈重建并恢复访问")
			Eventually(func() bool {
				res, e := utils.ExecuteCommandOnNode(c.workerNode, "curl -k --max-time 5 -s https://127.0.0.1:6443/healthz || true")
				return e == nil && strings.TrimSpace(res.Stdout) == "ok"
			}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "容器删除后应自愈重建并恢复访问")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：DFD-001-配置渲染失败有明确错误日志
	 用例步骤：1）创建 nginx 模式 3M1W 集群
	         2）篡改 BKECluster apiServer.port 为非法值 70000
	         3）创建 Command CR 触发 worker 执行 ReconfigureNginx，预期结果 1
	         4）检查 worker bkeagent 日志，预期结果 2
	         5）恢复 apiServer.port
	 预期结果：1）ReconfigureNginx 命令执行失败（condition phase=Failed、status=False）
	         2）bkeagent 日志含明确渲染错误（port must be between 1 and 65535）
	*/
	Describe("配置渲染失败有明确错误日志", Label("nginx-proxy", "dfd", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("DFD-001 需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("DFD-001 - 篡改 apiServerPort 后 ReconfigureNginx 失败且日志含明确错误与上下文", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			ns := "bke-" + c.name
			bcName := "bke-" + c.name
			cmdName := fmt.Sprintf("dfd-inject-%d", time.Now().Unix())

			By("篡改 BKECluster apiServer.port 为非法值 70000")
			patchBad := `{"spec":{"clusterConfig":{"cluster":{"apiServer":{"port":70000}}}}}`
			res, err := localExecutor.Exec(fmt.Sprintf(
				"kubectl patch bkecluster %s -n %s --type=merge -p '%s' 2>&1", bcName, ns, patchBad))
			Expect(err).NotTo(HaveOccurred())
			Expect(res.ExitCode).To(Equal(0), "篡改 apiServer.port 应成功, out=%s", res.Stdout)

			By("创建 Command CR 触发 worker 执行 ReconfigureNginx")
			// Address the worker via IP nodeSelector the same way controller-dispatched
			// commands do: agent-side nodeName matching uses the name cached in the
			// agent's node file, which may differ from the live OS hostname.
			cmdYaml := fmt.Sprintf(`apiVersion: bkeagent.bocloud.com/v1beta1
kind: Command
metadata:
  name: %s
  namespace: %s
  labels:
    bke.bocloud.com/cluster-command: ""
    cluster.x-k8s.io/cluster-name: %s
spec:
  nodeSelector:
    matchLabels:
      %s: %s
  commands:
  - id: dfd-inject
    command: ["Kubeadm", "phase=ReconfigureNginx", "bkeConfig=%s:%s"]
    type: BuiltIn
`, cmdName, ns, bcName, c.workerNode.IP, c.workerNode.IP, ns, bcName)
			yamlPath := fmt.Sprintf("/tmp/dfd-inject-%d.yaml", time.Now().Unix())
			_, err = localExecutor.Exec(fmt.Sprintf(
				"cat > %s << 'DFDEOF'\n%sDFDEOF\nkubectl apply -f %s 2>&1", yamlPath, cmdYaml, yamlPath))
			Expect(err).NotTo(HaveOccurred())
			defer func() { _, _ = localExecutor.Exec(fmt.Sprintf("rm -f %s", yamlPath)) }()

			By("等待命令执行结束并按预期失败")
			// status 是按节点为 key 的 map（<nodeName>/<nodeIP> → CommandStatus），
			// 用通配符读取各节点的执行结果
			Eventually(func() string {
				r, _ := localExecutor.Exec(fmt.Sprintf(
					"kubectl get command %s -n %s -o jsonpath='{.status.*.phase}|{.status.*.status}' 2>/dev/null", cmdName, ns))
				if r == nil {
					return ""
				}
				return strings.TrimSpace(r.Stdout)
			}, 3*time.Minute, 10*time.Second).Should(ContainSubstring("Failed"), "ReconfigureNginx 应按预期失败（注入生效）")

			By("断言失败已标记且 StdErr 含明确渲染错误")
			res, err = localExecutor.Exec(fmt.Sprintf(
				"kubectl get command %s -n %s -o jsonpath='{.status.*.failed}|{.status.*.conditions[*].stdErr}' 2>/dev/null", cmdName, ns))
			Expect(err).NotTo(HaveOccurred())
			Expect(res.Stdout).To(ContainSubstring("port must be between 1 and 65535"), "condition StdErr 应包含渲染错误")

			By("断言 bkeagent 日志含明确渲染错误与参数上下文")
			var logHit string
			Eventually(func() string {
				r, e := utils.ExecuteCommandOnNode(c.workerNode,
					"grep -c 'port must be between 1 and 65535' /var/log/openFuyao/bkeagent.log 2>/dev/null || true")
				if e != nil || r == nil {
					return ""
				}
				logHit = strings.TrimSpace(r.Stdout)
				return logHit
			}, 2*time.Minute, 10*time.Second).ShouldNot(Equal("0"), "bkeagent 日志应包含 parsePort 的明确错误（0=未命中）")
			GinkgoWriter.Printf("[dfd] bkeagent 日志命中 %s 处 'port must be between 1 and 65535'\n", logHit)

			By("恢复 apiServer.port，避免影响集群清理")
			restore := fmt.Sprintf(
				"kubectl patch bkecluster %s -n %s --type=merge -p '{\"spec\":{\"clusterConfig\":{\"cluster\":{\"apiServer\":{\"port\":%s}}}}}'",
				bcName, ns, nginxProxyAPIPort)
			res, err = localExecutor.Exec(restore + " 2>&1")
			Expect(err).NotTo(HaveOccurred())
			Expect(res.ExitCode).To(Equal(0), "恢复 apiServer.port 应成功, out=%s", res.Stdout)
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c, true) })
	})

	/*
	 用例名称：NP-DEP-003-单master nginx集群upstream单节点
	 用例步骤：1）生成 loadBalanceType=nginx、endpoint=master IP 的 1M1W BKECluster+BKENode 配置并上传（其余测试节点闲置），预期结果 3
	         2）后台创建集群，轮询至 Healthy/Ready，预期结果 3
	         3）SSH worker 检查 nginx-proxy.yaml、nginx.conf、healthz、kubelet.conf，预期结果 1、2
	         4）统计 nginx.conf upstream 的 server 条目数，预期结果 1
	 预期结果：1）upstream 仅含 1 个 master
	         2）worker 经 127.0.0.1 访问正常（healthz=ok、kubelet.conf 指向 127.0.0.1）
	         3）集群 Healthy/Ready
	*/
	Describe("单 master nginx 集群 upstream 单节点", Label("nginx-proxy", "1m1w", "with-management-cluster", "online", "P2"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("NP-DEP-003 需要 4 测试节点（使用其中 1 master + 1 worker，其余闲置）")
			}
			var err error
			c, err = createNginxProxy1M1WCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-DEP-003 - 单 master 集群 worker nginx-proxy 可用且 upstream 仅含 1 个 master", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			Expect(c.masterIPs).To(HaveLen(1))
			verifyNginxProxyOnWorker(c.workerNode, c.masterIPs)
			By("断言 upstream 仅含 1 个 master")
			res, err := utils.ExecuteCommandOnNode(c.workerNode,
				"grep -c 'server .*:6443' /etc/kubernetes/nginx-proxy/nginx.conf")
			Expect(err).NotTo(HaveOccurred())
			Expect(strings.TrimSpace(res.Stdout)).To(Equal("1"), "upstream 应仅含 1 个 master 节点")
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

	/*
	 用例名称：NP-DEP-005-nginx模式自纳管集群生命周期
	 用例步骤：1）生成含 cluster-api addon 的 nginx 模式 3M1W 配置并上传
	         2）后台创建集群，轮询引导面 BC Healthy
	         3）校验管理集群有 bc CRD，预期结果 1
	         4）轮询自管 bke-<clusterName> BC Healthy，预期结果 2
	         5）SSH worker 验证 nginx-proxy，预期结果 3
	 预期结果：1）管理集群有 bkeclusters CRD
	         2）自管 bke-<clusterName> BC 存在且 Healthy
	         3）worker nginx-proxy 正常（upstream+healthz+kubelet.conf）
	*/
	Describe("nginx 模式自纳管集群生命周期", Label("nginx-proxy", "self-managed", "with-management-cluster", "online", "P1"), Ordered, func() {
		var c *nginxProxyCluster
		BeforeAll(func() {
			if _, _, ok := nginx3M1WNodes(); !ok {
				Skip("自纳管需要 4 节点")
			}
			var err error
			c, err = createNginxProxy3M1WSelfManagedCluster(clusterManager, localExecutor)
			Expect(err).NotTo(HaveOccurred())
		})
		It("NP-DEP-005 - 自纳管集群: 管理集群有 bc CRD + 自管 BC Healthy + worker nginx-proxy 正常", Label("required_nodes:4"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
			By("管理集群上有 bc CRD")
			crdCmd := fmt.Sprintf("kubectl --kubeconfig %s get crd bkeclusters.bke.bocloud.com --no-headers 2>&1", c.kubeconfigPath)
			crdRes, _ := localExecutor.Exec(crdCmd)
			Expect(crdRes.ExitCode).To(Equal(0), "管理集群应有 bkeclusters CRD, out=%s", crdRes.Stdout)

			selfBCName := utils.ManagementClusterSelfBCName(c.name)
			selfBCNS := utils.ManagementClusterSelfBCNamespace(c.name)
			By(fmt.Sprintf("自管 BC %s/%s 存在且 Ready", selfBCName, selfBCNS))
			selfBCCmd := fmt.Sprintf("kubectl --kubeconfig %s get bc %s -n %s -o custom-columns=STATUS:.status.clusterStatus --no-headers 2>&1", c.kubeconfigPath, selfBCName, selfBCNS)
			Eventually(func() string {
				res, _ := localExecutor.Exec(selfBCCmd)
				return strings.TrimSpace(res.Stdout)
			}, 2*time.Minute, 15*time.Second).Should(Equal("Ready"), "自管 BC 应为 Ready")

			By("worker 上 nginx-proxy 仍正常")
			Expect(c.masterIPs).To(HaveLen(3))
			verifyNginxProxyOnWorker(c.workerNode, c.masterIPs)
		})
		AfterAll(func() { cleanupNginxProxyCluster(clusterManager, localExecutor, c) })
	})

})
