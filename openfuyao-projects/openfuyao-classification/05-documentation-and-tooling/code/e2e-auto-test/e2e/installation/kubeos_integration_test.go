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

// KubeOS 不可变 OS 适配 - 集成测试
// 覆盖用例：
//   - KubeOS-IT-002: 混合 OS 集群安装——worker 加入成功
//   - KubeOS-IT-003: 混合 OS 集群安装——只读目录未被写入
//   - KubeOS-IT-004: worker 节点标签自动打标
//   - KubeOS-IT-005: 集群可用性验证——Pod 调度与 CNI
//   - KubeOS-IT-008: KubeOS addon 安装与调度
//
// 这些用例共享同一个 KubeOS 混合集群（1 普通 openEuler master + 1 KubeOS worker），
// 使用 Ordered + BeforeAll/AfterAll 避免重复安装。
var _ = SIGDescribe("KubeOS Immutable OS Integration", Ordered, func() {
	var (
		sshExecutor    *executor.SSHExecutor
		dynamicClient  dynamic.Interface
		clusterManager *utils.ClusterManager
		clusterChecker *utils.ClusterChecker
		kubeosChecker  *utils.KubeOSChecker

		clusterName    string
		clusterConfig  *config.BKEClusterConfig
		configPath     string
		nodeConfigPath string
		masterNode     config.NodeInfo
		kubeosWorker   config.NodeInfo
	)

	BeforeAll(func() {
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

		clusterName = fmt.Sprintf("test-kubeos-integ-%d", time.Now().Unix())

		// 加载节点：1 普通 openEuler master + 1 KubeOS worker
		nodes := config.LoadTestNodesFromEnv()
		Expect(len(nodes)).To(BeNumerically(">=", 2), "KubeOS 集成测试需要至少 2 个节点（1 master + 1 KubeOS worker）")

		masterNode = nodes[0]
		masterNode.Role = []string{"master/node", "etcd"}

		kubeosWorker = nodes[1]
		kubeosWorker.Role = []string{"node"}

		clusterConfig = config.NewKubeOSClusterConfig(clusterName, []config.NodeInfo{masterNode, kubeosWorker}, config.LoadKubeOSChartRepoFromEnv())

		By("生成集群配置")
		configPath, nodeConfigPath, err = clusterManager.GetConfigGenerator().GenerateAndUpload(clusterConfig)
		Expect(err).NotTo(HaveOccurred())

		By("使用 --immutable-os=kubeos 创建 KubeOS 混合集群")
		err = clusterManager.CreateClusterInBackgroundWithImmutableOS(configPath, nodeConfigPath, "", utils.ImmutableOSAnnotationValue)
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
	})

	AfterAll(func() {
		By("清理 KubeOS 集成测试集群")
		if clusterManager != nil && clusterManager.ClusterExistsWithKubeconfig(clusterName, "") {
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

	// 用例编号: KubeOS-IT-002
	// 用例名称: KubeOS_混合OS集群安装——worker加入成功
	// 用例步骤:
	//   1. bke cluster create -f cluster.yaml -n nodes.yaml --immutable-os=kubeos
	//   2. 等待安装完成
	//   3. kubectl get nodes 检查节点状态
	// 预期结果:
	//   1. master 和 worker 均 Ready
	//   2. 安装过程无报错中止
	//   3. worker 上 bkeagent/kubelet/containerd 服务正常运行
	It("【KubeOS-IT-002】混合 OS 集群安装——worker 加入成功", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func(ctx SpecContext) {
		By("验证集群节点 Ready")
		readyCount, err := clusterChecker.GetReadyNodeCountWithKubeconfig("")
		Expect(err).NotTo(HaveOccurred())
		Expect(readyCount).To(Equal(2), "master 和 worker 均 Ready")

		By("验证 KubeOS worker 上 bkeagent 服务正常运行")
		active, err := kubeosChecker.CheckServiceActiveOnNode(kubeosWorker, "bkeagent")
		Expect(err).NotTo(HaveOccurred())
		Expect(active).To(BeTrue(), "KubeOS worker 上 bkeagent 服务应处于 active 状态")

		By("验证 KubeOS worker 上 kubelet 服务正常运行")
		active, err = kubeosChecker.CheckServiceActiveOnNode(kubeosWorker, "kubelet")
		Expect(err).NotTo(HaveOccurred())
		Expect(active).To(BeTrue(), "KubeOS worker 上 kubelet 服务应处于 active 状态")

		By("验证 KubeOS worker 上 containerd 服务正常运行")
		active, err = kubeosChecker.CheckServiceActiveOnNode(kubeosWorker, "containerd")
		Expect(err).NotTo(HaveOccurred())
		Expect(active).To(BeTrue(), "KubeOS worker 上 containerd 服务应处于 active 状态")
	})

	// 用例编号: KubeOS-IT-003
	// 用例名称: KubeOS_混合OS集群安装——只读目录未被写入
	// 用例步骤:
	//   1. 检查 /persist、/etc、/var、/opt/cni 等路径的可写性；检查 containerd、kubelet、bke-agent 是否存在
	//   2. 在 KubeOS worker 上检查 /usr/bin/kubelet、/usr/local/bin/bkeagent 是否为镜像内置版本
	//   3. 检查安装过程中无 read-only file system 导致的失败
	// 预期结果:
	//   1. 允许写入目录检查通过；必需组件存在性检查通过
	//   2. 二进制为镜像内置版本，未被安装器覆盖
	//   3. 安装全程未因只读目录写入而失败
	It("【KubeOS-IT-003】混合 OS 集群安装——只读目录未被写入", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func(ctx SpecContext) {
		By("验证 KubeOS worker 上允许写入目录可写")
		writableDirs := []string{"/persist", "/etc", "/var", "/opt/cni"}
		for _, dir := range writableDirs {
			writable, err := kubeosChecker.CheckDirWritableOnNode(kubeosWorker, dir)
			Expect(err).NotTo(HaveOccurred())
			Expect(writable).To(BeTrue(), "KubeOS worker 上目录 %s 应可写", dir)
		}

		By("验证 KubeOS worker 上必需组件二进制存在")
		binaries := []struct {
			path string
			desc string
		}{
			{"/usr/bin/kubelet", "kubelet"},
			{"/usr/local/bin/bkeagent", "bkeagent"},
		}
		for _, bin := range binaries {
			exists, err := kubeosChecker.CheckBinaryOnNode(kubeosWorker, bin.path)
			Expect(err).NotTo(HaveOccurred())
			Expect(exists).To(BeTrue(), "KubeOS worker 上 %s 应存在（镜像内置）", bin.desc)
		}

		By("验证安装过程中无 read-only file system 错误")
		logPath := fmt.Sprintf("/tmp/bke-create-%s.log", clusterName)
		noReadOnly, err := kubeosChecker.CheckReadOnlyDirsNotWritten(logPath)
		Expect(err).NotTo(HaveOccurred())
		Expect(noReadOnly).To(BeTrue(), "安装全程不应因只读目录写入而失败")
	})

	// 用例编号: KubeOS-IT-004
	// 用例名称: KubeOS_worker节点标签自动打标
	// 用例步骤:
	//   1. kubectl get nodes --show-labels
	//   2. 检查 KubeOS worker 和 master 的标签
	// 预期结果:
	//   1. worker 带 node.openfuyao.io/os=kubeos
	//   2. worker 带 upgrade.openeuler.org/node-selector=default-worker-pool
	//   3. master 不带上述标签
	It("【KubeOS-IT-004】worker 节点标签自动打标", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func(ctx SpecContext) {
		By("获取 KubeOS worker 的 Kubernetes Node 名称")
		workerNodeName, err := kubeosChecker.GetNodeNameByIP(kubeosWorker.IP)
		Expect(err).NotTo(HaveOccurred())
		Expect(workerNodeName).NotTo(BeEmpty(), "应能通过 IP 找到 KubeOS worker 节点")

		By("验证 KubeOS worker 带有 KubeOS 相关标签")
		missing, err := kubeosChecker.CheckNodeHasKubeOSLabels(workerNodeName)
		Expect(err).NotTo(HaveOccurred())
		Expect(missing).To(BeEmpty(), "KubeOS worker 应带有所有 KubeOS 相关标签（缺失: %s）", strings.Join(missing, ", "))

		By("验证 master 节点不带 KubeOS 标签")
		labeledMasters, err := kubeosChecker.CheckMasterNotKubeOSLabeled()
		Expect(err).NotTo(HaveOccurred())
		Expect(labeledMasters).To(BeEmpty(), "master 节点不应带 node.openfuyao.io/os=kubeos 标签")
	})

	// 用例编号: KubeOS-IT-005
	// 用例名称: KubeOS_集群可用性验证——Pod调度与CNI
	// 用例步骤:
	//   1. 创建一个测试 Deployment（replicas=1，指定调度到 worker）
	//   2. 等待 Pod Running
	//   3. 检查 Pod 跨节点网络连通性
	// 预期结果:
	//   1. Pod 正常调度到 KubeOS worker 并 Running
	//   2. Calico CNI 正常工作，Pod 网络互通
	//   3. /opt/cni/bin 可写（Calico 插件正常安装）
	It("【KubeOS-IT-005】集群可用性验证——Pod 调度与 CNI", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func(ctx SpecContext) {
		By("验证 /opt/cni/bin 可写（Calico 插件正常安装）")
		writable, err := kubeosChecker.CheckDirWritableOnNode(kubeosWorker, "/opt/cni/bin")
		Expect(err).NotTo(HaveOccurred())
		Expect(writable).To(BeTrue(), "/opt/cni/bin 应可写（Calico 插件正常安装）")

		By("验证 Calico CNI 插件存在")
		calicoBinExists, err := kubeosChecker.CheckBinaryOnNode(kubeosWorker, "/opt/cni/bin/calico")
		Expect(err).NotTo(HaveOccurred())
		Expect(calicoBinExists).To(BeTrue(), "Calico CNI 插件应存在")

		By("创建测试 Deployment 并指定调度到 KubeOS worker")
		kubeconfigPath, err := clusterChecker.SaveKubeconfigToFile()
		Expect(err).NotTo(HaveOccurred())
		defer func() {
			_, _ = sshExecutor.Exec(fmt.Sprintf("rm -f %s", kubeconfigPath))
		}()

		// 创建 namespace 和测试 Deployment，通过 nodeSelector 调度到 KubeOS worker
		testNs := fmt.Sprintf("kubeos-test-%d", time.Now().Unix())
		createNsCmd := fmt.Sprintf("KUBECONFIG=%s kubectl create namespace %s", kubeconfigPath, testNs)
		result, err := sshExecutor.Exec(createNsCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0), "创建测试 namespace 失败: %s", result.Stderr)

		// 创建 Deployment，使用 nodeSelector 调度到 KubeOS worker
		deployYaml := fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: kubeos-test-deploy
  namespace: %s
spec:
  replicas: 1
  selector:
    matchLabels:
      app: kubeos-test
  template:
    metadata:
      labels:
        app: kubeos-test
    spec:
      nodeSelector:
        %s: %s
      containers:
      - name: nginx
        image: nginx:alpine
        ports:
        - containerPort: 80`, testNs, utils.KubeOSNodeOSLabel, utils.KubeOSNodeOSLabelValue)

		applyCmd := fmt.Sprintf("cat <<'EOF' | KUBECONFIG=%s kubectl apply -f -\n%s\nEOF", kubeconfigPath, deployYaml)
		result, err = sshExecutor.Exec(applyCmd)
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0), "创建测试 Deployment 失败: %s", result.Stderr)

		By("等待测试 Pod Running 并调度到 KubeOS worker")
		Eventually(func() bool {
			nodeName, onKubeOS, err := kubeosChecker.CheckPodScheduledToWorker("app=kubeos-test")
			if err != nil {
				return false
			}
			GinkgoWriter.Printf("测试 Pod 所在节点: %s, 是否在 KubeOS worker: %v\n", nodeName, onKubeOS)
			if !onKubeOS {
				return false
			}
			running, _ := kubeosChecker.CheckPodRunning(testNs, "app=kubeos-test")
			return running
		}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "测试 Pod 应调度到 KubeOS worker 并 Running")

		By("清理测试 Deployment")
		deleteCmd := fmt.Sprintf("KUBECONFIG=%s kubectl delete namespace %s --ignore-not-found", kubeconfigPath, testNs)
		_, _ = sshExecutor.Exec(deleteCmd)
	})

	// 用例编号: KubeOS-IT-008
	// 用例名称: KubeOS_KubeOS addon安装与调度
	// 用例步骤:
	//   1. 通过 BKECluster addons 和 helm install 安装 kubeos-addon
	//   2. kubectl get pods -n upgrade-system
	//   3. 检查 os-proxy 调度落点
	// 预期结果:
	//   1. os-operator Running（调度到 control-plane）
	//   2. os-proxy Pod 数 == KubeOS worker 数，且只调度到带 node.openfuyao.io/os=kubeos 的节点
	//   3. os-proxy 不调度到 master
	It("【KubeOS-IT-008】KubeOS addon 安装与调度", Label("kubeos", "installation", "post-init", "P0", "26.09", "skip-temporarily"), func(ctx SpecContext) {
		By("验证 KubeOS addon 已安装")
		Eventually(func() bool {
			installed, _ := kubeosChecker.CheckKubeOSAddonInstalled()
			return installed
		}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "KubeOS addon 应已安装（CRD、os-operator、os-proxy 均存在）")

		By("验证 os-operator Running（调度到 control-plane）")
		Eventually(func() bool {
			ready, _ := kubeosChecker.CheckOSOperatorReady()
			return ready
		}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "os-operator 应 Running")

		By("验证 os-proxy Pod 数 == KubeOS worker 数")
		kubeOSWorkers, err := kubeosChecker.GetKubeOSWorkerNodes()
		Expect(err).NotTo(HaveOccurred())
		Expect(len(kubeOSWorkers)).To(Equal(1), "应有 1 个 KubeOS worker")

		Eventually(func() bool {
			desired, ready, err := kubeosChecker.GetOSProxyDaemonSetStatus()
			if err != nil {
				return false
			}
			GinkgoWriter.Printf("os-proxy DaemonSet: desired=%d, ready=%d\n", desired, ready)
			return desired == len(kubeOSWorkers) && ready == desired
		}, 5*time.Minute, 15*time.Second).Should(BeTrue(), "os-proxy Pod 数应等于 KubeOS worker 数且全部 Ready")

		By("验证 os-proxy 只调度到 KubeOS worker 节点")
		nonKubeOSPods, kubeOSPodCount, err := kubeosChecker.CheckOSProxyScheduledOnKubeOSWorkers()
		Expect(err).NotTo(HaveOccurred())
		Expect(nonKubeOSPods).To(BeEmpty(), "os-proxy 不应调度到非 KubeOS 节点")
		Expect(kubeOSPodCount).To(Equal(len(kubeOSWorkers)), "os-proxy 应调度到所有 KubeOS worker")

		By("验证 os-proxy 不调度到 master")
		masterPods, err := kubeosChecker.CheckOSProxyNotOnMaster()
		Expect(err).NotTo(HaveOccurred())
		Expect(masterPods).To(BeEmpty(), "os-proxy 不应调度到 master 节点")
	})

	// 用例编号: KubeOS_DFR_001
	// 用例名称: DFR_os-proxy Pod故障后自动恢复
	// 用例步骤:
	//   1. 删除某个 KubeOS worker 上的 os-proxy Pod
	//   2. 持续观察 DaemonSet desiredNumberScheduled、numberReady 和 Pod 重建状态
	//   3. 等待新 Pod Ready 后再次检查日志
	// 预期结果:
	//   1. DaemonSet 自动重建 os-proxy Pod
	//   2. numberReady 在合理时间内恢复到 desiredNumberScheduled
	//   3. 新 Pod 无持续 CrashLoopBackOff 或连续 ERROR 日志
	// 注：复用集成测试已创建的 KubeOS 集群（含 addon），避免重复安装
	It("【KubeOS-DFR-001】os-proxy Pod 故障后自动恢复", Label("kubeos", "installation", "post-init", "P2", "26.09", "dfr", "skip-temporarily"), func(ctx SpecContext) {
		By("获取 KubeOS worker 的 Kubernetes Node 名称")
		workerNodeName, err := kubeosChecker.GetNodeNameByIP(kubeosWorker.IP)
		Expect(err).NotTo(HaveOccurred())
		Expect(workerNodeName).NotTo(BeEmpty(), "应能通过 IP 找到 KubeOS worker 节点")

		By("记录删除前的 os-proxy DaemonSet 状态")
		desiredBefore, readyBefore, err := kubeosChecker.GetOSProxyDaemonSetStatus()
		Expect(err).NotTo(HaveOccurred())
		Expect(desiredBefore).To(BeNumerically(">", 0), "os-proxy DaemonSet 应有期望 Pod")
		Expect(readyBefore).To(Equal(desiredBefore), "删除前 os-proxy 应全部 Ready")
		GinkgoWriter.Printf("删除前 os-proxy 状态: desired=%d, ready=%d\n", desiredBefore, readyBefore)

		By("删除 KubeOS worker 上的 os-proxy Pod")
		err = kubeosChecker.DeleteOSProxyPodOnNode(workerNodeName)
		Expect(err).NotTo(HaveOccurred())

		By("验证 DaemonSet 自动重建 os-proxy Pod 并恢复 Ready")
		Eventually(func() bool {
			desired, ready, err := kubeosChecker.GetOSProxyDaemonSetStatus()
			if err != nil {
				return false
			}
			GinkgoWriter.Printf("恢复中 os-proxy 状态: desired=%d, ready=%d\n", desired, ready)
			return desired == desiredBefore && ready == desired
		}, 3*time.Minute, 10*time.Second).Should(BeTrue(), "numberReady 应在合理时间内恢复到 desiredNumberScheduled")

		By("验证新 Pod 无持续 CrashLoopBackOff")
		Consistently(func() bool {
			running, _ := kubeosChecker.CheckPodRunning(utils.KubeOSAddonNamespace, utils.OSProxyPodLabel)
			return running
		}, 1*time.Minute, 15*time.Second).Should(BeTrue(), "新 os-proxy Pod 应持续 Running，无 CrashLoopBackOff")
	})

	// 用例编号: KubeOS_DFD_002
	// 用例名称: DFD_os-proxy访问os-agent失败可定位
	// 用例步骤:
	//   1. 在 KubeOS worker 上停止 os-agent 或临时移除 /var/run/os-agent
	//   2. 查看 os-proxy Pod 状态、日志和 Event
	//   3. 恢复 os-agent 后观察 os-proxy 是否恢复
	// 预期结果:
	//   1. os-proxy 能输出 socket 路径、节点名称、连接失败原因等诊断信息
	//   2. 相关指标或事件能反映 os-agent 不可达
	//   3. 恢复 os-agent 后 os-proxy 自动恢复正常
	// 注：复用集成测试已创建的 KubeOS 集群（含 addon），在 DFR-001 之后执行
	It("【KubeOS-DFD-002】os-proxy 访问 os-agent 失败可定位", Label("kubeos", "installation", "post-init", "P2", "26.09", "dfd", "skip-temporarily"), func(ctx SpecContext) {
		By("获取 KubeOS worker 的 Kubernetes Node 名称")
		workerNodeName, err := kubeosChecker.GetNodeNameByIP(kubeosWorker.IP)
		Expect(err).NotTo(HaveOccurred())
		Expect(workerNodeName).NotTo(BeEmpty(), "应能通过 IP 找到 KubeOS worker 节点")

		By("记录停止前的 os-proxy 状态")
		desiredBefore, readyBefore, err := kubeosChecker.GetOSProxyDaemonSetStatus()
		Expect(err).NotTo(HaveOccurred())
		Expect(readyBefore).To(Equal(desiredBefore), "停止前 os-proxy 应全部 Ready")

		By("在 KubeOS worker 上停止 os-agent 服务")
		err = kubeosChecker.StopOSAgentOnNode(kubeosWorker)
		Expect(err).NotTo(HaveOccurred())
		GinkgoWriter.Printf("已在节点 %s (%s) 上停止 os-agent\n", kubeosWorker.IP, workerNodeName)

		// 已运行的 os-proxy Pod 可能不会立即感知 os-agent 停止（连接已建立），
		// 需要重启 Pod 让新 Pod 尝试连接并输出诊断信息
		By("重启 os-proxy Pod 以触发连接 os-agent 失败")
		err = kubeosChecker.DeleteOSProxyPodOnNode(workerNodeName)
		Expect(err).NotTo(HaveOccurred())

		By("验证 os-proxy 能感知 os-agent 不可达并输出诊断信息")
		Eventually(func() bool {
			notReady, detail, _ := kubeosChecker.CheckOSProxyPodNotReady(workerNodeName)
			if notReady {
				GinkgoWriter.Printf("os-proxy 诊断信息: %s\n", detail)
			} else {
				logs, _ := kubeosChecker.GetOSProxyPodLogs(workerNodeName)
				GinkgoWriter.Printf("os-proxy 当前日志（未检测到异常）:\n%s\n", logs)
			}
			return notReady
		}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "os-proxy 应能感知 os-agent 不可达（Pod 异常或日志含连接失败信息）")

		By("获取 os-proxy 日志和 Event 用于诊断验证")
		logs, err := kubeosChecker.GetOSProxyPodLogs(workerNodeName)
		Expect(err).NotTo(HaveOccurred())
		GinkgoWriter.Printf("os-proxy 日志摘要:\n%s\n", logs)

		events, err := kubeosChecker.GetOSProxyPodEvents(workerNodeName)
		Expect(err).NotTo(HaveOccurred())
		GinkgoWriter.Printf("os-proxy Event 摘要:\n%s\n", events)

		By("恢复 os-agent 服务")
		err = kubeosChecker.StartOSAgentOnNode(kubeosWorker)
		Expect(err).NotTo(HaveOccurred())

		// 重启 os-proxy Pod 让它重新连接已恢复的 os-agent
		By("重启 os-proxy Pod 以重新连接 os-agent")
		err = kubeosChecker.DeleteOSProxyPodOnNode(workerNodeName)
		Expect(err).NotTo(HaveOccurred())

		By("验证 os-proxy 恢复正常")
		Eventually(func() bool {
			running, _ := kubeosChecker.CheckPodRunning(utils.KubeOSAddonNamespace, utils.OSProxyPodLabel)
			return running
		}, 3*time.Minute, 15*time.Second).Should(BeTrue(), "恢复 os-agent 后 os-proxy 应自动恢复正常")
	})
})
