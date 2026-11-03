package system_integration

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"gopkg.in/yaml.v3"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

var _ = Describe("System Integration Tests For Post-init", Label("post-init", "system-integration", "online"), func() {
	var (
		localexecutor *executor.LocalExecutor
		k8sClient     *k8s.K8SClient
		ctx           context.Context
		cancel        context.CancelFunc
	)

	BeforeEach(func() {
		// 初始化上下文
		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)

		// 创建到引导节点的 SSH 执行器
		var err error
		localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")
	})

	AfterEach(func() {

		// 取消上下文
		if cancel != nil {
			cancel()
		}
	})

	It("应该能够连接到引导节点", Label("required_nodes:0"), func() {
		result, err := localexecutor.Exec("hostname")
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0))
		Expect(result.Stdout).NotTo(BeEmpty())
		GinkgoWriter.Printf("引导节点 hostname: %s\n", strings.TrimSpace(result.Stdout))
	})

	It("引导节点的所有Pod应该正常运行且符合预期", Label("required_nodes:0"), func() {
		VerifyClusterPods(k8sClient, ctx, BootstrapNodeExpectations)
	})

	It("应该能够读取并验证/bke/cluster/下的配置文件版本", Label("skip-temporarily"), func() {
		// 定义需要检查的配置文件
		configFiles := []string{
			"/bke/cluster/1master.yaml",
			"/bke/cluster/1master1node.yaml",
			"/bke/cluster/3master.yaml",
		}

		for _, configFile := range configFiles {
			By(fmt.Sprintf("检查配置文件: %s", configFile))

			// 读取文件内容
			result, err := localexecutor.Exec(fmt.Sprintf("cat %s", configFile))
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("应该能够读取文件: %s", configFile))
			Expect(result.ExitCode).To(Equal(0), fmt.Sprintf("读取文件失败: %s", configFile))
			Expect(result.Stdout).NotTo(BeEmpty(), fmt.Sprintf("文件内容为空: %s", configFile))

			// 解析YAML
			var config BKEClusterConfig
			err = yaml.Unmarshal([]byte(result.Stdout), &config)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("解析YAML失败: %s", configFile))

			// 验证addons
			verifyAddons(config.Spec.ClusterConfig.Addons, configFile)

			// 验证cluster配置
			verifyClusterConfig(config.Spec.ClusterConfig.Cluster, configFile)

			// 验证customExtra配置
			verifyCustomExtraConfig(config.Spec.ClusterConfig.CustomExtra, configFile)
		}
	})
})

var _ = Describe("System Integration Tests For Managenment", Label("with-management-cluster", "system-integration", "online"), func() {
	var (
		k8sClient *k8s.K8SClient
		ctx       context.Context
		cancel    context.CancelFunc
	)

	BeforeEach(func() {
		// 初始化上下文
		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)

		var err error
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	})

	AfterEach(func() {

		// 取消上下文
		if cancel != nil {
			cancel()
		}
	})

	It("管理集群的所有节点应该是Ready状态", Label("required_nodes:0"), func() {
		verifyClusterNodesReady(k8sClient, ctx)
	})

	It("管理集群的所有Pod应该正常运行且符合预期", Label("skip-temporarily"), func() {
		VerifyClusterPods(k8sClient, ctx, ManagerNodeExpectations)
	})

	It("管理集群的所有Pod的镜像应该符合预期", Label("skip-temporarily"), func() {
		verifyClusterPodImages(k8sClient, ctx, ManagerNodeExpectations)
	})
})

var _ = Describe("System Integration Tests For workload", Label("with-workload-cluster", "system-integration", "online"), func() {
	var (
		localexecutor *executor.LocalExecutor
		k8sClient     *k8s.K8SClient
		ctx           context.Context
		cancel        context.CancelFunc
	)

	BeforeEach(func() {
		// 初始化上下文
		ctx, cancel = context.WithTimeout(context.Background(), 5*time.Minute)

		var err error
		localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	})

	AfterEach(func() {
		// 清理临时 kubeconfig 文件

		// 取消上下文
		if cancel != nil {
			cancel()
		}
	})

	It("业务集群的所有节点应该是Ready状态", Label("required_nodes:3"), func() {
		verifyClusterNodesReady(k8sClient, ctx)
	})

	It("业务集群的所有Pod应该正常运行且符合预期", Label("skip-temporarily"), func() {
		VerifyClusterPods(k8sClient, ctx, BusinessNodeExpectations)
	})

	It("业务集群的所有Pod的镜像应该符合预期", Label("skip-temporarily"), func() {
		verifyClusterPodImages(k8sClient, ctx, BusinessNodeExpectations)
	})

	It("应该能够验证所有Pod的日志访问功能", Label("required_nodes:3"), func() {
		verifyWorkloadPodLogs(ctx, k8sClient, localexecutor)
	})

	It("应该能够通过Prometheus查询集群指标", Label("required_nodes:3"), func() {
		verifyPrometheusQueries(ctx, k8sClient, localexecutor)
	})

	It("应该能够验证Prometheus和Harbor服务的内部访问性", Label("required_nodes:3"), func() {
		verifyServiceAccessibility(ctx, k8sClient, localexecutor)
	})

	It("应该能够删除openfuyao-system-controller Pod并验证其自动重启(SSH方式)", Label("required_nodes:3"), func() {
		verifyPodRestart(ctx, k8sClient, localexecutor)
	})
})

var _ = Describe("Management Cluster Single Master Failure Test", Label("system_integration_master_failure", "with-workload-cluster", "system-integration", "skip-temporarily", "online"), func() {
	var (
		failMasterExecutor *executor.SSHExecutor
		master2Executor    *executor.SSHExecutor
		master3Executor    *executor.SSHExecutor
		bootstrapExecutor  *executor.SSHExecutor
		mgmtClientset      kubernetes.Interface
		workloadClientset  kubernetes.Interface
		testCtx            context.Context
		testCancel         context.CancelFunc
		targetMasterNode   string
	)

	BeforeEach(func() {
		// 初始化测试上下文
		testCtx, testCancel = context.WithTimeout(context.Background(), 10*time.Minute)

		var err error

		// 1. 创建bootstrap executor（用于获取管理集群kubeconfig）
		bootstrapHost := os.Getenv("BOOTSTRAP_TEST_IP")
		bootstrapPort := os.Getenv("BOOTSTRAP_TEST_PORT")
		bootstrapUser := os.Getenv("BOOTSTRAP_TEST_USER")
		bootstrapPass := os.Getenv("BOOTSTRAP_TEST_PASSWORD")

		Expect(bootstrapHost).NotTo(BeEmpty(), "BOOTSTRAP_TEST_IP环境变量必须设置")
		Expect(bootstrapPort).NotTo(BeEmpty(), "BOOTSTRAP_TEST_PORT环境变量必须设置")
		Expect(bootstrapUser).NotTo(BeEmpty(), "BOOTSTRAP_TEST_USER环境变量必须设置")

		bootstrapPortInt, _ := strconv.Atoi(bootstrapPort)
		bootstrapExecutor, err = executor.NewSSHExecutor(
			bootstrapHost,
			bootstrapPortInt,
			bootstrapUser,
			bootstrapPass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到引导节点")

		// 2. 创建到各master节点的SSH执行器
		// 故障节点执行器（VIP节点）
		failMasterHost := os.Getenv("MANAGEMENT_FAILOVER_TEST_IP")
		failMasterPort := os.Getenv("MANAGEMENT_FAILOVER_TEST_PORT")
		failMasterUser := os.Getenv("MANAGEMENT_FAILOVER_TEST_USER")
		failMasterPass := os.Getenv("MANAGEMENT_FAILOVER_TEST_PASSWORD")

		Expect(failMasterHost).NotTo(BeEmpty(), "MANAGEMENT_FAILOVER_TEST_IP环境变量必须设置")
		Expect(failMasterPort).NotTo(BeEmpty(), "MANAGEMENT_FAILOVER_TEST_PORT环境变量必须设置")
		Expect(failMasterUser).NotTo(BeEmpty(), "MANAGEMENT_FAILOVER_TEST_USER环境变量必须设置")

		failMasterPortInt, _ := strconv.Atoi(failMasterPort)
		failMasterExecutor, err = executor.NewSSHExecutor(
			failMasterHost,
			failMasterPortInt,
			failMasterUser,
			failMasterPass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到故障master节点")

		// master-2节点执行器
		master2Host := os.Getenv("MANAGEMENT_NODE2_TEST_IP")
		master2Port := os.Getenv("MANAGEMENT_NODE2_TEST_PORT")
		master2User := os.Getenv("MANAGEMENT_NODE2_TEST_USER")
		master2Pass := os.Getenv("MANAGEMENT_NODE2_TEST_PASSWORD")

		Expect(master2Host).NotTo(BeEmpty(), "MANAGEMENT_NODE2_TEST_IP环境变量必须设置")
		Expect(master2Port).NotTo(BeEmpty(), "MANAGEMENT_NODE2_TEST_PORT环境变量必须设置")
		Expect(master2User).NotTo(BeEmpty(), "MANAGEMENT_NODE2_TEST_USER环境变量必须设置")

		master2PortInt, _ := strconv.Atoi(master2Port)
		master2Executor, err = executor.NewSSHExecutor(
			master2Host,
			master2PortInt,
			master2User,
			master2Pass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到master-2节点")

		// master-3节点执行器
		master3Host := os.Getenv("MANAGEMENT_NODE3_TEST_IP")
		master3Port := os.Getenv("MANAGEMENT_NODE3_TEST_PORT")
		master3User := os.Getenv("MANAGEMENT_NODE3_TEST_USER")
		master3Pass := os.Getenv("MANAGEMENT_NODE3_TEST_PASSWORD")

		Expect(master3Host).NotTo(BeEmpty(), "MANAGEMENT_NODE3_TEST_IP环境变量必须设置")
		Expect(master3Port).NotTo(BeEmpty(), "MANAGEMENT_NODE3_TEST_PORT环境变量必须设置")
		Expect(master3User).NotTo(BeEmpty(), "MANAGEMENT_NODE3_TEST_USER环境变量必须设置")

		master3PortInt, _ := strconv.Atoi(master3Port)
		master3Executor, err = executor.NewSSHExecutor(
			master3Host,
			master3PortInt,
			master3User,
			master3Pass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到master-3节点")

		// 获取目标节点名称（通过hostname）
		result, err := failMasterExecutor.Exec("hostname")
		Expect(err).NotTo(HaveOccurred(), "应该能够获取节点hostname")
		targetMasterNode = strings.TrimSpace(result.Stdout)

		Expect(targetMasterNode).NotTo(BeEmpty(), "应该能够获取节点名称")
		GinkgoWriter.Printf("目标故障节点: %s (%s)\n", targetMasterNode, failMasterHost)

		// 初始化管理集群客户端
		mgmtClientset = createManagementClientset(bootstrapExecutor)
		Expect(mgmtClientset).NotTo(BeNil(), "应该能够初始化管理集群客户端")

		// 初始化业务集群客户端
		workloadClientset = createWorkloadClientset(master2Executor)
		Expect(workloadClientset).NotTo(BeNil(), "应该能够初始化业务集群客户端")

		GinkgoWriter.Printf("管理集群VIP: %s\n", ManagementClusterVIP)
		GinkgoWriter.Printf("VIP节点名称: %s\n", OriginalVIPHolder)

	})

	AfterEach(func() {
		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	It("应该能够模拟单master故障并验证集群恢复，包括ectd和keepalived", func() {
		// 第一步：停止目标master节点的kubelet
		By(fmt.Sprintf("停止 %s 节点的kubelet服务", targetMasterNode))

		// 停止kubelet服务
		result, err := failMasterExecutor.Exec("systemctl stop kubelet")
		Expect(err).NotTo(HaveOccurred(), "应该能够停止kubelet服务")
		Expect(result.ExitCode).To(Equal(0), "停止kubelet命令失败")

		GinkgoWriter.Printf("已停止 %s 节点的kubelet服务\n", targetMasterNode)

		// 2. 停止etcd容器（模拟etcd故障）
		By("停止etcd容器")
		result, err = failMasterExecutor.Exec("crictl ps --name etcd-master-1 -q 2>/dev/null")
		if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
			etcdContainerID := strings.TrimSpace(result.Stdout)
			result, err = failMasterExecutor.Exec(fmt.Sprintf("crictl stop %s", etcdContainerID))
			if err == nil && result.ExitCode == 0 {
				GinkgoWriter.Printf("已停止etcd容器: %s\n", etcdContainerID)
			} else {
				GinkgoWriter.Printf("停止etcd容器失败: %v\n", err)
			}
		} else {
			// 备用方法：通过容器名称停止
			result, err = failMasterExecutor.Exec("crictl stop $(crictl ps --name etcd -q) 2>/dev/null || true")
			if err == nil {
				GinkgoWriter.Printf("已停止etcd容器\n")
			}
		}

		// 3. 停止keepalived容器（让VIP转移）
		By("停止keepalived容器")
		result, err = failMasterExecutor.Exec("crictl ps --name keepalived-master-1 -q 2>/dev/null")
		if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
			keepalivedContainerID := strings.TrimSpace(result.Stdout)
			result, err = failMasterExecutor.Exec(fmt.Sprintf("crictl stop %s", keepalivedContainerID))
			if err == nil && result.ExitCode == 0 {
				GinkgoWriter.Printf("已停止keepalived容器: %s\n", keepalivedContainerID)
			} else {
				GinkgoWriter.Printf("停止keepalived容器失败: %v\n", err)
			}
		} else {
			// 备用方法：通过容器名称停止
			result, err = failMasterExecutor.Exec("crictl stop $(crictl ps --name keepalived -q) 2>/dev/null || true")
			if err == nil {
				GinkgoWriter.Printf("已停止keepalived容器\n")
			}
		}

		// 第二步：验证节点状态变为NotReady
		By("验证节点状态变为NotReady")

		Eventually(func() bool {
			node, err := mgmtClientset.CoreV1().Nodes().Get(testCtx, targetMasterNode, metav1.GetOptions{})
			if err != nil {
				GinkgoWriter.Printf("获取节点状态失败: %v\n", err)
				return false
			}

			for _, condition := range node.Status.Conditions {
				if condition.Type == corev1.NodeReady {
					isReady := condition.Status == corev1.ConditionTrue
					GinkgoWriter.Printf("节点 %s Ready状态: %v\n", targetMasterNode, isReady)
					return !isReady // 返回true表示节点NotReady
				}
			}
			return false
		}).WithTimeout(90*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
			fmt.Sprintf("节点 %s 应该在90秒内变为NotReady", targetMasterNode))

		GinkgoWriter.Printf("节点 %s 状态已变为NotReady\n", targetMasterNode)

		// 第三步：验证VIP转移
		By("验证VIP转移")

		// 等待几秒让keepalived检测到故障
		GinkgoWriter.Printf("等待keepalived检测故障（10秒）...\n")
		time.Sleep(10 * time.Second)

		// 检查VIP是否转移
		newVIPHolder := ""
		Eventually(func() bool {
			currentHolder := findCurrentVIPHolder([]*executor.SSHExecutor{master2Executor, master3Executor})
			GinkgoWriter.Printf("当前VIP持有者检查: %s\n", currentHolder)

			// VIP应该转移到其他节点
			if currentHolder != "" && currentHolder != OriginalVIPHolder {
				newVIPHolder = currentHolder
				GinkgoWriter.Printf("VIP已从 %s 转移到 %s\n", OriginalVIPHolder, currentHolder)
				return true
			}
			return false
		}).WithTimeout(60*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
			"VIP应该在60秒内转移到其他节点")

		GinkgoWriter.Printf("VIP转移完成: %s → %s\n", OriginalVIPHolder, newVIPHolder)

		// 第四步：验证集群API仍然可用
		By("验证集群API可用性")

		successCount := 0
		for i := 1; i <= 5; i++ {
			_, err := mgmtClientset.CoreV1().Namespaces().List(testCtx, metav1.ListOptions{})
			if err == nil {
				successCount++
				GinkgoWriter.Printf("第 %d 次检查: API Server可用\n", i)
			} else {
				GinkgoWriter.Printf("第 %d 次检查: API Server错误: %v\n", i, err)
			}
			time.Sleep(2 * time.Second)
		}
		Expect(successCount).To(BeNumerically(">=", 3), "API Server应该保持可用")
		GinkgoWriter.Printf("API Server验证通过\n")

		// 第五步：验证关键组件状态
		By("验证关键组件状态")

		// 检查etcd集群
		verifyEtcdHealth(testCtx, mgmtClientset)

		// 检查API Server
		verifyAPIServerHealth(testCtx, mgmtClientset)

		// 第六步：恢复故障节点
		By(fmt.Sprintf("恢复 %s 节点的kubelet服务", targetMasterNode))

		// 1. 恢复keepalived容器
		GinkgoWriter.Printf("恢复keepalived容器...\n")
		result, err = failMasterExecutor.Exec("crictl ps -a --name keepalived-master-1 -q 2>/dev/null")
		if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
			keepalivedContainerID := strings.TrimSpace(result.Stdout)
			result, err = failMasterExecutor.Exec(fmt.Sprintf("crictl start %s", keepalivedContainerID))
			if err == nil && result.ExitCode == 0 {
				GinkgoWriter.Printf("已启动keepalived容器: %s\n", keepalivedContainerID)
			} else {
				// 如果容器已被删除，尝试重启服务
				result, err = failMasterExecutor.Exec("systemctl restart keepalived 2>/dev/null || true")
				if err == nil {
					GinkgoWriter.Printf("已重启keepalived服务\n")
				}
			}
		}

		// 2. 恢复etcd容器
		GinkgoWriter.Printf("恢复etcd容器...\n")
		result, err = failMasterExecutor.Exec("crictl ps -a --name etcd-master-1 -q 2>/dev/null")
		if err == nil && result.ExitCode == 0 && strings.TrimSpace(result.Stdout) != "" {
			etcdContainerID := strings.TrimSpace(result.Stdout)
			result, err = failMasterExecutor.Exec(fmt.Sprintf("crictl start %s", etcdContainerID))
			if err == nil && result.ExitCode == 0 {
				GinkgoWriter.Printf("已启动etcd容器: %s\n", etcdContainerID)
			} else {
				// 备用方法：重启kubelet会重新启动etcd
				GinkgoWriter.Printf("启动etcd容器失败，等待kubelet恢复\n")
			}
		}

		// 3. 恢复kubelet服务
		GinkgoWriter.Printf("恢复kubelet服务...\n")
		result, err = failMasterExecutor.Exec("systemctl start kubelet")
		Expect(err).NotTo(HaveOccurred(), "应该能够启动kubelet服务")
		Expect(result.ExitCode).To(Equal(0), "启动kubelet命令失败")

		GinkgoWriter.Printf("已启动 %s 节点的kubelet服务\n", targetMasterNode)

		// 等待容器启动
		GinkgoWriter.Printf("等待容器启动（10秒）...\n")
		time.Sleep(10 * time.Second)

		// 第七步：验证节点恢复
		By("验证节点恢复为Ready状态")

		Eventually(func() bool {
			node, err := mgmtClientset.CoreV1().Nodes().Get(testCtx, targetMasterNode, metav1.GetOptions{})
			if err != nil {
				GinkgoWriter.Printf("获取节点状态失败: %v\n", err)
				return false
			}

			for _, condition := range node.Status.Conditions {
				if condition.Type == corev1.NodeReady {
					isReady := condition.Status == corev1.ConditionTrue
					GinkgoWriter.Printf("检查节点恢复: %s Ready=%v\n", targetMasterNode, isReady)
					return isReady
				}
			}
			return false
		}).WithTimeout(180*time.Second).WithPolling(10*time.Second).Should(BeTrue(),
			fmt.Sprintf("节点 %s 应该在180秒内恢复为Ready", targetMasterNode))

		GinkgoWriter.Printf("节点 %s 已恢复为Ready状态\n", targetMasterNode)

		// 等待一段时间让Pod稳定
		GinkgoWriter.Printf("等待Pod稳定（30秒）...\n")
		time.Sleep(30 * time.Second)

		// 第八步：验证管理集群完全恢复
		By("验证管理集群完全恢复")
		verifyManagementClusterRecovery(testCtx, mgmtClientset)

		// 第九步：验证业务集群仍然正常
		By("验证业务集群状态")
		verifyWorkloadClusterRecovery(testCtx, workloadClientset)

		GinkgoWriter.Printf("\n单master故障测试完成，集群验证通过\n")
	})
})

var _ = Describe("Workload Cluster Worker Node Failure Test", Label("system_integration_worker_failure", "with-workload-cluster", "system-integration", "skip-temporarily", "online"), func() {
	var (
		workloadExecutor  *executor.SSHExecutor
		masterExecutor    *executor.SSHExecutor
		workloadClientset kubernetes.Interface
		testCtx           context.Context
		testCancel        context.CancelFunc
		targetWorkerNode  string
		originalNode      *corev1.Node
	)

	BeforeEach(func() {
		// 初始化测试上下文
		testCtx, testCancel = context.WithTimeout(context.Background(), 10*time.Minute)

		var err error

		// 1. 创建业务集群的SSH执行器（worker节点）
		workloadHost := os.Getenv("WORKLOAD_FAILOVER_TEST_IP")
		if workloadHost == "" {
			// 如果未配置备用节点，使用默认业务集群节点
			workloadHost = os.Getenv("WORKLOAD_TEST_IP")
		}
		workloadPort := os.Getenv("WORKLOAD_FAILOVER_TEST_PORT")
		if workloadPort == "" {
			workloadPort = os.Getenv("WORKLOAD_TEST_PORT")
		}
		workloadUser := os.Getenv("WORKLOAD_FAILOVER_TEST_USER")
		if workloadUser == "" {
			workloadUser = os.Getenv("WORKLOAD_TEST_USER")
		}
		workloadPass := os.Getenv("WORKLOAD_FAILOVER_TEST_PASSWORD")
		if workloadPass == "" {
			workloadPass = os.Getenv("WORKLOAD_TEST_PASSWORD")
		}

		Expect(workloadHost).NotTo(BeEmpty(), "WORKLOAD_FAILOVER_TEST_IP或WORKLOAD_TEST_IP必须设置")
		Expect(workloadPort).NotTo(BeEmpty(), "WORKLOAD_FAILOVER_TEST_PORT或WORKLOAD_TEST_PORT必须设置")
		Expect(workloadUser).NotTo(BeEmpty(), "WORKLOAD_FAILOVER_TEST_USER或WORKLOAD_TEST_USER必须设置")

		workloadPortInt, _ := strconv.Atoi(workloadPort)
		workloadExecutor, err = executor.NewSSHExecutor(
			workloadHost,
			workloadPortInt,
			workloadUser,
			workloadPass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到业务集群worker节点")

		// 2. 创建管理集群master节点SSH执行器（用于获取kubeconfig）
		managementHost := os.Getenv("MANAGEMENT_TEST_IP")
		managementPort := os.Getenv("MANAGEMENT_TEST_PORT")
		managementUser := os.Getenv("MANAGEMENT_TEST_USER")
		managementPass := os.Getenv("MANAGEMENT_TEST_PASSWORD")

		Expect(managementHost).NotTo(BeEmpty(), "MANAGEMENT_TEST_IP环境变量必须设置")
		Expect(managementPort).NotTo(BeEmpty(), "MANAGEMENT_TEST_PORT环境变量必须设置")
		Expect(managementUser).NotTo(BeEmpty(), "MANAGEMENT_TEST_USER环境变量必须设置")

		managementPortInt, _ := strconv.Atoi(managementPort)
		masterExecutor, err = executor.NewSSHExecutor(
			managementHost,
			managementPortInt,
			managementUser,
			managementPass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到管理集群master节点")

		// 3. 初始化业务集群客户端
		workloadClientset = createWorkloadClientset(masterExecutor)
		Expect(workloadClientset).NotTo(BeNil(), "应该能够初始化业务集群客户端")

		// 4. 获取worker节点名称
		result, err := workloadExecutor.Exec("hostname")
		Expect(err).NotTo(HaveOccurred(), "应该能够获取worker节点hostname")
		targetWorkerNode = strings.TrimSpace(result.Stdout)
		Expect(targetWorkerNode).NotTo(BeEmpty(), "应该能够获取worker节点名称")

		// 5. 获取worker节点的原始状态
		originalNode, err = workloadClientset.CoreV1().Nodes().Get(testCtx, targetWorkerNode, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "应该能够获取worker节点信息")

		GinkgoWriter.Printf("目标worker节点: %s (%s)\n", targetWorkerNode, workloadHost)
		GinkgoWriter.Printf("原始调度状态: %v\n", !originalNode.Spec.Unschedulable)
	})

	AfterEach(func() {
		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	It("应该能够模拟worker节点宕机并验证Pod迁移和集群恢复", func() {
		// 第一步：记录初始状态
		By("记录初始Pod分布")
		initialPods := getPodsOnNode(testCtx, workloadClientset, targetWorkerNode)
		GinkgoWriter.Printf("节点 %s 上的初始Pod数量: %d\n", targetWorkerNode, len(initialPods))

		if len(initialPods) > 0 {
			GinkgoWriter.Printf("初始Pod列表:\n")
			for i, pod := range initialPods {
				GinkgoWriter.Printf("  %d. %s/%s (状态: %s, Ready: %v)\n",
					i+1, pod.Namespace, pod.Name, pod.Status.Phase, IsPodReady(&pod))
			}
		}

		// 第二步：标记节点为不可调度（模拟宕机）
		By(fmt.Sprintf("标记节点 %s 为不可调度", targetWorkerNode))

		// 使用kubectl命令标记节点不可调度
		cordonCmd := fmt.Sprintf("kubectl cordon %s", targetWorkerNode)
		result, err := workloadExecutor.Exec(cordonCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够执行cordon命令")
		Expect(result.ExitCode).To(Equal(0), "cordon命令失败: %s", result.Stderr)

		GinkgoWriter.Printf("已标记节点 %s 为不可调度\n", targetWorkerNode)

		// 验证节点调度状态
		Eventually(func() bool {
			node, err := workloadClientset.CoreV1().Nodes().Get(testCtx, targetWorkerNode, metav1.GetOptions{})
			if err != nil {
				GinkgoWriter.Printf("获取节点状态失败: %v\n", err)
				return false
			}
			isUnschedulable := node.Spec.Unschedulable
			GinkgoWriter.Printf("节点 %s 不可调度状态: %v\n", targetWorkerNode, isUnschedulable)
			return isUnschedulable
		}).WithTimeout(30*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
			fmt.Sprintf("节点 %s 应该在30秒内变为不可调度", targetWorkerNode))

		// 第三步：驱逐criticalNamespaces中的Pod
		By("驱逐criticalNamespaces中的Pod")

		// criticalNamespaces 定义需要驱逐的命名空间
		criticalNamespaces := []string{
			"openfuyao-system-controller",
			"openfuyao-system",
		}
		for _, namespace := range criticalNamespaces {
			// 获取该命名空间运行在目标节点上的Pod
			podsOnNodeCmd := fmt.Sprintf("kubectl get pods -n %s -o wide --field-selector spec.nodeName=%s --no-headers",
				namespace, targetWorkerNode)
			result, err := workloadExecutor.Exec(podsOnNodeCmd)

			// 删除该命名空间的所有Pod
			deleteCmd := fmt.Sprintf("kubectl delete pods --all -n %s --timeout=60s", namespace)
			GinkgoWriter.Printf("执行命令: %s\n", deleteCmd)

			result, err = workloadExecutor.Exec(deleteCmd)

			if err != nil {
				GinkgoWriter.Printf("  删除命令执行错误: %v\n", err)
				continue
			}

			if result.ExitCode == 0 {
				GinkgoWriter.Printf("  删除成功\n")
			} else {
				GinkgoWriter.Printf("  删除失败: %s\n", result.Stderr)
			}
		}

		GinkgoWriter.Printf("总共尝试删除 %d 个命名空间的Pod\n", len(criticalNamespaces))
		GinkgoWriter.Printf("等待Pod驱逐和重新调度（90秒）...\n")
		time.Sleep(90 * time.Second)

		// 第四步：验证Pod迁移
		By("验证Pod迁移情况")

		// 检查节点上剩余的Pod
		remainingPods := getPodsOnNode(testCtx, workloadClientset, targetWorkerNode)
		GinkgoWriter.Printf("节点 %s 上剩余Pod数量: %d\n", targetWorkerNode, len(remainingPods))

		// 检查criticalNamespaces的Pod是否已被驱逐
		for _, namespace := range criticalNamespaces {
			podsInNamespace := getPodsInNamespaceOnNode(testCtx, workloadClientset, namespace, targetWorkerNode)
			GinkgoWriter.Printf("命名空间 %s 在节点上的剩余Pod: %d\n", namespace, len(podsInNamespace))

			if len(podsInNamespace) > 0 {
				GinkgoWriter.Printf("未迁移的Pod:\n")
				for _, pod := range podsInNamespace {
					GinkgoWriter.Printf("  %s/%s\n", pod.Namespace, pod.Name)
				}
			}
		}

		// 第五步：验证集群整体健康状态
		By("验证集群整体健康状态")

		// 检查所有节点状态
		nodes, err := workloadClientset.CoreV1().Nodes().List(testCtx, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "应该能够获取所有节点")

		readyNodes := 0
		for _, node := range nodes.Items {
			isReady := false
			for _, condition := range node.Status.Conditions {
				if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
					isReady = true
					readyNodes++
					break
				}
			}
			GinkgoWriter.Printf("节点 %s: Ready=%v, 不可调度=%v\n",
				node.Name, isReady, node.Spec.Unschedulable)
		}

		Expect(readyNodes).To(BeNumerically(">=", 1), "至少应该有一个节点Ready")
		GinkgoWriter.Printf("集群节点状态: %d/%d Ready\n", readyNodes, len(nodes.Items))

		// 检查所有Pod状态
		By("检查所有Pod状态")
		allPods, err := workloadClientset.CoreV1().Pods("").List(testCtx, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "应该能够获取所有Pod")

		runningPods := 0
		readyPods := 0
		for _, pod := range allPods.Items {
			if pod.Status.Phase == corev1.PodRunning {
				runningPods++
			}
			if IsPodReady(&pod) {
				readyPods++
			}
		}

		GinkgoWriter.Printf("Pod状态统计:\n")
		GinkgoWriter.Printf("  总Pod数: %d\n", len(allPods.Items))
		GinkgoWriter.Printf("  Running状态: %d\n", runningPods)
		GinkgoWriter.Printf("  Ready状态: %d\n", readyPods)

		// 检查criticalNamespaces中的Pod状态
		By("检查criticalNamespaces中的Pod状态")

		for _, namespace := range criticalNamespaces {
			pods, err := workloadClientset.CoreV1().Pods(namespace).List(testCtx, metav1.ListOptions{})
			if err != nil {
				GinkgoWriter.Printf("无法获取命名空间 %s 的Pods: %v\n", namespace, err)
				continue
			}

			namespaceReady := 0
			for _, pod := range pods.Items {
				if IsPodReady(&pod) {
					namespaceReady++
				}
			}

			if len(pods.Items) > 0 {
				successRate := float64(namespaceReady) / float64(len(pods.Items)) * 100
				GinkgoWriter.Printf("命名空间 %s: %d/%d Pods Ready (%.1f%%)\n",
					namespace, namespaceReady, len(pods.Items), successRate)
			}
		}

		// 第六步：恢复节点为可调度
		By(fmt.Sprintf("恢复节点 %s 为可调度", targetWorkerNode))

		uncordonCmd := fmt.Sprintf("kubectl uncordon %s", targetWorkerNode)
		result, err = workloadExecutor.Exec(uncordonCmd)
		Expect(err).NotTo(HaveOccurred(), "应该能够执行uncordon命令")
		Expect(result.ExitCode).To(Equal(0), "uncordon命令失败: %s", result.Stderr)

		GinkgoWriter.Printf("已恢复节点 %s 为可调度\n", targetWorkerNode)

		// 验证节点调度状态恢复
		Eventually(func() bool {
			node, err := workloadClientset.CoreV1().Nodes().Get(testCtx, targetWorkerNode, metav1.GetOptions{})
			if err != nil {
				GinkgoWriter.Printf("获取节点状态失败: %v\n", err)
				return false
			}
			isSchedulable := !node.Spec.Unschedulable
			GinkgoWriter.Printf("节点 %s 可调度状态: %v\n", targetWorkerNode, isSchedulable)
			return isSchedulable
		}).WithTimeout(30*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
			fmt.Sprintf("节点 %s 应该在30秒内恢复为可调度", targetWorkerNode))

		// 第七步：删除criticalNamespaces中的Pod，让调度器重新调度到恢复的节点
		By("删除criticalNamespaces中的Pod以触发重新调度")

		for _, namespace := range criticalNamespaces {
			// 使用workerExecutor执行删除命令
			deleteCmd := fmt.Sprintf("kubectl delete pods --all -n %s --timeout=60s", namespace)
			GinkgoWriter.Printf("删除命名空间 %s 的所有Pod...\n", namespace)

			result, err := workloadExecutor.Exec(deleteCmd)
			if err != nil {
				GinkgoWriter.Printf("  删除命令执行错误: %v\n", err)
				continue
			}

			if result.ExitCode == 0 {
				GinkgoWriter.Printf("  删除成功\n")
				GinkgoWriter.Printf("  输出: %s\n", result.Stdout)
			} else {
				GinkgoWriter.Printf("  删除失败: %s\n", result.Stderr)
			}
		}

		// 等待Pod重新调度
		By("等待Pod重新调度")
		GinkgoWriter.Printf("等待Pod重新调度（120秒）...\n")
		time.Sleep(120 * time.Second)

		// 第八步：验证集群完全恢复
		By("验证集群完全恢复")

		// 检查所有节点都Ready
		Eventually(func() bool {
			nodes, err := workloadClientset.CoreV1().Nodes().List(testCtx, metav1.ListOptions{})
			if err != nil {
				return false
			}

			allReady := true
			for _, node := range nodes.Items {
				nodeReady := false
				for _, condition := range node.Status.Conditions {
					if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
						nodeReady = true
						break
					}
				}
				if !nodeReady {
					allReady = false
					break
				}
			}
			return allReady
		}).WithTimeout(120*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
			"所有节点应该在120秒内恢复为Ready")

		GinkgoWriter.Printf("所有节点已恢复为Ready状态\n")

		// 检查关键系统Pod全部Ready
		By("检查关键系统Pod全部Ready")

		Eventually(func() bool {
			allCriticalReady := true
			for _, namespace := range criticalNamespaces {
				pods, err := workloadClientset.CoreV1().Pods(namespace).List(testCtx, metav1.ListOptions{})
				if err != nil {
					continue
				}

				for _, pod := range pods.Items {
					if !IsPodReady(&pod) && pod.Status.Phase != corev1.PodSucceeded {
						allCriticalReady = false
						GinkgoWriter.Printf("Pod %s/%s 未Ready: %s\n",
							pod.Namespace, pod.Name, pod.Status.Phase)
						break
					}
				}
				if !allCriticalReady {
					break
				}
			}
			return allCriticalReady
		}).WithTimeout(180*time.Second).WithPolling(10*time.Second).Should(BeTrue(),
			"所有关键系统Pod应该在180秒内恢复为Ready")

		GinkgoWriter.Printf("\nWorker节点故障测试完成，集群验证通过\n")
		GinkgoWriter.Printf("总结:\n")
		GinkgoWriter.Printf("  1. 成功模拟worker节点宕机（标记为不可调度）\n")
		GinkgoWriter.Printf("  2. 验证了Pod迁移机制\n")
		GinkgoWriter.Printf("  3. 验证了集群在单节点故障下的可用性\n")
		GinkgoWriter.Printf("  4. 成功恢复节点并验证集群完全恢复\n")
	})
})

// 辅助函数
// verifyClusterNodesReady 验证集群的所有节点都是Ready状态
func verifyClusterNodesReady(k8sClient *k8s.K8SClient, ctx context.Context) {
	nodes, err := k8sClient.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	Expect(nodes.Items).NotTo(BeEmpty(), "集群应该至少有一个节点")

	for _, node := range nodes.Items {
		var isReady bool
		for _, condition := range node.Status.Conditions {
			if condition.Type == corev1.NodeReady {
				isReady = condition.Status == corev1.ConditionTrue
				break
			}
		}
		GinkgoWriter.Printf("节点 %s 状态: Ready=%v\n", node.Name, isReady)
		Expect(isReady).To(BeTrue(), fmt.Sprintf("节点 %s 应该是Ready状态", node.Name))
	}
}

// verifyClusterPodImages 通用方法：验证集群Pod的镜像
func verifyClusterPodImages(k8sClient *k8s.K8SClient, ctx context.Context, expectations map[string]ComponentExpectation) {
	totalChecked := 0
	totalPassed := 0
	totalFailed := 0

	GinkgoWriter.Printf("=== 开始镜像检查 ===\n")

	// 获取集群中所有命名空间的所有Pod
	allPods, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表失败")

	// 遍历所有期望组件
	for componentName, expectation := range expectations {
		// 如果没有配置期望镜像，跳过
		if len(expectation.ExpectedImages) == 0 {
			continue
		}

		totalChecked++

		// 查找匹配该组件的Pod
		var matchingPods []corev1.Pod
		for _, pod := range allPods.Items {
			if findMatchingComponent(pod.Name, map[string]ComponentExpectation{componentName: expectation}) != "" {
				matchingPods = append(matchingPods, pod)
			}
		}

		// 如果没有找到Pod，记录失败
		if len(matchingPods) == 0 {
			GinkgoWriter.Printf("组件 %s: 未找到Pod\n", componentName)
			totalFailed++
			Expect(len(matchingPods)).To(BeNumerically(">", 0),
				fmt.Sprintf("组件 %s 应该至少有一个Pod", componentName))
			continue
		}

		GinkgoWriter.Printf("\n组件 %s: 找到 %d 个Pod\n", componentName, len(matchingPods))

		// 检查每个Pod
		for podIndex, pod := range matchingPods {
			podImages := getAllPodImages(&pod)

			GinkgoWriter.Printf("Pod %d/%d: %s/%s\n",
				podIndex+1, len(matchingPods), pod.Namespace, pod.Name)
			GinkgoWriter.Printf("容器镜像: %v\n", podImages)

			// 检查每个期望镜像是否在Pod中存在
			allExpectedImagesFound := true
			for _, expectedImage := range expectation.ExpectedImages {
				if expectedImage == "" {
					// 空字符串表示通配符，总是匹配
					GinkgoWriter.Printf("期望镜像 [任意]: 跳过检查\n")
					totalPassed++
					continue
				}

				if contains(podImages, expectedImage) {
					GinkgoWriter.Printf("期望镜像 %s: 存在\n", expectedImage)
					totalPassed++
				} else {
					GinkgoWriter.Printf("期望镜像 %s: 不存在\n", expectedImage)
					allExpectedImagesFound = false
					totalFailed++
				}
			}

			if !allExpectedImagesFound {
				Expect(allExpectedImagesFound).To(BeTrue(),
					fmt.Sprintf("组件 %s 的Pod %s 应该包含所有期望镜像", componentName, pod.Name))
			}
		}
	}

	// 汇总报告
	GinkgoWriter.Printf("\n=== 镜像检查完成 ===\n")
	if totalChecked > 0 {
		GinkgoWriter.Printf("检查的组件数: %d\n", totalChecked)
		GinkgoWriter.Printf("通过: %d\n", totalPassed)
		GinkgoWriter.Printf("失败: %d\n", totalFailed)
	} else {
		GinkgoWriter.Printf("没有需要检查镜像的组件\n")
	}
}

// getAllPodImages 安全地获取Pod中的所有容器镜像
func getAllPodImages(pod *corev1.Pod) []string {
	var images []string

	// 安全地添加initContainers镜像（如果存在）
	if pod.Spec.InitContainers != nil {
		for _, container := range pod.Spec.InitContainers {
			images = append(images, container.Image)
		}
	}

	// 安全地添加containers镜像（如果存在）
	if pod.Spec.Containers != nil {
		for _, container := range pod.Spec.Containers {
			images = append(images, container.Image)
		}
	}

	return images
}

// contains 检查字符串是否在字符串切片中
func contains(slice []string, str string) bool {
	for _, item := range slice {
		if item == str {
			return true
		}
	}
	return false
}

// getBootstrapKubeconfigFromSSH 通过SSH从引导节点获取kubeconfig
func getBootstrapKubeconfigFromSSH(executor *executor.SSHExecutor, nodeIP string) (string, error) {
	// 先获取节点IP（
	if nodeIP == "" {
		result, err := executor.Exec("hostname -I | awk '{print $1}'")
		if err != nil {
			return "", fmt.Errorf("获取节点IP失败: %v", err)
		}
		if result.ExitCode == 0 && result.Stdout != "" {
			nodeIP = strings.TrimSpace(result.Stdout)
		}
	}

	// 尝试多个可能的kubeconfig位置
	kubeconfigSources := []struct {
		command string
		desc    string
	}{
		{"sudo cat /etc/rancher/k3s/k3s.yaml", "K3s默认配置文件"},
		{"sudo cat /var/lib/rancher/k3s/server/cred/admin.kubeconfig", "K3s admin kubeconfig"},
		{"sudo k3s kubectl config view --raw", "K3s kubectl配置"},
		{"sudo cat /root/.kube/config", "root用户kubeconfig"},
		{"kubectl config view --raw 2>/dev/null", "当前用户kubeconfig"},
	}

	for _, source := range kubeconfigSources {
		result, err := executor.Exec(source.command)
		if err != nil {
			continue
		}

		if result.ExitCode == 0 && result.Stdout != "" {
			config := result.Stdout

			// 验证是否是有效的kubeconfig
			if !strings.Contains(config, "apiVersion") {
				continue
			}

			// 更新server地址为节点IP
			if nodeIP != "" {
				// 替换所有可能的本地地址
				replacements := []struct {
					old string
					new string
				}{
					{"server: https://127.0.0.1:6443", fmt.Sprintf("server: https://%s:6443", nodeIP)},
					{"server: https://localhost:6443", fmt.Sprintf("server: https://%s:6443", nodeIP)},
					{`server: https://\[::1\]:6443`, fmt.Sprintf("server: https://%s:6443", nodeIP)},
				}

				for _, rep := range replacements {
					config = strings.Replace(config, rep.old, rep.new, 1)
				}
			}

			return config, nil
		}
	}

	return "", fmt.Errorf("无法在引导节点上获取有效的kubeconfig")
}

// verifyAddons 验证addons配置
func verifyAddons(addons []struct {
	Name    string                 `yaml:"name"`
	Param   map[string]interface{} `yaml:"param,omitempty"`
	Version string                 `yaml:"version,omitempty"`
	Block   bool                   `yaml:"block,omitempty"`
}, configFile string) {
	GinkgoWriter.Printf("验证配置文件 %s 的addons...\n", configFile)

	for _, addon := range addons {
		switch addon.Name {
		case "kubeproxy":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.KubeProxy),
				fmt.Sprintf("%s: kubeproxy版本应该是 %s", configFile, BootstrapNodeVersions.Addons.KubeProxy))

		case "calico":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.Calico),
				fmt.Sprintf("%s: calico版本应该是 %s", configFile, BootstrapNodeVersions.Addons.Calico))

		case "coredns":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.CoreDNS),
				fmt.Sprintf("%s: coredns版本应该是 %s", configFile, BootstrapNodeVersions.Addons.CoreDNS))

		case "bkeagent-deployer":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.BKEAgentDeployer),
				fmt.Sprintf("%s: bkeagent-deployer版本应该是 %s", configFile, BootstrapNodeVersions.Addons.BKEAgentDeployer))

			// 检查tagVersion
			if tagVersion, ok := addon.Param["tagVersion"].(string); ok {
				Expect(tagVersion).To(Equal(BootstrapNodeVersions.Addons.BKEAgentDeployerTag),
					fmt.Sprintf("%s: bkeagent-deployer tagVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.BKEAgentDeployerTag))
			} else {
				Fail(fmt.Sprintf("%s: bkeagent-deployer缺少tagVersion参数", configFile))
			}

		case "cluster-api":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.ClusterAPI),
				fmt.Sprintf("%s: cluster-api版本应该是 %s", configFile, BootstrapNodeVersions.Addons.ClusterAPI))

			// 检查各种参数
			if containerdVersion, ok := addon.Param["containerdVersion"].(string); ok {
				Expect(containerdVersion).To(Equal(BootstrapNodeVersions.Addons.ClusterAPIContainerd),
					fmt.Sprintf("%s: cluster-api containerdVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.ClusterAPIContainerd))
			}

			if manifestsVersion, ok := addon.Param["manifestsVersion"].(string); ok {
				Expect(manifestsVersion).To(Equal(BootstrapNodeVersions.Addons.ClusterAPIManifests),
					fmt.Sprintf("%s: cluster-api manifestsVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.ClusterAPIManifests))
			}

			if openFuyaoVersion, ok := addon.Param["openFuyaoVersion"].(string); ok {
				Expect(openFuyaoVersion).To(Equal(BootstrapNodeVersions.Addons.ClusterAPIOpenFuyao),
					fmt.Sprintf("%s: cluster-api openFuyaoVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.ClusterAPIOpenFuyao))
			}

			if providerVersion, ok := addon.Param["providerVersion"].(string); ok {
				Expect(providerVersion).To(Equal(BootstrapNodeVersions.Addons.ClusterAPIProvider),
					fmt.Sprintf("%s: cluster-api providerVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.ClusterAPIProvider))
			}

		case "openfuyao-system-controller":
			Expect(addon.Version).To(Equal(BootstrapNodeVersions.Addons.OpenFuyaoController),
				fmt.Sprintf("%s: openfuyao-system-controller版本应该是 %s", configFile, BootstrapNodeVersions.Addons.OpenFuyaoController))

			// 检查tagVersion
			if tagVersion, ok := addon.Param["tagVersion"].(string); ok {
				Expect(tagVersion).To(Equal(BootstrapNodeVersions.Addons.OpenFuyaoControllerTag),
					fmt.Sprintf("%s: openfuyao-system-controller tagVersion应该是 %s", configFile, BootstrapNodeVersions.Addons.OpenFuyaoControllerTag))
			}

			// 检查helmRepo
			if helmRepo, ok := addon.Param["helmRepo"].(string); ok {
				Expect(helmRepo).To(Equal(BootstrapNodeVersions.Addons.OpenFuyaoControllerRepo),
					fmt.Sprintf("%s: openfuyao-system-controller helmRepo应该是 %s", configFile, BootstrapNodeVersions.Addons.OpenFuyaoControllerRepo))
			}
		}
	}

	GinkgoWriter.Printf("配置文件 %s 的addons验证通过\n", configFile)
}

// verifyClusterConfig 验证cluster配置
func verifyClusterConfig(cluster struct {
	ContainerdVersion string `yaml:"containerdVersion"`
	EtcdVersion       string `yaml:"etcdVersion"`
	KubernetesVersion string `yaml:"kubernetesVersion"`
	OpenFuyaoVersion  string `yaml:"openFuyaoVersion"`
}, configFile string) {
	GinkgoWriter.Printf("验证配置文件 %s 的cluster配置...\n", configFile)

	Expect(cluster.ContainerdVersion).To(Equal(BootstrapNodeVersions.Cluster.ContainerdVersion),
		fmt.Sprintf("%s: cluster.containerdVersion应该是 %s", configFile, BootstrapNodeVersions.Cluster.ContainerdVersion))

	Expect(cluster.EtcdVersion).To(Equal(BootstrapNodeVersions.Cluster.EtcdVersion),
		fmt.Sprintf("%s: cluster.etcdVersion应该是 %s", configFile, BootstrapNodeVersions.Cluster.EtcdVersion))

	Expect(cluster.KubernetesVersion).To(Equal(BootstrapNodeVersions.Cluster.KubernetesVersion),
		fmt.Sprintf("%s: cluster.kubernetesVersion应该是 %s", configFile, BootstrapNodeVersions.Cluster.KubernetesVersion))

	Expect(cluster.OpenFuyaoVersion).To(Equal(BootstrapNodeVersions.Cluster.OpenFuyaoVersion),
		fmt.Sprintf("%s: cluster.openFuyaoVersion应该是 %s", configFile, BootstrapNodeVersions.Cluster.OpenFuyaoVersion))

	GinkgoWriter.Printf("配置文件 %s 的cluster配置验证通过\n", configFile)
}

// verifyCustomExtraConfig 验证customExtra配置
func verifyCustomExtraConfig(customExtra struct {
	ClusterAPI string `yaml:"clusterapi"`
	Containerd string `yaml:"containerd"`
}, configFile string) {
	GinkgoWriter.Printf("验证配置文件 %s 的customExtra配置...\n", configFile)

	Expect(customExtra.ClusterAPI).To(Equal(BootstrapNodeVersions.CustomExtra.ClusterAPI),
		fmt.Sprintf("%s: customExtra.clusterapi应该是 %s", configFile, BootstrapNodeVersions.CustomExtra.ClusterAPI))

	Expect(customExtra.Containerd).To(Equal(BootstrapNodeVersions.CustomExtra.Containerd),
		fmt.Sprintf("%s: customExtra.containerd应该是 %s", configFile, BootstrapNodeVersions.CustomExtra.Containerd))

	GinkgoWriter.Printf("配置文件 %s 的customExtra配置验证通过\n", configFile)
}

// createManagementClientset 创建管理集群客户端（参考之前的方法）
func createManagementClientset(bootstrapExecutor *executor.SSHExecutor) kubernetes.Interface {
	// 创建集群检查器
	managementChecker := utils.NewClusterCheckerWithParentKubeconfig(bootstrapExecutor, ManagementClusterName, utils.DefaultBootstraoClusterKubeconfig)

	// 获取kubeconfig内容
	managementKubeconfigStr, err := managementChecker.GetKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "应该能够获取管理集群的kubeconfig")

	// 为管理集群创建临时文件
	managementTmpFile, err := os.CreateTemp("", "kubeconfig-management-*.yaml")
	Expect(err).NotTo(HaveOccurred(), "应该能够创建管理集群临时文件")
	defer os.Remove(managementTmpFile.Name())

	if _, err := managementTmpFile.WriteString(managementKubeconfigStr); err != nil {
		Fail(fmt.Sprintf("写入管理集群kubeconfig失败: %v", err))
	}
	managementTmpFile.Close()

	// 创建Kubernetes客户端配置
	managementConfig, err := clientcmd.BuildConfigFromFlags("", managementTmpFile.Name())
	Expect(err).NotTo(HaveOccurred(), "应该能够构建管理集群配置")

	// 创建客户端
	clientset, err := kubernetes.NewForConfig(managementConfig)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建管理集群客户端")

	GinkgoWriter.Printf("管理集群客户端创建成功\n")
	return clientset
}

// createWorkloadClientset 创建业务集群客户端
func createWorkloadClientset(masterClusterExecutor *executor.SSHExecutor) kubernetes.Interface {
	// 创建集群检查器
	workloadChecker := utils.NewClusterChecker(masterClusterExecutor, WorkloadClusterName)

	// 获取kubeconfig内容
	workloadKubeconfigStr, err := workloadChecker.GetKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "应该能够获取业务集群的kubeconfig")

	// 为业务集群创建临时文件
	workloadTmpFile, err := os.CreateTemp("", "kubeconfig-workload-*.yaml")
	Expect(err).NotTo(HaveOccurred(), "应该能够创建业务集群临时文件")
	defer os.Remove(workloadTmpFile.Name())

	if _, err := workloadTmpFile.WriteString(workloadKubeconfigStr); err != nil {
		Fail(fmt.Sprintf("写入业务集群kubeconfig失败: %v", err))
	}
	workloadTmpFile.Close()

	// 创建Kubernetes客户端配置
	workloadConfig, err := clientcmd.BuildConfigFromFlags("", workloadTmpFile.Name())
	Expect(err).NotTo(HaveOccurred(), "应该能够构建业务集群配置")

	// 创建客户端
	clientset, err := kubernetes.NewForConfig(workloadConfig)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建业务集群客户端")

	GinkgoWriter.Printf("业务集群客户端创建成功\n")
	return clientset
}

// findCurrentVIPHolder 查找当前VIP持有者
func findCurrentVIPHolder(executors []*executor.SSHExecutor) string {
	for _, executor := range executors {
		if executor == nil {
			continue
		}

		// 检查节点是否有VIP
		result, err := executor.Exec(fmt.Sprintf("ip addr | grep '%s' && echo 'VIP_FOUND'", ManagementClusterVIP))
		if err == nil && result.ExitCode == 0 && strings.Contains(result.Stdout, "VIP_FOUND") {
			// 获取节点名称
			nodeResult, err := executor.Exec("hostname")
			if err == nil && nodeResult.ExitCode == 0 {
				return strings.TrimSpace(nodeResult.Stdout)
			}
		}
	}

	return ""
}

// verifyEtcdHealth 验证etcd集群健康
func verifyEtcdHealth(ctx context.Context, clientset kubernetes.Interface) {
	etcdPods, err := clientset.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "component=etcd",
	})
	if err != nil {
		GinkgoWriter.Printf("无法获取etcd Pods: %v\n", err)
		return
	}

	runningCount := 0
	for _, pod := range etcdPods.Items {
		if pod.Status.Phase == corev1.PodRunning && IsPodReady(&pod) {
			runningCount++
		}
	}

	GinkgoWriter.Printf("etcd集群状态: %d/%d 运行正常\n", runningCount, len(etcdPods.Items))
	Expect(runningCount).To(BeNumerically(">=", 2), "etcd集群应该至少2个节点健康")
	GinkgoWriter.Printf("etcd集群健康验证通过\n")
}

// verifyAPIServerHealth 验证API Server健康
func verifyAPIServerHealth(ctx context.Context, clientset kubernetes.Interface) {
	apiPods, err := clientset.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "component=kube-apiserver",
	})
	if err != nil {
		GinkgoWriter.Printf("无法获取API Server Pods: %v\n", err)
		return
	}

	runningCount := 0
	for _, pod := range apiPods.Items {
		if pod.Status.Phase == corev1.PodRunning && IsPodReady(&pod) {
			runningCount++
		}
	}

	GinkgoWriter.Printf("API Server状态: %d/%d 运行正常\n", runningCount, len(apiPods.Items))
	Expect(runningCount).To(BeNumerically(">=", 2), "API Server应该至少2个节点健康")
	GinkgoWriter.Printf("API Server健康验证通过\n")
}

// verifyManagementClusterRecovery 验证管理集群恢复
func verifyManagementClusterRecovery(ctx context.Context, clientset kubernetes.Interface) {
	GinkgoWriter.Printf("\n验证管理集群恢复:\n")

	// 检查所有节点状态
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取所有节点")

	readyNodes := 0
	for _, node := range nodes.Items {
		isReady := false
		for _, condition := range node.Status.Conditions {
			if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
				isReady = true
				readyNodes++
				break
			}
		}

		if isReady {
			GinkgoWriter.Printf(" 节点 %s: Ready\n", node.Name)
		} else {
			GinkgoWriter.Printf(" 节点 %s: NotReady\n", node.Name)
		}
	}

	Expect(readyNodes).To(Equal(len(nodes.Items)),
		fmt.Sprintf("所有管理集群节点都应该Ready，当前: %d/%d", readyNodes, len(nodes.Items)))
	GinkgoWriter.Printf("管理集群节点恢复验证通过\n")

	// 检查关键Deployment
	criticalDeployments := []struct {
		namespace string
		name      string
	}{
		{"kube-system", "coredns"},
		{"openfuyao-system-controller", "openfuyao-system-controller"},
	}

	for _, dep := range criticalDeployments {
		deployment, err := clientset.AppsV1().Deployments(dep.namespace).Get(ctx, dep.name, metav1.GetOptions{})
		if err != nil {
			GinkgoWriter.Printf("  Deployment %s/%s: 获取失败: %v\n", dep.namespace, dep.name, err)
			continue
		}

		if deployment.Status.ReadyReplicas == deployment.Status.Replicas {
			GinkgoWriter.Printf("  Deployment %s/%s: %d/%d Ready\n",
				dep.namespace, dep.name, deployment.Status.ReadyReplicas, deployment.Status.Replicas)
		} else {
			GinkgoWriter.Printf("  Deployment %s/%s: %d/%d Ready\n",
				dep.namespace, dep.name, deployment.Status.ReadyReplicas, deployment.Status.Replicas)
		}
	}
}

// verifyWorkloadClusterRecovery 验证业务集群恢复
func verifyWorkloadClusterRecovery(ctx context.Context, clientset kubernetes.Interface) {
	GinkgoWriter.Printf("\n验证业务集群恢复:\n")

	// 检查业务集群节点
	nodes, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("无法获取业务集群节点: %v\n", err)
		return
	}

	readyNodes := 0
	for _, node := range nodes.Items {
		isReady := false
		for _, condition := range node.Status.Conditions {
			if condition.Type == corev1.NodeReady && condition.Status == corev1.ConditionTrue {
				isReady = true
				readyNodes++
				break
			}
		}

		if isReady {
			GinkgoWriter.Printf("业务节点 %s: Ready\n", node.Name)
		} else {
			GinkgoWriter.Printf("业务节点 %s: NotReady\n", node.Name)
		}
	}

	GinkgoWriter.Printf("业务集群节点状态: %d/%d Ready\n", readyNodes, len(nodes.Items))

	// 检查关键业务服务
	criticalServices := []string{
		"openfuyao-system-controller",
		"openfuyao-system",
	}

	for _, namespace := range criticalServices {
		pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if err != nil {
			GinkgoWriter.Printf("无法获取命名空间 %s 的Pods: %v\n", namespace, err)
			continue
		}

		runningPods := 0
		for _, pod := range pods.Items {
			if pod.Status.Phase == corev1.PodRunning && IsPodReady(&pod) {
				runningPods++
			}
		}

		GinkgoWriter.Printf("命名空间 %s: %d/%d Pods运行正常\n",
			namespace, runningPods, len(pods.Items))
	}

	GinkgoWriter.Printf("业务集群验证完成\n")
}

// 新增辅助函数：验证业务集群Pod日志功能
func verifyWorkloadPodLogs(ctx context.Context, k8sClient *k8s.K8SClient, executor *executor.LocalExecutor) {
	By("验证业务集群Pod日志查看功能")

	// 获取所有命名空间的Pod
	allPods, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取所有Pod")

	GinkgoWriter.Printf("总共发现 %d 个Pod\n", len(allPods.Items))

	// 按类型分类Pod
	podCategories := map[string][]corev1.Pod{
		"system":     {},
		"openfuyao":  {},
		"monitoring": {},
		"ingress":    {},
		"other":      {},
	}

	for _, pod := range allPods.Items {
		namespace := pod.Namespace

		// 分类
		switch {
		case namespace == "kube-system":
			podCategories["system"] = append(podCategories["system"], pod)
		case strings.Contains(namespace, "openfuyao"):
			podCategories["openfuyao"] = append(podCategories["openfuyao"], pod)
		case namespace == "monitoring":
			podCategories["monitoring"] = append(podCategories["monitoring"], pod)
		case namespace == "ingress-nginx":
			podCategories["ingress"] = append(podCategories["ingress"], pod)
		default:
			podCategories["other"] = append(podCategories["other"], pod)
		}
	}

	// 测试每个类别的一些Pod
	for category, pods := range podCategories {
		if len(pods) == 0 {
			continue
		}

		GinkgoWriter.Printf("\n检查 %s 类别的Pod日志（共%d个）:\n", category, len(pods))

		testedPods := 0
		successfulPods := 0

		for i := 0; i < len(pods); i++ {
			pod := pods[i]

			// 只检查Running状态的Pod
			if pod.Status.Phase != corev1.PodRunning {
				GinkgoWriter.Printf("  %s/%s: 跳过（状态: %s）\n",
					pod.Namespace, pod.Name, pod.Status.Phase)
				continue
			}

			testedPods++

			// 获取日志
			logsCmd := fmt.Sprintf("kubectl logs --namespace=%s %s --tail=5",
				pod.Namespace, pod.Name)

			result, err := executor.Exec(logsCmd)

			if err != nil {
				GinkgoWriter.Printf("  %s/%s: 失败 - %v\n", pod.Namespace, pod.Name, err)
			} else if result.ExitCode != 0 {
				GinkgoWriter.Printf("  %s/%s: 失败 - 退出码: %d\n",
					pod.Namespace, pod.Name, result.ExitCode)
			} else {
				successfulPods++
				logLines := strings.Count(result.Stdout, "\n")
				GinkgoWriter.Printf("  %s/%s: 成功（%d行日志）\n",
					pod.Namespace, pod.Name, logLines)
			}
		}

		if testedPods > 0 {
			successRate := float64(successfulPods) / float64(testedPods) * 100
			GinkgoWriter.Printf("  %s类别日志检查成功率: %.1f%%\n", category, successRate)
		}
	}
}

func verifyPrometheusQueries(ctx context.Context, k8sClient *k8s.K8SClient, workloadExecutor *executor.LocalExecutor) {
	By("测试Prometheus查询功能")

	// 获取Prometheus服务的ClusterIP
	prometheusService, err := k8sClient.Clientset.CoreV1().Services("monitoring").Get(ctx, "prometheus-k8s", metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取Prometheus服务")

	prometheusIP := prometheusService.Spec.ClusterIP
	prometheusPort := prometheusService.Spec.Ports[0].Port

	GinkgoWriter.Printf("Prometheus服务地址: %s:%d\n", prometheusIP, prometheusPort)

	// 创建临时测试Pod用于在集群内部执行curl命令（ClusterIP只能在集群内部访问）
	By("创建临时测试Pod用于访问Prometheus")
	testPodName := "prometheus-query-test-pod"

	// 清理可能存在的旧Pod
	cleanupCmd := fmt.Sprintf("kubectl delete pod %s --ignore-not-found=true --timeout=10s", testPodName)
	result, err := workloadExecutor.Exec(cleanupCmd)
	if err != nil {
		GinkgoWriter.Printf("清理旧Pod时出错: %v\n", err)
	}
	time.Sleep(2 * time.Second)

	// 创建新的测试Pod
	createPodCmd := fmt.Sprintf("kubectl run %s --image=curlimages/curl:latest --restart=Never -- sh -c 'sleep 600'", testPodName)
	GinkgoWriter.Printf("执行命令: %s\n", createPodCmd)
	result, err = workloadExecutor.Exec(createPodCmd)
	if err != nil {
		GinkgoWriter.Printf("创建Pod命令执行错误: %v\n", err)
		Fail("创建临时测试Pod失败")
	}
	if result.ExitCode != 0 {
		GinkgoWriter.Printf("创建Pod命令失败，退出码: %d，错误: %s\n", result.ExitCode, result.Stderr)
		Fail(fmt.Sprintf("创建Pod命令失败，退出码: %d", result.ExitCode))
	}

	// 等待Pod变为Running状态
	By("等待测试Pod就绪")
	Eventually(func() bool {
		checkCmd := fmt.Sprintf("kubectl get pod %s -o jsonpath='{.status.phase}'", testPodName)
		result, err := workloadExecutor.Exec(checkCmd)
		if err != nil {
			return false
		}
		phase := strings.TrimSpace(result.Stdout)
		GinkgoWriter.Printf("Pod状态: %s\n", phase)
		return phase == "Running"
	}).WithTimeout(60*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
		"临时测试Pod应该在60秒内变为Running状态")

	// 确保在方法结束时清理测试Pod
	defer func() {
		By("清理临时测试Pod")
		deleteCmd := fmt.Sprintf("kubectl delete pod %s --ignore-not-found=true --timeout=30s", testPodName)
		workloadExecutor.Exec(deleteCmd)
	}()

	// 测试Prometheus查询
	testQueries := []struct {
		name        string
		query       string
		description string
		expectData  bool
	}{
		{
			name:        "集群节点状态",
			query:       "kube_node_status_condition",
			description: "查询集群节点状态条件",
			expectData:  true,
		},
		{
			name:        "节点CPU使用率",
			query:       "100 - (avg by (instance) (rate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)",
			description: "查询节点CPU使用率",
			expectData:  true,
		},
		{
			name:        "节点磁盘使用率",
			query:       "100 - ((node_filesystem_avail_bytes{mountpoint=\"/\"} * 100) / node_filesystem_size_bytes{mountpoint=\"/\"})",
			description: "查询节点磁盘使用率",
			expectData:  true,
		},
	}

	totalQueries := len(testQueries)
	successfulQueries := 0
	failedQueries := 0

	GinkgoWriter.Printf("\n开始测试 %d 个Prometheus查询...\n", totalQueries)

	for _, test := range testQueries {
		By(fmt.Sprintf("测试查询: %s", test.name))

		// 构建Prometheus API URL
		queryURL := fmt.Sprintf("http://%s:%d/api/v1/query?query=%s",
			prometheusIP, prometheusPort, url.QueryEscape(test.query))

		// 构造curl命令（通过kubectl exec在集群内Pod中执行，确保能访问ClusterIP）
		curlCmd := fmt.Sprintf("kubectl exec -i %s -- curl -s --max-time 30 '%s'", testPodName, queryURL)
		GinkgoWriter.Printf("  执行命令: %s\n", curlCmd)

		// 在测试Pod内部执行curl命令
		result, err := workloadExecutor.Exec(curlCmd)
		if err != nil {
			GinkgoWriter.Printf("  %s: 执行curl命令失败 - %v\n", test.name, err)
			failedQueries++
			continue
		}

		if result.ExitCode != 0 {
			GinkgoWriter.Printf("  %s: curl命令失败 - 退出码: %d\n", test.name, result.ExitCode)
			GinkgoWriter.Printf("    错误输出: %s\n", result.Stderr)
			failedQueries++
			continue
		}

		// 检查响应是否为空
		response := strings.TrimSpace(result.Stdout)
		if response == "" {
			GinkgoWriter.Printf("  %s: 响应为空\n", test.name)
			failedQueries++
			continue
		}

		GinkgoWriter.Printf("  %s: curl命令执行成功\n", test.name)

		// 尝试解析JSON响应
		var promResponse PrometheusQueryResponse
		err = json.Unmarshal([]byte(response), &promResponse)

		if err != nil {
			GinkgoWriter.Printf("  %s: 解析响应失败 - %v\n", test.name, err)
			failedQueries++
			continue
		}

		// 检查查询状态
		if promResponse.Status != "success" {
			GinkgoWriter.Printf("  %s: Prometheus查询失败 - 状态: %s\n",
				test.name, promResponse.Status)
			if promResponse.Error != "" {
				GinkgoWriter.Printf("    错误信息: %s\n", promResponse.Error)
			}
			failedQueries++
			continue
		}

		// 检查数据
		if test.expectData {
			if len(promResponse.Data.Result) == 0 {
				GinkgoWriter.Printf("  %s: 查询成功但无数据返回\n", test.name)
				// 这个可能不一定是失败，取决于实际情况
				failedQueries++
				continue
			}

			// 输出一些样本数据
			sampleCount := min(3, len(promResponse.Data.Result))
			GinkgoWriter.Printf("  %s: 查询成功，返回 %d 个数据点\n",
				test.name, len(promResponse.Data.Result))

			for i := 0; i < sampleCount; i++ {
				result := promResponse.Data.Result[i]
				valueStr := "无值"
				if len(result.Value) > 1 {
					// 尝试将值转换为可读格式
					value := result.Value[1]
					if strVal, ok := value.(string); ok {
						valueStr = strVal
					} else {
						valueStr = fmt.Sprintf("%v", value)
					}
				}

				// 美化metric输出
				metricJson, _ := json.Marshal(result.Metric)
				GinkgoWriter.Printf("    样本 %d: metric=%s, value=%s\n",
					i+1, string(metricJson), valueStr)
			}
		} else {
			GinkgoWriter.Printf("  %s: 查询成功\n", test.name)
		}

		successfulQueries++
	}

	// 汇总结果
	GinkgoWriter.Printf("\nPrometheus查询测试完成：\n")
	GinkgoWriter.Printf("  总共测试查询数: %d\n", totalQueries)
	GinkgoWriter.Printf("  成功查询: %d\n", successfulQueries)
	GinkgoWriter.Printf("  失败查询: %d\n", failedQueries)

	successRate := 0.0
	if totalQueries > 0 {
		successRate = float64(successfulQueries) / float64(totalQueries) * 100
	}
	GinkgoWriter.Printf("  成功率: %.1f%%\n", successRate)

	Expect(successRate).To(BeNumerically(">=", 80.0),
		fmt.Sprintf("Prometheus查询成功率应该至少80%%，当前: %.1f%%", successRate))

	GinkgoWriter.Printf("Prometheus查询功能验证通过\n")
}

// verifyServiceAccessibility 验证服务内部访问性
func verifyServiceAccessibility(ctx context.Context, k8sClient *k8s.K8SClient, workloadExecutor *executor.LocalExecutor) {
	By("验证Prometheus和Harbor服务的内部访问性")

	By("创建临时测试Pod")

	// 1. 先清理可能存在的旧Pod
	By("清理可能存在的旧Pod")
	cleanupCmd := "kubectl delete pod service-access-test-pod --ignore-not-found=true --timeout=10s"
	result, err := workloadExecutor.Exec(cleanupCmd)
	if err != nil {
		GinkgoWriter.Printf("清理旧Pod时出错: %v\n", err)
	}
	if result.ExitCode == 0 {
		GinkgoWriter.Printf("清理完成\n")
	}

	// 等待清理完成
	time.Sleep(2 * time.Second)

	// 2. 使用kubectl run创建Pod
	By("使用kubectl run创建临时测试Pod")
	createPodCmd := "kubectl run service-access-test-pod --image=curlimages/curl:latest --restart=Never -- sh -c 'sleep 600'"

	GinkgoWriter.Printf("执行命令: %s\n", createPodCmd)
	result, err = workloadExecutor.Exec(createPodCmd)

	if err != nil {
		GinkgoWriter.Printf("创建Pod命令执行错误: %v\n", err)
		GinkgoWriter.Printf("退出码: %d\n", result.ExitCode)
		GinkgoWriter.Printf("标准输出: %s\n", result.Stdout)
		GinkgoWriter.Printf("标准错误: %s\n", result.Stderr)
		Fail("创建临时测试Pod失败")
	}

	if result.ExitCode != 0 {
		GinkgoWriter.Printf("创建Pod命令失败:\n")
		GinkgoWriter.Printf("退出码: %d\n", result.ExitCode)
		GinkgoWriter.Printf("标准输出: %s\n", result.Stdout)
		GinkgoWriter.Printf("标准错误: %s\n", result.Stderr)
		Fail(fmt.Sprintf("创建Pod命令失败，退出码: %d", result.ExitCode))
	}

	GinkgoWriter.Printf("已创建临时测试Pod: service-access-test-pod\n")

	// 3. 等待Pod变为Running状态
	By("等待Pod就绪")
	Eventually(func() bool {
		checkCmd := "kubectl get pod service-access-test-pod -o jsonpath='{.status.phase}'"
		result, err := workloadExecutor.Exec(checkCmd)
		if err != nil {
			return false
		}
		phase := strings.TrimSpace(result.Stdout)
		GinkgoWriter.Printf("Pod状态: %s\n", phase)
		return phase == "Running"
	}).WithTimeout(60*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
		"临时测试Pod应该在60秒内变为Running状态")

	// 4. 在Pod内部测试访问Prometheus服务
	By("测试访问Prometheus服务（循环100次）")
	prometheusURL := "http://prometheus-k8s.monitoring.svc.cluster.local:9090"

	successCount := 0
	failCount := 0
	totalRequests := 100

	GinkgoWriter.Printf("开始测试Prometheus服务访问，总共 %d 次请求...\n", totalRequests)
	startTime := time.Now()

	for i := 1; i <= totalRequests; i++ {
		// 在Pod内部执行curl命令
		curlCmd := fmt.Sprintf("kubectl exec -i service-access-test-pod -- curl -s -o /dev/null -w '%%{http_code}' --connect-timeout 10 --max-time 15 %s", prometheusURL)

		result, err := workloadExecutor.Exec(curlCmd)

		if err != nil {
			GinkgoWriter.Printf("  第 %d 次请求失败 (命令执行错误): %v\n", i, err)
			failCount++
			continue
		}

		if result.ExitCode != 0 {
			GinkgoWriter.Printf("  第 %d 次请求失败 (退出码: %d): %s\n",
				i, result.ExitCode, result.Stderr)
			failCount++
			continue
		}

		httpCode := strings.TrimSpace(result.Stdout)
		if httpCode == "200" || httpCode == "302" || httpCode == "301" {
			successCount++
			if i%10 == 0 { // 每10次输出一次进度
				GinkgoWriter.Printf("  第 %d 次请求成功 (HTTP状态码: %s)\n", i, httpCode)
			}
		} else {
			failCount++
			GinkgoWriter.Printf("  第 %d 次请求失败 (HTTP状态码: %s)\n", i, httpCode)
		}

		// 添加短暂延迟，避免请求过于密集
		if i < totalRequests {
			time.Sleep(200 * time.Millisecond)
		}
	}

	elapsedTime := time.Since(startTime)
	prometheusSuccessRate := float64(successCount) / float64(totalRequests) * 100

	GinkgoWriter.Printf("\nPrometheus访问测试完成:\n")
	GinkgoWriter.Printf("  总请求数: %d\n", totalRequests)
	GinkgoWriter.Printf("  成功次数: %d\n", successCount)
	GinkgoWriter.Printf("  失败次数: %d\n", failCount)
	GinkgoWriter.Printf("  成功率: %.1f%%\n", prometheusSuccessRate)
	GinkgoWriter.Printf("  总耗时: %v\n", elapsedTime)
	GinkgoWriter.Printf("  平均耗时: %v/请求\n", elapsedTime/time.Duration(totalRequests))

	// 验证Prometheus访问成功率
	Expect(prometheusSuccessRate).To(BeNumerically(">=", 90.0),
		fmt.Sprintf("Prometheus服务访问成功率应该至少90%%，当前: %.1f%%", prometheusSuccessRate))

	// 5. 在Pod内部测试访问Harbor服务
	By("测试访问Harbor服务（循环50次）")
	harborURL := "http://local-harbor.openfuyao-system.svc.cluster.local"

	successCount = 0
	failCount = 0
	totalRequests = 50

	GinkgoWriter.Printf("\n开始测试Harbor服务访问，总共 %d 次请求...\n", totalRequests)
	startTime = time.Now()

	for i := 1; i <= totalRequests; i++ {
		// 在Pod内部执行curl命令
		curlCmd := fmt.Sprintf("kubectl exec -i service-access-test-pod -- curl -s -o /dev/null -w '%%{http_code}' --connect-timeout 10 --max-time 15 %s", harborURL)

		result, err := workloadExecutor.Exec(curlCmd)

		if err != nil {
			GinkgoWriter.Printf("  第 %d 次请求失败 (命令执行错误): %v\n", i, err)
			failCount++
			continue
		}

		if result.ExitCode != 0 {
			GinkgoWriter.Printf("  第 %d 次请求失败 (退出码: %d): %s\n",
				i, result.ExitCode, result.Stderr)
			failCount++
			continue
		}

		httpCode := strings.TrimSpace(result.Stdout)
		if httpCode == "200" || httpCode == "302" || httpCode == "301" || httpCode == "401" {
			// 401是正常的，因为需要登录认证
			successCount++
			if i%10 == 0 { // 每10次输出一次进度
				GinkgoWriter.Printf("  第 %d 次请求成功 (HTTP状态码: %s)\n", i, httpCode)
			}
		} else {
			failCount++
			GinkgoWriter.Printf("  第 %d 次请求失败 (HTTP状态码: %s)\n", i, httpCode)
		}

		// 添加短暂延迟，避免请求过于密集
		if i < totalRequests {
			time.Sleep(100 * time.Millisecond) // Harbor请求间隔更短
		}
	}

	elapsedTime = time.Since(startTime)
	harborSuccessRate := float64(successCount) / float64(totalRequests) * 100

	GinkgoWriter.Printf("\nHarbor访问测试完成:\n")
	GinkgoWriter.Printf("  总请求数: %d\n", totalRequests)
	GinkgoWriter.Printf("  成功次数: %d\n", successCount)
	GinkgoWriter.Printf("  失败次数: %d\n", failCount)
	GinkgoWriter.Printf("  成功率: %.1f%%\n", harborSuccessRate)
	GinkgoWriter.Printf("  总耗时: %v\n", elapsedTime)
	GinkgoWriter.Printf("  平均耗时: %v/请求\n", elapsedTime/time.Duration(totalRequests))

	// 验证Harbor访问成功率
	Expect(harborSuccessRate).To(BeNumerically(">=", 95.0),
		fmt.Sprintf("Harbor服务访问成功率应该至少95%%，当前: %.1f%%", harborSuccessRate))

	// 6. 清理临时Pod
	By("清理临时测试Pod")
	deletePodCmd := "kubectl delete pod service-access-test-pod --timeout=120s"
	result, err = workloadExecutor.Exec(deletePodCmd)
	Expect(err).NotTo(HaveOccurred(), "应该能够执行kubectl delete命令")

	GinkgoWriter.Printf("已清理临时测试Pod\n")

	// 7. 汇总报告
	GinkgoWriter.Printf("\n=== 服务访问测试总结 ===\n")
	GinkgoWriter.Printf("Prometheus服务访问:\n")
	GinkgoWriter.Printf("  - 请求次数: 50\n")
	GinkgoWriter.Printf("  - 成功率: %.1f%% (要求 >= 90%%)\n", prometheusSuccessRate)
	GinkgoWriter.Printf("Harbor服务访问:\n")
	GinkgoWriter.Printf("  - 请求次数: 100\n")
	GinkgoWriter.Printf("  - 成功率: %.1f%% (要求 >= 95%%)\n", harborSuccessRate)

	if prometheusSuccessRate >= 90.0 && harborSuccessRate >= 95.0 {
		GinkgoWriter.Printf("所有服务访问测试通过！\n")
	}
}

func verifyPodRestart(ctx context.Context, k8sClient *k8s.K8SClient, workloadExecutor *executor.LocalExecutor) {
	By("查找openfuyao-system-controller Pod")
	result, err := workloadExecutor.Exec("kubectl get pods -n openfuyao-system-controller -l app=openfuyao-system-controller -o jsonpath='{.items[0].metadata.name}'")
	Expect(err).NotTo(HaveOccurred(), "应该能够查找Pod")
	Expect(result.ExitCode).To(Equal(0), "查找Pod命令失败")
	Expect(result.Stdout).NotTo(BeEmpty(), "应该找到openfuyao-system-controller Pod")

	targetPodName := strings.TrimSpace(result.Stdout)
	GinkgoWriter.Printf("找到目标Pod: %s\n", targetPodName)

	By("删除openfuyao-system-controller Pod")
	result, err = workloadExecutor.Exec(fmt.Sprintf("kubectl delete pod %s -n openfuyao-system-controller", targetPodName))
	Expect(err).NotTo(HaveOccurred(), "应该能够删除Pod")
	Expect(result.ExitCode).To(Equal(0), "删除Pod命令失败")

	GinkgoWriter.Printf("已删除Pod: %s，等待Pod重启...\n", targetPodName)

	By("等待Pod重启")
	timeout := 60 * time.Second
	startTime := time.Now()
	var newPodName string

	// 等待新Pod出现
	Eventually(func() bool {
		result, err := workloadExecutor.Exec("kubectl get pods -n openfuyao-system-controller -l app=openfuyao-system-controller -o jsonpath='{.items[0].metadata.name}'")
		if err != nil || result.ExitCode != 0 {
			return false
		}

		currentPodName := strings.TrimSpace(result.Stdout)
		if currentPodName != "" && currentPodName != targetPodName {
			newPodName = currentPodName
			return true
		}
		return false
	}).WithTimeout(timeout).WithPolling(2*time.Second).Should(BeTrue(),
		"应该在60秒内看到新的openfuyao-system-controller Pod启动")

	GinkgoWriter.Printf("发现新Pod: %s (耗时: %v)\n", newPodName, time.Since(startTime))

	By("验证新Pod状态")
	Eventually(func() bool {
		result, err := workloadExecutor.Exec(fmt.Sprintf("kubectl get pod %s -n openfuyao-system-controller -o jsonpath='{..status.phase},{..status.conditions[?(@.type==\"Ready\")].status}'", newPodName))
		if err != nil || result.ExitCode != 0 {
			return false
		}

		statusParts := strings.Split(result.Stdout, ",")
		if len(statusParts) < 2 {
			return false
		}

		isRunning := statusParts[0] == "Running"
		isReady := statusParts[1] == "True"

		GinkgoWriter.Printf("Pod %s 状态: Phase=%s, Ready=%v\n", newPodName, statusParts[0], isReady)

		return isRunning && isReady
	}).WithTimeout(30*time.Second).WithPolling(2*time.Second).Should(BeTrue(),
		"新Pod应该在30秒内变为Ready和Running状态")

	GinkgoWriter.Printf("Pod重启验证成功: %s 已恢复Ready和Running状态\n", newPodName)
}

func getPodsOnNode(ctx context.Context, clientset kubernetes.Interface, nodeName string) []corev1.Pod {
	allPods, err := clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{
		FieldSelector: fmt.Sprintf("spec.nodeName=%s", nodeName),
	})
	if err != nil {
		GinkgoWriter.Printf("获取节点 %s 上的Pod失败: %v\n", nodeName, err)
		return []corev1.Pod{}
	}
	return allPods.Items
}

// 获取特定命名空间在特定节点上的Pod
func getPodsInNamespaceOnNode(ctx context.Context, clientset kubernetes.Interface, namespace string, nodeName string) []corev1.Pod {
	var pods []corev1.Pod

	podList, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return pods
	}

	for _, pod := range podList.Items {
		if pod.Spec.NodeName == nodeName {
			pods = append(pods, pod)
		}
	}

	return pods
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
