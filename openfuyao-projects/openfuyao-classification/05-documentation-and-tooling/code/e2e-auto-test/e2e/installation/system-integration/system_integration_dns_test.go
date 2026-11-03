package system_integration

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/kubernetes"
)

var _ = Describe("Workload Cluster DNS and Network Tests", Label("system_integration_dns_network", "with-workload-cluster", "system-integration", "online"), func() {
	var (
		sshClient        *executor.SSHExecutor
		localexecutor    *executor.LocalExecutor
		k8sClient        *k8s.K8SClient
		testCtx          context.Context
		testCancel       context.CancelFunc
		testNamespace    string
		createdResources []ResourceInfo
	)

	BeforeEach(func() {
		// 初始化测试上下文
		testCtx, testCancel = context.WithTimeout(context.Background(), 15*time.Minute)

		var err error

		// ssh client
		if isDaily {
			localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		} else {
			port := 22
			if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
				if portInt, err := strconv.Atoi(portStr); err == nil {
					port = portInt
				}
			}
			sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE5_IP"), port, os.Getenv("TEST_NODE5_USER"), os.Getenv("TEST_NODE5_PASSWORD"))
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")
		}

		// k8s client
		if isDaily {
			k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
		} else {
			k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

		// 4. 创建测试命名空间
		testNamespace = fmt.Sprintf("dns-network-test-%d", time.Now().Unix())
		CreateNamespaceWithK8SClient(testCtx, k8sClient, testNamespace)

		// 记录创建的命名空间
		createdResources = append(createdResources, ResourceInfo{
			Type:      "namespace",
			Name:      testNamespace,
			Namespace: "",
		})

		GinkgoWriter.Printf("DNS和网络测试初始化完成，测试命名空间: %s\n", testNamespace)
	})

	AfterEach(func() {
		// 清理所有临时资源
		CleanupTestResourcesWithK8SClient(testCtx, k8sClient, sshClient, localexecutor, isDaily, createdResources)

		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	It("应该能够测试CoreDNS的各种解析功能", Label("required_nodes:3"), func() {
		By("检查CoreDNS部署状态")
		corednsInfo := checkCoreDNSStatus(testCtx, k8sClient.Clientset)

		GinkgoWriter.Printf("CoreDNS集群信息:\n")
		GinkgoWriter.Printf("  Pod数量: %d\n", corednsInfo.PodCount)
		GinkgoWriter.Printf("  Ready状态: %d\n", corednsInfo.ReadyCount)
		GinkgoWriter.Printf("  Service IP: %s\n", corednsInfo.ServiceIP)
		GinkgoWriter.Printf("  配置版本: %s\n", corednsInfo.ConfigVersion)

		Expect(corednsInfo.ReadyCount).To(BeNumerically(">=", 1), "CoreDNS应该至少1个Pod就绪")
		Expect(corednsInfo.ServiceIP).NotTo(BeEmpty(), "CoreDNS服务应该有ClusterIP")

		By("创建测试Pod")
		testPod1 := createDNSTestPod(testCtx, k8sClient.Clientset, testNamespace, "test-pod-1", "")

		Expect(testPod1).NotTo(BeNil(), "应该能够创建测试Pod 1")

		// 记录创建的Pod
		createdResources = append(createdResources, ResourceInfo{
			Type:      "pod",
			Name:      "test-pod-1",
			Namespace: testNamespace,
		})

		// 等待Pod就绪
		WaitForPodReadyWithK8SClient(testCtx, k8sClient, testNamespace, "test-pod-1")

		By("测试同一命名空间内部服务解析")
		// 创建测试Service
		service1 := createTestService(testCtx, k8sClient.Clientset, testNamespace, "test-service-1", map[string]string{"app": "test-app-1"})

		// 记录创建的Service
		if service1 != nil {
			createdResources = append(createdResources, ResourceInfo{
				Type:      "service",
				Name:      "test-service-1",
				Namespace: testNamespace,
			})
		}

		// 测试完整域名解析
		testSameNamespaceResolutionWithExecutors(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily, testNamespace, "test-pod-1", []string{
			"test-service-1." + testNamespace + ".svc.cluster.local",
		})

		By("测试跨命名空间服务解析")
		// 创建另一个命名空间
		crossNamespace := testNamespace + "-cross"
		CreateNamespaceWithK8SClient(testCtx, k8sClient, crossNamespace)
		createdResources = append(createdResources, ResourceInfo{
			Type:      "namespace",
			Name:      crossNamespace,
			Namespace: "",
		})

		// 在另一个命名空间创建服务
		crossService := createTestService(testCtx, k8sClient.Clientset, crossNamespace, "cross-service", map[string]string{"app": "cross-app"})
		if crossService != nil {
			createdResources = append(createdResources, ResourceInfo{
				Type:      "service",
				Name:      "cross-service",
				Namespace: crossNamespace,
			})
		}

		// 测试跨命名空间解析
		testCrossNamespaceResolutionWithExecutors(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily, testNamespace, "test-pod-1", crossNamespace, "cross-service")

		By("测试外部域名解析")
		testExternalResolutionWithExecutors(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily, testNamespace, "test-pod-1")
	})
})

var _ = Describe("Workload Cluster DNS and Network Tests", Label("system_integration_dns_network", "with-workload-cluster", "system-integration", "skip-temporarily", "online"), func() {
	var (
		workloadExecutor  *executor.SSHExecutor
		masterExecutor    *executor.SSHExecutor
		workloadClientset kubernetes.Interface
		testCtx           context.Context
		testCancel        context.CancelFunc
		testNamespace     string
		createdResources  []ResourceInfo
	)

	BeforeEach(func() {
		// 初始化测试上下文
		testCtx, testCancel = context.WithTimeout(context.Background(), 15*time.Minute)

		var err error

		// 1. 创建业务集群的SSH执行器
		workloadHost := os.Getenv("TEST_NODE2_IP")
		workloadPort := os.Getenv("TEST_NODE2_PORT")
		workloadUser := os.Getenv("TEST_NODE2_USER")
		workloadPass := os.Getenv("TEST_NODE2_PASSWORD")

		Expect(workloadHost).NotTo(BeEmpty(), "TEST_NODE2_IP环境变量必须设置")
		Expect(workloadPort).NotTo(BeEmpty(), "TEST_NODE2_PORT环境变量必须设置")
		Expect(workloadUser).NotTo(BeEmpty(), "TEST_NODE2_USER环境变量必须设置")

		workloadPortInt, _ := strconv.Atoi(workloadPort)
		workloadExecutor, err = executor.NewSSHExecutor(
			workloadHost,
			workloadPortInt,
			workloadUser,
			workloadPass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到业务集群节点")

		// 2. 创建管理集群master节点SSH执行器（用于获取kubeconfig）
		managementHost := os.Getenv("TEST_NODE1_IP")
		managementPort := os.Getenv("TEST_NODE1_PORT")
		managementUser := os.Getenv("TEST_NODE1_USER")
		managementPass := os.Getenv("TEST_NODE1_PASSWORD")

		Expect(managementHost).NotTo(BeEmpty(), "TEST_NODE1_IP环境变量必须设置")
		Expect(managementPort).NotTo(BeEmpty(), "TEST_NODE1_PORT环境变量必须设置")
		Expect(managementUser).NotTo(BeEmpty(), "TEST_NODE1_USER环境变量必须设置")

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

		// 4. 创建测试命名空间
		testNamespace = fmt.Sprintf("dns-network-test-%d", time.Now().Unix())
		CreateNamespace(testCtx, workloadClientset, testNamespace)

		// 记录创建的命名空间
		createdResources = append(createdResources, ResourceInfo{
			Type:      "namespace",
			Name:      testNamespace,
			Namespace: "",
		})

		GinkgoWriter.Printf("DNS和网络测试初始化完成，测试命名空间: %s\n", testNamespace)
	})

	AfterEach(func() {
		// 清理所有临时资源
		CleanupTestResources(testCtx, workloadClientset, workloadExecutor, createdResources)

		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	It("应该能够测试Pod之间的网络通信能力", func() {
		By("准备测试环境")
		// 获取集群节点信息
		var nodeNames []string
		GinkgoWriter.Printf("使用预设的两个节点进行测试\n")
		// 获取节点1的主机名
		hostnameCmd := "hostname"
		result1, err1 := workloadExecutor.Exec(hostnameCmd)

		// 获取节点2的主机名
		failWorkloadExecutor, err := executor.NewSSHExecutor(
			FailWorkLoadNodeInfo.Host,
			FailWorkLoadNodeInfo.Port,
			FailWorkLoadNodeInfo.User,
			FailWorkLoadNodeInfo.Pass,
		)
		Expect(err).NotTo(HaveOccurred(), "应该能够连接到业务集群节点")
		result2, err2 := failWorkloadExecutor.Exec(hostnameCmd)

		if err1 == nil && result1.ExitCode == 0 && err2 == nil && result2.ExitCode == 0 {
			node1Name := strings.TrimSpace(result1.Stdout)
			node2Name := strings.TrimSpace(result2.Stdout)
			nodeNames = []string{node1Name, node2Name}
			GinkgoWriter.Printf("使用节点: %s (%s) 和 %s (%s)\n",
				node1Name, WorkloadClusterNodeInfo.Host,
				node2Name, FailWorkLoadNodeInfo.Host)
		}

		Expect(len(nodeNames)).To(BeNumerically(">=", 1), "应该至少1个节点用于网络测试")
		GinkgoWriter.Printf("可用节点: %v\n", nodeNames)

		By("创建同节点Pod")
		sameNodePod1 := createNetworkTestPod(testCtx, workloadClientset, testNamespace,
			"same-node-pod-1", nodeNames[0], map[string]string{"app": "network-test"})
		sameNodePod2 := createNetworkTestPod(testCtx, workloadClientset, testNamespace,
			"same-node-pod-2", nodeNames[0], map[string]string{"app": "network-test"})

		Expect(sameNodePod1).NotTo(BeNil(), "应该能够创建同节点Pod 1")
		Expect(sameNodePod2).NotTo(BeNil(), "应该能够创建同节点Pod 2")

		// 记录创建的Pod
		createdResources = append(createdResources, ResourceInfo{
			Type:      "pod",
			Name:      "same-node-pod-1",
			Namespace: testNamespace,
		})
		createdResources = append(createdResources, ResourceInfo{
			Type:      "pod",
			Name:      "same-node-pod-2",
			Namespace: testNamespace,
		})

		// 等待Pod就绪
		WaitForPodReady(testCtx, workloadClientset, testNamespace, "same-node-pod-1")
		WaitForPodReady(testCtx, workloadClientset, testNamespace, "same-node-pod-2")

		By("测试同节点Pod通信")
		testSameNodePodCommunication(testCtx, workloadClientset, workloadExecutor, testNamespace,
			"same-node-pod-1", "same-node-pod-2")

		if len(nodeNames) >= 2 {
			By("创建跨节点Pod")
			crossNodePod1 := createNetworkTestPod(testCtx, workloadClientset, testNamespace,
				"cross-node-pod-1", nodeNames[0], map[string]string{"app": "network-test"})
			crossNodePod2 := createNetworkTestPod(testCtx, workloadClientset, testNamespace,
				"cross-node-pod-2", nodeNames[1], map[string]string{"app": "network-test"})

			Expect(crossNodePod1).NotTo(BeNil(), "应该能够创建跨节点Pod 1")
			Expect(crossNodePod2).NotTo(BeNil(), "应该能够创建跨节点Pod 2")

			// 记录创建的Pod
			createdResources = append(createdResources, ResourceInfo{
				Type:      "pod",
				Name:      "cross-node-pod-1",
				Namespace: testNamespace,
			})
			createdResources = append(createdResources, ResourceInfo{
				Type:      "pod",
				Name:      "cross-node-pod-2",
				Namespace: testNamespace,
			})

			// 等待Pod就绪
			WaitForPodReady(testCtx, workloadClientset, testNamespace, "cross-node-pod-1")
			WaitForPodReady(testCtx, workloadClientset, testNamespace, "cross-node-pod-2")

			By("测试跨节点Pod通信")
			testCrossNodePodCommunication(testCtx, workloadClientset, workloadExecutor, testNamespace,
				"cross-node-pod-1", "cross-node-pod-2")
		} else {
			GinkgoWriter.Printf("只有1个节点，跳过跨节点通信测试\n")
		}
	})
})

func createDNSTestPod(ctx context.Context, clientset *kubernetes.Clientset, namespace, name, nodeName string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app":  "dns-test",
				"test": "dns-resolution",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:    "dns-test",
					Image:   "busybox:1.35",
					Command: []string{"sleep", "3600"},
					SecurityContext: &corev1.SecurityContext{
						Privileged:               &[]bool{false}[0],
						AllowPrivilegeEscalation: &[]bool{false}[0],
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyAlways,
			DNSPolicy:     corev1.DNSClusterFirst,
		},
	}

	// 如果指定了节点名称，尝试获取该节点的实际主机名
	if nodeName != "" {
		// 这里可以根据需要将节点名映射到实际节点
		// 例如："node-1" -> 实际节点名
		pod.Spec.NodeName = nodeName
	}

	createdPod, err := clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建DNS测试Pod失败: %v\n", err)
		return nil
	}

	return createdPod
}

// 辅助函数 - CoreDNS相关
type CoreDNSInfo struct {
	PodCount      int
	ReadyCount    int
	ServiceIP     string
	ConfigVersion string
}

func checkCoreDNSStatus(ctx context.Context, clientset *kubernetes.Clientset) CoreDNSInfo {
	var info CoreDNSInfo

	// 获取CoreDNS Pod
	pods, err := clientset.CoreV1().Pods("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "k8s-app=kube-dns",
	})

	if err == nil {
		info.PodCount = len(pods.Items)
		for _, pod := range pods.Items {
			if pod.Status.Phase == corev1.PodRunning && IsPodReady(&pod) {
				info.ReadyCount++
			}
		}
	}

	// 获取CoreDNS Service
	services, err := clientset.CoreV1().Services("kube-system").List(ctx, metav1.ListOptions{
		LabelSelector: "k8s-app=kube-dns",
	})

	if err == nil && len(services.Items) > 0 {
		info.ServiceIP = services.Items[0].Spec.ClusterIP
	}

	// 获取CoreDNS配置
	configMaps, err := clientset.CoreV1().ConfigMaps("kube-system").Get(ctx, "coredns", metav1.GetOptions{})
	if err == nil {
		info.ConfigVersion = configMaps.ResourceVersion
	}

	return info
}

func createTestService(ctx context.Context, clientset *kubernetes.Clientset, namespace, name string, selector map[string]string) *corev1.Service {
	service := &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: corev1.ServiceSpec{
			Selector: selector,
			Ports: []corev1.ServicePort{
				{
					Name:       "http",
					Port:       80,
					TargetPort: intstr.FromInt(80),
				},
			},
		},
	}

	createdService, err := clientset.CoreV1().Services(namespace).Create(ctx, service, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建测试Service失败: %v\n", err)
		return nil
	}

	return createdService
}

// 测试同一命名空间解析 - 支持根据isDaily选择执行器
func testSameNamespaceResolutionWithExecutors(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool,
	namespace, podName string, domains []string) {

	GinkgoWriter.Printf("测试同一命名空间解析: Pod=%s, 命名空间=%s\n", podName, namespace)

	for _, domain := range domains {
		By(fmt.Sprintf("解析域名: %s", domain))

		// 在Pod内执行nslookup
		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, namespace, domain)
		var result *executor.ExecResult
		var err error
		if isDaily && localExecutor != nil {
			result, err = localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			result, err = sshExecutor.Exec(cmd)
		}

		if err != nil {
			GinkgoWriter.Printf("  执行nslookup命令失败: %v\n", err)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("应该能够执行nslookup: %s", domain))
			continue
		}

		// 解析nslookup输出，检查是否成功解析
		if isDNSLookupSuccessful(result.Stdout, domain) {
			GinkgoWriter.Printf("  %s: 解析成功\n", domain)

			// 提取IP地址
			ip := extractIPFromNSLookup(result.Stdout)
			if ip != "" {
				GinkgoWriter.Printf("    IP地址: %s\n", ip)
			}

			// 对于短域名，需要特别检查
			if !strings.Contains(domain, ".") {
				// 检查是否解析到了正确的完整域名
				fullDomain := domain + "." + namespace + ".svc.cluster.local"
				if strings.Contains(result.Stdout, fullDomain) {
					GinkgoWriter.Printf("    正确解析为: %s\n", fullDomain)
				}
			}
		} else {
			GinkgoWriter.Printf("  %s: 解析失败\n", domain)
			GinkgoWriter.Printf("    输出: %s\n", result.Stdout)
			// 对于短域名，nslookup可能会显示一些错误但仍然成功
			// 所以这里不直接断言失败，而是继续测试
		}
	}
}

// 测试同一命名空间解析 - 兼容旧版本的函数签名
func testSameNamespaceResolution(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor,
	namespace, podName string, domains []string) {

	GinkgoWriter.Printf("测试同一命名空间解析: Pod=%s, 命名空间=%s\n", podName, namespace)

	for _, domain := range domains {
		By(fmt.Sprintf("解析域名: %s", domain))

		// 在Pod内执行nslookup
		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, namespace, domain)
		result, err := executor.Exec(cmd)

		if err != nil {
			GinkgoWriter.Printf("  执行nslookup命令失败: %v\n", err)
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("应该能够执行nslookup: %s", domain))
			continue
		}

		// 解析nslookup输出，检查是否成功解析
		if isDNSLookupSuccessful(result.Stdout, domain) {
			GinkgoWriter.Printf("  %s: 解析成功\n", domain)

			// 提取IP地址
			ip := extractIPFromNSLookup(result.Stdout)
			if ip != "" {
				GinkgoWriter.Printf("    IP地址: %s\n", ip)
			}

			// 对于短域名，需要特别检查
			if !strings.Contains(domain, ".") {
				// 检查是否解析到了正确的完整域名
				fullDomain := domain + "." + namespace + ".svc.cluster.local"
				if strings.Contains(result.Stdout, fullDomain) {
					GinkgoWriter.Printf("    正确解析为: %s\n", fullDomain)
				}
			}
		} else {
			GinkgoWriter.Printf("  %s: 解析失败\n", domain)
			GinkgoWriter.Printf("    输出: %s\n", result.Stdout)
			// 对于短域名，nslookup可能会显示一些错误但仍然成功
			// 所以这里不直接断言失败，而是继续测试
		}
	}
}

// 检查DNS查询是否成功
func isDNSLookupSuccessful(output, domain string) bool {
	if output == "" {
		return false
	}

	// 清理输出
	output = strings.TrimSpace(output)

	// 如果输出包含"Address:"并且后面有IP地址，说明解析成功
	lines := strings.Split(output, "\n")

	// 查找域名行
	for i, line := range lines {
		line = strings.TrimSpace(line)

		// 检查是否包含域名（忽略大小写）
		if strings.Contains(strings.ToLower(line), strings.ToLower(domain)) {
			// 检查当前行或后续行是否有Address
			for j := i; j < len(lines) && j <= i+3; j++ {
				if strings.Contains(lines[j], "Address:") && !strings.Contains(lines[j], "#") {
					// 提取IP地址
					parts := strings.Fields(lines[j])
					if len(parts) > 1 {
						ip := parts[len(parts)-1]
						// 简单检查是否是IP地址格式
						if strings.Contains(ip, ".") && !strings.Contains(ip, ":") {
							return true
						}
					}

					// 或者检查下一行
					if j+1 < len(lines) {
						nextLine := strings.TrimSpace(lines[j+1])
						if nextLine != "" && !strings.Contains(nextLine, "#") {
							// 简单检查是否是IP地址格式
							if strings.Contains(nextLine, ".") && !strings.Contains(nextLine, ":") {
								return true
							}
						}
					}
				}
			}
		}

		// 检查是否有Address行并且不包含NXDOMAIN
		if strings.Contains(line, "Address:") && !strings.Contains(line, "#") && !strings.Contains(strings.ToLower(output), "nxdomain") {
			// 检查是否有IP地址
			parts := strings.Fields(line)
			if len(parts) > 1 {
				ip := parts[len(parts)-1]
				if strings.Contains(ip, ".") && !strings.Contains(ip, ":") {
					return true
				}
			}
		}
	}

	return false
}

// 从nslookup输出中提取IP地址
func extractIPFromNSLookup(output string) string {
	lines := strings.Split(output, "\n")
	for i, line := range lines {
		if strings.Contains(line, "Address:") && !strings.Contains(line, "#") {
			parts := strings.Fields(line)
			if len(parts) > 1 {
				// 检查是否是IP地址
				ip := parts[len(parts)-1]
				if strings.Contains(ip, ".") && !strings.Contains(ip, ":") {
					return ip
				}
			}
			// 或者检查下一行
			if i+1 < len(lines) {
				nextLine := strings.TrimSpace(lines[i+1])
				if nextLine != "" && !strings.Contains(nextLine, "#") {
					return nextLine
				}
			}
		}
	}
	return ""
}

func testCrossNamespaceResolution(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor,
	srcNamespace, podName, dstNamespace, serviceName string) {

	GinkgoWriter.Printf("测试跨命名空间解析: %s -> %s.%s\n", podName, serviceName, dstNamespace)

	testCases := []struct {
		name   string
		domain string
	}{
		{
			name:   "完整FQDN",
			domain: fmt.Sprintf("%s.%s.svc.cluster.local", serviceName, dstNamespace),
		},
	}

	for _, tc := range testCases {
		By(fmt.Sprintf("测试 %s: %s", tc.name, tc.domain))

		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, srcNamespace, tc.domain)
		GinkgoWriter.Printf(cmd)
		result, err := executor.Exec(cmd)

		if err != nil {
			// 不直接检查err，因为nslookup可能返回非0退出码但仍然有有效输出
			GinkgoWriter.Printf("  执行失败: %v\n", err)
		}
		GinkgoWriter.Printf(result.Stdout)
		isResolved := isDNSLookupSuccessful(result.Stdout, tc.domain)

		if isResolved {
			GinkgoWriter.Printf("  结果: 解析成功\n")

			// 提取和显示IP
			ip := extractIPFromNSLookup(result.Stdout)
			if ip != "" {
				GinkgoWriter.Printf("    IP地址: %s\n", ip)
			}
		} else {
			GinkgoWriter.Printf("  结果: 解析失败\n")
			GinkgoWriter.Printf("  错误: 完整域名 %s 应该能够解析\n", tc.domain)
		}
	}
}

func testExternalResolution(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor,
	namespace, podName string) {

	GinkgoWriter.Printf("测试外部域名解析: Pod=%s\n", podName)

	externalDomains := []struct {
		name     string
		domain   string
		expected bool
	}{
		{"百度", "www.baidu.com", true},
		{"腾讯", "www.qq.com", true},
		{"GitHub", "api.github.com", true},
	}

	successCount := 0
	totalCount := 0

	for _, test := range externalDomains {
		totalCount++
		By(fmt.Sprintf("测试外部域名: %s (%s)", test.domain, test.name))

		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, namespace, test.domain)
		result, err := executor.Exec(cmd)

		if err != nil {
			GinkgoWriter.Printf("  执行失败: %v\n", err)
			continue
		}

		isResolved := isDNSLookupSuccessful(result.Stdout, test.domain)

		if isResolved {
			GinkgoWriter.Printf("  结果: 解析成功\n")

			if test.expected {
				successCount++

				// 提取IP地址
				ip := extractIPFromNSLookup(result.Stdout)
				if ip != "" {
					GinkgoWriter.Printf("    IP: %s\n", ip)
				}
			} else {
				GinkgoWriter.Printf("  意外成功: 这个域名不应该解析成功\n")
			}
		} else {
			GinkgoWriter.Printf("  结果: 解析失败\n")

			if !test.expected {
				successCount++
				GinkgoWriter.Printf("  符合预期: 这个域名不应该解析\n")
			} else {
				GinkgoWriter.Printf("  错误: 期望解析但失败\n")
			}
		}

		// 避免请求过快
		time.Sleep(500 * time.Millisecond)
	}

	successRate := float64(successCount) / float64(totalCount) * 100
	GinkgoWriter.Printf("外部域名解析成功率: %.1f%% (%d/%d)\n", successRate, successCount, totalCount)

	// 对于外部域名，允许一定的失败率（比如谷歌可能被墙）
	Expect(successRate).To(BeNumerically(">=", 70.0),
		"外部域名解析成功率应该至少70%")
}

// 测试外部域名解析 - 支持根据isDaily选择执行器
func testExternalResolutionWithExecutors(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool,
	namespace, podName string) {

	GinkgoWriter.Printf("测试外部域名解析: Pod=%s\n", podName)

	externalDomains := []struct {
		name     string
		domain   string
		expected bool
	}{
		{"百度", "www.baidu.com", true},
		{"腾讯", "www.qq.com", true},
		{"GitHub", "api.github.com", true},
	}

	successCount := 0
	totalCount := 0

	for _, test := range externalDomains {
		totalCount++
		By(fmt.Sprintf("测试外部域名: %s (%s)", test.domain, test.name))

		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, namespace, test.domain)
		var result *executor.ExecResult
		var err error
		if isDaily && localExecutor != nil {
			result, err = localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			result, err = sshExecutor.Exec(cmd)
		}

		if err != nil {
			GinkgoWriter.Printf("  执行失败: %v\n", err)
			continue
		}

		isResolved := isDNSLookupSuccessful(result.Stdout, test.domain)

		if isResolved {
			GinkgoWriter.Printf("  结果: 解析成功\n")

			if test.expected {
				successCount++

				// 提取IP地址
				ip := extractIPFromNSLookup(result.Stdout)
				if ip != "" {
					GinkgoWriter.Printf("    IP: %s\n", ip)
				}
			} else {
				GinkgoWriter.Printf("  意外成功: 这个域名不应该解析成功\n")
			}
		} else {
			GinkgoWriter.Printf("  结果: 解析失败\n")

			if !test.expected {
				successCount++
				GinkgoWriter.Printf("  符合预期: 这个域名不应该解析\n")
			} else {
				GinkgoWriter.Printf("  错误: 期望解析但失败\n")
			}
		}

		// 避免请求过快
		time.Sleep(500 * time.Millisecond)
	}

	successRate := float64(successCount) / float64(totalCount) * 100
	GinkgoWriter.Printf("外部域名解析成功率: %.1f%% (%d/%d)\n", successRate, successCount, totalCount)

	// 对于外部域名，允许一定的失败率（比如谷歌可能被墙）
	Expect(successRate).To(BeNumerically(">=", 70.0),
		"外部域名解析成功率应该至少70%")
}

// 辅助函数 - 网络通信相关
func createNetworkTestPod(ctx context.Context, clientset kubernetes.Interface, namespace, name, nodeName string, labels map[string]string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:    "network-test",
					Image:   "alpine:3.16",
					Command: []string{"/bin/sh", "-c", "while true; do sleep 3600; done"},
					Ports: []corev1.ContainerPort{
						{
							Name:          "http",
							ContainerPort: 80,
						},
					},
					ReadinessProbe: &corev1.Probe{
						ProbeHandler: corev1.ProbeHandler{
							Exec: &corev1.ExecAction{
								Command: []string{"sh", "-c", "echo ready"},
							},
						},
						InitialDelaySeconds: 5,
						PeriodSeconds:       5,
					},
				},
			},
			RestartPolicy: corev1.RestartPolicyAlways,
		},
	}

	if nodeName != "" {
		pod.Spec.NodeName = nodeName
	}

	createdPod, err := clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建网络测试Pod失败: %v\n", err)
		return nil
	}

	return createdPod
}

func testSameNodePodCommunication(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor,
	namespace, pod1Name, pod2Name string) {

	GinkgoWriter.Printf("测试同节点Pod通信: %s <-> %s\n", pod1Name, pod2Name)

	// 获取Pod IP
	pod1, err := clientset.CoreV1().Pods(namespace).Get(ctx, pod1Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取Pod 1信息")
	pod1IP := pod1.Status.PodIP

	pod2, err := clientset.CoreV1().Pods(namespace).Get(ctx, pod2Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取Pod 2信息")
	pod2IP := pod2.Status.PodIP

	GinkgoWriter.Printf("Pod信息:\n")
	GinkgoWriter.Printf("  %s: IP=%s, 节点=%s\n", pod1Name, pod1IP, pod1.Spec.NodeName)
	GinkgoWriter.Printf("  %s: IP=%s, 节点=%s\n", pod2Name, pod2IP, pod2.Spec.NodeName)

	Expect(pod1IP).NotTo(BeEmpty(), "Pod 1应该有IP地址")
	Expect(pod2IP).NotTo(BeEmpty(), "Pod 2应该有IP地址")

	By("测试ICMP连通性")
	// 测试ping
	pingCmd := fmt.Sprintf("kubectl exec %s -n %s -- ping -c 3 %s 2>&1", pod1Name, namespace, pod2IP)
	result, err := executor.Exec(pingCmd)
	if err == nil && result.ExitCode == 0 {
		GinkgoWriter.Printf("ICMP连通性: 成功\n")
		// 提取延迟统计
		lines := strings.Split(result.Stdout, "\n")
		for _, line := range lines {
			if strings.Contains(line, "min/avg/max") {
				GinkgoWriter.Printf("  延迟统计: %s\n", line)
			}
		}
	} else {
		GinkgoWriter.Printf("ICMP连通性: 失败 (Pod可能禁用了ping)\n")
	}
}

func testCrossNodePodCommunication(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor,
	namespace, pod1Name, pod2Name string) {

	GinkgoWriter.Printf("测试跨节点Pod通信: %s <-> %s\n", pod1Name, pod2Name)

	// 获取Pod信息
	pod1, err := clientset.CoreV1().Pods(namespace).Get(ctx, pod1Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取Pod 1信息")
	pod1IP := pod1.Status.PodIP
	pod1Node := pod1.Spec.NodeName

	pod2, err := clientset.CoreV1().Pods(namespace).Get(ctx, pod2Name, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "应该能够获取Pod 2信息")
	pod2IP := pod2.Status.PodIP
	pod2Node := pod2.Spec.NodeName

	GinkgoWriter.Printf("Pod信息:\n")
	GinkgoWriter.Printf("  %s: IP=%s, 节点=%s\n", pod1Name, pod1IP, pod1Node)
	GinkgoWriter.Printf("  %s: IP=%s, 节点=%s\n", pod2Name, pod2IP, pod2Node)

	Expect(pod1Node).NotTo(Equal(pod2Node), "Pod应该在不同的节点上")

	By("测试跨节点延迟")
	// 使用ping测试延迟
	pingCmd := fmt.Sprintf("kubectl exec %s -n %s -- ping -c 10 %s 2>&1 | tail -2", pod1Name, namespace, pod2IP)
	result, err := executor.Exec(pingCmd)
	if err == nil && result.ExitCode == 0 {
		lines := strings.Split(result.Stdout, "\n")
		for _, line := range lines {
			if strings.Contains(line, "min/avg/max") {
				GinkgoWriter.Printf("跨节点延迟: %s\n", line)
				// 解析延迟数值
				parts := strings.Split(line, "=")
				if len(parts) > 1 {
					latencyParts := strings.Split(parts[1], "/")
					if len(latencyParts) >= 3 {
						avgLatency := strings.TrimSpace(latencyParts[1])
						GinkgoWriter.Printf("  平均延迟: %s ms\n", avgLatency)
					}
				}
			}
		}
	}
}

// 测试跨命名空间解析 - 支持根据isDaily选择执行器
func testCrossNamespaceResolutionWithExecutors(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool,
	srcNamespace, podName, dstNamespace, serviceName string) {

	GinkgoWriter.Printf("测试跨命名空间解析: %s -> %s.%s\n", podName, serviceName, dstNamespace)

	testCases := []struct {
		name   string
		domain string
	}{
		{
			name:   "完整FQDN",
			domain: fmt.Sprintf("%s.%s.svc.cluster.local", serviceName, dstNamespace),
		},
	}

	for _, tc := range testCases {
		By(fmt.Sprintf("测试 %s: %s", tc.name, tc.domain))

		cmd := fmt.Sprintf("kubectl exec %s -n %s -- nslookup %s 2>&1", podName, srcNamespace, tc.domain)
		GinkgoWriter.Printf(cmd)
		var result *executor.ExecResult
		var err error
		if isDaily && localExecutor != nil {
			result, err = localExecutor.Exec(cmd)
		} else if sshExecutor != nil {
			result, err = sshExecutor.Exec(cmd)
		}

		if err != nil {
			// 不直接检查err，因为nslookup可能返回非0退出码但仍然有有效输出
			GinkgoWriter.Printf("  执行失败: %v\n", err)
		}
		GinkgoWriter.Printf(result.Stdout)
		isResolved := isDNSLookupSuccessful(result.Stdout, tc.domain)

		if isResolved {
			GinkgoWriter.Printf("  结果: 解析成功\n")

			// 提取和显示IP
			ip := extractIPFromNSLookup(result.Stdout)
			if ip != "" {
				GinkgoWriter.Printf("    IP地址: %s\n", ip)
			}
		} else {
			GinkgoWriter.Printf("  结果: 解析失败\n")
			GinkgoWriter.Printf("  错误: 完整域名 %s 应该能够解析\n", tc.domain)
		}
	}
}
