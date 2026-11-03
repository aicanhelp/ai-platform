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

var isDaily = true

var _ = Describe("Service Status and Connectivity Tests", Label("system_integration_services", "with-workload-cluster", "system-integration", "online"), func() {
	var (
		sshClient        *executor.SSHExecutor
		localexecutor    *executor.LocalExecutor
		k8sClient        *k8s.K8SClient
		testCtx          context.Context
		testCancel       context.CancelFunc
		testNamespace    string
		createdResources []ResourceInfo
		testPodName      string
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
		testNamespace = fmt.Sprintf("clusterapi-test-%d", time.Now().Unix())
		CreateNamespaceWithK8SClient(testCtx, k8sClient, testNamespace)

		// 记录创建的命名空间
		createdResources = append(createdResources, ResourceInfo{
			Type:      "namespace",
			Name:      testNamespace,
			Namespace: "",
		})

		// 5. 创建测试Pod用于连通性测试
		testPodName = "service-test-pod"
		testPod1 := createServiceTestPod(testCtx, k8sClient.Clientset, testNamespace, testPodName)
		Expect(testPod1).NotTo(BeNil(), "应该能够创建测试Pod 1")
		//Expect(err).NotTo(HaveOccurred(), "应该能够创建测试Pod")
		createdResources = append(createdResources, ResourceInfo{
			Type:      "pod",
			Name:      testPodName,
			Namespace: testNamespace,
		})

		// 等待测试Pod就绪
		WaitForPodReady(testCtx, k8sClient.Clientset, testNamespace, testPodName)

		GinkgoWriter.Printf("clusterAPI测试初始化完成，测试命名空间: %s，测试Pod: %s\n", testNamespace, testPodName)
	})

	AfterEach(func() {
		// 清理所有临时资源
		CleanupTestResourcesWithK8SClient(testCtx, k8sClient, sshClient, localexecutor, isDaily, createdResources)

		// 清理资源
		if testCancel != nil {
			testCancel()
		}
	})

	Describe("Service Status and Connectivity Tests", func() {
		It("应该能够验证所有Service的状态和连通性", Label("required_nodes:3"), func() {
			By("验证所有Service的存在和状态")
			serviceStatusResults := verifyServiceStatus(testCtx, k8sClient.Clientset)

			GinkgoWriter.Printf("Service状态检查结果:\n")
			GinkgoWriter.Printf("  总Service数: %d\n", serviceStatusResults.TotalServices)
			GinkgoWriter.Printf("  找到的Service数: %d\n", serviceStatusResults.FoundServices)
			GinkgoWriter.Printf("  类型正确的Service数: %d\n", serviceStatusResults.CorrectTypeServices)
			GinkgoWriter.Printf("  有ClusterIP的Service数: %d\n", serviceStatusResults.HasClusterIPServices)
			Expect(serviceStatusResults.FoundServices).To(Equal(serviceStatusResults.TotalServices), "所有期望的 Service 应被找到")

			By("测试关键Service的连通性")
			connectivityResults := testCriticalServicesConnectivityWithExecutors(testCtx, k8sClient.Clientset, sshClient, localexecutor, isDaily, testNamespace, testPodName)
			GinkgoWriter.Printf("Service连通性测试结果:\n")
			GinkgoWriter.Printf("  测试的Service数: %d\n", connectivityResults.TestedServices)
			GinkgoWriter.Printf("  连通成功的Service数: %d\n", connectivityResults.SuccessfulConnections)
			GinkgoWriter.Printf("  连通失败的Service数: %d\n", connectivityResults.FailedConnections)
			Expect(connectivityResults.SuccessfulConnections).To(Equal(connectivityResults.SuccessfulConnections), "关键Service连通率100%")
		})
	})
})

// Service状态检查结果
type ServiceStatusResult struct {
	TotalServices        int
	FoundServices        int
	CorrectTypeServices  int
	HasClusterIPServices int
	StatusDetails        map[string]ServiceStatusDetail
}

type ServiceStatusDetail struct {
	Exists       bool
	TypeMatch    bool
	HasClusterIP bool
	ClusterIP    string
	Ports        []corev1.ServicePort
	ErrorMessage string
}

// Service连通性测试结果
type ServiceConnectivityResult struct {
	TestedServices        int
	SuccessfulConnections int
	FailedConnections     int
	ConnectionDetails     map[string]ConnectionDetail
}

type ConnectionDetail struct {
	ServiceName  string
	StatusCode   int
	ResponseTime time.Duration
	Error        string
	IsAccessible bool
}

// 创建用于Service测试的Pod
func createServiceTestPod(ctx context.Context, clientset *kubernetes.Clientset, namespace, name string) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			Labels: map[string]string{
				"app":  "service-test",
				"test": "service-connectivity",
			},
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{
				{
					Name:    "service-test",
					Image:   "curlimages/curl:latest", // 使用包含curl的镜像
					Command: []string{"sh", "-c", "while true; do sleep 3600; done"},
				},
			},
			RestartPolicy: corev1.RestartPolicyAlways,
		},
	}

	createdPod, err := clientset.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建Service测试Pod失败: %v\n", err)
		return nil
	}

	return createdPod
}

// 验证Service状态
func verifyServiceStatus(ctx context.Context, clientset *kubernetes.Clientset) ServiceStatusResult {
	var result ServiceStatusResult
	result.StatusDetails = make(map[string]ServiceStatusDetail)
	result.TotalServices = len(ServiceExpectations)

	GinkgoWriter.Printf("=== 开始Service状态检查 (共%d个Service) ===\n", result.TotalServices)

	// 获取所有命名空间的Service
	allServices, err := clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("获取Service列表失败: %v\n", err)
		return result
	}

	// 创建Service查找映射
	serviceMap := make(map[string]map[string]corev1.Service) // namespace -> name -> Service
	for _, svc := range allServices.Items {
		if _, ok := serviceMap[svc.Namespace]; !ok {
			serviceMap[svc.Namespace] = make(map[string]corev1.Service)
		}
		serviceMap[svc.Namespace][svc.Name] = svc
	}

	// 检查每个期望的Service
	for serviceKey, expectation := range ServiceExpectations {
		var detail ServiceStatusDetail

		GinkgoWriter.Printf("\n检查Service: %s/%s\n", expectation.Namespace, expectation.Name)

		// 查找Service
		nsServices, nsExists := serviceMap[expectation.Namespace]
		if !nsExists {
			detail.ErrorMessage = fmt.Sprintf("命名空间 %s 不存在", expectation.Namespace)
			result.StatusDetails[serviceKey] = detail
			continue
		}

		actualService, svcExists := nsServices[expectation.Name]
		if !svcExists {
			detail.ErrorMessage = "Service不存在"
			result.StatusDetails[serviceKey] = detail
			continue
		}

		detail.Exists = true
		result.FoundServices++

		// 检查Service类型
		actualType := string(actualService.Spec.Type)
		detail.TypeMatch = (actualType == expectation.Type)
		if detail.TypeMatch {
			result.CorrectTypeServices++
			GinkgoWriter.Printf("类型正确: %s\n", actualType)
		} else {
			GinkgoWriter.Printf("类型不匹配: 期望 %s, 实际 %s\n",
				expectation.Type, actualType)
		}

		// 检查ClusterIP
		detail.ClusterIP = actualService.Spec.ClusterIP
		if detail.ClusterIP == "" {
			GinkgoWriter.Printf("没有ClusterIP\n")
		} else if detail.ClusterIP == "None" {
			GinkgoWriter.Printf("Headless Service (ClusterIP: None)\n")
			detail.HasClusterIP = false
		} else {
			detail.HasClusterIP = true
			result.HasClusterIPServices++
			GinkgoWriter.Printf("ClusterIP: %s\n", detail.ClusterIP)
		}

		// 检查端口
		detail.Ports = actualService.Spec.Ports
		if len(detail.Ports) == 0 {
			GinkgoWriter.Printf("没有端口配置\n")
		} else {
			portInfo := make([]string, 0)
			for _, port := range detail.Ports {
				portStr := fmt.Sprintf("%s:%d", port.Name, port.Port)
				if port.TargetPort.Type == intstr.Int {
					portStr += fmt.Sprintf("->%d", port.TargetPort.IntVal)
				} else {
					portStr += fmt.Sprintf("->%s", port.TargetPort.StrVal)
				}
				portInfo = append(portInfo, portStr)
			}
			GinkgoWriter.Printf("端口: %s\n", strings.Join(portInfo, ", "))
		}

		result.StatusDetails[serviceKey] = detail
	}

	GinkgoWriter.Printf("\n=== Service状态检查完成 ===\n")
	return result
}

// 测试关键Service的连通性
func testCriticalServicesConnectivity(ctx context.Context, clientset kubernetes.Interface,
	executor *executor.SSHExecutor, testNamespace, testPod string) ServiceConnectivityResult {

	var result ServiceConnectivityResult
	result.ConnectionDetails = make(map[string]ConnectionDetail)

	// 定义关键Service（必须测试的）
	criticalServices := []ServiceExpectation{
		// Kubernetes核心服务
		KubernetesDefaultService, // API Server

		// 监控服务
		PrometheusK8sService,    // Prometheus
		AlertManagerMainService, // AlertManager

		// openfuyao核心服务
		ConsoleService,               // 控制台服务
		ApplicationManagementService, // 应用管理服务
	}

	GinkgoWriter.Printf("=== 开始关键Service连通性测试 (共%d个) ===\n", len(criticalServices))

	for _, service := range criticalServices {
		result.TestedServices++
		serviceKey := fmt.Sprintf("%s/%s", service.Namespace, service.Name)

		GinkgoWriter.Printf("\n测试Service: %s\n", serviceKey)

		var detail ConnectionDetail
		detail.ServiceName = serviceKey

		// 获取Service信息
		svc, err := clientset.CoreV1().Services(service.Namespace).Get(ctx, service.Name, metav1.GetOptions{})
		if err != nil {
			detail.Error = fmt.Sprintf("获取Service失败: %v", err)
			detail.IsAccessible = false
			result.FailedConnections++
			result.ConnectionDetails[serviceKey] = detail
			GinkgoWriter.Printf("%s\n", detail.Error)
			continue
		}

		// 如果没有ClusterIP或没有端口，跳过连通性测试
		if svc.Spec.ClusterIP == "" || svc.Spec.ClusterIP == "None" || len(svc.Spec.Ports) == 0 {
			detail.Error = "Service没有ClusterIP或端口"
			detail.IsAccessible = false
			result.FailedConnections++
			result.ConnectionDetails[serviceKey] = detail
			GinkgoWriter.Printf("%s\n", detail.Error)
			continue
		}

		// 测试连通性
		startTime := time.Now()
		testResult := testSingleServiceConnectivity(executor, service.Namespace, service.Name,
			svc, testNamespace, testPod)
		responseTime := time.Since(startTime)

		detail.StatusCode = testResult.StatusCode
		detail.ResponseTime = responseTime
		detail.Error = testResult.Error
		detail.IsAccessible = testResult.IsAccessible

		if testResult.IsAccessible {
			result.SuccessfulConnections++
			GinkgoWriter.Printf("连通成功 (HTTP %d, 耗时: %v)\n",
				testResult.StatusCode, responseTime.Round(time.Millisecond))
		} else {
			result.FailedConnections++
			GinkgoWriter.Printf("连通失败")
			if testResult.Error != "" {
				GinkgoWriter.Printf(": %s", testResult.Error)
			}
			if testResult.StatusCode > 0 {
				GinkgoWriter.Printf(" (HTTP %d)", testResult.StatusCode)
			}
			GinkgoWriter.Printf("\n")
		}

		result.ConnectionDetails[serviceKey] = detail
	}

	GinkgoWriter.Printf("\n=== 关键Service连通性测试完成 ===\n")
	return result
}

// 测试关键Service的连通性 - 支持根据isDaily选择执行器
func testCriticalServicesConnectivityWithExecutors(ctx context.Context, clientset *kubernetes.Clientset, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool,
	testNamespace, testPod string) ServiceConnectivityResult {

	var result ServiceConnectivityResult
	result.ConnectionDetails = make(map[string]ConnectionDetail)

	// 定义关键Service（必须测试的）
	criticalServices := []ServiceExpectation{
		// Kubernetes核心服务
		KubernetesDefaultService, // API Server

		// 监控服务
		PrometheusK8sService,    // Prometheus
		AlertManagerMainService, // AlertManager

		// openfuyao核心服务
		ConsoleService,               // 控制台服务
		ApplicationManagementService, // 应用管理服务
	}

	GinkgoWriter.Printf("=== 开始关键Service连通性测试 (共%d个) ===\n", len(criticalServices))

	for _, service := range criticalServices {
		result.TestedServices++
		serviceKey := fmt.Sprintf("%s/%s", service.Namespace, service.Name)

		GinkgoWriter.Printf("\n测试Service: %s\n", serviceKey)

		var detail ConnectionDetail
		detail.ServiceName = serviceKey

		// 获取Service信息
		svc, err := clientset.CoreV1().Services(service.Namespace).Get(ctx, service.Name, metav1.GetOptions{})
		if err != nil {
			detail.Error = fmt.Sprintf("获取Service失败: %v", err)
			detail.IsAccessible = false
			result.FailedConnections++
			result.ConnectionDetails[serviceKey] = detail
			GinkgoWriter.Printf("%s\n", detail.Error)
			continue
		}

		// 如果没有ClusterIP或没有端口，跳过连通性测试
		if svc.Spec.ClusterIP == "" || svc.Spec.ClusterIP == "None" || len(svc.Spec.Ports) == 0 {
			detail.Error = "Service没有ClusterIP或端口"
			detail.IsAccessible = false
			result.FailedConnections++
			result.ConnectionDetails[serviceKey] = detail
			GinkgoWriter.Printf("%s\n", detail.Error)
			continue
		}

		// 测试连通性
		startTime := time.Now()
		testResult := testSingleServiceConnectivityWithExecutors(sshExecutor, localExecutor, isDaily, service.Namespace, service.Name,
			svc, testNamespace, testPod)
		responseTime := time.Since(startTime)

		detail.StatusCode = testResult.StatusCode
		detail.ResponseTime = responseTime
		detail.Error = testResult.Error
		detail.IsAccessible = testResult.IsAccessible

		if testResult.IsAccessible {
			result.SuccessfulConnections++
			GinkgoWriter.Printf("连通成功 (HTTP %d, 耗时: %v)\n",
				testResult.StatusCode, responseTime.Round(time.Millisecond))
		} else {
			result.FailedConnections++
			GinkgoWriter.Printf("连通失败")
			if testResult.Error != "" {
				GinkgoWriter.Printf(": %s", testResult.Error)
			}
			if testResult.StatusCode > 0 {
				GinkgoWriter.Printf(" (HTTP %d)", testResult.StatusCode)
			}
			GinkgoWriter.Printf("\n")
		}

		result.ConnectionDetails[serviceKey] = detail
	}

	GinkgoWriter.Printf("\n=== 关键Service连通性测试完成 ===\n")
	return result
}

// 测试单个Service的连通性
type ServiceTestResult struct {
	StatusCode   int
	Error        string
	IsAccessible bool
}

// 测试单个Service的连通性 - 支持根据isDaily选择执行器
func testSingleServiceConnectivityWithExecutors(sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool, svcNamespace, svcName string,
	svc *corev1.Service, testNamespace, testPod string) ServiceTestResult {

	var result ServiceTestResult

	// 使用第一个端口进行测试
	port := svc.Spec.Ports[0].Port
	protocol := svc.Spec.Ports[0].Protocol

	// 构建访问URL
	var url string
	if protocol == corev1.ProtocolTCP {
		// 对于TCP服务，使用curl进行测试
		url = fmt.Sprintf("http://%s:%d/", svc.Spec.ClusterIP, port)
	} else {
		// 对于UDP服务，可能需要不同的测试方法
		result.Error = fmt.Sprintf("暂不支持%s协议的连通性测试", protocol)
		result.IsAccessible = false
		return result
	}

	// 在测试Pod中执行curl命令
	cmd := fmt.Sprintf("kubectl exec %s -n %s -- curl -s -o /dev/null -w '%%{http_code}' %s 2>&1",
		testPod, testNamespace, url)

	var execResult *executor.ExecResult
	var err error
	if isDaily && localExecutor != nil {
		execResult, err = localExecutor.Exec(cmd)
	} else if sshExecutor != nil {
		execResult, err = sshExecutor.Exec(cmd)
	}

	if err != nil {
		result.Error = fmt.Sprintf("执行curl命令失败: %v", err)
		result.IsAccessible = false
		return result
	}

	// 解析状态码
	stdout := strings.TrimSpace(execResult.Stdout)
	if stdout == "" {
		result.Error = "curl命令没有返回状态码"
		result.IsAccessible = false
		return result
	}

	// 尝试解析状态码
	statusCode, err := strconv.Atoi(stdout)
	if err != nil {
		// 可能有其他输出，尝试提取状态码
		result.Error = fmt.Sprintf("解析状态码失败: %v，输出: %s", err, stdout)
		result.IsAccessible = false
		return result
	}

	result.StatusCode = statusCode

	// 2xx和3xx状态码都认为是成功的
	if statusCode >= 200 && statusCode < 400 {
		result.IsAccessible = true
	} else {
		result.IsAccessible = false
		result.Error = fmt.Sprintf("HTTP状态码: %d", statusCode)
	}

	return result
}

func testSingleServiceConnectivity(executor *executor.SSHExecutor, svcNamespace, svcName string,
	svc *corev1.Service, testNamespace, testPod string) ServiceTestResult {

	var result ServiceTestResult

	// 使用第一个端口进行测试
	port := svc.Spec.Ports[0].Port

	// 确定协议
	protocol := "http"
	if port == 443 || port == 6443 || svcName == "kubernetes" {
		protocol = "https"
	}

	// 构建测试URL
	svcURL := fmt.Sprintf("%s://%s.%s.svc.cluster.local:%d",
		protocol, svcName, svcNamespace, port)

	// 根据Service类型调整超时时间
	timeout := 5
	if strings.Contains(svcName, "kubernetes") {
		timeout = 10
	}

	// 执行curl命令
	cmd := fmt.Sprintf("kubectl exec %s -n %s -- timeout %d curl -k -s -o /dev/null -w '%%{http_code}' --connect-timeout %d --max-time %d %s",
		testPod, testNamespace, timeout, timeout, timeout, svcURL)

	curlResult, err := executor.Exec(cmd)

	if err != nil {
		result.Error = fmt.Sprintf("执行命令失败: %v", err)
		return result
	}

	// 解析HTTP状态码
	statusCodeStr := strings.TrimSpace(curlResult.Stdout)
	if statusCodeStr == "" {
		result.Error = "无HTTP响应"
		return result
	}

	statusCode, err := strconv.Atoi(statusCodeStr)
	if err != nil {
		result.Error = fmt.Sprintf("无效的HTTP状态码: %s", statusCodeStr)
		return result
	}

	result.StatusCode = statusCode

	// 判断连通性
	// 2xx/3xx/4xx 表示服务可达（4xx是客户端错误，但服务在运行）
	// 5xx 表示服务有问题
	result.IsAccessible = (statusCode < 500)

	return result
}
