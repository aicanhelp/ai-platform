package system_integration

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

// 基础辅助函数
func CreateNamespace(ctx context.Context, clientset kubernetes.Interface, namespace string) {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	}
	_, err := clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建命名空间 %s 失败: %v\n", namespace, err)
	} else {
		GinkgoWriter.Printf("创建命名空间: %s\n", namespace)
	}
}

// 创建命名空间 - 支持K8SClient版本
func CreateNamespaceWithK8SClient(ctx context.Context, k8sClient *k8s.K8SClient, namespace string) {
	ns := &corev1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name: namespace,
		},
	}
	_, err := k8sClient.Clientset.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
	if err != nil {
		GinkgoWriter.Printf("创建命名空间 %s 失败: %v\n", namespace, err)
	} else {
		GinkgoWriter.Printf("创建命名空间: %s\n", namespace)
	}
}

func WaitForPodReady(ctx context.Context, clientset kubernetes.Interface, namespace, podName string) {
	Eventually(func() bool {
		pod, err := clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false
		}

		if pod.Status.Phase == corev1.PodRunning && IsPodReady(pod) {
			return true
		}

		GinkgoWriter.Printf("等待Pod就绪: %s/%s (状态: %s)\n", namespace, podName, pod.Status.Phase)
		return false
	}).WithTimeout(120*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
		fmt.Sprintf("Pod %s/%s 应该在120秒内就绪", namespace, podName))
}

// 等待Pod就绪 - 支持K8SClient版本
func WaitForPodReadyWithK8SClient(ctx context.Context, k8sClient *k8s.K8SClient, namespace, podName string) {
	Eventually(func() bool {
		pod, err := k8sClient.Clientset.CoreV1().Pods(namespace).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false
		}

		if pod.Status.Phase == corev1.PodRunning && IsPodReady(pod) {
			return true
		}

		GinkgoWriter.Printf("等待Pod就绪: %s/%s (状态: %s)\n", namespace, podName, pod.Status.Phase)
		return false
	}).WithTimeout(120*time.Second).WithPolling(5*time.Second).Should(BeTrue(),
		fmt.Sprintf("Pod %s/%s 应该在120秒内就绪", namespace, podName))
}

// 从kubeconfig字符串创建K8SClient
func CreateK8SClientFromKubeconfig(kubeconfigStr string) *k8s.K8SClient {
	// 为业务集群创建临时文件
	tmpFile, err := os.CreateTemp("", "kubeconfig-workload-*.yaml")
	Expect(err).NotTo(HaveOccurred(), "应该能够创建业务集群临时文件")
	defer os.Remove(tmpFile.Name())

	if _, err := tmpFile.WriteString(kubeconfigStr); err != nil {
		Fail(fmt.Sprintf("写入业务集群kubeconfig失败: %v", err))
	}
	tmpFile.Close()

	// 创建Kubernetes客户端配置
	config, err := clientcmd.BuildConfigFromFlags("", tmpFile.Name())
	Expect(err).NotTo(HaveOccurred(), "应该能够构建业务集群配置")

	// 创建clientset
	clientset, err := kubernetes.NewForConfig(config)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建业务集群clientset")

	// 创建dynamic client
	dynamicClient, err := dynamic.NewForConfig(config)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建业务集群dynamic client")

	return &k8s.K8SClient{
		RestConfig:    config,
		Clientset:     clientset,
		DynamicClient: dynamicClient,
	}
}

// 清理测试资源
func CleanupTestResources(ctx context.Context, clientset kubernetes.Interface, executor *executor.SSHExecutor, resources []ResourceInfo) {
	GinkgoWriter.Printf("开始清理测试资源 (共%d个)\n", len(resources))

	// 按类型分组清理，先清理Pod，再清理Service，最后清理Namespace
	var pods []ResourceInfo
	var services []ResourceInfo
	var namespaces []ResourceInfo

	for _, resource := range resources {
		switch resource.Type {
		case "pod":
			pods = append(pods, resource)
		case "service":
			services = append(services, resource)
		case "namespace":
			namespaces = append(namespaces, resource)
		}
	}

	// 清理Pod
	for _, pod := range pods {
		By(fmt.Sprintf("清理Pod: %s/%s", pod.Namespace, pod.Name))
		err := clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{
			GracePeriodSeconds: &[]int64{0}[0], // 立即删除
		})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Pod %s/%s 失败: %v\n", pod.Namespace, pod.Name, err)
			// 尝试强制删除
			forceDeleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --force --grace-period=0 2>/dev/null || true",
				pod.Name, pod.Namespace)
			executor.Exec(forceDeleteCmd)
		} else {
			GinkgoWriter.Printf("删除Pod: %s/%s\n", pod.Namespace, pod.Name)
		}
	}

	// 清理Service
	for _, service := range services {
		By(fmt.Sprintf("清理Service: %s/%s", service.Namespace, service.Name))
		err := clientset.CoreV1().Services(service.Namespace).Delete(ctx, service.Name, metav1.DeleteOptions{})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Service %s/%s 失败: %v\n", service.Namespace, service.Name, err)
			// 尝试使用kubectl删除
			deleteCmd := fmt.Sprintf("kubectl delete service %s -n %s 2>/dev/null || true",
				service.Name, service.Namespace)
			executor.Exec(deleteCmd)
		} else {
			GinkgoWriter.Printf("删除Service: %s/%s\n", service.Namespace, service.Name)
		}
	}

	// 等待Pod完全清理
	if len(pods) > 0 {
		GinkgoWriter.Printf("等待Pod完全清理...\n")
		time.Sleep(5 * time.Second)
	}

	// 清理Namespace（最后清理，因为会清理其中的所有资源）
	for _, ns := range namespaces {
		By(fmt.Sprintf("清理Namespace: %s", ns.Name))
		// 首先检查命名空间是否还存在
		_, err := clientset.CoreV1().Namespaces().Get(ctx, ns.Name, metav1.GetOptions{})
		if err == nil {
			// 命名空间还存在，删除它
			err = clientset.CoreV1().Namespaces().Delete(ctx, ns.Name, metav1.DeleteOptions{})
			if err != nil && !strings.Contains(err.Error(), "not found") {
				GinkgoWriter.Printf("删除Namespace %s 失败: %v\n", ns.Name, err)
				// 尝试使用kubectl强制删除
				forceDeleteCmd := fmt.Sprintf("kubectl delete namespace %s --force --grace-period=0 2>/dev/null || true", ns.Name)
				executor.Exec(forceDeleteCmd)
			} else {
				GinkgoWriter.Printf("删除Namespace: %s\n", ns.Name)
			}
		}
	}

	// 等待所有资源完全清理
	GinkgoWriter.Printf("等待资源清理完成...\n")
	time.Sleep(3 * time.Second)

	GinkgoWriter.Printf("资源清理完成\n")
}

// 清理测试资源 - 支持根据isDaily选择执行器
func CleanupTestResourcesWithExecutors(ctx context.Context, clientset kubernetes.Interface, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool, resources []ResourceInfo) {
	GinkgoWriter.Printf("开始清理测试资源 (共%d个)\n", len(resources))

	// 按类型分组清理，先清理Pod，再清理Service，最后清理Namespace
	var pods []ResourceInfo
	var services []ResourceInfo
	var namespaces []ResourceInfo

	for _, resource := range resources {
		switch resource.Type {
		case "pod":
			pods = append(pods, resource)
		case "service":
			services = append(services, resource)
		case "namespace":
			namespaces = append(namespaces, resource)
		}
	}

	// 清理Pod
	for _, pod := range pods {
		By(fmt.Sprintf("清理Pod: %s/%s", pod.Namespace, pod.Name))
		err := clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{
			GracePeriodSeconds: &[]int64{0}[0], // 立即删除
		})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Pod %s/%s 失败: %v\n", pod.Namespace, pod.Name, err)
			// 尝试强制删除
			forceDeleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --force --grace-period=0 2>/dev/null || true",
				pod.Name, pod.Namespace)
			// 根据isDaily选择执行器
			if isDaily && localExecutor != nil {
				localExecutor.Exec(forceDeleteCmd)
			} else if sshExecutor != nil {
				sshExecutor.Exec(forceDeleteCmd)
			}
		} else {
			GinkgoWriter.Printf("删除Pod: %s/%s\n", pod.Namespace, pod.Name)
		}
	}

	// 清理Service
	for _, service := range services {
		By(fmt.Sprintf("清理Service: %s/%s", service.Namespace, service.Name))
		err := clientset.CoreV1().Services(service.Namespace).Delete(ctx, service.Name, metav1.DeleteOptions{})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Service %s/%s 失败: %v\n", service.Namespace, service.Name, err)
			// 尝试使用kubectl删除
			deleteCmd := fmt.Sprintf("kubectl delete service %s -n %s 2>/dev/null || true",
				service.Name, service.Namespace)
			// 根据isDaily选择执行器
			if isDaily && localExecutor != nil {
				localExecutor.Exec(deleteCmd)
			} else if sshExecutor != nil {
				sshExecutor.Exec(deleteCmd)
			}
		} else {
			GinkgoWriter.Printf("删除Service: %s/%s\n", service.Namespace, service.Name)
		}
	}

	// 等待Pod完全清理
	if len(pods) > 0 {
		GinkgoWriter.Printf("等待Pod完全清理...\n")
		time.Sleep(5 * time.Second)
	}

	// 清理Namespace（最后清理，因为会清理其中的所有资源）
	for _, ns := range namespaces {
		By(fmt.Sprintf("清理Namespace: %s", ns.Name))
		// 首先检查命名空间是否还存在
		_, err := clientset.CoreV1().Namespaces().Get(ctx, ns.Name, metav1.GetOptions{})
		if err == nil {
			// 命名空间还存在，删除它
			err = clientset.CoreV1().Namespaces().Delete(ctx, ns.Name, metav1.DeleteOptions{})
			if err != nil && !strings.Contains(err.Error(), "not found") {
				GinkgoWriter.Printf("删除Namespace %s 失败: %v\n", ns.Name, err)
				// 尝试使用kubectl强制删除
				forceDeleteCmd := fmt.Sprintf("kubectl delete namespace %s --force --grace-period=0 2>/dev/null || true", ns.Name)
				// 根据isDaily选择执行器
				if isDaily && localExecutor != nil {
					localExecutor.Exec(forceDeleteCmd)
				} else if sshExecutor != nil {
					sshExecutor.Exec(forceDeleteCmd)
				}
			} else {
				GinkgoWriter.Printf("删除Namespace: %s\n", ns.Name)
			}
		}
	}

	// 等待所有资源完全清理
	GinkgoWriter.Printf("等待资源清理完成...\n")
	time.Sleep(3 * time.Second)

	GinkgoWriter.Printf("资源清理完成\n")
}

// 清理测试资源 - 支持K8SClient版本
func CleanupTestResourcesWithK8SClient(ctx context.Context, k8sClient *k8s.K8SClient, sshExecutor *executor.SSHExecutor, localExecutor *executor.LocalExecutor, isDaily bool, resources []ResourceInfo) {
	GinkgoWriter.Printf("开始清理测试资源 (共%d个)\n", len(resources))

	// 按类型分组清理，先清理Pod，再清理Service，最后清理Namespace
	var pods []ResourceInfo
	var services []ResourceInfo
	var namespaces []ResourceInfo

	for _, resource := range resources {
		switch resource.Type {
		case "pod":
			pods = append(pods, resource)
		case "service":
			services = append(services, resource)
		case "namespace":
			namespaces = append(namespaces, resource)
		}
	}

	// 清理Pod
	for _, pod := range pods {
		By(fmt.Sprintf("清理Pod: %s/%s", pod.Namespace, pod.Name))
		err := k8sClient.Clientset.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, metav1.DeleteOptions{
			GracePeriodSeconds: &[]int64{0}[0], // 立即删除
		})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Pod %s/%s 失败: %v\n", pod.Namespace, pod.Name, err)
			// 尝试强制删除
			forceDeleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --force --grace-period=0 2>/dev/null || true",
				pod.Name, pod.Namespace)
			// 根据isDaily选择执行器
			if isDaily && localExecutor != nil {
				localExecutor.Exec(forceDeleteCmd)
			} else if sshExecutor != nil {
				sshExecutor.Exec(forceDeleteCmd)
			}
		} else {
			GinkgoWriter.Printf("删除Pod: %s/%s\n", pod.Namespace, pod.Name)
		}
	}

	// 清理Service
	for _, service := range services {
		By(fmt.Sprintf("清理Service: %s/%s", service.Namespace, service.Name))
		err := k8sClient.Clientset.CoreV1().Services(service.Namespace).Delete(ctx, service.Name, metav1.DeleteOptions{})
		if err != nil && !strings.Contains(err.Error(), "not found") {
			GinkgoWriter.Printf("删除Service %s/%s 失败: %v\n", service.Namespace, service.Name, err)
			// 尝试使用kubectl删除
			deleteCmd := fmt.Sprintf("kubectl delete service %s -n %s 2>/dev/null || true",
				service.Name, service.Namespace)
			// 根据isDaily选择执行器
			if isDaily && localExecutor != nil {
				localExecutor.Exec(deleteCmd)
			} else if sshExecutor != nil {
				sshExecutor.Exec(deleteCmd)
			}
		} else {
			GinkgoWriter.Printf("删除Service: %s/%s\n", service.Namespace, service.Name)
		}
	}

	// 等待Pod完全清理
	if len(pods) > 0 {
		GinkgoWriter.Printf("等待Pod完全清理...\n")
		time.Sleep(5 * time.Second)
	}

	// 清理Namespace（最后清理，因为会清理其中的所有资源）
	for _, ns := range namespaces {
		By(fmt.Sprintf("清理Namespace: %s", ns.Name))
		// 首先检查命名空间是否还存在
		_, err := k8sClient.Clientset.CoreV1().Namespaces().Get(ctx, ns.Name, metav1.GetOptions{})
		if err == nil {
			// 命名空间还存在，删除它
			err = k8sClient.Clientset.CoreV1().Namespaces().Delete(ctx, ns.Name, metav1.DeleteOptions{})
			if err != nil && !strings.Contains(err.Error(), "not found") {
				GinkgoWriter.Printf("删除Namespace %s 失败: %v\n", ns.Name, err)
				// 尝试使用kubectl强制删除
				forceDeleteCmd := fmt.Sprintf("kubectl delete namespace %s --force --grace-period=0 2>/dev/null || true", ns.Name)
				// 根据isDaily选择执行器
				if isDaily && localExecutor != nil {
					localExecutor.Exec(forceDeleteCmd)
				} else if sshExecutor != nil {
					sshExecutor.Exec(forceDeleteCmd)
				}
			} else {
				GinkgoWriter.Printf("删除Namespace: %s\n", ns.Name)
			}
		}
	}

	// 等待所有资源完全清理
	GinkgoWriter.Printf("等待资源清理完成...\n")
	time.Sleep(3 * time.Second)

	GinkgoWriter.Printf("资源清理完成\n")
}

// 创建组件配置辅助函数（可自定义数量）
func CreateComponentWithCount(base ComponentExpectation, count int) ComponentExpectation {
	return ComponentExpectation{
		Name:           base.Name,
		Namespace:      base.Namespace,
		ExpectedCount:  count, // 使用自定义数量
		ExpectedImages: base.ExpectedImages,
	}
}

// IsPodReady 检查 Pod 是否就绪
func IsPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			return condition.Status == corev1.ConditionTrue
		}
	}
	return false
}

// VerifyClusterPods 通用方法：验证集群的所有Pod状态、名称和数量
func VerifyClusterPods(k8sClient *k8s.K8SClient, ctx context.Context, expectations map[string]ComponentExpectation) {
	totalComponents := len(expectations)
	totalPassed := 0
	totalFailed := 0

	GinkgoWriter.Printf("=== 开始Pod检查 (共%d个组件) ===\n", totalComponents)

	// 获取集群中所有命名空间的所有Pod
	allPods, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表失败")

	// 按组件名称分组实际Pod
	actualPodsByComponent := make(map[string][]corev1.Pod)
	for _, pod := range allPods.Items {
		// 查找匹配的组件名称
		matchedComponent := findMatchingComponent(pod.Name, expectations)
		if matchedComponent != "" {
			actualPodsByComponent[matchedComponent] = append(actualPodsByComponent[matchedComponent], pod)
		}
	}

	// 检查每个期望组件
	for componentName, expectation := range expectations {
		GinkgoWriter.Printf("\n组件检查: %s (命名空间: %s)\n", componentName, expectation.Namespace)

		// 获取该组件的实际Pod
		actualPods := actualPodsByComponent[componentName]
		actualCount := len(actualPods)
		expectedCount := expectation.ExpectedCount

		GinkgoWriter.Printf("  期望数量: %d, 实际数量: %d", expectedCount, actualCount)

		if actualCount == expectedCount {
			GinkgoWriter.Printf(" PASS\n")
			totalPassed++
		} else {
			GinkgoWriter.Printf(" FAIL(不匹配)\n")
			totalFailed++
			Expect(actualCount).To(Equal(expectedCount),
				fmt.Sprintf("组件 %s 的Pod数量应该是 %d", componentName, expectedCount))
		}

		// 检查每个Pod的状态
		for _, pod := range actualPods {
			isReady := IsPodReady(&pod)
			isRunning := pod.Status.Phase == corev1.PodRunning

			statusIcon := "PASS"
			if !isReady || !isRunning {
				statusIcon = "FAIL"
				totalFailed++
			} else {
				totalPassed++
			}

			GinkgoWriter.Printf("  Pod: %s - Running:%v Ready:%v %s\n",
				pod.Name, isRunning, isReady, statusIcon)

			if expectedCount > 0 {
				Expect(isRunning).To(BeTrue(),
					fmt.Sprintf("Pod %s/%s 应该是Running状态", pod.Namespace, pod.Name))
				Expect(isReady).To(BeTrue(),
					fmt.Sprintf("Pod %s/%s 应该是Ready状态", pod.Namespace, pod.Name))
			}
		}

		// 如果没有找到Pod但期望数量>0，记录失败
		if actualCount == 0 && expectedCount > 0 {
			GinkgoWriter.Printf("警告: 未找到组件 %s 的任何Pod\n", componentName)
			totalFailed++
			Expect(actualCount).To(BeNumerically(">=", expectedCount),
				fmt.Sprintf("组件 %s 应该有至少 %d 个Pod", componentName, expectedCount))
		}
	}

	// 汇总报告
	GinkgoWriter.Printf("\n=== Pod检查完成 ===\n")
	GinkgoWriter.Printf("总组件数: %d\n", totalComponents)
	GinkgoWriter.Printf("通过: %d\n", totalPassed)
	GinkgoWriter.Printf("失败: %d\n", totalFailed)
}

// findMatchingComponent 查找Pod名称匹配的组件
func findMatchingComponent(podName string, expectations map[string]ComponentExpectation) string {
	// 先尝试完全匹配组件名称
	for componentName, expectation := range expectations {
		// 组件名称 + 随机后缀（如：kube-apiserver-master-1）
		if strings.HasPrefix(podName, expectation.Name+"-") {
			return componentName
		}
	}

	// 再尝试包含匹配
	for componentName, expectation := range expectations {
		// 检查Pod名称是否包含组件名称
		if strings.Contains(podName, expectation.Name) {
			// 避免误匹配，例如"kube-proxy"匹配到"kube-proxy-daemonset"
			parts := strings.Split(podName, "-")
			for i := 0; i < len(parts)-1; i++ {
				possibleName := strings.Join(parts[:i+1], "-")
				if possibleName == expectation.Name {
					return componentName
				}
			}
		}
	}
	return ""
}
