package npu_operator

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	npu_config "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/npu-operator-config"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ==================== NPU-Operator 组件间集成测试 ====================

var _ = Describe("npu-operator 组件间集成测试", Ordered, Label("npu-operator-components", "npu", "system-integration", "with-workload-cluster"), func() {

	var (
		helmConfig map[string]*helm.ReleaseConfig
		sshClient  *executor.SSHExecutor
		k8sClient  *k8s.K8SClient
		ctx        context.Context
		cancel     context.CancelFunc
	)

	BeforeAll(func() {
		var err error

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		// 从环境变量获取SSH连接信息
		host := os.Getenv("TEST_NODE5_IP")
		port := 22
		if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
			if portInt, err := strconv.Atoi(portStr); err == nil {
				port = portInt
			}
		}
		user := os.Getenv("TEST_NODE5_USER")
		password := os.Getenv("TEST_NODE5_PASSWORD")

		sshClient, err = executor.NewSSHExecutor(
			host,
			port,
			user,
			password,
		)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")

		// k8s client
		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

		// load helm config from framework/helm
		helmConfig, err = helm.LoadHelmConfig()
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm config 应当成功加载")
	})

	AfterEach(func() {
		for _, eachRelease := range helmConfig {
			var needUninstall bool
			var err error
			needUninstall, err = helm.CheckHelmRelease(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "helm release check 应该成功")
			if needUninstall == true {
				err = helm.HelmUninstall(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("helm uninstall %s 应当执行成功", eachRelease.ReleaseName))
				time.Sleep(15 * time.Second)
				GinkgoWriter.Printf("%s uninstall success\n", eachRelease.ReleaseName)
			}
		}
	})

	It("npu-operator & kae-operator", func() {
		var err error

		// step1: kae-operator install
		GinkgoWriter.Println("=====step1: kae-operator install=====")
		tempReleaseName := helmConfig["kae-operator"].ReleaseName
		tempHelmUrl := helmConfig["kae-operator"].HelmURrl
		tempVersion := helmConfig["kae-operator"].Version
		tempNamespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("kae-operator install success")

		// step2: npu-operator install (expected to fail due to conflict)
		GinkgoWriter.Println("=====step2: npu-operator install (expected conflict)=====")
		releaseName := helmConfig["npu-operator"].ReleaseName
		helmUrl := helmConfig["npu-operator"].HelmURrl
		version := helmConfig["npu-operator"].Version
		namespace := helmConfig["npu-operator"].Namespace

		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println("Expected conflict error:", err)
		}
		Expect(err).To(HaveOccurred(), "npu-operator 安装应当失败，因为与 kae-operator 冲突")

		GinkgoWriter.Println("npu-operator & kae-operator conflict detection pass")
	})

	// additional compatibility/conflict tests based on helm_config.yaml components
	It("npu-operator & colocation-package", func() {
		var err error

		// step1: colocation-package install
		GinkgoWriter.Println("=====step1: colocation-package install=====")
		temp := helmConfig["colocation-package"]
		GinkgoWriter.Println("start install colocation-package...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("colocation-package install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & many-core-scheduler", func() {
		var err error

		// step1: many-core-scheduler install
		GinkgoWriter.Println("=====step1: many-core-scheduler install=====")
		temp := helmConfig["many-core-scheduler"]
		GinkgoWriter.Println("start install many-core-scheduler...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install many-core-scheduler 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("many-core-scheduler install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & numa-affinity-package", func() {
		var err error

		// step1: numa-affinity-package install
		GinkgoWriter.Println("=====step1: numa-affinity-package install=====")
		temp := helmConfig["numa-affinity-package"]
		GinkgoWriter.Println("start install numa-affinity-package...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("numa-affinity-package install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & logging-package", func() {
		var err error

		// step1: logging-package install
		GinkgoWriter.Println("=====step1: logging-package install=====")
		temp := helmConfig["logging-package"]
		GinkgoWriter.Println("start install logging-package...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("logging-package install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & multi-cluster-service", func() {
		var err error

		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		temp := helmConfig["multi-cluster-service"]
		GinkgoWriter.Println("start install multi-cluster-service...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("multi-cluster-service install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & monitoring-dashboard", func() {
		var err error

		// step1: monitoring-dashboard install
		GinkgoWriter.Println("=====step1: monitoring-dashboard install=====")
		temp := helmConfig["monitoring-dashboard"]
		GinkgoWriter.Println("start install monitoring-dashboard...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("monitoring-dashboard install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})

	It("npu-operator & ray-package", func() {
		var err error

		// step1: ray-package install
		GinkgoWriter.Println("=====step1: ray-package install=====")
		temp := helmConfig["ray-package"]
		GinkgoWriter.Println("start install ray-package...")
		err = helm.HelmInstall(temp.ReleaseName, temp.HelmURrl, temp.Version, temp.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install ray-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("ray-package install success")

		// step2: npu-operator install
		GinkgoWriter.Println("=====step2: npu-operator install=====")
		npu := helmConfig["npu-operator"]
		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(npu.ReleaseName, npu.HelmURrl, npu.Version, npu.Namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: npu-operator installation check
		GinkgoWriter.Println("=====step3: npu-operator installation check=====")
		verifyNpuOperatorPods(k8sClient, ctx)
		GinkgoWriter.Println("npu-operator installation check pass")
	})
})

var _ = Describe("NPU Operator System Integration", Label("npu_operator_system_integration", "npu", "system-integration", "with-workload-cluster"), func() {
	var (
		k8sClient *k8s.K8SClient
		ctx       context.Context
		cancel    context.CancelFunc
		sshClient *executor.SSHExecutor
	)

	BeforeEach(func() {
		var err error
		ctx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)

		// ssh client
		// 从环境变量获取SSH连接信息
		host := os.Getenv("TEST_NODE5_IP")
		port := 22
		if portStr := os.Getenv("TEST_NODE5_PORT"); portStr != "" {
			if portInt, err := strconv.Atoi(portStr); err == nil {
				port = portInt
			}
		}
		user := os.Getenv("TEST_NODE5_USER")
		password := os.Getenv("TEST_NODE5_PASSWORD")

		sshClient, err = executor.NewSSHExecutor(
			host,
			port,
			user,
			password,
		)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")

		// k8s client
		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")
	})

	AfterEach(func() {
		if cancel != nil {
			cancel()
		}
	})

	It("should have NPU operator related pods running and images match expectations", func() {
		//校验pod状态和镜像
		verifyNpuOperatorPods(k8sClient, ctx)
	})

	It("should have complete npu-exporter metrics with proper format", func() {
		// 测试npu-exporter metrics的完整性
		verifyNpuExporterMetricsCompleteness(k8sClient, sshClient, ctx)
	})

	It("should have inference-gateway service accessible and response contains valid metrics", func() {
		// 测试访问ai服务后metrics正确性
		verifyInferenceGatewayService(k8sClient, sshClient, ctx)
	})

	// 在 Describe 块中添加新的测试用例
	It("should automatically recover NPU operator components after pod deletion", func() {
		verifyNPUComponentAutoRecovery(k8sClient, ctx, sshClient)
	})
})

// verifyNPUComponentAutoRecovery 验证NPU组件在Pod删除后的自动恢复能力
func verifyNPUComponentAutoRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("开始验证NPU Operator组件的自愈能力...\n")

	// 记录测试开始时间
	startTime := time.Now()

	// 记录失败的组件
	var failedComponents []string

	// 统计成功恢复的组件数量
	successfullyRecovered := 0
	totalComponents := len(npu_config.ExpectedImages)

	if totalComponents == 0 {
		GinkgoWriter.Printf("警告: ExpectedImages配置为空，跳过自愈能力测试\n")
		return
	}

	// 获取组件名称列表
	var componentNames []string
	for componentName := range npu_config.ExpectedImages {
		componentNames = append(componentNames, componentName)
	}

	for i, componentName := range componentNames {
		GinkgoWriter.Printf("\n=== 开始测试组件 (%d/%d): %s ===\n", i+1, totalComponents, componentName)

		// 1. 首先找到组件的当前Pod
		originalPod := findComponentPodByName(k8sClient, ctx, componentName)
		if originalPod == nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (未找到Pod)", componentName))
			GinkgoWriter.Printf("未找到 %s 组件的Pod\n", componentName)
			continue
		}

		originalPodName := originalPod.Name
		namespace := originalPod.Namespace
		GinkgoWriter.Printf("找到 %s 组件的Pod: %s/%s (状态: %s)\n",
			componentName, namespace, originalPodName, originalPod.Status.Phase)

		// 2. 删除Pod
		deleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --wait=false --timeout=30s", originalPodName, namespace)
		deleteRes, err := sshClient.Exec(deleteCmd)
		if err != nil {
			failedComponents = append(failedComponents,
				fmt.Sprintf("%s (删除失败: %v, stderr: %s)", componentName, err,
					func() string {
						if deleteRes != nil {
							return deleteRes.Stderr
						}
						return ""
					}()))
			GinkgoWriter.Printf("删除 %s 组件的Pod时出错: %v\n", componentName, err)
			if deleteRes != nil {
				GinkgoWriter.Printf("删除命令输出: stdout=%s, stderr=%s\n", deleteRes.Stdout, deleteRes.Stderr)
			}
			continue
		}
		GinkgoWriter.Printf("删除命令执行成功: %s\n", strings.TrimSpace(deleteRes.Stdout))
	}

	time.Sleep(5 * time.Minute) // 等待一段时间，避免过快检查
	for _, componentName := range componentNames {
		// 1. 首先找到组件的当前Pod
		newPod := findComponentPodByName(k8sClient, ctx, componentName)
		if newPod == nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (未找到Pod)", componentName))
			GinkgoWriter.Printf("未找到 %s 组件的Pod\n", componentName)
			continue
		}

		newPodName := newPod.Name
		namespace := newPod.Namespace
		GinkgoWriter.Printf("找到 %s 组件的Pod: %s/%s (状态: %s)\n",
			componentName, namespace, newPodName, newPod.Status.Phase)

		if newPod.Status.Phase != corev1.PodRunning {
			failedComponents = append(failedComponents,
				fmt.Sprintf("%s (Pod状态: %s)", componentName, newPod.Status.Phase))
			GinkgoWriter.Printf("恢复的Pod状态不是Running: %s\n", newPod.Status.Phase)
			continue
		}
		successfullyRecovered++
	}

	// 测试完成统计
	duration := time.Since(startTime)
	GinkgoWriter.Printf("\n=== NPU组件自愈能力测试完成 ===\n")
	GinkgoWriter.Printf("测试总时长: %v\n", duration.Round(time.Second))
	GinkgoWriter.Printf("组件总数: %d\n", totalComponents)
	GinkgoWriter.Printf("成功恢复: %d\n", successfullyRecovered)
	GinkgoWriter.Printf("恢复失败: %d\n", len(failedComponents))

	if len(failedComponents) > 0 {
		GinkgoWriter.Printf("失败组件详情:\n")
		for i, comp := range failedComponents {
			GinkgoWriter.Printf("  %d. %s\n", i+1, comp)
		}
	}

	// 要求全部组件都能成功恢复
	Expect(successfullyRecovered).To(Equal(totalComponents),
		"所有NPU组件都应成功恢复。总组件数: %d, 成功恢复: %d, 失败组件: %v",
		totalComponents, successfullyRecovered, failedComponents)

	GinkgoWriter.Printf("✓ NPU Operator所有组件自愈能力验证通过！\n")
}

// findComponentPodByName 根据组件名称查找对应的Pod
func findComponentPodByName(k8sClient *k8s.K8SClient, ctx context.Context, componentName string) *corev1.Pod {
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	if err != nil {
		GinkgoWriter.Printf("获取Pod列表时出错: %v\n", err)
		return nil
	}

	return findMatchingComponentPod(componentName, podList.Items)
}

// waitForNPUPodRecovery 等待NPU Pod恢复
func waitForNPUPodRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor,
	componentName, namespace string, timeout time.Duration) bool {

	startTime := time.Now()
	checkInterval := 5 * time.Second

	GinkgoWriter.Printf("开始监控Pod恢复 (开始时间: %v)...\n", startTime.Format("15:04:05"))

	for time.Since(startTime) < timeout {
		elapsed := time.Since(startTime).Round(time.Second)

		// 检查是否有新的Pod被创建
		newPod := findComponentPodByName(k8sClient, ctx, componentName)

		if newPod != nil {
			newPodName := newPod.Name

			// 检查Pod状态
			switch newPod.Status.Phase {
			case corev1.PodRunning:
				// 检查就绪状态
				for _, condition := range newPod.Status.Conditions {
					if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
						GinkgoWriter.Printf("[%v] 找到恢复的Pod: %s (状态: %s, 已就绪)\n",
							elapsed, newPodName, newPod.Status.Phase)
						return true
					}
				}

				// 获取Pod事件用于调试
				if elapsed > 30*time.Second {
					eventsCmd := fmt.Sprintf("kubectl get events -n %s --field-selector involvedObject.name=%s --sort-by='.lastTimestamp' | tail -5",
						namespace, newPodName)
					eventsRes, _ := sshClient.Exec(eventsCmd)
					if eventsRes != nil && eventsRes.Stdout != "" {
						GinkgoWriter.Printf("最近事件: %s\n", eventsRes.Stdout)
					}
				}

			case corev1.PodPending:
				GinkgoWriter.Printf("[%v] Pod %s 处于Pending状态...\n", elapsed, newPodName)
				// 检查调度状态
				for _, condition := range newPod.Status.Conditions {
					if condition.Type == corev1.PodScheduled && condition.Status == corev1.ConditionFalse {
						GinkgoWriter.Printf("等待调度原因: %s\n", condition.Reason)
					}
				}

			case corev1.PodFailed:
				GinkgoWriter.Printf("[%v] Pod %s 处于Failed状态\n", elapsed, newPodName)
				// 获取失败原因
				logsCmd := fmt.Sprintf("kubectl logs %s -n %s --tail=10 2>&1 || true", newPodName, namespace)
				logsRes, _ := sshClient.Exec(logsCmd)
				if logsRes != nil && logsRes.Stdout != "" {
					GinkgoWriter.Printf("失败日志: %s\n", logsRes.Stdout)
				}
				return false

			default:
				GinkgoWriter.Printf("[%v] Pod %s 处于 %s 状态\n", elapsed, newPodName, newPod.Status.Phase)
			}

		} else {
			GinkgoWriter.Printf("[%v] 等待 %s 组件新Pod创建...\n", elapsed, componentName)
		}

		time.Sleep(checkInterval)
	}

	GinkgoWriter.Printf("[%v] %s 组件恢复超时\n", timeout, componentName)
	return false
}

func verifyNpuOperatorPods(k8sClient *k8s.K8SClient, ctx context.Context) {
	// 获取所有 Pod 作为候选集合
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有 pods 时不应出错")
	Expect(len(podList.Items)).To(BeNumerically(">", 0), "集群中应至少存在一个 pod")

	// 遍历所有期望的组件名称（来自 ExpectedImages 的 key）
	for componentName, expectedImgs := range npu_config.ExpectedImages {
		matchedPod := findMatchingComponentPod(componentName, podList.Items)
		if matchedPod == nil {
			GinkgoWriter.Printf("未找到与组件 %s 匹配的 pod\n", componentName)
			continue
		}

		// 检查 Pod 的状态为 Running
		Expect(matchedPod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s 应为 Running", matchedPod.Name)
		allExpectedImagesFound := true
		for _, expectedImage := range expectedImgs {
			// 校验容器镜像至少匹配一项期望值（支持部分匹配）
			podImages := getAllPodImages(matchedPod)
			GinkgoWriter.Printf("容器镜像: %v\n", podImages)
			if expectedImage == "" {
				// 空字符串表示通配符，总是匹配
				GinkgoWriter.Printf("期望镜像 [任意]: 跳过检查\n")
				continue
			}
			if contains(podImages, expectedImage) {
				GinkgoWriter.Printf("期望镜像 %s: 存在\n", expectedImage)
			} else {
				GinkgoWriter.Printf("期望镜像 %s: 不存在\n", expectedImage)
				allExpectedImagesFound = false
			}
		}
		if !allExpectedImagesFound {
			Expect(allExpectedImagesFound).To(BeTrue(),
				fmt.Sprintf("组件 %s 应该包含期望镜像", componentName))
		}
	}
}

func verifyNpuExporterMetricsCompleteness(k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, ctx context.Context) {
	// 先获取 metrics 数据
	metricsData := verifyNpuExporterService(k8sClient, sshClient, ctx)

	// 进行详细的 metrics 内容检查
	verifyMetricsContent(metricsData)
}

func verifyNpuExporterService(k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, ctx context.Context) string {
	// 使用共用方法获取metrics数据（从npu-exporter-service）
	metricsData := fetchMetricsData(sshClient, k8sClient, ctx, "npu-exporter-service")

	// 基础检查：确保包含一些预期的 Prometheus 指标格式
	Expect(metricsData).To(ContainSubstring("# TYPE"), "metrics 应该包含 Prometheus 类型定义")
	Expect(metricsData).To(ContainSubstring("# HELP"), "metrics 应该包含 Prometheus 帮助文本")

	return metricsData
}

// verifyMetricsContent 进行详细的 metrics 内容检查
func verifyMetricsContent(metricsData string) {
	GinkgoWriter.Println("开始检查 metrics 内容完整性...")

	// 按行分割 metrics 数据
	lines := strings.Split(metricsData, "\n")

	// 用于存储发现的指标类型和对应的数据行
	metricTypes := make(map[string][]string)
	currentMetricType := ""

	// 正则表达式匹配 TYPE 行和指标数据行
	typeRegex := regexp.MustCompile(`^#\s*TYPE\s+(\w+)\s+(\w+)`)
	metricRegex := regexp.MustCompile(`^(\w+)\{`)
	emptyLineRegex := regexp.MustCompile(`^\s*$`)

	// 解析 metrics 数据
	for _, line := range lines {
		if emptyLineRegex.MatchString(line) {
			continue // 跳过空行
		}

		// 检查是否是 TYPE 定义行
		if matches := typeRegex.FindStringSubmatch(line); matches != nil {
			metricName := matches[1]
			metricType := matches[2]
			currentMetricType = metricName
			metricTypes[currentMetricType] = []string{}
			GinkgoWriter.Printf("发现指标类型: %s (类型: %s)\n", metricName, metricType)
			continue
		}

		// 检查是否是具体的指标数据行
		if matches := metricRegex.FindStringSubmatch(line); matches != nil {
			metricName := matches[1]
			if currentMetricType == metricName {
				// 这是当前指标类型的数据行
				metricTypes[metricName] = append(metricTypes[metricName], line)
			} else {
				// 这可能是一个没有 TYPE 定义的指标，或者是新的指标类型
				if _, exists := metricTypes[metricName]; !exists {
					metricTypes[metricName] = []string{line}
					GinkgoWriter.Printf("发现未声明的指标数据: %s\n", metricName)
				}
			}
		}

		// 检查machine_npu_nums特殊处理
		if strings.HasPrefix(strings.TrimSpace(line), "machine_npu_nums ") {
			metricName := "machine_npu_nums"
			if _, exists := metricTypes[metricName]; !exists {
				metricTypes[metricName] = []string{}
			}
			metricTypes[metricName] = append(metricTypes[metricName], line)
			GinkgoWriter.Printf("发现无标签指标: %s\n", metricName)
		}
	}

	// 检查每个指标类型是否有对应的数据行
	allMetricsHaveData := true
	for metricName, dataLines := range metricTypes {
		if len(dataLines) == 0 {
			GinkgoWriter.Printf("警告: 指标类型 %s 有定义但没有数据行\n", metricName)
			allMetricsHaveData = false
		} else {
			GinkgoWriter.Printf("指标 %s 有 %d 行数据\n", metricName, len(dataLines))
		}
	}

	// 统计指标数量
	totalMetrics := len(metricTypes)
	metricsWithData := 0
	for _, dataLines := range metricTypes {
		if len(dataLines) > 0 {
			metricsWithData++
		}
	}

	GinkgoWriter.Printf("Metrics 检查完成: 共发现 %d 个指标类型，其中 %d 个有数据行\n",
		totalMetrics, metricsWithData)

	// 要求所有声明的指标类型都有数据行
	Expect(allMetricsHaveData).To(BeTrue(), "所有声明的指标类型都应该有对应的数据行")
}

// findMatchingComponentPod 根据组件名称查找匹配的 Pod
// 采用两层匹配策略：先完全匹配（组件名称 + 连接符），再包含匹配
func findMatchingComponentPod(componentName string, pods []corev1.Pod) *corev1.Pod {
	// 策略1：完全匹配 - 查找 "componentName-*" 形式的 Pod
	for i := range pods {
		if strings.HasPrefix(pods[i].Name, componentName+"-") {
			return &pods[i]
		}
	}

	// 策略2：包含匹配 - 查找名称中包含组件名称的 Pod
	for i := range pods {
		if strings.Contains(pods[i].Name, componentName) {
			// 避免误匹配，例如 "kube-proxy" 不应该匹配 "kube-proxy-daemonset"
			parts := strings.Split(pods[i].Name, "-")
			for j := 0; j < len(parts); j++ {
				possibleName := strings.Join(parts[:j+1], "-")
				if possibleName == componentName {
					return &pods[i]
				}
			}
		}
	}

	return nil
}

// 获取Pod中的所有容器镜像
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

func verifyInferenceGatewayService(k8sClient *k8s.K8SClient, sshClient *executor.SSHExecutor, ctx context.Context) {
	// 获取所有命名空间中的服务
	svcList, err := k8sClient.Clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有服务时不应出错")

	// 查找名为 inference-gateway 的服务
	var inferenceGatewayService *corev1.Service
	for i, svc := range svcList.Items {
		if strings.Contains(svc.Name, "inference-gateway") {
			inferenceGatewayService = &svcList.Items[i]
			break
		}
	}

	Expect(inferenceGatewayService).NotTo(BeNil(), "应该找到 inference-gateway服务")

	// 获取服务的 ClusterIP 和端口
	clusterIP := inferenceGatewayService.Spec.ClusterIP
	Expect(clusterIP).NotTo(BeEmpty(), "inference-gateway 应该有 ClusterIP")
	Expect(clusterIP).NotTo(Equal("None"), "inference-gateway 应该不是 Headless Service")

	// 查找端口（优先使用80端口，否则使用第一个可用端口）
	var servicePort int32
	for _, port := range inferenceGatewayService.Spec.Ports {
		if port.Port == 80 {
			servicePort = port.Port
			break
		}
	}
	// 如果没有找到80端口，使用第一个端口
	if servicePort == 0 && len(inferenceGatewayService.Spec.Ports) > 0 {
		servicePort = inferenceGatewayService.Spec.Ports[0].Port
	}
	Expect(servicePort).NotTo(BeZero(), "应该找到 inference-gateway 的端口")

	GinkgoWriter.Printf("找到 inference-gateway: ClusterIP=%s, Port=%d, Namespace=%s\n",
		clusterIP, servicePort, inferenceGatewayService.Namespace)

	// 构建curl命令测试服务访问
	testAIServiceAccess(sshClient, k8sClient, ctx, clusterIP, servicePort)
}

func testAIServiceAccess(sshClient *executor.SSHExecutor, k8sClient *k8s.K8SClient, ctx context.Context, clusterIP string, servicePort int32) {
	// 构建curl命令
	curlCmd := fmt.Sprintf(`curl -X POST http://%s:%d/v1/chat/completions \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer dummy-key" \
     -d '{
           "model": "Qwen/Qwen3-8B",
           "messages": [{"role": "user", "content": "介绍一下北京的景点"}],
           "stream": false
         }' -s -f --connect-timeout 30`, clusterIP, servicePort)

	// 执行curl命令
	_, err := sshClient.Exec(curlCmd)
	Expect(err).NotTo(HaveOccurred(), "应该能够通过 curl 访问 inference-gateway 服务")

	// 访问metrics端点检查日志
	checkMetricsForContainerInfo(sshClient, k8sClient, ctx)
}

// checkMetricsForContainerInfo 检查metrics日志中是否包含所需的container信息
func checkMetricsForContainerInfo(sshClient *executor.SSHExecutor, k8sClient *k8s.K8SClient, ctx context.Context) {
	// 获取metrics数据（从inference-gateway服务）
	metricsData := fetchMetricsData(sshClient, k8sClient, ctx, "npu-exporter-service")

	// 解析并检查metrics中是否包含所需的信息
	verifyMetricsContainerInfo(metricsData)
}

// verifyMetricsContainerInfo 验证metrics中是否包含container信息
func verifyMetricsContainerInfo(metricsData string) {
	// 定义需要检查的字段
	requiredFields := []string{
		`container_name="`,
		`model_name="`,
		`namespace="`,
	}

	fieldsFound := make(map[string]bool)
	for _, field := range requiredFields {
		fieldsFound[field] = false
	}

	// 按行检查metrics数据
	lines := strings.Split(metricsData, "\n")
	for _, line := range lines {
		// 跳过注释行和空行
		if strings.HasPrefix(line, "#") || strings.TrimSpace(line) == "" {
			continue
		}

		// 检查是否包含所有需要的字段
		for _, field := range requiredFields {
			if strings.Contains(line, field) {
				// 检查字段是否有值（不是空字符串）
				pattern := fmt.Sprintf(`%s([^"]+)"`, strings.TrimSuffix(field, "="))
				re := regexp.MustCompile(pattern)
				matches := re.FindStringSubmatch(line)

				if len(matches) > 1 && matches[1] != "" {
					fieldsFound[field] = true
					GinkgoWriter.Printf("发现有效字段 %s: 值=%s\n",
						strings.TrimSuffix(strings.TrimSuffix(field, "="), `"`), matches[1])
				}
			}
		}

		// 检查是否已经找到所有字段
		allFound := true
		for _, found := range fieldsFound {
			if !found {
				allFound = false
				break
			}
		}
		if allFound {
			break
		}
	}

	// 输出检查结果
	GinkgoWriter.Println("Container信息检查结果:")
	for field, found := range fieldsFound {
		fieldName := strings.TrimSuffix(strings.TrimSuffix(field, "="), `"`)
		if found {
			GinkgoWriter.Printf("%s: 存在有效值\n", fieldName)
		} else {
			GinkgoWriter.Printf("%s: 未找到或值为空\n", fieldName)
		}
	}

	// 验证所有字段都找到了
	for field, found := range fieldsFound {
		Expect(found).To(BeTrue(),
			fmt.Sprintf("metrics中应该包含有效的%s字段",
				strings.TrimSuffix(strings.TrimSuffix(field, "="), `"`)))
	}
}

// fetchMetricsData 通过SSH从指定的服务获取metrics数据
// 包括查找服务、获取ClusterIP和端口、执行curl命令等完整逻辑
func fetchMetricsData(sshClient *executor.SSHExecutor, k8sClient *k8s.K8SClient, ctx context.Context, serviceName string) string {
	// 获取所有命名空间中的服务
	svcList, err := k8sClient.Clientset.CoreV1().Services("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有服务时不应出错")

	// 查找指定名称的服务
	var targetService *corev1.Service
	for i, svc := range svcList.Items {
		if svc.Name == serviceName {
			targetService = &svcList.Items[i]
			break
		}
	}

	Expect(targetService).NotTo(BeNil(), fmt.Sprintf("应该找到 %s 服务", serviceName))

	// 获取服务的 ClusterIP 和端口
	clusterIP := targetService.Spec.ClusterIP
	Expect(clusterIP).NotTo(BeEmpty(), fmt.Sprintf("%s 应该有 ClusterIP", serviceName))
	Expect(clusterIP).NotTo(Equal("None"), fmt.Sprintf("%s 应该不是 Headless Service", serviceName))

	// 查找端口
	var servicePort int32
	for _, port := range targetService.Spec.Ports {
		if port.Name == "metrics" || port.Port == 8082 || port.Port == 80 {
			servicePort = port.Port
			break
		}
	}
	// 如果没有找到特定端口，使用第一个端口
	if servicePort == 0 && len(targetService.Spec.Ports) > 0 {
		servicePort = targetService.Spec.Ports[0].Port
	}
	Expect(servicePort).NotTo(BeZero(), fmt.Sprintf("应该找到 %s 的端口", serviceName))

	GinkgoWriter.Printf("找到 %s: ClusterIP=%s, Port=%d\n", serviceName, clusterIP, servicePort)

	// 通过 SSH 在集群节点上执行 curl 命令获取metrics
	metricsData := fetchMetricsFromEndpoint(sshClient, clusterIP, servicePort)

	// 检查 curl 命令的输出
	Expect(metricsData).NotTo(BeEmpty(), "metrics 端点应该返回数据")

	GinkgoWriter.Printf("成功获取 metrics 数据，响应大小: %d 字节\n", len(metricsData))

	// 输出部分 metrics 内容用于调试
	if len(metricsData) > 500 {
		GinkgoWriter.Printf("Metrics 前500字符: %s\n", metricsData[:500])
	} else {
		GinkgoWriter.Printf("Metrics 内容: %s\n", metricsData)
	}

	return metricsData
}

// fetchMetricsFromEndpoint 从指定的 ClusterIP:Port 通过SSH获取metrics数据
// 先尝试标准的/metrics路径，如果失败则尝试其他常见的metrics端点路径
func fetchMetricsFromEndpoint(sshClient *executor.SSHExecutor, clusterIP string, servicePort int32) string {
	// 首先尝试标准metrics路径
	metricsCmd := fmt.Sprintf("curl -s -f --connect-timeout 30 http://%s:%d/metrics", clusterIP, servicePort)
	res, err := sshClient.Exec(metricsCmd)

	if err == nil && res != nil && res.Stdout != "" {
		GinkgoWriter.Printf("使用标准路径 /metrics 成功获取metrics\n")
		return res.Stdout
	}

	// 如果访问失败，尝试其他常见metrics路径
	alternativePaths := []string{
		"/actuator/prometheus",
		"/actuator/metrics",
		"/metrics/prometheus",
		"/prometheus/metrics",
	}

	for _, path := range alternativePaths {
		metricsCmd = fmt.Sprintf("curl -s -f --connect-timeout 30 http://%s:%d%s", clusterIP, servicePort, path)
		res, err = sshClient.Exec(metricsCmd)
		if err == nil && res != nil && res.Stdout != "" {
			GinkgoWriter.Printf("使用备选路径 %s 成功获取metrics\n", path)
			return res.Stdout
		}
	}

	// 所有尝试都失败了，返回空字符串
	GinkgoWriter.Printf("警告: 无法通过任何路径访问 metrics 端点 http://%s:%d\n", clusterIP, servicePort)
	return ""
}
