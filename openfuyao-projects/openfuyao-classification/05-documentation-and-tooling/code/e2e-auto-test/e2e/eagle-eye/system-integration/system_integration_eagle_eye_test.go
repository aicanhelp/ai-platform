package eagle_eye

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ==================== Eagle-Eye 组件间集成测试 ====================

var _ = Describe("eagle-eye 组件间集成测试", Ordered, Label("eagle-eye-components", "npu", "with-workload-cluster"), func() {

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
				time.Sleep(30 * time.Second)
				Eventually(
					func() bool {
						status, err := helm.HelmUninstallNamespaceCheck(eachRelease.ReleaseName, k8sClient, ctx)
						if err != nil {
							GinkgoWriter.Println(err)
						}
						return status
					}, 3*time.Minute, 1*time.Second,
				).Should(BeTrue(), "namespace deletion should be completed")
				GinkgoWriter.Printf("%s uninstall success\n", eachRelease.ReleaseName)
			}
		}
	})

	It("eagle-eye & logging-package", func() {
		var err error

		// step1: logging-package install
		GinkgoWriter.Println("=====step1: logging-package install=====")
		tempReleaseName := helmConfig["logging-package"].ReleaseName
		tempHelmUrl := helmConfig["logging-package"].HelmURrl
		tempVersion := helmConfig["logging-package"].Version
		tempNamespace := helmConfig["logging-package"].Namespace

		GinkgoWriter.Println("start install logging-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("logging-package install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & colocation-package", func() {
		var err error

		// step1: colocation-package install
		GinkgoWriter.Println("=====step1: colocation-package install=====")
		tempReleaseName := helmConfig["colocation-package"].ReleaseName
		tempHelmUrl := helmConfig["colocation-package"].HelmURrl
		tempVersion := helmConfig["colocation-package"].Version
		tempNamespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("colocation-package install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & many-core-scheduler", func() {
		var err error

		// step1: many-core-scheduler install
		GinkgoWriter.Println("=====step1: many-core-scheduler install=====")
		tempReleaseName := helmConfig["many-core-scheduler"].ReleaseName
		tempHelmUrl := helmConfig["many-core-scheduler"].HelmURrl
		tempVersion := helmConfig["many-core-scheduler"].Version
		tempNamespace := helmConfig["many-core-scheduler"].Namespace

		GinkgoWriter.Println("start install many-core-scheduler...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install many-core-scheduler 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("many-core-scheduler install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & numa-affinity-package", func() {
		var err error

		// step1: numa-affinity-package install
		GinkgoWriter.Println("=====step1: numa-affinity-package install=====")
		tempReleaseName := helmConfig["numa-affinity-package"].ReleaseName
		tempHelmUrl := helmConfig["numa-affinity-package"].HelmURrl
		tempVersion := helmConfig["numa-affinity-package"].Version
		tempNamespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("numa-affinity-package install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & multi-cluster-service", func() {
		var err error

		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		tempReleaseName := helmConfig["multi-cluster-service"].ReleaseName
		tempHelmUrl := helmConfig["multi-cluster-service"].HelmURrl
		tempVersion := helmConfig["multi-cluster-service"].Version
		tempNamespace := helmConfig["multi-cluster-service"].Namespace

		GinkgoWriter.Println("start install multi-cluster-service...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("multi-cluster-service install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & ray-package", func() {
		var err error

		// step1: ray-package install
		GinkgoWriter.Println("=====step1: ray-package install=====")
		tempReleaseName := helmConfig["ray-package"].ReleaseName
		tempHelmUrl := helmConfig["ray-package"].HelmURrl
		tempVersion := helmConfig["ray-package"].Version
		tempNamespace := helmConfig["ray-package"].Namespace

		GinkgoWriter.Println("start install ray-package...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install ray-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("ray-package install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & kae-operator", func() {
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

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & npu-operator", func() {
		var err error

		// step1: npu-operator install
		GinkgoWriter.Println("=====step1: npu-operator install=====")
		tempReleaseName := helmConfig["npu-operator"].ReleaseName
		tempHelmUrl := helmConfig["npu-operator"].HelmURrl
		tempVersion := helmConfig["npu-operator"].Version
		tempNamespace := helmConfig["npu-operator"].Namespace

		GinkgoWriter.Println("start install npu-operator...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("npu-operator install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & infernex", func() {
		var err error

		// step1: infernex install
		GinkgoWriter.Println("=====step1: infernex install=====")
		tempReleaseName := helmConfig["infernex"].ReleaseName
		tempHelmUrl := helmConfig["infernex"].HelmURrl
		tempVersion := helmConfig["infernex"].Version
		tempNamespace := helmConfig["infernex"].Namespace

		GinkgoWriter.Println("start install infernex...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("infernex install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})

	It("eagle-eye & monitoring-dashboard", func() {
		var err error

		// step1: monitoring-dashboard install
		GinkgoWriter.Println("=====step1: monitoring-dashboard install=====")
		tempReleaseName := helmConfig["monitoring-dashboard"].ReleaseName
		tempHelmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		tempVersion := helmConfig["monitoring-dashboard"].Version
		tempNamespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("monitoring-dashboard install success")

		// step2: eagle-eye install
		GinkgoWriter.Println("=====step2: eagle-eye install=====")
		releaseName := helmConfig["eagle-eye"].ReleaseName
		helmUrl := helmConfig["eagle-eye"].HelmURrl
		version := helmConfig["eagle-eye"].Version
		namespace := helmConfig["eagle-eye"].Namespace

		GinkgoWriter.Println("start install eagle-eye...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install eagle-eye 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: eagle-eye installation check
		GinkgoWriter.Println("=====step3: eagle-eye installation check=====")
		verifyEagleEyePods(k8sClient, ctx)
		GinkgoWriter.Println("eagle-eye installation check pass")
	})
})

var _ = Describe("Eagle-Eye System Integration", Label("eagle_eye_system_integration", "npu", "with-workload-cluster"), func() {
	var (
		k8sClient *k8s.K8SClient
		ctx       context.Context
		cancel    context.CancelFunc
		sshClient *executor.SSHExecutor
	)

	BeforeEach(func() {
		var err error
		ctx, cancel = context.WithTimeout(context.Background(), 10*time.Minute)

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
		Expect(err).NotTo(HaveOccurred(), "应该能够使用 SSH 连接到 Eagle-Eye 测试节点")

		// 创建 k8s 客户端
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

	It("should have Eagle-Eye related pods running and images match expectations", func() {
		// 校验Eagle-Eye相关pod的状态和镜像
		verifyEagleEyePods(k8sClient, ctx)
	})

	It("should have NATS service running and logs show readiness", func() {
		// 校验NATS服务是否运行且日志显示就绪
		verifyNATSServiceReadiness(sshClient)
	})

	It("should have hardware-diagnosis service running and logs show events detection", func() {
		// 校验hardware-diagnosis服务是否运行且日志显示事件检测
		verifyHardwareDiagnosisService(k8sClient, ctx, sshClient)
	})

	It("should have hardware-monitor service running and logs show hardware metrics collection", func() {
		// 校验hardware-monitor服务是否运行且日志显示硬件指标收集
		verifyHardwareMonitorService(k8sClient, ctx, sshClient)
	})

	It("should deploy subscriber and publisher to verify NATS functionality", func() {
		// 部署订阅方和发布方检查nats功能
		verifyNATSFunctionality(sshClient, k8sClient, ctx)
	})

	It("should automatically recover Eagle-Eye components after pod deletion", func() {
		// 按照ExpectedImages中的componentName，依次用kubectl delete pod的方式删除，然后检查pod在2分钟内是否恢复
		verifyComponentAutoRecovery(k8sClient, ctx, sshClient)
	})
})

// verifyComponentAutoRecovery 验证组件在Pod删除后的自动恢复能力
func verifyComponentAutoRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("开始验证Eagle-Eye组件的自愈能力...\n")

	// 记录测试开始时间
	startTime := time.Now()

	// 记录失败的组件
	var failedComponents []string

	// 统计成功恢复的组件数量
	successfullyRecovered := 0
	totalComponents := len(ExpectedImages)

	for componentName := range ExpectedImages {
		GinkgoWriter.Printf("\n=== 开始测试组件: %s ===\n", componentName)

		// 1. 首先找到组件的当前Pod
		originalPod := findComponentPodByName(k8sClient, ctx, componentName)
		if originalPod == nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (未找到Pod)", componentName))
			GinkgoWriter.Printf("未找到 %s 组件的Pod\n", componentName)
			continue
		}

		originalPodName := originalPod.Name
		namespace := originalPod.Namespace
		GinkgoWriter.Printf("找到 %s 组件的Pod: %s/%s\n", componentName, namespace, originalPodName)

		// 2. 删除Pod
		deleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --wait=false", originalPodName, namespace)
		deleteRes, err := sshClient.Exec(deleteCmd)
		if err != nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (删除失败: %v)", componentName, err))
			GinkgoWriter.Printf("删除 %s 组件的Pod时出错: %v\n", componentName, err)
			continue
		}
		GinkgoWriter.Printf("删除命令执行结果: %s\n", deleteRes.Stdout)

		// 3. 等待并验证Pod恢复
		GinkgoWriter.Printf("等待 %s 组件Pod恢复...\n", componentName)
		recovered := waitForPodRecovery(k8sClient, ctx, sshClient, componentName, namespace, 3*time.Minute)

		if recovered {
			successfullyRecovered++
			GinkgoWriter.Printf("%s 组件成功在3分钟内恢复\n", componentName)

			// 4. 验证新Pod的状态
			newPod := findComponentPodByName(k8sClient, ctx, componentName)
			if newPod == nil {
				failedComponents = append(failedComponents, fmt.Sprintf("%s (恢复后未找到Pod)", componentName))
				GinkgoWriter.Printf("恢复后未找到 %s 组件Pod\n", componentName)
				continue
			}
			if newPod.Status.Phase != corev1.PodRunning {
				failedComponents = append(failedComponents, fmt.Sprintf("%s (Pod状态: %s)", componentName, newPod.Status.Phase))
				GinkgoWriter.Printf("恢复的Pod状态不是Running: %s\n", newPod.Status.Phase)
				continue
			}
		} else {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (3分钟内未恢复)", componentName))
			GinkgoWriter.Printf("%s 组件未在3分钟内恢复\n", componentName)
		}

		// 7. 短暂等待，避免同时删除多个Pod造成压力
		if componentName != getLastComponentName(ExpectedImages) {
			GinkgoWriter.Printf("等待10秒后测试下一个组件...\n")
			time.Sleep(10 * time.Second)
		}
	}

	// 测试完成统计
	duration := time.Since(startTime)
	GinkgoWriter.Printf("\n=== 自愈能力测试完成 ===\n")
	GinkgoWriter.Printf("测试总时长: %v\n", duration)
	GinkgoWriter.Printf("组件总数: %d\n", totalComponents)
	GinkgoWriter.Printf("成功恢复: %d\n", successfullyRecovered)

	if len(failedComponents) > 0 {
		GinkgoWriter.Printf("失败组件:\n")
		for _, comp := range failedComponents {
			GinkgoWriter.Printf("  - %s\n", comp)
		}
	}

	// 要求全部组件都能成功恢复
	Expect(successfullyRecovered).To(Equal(totalComponents),
		"所有组件都应成功恢复。失败组件: %v", failedComponents)

	GinkgoWriter.Printf("Eagle-Eye组件自愈能力验证通过\n")
}

// findComponentPodByName 根据组件名称查找对应的Pod
func findComponentPodByName(k8sClient *k8s.K8SClient, ctx context.Context, componentName string) *corev1.Pod {
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表时不应出错")

	return findMatchingPod(componentName, podList.Items)
}

// waitForPodRecovery 等待Pod恢复
func waitForPodRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor,
	componentName, namespace string, timeout time.Duration) bool {

	startTime := time.Now()
	checkInterval := 5 * time.Second

	for time.Since(startTime) < timeout {
		// 检查是否有新的Pod被创建
		newPod := findComponentPodByName(k8sClient, ctx, componentName)

		if newPod != nil {
			// 找到新Pod，检查其状态
			if newPod.Status.Phase == corev1.PodRunning {
				// 检查所有容器是否就绪
				for _, condition := range newPod.Status.Conditions {
					if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
						GinkgoWriter.Printf("找到恢复的Pod: %s (状态: %s)\n", newPod.Name, newPod.Status.Phase)
						return true
					}
				}
			} else if newPod.Status.Phase == corev1.PodPending {
				GinkgoWriter.Printf("新Pod %s 处于Pending状态，等待调度...\n", newPod.Name)
			} else if newPod.Status.Phase == corev1.PodFailed {
				GinkgoWriter.Printf("新Pod %s 处于Failed状态\n", newPod.Name)
				return false
			}
		} else {
			GinkgoWriter.Printf("等待 %s 组件新Pod创建...\n", componentName)
		}

		time.Sleep(checkInterval)
	}

	return false
}

// getLastComponentName 获取最后一个组件名称
func getLastComponentName(components map[string][]string) string {
	lastKey := ""
	for key := range components {
		lastKey = key
	}
	return lastKey
}

// verifyNATSFunctionality 验证NATS功能：部署订阅方和发布方
func verifyNATSFunctionality(sshClient *executor.SSHExecutor, k8sClient *k8s.K8SClient, ctx context.Context) {
	GinkgoWriter.Printf("开始验证NATS功能：部署订阅方和发布方...\n")

	// 1. 查找eagle-eye目录
	baseDir := findEagleEyeDirectory(sshClient)
	Expect(baseDir).NotTo(BeEmpty(), "应该能够找到eagle-eye目录")
	GinkgoWriter.Printf("找到eagle-eye目录: %s\n", baseDir)

	// 2. 检查src目录和requirements.txt是否存在
	cmdCheck := fmt.Sprintf("ls -la %s/src/ %s/requirements.txt 2>/dev/null || echo 'File not found'", baseDir, baseDir)
	resCheck, err := sshClient.Exec(cmdCheck)
	Expect(err).NotTo(HaveOccurred(), "应该能够检查文件是否存在")
	GinkgoWriter.Printf("文件检查结果: %s\n", resCheck.Stdout)

	// 3. 打包源码和配置
	tarCmd := fmt.Sprintf(`
cd %s && set -e
if [ -d "src" ] && [ -f "requirements.txt" ]; then
  tar -czf source_code.tar.gz src/ requirements.txt
  echo "打包成功"
else
  echo "找不到src目录或requirements.txt文件"
  exit 1
fi`, baseDir)

	res2, err := sshClient.Exec(tarCmd)
	Expect(err).NotTo(HaveOccurred(), "应该能够打包源码和配置文件")
	GinkgoWriter.Printf("打包结果: %s\n", res2.Stdout)
	Expect(strings.Contains(res2.Stdout, "打包成功")).To(BeTrue(), "打包应该成功")

	// 4. 检查打包文件是否创建成功
	cmdCheckTar := fmt.Sprintf("ls -la %s/source_code.tar.gz 2>/dev/null || echo 'File not found'", baseDir)
	resCheckTar, err := sshClient.Exec(cmdCheckTar)
	Expect(err).NotTo(HaveOccurred(), "应该能够检查打包文件")
	GinkgoWriter.Printf("打包文件检查: %s\n", resCheckTar.Stdout)
	Expect(strings.Contains(resCheckTar.Stdout, "source_code.tar.gz")).To(BeTrue(), "应该创建了source_code.tar.gz文件")

	// 3. 检查并创建namespace
	cmd3 := "kubectl get namespace ai-inference || kubectl create namespace ai-inference"
	res3, err := sshClient.Exec(cmd3)
	Expect(err).NotTo(HaveOccurred(), "应该能够检查并创建namespace")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd3, res3)

	// 4. 删除已存在的ConfigMap
	cmd4 := "kubectl delete configmap vllm-source-code -n ai-inference --ignore-not-found"
	res4, err := sshClient.Exec(cmd4)
	Expect(err).NotTo(HaveOccurred(), "应该能够删除已存在的ConfigMap")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd4, res4)

	// 5. 创建新的ConfigMap
	cmd5 := fmt.Sprintf(`
kubectl create configmap vllm-source-code \
  --from-file=source_code.tar.gz=%s/source_code.tar.gz \
  -n ai-inference`, baseDir)
	res5, err := sshClient.Exec(cmd5)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建ConfigMap")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd5, res5)

	// 6. 创建订阅方
	cmd6 := fmt.Sprintf("kubectl apply -f %s/docs/routing-metrics/vllm_eagle_eye_subscriber.yaml", baseDir)
	res6, err := sshClient.Exec(cmd6)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建订阅方")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd6, res6)

	// 等待订阅方pod就绪
	GinkgoWriter.Printf("等待订阅方pod就绪...\n")
	var subscriberPodName string
	Eventually(func() bool {
		cmd := "kubectl get pods -n ai-inference --no-headers | grep vllm-eagle-eye-subscriber | awk '{print $1}'"
		res, err := sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		subscriberPodName = strings.TrimSpace(res.Stdout)
		if subscriberPodName == "" {
			return false
		}

		// 检查pod状态
		cmd = fmt.Sprintf("kubectl get pod %s -n ai-inference -o jsonpath='{.status.phase}'", subscriberPodName)
		res, err = sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		return strings.TrimSpace(res.Stdout) == "Running"
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "订阅方pod应该在2分钟内就绪")
	GinkgoWriter.Printf("订阅方pod已就绪: %s\n", subscriberPodName)

	// 7. 创建发布方
	cmd7 := fmt.Sprintf("kubectl apply -f %s/docs/routing-metrics/vllm_eagle_eye.yaml", baseDir)
	res7, err := sshClient.Exec(cmd7)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建发布方")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd7, res7)

	// 等待发布方pod就绪
	GinkgoWriter.Printf("等待发布方pod就绪...\n")
	//等待服务就绪，直接等待两分钟
	time.Sleep(2 * time.Minute)
	var publisherPodName string
	var podIP string
	Eventually(func() bool {
		cmd := "kubectl get pod -n ai-inference -l app=vllm-eagle-eye -o jsonpath='{.items[0].metadata.name}'"
		res, err := sshClient.Exec(cmd)
		if err != nil || res.Stdout == "" {
			return false
		}
		publisherPodName = strings.TrimSpace(res.Stdout)

		// 检查pod状态
		cmd = fmt.Sprintf("kubectl get pod %s -n ai-inference -o jsonpath='{.status.phase}'", publisherPodName)
		res, err = sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		phase := strings.TrimSpace(res.Stdout)

		if phase == "Running" {
			// 获取pod IP
			cmd = fmt.Sprintf("kubectl get pod %s -n ai-inference -o jsonpath='{.status.podIP}'", publisherPodName)
			res, err = sshClient.Exec(cmd)
			if err == nil && res.Stdout != "" {
				podIP = strings.TrimSpace(res.Stdout)
			}
			return true
		}
		return false
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "发布方pod应该在2分钟内就绪")
	GinkgoWriter.Printf("发布方pod已就绪: %s, IP: %s\n", publisherPodName, podIP)

	// 8. 创建测试pod并发送请求
	cmd8 := `kubectl run curl-test --image=curlimages/curl -n ai-inference --restart=Never -- /bin/sh -c "sleep 3600"`
	res8, err := sshClient.Exec(cmd8)
	Expect(err).NotTo(HaveOccurred(), "应该能够创建测试pod")
	GinkgoWriter.Printf("执行: %s\n结果: %v\n", cmd8, res8)

	// 等待测试pod就绪
	Eventually(func() bool {
		cmd := "kubectl get pod curl-test -n ai-inference -o jsonpath='{.status.phase}' 2>/dev/null || echo 'NotFound'"
		res, err := sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		return strings.TrimSpace(res.Stdout) == "Running"
	}, 2*time.Minute, 10*time.Second).Should(BeTrue(), "测试pod应该在2分钟内就绪")

	// 明确打印测试 pod 就绪，便于日志追踪
	GinkgoWriter.Printf("测试 pod 已就绪: curl-test\n")

	// 9. 发送请求到发布方
	// 使用更稳健的引用并保留 curl 的 stderr 以便排查连接问题（curl 返回码 7 表示无法连接）
	curlCmd := fmt.Sprintf(`kubectl exec -n ai-inference curl-test -- /bin/sh -c 'curl -sS -X POST "http://%s:8000/generate" -H "Content-Type: application/json" -d "{\"prompt\": \"Hello AI\"}" -w "%%{http_code}"'`, podIP)

	GinkgoWriter.Printf("发送请求到发布方...\n")
	Eventually(func() bool {
		res, err := sshClient.Exec(curlCmd)
		if err != nil {
			// 打印更多调试信息，包含 stdout/stderr
			GinkgoWriter.Printf("发送请求失败: %v\n", err)
			if res != nil {
				GinkgoWriter.Printf("curl stdout:\n%s\n", res.Stdout)
				GinkgoWriter.Printf("curl stderr:\n%s\n", res.Stderr)
			}
			return false
		}

		statusCode := strings.TrimSpace(res.Stdout)
		GinkgoWriter.Printf("HTTP状态码: %s\n", statusCode)
		return strings.Contains(statusCode, "200")
	}, 1*time.Minute, 10*time.Second).Should(BeTrue(), "应该能够成功发送请求到发布方")

	// 10. 检查订阅方日志
	GinkgoWriter.Printf("检查订阅方日志...\n")
	Eventually(func() bool {
		cmd := fmt.Sprintf("kubectl logs -n ai-inference %s --tail=20", subscriberPodName)
		res, err := sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		logs := res.Stdout
		return strings.Contains(logs, "Received message on topic eagle_eye.routing_metrics")
	}, 1*time.Minute, 10*time.Second).Should(BeTrue(), "订阅方日志中应该包含'Received message on topic eagle_eye.routing_metrics'")

	// 11. 检查发布方日志
	GinkgoWriter.Printf("检查发布方日志...\n")
	Eventually(func() bool {
		cmd := fmt.Sprintf("kubectl logs -n ai-inference %s --tail=20", publisherPodName)
		res, err := sshClient.Exec(cmd)
		if err != nil {
			return false
		}
		logs := res.Stdout
		return strings.Contains(logs, "Successfully published results to NATS topic eagle_eye.routing_metrics")
	}, 1*time.Minute, 10*time.Second).Should(BeTrue(), "发布方日志中应该包含'Successfully published results to NATS topic eagle_eye.routing_metrics'")

	// 12. 清理资源
	GinkgoWriter.Printf("开始清理资源...\n")

	// 删除发布方
	cleanupCmd1 := fmt.Sprintf("kubectl delete -f %s/docs/routing-metrics/vllm_eagle_eye.yaml --ignore-not-found", baseDir)
	_, err = sshClient.Exec(cleanupCmd1)
	Expect(err).NotTo(HaveOccurred(), "应该能够删除发布方")

	// 删除订阅方
	cleanupCmd2 := fmt.Sprintf("kubectl delete -f %s/docs/routing-metrics/vllm_eagle_eye_subscriber.yaml --ignore-not-found", baseDir)
	_, err = sshClient.Exec(cleanupCmd2)
	Expect(err).NotTo(HaveOccurred(), "应该能够删除订阅方")

	// 删除ConfigMap
	cleanupCmd3 := "kubectl delete configmap vllm-source-code -n ai-inference --ignore-not-found"
	_, err = sshClient.Exec(cleanupCmd3)
	Expect(err).NotTo(HaveOccurred(), "应该能够删除ConfigMap")

	// 删除测试pod
	cleanupCmd4 := "kubectl delete pod curl-test -n ai-inference --ignore-not-found"
	_, err = sshClient.Exec(cleanupCmd4)
	Expect(err).NotTo(HaveOccurred(), "应该能够删除测试pod")

	GinkgoWriter.Printf("NATS功能验证完成！\n")
}

// verifyEagleEyePods 验证Eagle-Eye相关的Pod状态和镜像
func verifyEagleEyePods(k8sClient *k8s.K8SClient, ctx context.Context) {
	// 获取所有 Pod 作为候选集合
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有 pods 时不应出错")
	Expect(len(podList.Items)).To(BeNumerically(">", 0), "集群中应至少存在一个 pod")

	GinkgoWriter.Printf("开始检查Eagle-Eye相关Pod...\n")

	// 遍历所有期望的组件名称（来自 ExpectedImages 的 key）并校验镜像
	componentsFound := 0
	for componentName, expectedImgs := range ExpectedImages {
		matchedPod := findMatchingPod(componentName, podList.Items)
		if matchedPod == nil {
			GinkgoWriter.Printf("未找到与组件 %s 匹配的 pod\n", componentName)
			continue
		}

		componentsFound++
		GinkgoWriter.Printf("检查 Pod: %s/%s\n", matchedPod.Namespace, matchedPod.Name)

		// 检查 Pod 的状态为 Running
		Expect(matchedPod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s 应为 Running", matchedPod.Name)

		// 如果没有配置期望镜像，跳过镜像断言
		if len(expectedImgs) == 0 {
			GinkgoWriter.Printf("未为组件 %s 配置期望镜像，跳过镜像断言\n", componentName)
			continue
		}

		// 校验容器镜像至少匹配一项期望值（支持部分匹配）
		podImages := getAllPodImages(matchedPod)
		GinkgoWriter.Printf("容器镜像: %v\n", podImages)
		allExpectedImagesFound := true
		for _, expectedImage := range expectedImgs {
			if expectedImage == "" {
				// 空字符串表示通配符，总是匹配
				GinkgoWriter.Printf("期望镜像 [任意]: 跳过检查\n")
				continue
			}
			if contains(podImages, expectedImage) || containsSubstringInSlice(podImages, expectedImage) {
				GinkgoWriter.Printf("期望镜像 %s: 存在\n", expectedImage)
			} else {
				GinkgoWriter.Printf("期望镜像 %s: 不存在\n", expectedImage)
				allExpectedImagesFound = false
			}
		}

		if !allExpectedImagesFound {
			Expect(allExpectedImagesFound).To(BeTrue(), fmt.Sprintf("组件 %s 应该包含期望镜像", componentName))
		}
	}

	GinkgoWriter.Printf("Eagle-Eye Pod 检查完成: 找到 %d 个组件的 Pod\n", componentsFound)
}

// verifyHardwareDiagnosisService 验证hardware-diagnosis服务是否运行且日志显示事件检测
func verifyHardwareDiagnosisService(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("开始检查 hardware-diagnosis...\n")

	// 获取 hardware-diagnosis pod
	matchedPod := getComponentPod(k8sClient, ctx, "hardware-diagnosis")

	// 获取 hardware-diagnosis pod 的日志（通过 SSH 执行 kubectl logs）
	getLogsCmd := fmt.Sprintf("kubectl logs -n %s %s --tail=100", matchedPod.Namespace, matchedPod.Name)
	res, err := sshClient.Exec(getLogsCmd)
	Expect(err).NotTo(HaveOccurred(), "执行 kubectl logs 获取 hardware-diagnosis 日志不应报错")
	Expect(res).NotTo(BeNil(), "kubectl logs 返回结果不应为空")

	logs := res.Stdout

	// 检查日志中是否包含指定的关键字
	expectedKeywords := []string{
		"hardware-diagnosis - INFO - Detected events",
		"Detected events", // 也检查简化的关键字
	}

	keywordFound := false
	var foundKeyword string

	for _, keyword := range expectedKeywords {
		if strings.Contains(logs, keyword) {
			keywordFound = true
			foundKeyword = keyword
			break
		}
	}

	Expect(keywordFound).To(BeTrue(),
		fmt.Sprintf("hardware-diagnosis 日志中应该包含事件检测关键字，当前日志中未找到: %v\nPod: %s/%s\n日志片段: %s",
			expectedKeywords, matchedPod.Namespace, matchedPod.Name, getLogSnippet(logs)))

	GinkgoWriter.Printf("hardware-diagnosis 日志检查通过，找到关键字: %s\n", foundKeyword)
}

// verifyHardwareMonitorService 验证hardware-monitor服务是否运行且日志显示硬件指标收集
func verifyHardwareMonitorService(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor) {
	GinkgoWriter.Printf("开始检查 hardware-monitor...\n")

	// 获取 hardware-monitor pod
	matchedPod := getComponentPod(k8sClient, ctx, "hardware-monitor")

	// 获取 hardware-monitor pod 的日志（通过 SSH 执行 kubectl logs）
	getLogsCmd := fmt.Sprintf("kubectl logs -n %s %s --tail=100", matchedPod.Namespace, matchedPod.Name)
	res, err := sshClient.Exec(getLogsCmd)
	Expect(err).NotTo(HaveOccurred(), "执行 kubectl logs 获取 hardware-monitor 日志不应报错")
	Expect(res).NotTo(BeNil(), "kubectl logs 返回结果不应为空")

	logs := res.Stdout
	// 检查日志中是否包含指定的关键字
	expectedKeywords := []string{
		// 成功收集各种硬件指标
		"collector.npu.ddr",
		"collector.npu.hbm",
		"collector.npu.npu",
		"collector.npu.npu",

		// 成功发布到NATS
		"Successfully published results to NATS topic",
	}

	var notFoundKeywords []string

	for _, keyword := range expectedKeywords {
		if !strings.Contains(logs, keyword) {
			notFoundKeywords = append(notFoundKeywords, keyword)
		}
	}

	// 检查是否所有关键字都被找到
	if len(notFoundKeywords) > 0 {
		GinkgoWriter.Printf("在 hardware-monitor 日志中未找到的关键字: %v\n", notFoundKeywords)
		Expect(len(notFoundKeywords)).To(Equal(0), fmt.Sprintf("hardware-monitor 日志中应该包含所有关键字，缺少: %v", notFoundKeywords))
	}

	GinkgoWriter.Printf("hardware-monitor 日志检查通过，所有关键字都已找到\n")
}

// findMatchingPod 根据组件名称查找匹配的 Pod
func findMatchingPod(componentName string, pods []corev1.Pod) *corev1.Pod {
	// 策略1：完全匹配 - 查找 "componentName-*" 形式的 Pod
	for i := range pods {
		if strings.HasPrefix(pods[i].Name, componentName+"-") {
			return &pods[i]
		}
	}

	// 策略2：包含匹配 - 查找名称中包含组件名称的 Pod
	for i := range pods {
		if strings.Contains(pods[i].Name, componentName) {
			// 避免误匹配
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

// getAllPodImages 获取Pod中的所有容器镜像
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

// containsSubstringInSlice 检查切片中是否有元素包含给定子串（支持部分匹配）
func containsSubstringInSlice(slice []string, substr string) bool {
	for _, item := range slice {
		if strings.Contains(item, substr) {
			return true
		}
	}
	return false
}

// verifyNATSServiceReadiness 验证NATS服务是否运行且日志显示就绪
func verifyNATSServiceReadiness(sshClient *executor.SSHExecutor) {
	// 使用 SSHExecutor 获取 NATS pod 的日志
	cmd := "kubectl logs -n nats $(kubectl get pods -n nats -l app.kubernetes.io/name=nats -o jsonpath='{.items[*].metadata.name}')"
	res, err := sshClient.Exec(cmd)
	Expect(err).NotTo(HaveOccurred(), "执行 kubectl logs 不应报错")
	Expect(res).NotTo(BeNil(), "kubectl logs 返回结果不应为空")

	logs := res.Stdout

	// 检查日志中是否包含 Server is ready
	Expect(strings.Contains(logs, "Server is ready")).To(BeTrue(), "NATS 日志中应该包含 'Server is ready'")
}

// getLogSnippet 获取日志的片段用于错误信息
func getLogSnippet(logs string) string {
	if len(logs) == 0 {
		return "[空日志]"
	}

	// 取最后200个字符作为片段
	if len(logs) > 200 {
		return logs[len(logs)-200:] + "..."
	}
	return logs
}

// getComponentPod 公共方法：根据组件名称获取对应的 Pod
func getComponentPod(k8sClient *k8s.K8SClient, ctx context.Context, componentName string) *corev1.Pod {
	GinkgoWriter.Printf("开始查找 %s pod...\n", componentName)

	// 获取所有 Pod 作为候选集合
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有 pods 时不应出错")

	// 使用 findMatchingPod 查找指定组件的 pod
	matchedPod := findMatchingPod(componentName, podList.Items)
	Expect(matchedPod).NotTo(BeNil(), "应该能够找到 %s pod", componentName)

	GinkgoWriter.Printf("找到 %s pod: %s/%s\n", componentName, matchedPod.Namespace, matchedPod.Name)
	return matchedPod
}

// findEagleEyeDirectory 查找eagle-eye目录
func findEagleEyeDirectory(sshClient *executor.SSHExecutor) string {
	// 尝试几个可能的路径
	possiblePaths := []string{
		"/root/eagle-eye",
		"~/eagle-eye",
		"/eagle-eye",
	}

	for _, path := range possiblePaths {
		cmd := fmt.Sprintf("ls -la %s 2>/dev/null | head -5", path)
		res, err := sshClient.Exec(cmd)
		if err == nil && !strings.Contains(res.Stdout, "No such file or directory") {
			GinkgoWriter.Printf("找到目录: %s\n", path)
			return path
		}
	}

	return ""
}
