package monitoring_dashboard

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

// ==================== monitoring-dashboard 组件间集成测试 ====================

var isDaily = true

var _ = Describe("monitoring-dashboard 组件间集成测试", Ordered, Label("monitoring-dashboard-components", "with-workload-cluster", "system-integration"), func() {

	var (
		helmConfig    map[string]*helm.ReleaseConfig
		sshClient     *executor.SSHExecutor
		localexecutor *executor.LocalExecutor
		k8sClient     *k8s.K8SClient
		ctx           context.Context
		cancel        context.CancelFunc
	)

	BeforeAll(func() {
		var err error

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

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

		// load helm config
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
			if isDaily {
				needUninstall, err = helm.CheckHelmReleaseLocal(eachRelease.ReleaseName, eachRelease.Namespace, localexecutor)
			} else {
				needUninstall, err = helm.CheckHelmRelease(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
			}
			if err != nil {
				GinkgoWriter.Println(err)
			}
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "helm release check 应该成功")
			if needUninstall == true {
				if isDaily {
					err = helm.HelmUninstallLocal(eachRelease.ReleaseName, eachRelease.Namespace, localexecutor)
				} else {
					err = helm.HelmUninstall(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
				}
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

	// 按照 helm_config.yaml 中记录的组件，逐一执行兼容性测试（与 eagle-eye 写法保持一致）
	It("monitoring-dashboard & logging-package", func() {
		var err error
		// step1: logging-package install
		GinkgoWriter.Println("=====step1: logging-package install=====")
		tempReleaseName := helmConfig["logging-package"].ReleaseName
		tempHelmUrl := helmConfig["logging-package"].HelmURrl
		tempVersion := helmConfig["logging-package"].Version
		tempNamespace := helmConfig["logging-package"].Namespace

		GinkgoWriter.Println("start install logging-package...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("logging-package install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: monitoring-dashboard installation check
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & colocation-package", func() {
		var err error
		// step1: colocation-package install
		GinkgoWriter.Println("=====step1: colocation-package install=====")
		tempReleaseName := helmConfig["colocation-package"].ReleaseName
		tempHelmUrl := helmConfig["colocation-package"].HelmURrl
		tempVersion := helmConfig["colocation-package"].Version
		tempNamespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("colocation-package install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)

		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & many-core-scheduler", func() {
		var err error
		// step1: many-core-scheduler install
		err = k8sClient.CreateNamespaceIfNotExists(ctx, "volcano-system")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ns: volcano-system should be created successfully")

		GinkgoWriter.Println("=====step1: many-core-scheduler install=====")
		tempReleaseName := helmConfig["many-core-scheduler"].ReleaseName
		tempHelmUrl := helmConfig["many-core-scheduler"].HelmURrl
		tempVersion := helmConfig["many-core-scheduler"].Version
		tempNamespace := helmConfig["many-core-scheduler"].Namespace

		GinkgoWriter.Println("start install many-core-scheduler...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install many-core-scheduler 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("many-core-scheduler install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & numa-affinity-package", func() {
		var err error
		// step1: numa-affinity-package install
		GinkgoWriter.Println("=====step1: numa-affinity-package install=====")
		tempReleaseName := helmConfig["numa-affinity-package"].ReleaseName
		tempHelmUrl := helmConfig["numa-affinity-package"].HelmURrl
		tempVersion := helmConfig["numa-affinity-package"].Version
		tempNamespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("numa-affinity-package install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & multi-cluster-service", func() {
		var err error
		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		tempReleaseName := helmConfig["multi-cluster-service"].ReleaseName
		tempHelmUrl := helmConfig["multi-cluster-service"].HelmURrl
		tempVersion := helmConfig["multi-cluster-service"].Version
		tempNamespace := helmConfig["multi-cluster-service"].Namespace

		GinkgoWriter.Println("start install multi-cluster-service...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("multi-cluster-service install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & ray-package", func() {
		var err error
		// step1: ray-package install
		GinkgoWriter.Println("=====step1: ray-package install=====")
		tempReleaseName := helmConfig["ray-package"].ReleaseName
		tempHelmUrl := helmConfig["ray-package"].HelmURrl
		tempVersion := helmConfig["ray-package"].Version
		tempNamespace := helmConfig["ray-package"].Namespace

		GinkgoWriter.Println("start install ray-package...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install ray-package 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("ray-package install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & kae-operator", Label("kae"), func() {
		var err error
		// step1: kae-operator install
		GinkgoWriter.Println("=====step1: kae-operator install=====")
		tempReleaseName := helmConfig["kae-operator"].ReleaseName
		tempHelmUrl := helmConfig["kae-operator"].HelmURrl
		tempVersion := helmConfig["kae-operator"].Version
		tempNamespace := helmConfig["kae-operator"].Namespace

		GinkgoWriter.Println("start install kae-operator...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install kae-operator 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("kae-operator install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & npu-operator", Label("npu"), func() {
		var err error
		// step1: npu-operator install
		GinkgoWriter.Println("=====step1: npu-operator install=====")
		tempReleaseName := helmConfig["npu-operator"].ReleaseName
		tempHelmUrl := helmConfig["npu-operator"].HelmURrl
		tempVersion := helmConfig["npu-operator"].Version
		tempNamespace := helmConfig["npu-operator"].Namespace

		GinkgoWriter.Println("start install npu-operator...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("npu-operator install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})

	It("monitoring-dashboard & infernex", Label("npu"), func() {
		var err error
		// step1: infernex install
		GinkgoWriter.Println("=====step1: infernex install=====")
		tempReleaseName := helmConfig["infernex"].ReleaseName
		tempHelmUrl := helmConfig["infernex"].HelmURrl
		tempVersion := helmConfig["infernex"].Version
		tempNamespace := helmConfig["infernex"].Namespace

		GinkgoWriter.Println("start install infernex...")
		if isDaily {
			err = helm.HelmInstallLocal(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, localexecutor)
		} else {
			err = helm.HelmInstall(tempReleaseName, tempHelmUrl, tempVersion, tempNamespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install infernex 应当执行成功")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("infernex install success")

		// step2: monitoring-dashboard install
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("=====step3: monitoring-dashboard installation check=====")
		verifyMonitoringDashboardPods(k8sClient, ctx)
		GinkgoWriter.Println("monitoring-dashboard installation check pass")
	})
})

var _ = Describe("Monitoring-dashboard System Integration", Label("monitoring_dashboard_system_integration", "with-workload-cluster", "system-integration"), func() {
	var (
		k8sClient     *k8s.K8SClient
		ctx           context.Context
		cancel        context.CancelFunc
		sshClient     *executor.SSHExecutor
		localexecutor *executor.LocalExecutor
		helmConfig    map[string]*helm.ReleaseConfig
	)

	BeforeEach(func() {
		var err error
		ctx, cancel = context.WithTimeout(context.Background(), 30*time.Minute)

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
			Expect(err).NotTo(HaveOccurred(), "应该能够使用 SSH 连接到 Monitoring-dashboard 测试节点")
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

		// load helm config
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
			if isDaily {
				needUninstall, err = helm.CheckHelmReleaseLocal(eachRelease.ReleaseName, eachRelease.Namespace, localexecutor)
			} else {
				needUninstall, err = helm.CheckHelmRelease(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
			}
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "helm release check 应该成功")
			if needUninstall == true {
				if isDaily {
					err = helm.HelmUninstallLocal(eachRelease.ReleaseName, eachRelease.Namespace, localexecutor)
				} else {
					err = helm.HelmUninstall(eachRelease.ReleaseName, eachRelease.Namespace, sshClient)
				}
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
		if cancel != nil {
			cancel()
		}
	})

	It("should have monitoring-dashboard related pods running and images match expectations", func() {
		// step2: monitoring-dashboard install
		var err error
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")
		time.Sleep(60 * time.Second)
		// 校验 monitoring-dashboard 相关 pod 的状态和镜像
		verifyMonitoringDashboardPods(k8sClient, ctx)
	})

	It("should automatically recover monitoring components after pod deletion", func() {
		// step2: monitoring-dashboard install
		var err error
		GinkgoWriter.Println("=====step2: monitoring-dashboard install=====")
		releaseName := helmConfig["monitoring-dashboard"].ReleaseName
		helmUrl := helmConfig["monitoring-dashboard"].HelmURrl
		version := helmConfig["monitoring-dashboard"].Version
		namespace := helmConfig["monitoring-dashboard"].Namespace

		GinkgoWriter.Println("start install monitoring-dashboard...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install monitoring-dashboard 应当执行成功")
		time.Sleep(60 * time.Second)
		// 按照 ExpectedImages 中的 componentName，依次用 kubectl delete pod 的方式删除，然后检查 pod 在2分钟内是否恢复
		verifyMonitoringComponentAutoRecovery(k8sClient, ctx, sshClient, localexecutor)
	})

})

// verifyMonitoringComponentAutoRecovery 验证组件在Pod删除后的自动恢复能力
func verifyMonitoringComponentAutoRecovery(k8sClient *k8s.K8SClient, ctx context.Context, sshClient *executor.SSHExecutor, localexecutor *executor.LocalExecutor) {
	GinkgoWriter.Printf("开始验证 monitoring 组件的自愈能力...\n")

	startTime := time.Now()
	var failedComponents []string
	successfullyRecovered := 0
	totalComponents := len(ExpectedImages)

	if totalComponents == 0 {
		GinkgoWriter.Printf("警告: ExpectedImages 配置为空，跳过自愈能力测试\n")
		return
	}

	var componentNames []string
	for componentName := range ExpectedImages {
		componentNames = append(componentNames, componentName)
	}

	GinkgoWriter.Printf("=== 第一阶段: 删除所有组件的Pod ===\n")
	for i, componentName := range componentNames {
		GinkgoWriter.Printf("处理组件 (%d/%d): %s\n", i+1, totalComponents, componentName)
		originalPod := findMonitoringComponentPodByName(k8sClient, ctx, componentName)
		if originalPod == nil {
			GinkgoWriter.Printf("警告: 未找到 %s 组件的Pod\n", componentName)
			continue
		}

		originalPodName := originalPod.Name
		namespace := originalPod.Namespace
		GinkgoWriter.Printf("  找到Pod: %s/%s (状态: %s)\n", namespace, originalPodName, originalPod.Status.Phase)

		deleteCmd := fmt.Sprintf("kubectl delete pod %s -n %s --wait=false --timeout=30s", originalPodName, namespace)
		var deleteRes *executor.ExecResult
		var err error
		if isDaily {
			deleteRes, err = localexecutor.Exec(deleteCmd)
		} else {
			deleteRes, err = sshClient.Exec(deleteCmd)
		}
		if err != nil {
			GinkgoWriter.Printf("  删除失败: %v\n", err)
			if deleteRes != nil {
				GinkgoWriter.Printf("  删除命令输出: stdout=%s, stderr=%s\n", deleteRes.Stdout, deleteRes.Stderr)
			}
			continue
		}
		GinkgoWriter.Printf("  删除成功: %s\n", strings.TrimSpace(deleteRes.Stdout))
	}

	GinkgoWriter.Printf("\n=== 第二阶段: 等待2分钟让Pod恢复 ===\n")
	time.Sleep(2 * time.Minute)

	GinkgoWriter.Printf("\n=== 第三阶段: 检查所有组件是否恢复 ===\n")
	for _, componentName := range componentNames {
		GinkgoWriter.Printf("检查组件: %s\n", componentName)
		newPod := findMonitoringComponentPodByName(k8sClient, ctx, componentName)
		if newPod == nil {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (恢复后未找到Pod)", componentName))
			GinkgoWriter.Printf("  未找到Pod\n")
			continue
		}

		newPodName := newPod.Name
		namespace := newPod.Namespace
		GinkgoWriter.Printf("  找到Pod: %s/%s (状态: %s)\n", namespace, newPodName, newPod.Status.Phase)

		if newPod.Status.Phase != corev1.PodRunning {
			failedComponents = append(failedComponents, fmt.Sprintf("%s (状态: %s)", componentName, newPod.Status.Phase))
			continue
		}

		successfullyRecovered++
	}

	duration := time.Since(startTime)
	GinkgoWriter.Printf("自愈测试完成: 成功恢复 %d/%d 个组件, 耗时: %s\n", successfullyRecovered, totalComponents, duration)

	if len(failedComponents) > 0 {
		GinkgoWriter.Printf("以下组件恢复失败: %v\n", failedComponents)
	}

	Expect(len(failedComponents)).To(Equal(0), "以下组件在删除后未能成功恢复: %v", failedComponents)
}

// verifyMonitoringDashboardPods 验证 monitoring-dashboard 相关的Pod状态和镜像
func verifyMonitoringDashboardPods(k8sClient *k8s.K8SClient, ctx context.Context) {
	GinkgoWriter.Printf("开始检查 monitoring-dashboard 相关Pod...\n")

	// 检查所有期望的组件
	checkAllExpectedComponents(k8sClient, ctx)
}

// checkAllExpectedComponents 检查所有期望的组件
func checkAllExpectedComponents(k8sClient *k8s.K8SClient, ctx context.Context) {
	// 获取所有 Pod 作为候选集合
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取所有 pods 时不应出错")
	Expect(len(podList.Items)).To(BeNumerically(">", 0), "集群中应至少存在一个 pod")

	// 遍历所有期望的组件名称（来自 ExpectedImages 的 key）并校验
	componentsFound := 0
	for componentName, expectedImgs := range ExpectedImages {
		matchedPod := findMatchingMonitoringPod(componentName, podList.Items)
		if matchedPod == nil {
			GinkgoWriter.Printf("未找到与组件 %s 匹配的 pod\n", componentName)
			continue
		}

		componentsFound++
		GinkgoWriter.Printf("检查 Pod: %s/%s (组件: %s)\n", matchedPod.Namespace, matchedPod.Name, componentName)

		// 检查 Pod 的状态
		checkPodStatus(matchedPod)

		// 如果有配置期望镜像，检查镜像匹配
		if len(expectedImgs) > 0 && !isDaily {
			checkPodImages(matchedPod, expectedImgs, componentName)
		}
	}

	Expect(componentsFound).To(Equal(len(ExpectedImages)),
		"应该找到所有期望的组件。期望: %d, 找到: %d", len(ExpectedImages), componentsFound)

	GinkgoWriter.Printf("monitoring-dashboard Pod 检查完成: 找到 %d 个组件的 Pod\n", componentsFound)
}

// checkPodStatus 检查Pod状态
func checkPodStatus(pod *corev1.Pod) {
	// 检查 Pod 的状态为 Running
	Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "pod %s 应为 Running", pod.Name)

	// 检查所有容器是否就绪
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady {
			Expect(condition.Status).To(Equal(corev1.ConditionTrue),
				"pod %s 应为 Ready 状态", pod.Name)
		}
	}

	GinkgoWriter.Printf("  Pod状态: %s (Ready: %v)\n",
		pod.Status.Phase, isPodReady(pod))
}

// checkPodImages 检查Pod镜像
func checkPodImages(pod *corev1.Pod, expectedImgs []string, componentName string) {
	// 获取Pod中的所有容器镜像
	podImages := getAllPodImages(pod)
	GinkgoWriter.Printf("  容器镜像: %v\n", podImages)

	// 校验容器镜像至少匹配一项期望值（支持部分匹配）
	allExpectedImagesFound := true
	for _, expectedImage := range expectedImgs {
		if expectedImage == "" {
			// 空字符串表示通配符，总是匹配
			GinkgoWriter.Printf("  期望镜像 [任意]: 跳过检查\n")
			continue
		}
		if contains(podImages, expectedImage) || containsSubstringInSlice(podImages, expectedImage) {
			GinkgoWriter.Printf("  期望镜像 %s: 存在\n", expectedImage)
		} else {
			GinkgoWriter.Printf("  期望镜像 %s: 不存在\n", expectedImage)
			allExpectedImagesFound = false
		}
	}

	if !allExpectedImagesFound {
		Expect(allExpectedImagesFound).To(BeTrue(),
			fmt.Sprintf("组件 %s 应该包含期望镜像", componentName))
	}
}

// findMonitoringComponentPodByName 根据组件名称查找对应的Pod
func findMonitoringComponentPodByName(k8sClient *k8s.K8SClient, ctx context.Context, componentName string) *corev1.Pod {
	podList, err := k8sClient.Clientset.CoreV1().Pods("").List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取Pod列表时不应出错")

	return findMatchingMonitoringPod(componentName, podList.Items)
}

// findMatchingMonitoringPod 根据组件名称查找匹配的 Pod
func findMatchingMonitoringPod(componentName string, pods []corev1.Pod) *corev1.Pod {
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

// isPodReady 检查Pod是否就绪
func isPodReady(pod *corev1.Pod) bool {
	for _, condition := range pod.Status.Conditions {
		if condition.Type == corev1.PodReady && condition.Status == corev1.ConditionTrue {
			return true
		}
	}
	return false
}
