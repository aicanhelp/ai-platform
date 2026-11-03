package infernex

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/system-integration/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var (
	helmConfig    map[string]*helm.ReleaseConfig
	sshClient     *executor.SSHExecutor
	localexecutor *executor.LocalExecutor
	k8sClient     *k8s.K8SClient
	ctx           context.Context
	cancel        context.CancelFunc
)

var isDaily = true

const (
	lwsHelmReleaseName = "lws"
	lwsHelmChartURL    = "oci://registry.k8s.io/lws/charts/lws"
	lwsHelmVersion     = "0.8.0"
	lwsNamespace       = "lws-system"
)

func setupClients() {
	ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
	DeferCleanup(cancel)

	var err error
	if isDaily {
		localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	} else {
		port, _ := strconv.Atoi(os.Getenv("TEST_NODE3_PORT"))
		sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE3_IP"), port, os.Getenv("TEST_NODE3_USER"), os.Getenv("TEST_NODE3_PASSWORD"))
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ssh Client 应当成功创建")

		k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
	}
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	helmConfig, err = helm.LoadHelmConfig()
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm config 应当成功加载")

	err = k8sClient.CreateNamespaceIfNotExists(ctx, "ai-inference")
	Expect(err).NotTo(HaveOccurred(), "namespace: ai-inference should existed or be created correctly")

	installLeaderWorkerSetIfNeeded()
}

func checkHelmRelease(releaseName, namespace string) (bool, error) {
	if isDaily {
		return helm.CheckHelmReleaseLocal(releaseName, namespace, localexecutor)
	}
	return helm.CheckHelmRelease(releaseName, namespace, sshClient)
}

func helmInstall(releaseName, helmUrl, version, namespace string, installArgs ...string) error {
	if isDaily {
		return helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor, installArgs...)
	}
	return helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient, installArgs...)
}

func helmInstallRelease(cfg *helm.ReleaseConfig, installArgs ...string) error {
	return helmInstall(cfg.ReleaseName, cfg.HelmURrl, cfg.Version, cfg.Namespace, installArgs...)
}

func helmUninstallRelease(releaseName, namespace string) error {
	if isDaily {
		return helm.HelmUninstallLocal(releaseName, namespace, localexecutor)
	}
	return helm.HelmUninstall(releaseName, namespace, sshClient)
}

func installLeaderWorkerSetIfNeeded() {
	installed, err := checkHelmRelease(lwsHelmReleaseName, lwsNamespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm release check for leaderworkerset 应该成功")
	if installed {
		GinkgoWriter.Println("leaderworkerset already installed, skip")
		return
	}

	err = k8sClient.CreateNamespaceIfNotExists(ctx, lwsNamespace)
	Expect(err).NotTo(HaveOccurred(), "namespace for leaderworkerset should be created")

	GinkgoWriter.Println("start install leaderworkerset...")
	err = helmInstall(lwsHelmReleaseName, lwsHelmChartURL, lwsHelmVersion, lwsNamespace, "--create-namespace")
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install leaderworkerset 应当执行成功")

	time.Sleep(60 * time.Second)
	GinkgoWriter.Println("leaderworkerset install success")
}

func uninstallLeaderWorkerSet() {
	installed, err := checkHelmRelease(lwsHelmReleaseName, lwsNamespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm release check for leaderworkerset 应该成功")
	if !installed {
		GinkgoWriter.Println("leaderworkerset already uninstalled, skip")
		return
	}

	GinkgoWriter.Println("start uninstall leaderworkerset...")
	err = helmUninstallRelease(lwsHelmReleaseName, lwsNamespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm uninstall leaderworkerset 应当执行成功")
	time.Sleep(15 * time.Second)
	GinkgoWriter.Println("leaderworkerset has been uninstalled")
}

func restoreLeaderWorkerSetIfNeeded() {
	installLeaderWorkerSetIfNeeded()
}

func uninstallNpuOperator() {
	cfg := helmConfig["npu-operator"]
	Expect(cfg).NotTo(BeNil(), "helm config 应包含 npu-operator")

	installed, err := checkHelmRelease(cfg.ReleaseName, cfg.Namespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "npu-operator helm release check 应该成功")
	if !installed {
		GinkgoWriter.Println("npu-operator already uninstalled, skip")
		return
	}

	GinkgoWriter.Println("start uninstall npu-operator...")
	err = helmUninstallRelease(cfg.ReleaseName, cfg.Namespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm uninstall npu-operator 应当执行成功")
	time.Sleep(30 * time.Second)
	Eventually(
		func() bool {
			status, err := helm.HelmUninstallNamespaceCheck(cfg.ReleaseName, k8sClient, ctx)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			return status
		}, 3*time.Minute, 1*time.Second,
	).Should(BeTrue(), "npu-operator namespace deletion should be completed")
	GinkgoWriter.Println("npu-operator uninstall success")
}

func restoreNpuOperatorIfNeeded() {
	cfg := helmConfig["npu-operator"]
	if cfg == nil {
		return
	}

	installed, err := checkHelmRelease(cfg.ReleaseName, cfg.Namespace)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "npu-operator helm release check 应该成功")
	if installed {
		GinkgoWriter.Println("npu-operator already installed, skip restore")
		return
	}

	GinkgoWriter.Println("start restore npu-operator...")
	err = helmInstallRelease(cfg)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install npu-operator 应当执行成功")
	time.Sleep(90 * time.Second)
	GinkgoWriter.Println("npu-operator restore success")
}

func cleanupHelmReleases() {
	for _, eachRelease := range helmConfig {
		if eachRelease.ReleaseName == "npu-operator" {
			continue
		}

		needUninstall, err := checkHelmRelease(eachRelease.ReleaseName, eachRelease.Namespace)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm release check 应该成功")
		if needUninstall == true {
			err = helmUninstallRelease(eachRelease.ReleaseName, eachRelease.Namespace)
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
}

func installInfernex() {
	cfg := helmConfig["infernex"]
	GinkgoWriter.Println("start install infernex...")
	err := helmInstallRelease(cfg)
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install infernex 应当执行成功")
	time.Sleep(90 * time.Second)
	GinkgoWriter.Println("infernex has been installed")
}

var _ = Describe("infernex组件间集成测试", Label("infernex-components", "npu", "with-workload-cluster"), func() {

	BeforeEach(func() {
		setupClients()
	})

	AfterEach(func() {
		cleanupHelmReleases()
	})

	It("infernex components integration test", func() {
		var err error

		// step1: 部署其他组件
		dependencyComponents := []struct {
			key         string
			waitSeconds int
			installArgs []string
		}{
			{"logging-package", 30, nil},
			{"multi-cluster-service", 120, nil},
			{"monitoring-dashboard", 30, nil},
			{"colocation-package", 90, nil},
			{"numa-affinity-package", 60, []string{"--set", "volcano-config-package.volcano.components.enabled=false"}},
			{"many-core-orchestrator", 60, []string{"--set", "mcoPlugin.enabled=false", "--set", "kata-deploy.enabled=false"}},
			{"ray-package", 90, nil},
		}

		for _, comp := range dependencyComponents {
			cfg := helmConfig[comp.key]
			GinkgoWriter.Printf("start install %s...\n", comp.key)
			err = helmInstallRelease(cfg, comp.installArgs...)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("helm install %s 应当执行成功", comp.key))

			time.Sleep(time.Duration(comp.waitSeconds) * time.Second)
			GinkgoWriter.Printf("%s install success\n", comp.key)
		}

		// step2：部署 infernex
		installInfernex()

		// step3：infernex 部署 check
		var status bool
		Eventually(
			func() bool {
				status, err = utils.InfernexInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 5*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("infernex installation check pass")
	})
})

var _ = Describe("infernex系统间集成测试", Label("infernex-system", "npu", "with-workload-cluster"), func() {

	BeforeEach(func() {
		setupClients()
	})

	AfterEach(func() {
		cleanupHelmReleases()
	})

	It("infernex system integration test", func() {
		installInfernex()

		var status bool
		var err error
		Eventually(
			func() bool {
				status, err = utils.InfernexInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 5*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("infernex installation check pass")
	})
})

var _ = Describe("infernex异常场景集成测试", Label("infernex-anomaly", "npu", "with-workload-cluster"), func() {

	BeforeEach(func() {
		setupClients()
	})

	AfterEach(func() {
		cleanupHelmReleases()
		restoreNpuOperatorIfNeeded()
		restoreLeaderWorkerSetIfNeeded()
	})

	It("anomaly-scene: lack of npu resources or NPU-operator", func() {
		uninstallNpuOperator()
		installInfernex()

		anomaly, err := utils.InfernexAnomalyCheckWithoutNPU(k8sClient, ctx, "ai-inference")
		Expect(err).NotTo(HaveOccurred(), "infernex anomaly check should be success")
		Expect(anomaly).To(BeTrue(), "infernex should not be installed correctly on NPU/NPU-operator lacking env")
		GinkgoWriter.Println("vllm pod is pending due to insufficient NPU/NPU-operator as expected")
	})

	It("anomaly-scene: lack of leaderworkerset", func() {
		uninstallLeaderWorkerSet()

		cfg := helmConfig["infernex"]
		GinkgoWriter.Println("start install infernex...")
		err := helmInstallRelease(cfg)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).To(HaveOccurred(), "helm install infernex should be failed due to lack of leaderworkerset")
		GinkgoWriter.Println("infernex installation failed as expected")
	})

	It("anomaly scene: system reboot", Label("skip-temporarily"), func() {
		if isDaily {
			Skip("system reboot 仅在 SSH 环境执行")
		}

		GinkgoWriter.Println("=====step1: infernex install=====")
		installInfernex()

		GinkgoWriter.Println("=====step2: system reboot=====")
		err := utils.SystemReboot(sshClient, 0)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "system reboot should be success")

		time.Sleep(3 * time.Minute)
		GinkgoWriter.Println("system has been restarted")

		GinkgoWriter.Println("=====step3: rebuild ssh client and k8s client======")
		Eventually(func() error {
			port, _ := strconv.Atoi(os.Getenv("TEST_NODE3_PORT"))
			sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE3_IP"), port, os.Getenv("TEST_NODE3_USER"), os.Getenv("TEST_NODE3_PASSWORD"))
			return err
		}, 10*time.Minute, 10*time.Second).Should(Succeed(), "SSH client should be recreated successfully")
		GinkgoWriter.Println("ssh client recreated successfully")

		Eventually(func() error {
			k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
			return err
		}, 10*time.Minute, 10*time.Second).Should(Succeed(), "k8s client should be recreated successfully")
		GinkgoWriter.Println("k8s client recreated successfully")

		time.Sleep(5 * time.Minute)

		GinkgoWriter.Println("=====step4: infernex installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.InfernexInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("infernex installation check pass")
	})
})
