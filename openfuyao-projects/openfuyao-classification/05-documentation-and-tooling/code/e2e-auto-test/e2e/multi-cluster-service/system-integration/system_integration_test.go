package system_integration

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/multi-cluster-service/system-integration/utils"
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

type peerComponent struct {
	key         string
	waitSeconds int
	installArgs []string
}

func basePeerComponents() []peerComponent {
	return []peerComponent{
		{"logging-package", 30, nil},
		{"numa-affinity-package", 60, nil},
		{"many-core-orchestrator", 60, []string{"--set", "mcoPlugin.enabled=false", "--set", "kata-deploy.enabled=false"}},
		{"monitoring-dashboard", 30, nil},
		{"colocation-package", 60, nil},
		{"ray-package", 60, nil},
	}
}

func installPeerComponents(components []peerComponent) {
	var err error
	for _, comp := range components {
		if comp.key == "infernex" {
			err = k8sClient.CreateNamespaceIfNotExists(ctx, "ai-inference")
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "ns: ai-inference should be created successfully")
			GinkgoWriter.Println("ns: ai-inference has been created")
		}

		cfg := helmConfig[comp.key]
		GinkgoWriter.Printf("start install %s...\n", comp.key)
		if isDaily {
			err = helm.HelmInstallLocal(cfg.ReleaseName, cfg.HelmURrl, cfg.Version, cfg.Namespace, localexecutor, comp.installArgs...)
		} else {
			err = helm.HelmInstall(cfg.ReleaseName, cfg.HelmURrl, cfg.Version, cfg.Namespace, sshClient, comp.installArgs...)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("helm install %s 应当执行成功", comp.key))

		time.Sleep(time.Duration(comp.waitSeconds) * time.Second)
		GinkgoWriter.Printf("%s install success\n", comp.key)
	}
}

func installMCSAndCheck() {
	var err error

	mcsCfg := helmConfig["multi-cluster-service"]
	GinkgoWriter.Println("start install multi-cluster-service...")
	if isDaily {
		err = helm.HelmInstallLocal(mcsCfg.ReleaseName, mcsCfg.HelmURrl, mcsCfg.Version, mcsCfg.Namespace, localexecutor)
	} else {
		err = helm.HelmInstall(mcsCfg.ReleaseName, mcsCfg.HelmURrl, mcsCfg.Version, mcsCfg.Namespace, sshClient)
	}
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service 应当执行成功")

	time.Sleep(60 * time.Second)

	var status bool
	Eventually(
		func() bool {
			status, err = utils.MCSInstallationCheck(k8sClient, ctx)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			return status
		}, 2*time.Minute, 10*time.Second,
	).Should(BeTrue())
	GinkgoWriter.Println("multi-cluster-service installation check pass")
}

var _ = Describe("mcs组件间集成测试", Label("mcs-components", "with-workload-cluster"), func() {

	BeforeEach(func() {

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		var err error
		if isDaily {
			localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		} else {
			port, _ := strconv.Atoi(os.Getenv("TEST_NODE5_PORT"))
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

	It("mcs components integration test", func() {
		installPeerComponents(basePeerComponents())
		installMCSAndCheck()
	})

	It("multi-cluster-service & kae-operator", Label("kae"), func() {
		installPeerComponents([]peerComponent{
			{"kae-operator", 60, nil},
		})
		installMCSAndCheck()
	})

	It("multi-cluster-service & npu-operator & infernex", Label("npu"), func() {
		installPeerComponents([]peerComponent{
			{"npu-operator", 90, nil},
			{"infernex", 120, nil},
		})
		installMCSAndCheck()
	})
})

var _ = Describe("组件与系统集成测试", Label("mcs-system", "with-workload-cluster"), func() {

	BeforeEach(func() {

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		var err error
		if isDaily {
			localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		} else {
			port, _ := strconv.Atoi(os.Getenv("TEST_NODE5_PORT"))
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

	It("system scene: installation success", func() {

		var err error

		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		releaseName := helmConfig["multi-cluster-service"].ReleaseName
		helmUrl := helmConfig["multi-cluster-service"].HelmURrl
		version := helmConfig["multi-cluster-service"].Version
		namespace := helmConfig["multi-cluster-service"].Namespace

		GinkgoWriter.Println("start install multi-cluster-service...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service should be successfully")

		time.Sleep(60 * time.Second)

		// step2: multi-cluster-service installation check
		GinkgoWriter.Println("=====step2: multi-cluster-service installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.MCSInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("multi-cluster-service installation check pass")
	})
})

var _ = Describe("异常场景测试", Ordered, Label("mcs-anomaly", "with-workload-cluster"), func() {

	BeforeEach(func() {

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
		var err error
		if isDaily {
			localexecutor = executor.NewLocalExecutor(10 * time.Minute)
		} else {
			port, _ := strconv.Atoi(os.Getenv("TEST_NODE5_PORT"))
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

	It("anomaly scene: system reboot", Label("skip-temporarily"), func() {

		var err error

		// step1: multi-cluster-service install
		GinkgoWriter.Println("=====step1: multi-cluster-service install=====")
		releaseName := helmConfig["multi-cluster-service"].ReleaseName
		helmUrl := helmConfig["multi-cluster-service"].HelmURrl
		version := helmConfig["multi-cluster-service"].Version
		namespace := helmConfig["multi-cluster-service"].Namespace

		GinkgoWriter.Println("start install multi-cluster-service...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install multi-cluster-service should be successfully")

		time.Sleep(90 * time.Second)

		// step2: system reboot
		GinkgoWriter.Println("=====step2: system reboot=====")
		err = utils.SystemReboot(sshClient, 0)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "system reboot should be success")

		time.Sleep(3 * time.Minute)
		GinkgoWriter.Println("system has been restarted")

		// step3: rebuild ssh client and k8s client
		GinkgoWriter.Println("=====step3: rebuild ssh client and k8s client======")
		Eventually(func() error {
			port, _ := strconv.Atoi(os.Getenv("TEST_NODE5_PORT"))
			sshClient, err = executor.NewSSHExecutor(os.Getenv("TEST_NODE5_IP"), port, os.Getenv("TEST_NODE5_USER"), os.Getenv("TEST_NODE5_PASSWORD"))
			return err
		}, 10*time.Minute, 10*time.Second).Should(Succeed(), "SSH client should be recreated successfully")
		GinkgoWriter.Println("ssh client recreated successfully")

		Eventually(func() error {
			if isDaily {
				k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
			} else {
				k8sClient, err = k8s.NewK8SClientViaSSH(sshClient)
			}
			return err
		}, 10*time.Minute, 10*time.Second).Should(Succeed(), "k8s client should be recreated successfully")
		GinkgoWriter.Println("k8s client recreated successfully")

		time.Sleep(5 * time.Minute)

		// step4: multi-cluster-service installation check
		GinkgoWriter.Println("=====step4: multi-cluster-service installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.MCSInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeFalse())
		GinkgoWriter.Println("multi-cluster-service not ready after system restarted as expected")
	})
})
