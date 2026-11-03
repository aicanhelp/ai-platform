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
	"gitcode.com/openFuyao/e2e-auto-test/e2e/logging/system-integration/utils"
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
		{"multi-cluster-service", 60, nil},
		{"numa-affinity-package", 60, nil},
		{"monitoring-dashboard", 30, nil},
		{"colocation-package", 60, nil},
		{"many-core-orchestrator", 60, []string{"--set", "mcoPlugin.enabled=false", "--set", "kata-deploy.enabled=false"}},
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

func installLoggingAndCheck() {
	var err error

	loggingCfg := helmConfig["logging-package"]
	GinkgoWriter.Println("start install logging-package...")
	if isDaily {
		err = helm.HelmInstallLocal(loggingCfg.ReleaseName, loggingCfg.HelmURrl, loggingCfg.Version, loggingCfg.Namespace, localexecutor)
	} else {
		err = helm.HelmInstall(loggingCfg.ReleaseName, loggingCfg.HelmURrl, loggingCfg.Version, loggingCfg.Namespace, sshClient)
	}
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install logging-package 应当执行成功")

	time.Sleep(30 * time.Second)

	var status bool
	Eventually(
		func() bool {
			status, err = utils.LoggingInstallationCheck(k8sClient, ctx)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "logging-package installation check 应当执行成功")
			return status
		}, 2*time.Minute, 10*time.Second,
	).Should(BeTrue())
	GinkgoWriter.Println("logging-package installation check pass")
}

var _ = Describe("logging组件间集成测试", Label("logging-components", "with-workload-cluster"), func() {

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

	It("logging components integration test", func() {
		installPeerComponents(basePeerComponents())
		installLoggingAndCheck()
	})

	It("logging-package & kae-operator", Label("kae"), func() {
		installPeerComponents([]peerComponent{
			{"kae-operator", 60, nil},
		})
		installLoggingAndCheck()
	})

	It("logging-package & npu-operator & infernex", Label("npu"), func() {
		installPeerComponents([]peerComponent{
			{"npu-operator", 90, nil},
			{"infernex", 120, nil},
		})
		installLoggingAndCheck()
	})
})

var _ = Describe("组件与系统集成测试", Label("logging-system", "with-workload-cluster"), func() {

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

		// step1: logging-package install
		GinkgoWriter.Println("=====step1: logging-package install=====")
		releaseName := helmConfig["logging-package"].ReleaseName
		helmUrl := helmConfig["logging-package"].HelmURrl
		version := helmConfig["logging-package"].Version
		namespace := helmConfig["logging-package"].Namespace

		GinkgoWriter.Println("start install logging-package...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package should be successfully")

		time.Sleep(30 * time.Second)

		// step2: logging-package installation check
		GinkgoWriter.Println("=====step2: logging-package installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.LoggingInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "logging-package installation check 应当执行成功")
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("logging-package installation check pass")
	})
})

var _ = Describe("异常场景测试", Label("logging-anomaly", "with-workload-cluster"), func() {

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

		// step1: logging-package install
		GinkgoWriter.Println("=====step2: logging-package install=====")
		releaseName := helmConfig["logging-package"].ReleaseName
		helmUrl := helmConfig["logging-package"].HelmURrl
		version := helmConfig["logging-package"].Version
		namespace := helmConfig["logging-package"].Namespace

		GinkgoWriter.Println("start install logging-package...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install logging-package should be successfully")

		time.Sleep(30 * time.Second)

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

		// step4: logging-package installation check
		GinkgoWriter.Println("=====step4: logging-package installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.LoggingInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "logging-package installation check 应当执行成功")
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("logging-package installation check pass")
	})
})
