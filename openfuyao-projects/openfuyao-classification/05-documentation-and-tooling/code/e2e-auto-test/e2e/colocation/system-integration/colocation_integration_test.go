package colocation

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/system-integration/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
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
		{"multi-cluster-service", 90, nil},
		{"numa-affinity-package", 60, nil},
		{"monitoring-dashboard", 30, nil},
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

func installColocationAndCheck() {
	var err error

	colocationCfg := helmConfig["colocation-package"]
	GinkgoWriter.Println("start install colocation-package...")
	if isDaily {
		err = helm.HelmInstallLocal(colocationCfg.ReleaseName, colocationCfg.HelmURrl, colocationCfg.Version, colocationCfg.Namespace, localexecutor)
	} else {
		err = helm.HelmInstall(colocationCfg.ReleaseName, colocationCfg.HelmURrl, colocationCfg.Version, colocationCfg.Namespace, sshClient)
	}
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

	time.Sleep(60 * time.Second)

	var status bool
	Eventually(
		func() bool {
			status, err = utils.ColocationInstallationCheck(k8sClient, ctx)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "colocation installation check 应当执行成功")
			return status
		}, 2*time.Minute, 10*time.Second,
	).Should(BeTrue())
	GinkgoWriter.Println("colocation installation check pass")
}

var _ = Describe("colocation 组件间集成测试", Label("colocation-components", "with-workload-cluster"), func() {

	BeforeEach(func() {

		var err error

		// context
		ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
		DeferCleanup(cancel)

		// ssh client
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

	It("colocation components integration test", func() {
		installPeerComponents(basePeerComponents())
		installColocationAndCheck()
	})

	It("colocation & kae-operator", Label("kae"), func() {
		installPeerComponents([]peerComponent{
			{"kae-operator", 60, nil},
		})
		installColocationAndCheck()
	})

	It("colocation & npu-operator & infernex", Label("npu"), func() {
		installPeerComponents([]peerComponent{
			{"npu-operator", 90, nil},
			{"infernex", 120, nil},
		})
		installColocationAndCheck()
	})
})

var _ = Describe("colocation 系统间集成测试", Label("colocation-system", "with-workload-cluster"), func() {

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

	It("system-scene: colocation deploy successfully", func() {

		var err error

		releaseName := helmConfig["colocation-package"].ReleaseName
		helmUrl := helmConfig["colocation-package"].HelmURrl
		version := helmConfig["colocation-package"].Version
		namespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package 应当执行成功")

		time.Sleep(60 * time.Second)

		// step4: colocation installation check
		var status bool
		Eventually(
			func() bool {
				status, err = utils.ColocationInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "colocation installation check 应当执行成功")
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("colocation installation check pass")
	})
})

var _ = Describe("colocation 异常场景测试", Label("colocation-anomaly", "with-workload-cluster"), func() {

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

		k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
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

	It("anomaly scene: lack of Prometheus", Label("skip-temporarily"), func() {

		var err error

		releaseName := helmConfig["colocation-package"].ReleaseName
		helmUrl := helmConfig["colocation-package"].HelmURrl
		version := helmConfig["colocation-package"].Version
		namespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package without openFuyao...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor, "--set colocation-website.openFuyao=true", "--set colocation-website.enableOAuth=true")
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient, "--set colocation-website.openFuyao=true", "--set colocation-website.enableOAuth=true")
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).To(HaveOccurred(), "colocation should not be installed successfully due to insufficiency of Prometheus")
		GinkgoWriter.Println("colocation installed failed as expected")
	})

	It("anomaly scene: system reboot", Label("skip-temporarily"), func() {

		var err error

		// step1: colocation-package install
		GinkgoWriter.Println("=====step1: colocation-package install=====")
		releaseName := helmConfig["colocation-package"].ReleaseName
		helmUrl := helmConfig["colocation-package"].HelmURrl
		version := helmConfig["colocation-package"].Version
		namespace := helmConfig["colocation-package"].Namespace

		GinkgoWriter.Println("start install colocation-package...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install colocation-package should be successfully")

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

		// step4: colocation-package installation check
		GinkgoWriter.Println("=====step4: colocation-package installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.ColocationInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("colocation-package installation check pass")
	})
})
