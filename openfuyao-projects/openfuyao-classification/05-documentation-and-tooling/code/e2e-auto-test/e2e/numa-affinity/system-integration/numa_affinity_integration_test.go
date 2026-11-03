package numa_affinity

import (
	"context"
	"fmt"
	"os"
	"strconv"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/numa-affinity/system-integration/utils"
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
		{"multi-cluster-service", 60, nil},
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

func installNumaAffinityAndCheck() {
	var err error

	numaCfg := helmConfig["numa-affinity-package"]
	GinkgoWriter.Println("start install numa-affinity-package...")
	if isDaily {
		err = helm.HelmInstallLocal(numaCfg.ReleaseName, numaCfg.HelmURrl, numaCfg.Version, numaCfg.Namespace, localexecutor)
	} else {
		err = helm.HelmInstall(numaCfg.ReleaseName, numaCfg.HelmURrl, numaCfg.Version, numaCfg.Namespace, sshClient)
	}
	if err != nil {
		GinkgoWriter.Println(err)
	}
	Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

	time.Sleep(60 * time.Second)

	var status bool
	Eventually(
		func() bool {
			status, err = utils.NumaAffinityInstallationCheck(k8sClient, ctx)
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "numa affinity installation check 应当执行成功")
			return status
		}, 2*time.Minute, 10*time.Second,
	).Should(BeTrue())
	GinkgoWriter.Println("numa affinity installation check pass")
}

var _ = Describe("numa affinity组件间集成测试", Label("numa-affinity-components", "with-workload-cluster"), func() {

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

	It("numa affinity components integration test", func() {
		installPeerComponents(basePeerComponents())
		installNumaAffinityAndCheck()
	})

	It("numa affinity & kae-operator", Label("kae"), func() {
		installPeerComponents([]peerComponent{
			{"kae-operator", 30, nil},
		})
		installNumaAffinityAndCheck()
	})

	It("numa affinity & npu-operator & infernex", Label("npu"), func() {
		installPeerComponents([]peerComponent{
			{"npu-operator", 90, nil},
			{"infernex", 120, nil},
		})
		installNumaAffinityAndCheck()
	})
})

var _ = Describe("系统间集成测试", Label("numa-affinity-system", "with-workload-cluster"), func() {

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

	It("system-scene: numa-affinity-package deploy success", func() {
		// 1. 部署numa-affinity-package
		var err error
		releaseName := helmConfig["numa-affinity-package"].ReleaseName
		helmUrl := helmConfig["numa-affinity-package"].HelmURrl
		version := helmConfig["numa-affinity-package"].Version
		namespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")
		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("numa-affinity-package has been installed")

		// 2. 检查numa-affinity部署状况
		var status bool
		Eventually(
			func() bool {
				status, err = utils.NumaAffinityInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "numa-affinity-package installation check 应当执行成功")
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("numa-affinity-package installation check pass")
	})
})

var _ = Describe("异常场景集成测试", Label("numa-affinity-anomaly", "with-workload-cluster"), func() {

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

		// delete volcano
		var flag bool
		var err error
		if isDaily {
			flag, err = helm.CheckHelmReleaseLocal("volcano", "volcano-system", localexecutor)
		} else {
			flag, err = helm.CheckHelmRelease("volcano", "volcano-system", sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "volcano helm release check should be successfully")
		if flag == true {
			if isDaily {
				err = helm.HelmUninstallLocal("volcano", "volcano-system", localexecutor)
			} else {
				err = helm.HelmUninstall("volcano", "volcano-system", sshClient)
			}
			if err != nil {
				GinkgoWriter.Println(err)
			}
			Expect(err).NotTo(HaveOccurred(), "volcano should be uninstalled successfully")
			time.Sleep(30 * time.Second)
			GinkgoWriter.Println("volcano has been uninstalled successfully")
		}

		err = k8sClient.DeleteNamespace(ctx, "volcano-system")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ns: volcano-system should be deleted successfully")
		time.Sleep(15 * time.Second)
		GinkgoWriter.Println("ns: volcano-system has been deleted successfully")
	})

	It("anomaly-scene: NPU operator has been installed", Label("npu", "skip-temporarily"), func() {
		// step1: install npu-operator
		var err error
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

		time.Sleep(60 * time.Second)
		GinkgoWriter.Println("npu-operator install success")

		// step2: install numa affinity
		releaseName := helmConfig["numa-affinity-package"].ReleaseName
		helmUrl := helmConfig["numa-affinity-package"].HelmURrl
		version := helmConfig["numa-affinity-package"].Version
		namespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).To(HaveOccurred(), "numa-affinity should not be installed correctly due to the difference of volcano")
		GinkgoWriter.Println("numa-affinity should not be installed successfully as expected")
	})

	It("anomaly scene: ns volcano-system already exist", func() {

		var err error

		// step1: create ns volcano-system
		err = k8sClient.CreateNamespaceIfNotExists(ctx, "volcano-system")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ns: volcano system should be created successfully")
		GinkgoWriter.Println("ns: volcano-system has been created")

		// step2: numa-affinity isntall
		releaseName := helmConfig["numa-affinity-package"].ReleaseName
		helmUrl := helmConfig["numa-affinity-package"].HelmURrl
		version := helmConfig["numa-affinity-package"].Version
		namespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa affinity...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package 应当执行成功")

		time.Sleep(60 * time.Second)

		// step3: numa affinity installation check
		var status bool
		Eventually(
			func() bool {
				status, err = utils.NumaAffinityInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				Expect(err).NotTo(HaveOccurred(), "numa affinity installation check 应当执行成功")
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("numa-affinity should be installed correctly although ns volcano-system already exist")
	})

	It("anomaly scene: volcano has been installed", Label("skip-temporarily"), func() {
		// step1: volcano install
		var err error

		if isDaily {
			err = utils.PreOpBeforeVolcanoInstallationLocal(localexecutor)
		} else {
			err = utils.PreOpBeforeVolcanoInstallation(sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "volcano helm repo should be added and updated successfully")

		err = k8sClient.CreateNamespaceIfNotExists(ctx, "volcano-system")
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "ns: volcano-system should existed")

		if isDaily {
			err = helm.HelmInstallLocal("volcano", "volcano-sh/volcano", "1.9.0", "volcano-system", localexecutor)
		} else {
			err = helm.HelmInstall("volcano", "volcano-sh/volcano", "1.9.0", "volcano-system", sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "volcano should be installed successfully")

		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("volcano install success")

		// step2: numa install
		releaseName := helmConfig["numa-affinity-package"].ReleaseName
		helmUrl := helmConfig["numa-affinity-package"].HelmURrl
		version := helmConfig["numa-affinity-package"].Version
		namespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		if isDaily {
			err = helm.HelmInstallLocal(releaseName, helmUrl, version, namespace, localexecutor)
		} else {
			err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		}
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).To(HaveOccurred(), "numa-affinity should not be installed correctly due to the difference of volcano")
		GinkgoWriter.Println("numa-affinity should not be installed successfully as expected")
	})

	It("anomaly scene: system reboot", Label("skip-temporarily"), func() {

		var err error

		// step1: numa-affinity-package install
		GinkgoWriter.Println("=====step1: numa-affinity-package install=====")
		releaseName := helmConfig["numa-affinity-package"].ReleaseName
		helmUrl := helmConfig["numa-affinity-package"].HelmURrl
		version := helmConfig["numa-affinity-package"].Version
		namespace := helmConfig["numa-affinity-package"].Namespace

		GinkgoWriter.Println("start install numa-affinity-package...")
		err = helm.HelmInstall(releaseName, helmUrl, version, namespace, sshClient)
		if err != nil {
			GinkgoWriter.Println(err)
		}
		Expect(err).NotTo(HaveOccurred(), "helm install numa-affinity-package should be successfully")

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

		// step4: ray-package installation check
		GinkgoWriter.Println("=====step4: ray-package installation check=====")
		var status bool
		Eventually(
			func() bool {
				status, err = utils.NumaAffinityInstallationCheck(k8sClient, ctx)
				if err != nil {
					GinkgoWriter.Println(err)
				}
				return status
			}, 2*time.Minute, 10*time.Second,
		).Should(BeTrue())
		GinkgoWriter.Println("numa-affinity-package installation check pass")
	})
})
