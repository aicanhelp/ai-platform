package numa_affinity_functional

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/helm"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/numa-affinity/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func captureBusinessPodBaseline() []utils.PodSnapshot {
	By("准备业务负载并生成重启前快照")
	createdByCase, err := utils.EnsureBusinessWorkload(sshClient, cfg)
	Expect(err).NotTo(HaveOccurred(), "should ensure business workload")
	if createdByCase {
		DeferCleanup(func() {
			utils.DeleteBusinessWorkload(sshClient, cfg)
		})
	}

	var baseline []utils.PodSnapshot
	Eventually(func() error {
		baseline, err = utils.GetRunningPodSnapshots(sshClient, cfg.BizNamespace, cfg.BizLabelSelector)
		if err != nil {
			// 记录诊断信息，便于在 Eventually 超时失败时定位根因
			GinkgoWriter.Printf("等待业务 Pod 就绪中: %v\n%s\n", err, utils.CollectPodDiagnostics(sshClient, cfg.BizNamespace, cfg.BizLabelSelector))
			return err
		}
		for _, pod := range baseline {
			if !pod.Ready {
				return fmt.Errorf("pod is not ready yet: %s", pod.Name)
			}
		}
		return nil
	}, 5*time.Minute, 5*time.Second).Should(Succeed())

	return baseline
}

func assertBusinessPodsStableAgainst(baseline []utils.PodSnapshot) {
	By("校验重启后业务 Pod 未被驱逐重建且无重启计数增长")
	Eventually(func() error {
		current, currentErr := utils.GetRunningPodSnapshots(sshClient, cfg.BizNamespace, cfg.BizLabelSelector)
		if currentErr != nil {
			return currentErr
		}
		return utils.AssertPodSnapshotsStable(baseline, current)
	}, 3*time.Minute, 5*time.Second).Should(Succeed())
}

var _ = SIGDescribe("NUMA亲和插件功能测试", Ordered, Label("numa-affinity-functional", "with-workload-cluster"), func() {
	It("【NUMA2503-040】亲和策略配置-运行时Pod亲和优化-自动化配置-策略未打开-重启containerd对现有业务的影响", Label("NUMA2503-040", "P2", "e2e"), func() {
		baseline := captureBusinessPodBaseline()

		By("重启 containerd 服务")
		err := utils.RestartSystemdServiceAndWaitActive(sshClient, cfg.ContainerdServiceName)
		Expect(err).NotTo(HaveOccurred(), "containerd should be active after restart")

		assertBusinessPodsStableAgainst(baseline)
	})

	It("【NUMA2503-025】亲和策略配置-最优NUMAdistance-自动化配置-策略未打开-重启kubelet对现有业务的影响", Label("NUMA2503-025", "P2", "e2e"), func() {
		By("备份 kubelet 配置文件")
		backupPath, err := utils.BackupFile(sshClient, cfg.KubeletConfigPath)
		Expect(err).NotTo(HaveOccurred(), "should backup kubelet config")

		DeferCleanup(func() {
			if !cfg.AutoRestore {
				utils.RemoveFile(sshClient, backupPath)
				return
			}

			restoreErr := utils.RestoreFile(sshClient, backupPath, cfg.KubeletConfigPath)
			Expect(restoreErr).NotTo(HaveOccurred(), "should restore kubelet config")

			restartErr := utils.RestartSystemdServiceAndWaitActive(sshClient, cfg.KubeletServiceName)
			Expect(restartErr).NotTo(HaveOccurred(), "kubelet should be active after restore restart")

			utils.RemoveFile(sshClient, backupPath)
		})

		By("确保 prefer-closest-numa-nodes 字段不为 true")
		_, err = utils.EnsurePreferClosestNumaNodesFalse(sshClient, cfg.KubeletConfigPath)
		Expect(err).NotTo(HaveOccurred(), "should ensure prefer-closest-numa-nodes is false or unchanged")

		baseline := captureBusinessPodBaseline()

		By("重启 kubelet 服务")
		err = utils.RestartSystemdServiceAndWaitActive(sshClient, cfg.KubeletServiceName)
		Expect(err).NotTo(HaveOccurred(), "kubelet should be active after restart")

		assertBusinessPodsStableAgainst(baseline)
	})

	It("【NUMA2503-052】独立部署-部署后volcano基础功能验证", Label("NUMA2503-052", "P0", "e2e"), func() {
		By("检查 volcano-scheduler-configmap 中存在 numa-aware 配置")
		configMapCmd := fmt.Sprintf("kubectl get cm %s -n %s -o yaml", cfg.VolcanoConfigMapName, cfg.VolcanoNamespace)
		configMapContent, err := utils.ExecStdout(sshClient, configMapCmd)
		Expect(err).NotTo(HaveOccurred(), "should query volcano scheduler configmap")
		Expect(configMapContent).To(ContainSubstring("numa-aware"), "volcano configmap should contain numa-aware plugin")

		namespace := utils.GenerateCaseNamespace("numa2503-052")
		deploymentName := "numa-functional-volcano"
		podLabel := "numa-functional-volcano"

		By("准备测试命名空间")
		_, err = utils.ExecStdout(sshClient, fmt.Sprintf("kubectl create namespace %s --dry-run=client -o yaml | kubectl apply -f -", namespace))
		Expect(err).NotTo(HaveOccurred(), "should create or update test namespace")

		DeferCleanup(func() {
			_, _ = utils.ExecStdout(sshClient, fmt.Sprintf("kubectl delete deployment %s -n %s --ignore-not-found=true", deploymentName, namespace))
			_, _ = utils.ExecStdout(sshClient, fmt.Sprintf("kubectl delete namespace %s --ignore-not-found=true", namespace))
		})

		By("部署 schedulerName=volcano 的测试负载")
		yamlContent := utils.BuildVolcanoDeploymentYAML(namespace, deploymentName, podLabel, cfg.VolcanoSchedulerName)
		applyCmd := fmt.Sprintf("cat <<'EOF' | kubectl apply -f -\n%s\nEOF", yamlContent)
		_, err = utils.ExecStdout(sshClient, applyCmd)
		Expect(err).NotTo(HaveOccurred(), "should apply test deployment")

		By("校验 Pod 使用的调度器为 volcano，而不是 kube-scheduler")
		Eventually(func() error {
			podCmd := fmt.Sprintf("kubectl get pods -n %s -l app=%s -o jsonpath='{.items[0].metadata.name}'", namespace, podLabel)
			podNameRaw, podErr := utils.ExecStdout(sshClient, podCmd)
			if podErr != nil {
				return podErr
			}
			podName := utils.NormalizeOutput(podNameRaw)
			if podName == "" {
				return fmt.Errorf("pod has not been created yet")
			}

			schedulerCmd := fmt.Sprintf("kubectl get pod %s -n %s -o jsonpath='{.spec.schedulerName}'", podName, namespace)
			schedulerRaw, schedulerErr := utils.ExecStdout(sshClient, schedulerCmd)
			if schedulerErr != nil {
				return schedulerErr
			}

			schedulerName := utils.NormalizeOutput(schedulerRaw)
			if schedulerName == "" {
				return fmt.Errorf("scheduler name is empty")
			}
			if schedulerName != cfg.VolcanoSchedulerName {
				return fmt.Errorf("expected scheduler %q but got %q", cfg.VolcanoSchedulerName, schedulerName)
			}
			if strings.EqualFold(schedulerName, "kube-scheduler") {
				return fmt.Errorf("scheduler should not be kube-scheduler")
			}

			return nil
		}, 2*time.Minute, 5*time.Second).Should(Succeed())
	})

	It("【NUMA2503-053】独立部署-helm卸载-基础功能验证", Label("NUMA2503-053", "P0", "e2e", "destructive"), func() {
		By("确认待卸载 release 当前存在")
		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred(), "helm list should run successfully")
		Expect(exists).To(BeTrue(), "target release should exist before uninstall")

		if cfg.AutoRestore {
			DeferCleanup(func() {
				currentExists, checkErr := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
				Expect(checkErr).NotTo(HaveOccurred(), "helm list should be available in cleanup")
				if currentExists {
					return
				}
				Expect(cfg.HelmURL).NotTo(Equal(""), "helm url should not be empty when auto restore is enabled")
				Expect(cfg.HelmVersion).NotTo(Equal(""), "helm version should not be empty when auto restore is enabled")

				installErr := helm.HelmInstall(cfg.HelmReleaseName, cfg.HelmURL, cfg.HelmVersion, cfg.HelmNamespace, sshClient)
				Expect(installErr).NotTo(HaveOccurred(), "helm install should restore released component")

				Eventually(func() bool {
					restoredExists, restoredErr := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
					Expect(restoredErr).NotTo(HaveOccurred(), "helm list should work when checking restoration")
					return restoredExists
				}, 2*time.Minute, 5*time.Second).Should(BeTrue())

				// 重装后等待 volcano 核心组件就绪，避免后续用例与 admission webhook 竞态
				Eventually(func() error {
					return utils.WaitForVolcanoComponentsReady(sshClient, cfg.VolcanoNamespace, 10*time.Second)
				}, 5*time.Minute, 5*time.Second).Should(Succeed(), "重装后 volcano 核心组件应在 5 分钟内就绪")
			})
		}

		By("执行 helm uninstall")
		err = helm.HelmUninstall(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred(), "helm uninstall should succeed")

		By("确认 release 已从 helm list 中移除")
		Eventually(func() bool {
			exists, err = helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
			Expect(err).NotTo(HaveOccurred(), "helm list should still be available after uninstall")
			return !exists
		}, 2*time.Minute, 5*time.Second).Should(BeTrue())
	})

	It("【NUMA2503-033】亲和策略配置-运行时Pod亲和优化-自动化配置-策略未打开-基础功能验证", Label("NUMA2503-033", "P0", "e2e", "skip-temporarily"), func() {
		By("检查 /etc/containerd/config.toml 中是否存在 NRI 插件配置并启用")
		containerdConfCmd := "cat /etc/containerd/config.toml"
		out, err := utils.ExecStdout(sshClient, containerdConfCmd)
		Expect(err).NotTo(HaveOccurred(), "should read containerd config")

		nriSectionRe := regexp.MustCompile(`\[plugins\.(?:"|')io\.containerd\.nri\.v1\.nri(?:"|')\]`)
		disableFalseRe := regexp.MustCompile(`(?m)^\s*disable\s*=\s*false\s*$`)
		if !nriSectionRe.MatchString(out) || !disableFalseRe.MatchString(out) {
			Skip("当前环境 /etc/containerd/config.toml 未找到 NRI 插件或 disable = false 开启配置，跳过用例")
		}

		By("检查 net_rship 进程是否运行 (oeaware 部分)")
		netRshipCmd := "ps -ef | grep net_rship | grep -v grep"
		out, err = utils.ExecStdout(sshClient, netRshipCmd)
		if err != nil || out == "" {
			Skip("当前环境未找到正在运行的 net_rship 进程 (oeAware/NUMAfast插件尚未安装或启动)，跳过用例")
		}
	})

	It("【NUMA2503-009】亲和策略配置-NUMA亲和性调度-static策略未打开-重启kubelet对现有业务的影响", Label("NUMA2503-009", "P2", "e2e"), func() {
		By("备份 kubelet 配置文件")
		backupPath, err := utils.BackupFile(sshClient, cfg.KubeletConfigPath)
		Expect(err).NotTo(HaveOccurred(), "should backup kubelet config")

		DeferCleanup(func() {
			if !cfg.AutoRestore {
				utils.RemoveFile(sshClient, backupPath)
				return
			}

			restoreErr := utils.RestoreFile(sshClient, backupPath, cfg.KubeletConfigPath)
			Expect(restoreErr).NotTo(HaveOccurred(), "should restore kubelet config")

			restartErr := utils.RestartSystemdServiceAndWaitActive(sshClient, cfg.KubeletServiceName)
			Expect(restartErr).NotTo(HaveOccurred(), "kubelet should be active after restore restart")

			utils.RemoveFile(sshClient, backupPath)
		})

		By("确保 cpuManagerPolicy 为 none (static 策略未打开)")
		_, err = utils.ExecStdout(sshClient, fmt.Sprintf("sed -i 's/cpuManagerPolicy: static/cpuManagerPolicy: none/g' %s", cfg.KubeletConfigPath))
		Expect(err).NotTo(HaveOccurred(), "should set cpuManagerPolicy to none")

		baseline := captureBusinessPodBaseline()

		By("重启 kubelet 服务")
		err = utils.RestartSystemdServiceAndWaitActive(sshClient, cfg.KubeletServiceName)
		Expect(err).NotTo(HaveOccurred(), "kubelet should be active after restart")

		assertBusinessPodsStableAgainst(baseline)
	})

	It("【NUMA2503-054】独立部署-环境正交-openEuler22.03-x86", Label("NUMA2503-054", "P2", "e2e", "skip-temporarily"), func() {
		By("检查当前架构是否为 x86_64")
		archCmd := "uname -m"
		arch, err := utils.ExecStdout(sshClient, archCmd)
		Expect(err).NotTo(HaveOccurred(), "should execute uname successfully")
		if !strings.Contains(arch, "x86_64") {
			Skip("当前环境非 x86_64 架构，跳过 x86 正交部署测试")
		}

		By("确认 components 已在 x86 环境独立部署成功")
		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred(), "helm list should execute successfully")
		Expect(exists).To(BeTrue(), "numa-affinity-package should be installed successfully on x86")
	})

	It("【NUMA2503-055】独立部署-环境正交-openEuler22.03-arm", Label("NUMA2503-055", "P2", "e2e"), func() {
		By("检查当前架构是否为 aarch64")
		archCmd := "uname -m"
		arch, err := utils.ExecStdout(sshClient, archCmd)
		Expect(err).NotTo(HaveOccurred(), "should execute uname successfully")
		if !strings.Contains(arch, "aarch64") {
			Skip("当前环境非 arm 架构，跳过 arm 正交部署测试")
		}

		By("确认 components 已在 arm 环境独立部署成功")
		exists, err := helm.CheckHelmRelease(cfg.HelmReleaseName, cfg.HelmNamespace, sshClient)
		Expect(err).NotTo(HaveOccurred(), "helm list should execute successfully")
		Expect(exists).To(BeTrue(), "numa-affinity-package should be installed successfully on arm")
	})

})
