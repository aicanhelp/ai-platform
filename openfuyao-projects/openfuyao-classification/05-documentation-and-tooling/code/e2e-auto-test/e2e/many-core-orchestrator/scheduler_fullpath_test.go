package mco

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

var _ = Describe("众核调度插件全链路测试", Ordered, Label("many-core-orchestrator", "scheduler", "scheduler-fullpath", "with-workload-cluster"), func() {
	var installedByTest bool

	BeforeAll(func(ctx SpecContext) {
		By("清理历史测试 release 和残留 CR")
		Expect(UninstallExistingReleaseAndWait(ctx, WaitSchedulerResourcesDeleted)).To(Succeed())
		// 清理可能残留的 NodeInterferenceReport CR（Filter 测试可能留了 patch 数据）
		_ = DeleteNodeInterferenceReport(ctx, MCONodeInterferenceReport)
	}, NodeTimeout(5*time.Minute))

	AfterAll(func(ctx SpecContext) {
		if installedByTest {
			By("卸载 MCO 测试 release 并等待清理完成")
			_ = UninstallExistingReleaseAndWait(ctx, WaitSchedulerResourcesDeleted)
		}
	}, NodeTimeout(5*time.Minute))

	It("众核-SCHED-IT-014 Helm 安装部署", SpecTimeout(3*time.Minute), func(ctx SpecContext) {
		By("执行 helm install 部署 chart（volcano.enabled=true, mcoPlugin.enabled=true）")
		chartPath := GetChartPath()
		Expect(HelmInstallLocalChart(ctx, MCOHelmConfig, chartPath, SchedulerPhaseHelmSets...)).To(Succeed())
		installedByTest = true
		Expect(WaitHelmReleaseDeployed(ctx, MCOHelmConfig["releaseName"], 120*time.Second, 5*time.Second)).To(Succeed())

		By("等待 mco-volcano-scheduler Pod Running")
		Expect(WaitDeploymentReady(k8sClient, ctx, MCOSchedulerDeploymentName, MCOVolcanoNamespace, 120*time.Second)).To(Succeed())

		By("等待 Volcano admission webhook 就绪")
		Expect(WaitVolcanoAdmissionReady(k8sClient, ctx, 120*time.Second)).To(Succeed())

		By("验证 mco-volcano-scheduler Pod 日志包含 mco-plugin 初始化信息且无 panic/fatal")
		pods, err := GetSchedulerPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty(), "mco-volcano-scheduler Pod 应存在")

		for _, pod := range pods.Items {
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "mco-volcano-scheduler Pod 应为 Running: %s/%s", pod.Namespace, pod.Name)
			for _, cs := range pod.Status.ContainerStatuses {
				Expect(cs.Ready).To(BeTrue(), "mco-volcano-scheduler 容器应为 Ready: %s/%s/%s", pod.Namespace, pod.Name, cs.Name)
			}

			logs, err := GetContainerLogs(k8sClient, ctx, &pod)
			Expect(err).NotTo(HaveOccurred())
			Expect(logs).NotTo(ContainSubstring("panic"), "调度器日志不应包含 panic")
			Expect(logs).NotTo(ContainSubstring("fatal"), "调度器日志不应包含 fatal")
		}
	})

	It("众核-SCHED-IT-015 调度器接管 Volcano Job", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		jobName := fmt.Sprintf("test-takeover-%d", time.Now().UnixNano())

		By("创建 Volcano Job（schedulerName: volcano，webhook 瞬态错误自动重试）")
		var createErr error
		for retry := 0; retry < 5; retry++ {
			_, createErr = CreateVolcanoJob(k8sClient, ctx, jobName, KataTestNamespace)
			if createErr == nil {
				break
			}
			if retry < 4 {
				By(fmt.Sprintf("Volcano Job 创建失败（%v），2s 后重试 (%d/5)", createErr, retry+1))
				time.Sleep(2 * time.Second)
			}
		}
		Expect(createErr).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeleteVolcanoJob(k8sClient, ctx, jobName, KataTestNamespace)
			time.Sleep(3 * time.Second)
		})

		By("等待 Volcano Job 的 Pod 进入 Running 状态")
		Eventually(func(g Gomega) {
			pods, err := k8sClient.Clientset.CoreV1().Pods(KataTestNamespace).List(ctx, metav1.ListOptions{
				LabelSelector: fmt.Sprintf("volcano.sh/job-name=%s", jobName),
			})
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(pods.Items).NotTo(BeEmpty(), "Volcano Job 应创建 Pod")
			for _, pod := range pods.Items {
				g.Expect(pod.Status.Phase).To(Or(Equal(corev1.PodRunning), Equal(corev1.PodSucceeded)),
					"Volcano Job Pod 应为 Running 或 Succeeded: %s/%s phase=%s", pod.Namespace, pod.Name, pod.Status.Phase)
			}
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		By("验证调度器日志包含 mco-plugin 调度决策记录")
		_, err := WaitSchedulerLogsContainAny(k8sClient, ctx, []string{"mco-plugin", "mco_plugin", "[mco-plugin]", "score:", "filter:"}, 30*time.Second)
		Expect(err).NotTo(HaveOccurred(), "调度器日志应包含 mco-plugin 调度决策记录")
	})

	It("众核-SCHED-IT-016 Filter/Score 决策日志验证", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		podName := fmt.Sprintf("test-log-%d", time.Now().UnixNano())

		By("创建使用 volcano 调度器的测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, podName, KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, podName, KataTestNamespace, 30*time.Second)
		})

		By("等待 Pod 被调度")
		_, err = WaitPodScheduledOnNode(k8sClient, ctx, podName, KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred(), "Pod 应被成功调度")

		By("检查调度器日志包含 filter 和 score 决策记录")
		time.Sleep(2 * time.Second)
		pods, err := GetSchedulerPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty())

		var allLogs string
		for _, pod := range pods.Items {
			logs, err := GetContainerLogs(k8sClient, ctx, &pod)
			if err == nil {
				allLogs += logs
			}
		}

		hasFilterOrScore := false
		filterScoreKeywords := []string{"filter:", "score:", "mco-plugin", "[mco-plugin]", "filter ", "score ", "interference"}
		for _, keyword := range filterScoreKeywords {
			if strings.Contains(allLogs, keyword) {
				hasFilterOrScore = true
				break
			}
		}
		Expect(hasFilterOrScore).To(BeTrue(), "调度器日志应包含 filter/score 决策相关记录")
	})

	It("众核-SCHED-IT-017 卸载清理", SpecTimeout(5*time.Minute), func(ctx SpecContext) {
		By("记录卸载前的 MCO 调度器资源")
		resourcesBefore, err := GetSchedulerResources(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(resourcesBefore)).To(BeNumerically(">=", 1), "卸载前应存在 MCO 调度器资源")

		By("执行 helm uninstall")
		Expect(UninstallReleaseAndWait(ctx, WaitSchedulerResourcesDeleted)).To(Succeed())
		installedByTest = false

		By("验证 MCO 调度器资源被正确清理")
		resourcesAfter, err := GetSchedulerResources(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		cleanedKeys := []string{
			"Deployment/" + MCOSchedulerDeploymentName,
			"ConfigMap/" + MCOSchedulerConfigMapName,
			"ServiceAccount/" + MCOSchedulerServiceAccount,
		}
		for _, key := range cleanedKeys {
			Expect(resourcesAfter).NotTo(HaveKey(key), "卸载后 %s 应被删除", key)
		}

		By("验证 ClusterRole 和 ClusterRoleBinding 被清理")
		Expect(resourcesAfter).NotTo(HaveKey("ClusterRole/"+MCOSchedulerServiceAccount), "卸载后 ClusterRole 应被删除")
		Expect(resourcesAfter).NotTo(HaveKey("ClusterRoleBinding/"+MCOSchedulerServiceAccount), "卸载后 ClusterRoleBinding 应被删除")
	})
})
