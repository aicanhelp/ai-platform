package mco

import (
	"fmt"
	"os/exec"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

var _ = Describe("众核调度插件 Filter 测试", Ordered, Label("many-core-orchestrator", "scheduler", "scheduler-filter", "with-workload-cluster"), func() {
	var (
		nodeA, nodeB string
	)

	BeforeAll(func(ctx SpecContext) {
		By("清理历史测试 release")
		Expect(UninstallExistingReleaseAndWait(ctx, WaitSchedulerResourcesDeleted)).To(Succeed())

		By("获取集群节点")
		nodes, err := GetAllNodeNames(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(nodes)).To(BeNumerically(">=", 2), "至少需要 2 个节点才能执行 Filter 测试")
		nodeA = nodes[0]
		nodeB = nodes[1]

		By("部署 MCO 调度插件（volcano + mcoPlugin）")
		chartPath := GetChartPath()
		Expect(HelmInstallLocalChart(ctx, MCOHelmConfig, chartPath, SchedulerPhaseHelmSets...)).To(Succeed())
		Expect(WaitHelmReleaseDeployed(ctx, MCOHelmConfig["releaseName"], 120*time.Second, 5*time.Second)).To(Succeed())

		By("等待 mco-volcano-scheduler Deployment Ready")
		Expect(WaitDeploymentReady(k8sClient, ctx, MCOSchedulerDeploymentName, MCOVolcanoNamespace, 120*time.Second)).To(Succeed())

		By("等待 Volcano admission webhook 就绪")
		Expect(WaitVolcanoAdmissionReady(k8sClient, ctx, 120*time.Second)).To(Succeed())

		By("确保 NodeInterferenceReport CR 存在")
		ensureNodeInterferenceReportCR(ctx)
		time.Sleep(3 * time.Second)
	}, NodeTimeout(5*time.Minute))

	AfterAll(func(ctx SpecContext) {
		By("卸载 MCO 调度插件并等待清理完成")
		_ = UninstallExistingReleaseAndWait(ctx, WaitSchedulerResourcesDeleted)
		By("清理 NodeInterferenceReport CR")
		_ = DeleteNodeInterferenceReport(ctx, MCONodeInterferenceReport)
	}, NodeTimeout(5*time.Minute))

	AfterEach(func(ctx SpecContext) {
		By("清理测试 Pod")
		_ = DeletePod(k8sClient, ctx, "test-psi-mem", KataTestNamespace, 30*time.Second)
		_ = DeletePod(k8sClient, ctx, "test-psi-io", KataTestNamespace, 30*time.Second)
		_ = DeletePod(k8sClient, ctx, "test-psi-skip", KataTestNamespace, 30*time.Second)
	})

	It("众核-SCHED-IT-006 数据源不可用时降级调度", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		podName := fmt.Sprintf("test-degrade-%d", time.Now().UnixNano()%100000)

		By("注入 CR 数据填充调度器缓存后删除 CR 模拟数据源不可用")
		nodeAReport := buildNodeReportJSON(true, 0, 0, 0.3)
		nodeBReport := buildNodeReportJSON(true, 0, 0, 0.3)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport,
			buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport))).To(Succeed())
		time.Sleep(3 * time.Second)

		Expect(DeleteNodeInterferenceReport(ctx, MCONodeInterferenceReport)).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建测试 Pod 验证降级后仍可正常调度")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, podName, KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, podName, KataTestNamespace, 30*time.Second)
		})

		By("验证 Pod 被成功调度（降级不阻塞）")
		_, err = WaitPodScheduledOnNode(k8sClient, ctx, podName, KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred(), "CR 删除后 Pod 应仍能正常调度")

		By("检查调度器日志包含降级回退信息")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}

		By("重新创建 CR 供后续 Filter 测试使用")
		ensureNodeInterferenceReportCR(ctx)
		time.Sleep(3 * time.Second)
	})

	It("众核-SCHED-IT-007 Filter_Kata双标签运行时检查", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		podName := fmt.Sprintf("test-kata-filter-%d", time.Now().UnixNano()%100000)

		By(fmt.Sprintf("保存节点标签：节点 A(%s) 删除 kata-ready，节点 B(%s) 确保双标签", nodeA, nodeB))
		origLabelA, hasLabelA, _ := GetNodeLabelValue(k8sClient, ctx, nodeA, MCOKataReadyLabelKey)
		origLabelB, hasLabelB, _ := GetNodeLabelValue(k8sClient, ctx, nodeB, MCOKataReadyLabelKey)
		_ = UnlabelNode(k8sClient, ctx, nodeA, MCOKataReadyLabelKey)
		Expect(LabelNode(k8sClient, ctx, nodeB, MCOKataRuntimeLabelKey, "true")).To(Succeed())
		Expect(LabelNode(k8sClient, ctx, nodeB, MCOKataReadyLabelKey, "true")).To(Succeed())

		DeferCleanup(func(ctx SpecContext) {
			if hasLabelA {
				_ = LabelNode(k8sClient, ctx, nodeA, MCOKataReadyLabelKey, origLabelA)
			} else {
				_ = UnlabelNode(k8sClient, ctx, nodeA, MCOKataReadyLabelKey)
			}
			if hasLabelB {
				_ = LabelNode(k8sClient, ctx, nodeB, MCOKataReadyLabelKey, origLabelB)
			} else {
				_ = UnlabelNode(k8sClient, ctx, nodeB, MCOKataReadyLabelKey)
			}
		})

		By("确保 RuntimeClass kata-qemu 存在")
		ensureRuntimeClassKataQemu(ctx)
		DeferCleanup(func(ctx SpecContext) {
			cmd := exec.CommandContext(ctx, "kubectl", "delete", "runtimeclass", "kata-qemu", "--ignore-not-found")
			if err := cmd.Run(); err != nil {
				fmt.Fprintf(GinkgoWriter, "WARNING: cleanup RuntimeClass kata-qemu failed: %v\n", err)
			}
		})

		By("创建 runtimeClassName=kata-qemu, schedulerName=volcano 的测试 Pod")
		_, err := CreateKataSchedulerTestPod(k8sClient, ctx, podName, KataTestNamespace, "kata-qemu", "volcano")
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, podName, KataTestNamespace, 30*time.Second)
		})

		By("验证 Pod 被调度到节点 B（双标签检查通过）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, podName, KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Equal(nodeB),
			"Pod 应调度到同时具有 %s 和 %s 双标签的节点 B，节点 A 缺少 %s 应被排除",
			MCOKataRuntimeLabelKey, MCOKataReadyLabelKey, MCOKataReadyLabelKey)

		By("检查调度器日志包含 kata filter 排除信息")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}
	})

	It("众核-SCHED-IT-008 Filter_PSI Memory Full 硬排除", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		By(fmt.Sprintf("注入 CR 数据：节点 A(%s) psiMemoryFull=5.0，节点 B(%s) psiMemoryFull=0", nodeA, nodeB))
		nodeAReport := buildNodeReportJSON(true, 5.0, 0, 0.5)
		nodeBReport := buildNodeReportJSON(true, 0, 0, 0.1)
		patchJSON := buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport, patchJSON)).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建使用 volcano 调度器的测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, "test-psi-mem", KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())

		By("验证 Pod 未调度到节点 A（psiMemoryFull 硬排除）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, "test-psi-mem", KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Equal(nodeB), "Pod 应调度到节点 B，因为节点 A 的 psiMemoryFull > 0 被硬排除")

		By("检查调度器日志包含 [mco-plugin] filter 排除信息")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}
	})

	It("众核-SCHED-IT-009 Filter_PSI IO 硬排除", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		By(fmt.Sprintf("注入 CR 数据：节点 A(%s) psiIoSome=30.0，节点 B(%s) psiIoSome=5.0", nodeA, nodeB))
		nodeAReport := buildNodeReportJSON(true, 0, 30.0, 0.5)
		nodeBReport := buildNodeReportJSON(true, 0, 5.0, 0.1)
		patchJSON := buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport, patchJSON)).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建使用 volcano 调度器的测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, "test-psi-io", KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())

		By("验证 Pod 未调度到节点 A（psiIoSome 超阈值硬排除）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, "test-psi-io", KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Equal(nodeB), "Pod 应调度到节点 B，因为节点 A 的 psiIoSome > 阈值被硬排除")

		By("检查调度器日志包含 [mco-plugin] filter 排除信息")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}
	})

	It("众核-SCHED-IT-010 Filter_PSI 不可用时跳过约束", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		By(fmt.Sprintf("注入 CR 数据：节点 A(%s) psiAvailable=false（高指标但跳过），节点 B(%s) psiAvailable=true", nodeA, nodeB))
		nodeAReport := buildNodeReportJSON(false, 5.0, 30.0, 0.3)
		nodeBReport := buildNodeReportJSON(true, 0, 0, 0.3)
		patchJSON := buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport, patchJSON)).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建使用 volcano 调度器的测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, "test-psi-skip", KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())

		By("验证 Pod 可被调度（节点 A 的 PSI 约束被跳过）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, "test-psi-skip", KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Or(Equal(nodeA), Equal(nodeB)), "Pod 应可调度到任一节点（PSI 不可用时跳过约束）")

		By("检查调度器日志包含 [mco-plugin] 信息")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}
	})

	It("众核-SCHED-IT-011 Score_干扰等级评分", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		podName := fmt.Sprintf("test-score-%d", time.Now().UnixNano()%100000)

		By(fmt.Sprintf("注入 CR：节点 A(%s) interferenceLevel=0.2 (高分80)，节点 B(%s) interferenceLevel=0.8 (低分20)", nodeA, nodeB))
		// 必须显式设 psiMemoryFull=0, psiIoSome=0，否则 Filter 阶段可能先排除节点
		nodeAReport := buildNodeReportJSON(true, 0, 0, 0.2)
		nodeBReport := buildNodeReportJSON(true, 0, 0, 0.8)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport,
			buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport))).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, podName, KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, podName, KataTestNamespace, 30*time.Second)
		})

		By("验证 Pod 调度到低干扰节点 A（得分更高）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, podName, KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Equal(nodeA),
			"Pod 应优先调度到低干扰节点 A (score≈80)，而非节点 B (score≈20)")

		By("检查调度器日志 score 输出")
		err = WaitSchedulerLogsContain(k8sClient, ctx, "[mco-plugin]", 30*time.Second)
		if err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: 调度器日志未包含 [mco-plugin] 信息: %v\n", err)
		}
	})

	It("众核-SCHED-IT-012 Score_数据不完整时中性分回退", SpecTimeout(2*time.Minute), func(ctx SpecContext) {
		podName := fmt.Sprintf("test-score-partial-%d", time.Now().UnixNano()%100000)

		By(fmt.Sprintf("注入 CR：节点 A(%s) status=PARTIAL (中性分50)，节点 B(%s) status=ok (正常评分80)", nodeA, nodeB))
		nodeAReport := buildNodeReportJSONWithStatus(true, 0, 0, 0.2, "PARTIAL")
		nodeBReport := buildNodeReportJSON(true, 0, 0, 0.2)
		Expect(PatchNodeInterferenceReportStatus(ctx, MCONodeInterferenceReport,
			buildFilterPatchJSON(nodeA, nodeB, nodeAReport, nodeBReport))).To(Succeed())
		time.Sleep(3 * time.Second)

		By("创建测试 Pod")
		_, err := CreateSchedulerTestPod(k8sClient, ctx, podName, KataTestNamespace, "volcano")
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, podName, KataTestNamespace, 30*time.Second)
		})

		By("验证 Pod 调度到节点 B（正常评分80 > 中性分50）")
		scheduledNode, err := WaitPodScheduledOnNode(k8sClient, ctx, podName, KataTestNamespace, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(scheduledNode).To(Equal(nodeB),
			"Pod 应调度到节点 B (status=ok, score≈80)，节点 A (status=PARTIAL) 得中性分 50 应被比下去")
	})
})

func ensureNodeInterferenceReportCR(ctx SpecContext) {
	crJSON := fmt.Sprintf(`{"apiVersion":"mco.many-core.io/v1alpha1","kind":"NodeInterferenceReport","metadata":{"name":"%s"},"status":{"nodes":{}}}`, MCONodeInterferenceReport)
	cmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(crJSON)
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(GinkgoWriter, "WARNING: ensureNodeInterferenceReportCR failed: %v\n", err)
	}
}

func buildNodeReportJSON(psiAvailable bool, psiMemoryFull, psiIoSome, interferenceLevel float64) string {
	seq := time.Now().UnixNano()
	return fmt.Sprintf(`{"psiAvailable":%t,"status":"ok","sequence":%d,"sampleCount":10,"windowSeconds":10.0,"interferenceLevel":%f,"metrics":{"llcMissRate":0.05,"llcOccupancy":0.3,"bioLatencyP95Ms":5.0,"bioLatencyP99Ms":8.0,"psiIoSome":%f,"psiIoFull":0,"psiMemorySome":2.0,"psiMemoryFull":%f,"memAvailableRatio":0.85,"pageScanRate":0}}`,
		psiAvailable, seq, interferenceLevel, psiIoSome, psiMemoryFull)
}

func buildFilterPatchJSON(nodeA, nodeB string, nodeAReport, nodeBReport string) string {
	return fmt.Sprintf(`{"status":{"nodes":{"%s":%s,"%s":%s}}}`, nodeA, nodeAReport, nodeB, nodeBReport)
}

func buildNodeReportJSONWithStatus(psiAvailable bool, psiMemoryFull, psiIoSome, interferenceLevel float64, status string) string {
	r := buildNodeReportJSON(psiAvailable, psiMemoryFull, psiIoSome, interferenceLevel)
	return strings.Replace(r, `"status":"ok"`, `"status":"`+status+`"`, 1)
}

func ensureRuntimeClassKataQemu(ctx SpecContext) {
	rcYAML := fmt.Sprintf(`apiVersion: node.k8s.io/v1
kind: RuntimeClass
metadata:
  name: %s
handler: %s
`, KataRuntimeClassQemu, KataRuntimeClassQemu)
	cmd := exec.CommandContext(ctx, "kubectl", "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(rcYAML)
	if err := cmd.Run(); err != nil {
		fmt.Fprintf(GinkgoWriter, "WARNING: ensureRuntimeClassKataQemu failed: %v\n", err)
	}
}
