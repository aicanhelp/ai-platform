package mco

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

var _ = Describe("Analyzer 集成测试", Ordered, Label("many-core-orchestrator", "analyzer", "with-workload-cluster"), func() {
	BeforeAll(func(ctx SpecContext) {
		By("清理历史测试 release，确保 Analyzer 用例从头部署")
		Expect(UninstallExistingCollectorAnalyzerRelease(ctx)).To(Succeed())

		By("部署 mco-analyzer Deployment 和 mco-collector DaemonSet")
		chartPath := GetChartPath()
		Expect(HelmInstallLocalChart(ctx, MCOHelmConfig, chartPath, CollectorPhaseHelmSets...)).To(Succeed())
		Expect(WaitHelmReleaseDeployed(ctx, MCOHelmConfig["releaseName"], 120*time.Second, 5*time.Second)).To(Succeed())
		By("等待 Collector/Analyzer 控制面 Ready 且数据面完成首轮采集上报")
		Expect(WaitCollectorAnalyzerReady(ctx)).To(Succeed())
	}, NodeTimeout(5*time.Minute))

	AfterAll(func(ctx SpecContext) {
		By("卸载 mco Analyzer/Collector 测试 release 并等待清理完成")
		_ = UninstallExistingCollectorAnalyzerRelease(ctx)
	}, NodeTimeout(5*time.Minute))

	It("mco-analyzer_FIT_1 Analyzer 部署与 CR 初始化", func(ctx SpecContext) {
		By("检查 mco-analyzer Deployment 和 Pod 处于 Running/Ready")
		deploy, err := k8sClient.Clientset.AppsV1().Deployments(MCODeployNamespace).Get(ctx, MCOAnalyzerDeploymentName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(deploy.Status.ReadyReplicas).To(Equal(deploy.Status.Replicas), "Analyzer ready replicas should equal replicas")
		Expect(deploy.Status.AvailableReplicas).To(Equal(deploy.Status.Replicas), "Analyzer available replicas should equal replicas")

		analyzerPods, err := GetAnalyzerPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(analyzerPods.Items).NotTo(BeEmpty(), "Analyzer Pod should exist")
		for _, pod := range analyzerPods.Items {
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "Analyzer Pod should be Running: %s/%s", pod.Namespace, pod.Name)
			for _, cs := range pod.Status.ContainerStatuses {
				Expect(cs.Ready).To(BeTrue(), "Analyzer container should be Ready: %s/%s/%s", pod.Namespace, pod.Name, cs.Name)
				Expect(cs.RestartCount).To(BeZero(), "Analyzer container should not restart: %s/%s/%s", pod.Namespace, pod.Name, cs.Name)
			}
		}

		By("检查 mco-analyzer Service 暴露 gRPC 和 metrics 端口")
		svc, err := GetService(k8sClient, ctx, MCOAnalyzerServiceName, MCODeployNamespace)
		Expect(err).NotTo(HaveOccurred())
		ports := map[int32]bool{}
		for _, port := range svc.Spec.Ports {
			ports[port.Port] = true
		}
		Expect(ports[MCOAnalyzerGRPCPort]).To(BeTrue(), "Analyzer Service should expose gRPC port")
		Expect(ports[MCOAnalyzerMetricsPort]).To(BeTrue(), "Analyzer Service should expose metrics port")

		By("等待 Analyzer 写入 cluster NodeInterferenceReport")
		collectorPods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(collectorPods.Items).NotTo(BeEmpty(), "Collector Pod should exist and report snapshots to Analyzer")
		expectedNodes := make([]string, 0, len(collectorPods.Items))
		for _, pod := range collectorPods.Items {
			_, err := WaitCollectorSnapshotLogTimes(k8sClient, ctx, &pod, 1, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			expectedNodes = append(expectedNodes, pod.Spec.NodeName)
		}

		report, err := WaitNodeInterferenceReportHasNodes(k8sClient, ctx, MCONodeInterferenceReport, expectedNodes, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		nodes, err := NodeInterferenceReportNodes(report)
		Expect(err).NotTo(HaveOccurred())
		Expect(nodes).NotTo(BeEmpty(), "NodeInterferenceReport status.nodes should contain reported nodes")
	})

	It("mco-analyzer_FIT_2 Analyzer 识别缓存污染干扰", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit2-cache"
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, 1, 0.28, 0.75, 35, 5, 0, 3, 0, 0.8, 100, true),
			newAnalyzerSnapshot(nodeName, 2, 0.28, 0.75, 35, 5, 0, 3, 0, 0.8, 100, true),
		})).To(Succeed())

		report, err := WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 2, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(NodeReportDetectionsContain(report, "cache-pollution", "LLC miss rate", "LLC occupancy")).To(BeTrue())
		Expect(nodeReportFloat(report, "metrics", "llcMissRate")).To(BeNumerically("~", 0.28, 0.001))
		Expect(nodeReportFloat(report, "metrics", "llcOccupancy")).To(BeNumerically("~", 0.75, 0.001))
		Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically(">", 0))
	})

	It("mco-analyzer_FIT_3 Analyzer 识别 IO 队列拥塞干扰", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit3-io"
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, 1, 0.1, 0.2, 65, 25, 0, 3, 0, 0.8, 100, true),
			newAnalyzerSnapshot(nodeName, 2, 0.1, 0.2, 65, 25, 0, 3, 0, 0.8, 100, true),
		})).To(Succeed())

		report, err := WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 2, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(NodeReportDetectionsContain(report, "io-queue-contention", "Bio latency P99", "PSI I/O")).To(BeTrue())
		Expect(nodeReportFloat(report, "interferenceLevel")).To(And(BeNumerically(">=", 0), BeNumerically("<=", 1)))
		Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.5323, 0.03))
	})

	It("mco-analyzer_FIT_4 Analyzer 识别内存压力干扰", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit4-memory"
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, 1, 0.1, 0.2, 35, 5, 0, 12, 0.5, 0.8, 100, true),
			newAnalyzerSnapshot(nodeName, 2, 0.1, 0.2, 35, 5, 0, 12, 0.5, 0.8, 100, true),
		})).To(Succeed())

		report, err := WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 2, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(NodeReportDetectionsContain(report, "memory-pressure", "PSI memory")).To(BeTrue())
		Expect(nodeReportFloat(report, "interferenceLevel")).To(And(BeNumerically(">=", 0), BeNumerically("<=", 1)))
		Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.2463, 0.03))
	})

	It("mco-analyzer_FIT_5 Analyzer 按权重计算干扰等级", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit5-weight"
		snapshots := []AnalyzerMetricSnapshot{}
		for i := 0; i < 3; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.253, 0.553, 65.0, 18.0, 0, 12.0, 0, 0.8, 100, true))
		}
		for i := 0; i < 3; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.254, 0.554, 65.1, 18.1, 0, 12.1, 0, 0.8, 100, true))
		}
		for i := 0; i < 3; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.255, 0.555, 65.2, 18.2, 0, 12.2, 0, 0.8, 100, true))
		}
		snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.613, 0.899, 82.0, 32.0, 0, 18.0, 3, 0.8, 100, true))
		Expect(InjectAnalyzerMetricSnapshots(ctx, snapshots)).To(Succeed())

		report, err := WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 10, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		level := nodeReportFloat(report, "interferenceLevel")
		Expect(level).To(And(BeNumerically(">=", 0), BeNumerically("<=", 1)))
		Expect(level).To(BeNumerically("~", 0.7584, 0.03))
		Expect(nodeReportFloat(report, "metrics", "llcMissRate")).To(BeNumerically("~", 0.2899, 0.001))
		Expect(nodeReportFloat(report, "metrics", "llcMissRate")).NotTo(BeNumerically("~", snapshots[len(snapshots)-1].LLCMissRate, 0.001))
		Expect(nodeReportFloat(report, "metrics", "llcOccupancy")).To(BeNumerically("~", 0.5885, 0.001))
		Expect(level).NotTo(BeNumerically("~", 0.888, 0.03))

		metrics, err := FetchAnalyzerMetricSamples(ctx, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		metricLevel, ok := MetricValueForNode(metrics, "mco_node_interference_level", nodeName)
		Expect(ok).To(BeTrue(), "mco_node_interference_level should contain node=%s", nodeName)
		Expect(metricLevel).To(BeNumerically("~", level, 0.001))
	})

	It("mco-analyzer_FIT_6 Analyzer 写入集群级 NodeInterferenceReport 状态", func(ctx SpecContext) {
		collectorPods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		if len(collectorPods.Items) < 2 {
			Skip("need at least two Collector Pods to verify multi-node NodeInterferenceReport status")
		}

		expectedNodes := make([]string, 0, len(collectorPods.Items))
		for _, pod := range collectorPods.Items {
			_, err := WaitCollectorSnapshotLogTimes(k8sClient, ctx, &pod, 1, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			expectedNodes = append(expectedNodes, pod.Spec.NodeName)
		}
		report, err := WaitNodeInterferenceReportHasNodes(k8sClient, ctx, MCONodeInterferenceReport, expectedNodes, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
		nodes, err := NodeInterferenceReportNodes(report)
		Expect(err).NotTo(HaveOccurred())
		for _, nodeName := range expectedNodes {
			nodeReport, ok := nodes[nodeName].(map[string]interface{})
			Expect(ok).To(BeTrue(), "status.nodes.%s should be an object", nodeName)
			Expect(nodeReportFloat(nodeReport, "interferenceLevel")).To(And(BeNumerically(">=", 0), BeNumerically("<=", 1)))
			_, ok = NodeReportBool(nodeReport, "psiAvailable")
			Expect(ok).To(BeTrue(), "status.nodes.%s.psiAvailable should exist", nodeName)
			Expect(nodeReport).To(HaveKey("metrics"))
			if detections, exists := nodeReport["detections"]; exists {
				Expect(detections).To(BeAssignableToTypeOf([]interface{}{}), "status.nodes.%s.detections should be an array when present", nodeName)
				for _, item := range detections.([]interface{}) {
					detection, ok := item.(map[string]interface{})
					Expect(ok).To(BeTrue(), "status.nodes.%s.detections item should be an object", nodeName)
					Expect(detection).To(HaveKey("type"), "status.nodes.%s.detections item should contain type", nodeName)
				}
			}
		}
	})

	It("mco-analyzer_FIT_7 Analyzer 重启后快速恢复 CR 更新", func(ctx SpecContext) {
		reportBefore, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
		Expect(err).NotTo(HaveOccurred())
		nodesBefore, err := NodeInterferenceReportNodes(reportBefore)
		Expect(err).NotTo(HaveOccurred())
		Expect(nodesBefore).NotTo(BeEmpty(), "NodeInterferenceReport should already contain node status")

		sequenceBefore := map[string]float64{}
		for nodeName, raw := range nodesBefore {
			nodeReport, ok := raw.(map[string]interface{})
			Expect(ok).To(BeTrue(), "status.nodes.%s should be an object", nodeName)
			sequence, ok := NodeReportFloat(nodeReport, "sequence")
			Expect(ok).To(BeTrue(), "status.nodes.%s.sequence should exist before restart", nodeName)
			sequenceBefore[nodeName] = sequence
		}

		By("重启 mco-analyzer Deployment")
		Expect(RolloutRestartDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace)).To(Succeed())
		reportRetained, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
		Expect(err).NotTo(HaveOccurred())
		nodesRetained, err := NodeInterferenceReportNodes(reportRetained)
		Expect(err).NotTo(HaveOccurred())
		for nodeName := range sequenceBefore {
			Expect(nodesRetained).To(HaveKey(nodeName), "old CR node %s should be retained during Analyzer restart", nodeName)
		}
		Expect(WaitDeploymentReady(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 120*time.Second)).To(Succeed())

		By("检查 Analyzer 恢复后 2s 内重新写入最新 CR")
		Eventually(func(g Gomega) {
			reportAfter, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
			g.Expect(err).NotTo(HaveOccurred())
			nodesAfter, err := NodeInterferenceReportNodes(reportAfter)
			g.Expect(err).NotTo(HaveOccurred())
			updated := false
			for nodeName, beforeSequence := range sequenceBefore {
				raw, ok := nodesAfter[nodeName]
				g.Expect(ok).To(BeTrue(), "status.nodes.%s should still exist after Analyzer restart", nodeName)
				nodeReport, ok := raw.(map[string]interface{})
				g.Expect(ok).To(BeTrue(), "status.nodes.%s should be an object", nodeName)
				sequence, ok := NodeReportFloat(nodeReport, "sequence")
				g.Expect(ok).To(BeTrue(), "status.nodes.%s.sequence should exist after restart", nodeName)
				if sequence > beforeSequence {
					updated = true
				}
			}
			g.Expect(updated).To(BeTrue(), "at least one node report should be overwritten by a new snapshot after Analyzer restart")
		}, 2*time.Second, 200*time.Millisecond).Should(Succeed())
	})

	It("mco-analyzer_FIT_8 Analyzer 暴露计算产出指标", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit8-metrics"
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, 1, 0.22, 0.72, 55, 22, 0, 11, 0, 0.8, 100, true),
		})).To(Succeed())
		_, err := WaitNodeInterferenceReportNode(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		report, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
		Expect(err).NotTo(HaveOccurred())
		nodes, err := NodeInterferenceReportNodes(report)
		Expect(err).NotTo(HaveOccurred())
		_, ok := nodes[nodeName]
		Expect(ok).To(BeTrue(), "NodeInterferenceReport status.nodes should contain node key %s", nodeName)

		metrics, err := FetchAnalyzerMetricSamples(ctx, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		_, ok = MetricValueForNode(metrics, "mco_node_interference_level", nodeName)
		Expect(ok).To(BeTrue(), "mco_node_interference_level should contain node=%s", nodeName)
		Expect(metricNameHasPrefix(metrics, "mco_analyzer_report_generation_duration_seconds")).To(BeTrue(), "analyzer report generation duration histogram should be exported")
		Expect(metricExists(metrics, "mco_analyzer_cr_updates_total")).To(BeTrue())
	})

	It("mco-analyzer_FIT_9 Analyzer 执行 PSI 不可用降级评估", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit9-degraded"
		snapshots := []AnalyzerMetricSnapshot{}
		for i := 0; i < 3; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.150, 0.550, 35.0, -1, -1, -1, -1, 0.750, 6400, false))
		}
		for i := 0; i < 3; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.155, 0.555, 35.5, -1, -1, -1, -1, 0.745, 6500, false))
		}
		for i := 0; i < 4; i++ {
			snapshots = append(snapshots, newAnalyzerSnapshot(nodeName, uint64(len(snapshots)+1), 0.160, 0.560, 36.0, -1, -1, -1, -1, 0.740, 6600, false))
		}
		Expect(InjectAnalyzerMetricSnapshots(ctx, snapshots)).To(Succeed())

		report, err := WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 10, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		psiAvailable, ok := NodeReportBool(report, "psiAvailable")
		Expect(ok).To(BeTrue())
		Expect(psiAvailable).To(BeFalse())
		Expect(nodeReportFloat(report, "metrics", "psiIoSome")).To(Equal(float64(-1)))
		Expect(nodeReportFloat(report, "metrics", "psiIoFull")).To(Equal(float64(-1)))
		Expect(nodeReportFloat(report, "metrics", "psiMemorySome")).To(Equal(float64(-1)))
		Expect(nodeReportFloat(report, "metrics", "psiMemoryFull")).To(Equal(float64(-1)))
		Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.5213, 0.03))
		Expect(NodeReportAvailabilityHas(report, "psiIoSome", false, "fallback")).To(BeTrue())
		Expect(NodeReportAvailabilityHas(report, "psiIoFull", false, "fallback")).To(BeTrue())
		Expect(NodeReportAvailabilityHas(report, "psiMemorySome", false, "fallback")).To(BeTrue())
		Expect(NodeReportAvailabilityHas(report, "psiMemoryFull", false, "fallback")).To(BeTrue())
		Expect(NodeReportAvailabilityHas(report, "memAvailableRatio", true, "ok")).To(BeTrue())
		Expect(NodeReportAvailabilityHas(report, "pageScanRate", true, "ok")).To(BeTrue())
	})

	It("mco-analyzer_FIT_10 Analyzer 热更新 DetectionRuleConfig 生效", func(ctx SpecContext) {
		nodeName := "e2e-analyzer-fit10-rule"
		By("确认默认规则下干扰等级")
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, 1, 0.253, 0.553, 65.0, 18.0, 0, 12.0, 0, 0.8, 100, true),
		})).To(Succeed())
		report, err := WaitNodeInterferenceReportNode(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.6718, 0.03))
		originalHash := reportHash(ctx)

		By("修改 DetectionRuleConfig 权重并等待规则 hash 更新")
		Expect(UpdateDetectionRuleWeights(ctx, 0.10, 0.10, 0.80)).To(Succeed())
		DeferCleanup(func(ctx SpecContext) {
			_ = UpdateDetectionRuleWeights(ctx, 0.30, 0.30, 0.40)
		})
		nextSequence := uint64(2)
		Expect(WaitRuleHashWithSnapshots(ctx, nodeName, &nextSequence, 60*time.Second,
			func(hash string) bool { return hash != originalHash },
			"change from original rule config hash")).To(Succeed())

		By("规则 hash 生效后再次发送快照并验证新权重计算")
		verifySequence := nextSequence
		nextSequence++
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, verifySequence, 0.253, 0.553, 65.0, 18.0, 0, 12.0, 0, 0.8, 100, true),
		})).To(Succeed())
		Eventually(func(g Gomega) {
			report, err := WaitNodeInterferenceReportNode(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 5*time.Second)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(nodeReportFloat(report, "sequence")).To(BeNumerically(">=", verifySequence))
			g.Expect(reportHash(ctx)).NotTo(Equal(originalHash))
			g.Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.8239, 0.03))
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		By("恢复默认配置并等待规则 hash 恢复")
		changedHash := reportHash(ctx)
		Expect(UpdateDetectionRuleWeights(ctx, 0.30, 0.30, 0.40)).To(Succeed())
		Expect(WaitRuleHashWithSnapshots(ctx, nodeName, &nextSequence, 60*time.Second,
			func(hash string) bool { return hash != changedHash && hash == originalHash },
			"restore original rule config hash")).To(Succeed())

		By("默认规则 hash 恢复后再次发送快照并验证干扰等级恢复")
		restoreVerifySequence := nextSequence
		nextSequence++
		Expect(InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, restoreVerifySequence, 0.253, 0.553, 65.0, 18.0, 0, 12.0, 0, 0.8, 100, true),
		})).To(Succeed())
		Eventually(func(g Gomega) {
			report, err := WaitNodeInterferenceReportNode(k8sClient, ctx, MCONodeInterferenceReport, nodeName, 5*time.Second)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(nodeReportFloat(report, "sequence")).To(BeNumerically(">=", restoreVerifySequence))
			g.Expect(reportHash(ctx)).To(Equal(originalHash))
			g.Expect(nodeReportFloat(report, "interferenceLevel")).To(BeNumerically("~", 0.6718, 0.03))
		}, 60*time.Second, 2*time.Second).Should(Succeed())
	})
})

func newAnalyzerSnapshot(node string, sequence uint64, llcMiss, llcOccupancy, bioP99, psiIO, psiIOFull, psiMemory, psiMemoryFull, memAvailableRatio, pageScanRate float64, psiAvailable bool) AnalyzerMetricSnapshot {
	status := "ok"
	if !psiAvailable {
		status = "fallback"
	}
	return AnalyzerMetricSnapshot{
		Node:              node,
		Sequence:          sequence,
		Status:            "ok",
		LLCMissRate:       llcMiss,
		LLCOccupancy:      llcOccupancy,
		BioLatencyP95Ms:   bioP99 * 0.8,
		BioLatencyP99Ms:   bioP99,
		PSIAvailable:      psiAvailable,
		PSIIOAvg10:        psiIO,
		PSIIOFullAvg10:    psiIOFull,
		PSIMemAvg10:       psiMemory,
		PSIMemFullAvg10:   psiMemoryFull,
		MemAvailableRatio: memAvailableRatio,
		PageScanRate:      pageScanRate,
		Availability: []AnalyzerMetricAvailability{
			{Collector: "psi", Metric: "psiIoSome", Available: psiAvailable, Status: status},
			{Collector: "psi", Metric: "psiIoFull", Available: psiAvailable, Status: status},
			{Collector: "psi", Metric: "psiMemorySome", Available: psiAvailable, Status: status},
			{Collector: "psi", Metric: "psiMemoryFull", Available: psiAvailable, Status: status},
			{Collector: "memory", Metric: "memAvailableRatio", Available: true, Status: "ok"},
			{Collector: "memory", Metric: "pageScanRate", Available: true, Status: "ok"},
		},
	}
}

func nodeReportFloat(report map[string]interface{}, fields ...string) float64 {
	value, ok := NodeReportFloat(report, fields...)
	Expect(ok).To(BeTrue(), "node report field %v should exist and be numeric", fields)
	return value
}

func metricExists(samples []MetricSample, name string) bool {
	for _, sample := range samples {
		if sample.Name == name {
			return true
		}
	}
	return false
}

func metricNameHasPrefix(samples []MetricSample, prefix string) bool {
	for _, sample := range samples {
		if strings.HasPrefix(sample.Name, prefix) {
			return true
		}
	}
	return false
}

func reportHash(ctx SpecContext) string {
	report, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
	Expect(err).NotTo(HaveOccurred())
	hash, _, _ := unstructured.NestedString(report.Object, "status", "ruleConfigHash")
	return hash
}

func WaitRuleHashWithSnapshots(ctx SpecContext, nodeName string, nextSequence *uint64, timeout time.Duration, acceptHash func(string) bool, description string) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	var lastHash string
	var lastSequence float64
	var lastInjected uint64

	for time.Now().Before(deadline) {
		hash, sequence, err := fit10ReportState(ctx, nodeName)
		if err == nil {
			lastHash = hash
			lastSequence = sequence
			if acceptHash(hash) {
				return nil
			}
		} else {
			lastErr = err
		}

		injectedSequence := *nextSequence
		(*nextSequence)++
		lastInjected = injectedSequence
		if err := InjectAnalyzerMetricSnapshots(ctx, []AnalyzerMetricSnapshot{
			newAnalyzerSnapshot(nodeName, injectedSequence, 0.253, 0.553, 65.0, 18.0, 0, 12.0, 0, 0.8, 100, true),
		}); err != nil {
			lastErr = err
			time.Sleep(2 * time.Second)
			continue
		}

		waitUntil := time.Now().Add(5 * time.Second)
		if deadline.Before(waitUntil) {
			waitUntil = deadline
		}
		processed := false
		for time.Now().Before(waitUntil) {
			hash, currentSequence, err := fit10ReportState(ctx, nodeName)
			if err != nil {
				lastErr = err
				time.Sleep(500 * time.Millisecond)
				continue
			}
			lastHash = hash
			lastSequence = currentSequence
			if currentSequence >= float64(injectedSequence) {
				processed = true
				if acceptHash(hash) {
					return nil
				}
				break
			}
			time.Sleep(500 * time.Millisecond)
		}

		if !processed {
			lastErr = fmt.Errorf("injected snapshot sequence %d was not reflected in report before retry", injectedSequence)
		}
		time.Sleep(2 * time.Second)
	}

	return fmt.Errorf("%s not observed within %v: lastHash=%q lastSequence=%.0f lastInjected=%d lastErr=%v",
		description, timeout, lastHash, lastSequence, lastInjected, lastErr)
}

func fit10ReportState(ctx SpecContext, nodeName string) (string, float64, error) {
	report, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
	if err != nil {
		return "", 0, err
	}
	hash, _, _ := unstructured.NestedString(report.Object, "status", "ruleConfigHash")
	nodes, err := NodeInterferenceReportNodes(report)
	if err != nil {
		return hash, 0, err
	}
	raw, ok := nodes[nodeName]
	if !ok {
		return hash, 0, fmt.Errorf("node %s not found in report %s", nodeName, MCONodeInterferenceReport)
	}
	nodeReport, ok := raw.(map[string]interface{})
	if !ok {
		return hash, 0, fmt.Errorf("node %s report has unexpected type %T", nodeName, raw)
	}
	sequence, ok := NodeReportFloat(nodeReport, "sequence")
	if !ok {
		return hash, 0, fmt.Errorf("node %s report sequence is missing or non-numeric", nodeName)
	}
	return hash, sequence, nil
}

func UpdateDetectionRuleWeights(ctx SpecContext, cache, memory, io float64) error {
	gvr := schema.GroupVersionResource{Group: "mco.many-core.io", Version: "v1alpha1", Resource: "detectionruleconfigs"}
	obj, err := k8sClient.DynamicClient.Resource(gvr).Namespace(MCODeployNamespace).Get(ctx, MCODetectionRuleConfig, metav1.GetOptions{})
	if err != nil {
		return err
	}
	weights := map[string]interface{}{
		"cache":  cache,
		"memory": memory,
		"io":     io,
	}
	if err := unstructured.SetNestedMap(obj.Object, weights, "spec", "weights"); err != nil {
		return err
	}
	_, err = k8sClient.DynamicClient.Resource(gvr).Namespace(MCODeployNamespace).Update(ctx, obj, metav1.UpdateOptions{})
	return err
}
