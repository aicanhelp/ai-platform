package mco

import (
	"context"
	"fmt"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

func WaitCollectorAnalyzerReady(ctx context.Context) error {
	if err := WaitDaemonSetReady(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, 120*time.Second); err != nil {
		return err
	}
	if err := WaitDeploymentReady(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 120*time.Second); err != nil {
		return err
	}
	if err := AnalyzerServiceReady(k8sClient, ctx); err != nil {
		return err
	}

	expectedNodes, err := WaitCollectorDataPlaneReady(ctx, 120*time.Second)
	if err != nil {
		return err
	}
	_, err = WaitNodeInterferenceReportHasNodes(k8sClient, ctx, MCONodeInterferenceReport, expectedNodes, 90*time.Second)
	return err
}

func WaitCollectorDataPlaneReady(ctx context.Context, timeout time.Duration) ([]string, error) {
	pods, err := GetCollectorPods(k8sClient, ctx)
	if err != nil {
		return nil, err
	}
	if len(pods.Items) == 0 {
		return nil, fmt.Errorf("no Collector Pods found")
	}

	expectedNodes := make([]string, 0, len(pods.Items))
	for _, pod := range pods.Items {
		if err := WaitCollectorSnapshotLogCount(k8sClient, ctx, &pod, 1, timeout); err != nil {
			return nil, err
		}
		if _, err := FetchCollectorMetricSamples(ctx, &pod, timeout); err != nil {
			return nil, err
		}
		expectedNodes = append(expectedNodes, pod.Spec.NodeName)
	}
	return expectedNodes, nil
}

func WaitCollectorAnalyzerDeleted(ctx context.Context) error {
	if err := WaitNamedResourcesDeleted(k8sClient, ctx, MCODeployNamespace, 60*time.Second,
		"daemonset/"+MCOCollectorDaemonSetName,
		"deployment/"+MCOAnalyzerDeploymentName,
		"service/"+MCOCollectorDaemonSetName,
		"service/"+MCOAnalyzerServiceName,
	); err != nil {
		return err
	}
	return WaitPodsWithLabelsDeleted(k8sClient, ctx, MCODeployNamespace, 60*time.Second,
		MCOCollectorAppLabel,
		"app="+MCOAnalyzerServiceName,
	)
}

func UninstallExistingCollectorAnalyzerRelease(ctx context.Context) error {
	return UninstallExistingReleaseAndWait(ctx, WaitCollectorAnalyzerDeleted)
}

var _ = Describe("Collector 集成测试", Ordered, Label("many-core-orchestrator", "collector", "with-workload-cluster"), func() {
	BeforeAll(func(ctx SpecContext) {
		By("清理历史测试 release，确保 Collector 用例从头部署")
		Expect(UninstallExistingCollectorAnalyzerRelease(ctx)).To(Succeed())

		By("部署 mco-collector DaemonSet 和 mco-analyzer Deployment")
		chartPath := GetChartPath()
		Expect(HelmInstallLocalChart(ctx, MCOHelmConfig, chartPath, CollectorPhaseHelmSets...)).To(Succeed())
		Expect(WaitHelmReleaseDeployed(ctx, MCOHelmConfig["releaseName"], 120*time.Second, 5*time.Second)).To(Succeed())
		By("等待 Collector/Analyzer 控制面 Ready 且数据面完成首轮采集上报")
		Expect(WaitCollectorAnalyzerReady(ctx)).To(Succeed())
	}, NodeTimeout(5*time.Minute))

	AfterAll(func(ctx SpecContext) {
		By("卸载 mco Collector/Analyzer 测试 release 并等待清理完成")
		_ = UninstallExistingCollectorAnalyzerRelease(ctx)
	}, NodeTimeout(5*time.Minute))

	It("mco-collector_FIT_1 Collector 部署与健康验证", func(ctx SpecContext) {
		By("检查 mco-collector DaemonSet 副本数与可调度节点数量一致")
		schedulableNodes, err := CountSchedulableNodes(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(schedulableNodes).To(BeNumerically(">", 0), "cluster should have at least one schedulable node")

		ds, err := GetDaemonSet(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(int(ds.Status.DesiredNumberScheduled)).To(Equal(schedulableNodes), "DaemonSet DESIRED should equal schedulable node count")
		Expect(int(ds.Status.CurrentNumberScheduled)).To(Equal(schedulableNodes), "DaemonSet CURRENT should equal schedulable node count")
		Expect(int(ds.Status.NumberReady)).To(Equal(schedulableNodes), "DaemonSet READY should equal schedulable node count")

		By("检查每个节点上的 Collector Pod 均处于 Running 且未发生重启")
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).To(HaveLen(schedulableNodes), "Collector Pod count should equal schedulable node count")
		for _, pod := range pods.Items {
			Expect(pod.Spec.NodeName).NotTo(BeEmpty(), "Collector Pod should bind to a node: %s", pod.Name)
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "Collector Pod should be Running: %s/%s", pod.Namespace, pod.Name)
			for _, cs := range pod.Status.ContainerStatuses {
				Expect(cs.Ready).To(BeTrue(), "Collector container should be Ready: %s/%s/%s", pod.Namespace, pod.Name, cs.Name)
				Expect(cs.RestartCount).To(BeZero(), "Collector container should not restart: %s/%s/%s", pod.Namespace, pod.Name, cs.Name)
			}
		}

		By("检查 Collector 日志包含健康采集记录")
		Expect(WaitCollectorLogsHealthy(k8sClient, ctx, &pods.Items[0], 30*time.Second)).To(Succeed())
	})

	It("mco-collector_FIT_2 Collector 采集核心宿主机指标", Label("skip-temporarily"), func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty(), "at least one Collector Pod should exist")
		pod := pods.Items[0]

		By("检查 Collector /metrics 暴露核心指标清单")
		samples, err := FetchCollectorMetricSamples(ctx, &pod, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())

		requiredMetrics := []string{
			"mco_llc_miss_rate_raw",
			"mco_llc_occupancy_raw",
			"mco_bio_latency_p99_ms_raw",
			"mco_psi_io_some_avg10_raw",
			"mco_psi_memory_some_avg10_raw",
			"mco_snapshot_timestamp_ms",
		}
		for _, metricName := range requiredMetrics {
			_, ok := MetricValueForNode(samples, metricName, pod.Spec.NodeName)
			Expect(ok).To(BeTrue(), "metric %s should exist with node=%s", metricName, pod.Spec.NodeName)
		}

		By("检查 snapshot timestamp 与日志采集周期约 1s")
		_, ok := MetricValueForNode(samples, "mco_snapshot_timestamp_ms", pod.Spec.NodeName)
		Expect(ok).To(BeTrue())

		snapshotTimes, err := WaitCollectorSnapshotLogTimes(k8sClient, ctx, &pod, 3, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		for i := 1; i < len(snapshotTimes); i++ {
			interval := snapshotTimes[i].Sub(snapshotTimes[i-1])
			Expect(interval).To(And(
				BeNumerically(">=", 800*time.Millisecond),
				BeNumerically("<=", 2500*time.Millisecond),
			), "snapshot collection interval should be about 1s")
		}
	})

	It("mco-collector_FIT_3 Collector 按节点标签暴露指标", func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		schedulableNodes, err := CountSchedulableNodes(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		if schedulableNodes < 2 {
			Skip("need at least two schedulable nodes to verify multi-node label isolation")
		}
		Expect(pods.Items).To(HaveLen(schedulableNodes), "Collector Pod count should equal schedulable node count before multi-node validation")

		for _, pod := range pods.Items {
			By("检查 " + pod.Name + " 暴露的节点级指标 node 标签")
			samples, err := FetchCollectorMetricSamples(ctx, &pod, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			Expect(samples).NotTo(BeEmpty(), "Collector /metrics should expose mco_ metrics")

			for _, sample := range samples {
				node, ok := sample.Labels["node"]
				Expect(ok).To(BeTrue(), "node-level metric %s should contain node label", sample.Name)
				Expect(node).To(Equal(pod.Spec.NodeName), "metric %s node label should match Pod spec.nodeName", sample.Name)
			}
		}
	})

	It("mco-collector_FIT_4 Collector 向 Analyzer 流式上报快照", func(ctx SpecContext) {
		By("检查 Analyzer Service 暴露 gRPC 50051 端口")
		svc, err := GetService(k8sClient, ctx, MCOAnalyzerServiceName, MCODeployNamespace)
		Expect(err).NotTo(HaveOccurred())
		foundGRPCPort := false
		for _, port := range svc.Spec.Ports {
			if port.Port == MCOAnalyzerGRPCPort {
				foundGRPCPort = true
				break
			}
		}
		Expect(foundGRPCPort).To(BeTrue(), "mco-analyzer Service should expose gRPC port 50051")

		By("重启一个 Collector Pod 以重新建立 gRPC 流连接")
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty())
		oldPod := pods.Items[0]
		Expect(DeletePod(k8sClient, ctx, oldPod.Name, oldPod.Namespace, 30*time.Second)).To(Succeed())
		newPod, err := WaitCollectorPodRecreatedOnNode(k8sClient, ctx, oldPod.Spec.NodeName, oldPod.Name, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())

		By("等待重建后的 Collector 输出 MetricSnapshot")
		snapshotTimes, err := WaitCollectorSnapshotLogTimes(k8sClient, ctx, newPod, 3, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())

		By("检查重建后的 Collector 持续输出 MetricSnapshot 字段")
		logs, err := GetContainerLogs(k8sClient, ctx, newPod)
		Expect(err).NotTo(HaveOccurred())
		requiredFields := []string{
			`"time":"`,
			`"msg":"snapshot collected"`,
			`"node":"`,
			`"llc_miss":"`,
			`"llc_occupancy":"`,
			`"bio_p99_ms":"`,
			`"psi_available":"`,
		}
		for _, field := range requiredFields {
			Expect(logs).To(ContainSubstring(field), "Collector snapshot logs should contain field %s", field)
		}

		By("检查 MetricSnapshot 发送间隔约 1s")
		for i := 1; i < len(snapshotTimes); i++ {
			interval := snapshotTimes[i].Sub(snapshotTimes[i-1])
			Expect(interval).To(And(BeNumerically(">=", 800*time.Millisecond), BeNumerically("<=", 2500*time.Millisecond)),
				"MetricSnapshot interval should be about 1s")
		}
	})

	It("mco-collector_FIT_6 Collector 指标端点不依赖 Analyzer 可用性", func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty(), "at least one Collector Pod should exist")
		pod := pods.Items[0]

		samplesBefore, err := FetchCollectorMetricSamples(ctx, &pod, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		timestampBefore, ok := MetricValueForNode(samplesBefore, "mco_snapshot_timestamp_ms", pod.Spec.NodeName)
		Expect(ok).To(BeTrue())

		By("将 Analyzer 副本数调整为 0 并等待不少于 3 个采集周期")
		Expect(ScaleDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 0)).To(Succeed())
		DeferCleanup(func(ctx SpecContext) {
			_ = ScaleDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 1)
			_ = WaitCollectorAnalyzerReady(ctx)
		})
		Expect(WaitDeploymentReplicas(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 0, 60*time.Second)).To(Succeed())
		time.Sleep(4 * time.Second)

		By("检查 Collector /metrics 仍返回 Prometheus 指标且 timestamp 递增")
		metricsText, err := FetchPodMetricsViaPortForward(ctx, pod.Name, pod.Namespace, MCOCollectorMetricsPort, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		Expect(metricsText).To(ContainSubstring("# HELP"))
		Expect(metricsText).To(ContainSubstring("mco_snapshot_timestamp_ms"))
		samplesAfter, err := ParsePrometheusMetrics(metricsText)
		Expect(err).NotTo(HaveOccurred())
		timestampAfter, ok := MetricValueForNode(samplesAfter, "mco_snapshot_timestamp_ms", pod.Spec.NodeName)
		Expect(ok).To(BeTrue())
		Expect(timestampAfter).To(BeNumerically(">", timestampBefore), "mco_snapshot_timestamp_ms should keep increasing while Analyzer is unavailable")
	})

	It("mco-collector_FIT_7 Collector 识别 PSI 可用路径", Label("skip-temporarily"), func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		for _, pod := range pods.Items {
			pressureAvailable, err := CollectorPodPressureAvailable(k8sClient, ctx, &pod)
			Expect(err).NotTo(HaveOccurred())
			if !pressureAvailable {
				continue
			}

			samples, err := FetchCollectorMetricSamples(ctx, &pod, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			psiAvailable, ok := MetricValueForNode(samples, "mco_collector_psi_available", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			Expect(psiAvailable).To(Equal(float64(1)), "Collector should report PSI available when /proc/pressure is accessible")

			psiIO, ok := MetricValueForNode(samples, "mco_psi_io_some_avg10_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			psiMemory, ok := MetricValueForNode(samples, "mco_psi_memory_some_avg10_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			Expect(psiIO).NotTo(Equal(float64(-1)), "PSI available node psi_io should not be -1")
			Expect(psiMemory).NotTo(Equal(float64(-1)), "PSI available node psi_memory should not be -1")

			By("检查对应 Collector snapshot 日志记录 PSI 可用且填充实际采样值")
			Expect(WaitCollectorPSIAvailableSnapshotLog(k8sClient, ctx, &pod, 30*time.Second)).To(Succeed())
			return
		}

		Skip("no Collector node with accessible /proc/pressure found in current cluster")
	})

	It("mco-collector_FIT_8 Collector 执行 PSI 不可用降级采集", func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		for _, pod := range pods.Items {
			pressureAvailable, err := CollectorPodPressureAvailable(k8sClient, ctx, &pod)
			Expect(err).NotTo(HaveOccurred())
			if pressureAvailable {
				continue
			}

			samples, err := FetchCollectorMetricSamples(ctx, &pod, 30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			psiAvailable, ok := MetricValueForNode(samples, "mco_collector_psi_available", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			Expect(psiAvailable).To(Equal(float64(0)), "Collector should report PSI unavailable when /proc/pressure is inaccessible")

			psiIO, ok := MetricValueForNode(samples, "mco_psi_io_some_avg10_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			psiMemory, ok := MetricValueForNode(samples, "mco_psi_memory_some_avg10_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			memAvailableRatio, ok := MetricValueForNode(samples, "mco_mem_available_ratio_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())
			pageScanRate, ok := MetricValueForNode(samples, "mco_pgscan_rate_pages_per_second_raw", pod.Spec.NodeName)
			Expect(ok).To(BeTrue())

			Expect(psiIO).To(Equal(float64(-1)), "PSI unavailable node psi_io should fall back to -1")
			Expect(psiMemory).To(Equal(float64(-1)), "PSI unavailable node psi_memory should fall back to -1")
			Expect(memAvailableRatio).To(And(BeNumerically(">=", 0), BeNumerically("<=", 1)), "fallback memory available ratio should be within 0~1")
			Expect(pageScanRate).To(BeNumerically(">=", 0), "fallback page scan rate should be non-negative")

			By("检查对应 Collector snapshot 日志记录 PSI 不可用并填充降级值")
			Expect(WaitCollectorPSIUnavailableSnapshotLog(k8sClient, ctx, &pod, 30*time.Second)).To(Succeed())
			return
		}

		Skip("no Collector node with inaccessible /proc/pressure found in current cluster")
	})

	It("mco-collector_DFX_1 Collector 长时间断链本地缓存与恢复", func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty(), "at least one Collector Pod should exist")
		pod := pods.Items[0]

		report, err := WaitNodeInterferenceReportNode(k8sClient, ctx, MCONodeInterferenceReport, pod.Spec.NodeName, 30*time.Second)
		Expect(err).NotTo(HaveOccurred())
		sequenceBefore, ok := NodeReportFloat(report, "sequence")
		Expect(ok).To(BeTrue(), "node report sequence should exist before disconnect")

		By("停止 Analyzer 60s 以上以模拟 Collector 长时间断链")
		Expect(ScaleDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 0)).To(Succeed())
		DeferCleanup(func(ctx SpecContext) {
			_ = ScaleDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 1)
			_ = WaitCollectorAnalyzerReady(ctx)
		})
		Expect(WaitDeploymentReplicas(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 0, 60*time.Second)).To(Succeed())
		time.Sleep(65 * time.Second)

		By("检查 Collector 持续运行且记录限频失败/本地缓存日志")
		currentPod, err := k8sClient.Clientset.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(currentPod.Status.Phase).To(Equal(corev1.PodRunning))
		for _, cs := range currentPod.Status.ContainerStatuses {
			Expect(cs.RestartCount).To(BeZero(), "Collector should not restart during long Analyzer disconnect")
		}
		Expect(WaitCollectorLogContainsAny(k8sClient, ctx, &pod, 10*time.Second,
			"analyzer send failed; writing local cache",
			"send queue full; writing local cache",
		)).To(Succeed())

		By("检查本地 JSONL 缓存文件已生成并写入快照")
		stdout, stderr, err := ExecInPodOutput(k8sClient, ctx, pod.Namespace, pod.Name, pod.Spec.Containers[0].Name, []string{"sh", "-c", "test -s /var/log/mco-collector/metrics.jsonl && tail -n 1 /var/log/mco-collector/metrics.jsonl"})
		Expect(err).NotTo(HaveOccurred(), "local cache should be non-empty: stderr=%s", stderr)
		Expect(stdout).To(ContainSubstring(`"node"`))
		Expect(stdout).To(ContainSubstring(`"metrics"`))

		By("恢复 Analyzer 后检查 Collector 自动重连并继续上报最新快照")
		Expect(ScaleDeployment(k8sClient, ctx, MCOAnalyzerDeploymentName, MCODeployNamespace, 1)).To(Succeed())
		Expect(WaitCollectorAnalyzerReady(ctx)).To(Succeed())
		recoveryTime := time.Now().Add(-1 * time.Second)
		Eventually(func(g Gomega) {
			logs, err := GetContainerLogs(k8sClient, ctx, &pod)
			g.Expect(err).NotTo(HaveOccurred())
			snapshotTimes, err := CollectorSnapshotLogTimes(logs, pod.Spec.NodeName)
			g.Expect(err).NotTo(HaveOccurred())
			countAfterRecovery := 0
			for _, snapshotTime := range snapshotTimes {
				if !snapshotTime.Before(recoveryTime) {
					countAfterRecovery++
				}
			}
			g.Expect(countAfterRecovery).To(BeNumerically(">=", 5), "Collector should keep collecting for at least 5 cycles after Analyzer recovers")
		}, 15*time.Second, 1*time.Second).Should(Succeed())
		_, err = WaitNodeInterferenceReportNodeSequence(k8sClient, ctx, MCONodeInterferenceReport, pod.Spec.NodeName, sequenceBefore+1, 60*time.Second)
		Expect(err).NotTo(HaveOccurred())
	})

	It("mco-collector_DFX_3 Collector 权限最小化验证", Label("skip-temporarily"), func(ctx SpecContext) {
		pods, err := GetCollectorPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(pods.Items).NotTo(BeEmpty(), "at least one Collector Pod should exist")
		nodeNames := make([]string, 0, len(pods.Items))
		for _, pod := range pods.Items {
			nodeNames = append(nodeNames, pod.Spec.NodeName)
		}

		ds, err := GetDaemonSet(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(ds.Spec.Template.Spec.Containers).NotTo(BeEmpty())
		containerName := ds.Spec.Template.Spec.Containers[0].Name

		By("去掉 Collector SYS_ADMIN capability 并等待 DaemonSet 滚动完成")
		originalCaps, err := PatchDaemonSetContainerCapabilities(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, containerName, []corev1.Capability{"SYS_RESOURCE"})
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_, _ = PatchDaemonSetContainerCapabilities(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, containerName, originalCaps)
			_ = WaitDaemonSetReady(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, 120*time.Second)
		})
		Expect(WaitDaemonSetReady(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, 120*time.Second)).To(Succeed())

		errorMetrics := []string{"bioLatencyP95Ms", "bioLatencyP99Ms", "bioSamples", "cacheReferences", "cacheMisses", "llcMissRate"}
		By("检查 bcc/perf 相关采集器 availability 变为 error")
		Eventually(func(g Gomega) {
			report, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
			g.Expect(err).NotTo(HaveOccurred())
			nodes, err := NodeInterferenceReportNodes(report)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(nodes).NotTo(BeEmpty())
			for nodeName, raw := range nodes {
				nodeReport, ok := raw.(map[string]interface{})
				g.Expect(ok).To(BeTrue(), "status.nodes.%s should be an object", nodeName)
				for _, metric := range errorMetrics {
					g.Expect(NodeReportAvailabilityHas(nodeReport, metric, false, "error")).To(BeTrue(), "node %s metric %s should be unavailable with error", nodeName, metric)
				}
			}
		}, 60*time.Second, 2*time.Second).Should(Succeed())

		By("恢复 Collector capability 并检查 bcc/perf 采集器恢复 ok")
		_, err = PatchDaemonSetContainerCapabilities(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, containerName, originalCaps)
		Expect(err).NotTo(HaveOccurred())
		Expect(WaitDaemonSetReady(k8sClient, ctx, MCOCollectorDaemonSetName, MCODeployNamespace, 120*time.Second)).To(Succeed())
		Eventually(func(g Gomega) {
			report, err := GetNodeInterferenceReport(k8sClient, ctx, MCONodeInterferenceReport)
			g.Expect(err).NotTo(HaveOccurred())
			nodes, err := NodeInterferenceReportNodes(report)
			g.Expect(err).NotTo(HaveOccurred())
			g.Expect(nodes).NotTo(BeEmpty())
			for nodeName, raw := range nodes {
				nodeReport, ok := raw.(map[string]interface{})
				g.Expect(ok).To(BeTrue(), "status.nodes.%s should be an object", nodeName)
				for _, metric := range errorMetrics {
					g.Expect(NodeReportAvailabilityHas(nodeReport, metric, true, "ok")).To(BeTrue(), "node %s metric %s should recover to ok", nodeName, metric)
				}
			}
		}, 60*time.Second, 2*time.Second).Should(Succeed())
	})
})
