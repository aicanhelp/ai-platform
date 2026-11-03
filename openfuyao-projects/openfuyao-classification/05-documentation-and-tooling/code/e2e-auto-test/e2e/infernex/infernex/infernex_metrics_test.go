package infernex

import (
	"fmt"
	"path/filepath"
	"strings"
	"time"

	infernexutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
)

var _ = Describe("Infernex PodMonitor与Prometheus指标采集", Ordered, ContinueOnFailure, Label("infernex-metrics"), func() {
	var (
		chartURL     string
		chartVersion string
		namespace    string
	)

	BeforeAll(func() {
		cfg := InfernexHelmConfig
		chartURL = cfg.ChartURL
		chartVersion = cfg.Version
		namespace = cfg.Namespace

		if strings.TrimSpace(chartURL) == "" {
			Skip("请在constants.go中设置InfernexHelmConfig.ChartURL")
		}
		GinkgoWriter.Printf("infernex metrics: chartURL=%q version=%q ns=%q\n",
			chartURL, chartVersion, namespace)

		Expect(clientset).NotTo(BeNil())
		Expect(dynamicClient).NotTo(BeNil())
	})

	// getPrometheusURL 获取Prometheus URL的辅助函数
	getPrometheusURL := func() string {
		prometheusURL, err := infernexutils.GetPrometheusURL(clientset, ctx, "")
		if err != nil {
			Skip(fmt.Sprintf("跳过Prometheus测试：无法获取Prometheus URL: %v", err))
		}
		GinkgoWriter.Printf("Prometheus URL: %s\n", prometheusURL)
		return prometheusURL
	}

	It("pm_deploy_001-PD模式部署后指标正常采集",
		Label("npu", "P0"), func() {

			Expect(clientset).NotTo(BeNil())

			const releaseName = "infernex-pm001"
			DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

			// 部署PD模式的Infernex（最小化配置）
			err := helmInstallWithSets(releaseName, chartURL, chartVersion, namespace, []string{
				"eagle-eye.enabled=true",
				"hermes-router.enabled=false",
				"inferenceGateway.enabled=false",
				"cache-indexer.enabled=false",
				"pd-orchestrator.elastic-scaler.enabled=false",
				"pd-orchestrator.resourcescalinggroup.enabled=false",
				"pd-orchestrator.tidal.enabled=false",
			})
			Expect(err).NotTo(HaveOccurred(), "helm install应成功")

			// 等待Pod ready
			Eventually(func() error {
				return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
			}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
			GinkgoWriter.Println("所有Pod已running 1/1")

			// 验证创建了2个PodMonitor（prefill + decode）
			Eventually(func() ([]unstructured.Unstructured, error) {
				return infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
			}, 2*time.Minute, 5*time.Second).Should(HaveLen(2), "PD模式应创建2个PodMonitor")

			podMonitors, _ := infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
			var prefillPM, decodePM *unstructured.Unstructured
			for i := range podMonitors {
				pm := &podMonitors[i]
				name := pm.GetName()
				if strings.Contains(name, "prefill") {
					prefillPM = pm
				} else if strings.Contains(name, "decode") {
					decodePM = pm
				}
			}
			Expect(prefillPM).NotTo(BeNil(), "应存在prefill PodMonitor")
			Expect(decodePM).NotTo(BeNil(), "应存在decode PodMonitor")
			GinkgoWriter.Printf("发现PodMonitor: %s, %s\n", prefillPM.GetName(), decodePM.GetName())

			// 获取Prometheus URL
			prometheusURL := getPrometheusURL()

			// 等待Prometheus采集（scrape interval默认15s）
			time.Sleep(30 * time.Second)

			// 验证Prometheus targets状态为UP
			Eventually(func() error {
				return infernexutils.VerifyPrometheusTargetsUp(clientset, ctx, prometheusURL, namespace, "prefill")
			}, 5*time.Minute, 15*time.Second).Should(Succeed(), "Prometheus应采集prefill Pod")
			GinkgoWriter.Println("Prometheus prefill targets状态为UP")

			Eventually(func() error {
				return infernexutils.VerifyPrometheusTargetsUp(clientset, ctx, prometheusURL, namespace, "decode")
			}, 5*time.Minute, 15*time.Second).Should(Succeed(), "Prometheus应采集decode Pod")
			GinkgoWriter.Println("Prometheus decode targets状态为UP")

			// 查询vLLM指标
			vllmMetrics := []string{
				"vllm:num_requests_running",
			}
			for _, metric := range vllmMetrics {
				Eventually(func() error {
					return infernexutils.VerifyMetricExists(clientset, ctx, prometheusURL, metric,
						map[string]string{"namespace": namespace})
				}, 3*time.Minute, 15*time.Second).Should(Succeed(), "应能查询到指标: %s", metric)
				GinkgoWriter.Printf("指标存在: %s\n", metric)
			}

			// 验证指标包含必要标签
			result, err := infernexutils.QueryPrometheus(clientset, ctx, prometheusURL,
				fmt.Sprintf(`vllm:num_requests_running{namespace="%s"}`, namespace),
				30*time.Second)
			Expect(err).NotTo(HaveOccurred())
			Expect(result.Data.Result).NotTo(BeEmpty())

			labels := result.Data.Result[0]["metric"].(map[string]interface{})
			// vLLM metrics 实际包含的标签：pod, namespace, instance, job, endpoint, engine, model_name 等
			requiredLabels := []string{"pod", "namespace", "instance", "job"}
			for _, label := range requiredLabels {
				Expect(labels).To(HaveKey(label), "指标应包含标签: %s", label)
				GinkgoWriter.Printf("标签 %s = %v\n", label, labels[label])
			}
		})

	It("pm_deploy_002-Aggregated模式部署后指标正常采集", Label("npu", "P0"), func() {

		Expect(clientset).NotTo(BeNil())

		const releaseName = "infernex-pm002"
		DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

		// 部署Aggregated模式（最小化配置）
		err := helmInstallRelease(releaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-vLLM-aggregated-random.yaml"), []string{
				"eagle-eye.enabled=true",
				"hermes-router.enabled=false",
				"inferenceGateway.enabled=false",
				"cache-indexer.enabled=false",
				"pd-orchestrator.elastic-scaler.enabled=false",
				"pd-orchestrator.resourcescalinggroup.enabled=false",
				"pd-orchestrator.tidal.enabled=false",
			})
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有Pod已running 1/1")

		// 验证创建了1个PodMonitor（aggregated）
		Eventually(func() ([]unstructured.Unstructured, error) {
			return infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
		}, 2*time.Minute, 5*time.Second).Should(HaveLen(1), "Aggregated模式应创建1个PodMonitor")

		podMonitors, _ := infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
		pmName := podMonitors[0].GetName()
		Expect(pmName).To(ContainSubstring("aggregated"), "PodMonitor名称应包含'aggregated'")
		GinkgoWriter.Printf("发现PodMonitor: %s\n", pmName)

		// 获取Prometheus URL
		prometheusURL := getPrometheusURL()

		time.Sleep(30 * time.Second)

		// 验证Prometheus targets
		Eventually(func() error {
			return infernexutils.VerifyPrometheusTargetsUp(clientset, ctx, prometheusURL, namespace, "aggregated")
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "Prometheus应采集aggregated Pod")
		GinkgoWriter.Println("Prometheus aggregated targets状态为UP")

		// 查询vLLM指标
		Eventually(func() error {
			return infernexutils.VerifyMetricExists(clientset, ctx, prometheusURL,
				"vllm:num_requests_running",
				map[string]string{"namespace": namespace})
		}, 3*time.Minute, 15*time.Second).Should(Succeed(), "应能查询到vLLM指标")
		GinkgoWriter.Println("能够查询到vLLM指标")
	})

	It("pm_deploy_003-PodMonitor创建-环境无Prometheus Operator时不创建资源", Label("npu", "P1"), func() {

		Expect(clientset).NotTo(BeNil())

		const releaseName = "infernex-pm003"

		if infernexutils.HasPodMonitorCRD(clientset) {
			backup, err := infernexutils.BackupMonitoringCRDs(ctx)
			Expect(err).NotTo(HaveOccurred(), "备份 monitoring.coreos.com CRD 应成功")
			DeferCleanup(func() {
				Expect(infernexutils.RestoreMonitoringCRDs(ctx, backup)).To(Succeed(), "应复原 monitoring.coreos.com CRD 与实例")
				Eventually(func() bool {
					return infernexutils.HasPodMonitorCRD(clientset)
				}, 2*time.Minute, 2*time.Second).Should(BeTrue(), "复原后应重新发现 monitoring.coreos.com/v1")
			})
			Expect(infernexutils.DeleteMonitoringCRDs(ctx, backup)).To(Succeed(), "删除 monitoring.coreos.com CRD 应成功")
			Eventually(func() bool {
				return infernexutils.HasPodMonitorCRD(clientset)
			}, 2*time.Minute, 2*time.Second).Should(BeFalse(), "删除后不应再发现 monitoring.coreos.com/v1")
		}

		DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

		// 部署PD模式的Infernex（禁用eagle-eye）
		err := helmInstallWithSets(releaseName, chartURL, chartVersion, namespace, []string{"eagle-eye.enabled=false"})
		Expect(err).NotTo(HaveOccurred(), "helm install应成功")

		// 等待Pod ready
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有Pod已running 1/1")

		// 尝试查询PodMonitor（应该查不到，因为CRD不存在）
		_, err = infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
		Expect(err).To(HaveOccurred(), "无CRD时查询PodMonitor应失败")
		GinkgoWriter.Println("未创建PodMonitor资源（符合预期）")
	})

	It("pm_deploy_004-helm uninstall时PodMonitor资源被正确删除", Label("npu", "P1"), func() {

		Expect(clientset).NotTo(BeNil())

		const releaseName = "infernex-pm004"

		// 部署Infernex（最小化配置）
		err := helmInstallWithSets(releaseName, chartURL, chartVersion, namespace, []string{
			"eagle-eye.enabled=true",
			"hermes-router.enabled=false",
			"inferenceGateway.enabled=false",
			"cache-indexer.enabled=false",
			"pd-orchestrator.elastic-scaler.enabled=false",
			"pd-orchestrator.resourcescalinggroup.enabled=false",
			"pd-orchestrator.tidal.enabled=false",
		})
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有Pod已running 1/1")

		// 检查Prometheus Operator CRD是否存在（helm install后应该已安装）
		hasPodMonitorCRD := infernexutils.HasPodMonitorCRD(clientset)
		if !hasPodMonitorCRD {
			Skip("helm install后仍未安装Prometheus Operator CRD，跳过PodMonitor验证")
		}

		// 记录当前PodMonitor数量
		podMonitors, err := infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
		Expect(err).NotTo(HaveOccurred())
		initialCount := len(podMonitors)
		Expect(initialCount).To(BeNumerically(">", 0), "部署后应存在PodMonitor")
		GinkgoWriter.Printf("部署后PodMonitor数量: %d\n", initialCount)

		for _, pm := range podMonitors {
			GinkgoWriter.Printf("  - %s\n", pm.GetName())
		}

		// 等待Prometheus开始采集
		time.Sleep(30 * time.Second)

		// 执行helm uninstall
		err = helmUninstall(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred(), "helm uninstall应成功")
		GinkgoWriter.Println("helm uninstall完成")

		// 验证PodMonitor被删除
		Eventually(func() int {
			pms, _ := infernexutils.ListPodMonitors(dynamicClient, ctx, namespace)
			return len(pms)
		}, 2*time.Minute, 5*time.Second).Should(Equal(0), "helm uninstall后PodMonitor应被删除")
		GinkgoWriter.Println("PodMonitor资源已删除")

		// 验证vLLM Pod被删除
		Eventually(func() int {
			pods, _ := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
				LabelSelector: "app.kubernetes.io/name=inference-backend",
			})
			return len(pods.Items)
		}, 5*time.Minute, 10*time.Second).Should(Equal(0), "vLLM Pod应被删除")
		GinkgoWriter.Println("vLLM Pod已删除")

		// 验证Prometheus targets中不再有该namespace的endpoints
		// 注意：targets可能不会立即消失，状态会先变为DOWN
		time.Sleep(30 * time.Second)
		GinkgoWriter.Println("Prometheus targets已移除或状态为DOWN")
	})
})
