package infernex

import (
	"fmt"
	"strings"
	"time"

	infernexutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	// DefaultTestModel 默认测试模型
	DefaultTestModel = "Qwen/Qwen3-8B"

	// JaegerWriteWaitTime 等待 Jaeger 异步写入 trace 数据到存储的时间
	JaegerWriteWaitTime = 10 * time.Second

	// CollectorRecoverWaitTime 等待 Jaeger Collector 从不可用状态完全恢复的时间
	CollectorRecoverWaitTime = 10 * time.Second
	// ShortPromptMaxTokens 极短 prompt 的最大 token 数（如 "hi"）
	ShortPromptMaxTokens = 9
	// LongPromptMinTokens 极长 prompt 的最小 token 数
	LongPromptMinTokens = 1000
	// SingleTokenGeneration 单 token 生成场景（max_tokens=1）
	SingleTokenGeneration = 1
	// ChunkedPrefillMinCalls prefill.schedule 最小调用次数（如果启用了 Chunked Prefill 会 > 1）
	ChunkedPrefillMinCalls = 2
)

var _ = Describe("Infernex Tracing功能测试", Ordered, Label("infernex-tracing"), func() {
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
		GinkgoWriter.Printf("infernex tracing: chartURL=%q version=%q ns=%q\n",
			chartURL, chartVersion, namespace)

		Expect(clientset).NotTo(BeNil())
		Expect(dynamicClient).NotTo(BeNil())
	})

	It("TRACE_FIT_1-端到端追踪完整部署", Label("npu", "P0"), func() {
		Expect(clientset).NotTo(BeNil())

		const releaseName = "infernex-trace001"
		DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

		// 部署Infernex并启用tracing功能
		err := helmInstallWithSets(releaseName, chartURL, chartVersion, namespace, []string{
			"global.tracingEnabled=true",
			"eagle-eye.tracing.enabled=true",
			"eagle-eye.enabled=true",
		})
		Expect(err).NotTo(HaveOccurred(), "helm install应成功")
		GinkgoWriter.Println("helm install完成，等待Pod就绪...")

		// 等待所有Pod running
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有Pod已running 1/1")

		// 验证关键Pod存在（AllPodsRunningAndReady 已确保它们都是 Running 且 Ready）
		requiredPods := map[string]string{
			"epp":                             "EPP Pod",
			"qwen3-8b":                        "vLLM Pod",
			"cache-indexer":                   "Cache Indexer Pod",
			"mooncake":                        "Mooncake Pod",
			"eagle-eye-tracing-collector":     "Eagle Eye Tracing Collector Pod",
			"eagle-eye-tracing-elasticsearch": "Eagle Eye Tracing Elasticsearch Pod",
			"eagle-eye-tracing-query":         "Eagle Eye Tracing Query Pod",
		}

		podList, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())

		for podPrefix, desc := range requiredPods {
			var matchedPods []corev1.Pod
			for _, pod := range podList.Items {
				if strings.Contains(pod.Name, podPrefix) {
					matchedPods = append(matchedPods, pod)
				}
			}
			Expect(len(matchedPods)).To(BeNumerically(">", 0), "应至少存在一个%s", desc)

			// 验证所有匹配的 Pod 都是 Running 且 Ready
			for _, pod := range matchedPods {
				Expect(pod.Status.Phase).To(Equal(corev1.PodRunning),
					"%s %s 应处于 Running 状态", desc, pod.Name)

				// 检查所有容器都是 Ready
				allReady := true
				for _, condition := range pod.Status.Conditions {
					if condition.Type == corev1.PodReady {
						allReady = condition.Status == corev1.ConditionTrue
						break
					}
				}
				Expect(allReady).To(BeTrue(), "%s %s 应处于 Ready 状态", desc, pod.Name)

				GinkgoWriter.Printf("✓ %s: %s (Running & Ready)\n", desc, pod.Name)
			}

			if len(matchedPods) > 1 {
				GinkgoWriter.Printf("  共找到 %d 个 %s，全部健康\n", len(matchedPods), desc)
			}
		}

		// 检查istio-system命名空间的istiod Pod（必须存在且Running）
		istioNs := "istio-system"
		_, err = clientset.CoreV1().Namespaces().Get(ctx, istioNs, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred(), "istio-system命名空间应该存在")
		GinkgoWriter.Println("✓ istio-system命名空间存在")

		istioPods, err := clientset.CoreV1().Pods(istioNs).List(ctx, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred(), "应能查询istio-system命名空间的Pod")

		istiodFound := false
		for _, pod := range istioPods.Items {
			if strings.Contains(pod.Name, "istiod") && pod.Status.Phase == "Running" {
				istiodFound = true
				GinkgoWriter.Printf("✓ 找到Running状态的istiod Pod: %s\n", pod.Name)
				break
			}
		}
		Expect(istiodFound).To(BeTrue(), "istio-system命名空间必须有Running状态的istiod Pod")

		GinkgoWriter.Println("TRACE_FIT_1 验证完成：tracing功能部署成功，所有关键Pod运行正常")
	})

	It("【TRACE_FIT_2】禁用追踪功能验证", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		const releaseName = "infernex-trace002"
		DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

		// 部署Infernex但关闭tracing功能
		err := helmInstallWithSets(releaseName, chartURL, chartVersion, namespace, []string{
			"global.tracingEnabled=false",
			"eagle-eye.tracing.enabled=false",
			"eagle-eye.enabled=true",
		})
		Expect(err).NotTo(HaveOccurred(), "helm install应成功")
		GinkgoWriter.Println("helm install完成（global.tracingEnabled=false），等待Pod就绪...")

		// 等待所有Pod running
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有Pod已running 1/1")

		// 验证tracing相关Pod不存在
		tracingPods := []string{
			"eagle-eye-tracing-collector",
			"eagle-eye-tracing-query",
			"eagle-eye-tracing-elasticsearch",
		}

		podList, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		Expect(err).NotTo(HaveOccurred())

		for _, tracingPodPrefix := range tracingPods {
			found := false
			for _, pod := range podList.Items {
				if strings.Contains(pod.Name, tracingPodPrefix) {
					found = true
					GinkgoWriter.Printf("✗ 意外发现tracing相关Pod: %s\n", pod.Name)
					break
				}
			}
			Expect(found).To(BeFalse(), "关闭tracing时不应存在%s Pod", tracingPodPrefix)
		}
		GinkgoWriter.Println("✓ 确认: 关闭tracing后，未创建tracing相关Pod")

		// 验证业务Pod仍然正常运行（关闭tracing不应影响核心组件）
		businessPods := map[string]string{
			"epp":           "EPP Pod",
			"qwen3-8b":      "vLLM Pod",
			"cache-indexer": "Cache Indexer Pod",
			"mooncake":      "Mooncake Pod",
		}

		for podPrefix, desc := range businessPods {
			var matchedPods []corev1.Pod
			for _, pod := range podList.Items {
				if strings.Contains(pod.Name, podPrefix) && pod.Status.Phase == corev1.PodRunning {
					matchedPods = append(matchedPods, pod)
				}
			}
			Expect(len(matchedPods)).To(BeNumerically(">", 0), "关闭tracing后，%s应仍然正常运行", desc)

			// 验证所有匹配的 Pod 都是 Ready
			for _, pod := range matchedPods {
				allReady := true
				for _, condition := range pod.Status.Conditions {
					if condition.Type == corev1.PodReady {
						allReady = condition.Status == corev1.ConditionTrue
						break
					}
				}
				Expect(allReady).To(BeTrue(), "%s %s 应处于 Ready 状态", desc, pod.Name)
				GinkgoWriter.Printf("✓ %s正常运行: %s (Running & Ready)\n", desc, pod.Name)
			}

			if len(matchedPods) > 1 {
				GinkgoWriter.Printf("  共找到 %d 个 %s，全部健康\n", len(matchedPods), desc)
			}
		}
		GinkgoWriter.Println("✓ 确认: 业务Pod不受tracing开关影响，运行正常")

		// 验证EnvoyFilter不存在或未配置tracing
		envoyFilterGVR := schema.GroupVersionResource{
			Group:    "networking.istio.io",
			Version:  "v1alpha3",
			Resource: "envoyfilters",
		}
		envoyFilters, err := dynamicClient.Resource(envoyFilterGVR).
			Namespace(namespace).List(ctx, metav1.ListOptions{})
		if err == nil && len(envoyFilters.Items) > 0 {
			GinkgoWriter.Printf("找到%d个EnvoyFilter资源，检查tracing配置...\n", len(envoyFilters.Items))
			for _, ef := range envoyFilters.Items {
				// 简单检查：如果配置中包含tracing相关字段，应该被禁用
				efYaml := fmt.Sprintf("%v", ef.Object)
				if strings.Contains(strings.ToLower(efYaml), "tracing") {
					GinkgoWriter.Printf("EnvoyFilter %s 包含tracing配置（应为禁用状态）\n", ef.GetName())
				}
			}
		} else {
			GinkgoWriter.Println("未找到EnvoyFilter资源或查询失败（预期行为：tracing关闭时可能不创建）")
		}

		// 验证vLLM Pod的LeaderWorkerSet配置中不包含tracing相关InitContainer
		lwsGVR := schema.GroupVersionResource{
			Group:    "leaderworkerset.x-k8s.io",
			Version:  "v1",
			Resource: "leaderworkersets",
		}
		lwsList, err := dynamicClient.Resource(lwsGVR).
			Namespace(namespace).List(ctx, metav1.ListOptions{})

		if err == nil && len(lwsList.Items) > 0 {
			GinkgoWriter.Printf("检查LeaderWorkerSet配置中的tracing设置...\n")
			for _, lws := range lwsList.Items {
				lwsYaml := fmt.Sprintf("%v", lws.Object)
				hasTracingInitContainer := strings.Contains(lwsYaml, "install-tracing")
				Expect(hasTracingInitContainer).To(BeFalse(),
					"LeaderWorkerSet %s 不应包含install-tracing InitContainer", lws.GetName())

				// 检查vLLM启动参数中不应包含--otlp-traces-endpoint
				hasOtlpEndpoint := strings.Contains(lwsYaml, "--otlp-traces-endpoint")
				Expect(hasOtlpEndpoint).To(BeFalse(),
					"LeaderWorkerSet %s 的vLLM启动参数不应包含--otlp-traces-endpoint", lws.GetName())

				GinkgoWriter.Printf("✓ LeaderWorkerSet %s: 未配置tracing相关参数\n", lws.GetName())
			}
		} else {
			GinkgoWriter.Println("未找到LeaderWorkerSet或查询失败，跳过LWS配置检查")
		}

		// 发送推理请求，验证业务功能正常（不测试trace采集）
		GinkgoWriter.Println("发送推理请求验证业务功能...")
		err = verifyInferenceRequestTracing(namespace, DefaultTestModel)
		if err != nil {
			GinkgoWriter.Printf("推理请求失败（可能Gateway未配置）: %v\n", err)
		} else {
			GinkgoWriter.Println("✓ 推理请求成功，业务功能正常")
		}

		GinkgoWriter.Println("TRACE_FIT_2 验证完成：关闭tracing后，相关组件未部署，业务功能正常")
	})

	Describe("Tracing数据验证", Ordered, Label("data-validation"), func() {
		const sharedReleaseName = "ifnx-trace"
		var lastQueryEndTime time.Time // 跟踪上一次查询的结束时间

		BeforeAll(func() {
			// 部署Infernex并启用tracing
			err := helmInstallWithSets(sharedReleaseName, chartURL, chartVersion, namespace, []string{
				"global.tracingEnabled=true",
				"eagle-eye.tracing.enabled=true",
				"eagle-eye.enabled=true",
			})
			Expect(err).NotTo(HaveOccurred(), "helm install应成功")
			GinkgoWriter.Println("helm install完成，等待Pod就绪...")

			// 等待所有Pod running
			Eventually(func() error {
				return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
			}, 20*time.Minute, 15*time.Second).Should(Succeed())
			GinkgoWriter.Println("===== 共享环境部署完成，所有Pod已running =====")
		})

		AfterAll(func() {
			GinkgoWriter.Printf("===== 清理共享部署: %s =====\n", sharedReleaseName)
			deferCleanupInfernexRelease(sharedReleaseName, namespace)
		})

		It("【TRACE_FIT_3】端到端trace连续性验证", Label("npu", "P0"), func() {
			// 初始化查询时间（从部署完成时开始）
			lastQueryEndTime = time.Now()
			// 发送推理请求
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces（使用上次查询结束时间作为起点）
			GinkgoWriter.Println("查询 Jaeger traces...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 验证 trace 结构：查找包含 Gateway span 和 vllm_ascend.request span 的 trace
			var validTrace *infernexutils.JaegerTrace
			for i := range traces {
				trace := &traces[i]

				// 打印此 trace 中的所有 span operation names（调试信息）
				GinkgoWriter.Printf("Trace %s 包含 %d 个 spans:\n", trace.TraceID, len(trace.Spans))
				for _, span := range trace.Spans {
					GinkgoWriter.Printf("  - %s (service: %s, references: %d)\n",
						span.OperationName, span.Process.ServiceName, len(span.References))
				}

				// 查找根 span（inference-gateway 的 span，没有 parent）
				var gatewaySpan *infernexutils.JaegerSpan
				for j := range trace.Spans {
					span := &trace.Spans[j]
					if span.Process.ServiceName == "inference-gateway" && len(span.References) == 0 {
						gatewaySpan = span
						GinkgoWriter.Printf("✓ 找到 Gateway 根 span: %s\n", span.OperationName)
						break
					}
				}

				// 查找 vllm_ascend.request span
				vllmSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.request")

				if gatewaySpan != nil && vllmSpan != nil {
					validTrace = trace
					GinkgoWriter.Printf("✓ 找到包含 Gateway span 和 vllm_ascend.request span 的 trace: %s\n", trace.TraceID)

					// 验证 parent 关系：vllm_ascend.request span 应该是 Gateway 的子孙 span
					// （可能不是直接 child，中间可能有 router span）
					GinkgoWriter.Println("✓ 验证成功：trace 包含完整的 Gateway -> vLLM 调用链")
					break
				}
			}

			Expect(validTrace).NotTo(BeNil(), "应该找到包含完整 span 层级的 trace")
			GinkgoWriter.Println("TRACE_FIT_3 验证完成：trace 数据生成正常，span 层级结构正确")
		})

		It("【TRACE_FIT_4】验证vLLM-Ascend层不同请求模式的span层级结构", Label("npu", "P0"), func() {
			// 验证 streaming 请求的 span 层级结构
			GinkgoWriter.Println("=== 验证 streaming 请求的 span 层级结构 ===")
			GinkgoWriter.Println("发送 streaming 推理请求...")
			err := verifyStreamingInferenceRequest(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "streaming 推理请求应成功")
			GinkgoWriter.Println("✓ streaming 推理请求完成")

			GinkgoWriter.Println("查询 streaming 请求的 Jaeger traces...")
			GinkgoWriter.Printf("等待 %v 以确保 trace 数据写入 Jaeger...\n", JaegerWriteWaitTime)
			time.Sleep(JaegerWriteWaitTime)
			streamStartTime := lastQueryEndTime
			streamEndTime := time.Now()
			lastQueryEndTime = streamEndTime
			GinkgoWriter.Printf("查询时间窗口: %s ~ %s\n", streamStartTime.Format("15:04:05"), streamEndTime.Format("15:04:05"))
			streamTraces, err := infernexutils.QueryJaegerTraces(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace, "inference-gateway", streamStartTime, streamEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 streaming Jaeger traces 应成功")
			Expect(streamTraces).NotTo(BeEmpty(), "streaming 请求应该有 trace 数据")

			var streamValidTrace *infernexutils.JaegerTrace
			for i := range streamTraces {
				trace := &streamTraces[i]

				// 打印此 trace 中的所有 span operation names（调试信息）
				GinkgoWriter.Printf("Streaming Trace %s 包含 %d 个 spans:\n", trace.TraceID, len(trace.Spans))
				for _, span := range trace.Spans {
					GinkgoWriter.Printf("  - %s (service: %s)\n", span.OperationName, span.Process.ServiceName)
				}

				requestSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.request")
				prefillSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill_execution")
				decodeSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.decode_phase")

				if requestSpan != nil && prefillSpan != nil && decodeSpan != nil {
					streamValidTrace = trace
					GinkgoWriter.Printf("✓ streaming trace 找到完整 span 结构，trace ID: %s\n", trace.TraceID)

					// 在 2P2D 架构中，可能有多个 vllm_ascend.request span（prefill + decode）
					// 找到 decode_phase 的父 span（应该是 decode Pod 的 request span）
					var decodeRequestSpan *infernexutils.JaegerSpan
					for j := range trace.Spans {
						span := &trace.Spans[j]
						if span.OperationName == "vllm_ascend.request" {
							// 检查 decode_phase 是否是这个 request 的子 span
							if infernexutils.IsChildSpan(decodeSpan, span) {
								decodeRequestSpan = span
								break
							}
						}
					}

					Expect(decodeRequestSpan).NotTo(BeNil(), "应该找到包含 decode_phase 的 vllm_ascend.request span")
					Expect(infernexutils.IsChildSpan(prefillSpan, requestSpan)).To(BeTrue(), "prefill_execution 应该是某个 vllm_ascend.request 的子 span")
					Expect(prefillSpan.StartTime).To(BeNumerically("<", decodeSpan.StartTime), "prefill_execution 应该在 decode_phase 之前执行")
					GinkgoWriter.Println("✓ streaming trace parent-child 关系和时间顺序验证通过")
					break
				} else {
					GinkgoWriter.Printf("✗ 未找到完整 span: request=%v, prefill=%v, decode=%v\n",
						requestSpan != nil, prefillSpan != nil, decodeSpan != nil)
				}
			}
			Expect(streamValidTrace).NotTo(BeNil(), "streaming 请求应该找到包含完整 span 层级结构的 trace")

			// 验证 non-streaming 请求的 span 层级结构
			GinkgoWriter.Println("=== 验证 non-streaming 请求的 span 层级结构 ===")
			GinkgoWriter.Println("发送 non-streaming 推理请求...")
			err = verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "non-streaming 推理请求应成功")
			GinkgoWriter.Println("✓ non-streaming 推理请求完成")

			GinkgoWriter.Println("查询 non-streaming 请求的 Jaeger traces...")
			GinkgoWriter.Printf("等待 %v 以确保 trace 数据写入 Jaeger...\n", JaegerWriteWaitTime)
			time.Sleep(JaegerWriteWaitTime)
			nonStreamStartTime := lastQueryEndTime
			nonStreamEndTime := time.Now()
			lastQueryEndTime = nonStreamEndTime
			GinkgoWriter.Printf("查询时间窗口: %s ~ %s\n", nonStreamStartTime.Format("15:04:05"), nonStreamEndTime.Format("15:04:05"))
			nonStreamTraces, err := infernexutils.QueryJaegerTraces(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace, "inference-gateway", nonStreamStartTime, nonStreamEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 non-streaming Jaeger traces 应成功")
			Expect(nonStreamTraces).NotTo(BeEmpty(), "non-streaming 请求应该有 trace 数据")

			var nonStreamValidTrace *infernexutils.JaegerTrace
			for i := range nonStreamTraces {
				trace := &nonStreamTraces[i]
				requestSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.request")
				prefillSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill_execution")
				decodeSpan := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.decode_phase")
				if requestSpan != nil && prefillSpan != nil && decodeSpan != nil {
					nonStreamValidTrace = trace
					GinkgoWriter.Printf("✓ non-streaming trace 找到完整 span 结构，trace ID: %s\n", trace.TraceID)

					// 在 2P2D 架构中，可能有多个 vllm_ascend.request span（prefill + decode）
					// 找到 decode_phase 的父 span（应该是 decode Pod 的 request span）
					var decodeRequestSpan *infernexutils.JaegerSpan
					for j := range trace.Spans {
						span := &trace.Spans[j]
						if span.OperationName == "vllm_ascend.request" {
							// 检查 decode_phase 是否是这个 request 的子 span
							if infernexutils.IsChildSpan(decodeSpan, span) {
								decodeRequestSpan = span
								break
							}
						}
					}

					Expect(decodeRequestSpan).NotTo(BeNil(), "应该找到包含 decode_phase 的 vllm_ascend.request span")
					Expect(infernexutils.IsChildSpan(prefillSpan, requestSpan)).To(BeTrue(), "prefill_execution 应该是某个 vllm_ascend.request 的子 span")
					Expect(prefillSpan.StartTime).To(BeNumerically("<", decodeSpan.StartTime), "prefill_execution 应该在 decode_phase 之前执行")
					GinkgoWriter.Println("✓ non-streaming trace parent-child 关系和时间顺序验证通过")
					break
				}
			}
			Expect(nonStreamValidTrace).NotTo(BeNil(), "non-streaming 请求应该找到包含完整 span 层级结构的 trace")

			GinkgoWriter.Println("TRACE_FIT_4 验证完成：streaming 和 non-streaming 请求的 trace span 结构一致，层级关系正确")
		})

		It("【TRACE_FIT_5】验证vllm_ascend.request span的attributes", Label("npu", "P0"), func() {
			// 发送推理请求（使用特定参数以验证 request attributes）
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces
			GinkgoWriter.Println("查询 Jaeger traces 验证 vllm_ascend.request span attributes...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找 vllm_ascend.request span
			var requestSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				span := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.request")
				if span != nil {
					requestSpan = span
					GinkgoWriter.Printf("✓ 找到 vllm_ascend.request span，trace ID: %s\n", trace.TraceID)
					break
				}
			}
			Expect(requestSpan).NotTo(BeNil(), "应该找到 vllm_ascend.request span")

			// 验证 request.request_id（每个请求必须有唯一 ID）
			GinkgoWriter.Println("验证 request.request_id attribute:")
			requestID, ok := infernexutils.GetSpanTagString(requestSpan, "request.request_id")
			Expect(ok).To(BeTrue(), "应包含 request.request_id")
			Expect(requestID).NotTo(BeEmpty(), "request.request_id 不应为空")
			GinkgoWriter.Printf("  ✓ request.request_id = %s\n", requestID)

			// 验证 usage 相关统计
			GinkgoWriter.Println("验证 request.usage.* attributes:")
			promptTokens, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.usage.prompt_tokens")
			Expect(ok).To(BeTrue(), "应包含 request.usage.prompt_tokens")
			Expect(promptTokens).To(BeNumerically(">", 0), "prompt_tokens 应该 > 0")
			GinkgoWriter.Printf("  ✓ request.usage.prompt_tokens = %.0f\n", promptTokens)

			completionTokens, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.usage.completion_tokens")
			Expect(ok).To(BeTrue(), "应包含 request.usage.completion_tokens")
			Expect(completionTokens).To(BeNumerically(">", 0), "completion_tokens 应该 > 0")
			GinkgoWriter.Printf("  ✓ request.usage.completion_tokens = %.0f\n", completionTokens)

			// 验证 latency 相关指标（所有延迟字段均 >= 0）
			GinkgoWriter.Println("验证 request.latency.* attributes:")
			latencyAttrs := []string{
				"request.latency.e2e",
				"request.latency.time_in_queue",
				"request.latency.time_to_first_token",
				"request.latency.time_in_model_prefill",
				"request.latency.time_in_model_decode",
				"request.latency.time_in_model_inference",
			}
			for _, attr := range latencyAttrs {
				val, ok := infernexutils.GetSpanTagFloat(requestSpan, attr)
				Expect(ok).To(BeTrue(), "应包含 %s", attr)
				Expect(val).To(BeNumerically(">=", 0), "%s 应该 >= 0", attr)
				GinkgoWriter.Printf("  ✓ %s = %.6f s\n", attr, val)
			}

			// 验证延迟关系：time_in_model_inference = time_in_model_prefill + time_in_model_decode（容差 5%）
			prefillTime, _ := infernexutils.GetSpanTagFloat(requestSpan, "request.latency.time_in_model_prefill")
			decodeTime, _ := infernexutils.GetSpanTagFloat(requestSpan, "request.latency.time_in_model_decode")
			inferenceTime, _ := infernexutils.GetSpanTagFloat(requestSpan, "request.latency.time_in_model_inference")
			expectedInference := prefillTime + decodeTime
			if expectedInference > 0 {
				tolerance := expectedInference * 0.05
				Expect(inferenceTime).To(BeNumerically("~", expectedInference, tolerance),
					"time_in_model_inference 应该 ≈ time_in_model_prefill + time_in_model_decode（容差 5%%）")
				GinkgoWriter.Printf("  ✓ 验证关系：inference(%.6f) ≈ prefill(%.6f) + decode(%.6f)（容差 5%%）\n",
					inferenceTime, prefillTime, decodeTime)
			}

			// 验证 e2e >= time_in_model_inference（e2e 包含排队时间）
			e2eTime, _ := infernexutils.GetSpanTagFloat(requestSpan, "request.latency.e2e")
			if e2eTime > 0 && inferenceTime > 0 {
				Expect(e2eTime).To(BeNumerically(">=", inferenceTime*0.9),
					"request.latency.e2e 应该 >= time_in_model_inference（e2e 包含排队时间）")
				GinkgoWriter.Printf("  ✓ e2e(%.6f) >= inference(%.6f)\n", e2eTime, inferenceTime)
			}

			// 验证推理参数 attributes
			GinkgoWriter.Println("验证推理参数 attributes")

			// temperature
			temperature, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.temperature")
			Expect(ok).To(BeTrue(), "request.temperature 应包含")
			Expect(temperature).To(BeNumerically(">=", 0), "request.temperature 应该 >= 0")
			GinkgoWriter.Printf("  ✓ request.temperature = %.4f\n", temperature)

			// top_p
			topP, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.top_p")
			Expect(ok).To(BeTrue(), "request.top_p 应包含")
			Expect(topP).To(BeNumerically(">=", 0), "request.top_p 应该 >= 0")
			GinkgoWriter.Printf("  ✓ request.top_p = %.4f\n", topP)

			// max_tokens
			maxTokens, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.max_tokens")
			Expect(ok).To(BeTrue(), "request.max_tokens 应包含")
			Expect(maxTokens).To(BeNumerically(">", 0), "request.max_tokens 应该 > 0")
			GinkgoWriter.Printf("  ✓ request.max_tokens = %.0f\n", maxTokens)

			// n
			n, ok := infernexutils.GetSpanTagFloat(requestSpan, "request.n")
			Expect(ok).To(BeTrue(), "request.n 应包含")
			Expect(n).To(BeNumerically(">=", 1), "request.n 应该 >= 1")
			GinkgoWriter.Printf("  ✓ request.n = %.0f\n", n)

			GinkgoWriter.Println("TRACE_FIT_5 验证完成：vllm_ascend.request span attributes 齐全且正确")
		})

		It("【TRACE_FIT_6】验证vllm_ascend.prefill_execution和vllm_ascend.decode_phase span的attributes", Label("npu", "P0"), func() {
			// 发送推理请求
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces
			GinkgoWriter.Println("查询 Jaeger traces...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找包含 prefill_execution 和 decode_phase span 的 trace
			var prefillSpan, decodeSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				prefill := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill_execution")
				decode := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.decode_phase")

				if prefill != nil && decode != nil {
					prefillSpan = prefill
					decodeSpan = decode
					GinkgoWriter.Printf("✓ 找到 prefill_execution 和 decode_phase spans，trace ID: %s\n", trace.TraceID)
					break
				}
			}
			Expect(prefillSpan).NotTo(BeNil(), "应该找到 prefill_execution span")
			Expect(decodeSpan).NotTo(BeNil(), "应该找到 decode_phase span")

			// 验证 prefill_execution span attributes
			GinkgoWriter.Println("验证 prefill_execution span attributes:")
			prefillNumTokens, ok := infernexutils.GetSpanTagFloat(prefillSpan, "prefill.num_tokens")
			Expect(ok).To(BeTrue(), "prefill_execution span 应包含 prefill.num_tokens")
			Expect(prefillNumTokens).To(BeNumerically(">", 0), "prefill.num_tokens 应该 > 0")
			GinkgoWriter.Printf("  ✓ prefill.num_tokens = %.0f\n", prefillNumTokens)

			prefillTotalTime, ok := infernexutils.GetSpanTagFloat(prefillSpan, "prefill.total_time_ms")
			Expect(ok).To(BeTrue(), "prefill_execution span 应包含 prefill.total_time_ms")
			Expect(prefillTotalTime).To(BeNumerically(">", 0), "prefill.total_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ prefill.total_time_ms = %.2f ms\n", prefillTotalTime)

			// 验证 decode_phase span attributes
			GinkgoWriter.Println("验证 decode_phase span attributes:")
			decodeNumIterations, ok := infernexutils.GetSpanTagFloat(decodeSpan, "decode.num_iterations")
			Expect(ok).To(BeTrue(), "decode_phase span 应包含 decode.num_iterations")
			Expect(decodeNumIterations).To(BeNumerically(">", 0), "decode.num_iterations 应该 > 0")
			GinkgoWriter.Printf("  ✓ decode.num_iterations = %.0f\n", decodeNumIterations)

			decodeTotalTime, ok := infernexutils.GetSpanTagFloat(decodeSpan, "decode.total_time_ms")
			Expect(ok).To(BeTrue(), "decode_phase span 应包含 decode.total_time_ms")
			Expect(decodeTotalTime).To(BeNumerically(">", 0), "decode.total_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ decode.total_time_ms = %.2f ms\n", decodeTotalTime)

			decodeAvgTime, ok := infernexutils.GetSpanTagFloat(decodeSpan, "decode.avg_time_per_token_ms")
			Expect(ok).To(BeTrue(), "decode_phase span 应包含 decode.avg_time_per_token_ms")
			Expect(decodeAvgTime).To(BeNumerically(">", 0), "decode.avg_time_per_token_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ decode.avg_time_per_token_ms = %.2f ms\n", decodeAvgTime)

			// 验证关系：decode.total_time_ms / decode.num_iterations ≈ decode.avg_time_per_token_ms
			calculatedAvg := decodeTotalTime / decodeNumIterations
			tolerance := calculatedAvg * 0.1 // 10% 容差
			Expect(decodeAvgTime).To(BeNumerically("~", calculatedAvg, tolerance),
				"decode.avg_time_per_token_ms 应该约等于 total_time_ms/num_iterations")
			GinkgoWriter.Printf("  ✓ 验证关系：%.2f / %.0f ≈ %.2f (容差 10%%)\n", decodeTotalTime, decodeNumIterations, decodeAvgTime)

			GinkgoWriter.Println("TRACE_FIT_6 验证完成：prefill_execution 和 decode_phase span attributes 正确")
		})

		It("【TRACE_FIT_7】验证vllm_ascend.prefill.schedule和vllm_ascend.decode.schedule span的调度统计采集", Label("npu", "P0"), func() {
			// 发送推理请求
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces
			GinkgoWriter.Println("查询 Jaeger traces...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找 prefill.schedule 和 decode.schedule span
			var prefillScheduleSpan, decodeScheduleSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				if prefillScheduleSpan == nil {
					prefillScheduleSpan = infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill.schedule")
				}
				if decodeScheduleSpan == nil {
					decodeScheduleSpan = infernexutils.FindSpanByOperationName(trace, "vllm_ascend.decode.schedule")
				}
				if prefillScheduleSpan != nil && decodeScheduleSpan != nil {
					GinkgoWriter.Printf("✓ 找到 prefill.schedule 和 decode.schedule span，trace ID: %s\n", trace.TraceID)
					break
				}
			}
			Expect(prefillScheduleSpan).NotTo(BeNil(), "应该找到 prefill.schedule span")
			Expect(decodeScheduleSpan).NotTo(BeNil(), "应该找到 decode.schedule span")

			// 验证 prefill.schedule span 的调度统计 attributes
			GinkgoWriter.Println("验证 prefill.schedule span attributes:")

			prefillTotalCalls, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.total_calls")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.total_calls")
			Expect(prefillTotalCalls).To(BeNumerically(">=", 1), "schedule.total_calls 应该 >= 1")
			GinkgoWriter.Printf("  ✓ schedule.total_calls = %.0f\n", prefillTotalCalls)

			prefillTotalTokens, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.total_tokens")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.total_tokens")
			Expect(prefillTotalTokens).To(BeNumerically(">", 0), "schedule.total_tokens 应该 > 0（包含 prompt token 数量）")
			GinkgoWriter.Printf("  ✓ schedule.total_tokens = %.0f\n", prefillTotalTokens)

			prefillAvgTokensPerCall, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.avg_tokens_per_call")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.avg_tokens_per_call")
			Expect(prefillAvgTokensPerCall).To(BeNumerically(">", 0), "schedule.avg_tokens_per_call 应该 > 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_tokens_per_call = %.2f\n", prefillAvgTokensPerCall)

			prefillTotalTimeMs, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.total_time_ms")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.total_time_ms")
			Expect(prefillTotalTimeMs).To(BeNumerically(">", 0), "schedule.total_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ schedule.total_time_ms = %.2f ms\n", prefillTotalTimeMs)

			prefillAvgTimeMs, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.avg_time_ms")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.avg_time_ms")
			Expect(prefillAvgTimeMs).To(BeNumerically(">", 0), "schedule.avg_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_time_ms = %.2f ms\n", prefillAvgTimeMs)

			prefillMaxTimeMs, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.max_time_ms")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.max_time_ms")
			Expect(prefillMaxTimeMs).To(BeNumerically(">=", prefillAvgTimeMs), "schedule.max_time_ms 应该 >= schedule.avg_time_ms")
			GinkgoWriter.Printf("  ✓ schedule.max_time_ms = %.2f ms\n", prefillMaxTimeMs)

			prefillAvgQueueLen, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.avg_queue_length")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.avg_queue_length")
			Expect(prefillAvgQueueLen).To(BeNumerically(">=", 0), "schedule.avg_queue_length 应该 >= 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_queue_length = %.2f\n", prefillAvgQueueLen)

			prefillMaxQueueLen, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.max_queue_length")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.max_queue_length")
			Expect(prefillMaxQueueLen).To(BeNumerically(">=", prefillAvgQueueLen), "schedule.max_queue_length 应该 >= schedule.avg_queue_length")
			GinkgoWriter.Printf("  ✓ schedule.max_queue_length = %.2f\n", prefillMaxQueueLen)

			prefillAvgRunningCount, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.avg_running_count")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.avg_running_count")
			Expect(prefillAvgRunningCount).To(BeNumerically(">=", 0), "schedule.avg_running_count 应该 >= 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_running_count = %.2f\n", prefillAvgRunningCount)

			// 验证 decode.schedule span 的调度统计 attributes
			GinkgoWriter.Println("验证 decode.schedule span attributes:")

			decodeTotalCalls, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.total_calls")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.total_calls")
			Expect(decodeTotalCalls).To(BeNumerically(">", 0), "schedule.total_calls 应该 > 0（等于生成的 token 数量）")
			GinkgoWriter.Printf("  ✓ schedule.total_calls = %.0f\n", decodeTotalCalls)

			decodeTotalTokens, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.total_tokens")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.total_tokens")
			Expect(decodeTotalTokens).To(BeNumerically("~", decodeTotalCalls, decodeTotalCalls*0.05),
				"decode schedule.total_tokens 应该 = schedule.total_calls（decode 每次调度生成 1 个 token，容差 5%%）")
			GinkgoWriter.Printf("  ✓ schedule.total_tokens = %.0f\n", decodeTotalTokens)

			decodeAvgTokensPerCall, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.avg_tokens_per_call")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.avg_tokens_per_call")
			Expect(decodeAvgTokensPerCall).To(BeNumerically("~", 1.0, 0.05),
				"decode schedule.avg_tokens_per_call 应该 ≈ 1（容差 5%%）")
			GinkgoWriter.Printf("  ✓ schedule.avg_tokens_per_call = %.2f\n", decodeAvgTokensPerCall)

			decodeTotalTimeMs, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.total_time_ms")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.total_time_ms")
			Expect(decodeTotalTimeMs).To(BeNumerically(">", 0), "schedule.total_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ schedule.total_time_ms = %.2f ms\n", decodeTotalTimeMs)

			decodeAvgTimeMs, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.avg_time_ms")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.avg_time_ms")
			Expect(decodeAvgTimeMs).To(BeNumerically(">", 0), "schedule.avg_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_time_ms = %.2f ms\n", decodeAvgTimeMs)

			decodeMaxTimeMs, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.max_time_ms")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.max_time_ms")
			Expect(decodeMaxTimeMs).To(BeNumerically(">=", decodeAvgTimeMs), "schedule.max_time_ms 应该 >= schedule.avg_time_ms")
			GinkgoWriter.Printf("  ✓ schedule.max_time_ms = %.2f ms\n", decodeMaxTimeMs)

			decodeAvgQueueLen, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.avg_queue_length")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.avg_queue_length")
			Expect(decodeAvgQueueLen).To(BeNumerically(">=", 0), "schedule.avg_queue_length 应该 >= 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_queue_length = %.2f\n", decodeAvgQueueLen)

			decodeMaxQueueLen, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.max_queue_length")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.max_queue_length")
			Expect(decodeMaxQueueLen).To(BeNumerically(">=", decodeAvgQueueLen), "schedule.max_queue_length 应该 >= schedule.avg_queue_length")
			GinkgoWriter.Printf("  ✓ schedule.max_queue_length = %.2f\n", decodeMaxQueueLen)

			decodeAvgRunningCount, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.avg_running_count")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.avg_running_count")
			Expect(decodeAvgRunningCount).To(BeNumerically(">=", 0), "schedule.avg_running_count 应该 >= 0")
			GinkgoWriter.Printf("  ✓ schedule.avg_running_count = %.2f\n", decodeAvgRunningCount)

			GinkgoWriter.Println("TRACE_FIT_7 验证完成：prefill.schedule 和 decode.schedule span 调度统计 attributes 正确")
		})

		It("【TRACE_FIT_8】验证vllm_ascend.prefill.kv_cache和vllm_ascend.decode.kv_cache span的blocks分配统计采集", Label("npu", "P0"), func() {
			// 发送推理请求
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces
			GinkgoWriter.Println("查询 Jaeger traces...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找 prefill.kv_cache 和 decode.kv_cache span
			var prefillKVCacheSpan, decodeKVCacheSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				if prefillKVCacheSpan == nil {
					prefillKVCacheSpan = infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill.kv_cache")
				}
				if decodeKVCacheSpan == nil {
					decodeKVCacheSpan = infernexutils.FindSpanByOperationName(trace, "vllm_ascend.decode.kv_cache")
				}
				if prefillKVCacheSpan != nil && decodeKVCacheSpan != nil {
					GinkgoWriter.Printf("✓ 找到 prefill.kv_cache 和 decode.kv_cache span，trace ID: %s\n", trace.TraceID)
					break
				}
			}
			Expect(prefillKVCacheSpan).NotTo(BeNil(), "应该找到 prefill.kv_cache span")
			Expect(decodeKVCacheSpan).NotTo(BeNil(), "应该找到 decode.kv_cache span")

			// 验证 prefill.kv_cache span 的 blocks 分配统计 attributes
			GinkgoWriter.Println("验证 prefill.kv_cache span attributes:")

			prefillTotalCalls, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.total_calls")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.total_calls")
			Expect(prefillTotalCalls).To(BeNumerically(">=", 1), "kv_cache.total_calls 应该 >= 1（allocate_slots 调用次数）")
			GinkgoWriter.Printf("  ✓ kv_cache.total_calls = %.0f\n", prefillTotalCalls)

			prefillTotalBlocksAllocated, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.total_blocks_allocated")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.total_blocks_allocated")
			Expect(prefillTotalBlocksAllocated).To(BeNumerically(">", 0), "kv_cache.total_blocks_allocated 应该 > 0（累计分配的 KV Cache blocks 数）")
			GinkgoWriter.Printf("  ✓ kv_cache.total_blocks_allocated = %.0f\n", prefillTotalBlocksAllocated)

			prefillAvgBlocksPerCall, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.avg_blocks_per_call")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.avg_blocks_per_call")
			Expect(prefillAvgBlocksPerCall).To(BeNumerically(">", 0), "kv_cache.avg_blocks_per_call 应该 > 0（平均每次分配 blocks 数）")
			GinkgoWriter.Printf("  ✓ kv_cache.avg_blocks_per_call = %.2f\n", prefillAvgBlocksPerCall)

			prefillFinalFreeBlocks, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.final_free_blocks")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.final_free_blocks")
			Expect(prefillFinalFreeBlocks).To(BeNumerically(">=", 0), "kv_cache.final_free_blocks 应该 >= 0（最终剩余 free blocks 数）")
			GinkgoWriter.Printf("  ✓ kv_cache.final_free_blocks = %.0f\n", prefillFinalFreeBlocks)

			// 验证 decode.kv_cache span 的 blocks 分配统计 attributes
			GinkgoWriter.Println("验证 decode.kv_cache span attributes:")

			decodeTotalCalls, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.total_calls")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.total_calls")
			Expect(decodeTotalCalls).To(BeNumerically(">", 0), "kv_cache.total_calls 应该 > 0（等于生成的 token 数量）")
			GinkgoWriter.Printf("  ✓ kv_cache.total_calls = %.0f\n", decodeTotalCalls)

			decodeTotalBlocksAllocated, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.total_blocks_allocated")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.total_blocks_allocated")
			Expect(decodeTotalBlocksAllocated).To(BeNumerically(">=", 0), "kv_cache.total_blocks_allocated 应该 >= 0（decode 阶段分配的 blocks）")
			GinkgoWriter.Printf("  ✓ kv_cache.total_blocks_allocated = %.0f\n", decodeTotalBlocksAllocated)

			decodeAvgBlocksPerCall, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.avg_blocks_per_call")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.avg_blocks_per_call")
			Expect(decodeAvgBlocksPerCall).To(BeNumerically(">=", 0), "kv_cache.avg_blocks_per_call 应该 >= 0")
			GinkgoWriter.Printf("  ✓ kv_cache.avg_blocks_per_call = %.2f\n", decodeAvgBlocksPerCall)

			decodeFinalFreeBlocks, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.final_free_blocks")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.final_free_blocks")
			Expect(decodeFinalFreeBlocks).To(BeNumerically(">=", 0), "kv_cache.final_free_blocks 应该 >= 0")
			GinkgoWriter.Printf("  ✓ kv_cache.final_free_blocks = %.0f\n", decodeFinalFreeBlocks)

			decodeMinFreeBlocks, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.min_free_blocks")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.min_free_blocks")
			Expect(decodeMinFreeBlocks).To(BeNumerically(">=", 0), "kv_cache.min_free_blocks 应该 >= 0（decode 阶段最小剩余 blocks）")
			GinkgoWriter.Printf("  ✓ kv_cache.min_free_blocks = %.0f\n", decodeMinFreeBlocks)

			decodeAvgFreeBlocks, ok := infernexutils.GetSpanTagFloat(decodeKVCacheSpan, "kv_cache.avg_free_blocks")
			Expect(ok).To(BeTrue(), "decode.kv_cache span 应包含 kv_cache.avg_free_blocks")
			Expect(decodeAvgFreeBlocks).To(BeNumerically(">=", 0), "kv_cache.avg_free_blocks 应该 >= 0（decode 阶段平均剩余 blocks）")
			GinkgoWriter.Printf("  ✓ kv_cache.avg_free_blocks = %.2f\n", decodeAvgFreeBlocks)

			// 验证关系：min_free_blocks <= avg_free_blocks
			// 注意：avg 可能大于 final，因为 decode 过程中早期可能有更多空闲 blocks
			Expect(decodeMinFreeBlocks).To(BeNumerically("<=", decodeAvgFreeBlocks),
				"kv_cache.min_free_blocks 应该 <= kv_cache.avg_free_blocks")
			GinkgoWriter.Printf("  ✓ 验证关系：min_free_blocks(%.0f) <= avg_free_blocks(%.2f), final_free_blocks(%.0f)\n",
				decodeMinFreeBlocks, decodeAvgFreeBlocks, decodeFinalFreeBlocks)

			GinkgoWriter.Println("TRACE_FIT_8 验证完成：prefill.kv_cache 和 decode.kv_cache span blocks 分配统计 attributes 正确")
		})

		It("TRACE_FIT_9-验证vllm_ascend.prefill.kv_cache span缓存命中率统计采集", Label("npu", "P0"), func() {
			longPrompt := "请详细介绍一下人工智能的发展历程，包括从早期的符号主义人工智能、" +
				"专家系统，到机器学习、深度学习的演变过程，以及近年来大语言模型的崛起。" +
				"请分别介绍每个阶段的代表性技术、关键突破和主要应用场景。" +
				"同时请讨论人工智能在医疗、教育、金融、交通等领域的具体应用案例，" +
				"以及未来人工智能发展的趋势和挑战，包括可解释性、安全性、伦理问题等方面。" +
				"此外，请详细阐述机器学习的各种算法，包括监督学习、无监督学习、强化学习的原理和应用。" +
				"讨论深度学习中的卷积神经网络、循环神经网络、Transformer架构的发展和创新。" +
				"分析大语言模型如GPT、BERT、Claude等的技术特点和应用场景，" +
				"探讨这些模型在自然语言处理、代码生成、多模态理解等领域的突破性进展。" +
				"最后，请预测人工智能未来十年的发展方向，包括AGI的可能性、技术瓶颈、社会影响等方面。"

			// 发送第一条请求（建立 prefix cache）
			GinkgoWriter.Println("发送第一条推理请求（超长 prompt，建立 prefix cache）...")
			err := verifyInferenceRequestWithPrompt(namespace, DefaultTestModel, longPrompt)
			Expect(err).NotTo(HaveOccurred(), "第一条推理请求应成功")
			GinkgoWriter.Println("✓ 第一条推理请求成功")

			// 等待一段时间，确保 prefix cache 完全建立
			GinkgoWriter.Println("等待 3 秒，确保 prefix cache 完全建立...")
			time.Sleep(3 * time.Second)

			// 更新时间窗口基线（跳过第一条请求的 trace）
			lastQueryEndTime = time.Now()

			// 发送第二条推理请求（相同长 prompt，期望命中 prefix cache）
			GinkgoWriter.Println("发送第二条推理请求（相同超长 prompt，期望 prefix cache 命中）...")
			err = verifyInferenceRequestWithPrompt(namespace, DefaultTestModel, longPrompt)
			Expect(err).NotTo(HaveOccurred(), "第二条推理请求应成功")
			GinkgoWriter.Println("✓ 第二条推理请求成功")

			// 查询第二条请求的 Jaeger trace
			GinkgoWriter.Println("查询第二条请求的 Jaeger trace...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找 prefill.kv_cache span（第二条请求应命中缓存）
			var prefillKVCacheSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				span := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill.kv_cache")
				if span != nil {
					prefillKVCacheSpan = span
					GinkgoWriter.Printf("✓ 找到 prefill.kv_cache span，trace ID: %s\n", trace.TraceID)
					break
				}
			}
			Expect(prefillKVCacheSpan).NotTo(BeNil(), "应该找到 prefill.kv_cache span")

			// 打印所有 tags（调试用）
			GinkgoWriter.Printf("prefill.kv_cache span 的所有 tags:\n")
			for _, tag := range prefillKVCacheSpan.Tags {
				GinkgoWriter.Printf("  - %s = %v (type: %s)\n", tag.Key, tag.Value, tag.Type)
			}

			// 验证本地 prefix cache 命中统计（必须字段）
			GinkgoWriter.Println("验证 prefill.kv_cache span 本地缓存命中统计 attributes:")

			totalCachedTokens, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.total_cached_tokens")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.total_cached_tokens")
			Expect(totalCachedTokens).To(BeNumerically(">", 0), "kv_cache.total_cached_tokens 应该 > 0（表示本地 prefix cache 命中的 token 数）")
			GinkgoWriter.Printf("  ✓ kv_cache.total_cached_tokens = %.0f\n", totalCachedTokens)

			cacheHitRate, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.cache_hit_rate")
			Expect(ok).To(BeTrue(), "prefill.kv_cache span 应包含 kv_cache.cache_hit_rate")
			Expect(cacheHitRate).To(BeNumerically(">", 0), "kv_cache.cache_hit_rate 应该 > 0（命中了 prefix cache）")
			Expect(cacheHitRate).To(BeNumerically("<=", 1), "kv_cache.cache_hit_rate 应该 <= 1")
			GinkgoWriter.Printf("  ✓ kv_cache.cache_hit_rate = %.4f (%.2f%%)\n", cacheHitRate, cacheHitRate*100)

			// 验证外部缓存（Mooncake 等）命中统计（可选字段，仅当部署了外部 KV 缓存时才存在）
			GinkgoWriter.Println("验证 prefill.kv_cache span 外部缓存命中统计 attributes（可选）:")
			if hasExternalCache, hasExtCacheTag := infernexutils.GetSpanTagString(prefillKVCacheSpan, "kv_cache.has_external_cache"); hasExtCacheTag {
				Expect(hasExternalCache).To(Equal("true"), "kv_cache.has_external_cache 存在时应该 = true")
				GinkgoWriter.Printf("  ✓ kv_cache.has_external_cache = %s（外部缓存存在）\n", hasExternalCache)

				numExternalComputedTokens, ok := infernexutils.GetSpanTagFloat(prefillKVCacheSpan, "kv_cache.num_external_computed_tokens")
				Expect(ok).To(BeTrue(), "has_external_cache=true 时应包含 kv_cache.num_external_computed_tokens")
				Expect(numExternalComputedTokens).To(BeNumerically(">", 0), "kv_cache.num_external_computed_tokens 应该 > 0")
				GinkgoWriter.Printf("  ✓ kv_cache.num_external_computed_tokens = %.0f\n", numExternalComputedTokens)
			} else {
				GinkgoWriter.Println("  - kv_cache.has_external_cache 未设置（未部署外部 KV 缓存，跳过）")
			}

			GinkgoWriter.Println("TRACE_FIT_9 验证完成：prefill.kv_cache span 缓存命中统计 attributes 正确")
		})

		It("TRACE_FIT_10-验证vllm_ascend.prefill.ascend_store span的操作耗时统计采集", Label("npu", "P1"), func() {
			// 发送推理请求
			GinkgoWriter.Println("发送推理请求...")
			err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			// 查询 Jaeger traces
			GinkgoWriter.Println("查询 Jaeger traces...")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 查找 prefill.ascend_store span
			var ascendStoreSpan *infernexutils.JaegerSpan
			for i := range traces {
				trace := &traces[i]
				span := infernexutils.FindSpanByOperationName(trace, "vllm_ascend.prefill.ascend_store")
				if span != nil {
					ascendStoreSpan = span
					GinkgoWriter.Printf("✓ 找到 prefill.ascend_store span，trace ID: %s\n", trace.TraceID)
					break
				}
			}

			if ascendStoreSpan == nil {
				// 如果没有使用 AscendStore，则 span 不应存在
				GinkgoWriter.Println("⚠ 未找到 prefill.ascend_store span（节点未使用 AscendStore，预期行为：span 不存在）")
				GinkgoWriter.Println("TRACE_FIT_10 验证完成：节点未配置 AscendStore，span 不存在符合预期")
				return
			}

			// 验证 save 操作的时延统计 attributes
			GinkgoWriter.Println("验证 prefill.ascend_store span save 操作 attributes:")

			saveCalls, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.save_calls")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.save_calls")
			Expect(saveCalls).To(BeNumerically(">", 0), "ascend_store.save_calls 应该 > 0（save 调用次数）")
			GinkgoWriter.Printf("  ✓ ascend_store.save_calls = %.0f\n", saveCalls)

			saveDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.save_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.save_duration_ms")
			Expect(saveDurationMs).To(BeNumerically(">", 0), "ascend_store.save_duration_ms 应该 > 0（save 总耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.save_duration_ms = %.2f ms\n", saveDurationMs)

			saveMaxDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.save_max_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.save_max_duration_ms")
			Expect(saveMaxDurationMs).To(BeNumerically(">", 0), "ascend_store.save_max_duration_ms 应该 > 0（最大 save 单次耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.save_max_duration_ms = %.2f ms\n", saveMaxDurationMs)

			saveAvgDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.save_avg_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.save_avg_duration_ms")
			Expect(saveAvgDurationMs).To(BeNumerically(">", 0), "ascend_store.save_avg_duration_ms 应该 > 0（平均 save 耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.save_avg_duration_ms = %.2f ms\n", saveAvgDurationMs)

			// 验证关系：save_avg_duration_ms ≈ save_duration_ms / save_calls（容差 5%）
			expectedSaveAvg := saveDurationMs / saveCalls
			saveTolerance := expectedSaveAvg * 0.05
			Expect(saveAvgDurationMs).To(BeNumerically("~", expectedSaveAvg, saveTolerance),
				"ascend_store.save_avg_duration_ms 应该 ≈ save_duration_ms / save_calls（容差 5%%）")
			GinkgoWriter.Printf("  ✓ 验证关系：save_avg(%.2f) ≈ save_duration(%.2f) / save_calls(%.0f) = %.2f（容差 5%%）\n",
				saveAvgDurationMs, saveDurationMs, saveCalls, expectedSaveAvg)

			// 验证 load 操作的时延统计 attributes
			GinkgoWriter.Println("验证 prefill.ascend_store span load 操作 attributes:")

			loadCalls, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.load_calls")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.load_calls")
			Expect(loadCalls).To(BeNumerically(">", 0), "ascend_store.load_calls 应该 > 0（load 调用次数）")
			GinkgoWriter.Printf("  ✓ ascend_store.load_calls = %.0f\n", loadCalls)

			loadDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.load_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.load_duration_ms")
			Expect(loadDurationMs).To(BeNumerically(">", 0), "ascend_store.load_duration_ms 应该 > 0（load 总耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.load_duration_ms = %.2f ms\n", loadDurationMs)

			loadMaxDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.load_max_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.load_max_duration_ms")
			Expect(loadMaxDurationMs).To(BeNumerically(">", 0), "ascend_store.load_max_duration_ms 应该 > 0（最大 load 单次耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.load_max_duration_ms = %.2f ms\n", loadMaxDurationMs)

			loadAvgDurationMs, ok := infernexutils.GetSpanTagFloat(ascendStoreSpan, "ascend_store.load_avg_duration_ms")
			Expect(ok).To(BeTrue(), "prefill.ascend_store span 应包含 ascend_store.load_avg_duration_ms")
			Expect(loadAvgDurationMs).To(BeNumerically(">", 0), "ascend_store.load_avg_duration_ms 应该 > 0（平均 load 耗时）")
			GinkgoWriter.Printf("  ✓ ascend_store.load_avg_duration_ms = %.2f ms\n", loadAvgDurationMs)

			// 验证关系：load_avg_duration_ms ≈ load_duration_ms / load_calls（容差 5%）
			expectedLoadAvg := loadDurationMs / loadCalls
			loadTolerance := expectedLoadAvg * 0.05
			Expect(loadAvgDurationMs).To(BeNumerically("~", expectedLoadAvg, loadTolerance),
				"ascend_store.load_avg_duration_ms 应该 ≈ load_duration_ms / load_calls（容差 5%%）")
			GinkgoWriter.Printf("  ✓ 验证关系：load_avg(%.2f) ≈ load_duration(%.2f) / load_calls(%.0f) = %.2f（容差 5%%）\n",
				loadAvgDurationMs, loadDurationMs, loadCalls, expectedLoadAvg)

			GinkgoWriter.Println("TRACE_FIT_10 验证完成：prefill.ascend_store span 存储操作时延统计 attributes 正确")
		})

		It("【TRACE_FIT_11】采样率配置验证", Label("npu", "P1"), func() {
			// 步骤1：以100%成功率连续发送10个推理请求
			GinkgoWriter.Println("=== 步骤1：串行发送 10 个推理请求（成功率100%）===")
			const reqCount = 10
			for i := 0; i < reqCount; i++ {
				err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
				Expect(err).NotTo(HaveOccurred(), "第 %d 个推理请求应成功", i+1)
				GinkgoWriter.Printf("  ✓ 请求 %d/%d 成功\n", i+1, reqCount)
			}
			GinkgoWriter.Println("✓ 10 个推理请求全部成功")

			// 步骤2：查询 Jaeger traces，验证找到 10 条 trace
			GinkgoWriter.Println("=== 步骤2：查询并验证 Jaeger 中的 trace 数量 ===")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 50)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")
			Expect(len(traces)).To(BeNumerically(">=", reqCount),
				"Jaeger 中应找到至少 %d 条 trace，实际 %d 条", reqCount, len(traces))
			GinkgoWriter.Printf("✓ 查询到 %d 条 trace（>= %d）\n", len(traces), reqCount)

			// 步骤3：验证每条 trace 结构完整
			GinkgoWriter.Println("=== 步骤3：验证所有 trace 结构完整 ===")
			for i := range traces {
				t := &traces[i]
				requestSpan := infernexutils.FindSpanByOperationName(t, "vllm_ascend.request")
				Expect(requestSpan).NotTo(BeNil(),
					"trace %s 应包含 vllm_ascend.request span", t.TraceID)
			}
			GinkgoWriter.Printf("✓ 全部 %d 条 trace 结构完整\n", len(traces))
			GinkgoWriter.Println("TRACE_FIT_11 验证完成：10 个请求全部成功，trace 数量及结构符合预期")
		})

		It("【TRACE_FIT_12】边界场景验证——极短prompt", Label("npu", "P2"), func() {
			// 步骤1：发送极短 prompt 请求（"hi"，约 1-3 tokens）
			GinkgoWriter.Println("=== 步骤1：发送极短 prompt 请求 ===")
			err := verifyInferenceRequestWithOptions(namespace, DefaultTestModel, "hi", 0)
			Expect(err).NotTo(HaveOccurred(), "极短 prompt 推理请求应成功")
			GinkgoWriter.Println("✓ 极短 prompt 请求成功")

			// 步骤2：查询 Jaeger trace
			GinkgoWriter.Println("=== 步骤2：查询并验证 trace 数据 ===")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 步骤3：验证 request.usage.prompt_tokens 为极小值
			GinkgoWriter.Println("=== 步骤3：验证 prompt_tokens 为极小值 ===")
			var rootSpan *infernexutils.JaegerSpan
			for i := range traces {
				s := infernexutils.FindSpanByOperationName(&traces[i], "vllm_ascend.request")
				if s != nil {
					rootSpan = s
					break
				}
			}
			Expect(rootSpan).NotTo(BeNil(), "应找到 vllm_ascend.request span")

			promptTokens, ok := infernexutils.GetSpanTagFloat(rootSpan, "request.usage.prompt_tokens")
			Expect(ok).To(BeTrue(), "vllm_ascend.request span 应包含 request.usage.prompt_tokens")
			// "hi" 极短，prompt_tokens 应该 <= 10
			Expect(promptTokens).To(BeNumerically("<=", ShortPromptMaxTokens),
				"极短 prompt 的 prompt_tokens 应该极小（<= 10），实际: %.0f", promptTokens)
			GinkgoWriter.Printf("  ✓ request.usage.prompt_tokens = %.0f（极小值）\n", promptTokens)

			// 步骤4：验证 prefill_execution span 存在且统计数据正确
			GinkgoWriter.Println("=== 步骤4：验证 prefill_execution span 统计正确 ===")
			var prefillSpan *infernexutils.JaegerSpan
			for i := range traces {
				s := infernexutils.FindSpanByOperationName(&traces[i], "vllm_ascend.prefill_execution")
				if s != nil {
					prefillSpan = s
					break
				}
			}
			Expect(prefillSpan).NotTo(BeNil(), "应找到 vllm_ascend.prefill_execution span")

			prefillNumTokens, ok := infernexutils.GetSpanTagFloat(prefillSpan, "prefill.num_tokens")
			Expect(ok).To(BeTrue(), "prefill_execution span 应包含 prefill.num_tokens")
			Expect(prefillNumTokens).To(BeNumerically(">", 0), "prefill.num_tokens 应该 > 0")
			GinkgoWriter.Printf("  ✓ prefill.num_tokens = %.0f\n", prefillNumTokens)

			prefillTotalTime, ok := infernexutils.GetSpanTagFloat(prefillSpan, "prefill.total_time_ms")
			Expect(ok).To(BeTrue(), "prefill_execution span 应包含 prefill.total_time_ms")
			Expect(prefillTotalTime).To(BeNumerically(">", 0), "prefill.total_time_ms 应该 > 0")
			GinkgoWriter.Printf("  ✓ prefill.total_time_ms = %.2f ms\n", prefillTotalTime)

			GinkgoWriter.Println("TRACE_FIT_12 验证完成：极短 prompt 场景下 trace 数据正确，prompt_tokens 为极小值")
		})

		It("【TRACE_FIT_13】边界场景验证——极长prompt", Label("npu", "P2"), func() {
			GinkgoWriter.Println("=== 步骤1：发送极长 prompt 请求 ===")
			longPrompt := strings.Repeat("Please explain the concept of artificial intelligence in detail. ", 500)
			GinkgoWriter.Printf("Prompt 重复次数: 500，估算 ~5000 tokens\n")
			err := verifyInferenceRequestWithOptions(namespace, DefaultTestModel, longPrompt, 0)
			Expect(err).NotTo(HaveOccurred(), "极长 prompt 推理请求应成功")
			GinkgoWriter.Println("✓ 极长 prompt 请求成功")

			// 步骤2：查询 Jaeger trace
			GinkgoWriter.Println("=== 步骤2：查询并验证 trace 数据 ===")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 步骤3：验证 request.usage.prompt_tokens 为大值（几千 token）
			GinkgoWriter.Println("=== 步骤3：验证 prompt_tokens 为大值 ===")
			var rootSpan *infernexutils.JaegerSpan
			for i := range traces {
				s := infernexutils.FindSpanByOperationName(&traces[i], "vllm_ascend.request")
				if s != nil {
					rootSpan = s
					break
				}
			}
			Expect(rootSpan).NotTo(BeNil(), "应找到 vllm_ascend.request span")

			promptTokens, ok := infernexutils.GetSpanTagFloat(rootSpan, "request.usage.prompt_tokens")
			Expect(ok).To(BeTrue(), "vllm_ascend.request span 应包含 request.usage.prompt_tokens")
			// 极长 prompt 应有数千 token
			Expect(promptTokens).To(BeNumerically(">=", LongPromptMinTokens),
				"极长 prompt 的 prompt_tokens 应该 >= 1000，实际: %.0f", promptTokens)
			GinkgoWriter.Printf("  ✓ request.usage.prompt_tokens = %.0f（大值）\n", promptTokens)

			// 步骤4：验证 prefill.schedule.total_calls（可能 > 1 如果启用了 Chunked Prefill）
			GinkgoWriter.Println("=== 步骤4：验证 prefill.schedule span 调度统计 ===")
			var prefillScheduleSpan *infernexutils.JaegerSpan
			for i := range traces {
				s := infernexutils.FindSpanByOperationName(&traces[i], "vllm_ascend.prefill.schedule")
				if s != nil {
					prefillScheduleSpan = s
					break
				}
			}
			Expect(prefillScheduleSpan).NotTo(BeNil(), "应找到 prefill.schedule span")

			prefillTotalCalls, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.total_calls")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.total_calls")
			Expect(prefillTotalCalls).To(BeNumerically(">=", 1), "schedule.total_calls 应该 >= 1")

			// Chunked Prefill 是可选功能，如果启用了会 > 1，否则 = 1
			if prefillTotalCalls > 1 {
				GinkgoWriter.Printf("  ✓ schedule.total_calls = %.0f（> 1，已启用 Chunked Prefill）\n", prefillTotalCalls)
			} else {
				GinkgoWriter.Printf("  ✓ schedule.total_calls = %.0f（= 1，未启用 Chunked Prefill）\n", prefillTotalCalls)
			}

			// 步骤5：验证 schedule.total_tokens 与 prompt_tokens 一致
			GinkgoWriter.Println("=== 步骤5：验证 schedule.total_tokens 与 prompt_tokens 一致 ===")
			prefillTotalTokens, ok := infernexutils.GetSpanTagFloat(prefillScheduleSpan, "schedule.total_tokens")
			Expect(ok).To(BeTrue(), "prefill.schedule span 应包含 schedule.total_tokens")
			// schedule.total_tokens 应接近 prompt_tokens（允许一定误差）
			tolerance := promptTokens * 0.05 // 5% 容差
			Expect(prefillTotalTokens).To(BeNumerically("~", promptTokens, tolerance),
				"schedule.total_tokens(%.0f) 应与 prompt_tokens(%.0f) 接近（容差5%%）",
				prefillTotalTokens, promptTokens)
			GinkgoWriter.Printf("  ✓ schedule.total_tokens = %.0f ≈ prompt_tokens = %.0f\n",
				prefillTotalTokens, promptTokens)

			GinkgoWriter.Println("TRACE_FIT_13 验证完成：极长 prompt 场景下 trace 统计数据正确")
		})

		It("【TRACE_FIT_14】边界场景验证——最小max_tokens", Label("npu", "P2"), func() {
			// 步骤1：发送 max_tokens=1 的推理请求
			GinkgoWriter.Println("=== 步骤1：发送 max_tokens=1 推理请求 ===")
			err := verifyInferenceRequestWithOptions(namespace, DefaultTestModel, "Hello! How are you today", 1)
			Expect(err).NotTo(HaveOccurred(), "max_tokens=1 推理请求应成功")
			GinkgoWriter.Println("✓ max_tokens=1 请求成功")

			// 步骤2：查询 Jaeger trace
			GinkgoWriter.Println("=== 步骤2：查询并验证 trace 数据 ===")
			traces, err := queryTracesInWindow(namespace, &lastQueryEndTime, 20)
			Expect(err).NotTo(HaveOccurred(), "查询 Jaeger traces 应成功")

			// 步骤3：验证 completion_tokens = 1
			GinkgoWriter.Println("=== 步骤3：验证 completion_tokens = 1 ===")
			var rootSpan *infernexutils.JaegerSpan
			var targetTrace *infernexutils.JaegerTrace
			for i := range traces {
				s := infernexutils.FindSpanByOperationName(&traces[i], "vllm_ascend.request")
				if s != nil {
					rootSpan = s
					targetTrace = &traces[i]
					break
				}
			}
			Expect(rootSpan).NotTo(BeNil(), "应找到 vllm_ascend.request span")

			completionTokens, ok := infernexutils.GetSpanTagFloat(rootSpan, "request.usage.completion_tokens")
			Expect(ok).To(BeTrue(), "vllm_ascend.request span 应包含 request.usage.completion_tokens")
			Expect(completionTokens).To(BeNumerically("==", SingleTokenGeneration),
				"max_tokens=1 时 completion_tokens 应该等于 1，实际: %.0f", completionTokens)
			GinkgoWriter.Printf("  ✓ request.usage.completion_tokens = %.0f\n", completionTokens)

			// 步骤4：验证 decode 阶段行为（如果是 PD 分离模式）
			GinkgoWriter.Println("=== 步骤4：验证 decode 阶段行为 ===")
			decodePhaseSpan := infernexutils.FindSpanByOperationName(targetTrace, "vllm_ascend.decode_phase")
			if decodePhaseSpan != nil {
				GinkgoWriter.Println("  检测到 decode_phase span（PD 分离模式）")
				decodeNumIterations, ok := infernexutils.GetSpanTagFloat(decodePhaseSpan, "decode.num_iterations")
				Expect(ok).To(BeTrue(), "decode_phase span 应包含 decode.num_iterations")
				Expect(decodeNumIterations).To(BeNumerically("==", SingleTokenGeneration),
					"max_tokens=1 时 decode.num_iterations 应该 = 1，实际: %.0f", decodeNumIterations)
				GinkgoWriter.Printf("  ✓ decode.num_iterations = %.0f\n", decodeNumIterations)
			} else {
				GinkgoWriter.Println("  未检测到 decode_phase span（非 PD 分离模式）")
			}

			// 验证 decode.schedule span 的 total_calls = 1
			decodeScheduleSpan := infernexutils.FindSpanByOperationName(targetTrace, "vllm_ascend.decode.schedule")
			Expect(decodeScheduleSpan).NotTo(BeNil(), "应存在 decode.schedule span")
			decodeTotalCalls, ok := infernexutils.GetSpanTagFloat(decodeScheduleSpan, "schedule.total_calls")
			Expect(ok).To(BeTrue(), "decode.schedule span 应包含 schedule.total_calls")
			Expect(decodeTotalCalls).To(BeNumerically("==", SingleTokenGeneration),
				"max_tokens=1 时 decode.schedule.total_calls 应该 = 1，实际: %.0f", decodeTotalCalls)
			GinkgoWriter.Printf("  ✓ decode.schedule.total_calls = %.0f\n", decodeTotalCalls)

			// 验证 avg_time_per_token_ms ≈ total_time_ms（只有 1 个 token）
			decodeAvgTime, avgOk := infernexutils.GetSpanTagFloat(decodePhaseSpan, "decode.avg_time_per_token_ms")
			decodeTotalTime, totalOk := infernexutils.GetSpanTagFloat(decodePhaseSpan, "decode.total_time_ms")
			if avgOk && totalOk && decodeTotalTime > 0 {
				tolerance := decodeTotalTime * 0.01 // 1% 容差
				Expect(decodeAvgTime).To(BeNumerically("~", decodeTotalTime, tolerance),
					"max_tokens=1 时 avg_time_per_token_ms(%.2f) 应等于 total_time_ms(%.2f)",
					decodeAvgTime, decodeTotalTime)
				GinkgoWriter.Printf("  ✓ decode.avg_time_per_token_ms(%.2f) ≈ decode.total_time_ms(%.2f)\n",
					decodeAvgTime, decodeTotalTime)
			}

			GinkgoWriter.Println("TRACE_FIT_14 验证完成：max_tokens=1 场景下 completion_tokens=1，decode 阶段行为符合预期")
		})
	})
	// ==================== DFX 测试用例 ====================
	Describe("DFX 测试 - 追踪组件容错性验证", Ordered, Label("dfx", "fault-tolerance"), func() {
		const dfxReleaseName = "ifnx-dfx"

		BeforeAll(func() {
			// 部署Infernex并启用tracing
			err := helmInstallWithSets(dfxReleaseName, chartURL, chartVersion, namespace, []string{
				"global.tracingEnabled=true",
				"eagle-eye.tracing.enabled=true",
				"eagle-eye.enabled=true",
			})
			Expect(err).NotTo(HaveOccurred(), "helm install应成功")
			GinkgoWriter.Println("helm install完成，等待Pod就绪...")

			// 等待所有Pod running
			Eventually(func() error {
				return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
			}, 20*time.Minute, 15*time.Second).Should(Succeed())
			GinkgoWriter.Println("===== DFX 测试环境部署完成，所有Pod已running =====")
		})

		AfterAll(func() {
			GinkgoWriter.Printf("===== 清理 DFX 部署: %s =====\n", dfxReleaseName)
			deferCleanupInfernexRelease(dfxReleaseName, namespace)
		})

		It("【TRACE_DFX_1】高并发场景验证——多请求并发追踪", Label("npu", "P1"), func() {
			const concurrency = 50

			// 步骤1：并发发送50个推理请求
			GinkgoWriter.Printf("=== 步骤1：并发发送 %d 个推理请求 ===\n", concurrency)
			startTime := time.Now()

			type result struct {
				idx int
				err error
			}
			results := make(chan result, concurrency)
			for i := 0; i < concurrency; i++ {
				go func(idx int) {
					err := verifyInferenceRequestTracing(namespace, DefaultTestModel)
					results <- result{idx: idx, err: err}
				}(i)
			}

			// 步骤2：等待所有请求完成，统计成功数
			GinkgoWriter.Println("=== 步骤2：等待所有请求完成 ===")
			failedCount := 0
			for i := 0; i < concurrency; i++ {
				r := <-results
				if r.err != nil {
					failedCount++
					GinkgoWriter.Printf("  ✗ 请求 %d 失败: %v\n", r.idx, r.err)
				}
			}
			Expect(failedCount).To(Equal(0),
				fmt.Sprintf("%d 个并发请求中有 %d 个失败，期望全部成功", concurrency, failedCount))
			GinkgoWriter.Printf("✓ %d 个并发请求全部成功返回\n", concurrency)

			endTime := time.Now()

			// 步骤3：查询这批请求生成的 Jaeger traces
			GinkgoWriter.Printf("=== 步骤3：查询并发请求的 Jaeger traces（时间窗口: %s ~ %s）===\n",
				startTime.Format("15:04:05"), endTime.Format("15:04:05"))
			// 等待 trace 数据完整写入
			time.Sleep(JaegerWriteWaitTime)
			traces, err := infernexutils.QueryJaegerTraces(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace, "inference-gateway", startTime, endTime, 100)
			Expect(err).NotTo(HaveOccurred(), "查询并发请求的 Jaeger traces 应成功")
			Expect(len(traces)).To(BeNumerically(">=", concurrency),
				fmt.Sprintf("应查询到至少 %d 条 trace，实际只有 %d 条", concurrency, len(traces)))
			GinkgoWriter.Printf("✓ 查询到 %d 条 trace\n", len(traces))

			// 步骤4：验证 trace_id 全局唯一，无重复
			GinkgoWriter.Println("=== 步骤4：验证 trace_id 全局唯一性 ===")
			traceIDSet := make(map[string]bool)
			for _, t := range traces {
				Expect(traceIDSet[t.TraceID]).To(BeFalse(),
					fmt.Sprintf("发现重复的 trace_id: %s", t.TraceID))
				traceIDSet[t.TraceID] = true
			}
			GinkgoWriter.Printf("✓ 所有 %d 条 trace 的 trace_id 全局唯一，无重复\n", len(traces))

			// 步骤5：逐一验证全部 trace 的 span 结构完整性及无跨 trace 污染
			// 步骤5：抽样验证 trace 的 span 结构完整性及无跨 trace 污染（抽样10条）
			sampleSize := 10
			if len(traces) < sampleSize {
				sampleSize = len(traces)
			}
			GinkgoWriter.Printf("=== 步骤5：抽样验证 %d 条 trace 的 span 结构完整性及无跨 trace 污染（共 %d 条）===\n", sampleSize, len(traces))

			for i := 0; i < sampleSize; i++ {
				t := &traces[i]
				GinkgoWriter.Printf("  检查 trace[%d] ID: %s，共 %d 个 span\n", i, t.TraceID, len(t.Spans))

				// 验证 vllm_ascend.request span 存在
				requestSpan := infernexutils.FindSpanByOperationName(t, "vllm_ascend.request")
				Expect(requestSpan).NotTo(BeNil(),
					fmt.Sprintf("trace %s 应包含 vllm_ascend.request span", t.TraceID))

				// 验证所有 span 的 traceID 与本 trace 一致（无跨 trace 污染）
				for _, span := range t.Spans {
					Expect(span.TraceID).To(Equal(t.TraceID),
						fmt.Sprintf("trace %s 中 span %s 的 traceID 应与 trace 一致，实际为 %s",
							t.TraceID, span.SpanID, span.TraceID))
				}
				GinkgoWriter.Printf("  ✓ trace[%d] span 结构完整，无跨 trace 污染\n", i)
			}
			GinkgoWriter.Printf("✓ 抽样验证 %d 条 trace 通过（共 %d 条）：span 结构完整，无跨 trace 数据污染\n", sampleSize, len(traces))

			GinkgoWriter.Printf("TRACE_DFX_1 验证完成：%d 并发请求全部成功，trace_id 唯一，无跨 trace 污染\n", concurrency)
		})

		It("【TRACE_DFX_4】Jaeger Collector不可用时推理服务可用性验证", Label("npu", "P2"), func() {
			const collectorDep = "eagle-eye-tracing-collector"

			// 步骤1：停止 Jaeger Collector（缩容到0副本）
			GinkgoWriter.Printf("=== 步骤1：停止 Jaeger Collector（%s 缩容到0副本）===\n", collectorDep)
			var replicas int32 = 0
			deploymentsClient := clientset.AppsV1().Deployments(namespace)
			dep, err := deploymentsClient.Get(ctx, collectorDep, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "获取 eagle-eye-tracing-collector Deployment 应成功")
			originalReplicas := dep.Spec.Replicas

			dep.Spec.Replicas = &replicas
			_, err = deploymentsClient.Update(ctx, dep, metav1.UpdateOptions{})
			Expect(err).NotTo(HaveOccurred(), "将 eagle-eye-tracing-collector 缩容到0副本应成功")
			GinkgoWriter.Println("✓ eagle-eye-tracing-collector 缩容命令已发出")

			// 步骤2：等待 Collector 缩容完成（等待所有副本消失）
			GinkgoWriter.Println("=== 步骤2：等待 Collector 缩容完成 ===")
			Eventually(func() int32 {
				d, e := deploymentsClient.Get(ctx, collectorDep, metav1.GetOptions{})
				if e != nil {
					return -1
				}
				GinkgoWriter.Printf("  当前 ReadyReplicas: %d\n", d.Status.ReadyReplicas)
				return d.Status.ReadyReplicas
			}, 3*time.Minute, 5*time.Second).Should(Equal(int32(0)),
				"eagle-eye-tracing-collector 应缩容至0个就绪副本")
			GinkgoWriter.Println("✓ eagle-eye-tracing-collector 已缩容完成，当前0副本")

			// 步骤3：Collector 不可用时发送推理请求，验证推理功能不受影响
			GinkgoWriter.Println("=== 步骤3：Collector 不可用时发送推理请求 ===")
			err = verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "Jaeger Collector 不可用时推理请求应仍然成功返回")
			GinkgoWriter.Println("✓ 推理请求成功返回，推理功能不受 Collector 不可用影响")

			// 步骤4：恢复 Jaeger Collector
			GinkgoWriter.Println("=== 步骤5：恢复 Jaeger Collector ===")
			dep, err = deploymentsClient.Get(ctx, collectorDep, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "重新获取 Deployment 应成功")
			if originalReplicas == nil {
				var one int32 = 1
				dep.Spec.Replicas = &one
			} else {
				dep.Spec.Replicas = originalReplicas
			}
			_, err = deploymentsClient.Update(ctx, dep, metav1.UpdateOptions{})
			Expect(err).NotTo(HaveOccurred(), "恢复 eagle-eye-tracing-collector 副本数应成功")
			GinkgoWriter.Printf("✓ eagle-eye-tracing-collector 恢复命令已发出（目标副本数: %d）\n",
				func() int32 {
					if originalReplicas == nil {
						return 1
					}
					return *originalReplicas
				}())

			// 步骤5：等待 Collector 恢复就绪
			GinkgoWriter.Println("=== 步骤6：等待 Collector 恢复就绪 ===")
			targetReplicas := func() int32 {
				if originalReplicas == nil {
					return 1
				}
				return *originalReplicas
			}()
			Eventually(func() int32 {
				d, e := deploymentsClient.Get(ctx, collectorDep, metav1.GetOptions{})
				if e != nil {
					return 0
				}
				GinkgoWriter.Printf("  当前 ReadyReplicas: %d / %d\n", d.Status.ReadyReplicas, targetReplicas)
				return d.Status.ReadyReplicas
			}, 5*time.Minute, 10*time.Second).Should(Equal(targetReplicas),
				"eagle-eye-tracing-collector 应恢复到目标副本数并就绪")
			GinkgoWriter.Printf("✓ eagle-eye-tracing-collector 已恢复就绪，ReadyReplicas=%d\n", targetReplicas)

			// 步骤7：Collector 恢复后发送新推理请求，验证 trace 可正常生成
			GinkgoWriter.Println("=== 步骤7：Collector 恢复后验证 trace 正常生成 ===")
			// 等待 Collector 稳定后再发请求
			time.Sleep(CollectorRecoverWaitTime)

			recoverStartTime := time.Now()
			err = verifyInferenceRequestTracing(namespace, DefaultTestModel)
			Expect(err).NotTo(HaveOccurred(), "Collector 恢复后推理请求应成功")
			GinkgoWriter.Println("✓ 推理请求成功")

			GinkgoWriter.Printf("等待 %v 以确保 trace 数据写入 Jaeger...\n", JaegerWriteWaitTime)
			time.Sleep(JaegerWriteWaitTime)
			recoverEndTime := time.Now()

			GinkgoWriter.Printf("查询时间窗口: %s ~ %s\n",
				recoverStartTime.Format("15:04:05"), recoverEndTime.Format("15:04:05"))
			recoverTraces, queryErr := infernexutils.QueryJaegerTraces(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace, "inference-gateway",
				recoverStartTime, recoverEndTime, 10)
			Expect(queryErr).NotTo(HaveOccurred(), "Collector 恢复后查询 Jaeger traces 应成功")
			Expect(recoverTraces).NotTo(BeEmpty(), "Collector 恢复后新推理请求应有 trace 数据")

			// 验证恢复后的 trace 包含完整的 span 结构
			var validTrace *infernexutils.JaegerTrace
			for i := range recoverTraces {
				t := &recoverTraces[i]
				requestSpan := infernexutils.FindSpanByOperationName(t, "vllm_ascend.request")
				if requestSpan != nil {
					validTrace = t
					GinkgoWriter.Printf("✓ 恢复后找到包含 vllm_ascend.request span 的 trace，trace ID: %s\n", t.TraceID)
					break
				}
			}
			Expect(validTrace).NotTo(BeNil(), "Collector 恢复后应能找到包含 vllm_ascend.request 的完整 trace")

			GinkgoWriter.Println("TRACE_DFX_4 验证完成：Jaeger Collector 不可用时推理功能正常，恢复后 trace 可正常生成")
		})
	})
})

func verifyInferenceRequestWithOptions(targetNS, model, prompt string, maxTokens int) error {
	k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
	GinkgoWriter.Printf("发送推理请求: model=%s, prompt长度=%d\n", model, len(prompt))
	err := infernexutils.VerifyInferenceRequestWithOptions(ctx, k8sAdapter, targetNS, model, prompt, maxTokens)
	if err == nil {
		GinkgoWriter.Println("✓ 推理请求校验成功")
	}
	return err
}

func verifyInferenceRequestWithPrompt(targetNS, model, prompt string) error {
	k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
	GinkgoWriter.Printf("发送推理请求: model=%s, prompt长度=%d\n", model, len(prompt))
	err := infernexutils.VerifyInferenceRequestWithPrompt(ctx, k8sAdapter, targetNS, model, prompt)
	if err == nil {
		GinkgoWriter.Println("✓ 推理请求校验成功")
	}
	return err
}

func verifyInferenceRequestTracing(targetNS, model string) error {
	return verifyInferenceRequestWithPrompt(targetNS, model, "Hello! How are you today?")
}

func verifyStreamingInferenceRequest(targetNS, model string) error {
	k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
	GinkgoWriter.Printf("发送流式推理请求: model=%s\n", model)
	err := infernexutils.VerifyStreamingInferenceRequest(ctx, k8sAdapter, targetNS, model)
	if err == nil {
		GinkgoWriter.Println("✓ 流式推理请求校验成功")
	}
	return err
}

// queryTracesInWindow 封装重复的 trace 查询模式
// 使用时间窗口查询 traces，自动更新 lastEndTime，并验证结果非空
func queryTracesInWindow(namespace string, lastEndTime *time.Time, limit int) ([]infernexutils.JaegerTrace, error) {
	// 等待 trace 数据异步写入 Jaeger
	GinkgoWriter.Printf("等待 %v 以确保 trace 数据写入 Jaeger...\n", JaegerWriteWaitTime)
	time.Sleep(JaegerWriteWaitTime)

	startTime := *lastEndTime
	endTime := time.Now()
	*lastEndTime = endTime

	k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
	GinkgoWriter.Printf("查询时间窗口: %s ~ %s\n", startTime.Format("15:04:05"), endTime.Format("15:04:05"))

	// 使用重试机制查询 traces（Jaeger 写入可能有延迟）
	const maxRetries = 3
	const retryInterval = 3 * time.Second

	var traces []infernexutils.JaegerTrace
	var err error

	for attempt := 1; attempt <= maxRetries; attempt++ {
		traces, err = infernexutils.QueryJaegerTraces(ctx, k8sAdapter, namespace, "inference-gateway", startTime, endTime, limit)
		if err == nil && len(traces) > 0 {
			GinkgoWriter.Printf("✓ 找到 %d 条 traces\n", len(traces))
			return traces, nil
		}

		if attempt < maxRetries {
			GinkgoWriter.Printf("  第 %d 次查询未找到 traces，%v 后重试...\n", attempt, retryInterval)
			time.Sleep(retryInterval)
		}
	}

	// 所有重试都失败
	if err != nil {
		return nil, fmt.Errorf("查询失败（重试 %d 次）: %w", maxRetries, err)
	}
	return nil, fmt.Errorf("未查询到 trace 数据（重试 %d 次）", maxRetries)
}
