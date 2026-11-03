package infernex

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	esutils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	infernexutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"gopkg.in/yaml.v3"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

var _ = Describe("Infernex helm deploy", Ordered, ContinueOnFailure, Label("infernex-deploy"), func() {
	var (
		chartURL     string
		chartVersion string
		releaseName  string
		namespace    string
	)

	BeforeAll(func() {
		cfg := InfernexHelmConfig
		chartURL = cfg.ChartURL
		chartVersion = cfg.Version
		releaseName = cfg.ReleaseName
		namespace = cfg.Namespace

		if strings.TrimSpace(chartURL) == "" {
			Skip("请在constants.go中设置InfernexHelmConfig.ChartURL")
		}
		GinkgoWriter.Printf("infernex deploy: chartURL=%q version=%q release=%q ns=%q\n",
			chartURL, chartVersion, releaseName, namespace)

		Expect(clientset).NotTo(BeNil())
		ensureDeploySuiteCleanSlate(releaseName, namespace)
	})

	It("InferNex_FIT_1-在线一键部署;InferNex_FIT_12-集成部署PD架构非流式推理;InferNex_FIT_13-集成部署PD架构流式推理;InferNex_FIT_21-推理后端使用mooncake multi connector", Label("npu", "P0"), func() {
		Expect(clientset).NotTo(BeNil())

		DeferCleanup(deferCleanupInfernexRelease, releaseName, namespace)

		err := helmInstall(releaseName, chartURL, chartVersion, namespace)
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}

		// 校验流式推理请求
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "", defaultModelName(), 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败")
		GinkgoWriter.Println("流式推理请求校验成功")

		// 校验非流式推理请求
		Eventually(func() error {
			return verifyNonStreamInferenceRequest(namespace, defaultModelName())
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "非流式推理请求校验失败")
		GinkgoWriter.Println("非流式推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_14-集成部署聚合架构流式推理", Label("npu", "P0"), func() {
		Expect(clientset).NotTo(BeNil())

		const fit14Release = "infernex-fit14"
		const aggregatedModel = "Qwen/Qwen3-8B"

		DeferCleanup(deferCleanupInfernexRelease, fit14Release, namespace)

		err := helmInstallRelease(fit14Release, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-vLLM-aggregated-random.yaml"), nil)
		Expect(err).NotTo(HaveOccurred(), "FIT_14: helm install（aggregated values）应成功")

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_14: 等待所有 Pod running 1/1 超时")
		GinkgoWriter.Println("FIT_14: 所有 Pod 已 running 1/1")

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace,
				"what's the result of 100+100?", aggregatedModel, 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "FIT_14: 流式推理请求校验失败")
		GinkgoWriter.Println("FIT_14: 流式推理请求校验成功")
	})

	It("InferNex_FIT_2-必选组件一键部署", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())

		minimalReleaseName := releaseName + "-minimal"

		DeferCleanup(deferCleanupInfernexRelease, minimalReleaseName, namespace)

		// 部署，禁用四个可选组件
		err := helmInstallWithSets(minimalReleaseName, chartURL, chartVersion, namespace, []string{
			"hermes-router.enabled=false",
			"inferenceGateway.enabled=false",
			"eagle-eye.enabled=false",
			"cache-indexer.enabled=false",
		})
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求（gateway 已禁用，回退到后端 Service）
		Eventually(func() error {
			return verifyInferenceRequest(namespace, defaultModelName())
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(minimalReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_3-集成部署一键删除", Label("npu", "P0"), func() {
		Expect(clientset).NotTo(BeNil())

		// FIT_3 自包含：独立 release/完整部署，不依赖 FIT_1 或其他用例
		const fit3ReleaseName = "infernex-fit3"
		fit3Namespace := namespace

		DeferCleanup(deferCleanupInfernexRelease, fit3ReleaseName, fit3Namespace)

		err := helmInstall(fit3ReleaseName, chartURL, chartVersion, fit3Namespace)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, fit3Namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_3: 等待 InferNex 部署就绪超时")
		GinkgoWriter.Println("FIT_3: InferNex 部署成功，开始 helm uninstall")

		err = helmUninstall(fit3ReleaseName, fit3Namespace)
		Expect(err).NotTo(HaveOccurred(), "FIT_3: helm uninstall 应成功")

		// uninstall 后由 chart/控制器自行回收资源；轮询直到满足 FIT_3 预期（RSG 扩缩容残留除外）
		Eventually(func() error {
			return verifyHelmUninstallResourceCleanup(fit3Namespace, fit3ReleaseName)
		}, 10*time.Minute, 15*time.Second).Should(Succeed(), "FIT_3: 卸载后资源校验失败")
		GinkgoWriter.Println("FIT_3: 卸载后各命名空间资源校验通过")
	})

	It("InferNex_FIT_4-集成部署命名空间可配置", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		testNS := "test-inference"
		testReleaseName := releaseName + "-testns"

		DeferCleanup(deferCleanupInfernexRelease, testReleaseName, testNS)

		// 部署到 test-inference 命名空间，自动创建 namespace
		err := helmInstallWithCreateNS(testReleaseName, chartURL, chartVersion, testNS)
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, testNS)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求
		Eventually(func() error {
			return verifyInferenceRequest(testNS, defaultModelName())
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(testReleaseName, testNS)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_5-在线一键部署Qwen3-32B（是否要切Qwen3.5）", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		qwenReleaseName := releaseName + "-qwen32b"

		// 部署，替换模型为 Qwen3-32B，设置 tp=4（通过 values 文件覆盖数组配置）
		DeferCleanup(deferCleanupInfernexRelease, qwenReleaseName, namespace)

		err := helmInstallRelease(qwenReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-32B-tp4.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1（32B 在线拉取/加载比 8B 更久，仅本用例放宽到 45 分钟）
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 45*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（45分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求（使用 Qwen3-32B 模型）
		Eventually(func() error {
			return verifyInferenceRequest(namespace, "Qwen/Qwen3-32B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 校验非流式推理请求（使用 Qwen3-32B 模型）
		Eventually(func() error {
			return verifyNonStreamInferenceRequest(namespace, "Qwen/Qwen3-32B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "非流式推理请求校验失败")
		GinkgoWriter.Println("非流式推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(qwenReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_9-在线一键部署vllm-ascend 0.19.1rc1版本（看版本变动）;InferNex_FIT_18-集成部署使用vllm-ascned 0.19.0rc1版本", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		vllmReleaseName := releaseName + "-vllm0191"

		// 部署，替换 vLLM 推理引擎镜像
		err := helmInstallWithSets(vllmReleaseName, chartURL, chartVersion, namespace, []string{
			"inference-backend.images.inferenceEngine.repository=quay.io/ascend/vllm-ascend",
			"inference-backend.images.inferenceEngine.tag=v0.19.1rc1",
		})
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, vllmReleaseName, namespace)

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求
		Eventually(func() error {
			return verifyInferenceRequest(namespace, defaultModelName())
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(vllmReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_17-集成部署使用vllm-ascned 0.13.0版本", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		vllmReleaseName := releaseName + "-vllm013"

		// 仅替换镜像 tag 会沿用 chart 默认 extraArgs（含 v0.18 才支持的参数），v0.13.0 无法启动。
		err := helmInstallRelease(vllmReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-vLLM-v0.13.0.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, vllmReleaseName, namespace)

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		Eventually(func() error { return ensureNamespaceActive("istio-system") }, 5*time.Minute, 5*time.Second).
			Should(Succeed(), "FIT_17: 等待 istio-system Active")
		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			_, err := infernexutils.ResolveGatewayURL(ctx, k8sAdapter, []string{namespace})
			return err
		}, 5*time.Minute, 10*time.Second).Should(Succeed(), "FIT_17: 解析推理网关失败")

		// CSV FIT_17：stream=true，至少一个分片并收到 done。走网关 NodePort，避免打到 backend ClusterIP 挂死。
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "what's the result of 100+100?", defaultModelName(), 0)
		}, 8*time.Minute, 20*time.Second).Should(Succeed(), "FIT_17: 流式推理请求校验失败")
		GinkgoWriter.Println("FIT_17: 流式推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(vllmReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_19-推理后端使用mooncake p2p connector", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		mooncakeReleaseName := releaseName + "-mooncake"

		// 部署，使用 Qwen3-8B 单独指定 MooncakeConnectorV1
		DeferCleanup(deferCleanupInfernexRelease, mooncakeReleaseName, namespace)

		err := helmInstallRelease(mooncakeReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-mooncake-connector.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求（使用 Qwen3-8B 模型）
		Eventually(func() error {
			return verifyInferenceRequest(namespace, "Qwen/Qwen3-8B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 校验非流式推理请求（使用 Qwen3-8B 模型）
		Eventually(func() error {
			return verifyNonStreamInferenceRequest(namespace, "Qwen/Qwen3-8B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "非流式推理请求校验失败")
		GinkgoWriter.Println("非流式推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(mooncakeReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_20-推理后端使用mooncake store connector", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		ascendStoreReleaseName := releaseName + "-ascendstore"

		// 部署，使用 Qwen3-8B 单独指定 AscendStoreConnector
		DeferCleanup(deferCleanupInfernexRelease, ascendStoreReleaseName, namespace)

		err := helmInstallRelease(ascendStoreReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-ascend-store-connector.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		// 等待所有 Pod running 1/1
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		// 校验推理请求（使用 Qwen3-8B 模型）
		Eventually(func() error {
			return verifyInferenceRequest(namespace, "Qwen/Qwen3-8B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "推理请求校验失败")
		GinkgoWriter.Println("推理请求校验成功")

		// 校验非流式推理请求（使用 Qwen3-8B 模型）
		Eventually(func() error {
			return verifyNonStreamInferenceRequest(namespace, "Qwen/Qwen3-8B")
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "非流式推理请求校验失败")
		GinkgoWriter.Println("非流式推理请求校验成功")

		// 检查 helm 安装了哪些资源
		resources, err := infernexutils.HelmGetManifest(ascendStoreReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_22-自定义prefill节点tp;InferNex_FIT_23-自定义prefill节点prefixCaching;InferNex_FIT_24-自定义prefill节点maxModellength;InferNex_FIT_25-自定义prefill节点maxNumBatchedTokens;InferNex_FIT_26-自定义prefill节点gpuMemoryUtilization", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		prefillReleaseName := releaseName + "-fit22"

		// FIT_22~26：使用外挂 yaml 配置
		err := helmInstallRelease(prefillReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-pd-vllm-config-params.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, prefillReleaseName, namespace)

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		Expect(restConfig).NotTo(BeNil())
		verifyPDEngineExtraArgs(namespace, infernexutils.PrefillPDRole, "InferNex_FIT_23", PDVllmConfigExtraArgMarkers)

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "what's the result of 100+100?", "", 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败（FIT_22~26）")
		GinkgoWriter.Println("流式推理请求校验成功（FIT_22~26）")

		resources, err := infernexutils.HelmGetManifest(prefillReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_27-自定义prefill多dp parallel配置（dp local rank 1）;InferNex_FIT_35-自定义decode多dp parallel配置（dp local rank 1）", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())
		Expect(dynamicClient).NotTo(BeNil())

		dpReleaseName := releaseName + "-dp-fit2735"

		Expect(infernexutils.EnsureLeaderWorkerSetAPI(clientset)).NotTo(HaveOccurred())

		// FIT_27/FIT_35：使用外挂 yaml 配置 DP 参数
		err := helmInstallRelease(dpReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-Coder-30B-A3B-Instruct-vllm-dp-1-local.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, dpReleaseName, namespace)

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		const expectedLWSSize int64 = 2 // dataParallelSize / dataParallelSizeLocal
		expectedRanks := []int{0, 1}

		Eventually(func() error {
			prefillLWS, err := infernexutils.FindLeaderWorkerSetNameByRole(dynamicClient, ctx, namespace, "prefill")
			if err != nil {
				return err
			}
			if err := infernexutils.VerifyLeaderWorkerSetTemplateSize(dynamicClient, ctx, namespace, prefillLWS, expectedLWSSize); err != nil {
				return fmt.Errorf("FIT_27 prefill LWS: %w", err)
			}
			return infernexutils.VerifyLeaderWorkerSetPodsDataParallelStartRanks(restConfig, clientset, ctx, namespace, prefillLWS, expectedRanks)
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "FIT_27 prefill LeaderWorkerSet 校验失败")
		GinkgoWriter.Println("FIT_27 prefill LeaderWorkerSet size 与 data-parallel-start-rank 校验成功")

		Eventually(func() error {
			decodeLWS, err := infernexutils.FindLeaderWorkerSetNameByRole(dynamicClient, ctx, namespace, "decode")
			if err != nil {
				return err
			}
			if err := infernexutils.VerifyLeaderWorkerSetTemplateSize(dynamicClient, ctx, namespace, decodeLWS, expectedLWSSize); err != nil {
				return fmt.Errorf("FIT_35 decode LWS: %w", err)
			}
			return infernexutils.VerifyLeaderWorkerSetPodsDataParallelStartRanks(restConfig, clientset, ctx, namespace, decodeLWS, expectedRanks)
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "FIT_35 decode LeaderWorkerSet 校验失败")
		GinkgoWriter.Println("FIT_35 decode LeaderWorkerSet size 与 data-parallel-start-rank 校验成功")

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		const moeModel = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "what's the result of 100+100?", moeModel, 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败（FIT_27/FIT_35）")
		GinkgoWriter.Println("流式推理请求校验成功（FIT_27/FIT_35）")

		resources, err := infernexutils.HelmGetManifest(dpReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
	})

	It("InferNex_FIT_28-自定义prefill多dp parallel配置（multiple dp local rank ）;InferNex_FIT_36-自定义decode多dp parallel配置（multiple dp local rank ）", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())
		Expect(dynamicClient).NotTo(BeNil())

		dpReleaseName := releaseName + "-dp-fit2836"

		Expect(infernexutils.EnsureLeaderWorkerSetAPI(clientset)).NotTo(HaveOccurred())

		// MoE 模型 PD 部署 + FIT_28/FIT_36：dataParallelSize=4, dataParallelSizeLocal=2
		err := helmInstallRelease(dpReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-Coder-30B-A3B-Instruct-vllm-dp-multi-local.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, dpReleaseName, namespace)

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		const expectedLWSSize int64 = 2 // 4 / 2
		expectedRanks := []int{0, 2}

		Eventually(func() error {
			prefillLWS, err := infernexutils.FindLeaderWorkerSetNameByRole(dynamicClient, ctx, namespace, "prefill")
			if err != nil {
				return err
			}
			if err := infernexutils.VerifyLeaderWorkerSetTemplateSize(dynamicClient, ctx, namespace, prefillLWS, expectedLWSSize); err != nil {
				return fmt.Errorf("FIT_28 prefill LWS: %w", err)
			}
			return infernexutils.VerifyLeaderWorkerSetPodsDataParallelStartRanks(restConfig, clientset, ctx, namespace, prefillLWS, expectedRanks)
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "FIT_28 prefill LeaderWorkerSet 校验失败")
		GinkgoWriter.Println("FIT_28 prefill LeaderWorkerSet size 与 data-parallel-start-rank 校验成功")

		Eventually(func() error {
			decodeLWS, err := infernexutils.FindLeaderWorkerSetNameByRole(dynamicClient, ctx, namespace, "decode")
			if err != nil {
				return err
			}
			if err := infernexutils.VerifyLeaderWorkerSetTemplateSize(dynamicClient, ctx, namespace, decodeLWS, expectedLWSSize); err != nil {
				return fmt.Errorf("FIT_36 decode LWS: %w", err)
			}
			return infernexutils.VerifyLeaderWorkerSetPodsDataParallelStartRanks(restConfig, clientset, ctx, namespace, decodeLWS, expectedRanks)
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "FIT_36 decode LeaderWorkerSet 校验失败")
		GinkgoWriter.Println("FIT_36 decode LeaderWorkerSet size 与 data-parallel-start-rank 校验成功")

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		const moeModel = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "what's the result of 100+100?", moeModel, 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败（FIT_28/FIT_36）")
		GinkgoWriter.Println("流式推理请求校验成功（FIT_28/FIT_36）")

		resources, err := infernexutils.HelmGetManifest(dpReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
	})

	It("InferNex_FIT_29-自定义prefill节点不匹配的dp parallel配置", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		fit29ReleaseName := releaseName + "-fit29"

		DeferCleanup(deferCleanupInfernexRelease, fit29ReleaseName, namespace)

		err := helmInstallRelease(fit29ReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-Coder-30B-A3B-Instruct-vllm-dp-error-local-prefill.yaml"), nil)
		expectHelmInstallDataParallelDivisibilityError(err, "qwen3-coder-pd-0", "prefill", 4, 3)
		Expect(infernexutils.HelmReleaseExists(fit29ReleaseName, namespace)).To(BeFalse(), "FIT_29: 安装失败后 release 不应存在")
	})

	It("InferNex_FIT_30-自定义decode节点tp;InferNex_FIT_31-自定义decode节点prefixCaching;InferNex_FIT_32-自定义decode节点maxModellength;InferNex_FIT_33-自定义decode节点maxNumBatchedTokends;InferNex_FIT_34-自定义decode节点gpuMemoryUtilization", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		decodeReleaseName := releaseName + "-fit3034"

		// FIT_30~34：使用外挂 yaml 配置
		err := helmInstallRelease(decodeReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-8B-pd-vllm-config-params.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		DeferCleanup(deferCleanupInfernexRelease, decodeReleaseName, namespace)

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有Pod running 1/1超时（20分钟）")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		Expect(restConfig).NotTo(BeNil())
		verifyPDEngineExtraArgs(namespace, infernexutils.DecodePDRole, "InferNex_FIT_31", PDVllmConfigExtraArgMarkers)

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace, "what's the result of 100+100?", "", 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败（FIT_30~34）")
		GinkgoWriter.Println("流式推理请求校验成功（FIT_30~34）")

		resources, err := infernexutils.HelmGetManifest(decodeReleaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(resources).NotTo(BeEmpty(), "helm应该安装了至少一个资源")
		GinkgoWriter.Printf("\nhelm 安装了 %d 个资源\n", len(resources))
	})

	It("InferNex_FIT_37-自定义decode节点不匹配的dp parallel配置", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())

		fit37ReleaseName := releaseName + "-fit37"

		DeferCleanup(deferCleanupInfernexRelease, fit37ReleaseName, namespace)

		err := helmInstallRelease(fit37ReleaseName, chartURL, chartVersion, namespace,
			filepath.Join("config", "Qwen3-Coder-30B-A3B-Instruct-vllm-dp-error-local-decode.yaml"), nil)
		expectHelmInstallDataParallelDivisibilityError(err, "qwen3-coder-pd-0", "decode", 4, 3)
		Expect(infernexutils.HelmReleaseExists(fit37ReleaseName, namespace)).To(BeFalse(), "FIT_37: 安装失败后 release 不应存在")
	})

	It("InferNex_FIT_38-自定义prefill节点环境变量;InferNex_FIT_39-自定义decode节点环境变量;InferNex_FIT_40-自定义推理节点节点环境变量", Label("npu", "P2"), func() {
		Expect(clientset).NotTo(BeNil())
		Expect(restConfig).NotTo(BeNil())

		const pdTestEnvRelease = "infernex-pd-test-env"
		pdTestEnvValues := filepath.Join("config", "Qwen3-8B-pd-test-env.yaml")

		DeferCleanup(deferCleanupInfernexRelease, pdTestEnvRelease, namespace)

		err := helmInstallRelease(pdTestEnvRelease, chartURL, chartVersion, namespace, pdTestEnvValues, nil)
		Expect(err).NotTo(HaveOccurred(), "helm install（Qwen3-8B-pd-test-env.yaml）应成功")

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有 Pod running 1/1 超时")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		verifyPDEngineContainerEnv(namespace, infernexutils.PrefillPDRole, "InferNex_FIT_38", PDTestEnvVarName, PDTestEnvVarValue)
		verifyPDEngineContainerEnv(namespace, infernexutils.DecodePDRole, "InferNex_FIT_39", PDTestEnvVarName, PDTestEnvVarValue)
		verifyPDEngineContainerEnv(namespace, infernexutils.PrefillPDRole, "InferNex_FIT_40", PDBackendTestEnvVar1Name, PDBackendTestEnvVar1Value)
		verifyPDEngineContainerEnv(namespace, infernexutils.DecodePDRole, "InferNex_FIT_40", PDBackendTestEnvVar1Name, PDBackendTestEnvVar1Value)

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace,
				"what's the result of 100+100?", defaultModelName(), 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败")
		GinkgoWriter.Println("流式推理请求校验成功")
	})

	It("InferNex_FIT_41-集成部署xPyD流式推理", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())

		// FIT_41 自包含：独立 release，覆盖 PD 副本数后默认 chart 一键部署
		const fit41ReleaseName = "infernex-fit41"
		fit41Namespace := namespace

		DeferCleanup(deferCleanupInfernexRelease, fit41ReleaseName, fit41Namespace)

		err := helmInstallRelease(fit41ReleaseName, chartURL, chartVersion, fit41Namespace,
			filepath.Join("config", "Qwen3-8B-xpyd.yaml"), nil)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, fit41Namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_41: 等待所有 Pod running 1/1 超时")
		GinkgoWriter.Println("FIT_41: 所有 Pod 已 running 1/1")

		// FIT_41 使用 LWS 架构，WaitForInfernexBackendReplicas 仅适用于 Deployment 架构，故跳过该校验
		// infernexutils.WaitForInfernexBackendReplicas(clientset, ctx, fit41Namespace, 3, 2, 10*time.Minute)
		GinkgoWriter.Println("FIT_41: 跳过 Deployment 副本数校验（LWS 架构）")

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, fit41Namespace,
				"what's the result of 100+100?", defaultModelName(), 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "FIT_41: 流式推理请求校验失败")
		GinkgoWriter.Println("FIT_41: 流式推理请求校验成功")
	})

	It("InferNex_FIT_42-自定义prefill节点额外启动参数;InferNex_FIT_43-自定义decode节点额外启动参数", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())
		Expect(restConfig).NotTo(BeNil())

		const pdExtraArgsRelease = "infernex-fit42"
		pdExtraArgsValues := filepath.Join("config", "Qwen3-8B-pd-extra-args.yaml")

		DeferCleanup(deferCleanupInfernexRelease, pdExtraArgsRelease, namespace)

		err := helmInstallRelease(pdExtraArgsRelease, chartURL, chartVersion, namespace, pdExtraArgsValues, nil)
		Expect(err).NotTo(HaveOccurred(), "helm install（Qwen3-8B-pd-extra-args.yaml）应成功")

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "等待所有 Pod running 1/1 超时")
		GinkgoWriter.Println("所有 Pod 已 running 1/1")

		verifyPDEngineExtraArgs(namespace, infernexutils.PrefillPDRole, "InferNex_FIT_42", PDExtraArgMarkers)
		verifyPDEngineExtraArgs(namespace, infernexutils.DecodePDRole, "InferNex_FIT_43", PDExtraArgMarkers)

		k8sAdapter := &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
		Eventually(func() error {
			return infernexutils.VerifyStreamChatCompletionsInference(ctx, k8sAdapter, namespace,
				"what's the result of 100+100?", defaultModelName(), 0)
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "流式推理请求校验失败")
		GinkgoWriter.Println("流式推理请求校验成功")
	})

	It("InferNex_FIT_46-Helm参数校验-非法参数部署失败并提示明确错误", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())

		const fit46Release = "infernex-fit46"
		DeferCleanup(deferCleanupInfernexRelease, fit46Release, namespace)

		modeOverlay, err := writeFirstServiceOverlay(chartURL, chartVersion, func(svc map[string]any) {
			svc["mode"] = "foo"
		})
		Expect(err).NotTo(HaveOccurred(), "FIT_46: 生成非法 mode overlay 失败")
		defer os.Remove(modeOverlay)

		err = helmInstallRelease(fit46Release, chartURL, chartVersion, namespace, modeOverlay, nil)
		Expect(err).To(HaveOccurred(), "FIT_46: mode=foo 时 helm install 应失败")
		Expect(err.Error()).To(ContainSubstring("services[qwen3-8b-2p2d-01].mode must be 'pd' or 'aggregated'"),
			"FIT_46: 期望包含 mode 校验错误，实际: %v", err)
		Expect(infernexutils.HelmReleaseExists(fit46Release, namespace)).To(BeFalse(), "FIT_46: mode 失败后 release 不应存在")
		GinkgoWriter.Println("FIT_46: 非法 mode 校验错误符合预期")

		kvOverlay, err := writeFirstServiceOverlay(chartURL, chartVersion, func(svc map[string]any) {
			delete(svc, "kvTransferConfig")
		})
		Expect(err).NotTo(HaveOccurred(), "FIT_46: 生成缺 kvTransferConfig overlay 失败")
		defer os.Remove(kvOverlay)

		err = helmInstallRelease(fit46Release, chartURL, chartVersion, namespace, kvOverlay, nil)
		Expect(err).To(HaveOccurred(), "FIT_46: 缺少 kvTransferConfig 时 helm install 应失败")
		Expect(err.Error()).To(ContainSubstring("services[qwen3-8b-2p2d-01].kvTransferConfig is required when mode is 'pd'"),
			"FIT_46: 期望包含 kvTransferConfig 校验错误，实际: %v", err)
		Expect(infernexutils.HelmReleaseExists(fit46Release, namespace)).To(BeFalse(), "FIT_46: kv 失败后 release 不应存在")
		GinkgoWriter.Println("FIT_46: 缺少 kvTransferConfig 校验错误符合预期")
	})

	It("InferNex_FIT_44-inference-gateway兼容HTTP/1.0客户端;InferNex_FIT_47-proxy-server透传后端HTTP错误码与错误信息;InferNex_FIT_48-推理过程中vLLM与proxy-server日志无噪声刷屏", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())
		Expect(dynamicClient).NotTo(BeNil())

		const fit44Release = "infernex-fit44"
		DeferCleanup(deferCleanupInfernexRelease, fit44Release, namespace)

		err := helmInstall(fit44Release, chartURL, chartVersion, namespace)
		Expect(err).NotTo(HaveOccurred(), "FIT_44/47/48: helm install 默认配置应成功")

		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_44/47/48: 等待所有 Pod running 1/1 超时")

		By("FIT_47: 不存在的模型名应透传后端 4xx，而不是 proxy-server 统一 500")
		Eventually(func() error {
			status, body, reqErr := postChatCompletionsRaw(namespace, map[string]any{
				"model": "model-not-exist",
				"messages": []map[string]string{
					{"role": "user", "content": "hello"},
				},
				"stream": false,
			})
			if reqErr != nil {
				return reqErr
			}
			if status == http.StatusInternalServerError {
				return fmt.Errorf("FIT_47 未知模型: 得到 500，响应: %s", body)
			}
			if status < 400 || status > 499 {
				return fmt.Errorf("FIT_47 未知模型: 期望 4xx，实际 %d，响应: %s", status, body)
			}
			if strings.TrimSpace(body) == "" {
				return fmt.Errorf("FIT_47 未知模型: 响应体为空")
			}
			GinkgoWriter.Printf("FIT_47 未知模型: status=%d body=%s\n", status, truncateForLog(body, 512))
			return nil
		}, 5*time.Minute, 15*time.Second).Should(Succeed())

		By("FIT_47: 缺少 messages 的非法请求体应透传后端错误信息")
		status, body, reqErr := postChatCompletionsRaw(namespace, map[string]any{
			"model":  defaultModelName(),
			"stream": false,
		})
		Expect(reqErr).NotTo(HaveOccurred(), "FIT_47 缺 messages: 请求应发出")
		Expect(status).NotTo(Equal(http.StatusInternalServerError), "FIT_47 缺 messages: 不应被改写为 500，响应: %s", body)
		Expect(status).To(BeNumerically(">=", 400), "FIT_47 缺 messages: 期望错误状态码，实际 %d", status)
		Expect(strings.TrimSpace(body)).NotTo(BeEmpty(), "FIT_47 缺 messages: 响应体应包含后端错误原文")
		GinkgoWriter.Printf("FIT_47 缺 messages: status=%d body=%s\n", status, truncateForLog(body, 512))

		By("FIT_44: helm upgrade 开启 inferenceGateway.enable_http10")
		Eventually(func() error {
			return ensureNamespaceActive("scaling-system")
		}, 5*time.Minute, 5*time.Second).Should(Succeed(), "FIT_44: scaling-system 应处于 Active 后再 upgrade")
		err = helmUpgradeWithSets(fit44Release, chartURL, chartVersion, namespace, []string{
			"inferenceGateway.enable_http10=true",
		}, true, "--no-hooks")
		Expect(err).NotTo(HaveOccurred(), "FIT_44: helm upgrade enable_http10 应成功")

		efGVR := schema.GroupVersionResource{Group: "networking.istio.io", Version: "v1alpha3", Resource: "envoyfilters"}
		Eventually(func() error {
			_, getErr := dynamicClient.Resource(efGVR).Namespace(namespace).Get(ctx, "inference-gateway-accept-http-10", metav1.GetOptions{})
			return getErr
		}, 2*time.Minute, 5*time.Second).Should(Succeed(), "FIT_44: 未找到 EnvoyFilter inference-gateway-accept-http-10")
		GinkgoWriter.Println("FIT_44: EnvoyFilter inference-gateway-accept-http-10 已创建")

		Eventually(func() error {
			httpStatus, respBody, httpErr := sendHTTP10ChatCompletion(namespace, defaultModelName(), "what's the result of 100+100?")
			if httpErr != nil {
				return httpErr
			}
			if httpStatus != http.StatusOK {
				return fmt.Errorf("FIT_44 HTTP/1.0 期望 200，实际 %d，响应: %s", httpStatus, truncateForLog(respBody, 512))
			}
			if !strings.Contains(respBody, "choices") && !strings.Contains(respBody, "content") {
				return fmt.Errorf("FIT_44 HTTP/1.0 响应缺少推理结果: %s", truncateForLog(respBody, 512))
			}
			return nil
		}, 5*time.Minute, 15*time.Second).Should(Succeed(), "FIT_44: HTTP/1.0 无 Host 请求应被接受")
		GinkgoWriter.Println("FIT_44: HTTP/1.0 推理请求校验成功")

		By("FIT_48: 完成一次推理后空闲 60s，检查 proxy-server / vLLM 访问日志无高频噪声")
		Eventually(func() error {
			return verifyInferenceRequest(namespace, defaultModelName())
		}, 5*time.Minute, 30*time.Second).Should(Succeed(), "FIT_48: 推理请求失败")
		GinkgoWriter.Println("FIT_48: 等待 60s 空闲...")
		time.Sleep(60 * time.Second)
		Expect(assertNoNoisyLogs(namespace, "proxy-server", 5)).NotTo(HaveOccurred(), "FIT_48: proxy-server 日志噪声超标")
		Expect(assertNoNoisyLogs(namespace, "prefill-engine", 5)).NotTo(HaveOccurred(), "FIT_48: prefill 引擎日志噪声超标")
		Expect(assertNoNoisyLogs(namespace, "decode-engine", 5)).NotTo(HaveOccurred(), "FIT_48: decode 引擎日志噪声超标")
		GinkgoWriter.Println("FIT_48: 日志噪声校验通过")
	})

	It("InferNex_FIT_45-外部mooncake master复用部署", Label("npu", "P1"), func() {
		Expect(clientset).NotTo(BeNil())

		const (
			fit45ReleaseA = "infernex-fit45-a"
			fit45ReleaseB = "infernex-fit45-b"
		)
		valuesFile := filepath.Join("config", "Qwen3-8B-pd-external-mooncake.yaml")

		for _, ns := range []string{fit45ExternalMooncakeNS, "scaling-system", "istio-system"} {
			Eventually(func() error { return ensureNamespaceActive(ns) }, 5*time.Minute, 5*time.Second).
				Should(Succeed(), "FIT_45: 确保 namespace %s Active", ns)
		}
		_ = helmUninstallIfExists("infernex-fit44", namespace)
		Eventually(func() bool {
			return !infernexutils.HelmReleaseExists("infernex-fit44", namespace)
		}, 3*time.Minute, 5*time.Second).Should(BeTrue(), "FIT_45: 等待 infernex-fit44 卸载完成")
		DeferCleanup(func() {
			GinkgoWriter.Printf("DeferCleanup: 删除外部 mooncake namespace %s\n", fit45ExternalMooncakeNS)
			_ = clientset.CoreV1().Namespaces().Delete(ctx, fit45ExternalMooncakeNS, metav1.DeleteOptions{})
		})
		Expect(applyMooncakeStoreFromChart(chartURL, chartVersion, fit45ExternalMooncakeNS)).NotTo(HaveOccurred(),
			"FIT_45: 部署外部 mooncake master/redis 失败")
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, fit45ExternalMooncakeNS)
		}, 5*time.Minute, 10*time.Second).Should(Succeed(), "FIT_45: 外部 mooncake/redis 未就绪")

		DeferCleanup(func() { deferCleanupReleaseOnly(fit45ReleaseA, namespace) })
		DeferCleanup(func() { deferCleanupReleaseOnly(fit45ReleaseB, fit45SecondNamespace) })

		err := helmInstallRelease(fit45ReleaseA, chartURL, chartVersion, namespace, valuesFile, nil)
		Expect(err).NotTo(HaveOccurred(), "FIT_45: 第一套 InferNex helm install 应成功")
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, namespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_45: 第一套 Pod 未就绪")
		assertNoBuiltInMooncake(namespace)

		err = helmInstallRelease(fit45ReleaseB, chartURL, chartVersion, fit45SecondNamespace, valuesFile, []string{
			fmt.Sprintf("inferenceGateway.service.nodePorts.http=%d", fit45SecondGatewayPort),
			"inferenceGateway.istio.enabled=false",
		})
		Expect(err).NotTo(HaveOccurred(), "FIT_45: 第二套 InferNex helm install 应成功")
		Eventually(func() error {
			return infernexutils.AllPodsRunningAndReady(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset}, fit45SecondNamespace)
		}, 20*time.Minute, 15*time.Second).Should(Succeed(), "FIT_45: 第二套 Pod 未就绪")
		assertNoBuiltInMooncake(fit45SecondNamespace)

		longPrompt := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 200)
		Expect(len(strings.Fields(longPrompt))).To(BeNumerically(">=", 1024), "FIT_45: 长 prompt 应超过 1024 token")

		Eventually(func() error {
			st, body, reqErr := postChatCompletionsRaw(namespace, map[string]any{
				"model": defaultModelName(),
				"messages": []map[string]string{
					{"role": "user", "content": longPrompt},
				},
				"max_tokens": 16,
				"stream":     false,
			})
			if reqErr != nil {
				return reqErr
			}
			if st != http.StatusOK {
				return fmt.Errorf("FIT_45 第一套推理失败 status=%d body=%s", st, truncateForLog(body, 512))
			}
			return nil
		}, 8*time.Minute, 20*time.Second).Should(Succeed(), "FIT_45: 第一套长文本推理失败")
		GinkgoWriter.Println("FIT_45: 第一套长文本推理成功，等待 mooncake L3 摄入后再打第二套")
		time.Sleep(30 * time.Second)

		Eventually(func() error {
			st, body, reqErr := postChatCompletionsRaw(fit45SecondNamespace, map[string]any{
				"model": defaultModelName(),
				"messages": []map[string]string{
					{"role": "user", "content": longPrompt},
				},
				"max_tokens": 16,
				"stream":     false,
			})
			if reqErr != nil {
				return reqErr
			}
			if st != http.StatusOK {
				return fmt.Errorf("FIT_45 第二套推理失败 status=%d body=%s", st, truncateForLog(body, 512))
			}
			return nil
		}, 8*time.Minute, 20*time.Second).Should(Succeed(), "FIT_45: 第二套长文本推理失败")
		GinkgoWriter.Println("FIT_45: 第二套长文本推理成功")

		Eventually(func() error {
			return assertExternalPrefixCacheHit(fit45SecondNamespace)
		}, 3*time.Minute, 15*time.Second).Should(Succeed(), "FIT_45: 第二套引擎日志未出现较高的 external prefix cache hit")
		GinkgoWriter.Println("FIT_45: 外部 prefix cache hit 校验通过")
	})

})

// verifyPDEngineExtraArgs 校验指定 pdRole 的 vllm serve 命令包含 expected extraArgs 片段。
func verifyPDEngineExtraArgs(namespace, pdRole, fitLabel string, extraArgMarkers []string) {
	Expect(clientset).NotTo(BeNil())
	Expect(restConfig).NotTo(BeNil())

	Eventually(func() error {
		targets, err := infernexutils.ListPDEngineTargets(clientset, ctx, namespace, pdRole)
		if err != nil {
			return err
		}
		for _, t := range targets {
			GinkgoWriter.Printf("%s 步骤1: pod=%s container=%s (label %s=%s)\n",
				fitLabel, t.Pod.Name, t.Container, infernexutils.PDRoleLabelKey, pdRole)
		}
		return nil
	}, 10*time.Minute, 30*time.Second).Should(Succeed(), "%s: 步骤1 定位 Pod/容器失败", fitLabel)

	Eventually(func() error {
		targets, err := infernexutils.ListPDEngineTargets(clientset, ctx, namespace, pdRole)
		if err != nil {
			return err
		}
		for _, t := range targets {
			cmdline, err := infernexutils.GetVLLMServeCommand(restConfig, clientset, ctx, namespace, t)
			if err != nil {
				return err
			}
			GinkgoWriter.Printf("%s 步骤2: pod=%s vllm serve 命令:\n%s\n", fitLabel, t.Pod.Name, cmdline)
			if err := infernexutils.VerifyVLLMCommandContainsExtraArgs(cmdline, extraArgMarkers); err != nil {
				return fmt.Errorf("%s 步骤3 pod=%s: %w", fitLabel, t.Pod.Name, err)
			}
			GinkgoWriter.Printf("%s 步骤3: pod=%s extraArgs 校验通过\n", fitLabel, t.Pod.Name)
		}
		return nil
	}, 10*time.Minute, 30*time.Second).Should(Succeed(), "%s: vllm extraArgs 校验失败", fitLabel)
}

// verifyPDEngineContainerEnv 定位 pdRole 对应 engine Pod/容器，在容器内 echo 校验环境变量是否符合预期。
func verifyPDEngineContainerEnv(namespace, pdRole, fitLabel, expectEnvName, expectEnvValue string) {
	Expect(clientset).NotTo(BeNil())
	Expect(restConfig).NotTo(BeNil())

	Eventually(func() error {
		targets, err := infernexutils.ListPDEngineTargets(clientset, ctx, namespace, pdRole)
		if err != nil {
			return err
		}
		for _, t := range targets {
			GinkgoWriter.Printf("%s 步骤1: pod=%s container=%s (label %s=%s)\n",
				fitLabel, t.Pod.Name, t.Container, infernexutils.PDRoleLabelKey, pdRole)
		}
		return nil
	}, 10*time.Minute, 30*time.Second).Should(Succeed(), "%s: 步骤1 定位 Pod/容器失败", fitLabel)

	Eventually(func() error {
		targets, err := infernexutils.ListPDEngineTargets(clientset, ctx, namespace, pdRole)
		if err != nil {
			return err
		}
		for _, t := range targets {
			got, err := infernexutils.GetContainerEnvVarValue(restConfig, clientset, ctx, namespace, t, expectEnvName)
			if err != nil {
				return err
			}
			GinkgoWriter.Printf("%s 步骤2: pod=%s echo $%s => %q\n", fitLabel, t.Pod.Name, expectEnvName, got)
			if got != expectEnvValue {
				return fmt.Errorf("%s 步骤3 pod=%s: %s=%q，期望 %q",
					fitLabel, t.Pod.Name, expectEnvName, got, expectEnvValue)
			}
			GinkgoWriter.Printf("%s 步骤3: pod=%s 环境变量 %s 校验通过\n", fitLabel, t.Pod.Name, expectEnvName)
		}
		return nil
	}, 10*time.Minute, 30*time.Second).Should(Succeed(), "%s: 环境变量 %s 校验失败", fitLabel, expectEnvName)
}

// --- helm 操作辅助函数 ---

// expectHelmInstallDataParallelDivisibilityError 断言 helm install 因 DP 无法整除失败，且 stderr/stdout 含 chart 校验文案。
func expectHelmInstallDataParallelDivisibilityError(err error, serviceName, pdRole string, dataParallelSize, dataParallelSizeLocal int) {
	Expect(err).To(HaveOccurred(), "helm install 应失败（dataParallelSize=%d, dataParallelSizeLocal=%d）",
		dataParallelSize, dataParallelSizeLocal)
	msg := err.Error()
	want := fmt.Sprintf("services[%s].pd.%s.dataParallelSize (%d) must be divisible by dataParallelSizeLocal (%d)",
		serviceName, pdRole, dataParallelSize, dataParallelSizeLocal)
	Expect(msg).To(ContainSubstring(want), "期望报错包含: %s\n实际: %s", want, msg)
	GinkgoWriter.Printf("helm install 按预期失败: %s\n", want)
}

// isLocalChartPath 判断 chartURL 是否为本地路径（非 OCI URL）。
func isLocalChartPath(chartURL string) bool {
	return chartURL != "" && !strings.HasPrefix(chartURL, "oci://")
}

// helmInstallWithCreateNS 使用 helm install 部署 chart，带 --create-namespace 参数。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
func helmInstallWithCreateNS(releaseName, chartURL, version, namespace string) error {
	return helmInstallWithSetsAndCreateNS(releaseName, chartURL, version, namespace, nil)
}

// helmInstallWithSetsAndCreateNS 使用 helm install 部署 chart，支持 --create-namespace 和 --set。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
func helmInstallWithSetsAndCreateNS(releaseName, chartURL, version, namespace string, setValues []string) error {
	args := []string{"install", releaseName, chartURL}
	if !isLocalChartPath(chartURL) {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	args = append(args, "-n", namespace, "--create-namespace")
	for _, sv := range setValues {
		args = append(args, "--set", sv)
	}

	GinkgoWriter.Printf("执行: helm %s\n", strings.Join(args, " "))

	cmd := exec.Command("helm", args...)
	cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	GinkgoWriter.Printf("helm install 成功: %s\n", stdout.String())
	return nil
}

// helmInstall 使用 helm install 部署 chart（不传额外 --set）。
// 支持 OCI URL (oci://...) 和本地 chart 路径。
func helmInstall(releaseName, chartURL, version, namespace string) error {
	return helmInstallWithSets(releaseName, chartURL, version, namespace, nil)
}

// helmInstallWithSets 使用 helm install 部署 chart，支持 --set 参数。
func helmInstallWithSets(releaseName, chartURL, version, namespace string, setValues []string) error {
	return helmInstallOCIWithSets(releaseName, chartURL, version, namespace, setValues)
}

// helmInstallRelease 使用 helm install 部署 chart，通过 -f 指定 values 文件，可选额外 --set。
func helmInstallRelease(releaseName, chartURL, version, namespace string, valuesFile string, setValues []string) error {
	args := []string{"install", releaseName, chartURL}
	if !strings.HasPrefix(chartURL, "/") && !strings.HasPrefix(chartURL, ".") && !strings.HasSuffix(chartURL, ".tgz") {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	args = append(args, "-n", namespace, "--create-namespace")
	if valuesFile != "" {
		args = append(args, "-f", valuesFile)
	}
	for _, sv := range setValues {
		args = append(args, "--set", sv)
	}

	GinkgoWriter.Printf("执行: helm %s\n", strings.Join(args, " "))

	cmd := exec.Command("helm", args...)
	cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	GinkgoWriter.Printf("helm install 成功: %s\n", stdout.String())
	return nil
}

func helmUpgradeWithSets(releaseName, chartURL, version, namespace string, setValues []string, reuseValues bool, extraArgs ...string) error {
	args := []string{"upgrade", releaseName, chartURL, "-n", namespace}
	if reuseValues {
		args = append(args, "--reuse-values")
	}
	if !isLocalChartPath(chartURL) {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	for _, sv := range setValues {
		args = append(args, "--set", sv)
	}
	args = append(args, extraArgs...)
	GinkgoWriter.Printf("执行: helm %s\n", strings.Join(args, " "))
	cmd := exec.Command("helm", args...)
	cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm upgrade failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	GinkgoWriter.Printf("helm upgrade 成功: %s\n", stdout.String())
	return nil
}

func appendHelmVersion(args []string, chartURL, version string) []string {
	if isLocalChartPath(chartURL) {
		return args
	}
	v := strings.TrimSpace(version)
	if v != "" && !strings.EqualFold(v, "latest") {
		return append(args, "--version", v)
	}
	return args
}

func writeFirstServiceOverlay(chartURL, version string, mutate func(map[string]any)) (string, error) {
	args := appendHelmVersion([]string{"show", "values", chartURL}, chartURL, version)
	cmd := exec.Command("helm", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return "", fmt.Errorf("helm show values: %w\n%s", err, stderr.String())
	}
	raw := stdout.Bytes()
	var root map[string]any
	if err := yaml.Unmarshal(raw, &root); err != nil {
		return "", fmt.Errorf("unmarshal chart values.yaml: %w", err)
	}
	ib, _ := root["inference-backend"].(map[string]any)
	if ib == nil {
		return "", fmt.Errorf("chart values.yaml 缺少 inference-backend")
	}
	services, _ := ib["services"].([]any)
	if len(services) == 0 {
		return "", fmt.Errorf("chart values.yaml 缺少 inference-backend.services[0]")
	}
	svc, _ := services[0].(map[string]any)
	if svc == nil {
		return "", fmt.Errorf("chart values.yaml services[0] 不是 map")
	}
	mutate(svc)
	overlay := map[string]any{
		"inference-backend": map[string]any{
			"services": []any{svc},
		},
	}
	out, err := yaml.Marshal(overlay)
	if err != nil {
		return "", err
	}
	f, err := os.CreateTemp("", "infernex-fit46-*.yaml")
	if err != nil {
		return "", err
	}
	if _, err := f.Write(out); err != nil {
		_ = f.Close()
		_ = os.Remove(f.Name())
		return "", err
	}
	if err := f.Close(); err != nil {
		_ = os.Remove(f.Name())
		return "", err
	}
	return f.Name(), nil
}

func deferCleanupReleaseOnly(releaseName, namespace string) {
	GinkgoWriter.Printf("DeferCleanup: helm uninstall %s/%s（若存在，不删除共享 ns）\n", namespace, releaseName)
	if err := helmUninstallIfExists(releaseName, namespace); err != nil {
		GinkgoWriter.Printf("DeferCleanup: helm uninstall 失败: %v\n", err)
	}
	time.Sleep(infernexDeferCleanupWait)
}

func ensureNamespace(namespace string) error {
	return ensureNamespaceActive(namespace)
}

func ensureNamespaceActive(namespace string) error {
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if apierrors.IsNotFound(err) {
		_, err = clientset.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: namespace},
		}, metav1.CreateOptions{})
		return err
	}
	if err != nil {
		return err
	}
	if ns.Status.Phase == corev1.NamespaceTerminating {
		if len(ns.Spec.Finalizers) > 0 {
			ns.Spec.Finalizers = nil
			_, _ = clientset.CoreV1().Namespaces().Finalize(ctx, ns, metav1.UpdateOptions{})
		}
		return fmt.Errorf("namespace %s 仍处于 Terminating", namespace)
	}
	if ns.Status.Phase != corev1.NamespaceActive {
		return fmt.Errorf("namespace %s phase=%s", namespace, ns.Status.Phase)
	}
	return nil
}

func applyMooncakeStoreFromChart(chartURL, version, namespace string) error {
	if err := ensureNamespace(namespace); err != nil {
		return err
	}
	args := appendHelmVersion([]string{"template", "mooncake-ext", chartURL, "-n", namespace}, chartURL, version)
	cmd := exec.Command("helm", args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm template mooncake store: %w\n%s", err, stderr.String())
	}
	out := stdout.Bytes()
	keep := map[string]struct{}{
		"mooncake-master-deployment": {},
		"mooncake-master-service":    {},
		"redis-server-deployment":    {},
		"redis-service":              {},
	}
	dec := yaml.NewDecoder(bytes.NewReader(out))
	var docs []any
	for {
		var doc map[string]any
		if decErr := dec.Decode(&doc); decErr != nil {
			if decErr == io.EOF {
				break
			}
			return fmt.Errorf("decode helm template: %w", decErr)
		}
		if len(doc) == 0 {
			continue
		}
		md, _ := doc["metadata"].(map[string]any)
		name, _ := md["name"].(string)
		if _, ok := keep[name]; ok {
			if ns, _ := md["namespace"].(string); ns == "" {
				md["namespace"] = namespace
				doc["metadata"] = md
			}
			docs = append(docs, doc)
		}
	}
	if len(docs) < 4 {
		return fmt.Errorf("helm template 仅解析到 %d 个 mooncake/redis 资源，期望 4 个", len(docs))
	}
	var buf bytes.Buffer
	enc := yaml.NewEncoder(&buf)
	for _, d := range docs {
		if err := enc.Encode(d); err != nil {
			_ = enc.Close()
			return err
		}
	}
	_ = enc.Close()
	apply := exec.Command("kubectl", "apply", "-f", "-")
	apply.Stdin = &buf
	applied, err := apply.CombinedOutput()
	if err != nil {
		return fmt.Errorf("kubectl apply mooncake store: %w\n%s", err, string(applied))
	}
	GinkgoWriter.Printf("FIT_45: 外部 mooncake/redis 已应用到 %s: %s\n", namespace, string(applied))
	return nil
}

func assertNoBuiltInMooncake(namespace string) {
	_, err := clientset.AppsV1().Deployments(namespace).Get(ctx, "mooncake-master-deployment", metav1.GetOptions{})
	Expect(apierrors.IsNotFound(err)).To(BeTrue(), "%s 不应存在 mooncake-master-deployment", namespace)
	_, err = clientset.CoreV1().Services(namespace).Get(ctx, "mooncake-master-service", metav1.GetOptions{})
	Expect(apierrors.IsNotFound(err)).To(BeTrue(), "%s 不应存在 mooncake-master-service", namespace)
}

func postChatCompletionsRaw(namespace string, reqBody map[string]any) (int, string, error) {
	baseURL, err := resolveInferenceURL(namespace)
	if err != nil {
		return 0, "", fmt.Errorf("解析推理服务 URL 失败: %w", err)
	}
	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"
	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return 0, "", err
	}
	GinkgoWriter.Printf("POST %s body=%s\n", fullURL, truncateForLog(string(bodyBytes), 256))
	client := &http.Client{Timeout: 180 * time.Second}
	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return 0, "", err
	}
	req.Header.Set("Content-Type", "application/json")
	resp, err := client.Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return resp.StatusCode, "", err
	}
	return resp.StatusCode, string(respBody), nil
}

func sendHTTP10ChatCompletion(namespace, model, message string) (int, string, error) {
	baseURL, err := resolveInferenceURL(namespace)
	if err != nil {
		return 0, "", err
	}
	u, err := url.Parse(baseURL)
	if err != nil {
		return 0, "", err
	}
	hostPort := u.Host
	if !strings.Contains(hostPort, ":") {
		hostPort += ":80"
	}
	payload, err := json.Marshal(map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": message},
		},
		"stream":     false,
		"max_tokens": 16,
	})
	if err != nil {
		return 0, "", err
	}
	req := fmt.Sprintf("POST /v1/chat/completions HTTP/1.0\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n%s",
		len(payload), payload)
	GinkgoWriter.Printf("FIT_44: raw HTTP/1.0 -> %s\n", hostPort)
	conn, err := net.DialTimeout("tcp", hostPort, 15*time.Second)
	if err != nil {
		return 0, "", err
	}
	defer conn.Close()
	_ = conn.SetDeadline(time.Now().Add(180 * time.Second))
	if _, err := io.WriteString(conn, req); err != nil {
		return 0, "", err
	}
	respBytes, err := io.ReadAll(conn)
	if err != nil && len(respBytes) == 0 {
		return 0, "", err
	}
	resp := string(respBytes)
	reader := bufio.NewReader(strings.NewReader(resp))
	statusLine, err := reader.ReadString('\n')
	if err != nil {
		return 0, resp, fmt.Errorf("读取 HTTP/1.0 状态行失败: %w raw=%s", err, truncateForLog(resp, 256))
	}
	fields := strings.Fields(statusLine)
	if len(fields) < 2 {
		return 0, resp, fmt.Errorf("无法解析 HTTP/1.0 状态行: %q", statusLine)
	}
	var status int
	if _, err := fmt.Sscanf(fields[1], "%d", &status); err != nil {
		return 0, resp, fmt.Errorf("无法解析状态码: %q", statusLine)
	}
	parts := strings.SplitN(resp, "\r\n\r\n", 2)
	body := resp
	if len(parts) == 2 {
		body = parts[1]
	}
	return status, body, nil
}

var logTimestampPrefix = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}[^\s]*\s+`)

func assertNoNoisyLogs(namespace, container string, maxRepeat int) error {
	since := int64(5 * 60)
	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return err
	}
	found := false
	for i := range pods.Items {
		pod := &pods.Items[i]
		if !podHasNamedContainer(pod, container) {
			continue
		}
		found = true
		opts := &corev1.PodLogOptions{Container: container, SinceSeconds: &since}
		raw, logErr := clientset.CoreV1().Pods(namespace).GetLogs(pod.Name, opts).Do(ctx).Raw()
		if logErr != nil {
			return fmt.Errorf("读取 %s/%s 日志失败: %w", pod.Name, container, logErr)
		}
		counts := map[string]int{}
		for _, line := range strings.Split(string(raw), "\n") {
			norm := strings.TrimSpace(logTimestampPrefix.ReplaceAllString(line, ""))
			if norm == "" {
				continue
			}
			lower := strings.ToLower(norm)
			if strings.Contains(lower, "traceback") || strings.Contains(norm, "Panic") {
				return fmt.Errorf("%s/%s 出现异常堆栈: %s", pod.Name, container, truncateForLog(norm, 300))
			}
			if !isLogNoiseCandidate(norm, lower) {
				continue
			}
			counts[norm]++
		}
		for msg, n := range counts {
			if n > maxRepeat {
				return fmt.Errorf("%s/%s 同一噪声消息重复 %d 次（>%d）: %s", pod.Name, container, n, maxRepeat, truncateForLog(msg, 300))
			}
		}
	}
	if !found {
		return fmt.Errorf("namespace %s 中未找到容器 %s", namespace, container)
	}
	return nil
}

func isLogNoiseCandidate(norm, lower string) bool {
	if strings.Contains(lower, "traceback") || strings.Contains(norm, "Panic") {
		return true
	}
	if strings.Contains(lower, "/health") || strings.Contains(lower, "/metrics") {
		return true
	}
	return false
}

func podHasNamedContainer(pod *corev1.Pod, container string) bool {
	for _, c := range pod.Spec.Containers {
		if c.Name == container {
			return true
		}
	}
	return false
}

func assertExternalPrefixCacheHit(namespace string) error {
	roles := []struct {
		role      string
		container string
	}{
		{infernexutils.PrefillPDRole, infernexutils.PrefillEngineContainer},
		{infernexutils.DecodePDRole, infernexutils.DecodeEngineContainer},
	}
	hitRe := regexp.MustCompile(`(?i)External prefix cache hit rate:\s*([0-9.]+)`)
	var snippets []string
	var latestRate float64
	sawLine := false
	for _, r := range roles {
		pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: infernexutils.PDRoleLabelKey + "=" + r.role,
		})
		if err != nil {
			return err
		}
		for i := range pods.Items {
			pod := &pods.Items[i]
			if !podHasNamedContainer(pod, r.container) {
				continue
			}
			raw, logErr := clientset.CoreV1().Pods(namespace).GetLogs(pod.Name, &corev1.PodLogOptions{Container: r.container}).Do(ctx).Raw()
			if logErr != nil {
				return logErr
			}
			text := string(raw)
			snippets = append(snippets, fmt.Sprintf("%s/%s len=%d", pod.Name, r.container, len(text)))
			matches := hitRe.FindAllStringSubmatch(text, -1)
			if len(matches) == 0 {
				continue
			}
			sawLine = true
			last := matches[len(matches)-1]
			var rate float64
			if _, scanErr := fmt.Sscanf(last[1], "%f", &rate); scanErr != nil {
				continue
			}
			latestRate = rate
			GinkgoWriter.Printf("FIT_45: %s 最近一次 external prefix cache hit rate=%s\n", pod.Name, last[1])
			if rate > 0 {
				return nil
			}
		}
	}
	if sawLine {
		return fmt.Errorf("第二套引擎 External prefix cache hit rate 最近一次为 %.4f，期望 >0（%s）", latestRate, strings.Join(snippets, "; "))
	}
	return fmt.Errorf("未在第二套 prefill/decode 日志中找到 External prefix cache hit（%s）", strings.Join(snippets, "; "))
}

func truncateForLog(s string, n int) string {
	s = strings.TrimSpace(s)
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}

// helmInstallOCIWithSets 使用 helm install 部署 chart（支持 OCI 和本地 tgz），支持额外 --set 参数。
func helmInstallOCIWithSets(releaseName, chartURL, version, namespace string, setValues []string) error {
	args := []string{"install", releaseName, chartURL}
	if !strings.HasPrefix(chartURL, "/") && !strings.HasPrefix(chartURL, ".") && !strings.HasSuffix(chartURL, ".tgz") {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	args = append(args, "-n", namespace, "--create-namespace")
	for _, sv := range setValues {
		args = append(args, "--set", sv)
	}

	GinkgoWriter.Printf("执行: helm %s\n", strings.Join(args, " "))

	cmd := exec.Command("helm", args...)
	cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	GinkgoWriter.Printf("helm install 成功: %s\n", stdout.String())
	return nil
}

// helmUninstall 卸载 helm release
func helmUninstall(releaseName, namespace string) error {
	run := func(noHooks bool) error {
		args := []string{"uninstall", releaseName, "-n", namespace}
		if noHooks {
			args = append(args, "--no-hooks")
		}
		cmd := exec.Command("helm", args...)
		cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
		var stderr bytes.Buffer
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return fmt.Errorf("helm uninstall %s/%s: %w stderr: %s", namespace, releaseName, err, stderr.String())
		}
		return nil
	}
	if err := run(false); err != nil {
		GinkgoWriter.Printf("helm uninstall 带 hook 失败，改用 --no-hooks: %v\n", err)
		return run(true)
	}
	return nil
}

// helmInstallOCIWithValuesFile 使用 helm install 部署 chart（支持 OCI 和本地 tgz），通过 -f 指定 values 文件。
func helmInstallOCIWithValuesFile(releaseName, chartURL, version, namespace, valuesFile string) error {
	args := []string{"install", releaseName, chartURL}
	if !strings.HasPrefix(chartURL, "/") && !strings.HasPrefix(chartURL, ".") && !strings.HasSuffix(chartURL, ".tgz") {
		v := strings.TrimSpace(version)
		if v != "" && !strings.EqualFold(v, "latest") {
			args = append(args, "--version", v)
		}
	}
	args = append(args, "-n", namespace, "--create-namespace", "-f", valuesFile)

	GinkgoWriter.Printf("执行: helm %s\n", strings.Join(args, " "))

	cmd := exec.Command("helm", args...)
	cmd.Env = append(os.Environ(), "HELM_DRIVER=configmap")
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("helm install failed: %w\nstdout: %s\nstderr: %s", err, stdout.String(), stderr.String())
	}
	GinkgoWriter.Printf("helm install 成功: %s\n", stdout.String())
	return nil
}

// helmUninstallIfExists 若 release 存在则卸载
func helmUninstallIfExists(releaseName, namespace string) error {
	if !infernexutils.HelmReleaseExists(releaseName, namespace) {
		return nil
	}
	GinkgoWriter.Printf("发现已有 release %s，正在卸载...\n", releaseName)
	return helmUninstall(releaseName, namespace)
}

// infernexSharedNamespaces infernex chart 创建的共享 namespace：清场时删除，FIT_3 卸载后校验本 release 无残留。
var infernexSharedNamespaces = []string{"nats", "eagle-eye", "istio-system", "scaling-system"}

func deployK8sAdapter() infernexutils.K8SClientInterface {
	return &infernexutils.ClientsetK8SAdapter{Clientset: clientset}
}

// ensureDeploySuiteCleanSlate 套件 BeforeAll：卸载相关命名空间内全部 helm release 并做完整清场。
func ensureDeploySuiteCleanSlate(baseReleaseName, primaryNamespace string) {
	GinkgoWriter.Println("套件 BeforeAll：清场开始...")
	for _, ns := range []string{primaryNamespace, "test-inference"} {
		releases, err := infernexutils.ListHelmReleasesInNamespaceIfExists(ctx, deployK8sAdapter(), ns)
		Expect(err).NotTo(HaveOccurred())
		for _, rel := range releases {
			err := helmUninstallIfExists(rel, ns)
			Expect(err).NotTo(HaveOccurred())
		}
		infernexutils.CleanupRSGScaledResources(clientset, dynamicClient, ctx, ns)
	}
	finalizeInfernexUninstall(baseReleaseName, primaryNamespace)

	// 清理残留的集群级资源（ClusterRole/ClusterRoleBinding/Webhook），这些资源在 helm install 失败后不会被自动清理
	cleanupOrphanedClusterScopedResources()

	// 强制删除所有共享命名空间（不管是否有 helm 注解）
	for _, ns := range infernexSharedNamespaces {
		if infernexutils.NamespaceExists(ctx, deployK8sAdapter(), ns) {
			GinkgoWriter.Printf("主动删除共享 namespace: %s\n", ns)
			_ = clientset.CoreV1().Namespaces().Delete(ctx, ns, metav1.DeleteOptions{})
		}
	}

	GinkgoWriter.Println("套件 BeforeAll：清场完成")
}

// cleanupOrphanedClusterScopedResources 删除所有带有 infernex 相关 helm 注解的集群级资源。
// 这些资源在 helm install 失败后不会被自动清理，会导致后续用例的 helm install 失败。
func cleanupOrphanedClusterScopedResources() {
	// infernex 相关的 release 名前缀列表（包括其他用户可能使用的变体）
	releasePrefixes := []string{"infernex", "kzj-infer"}

	// 判断 release 名是否是 infernex 相关的
	isInfernexRelease := func(relName string) bool {
		for _, prefix := range releasePrefixes {
			if strings.HasPrefix(relName, prefix) {
				return true
			}
		}
		return false
	}

	// 删除 ClusterRole
	crList, err := clientset.RbacV1().ClusterRoles().List(ctx, metav1.ListOptions{LabelSelector: "app.kubernetes.io/managed-by=Helm"})
	if err == nil {
		for _, cr := range crList.Items {
			if relName, ok := cr.Annotations["meta.helm.sh/release-name"]; ok && isInfernexRelease(relName) {
				GinkgoWriter.Printf("清理残留 ClusterRole: %s (release=%s)\n", cr.Name, relName)
				_ = clientset.RbacV1().ClusterRoles().Delete(ctx, cr.Name, metav1.DeleteOptions{})
			}
		}
	}

	// 删除 ClusterRoleBinding
	crbList, err := clientset.RbacV1().ClusterRoleBindings().List(ctx, metav1.ListOptions{LabelSelector: "app.kubernetes.io/managed-by=Helm"})
	if err == nil {
		for _, crb := range crbList.Items {
			if relName, ok := crb.Annotations["meta.helm.sh/release-name"]; ok && isInfernexRelease(relName) {
				GinkgoWriter.Printf("清理残留 ClusterRoleBinding: %s (release=%s)\n", crb.Name, relName)
				_ = clientset.RbacV1().ClusterRoleBindings().Delete(ctx, crb.Name, metav1.DeleteOptions{})
			}
		}
	}

	// 删除 ValidatingWebhookConfiguration
	vwcList, err := clientset.AdmissionregistrationV1().ValidatingWebhookConfigurations().List(ctx, metav1.ListOptions{LabelSelector: "app.kubernetes.io/managed-by=Helm"})
	if err == nil {
		for _, vwc := range vwcList.Items {
			if relName, ok := vwc.Annotations["meta.helm.sh/release-name"]; ok && isInfernexRelease(relName) {
				GinkgoWriter.Printf("清理残留 ValidatingWebhookConfiguration: %s (release=%s)\n", vwc.Name, relName)
				_ = clientset.AdmissionregistrationV1().ValidatingWebhookConfigurations().Delete(ctx, vwc.Name, metav1.DeleteOptions{})
			}
		}
	}

	// 删除 MutatingWebhookConfiguration
	mwcList, err := clientset.AdmissionregistrationV1().MutatingWebhookConfigurations().List(ctx, metav1.ListOptions{LabelSelector: "app.kubernetes.io/managed-by=Helm"})
	if err == nil {
		for _, mwc := range mwcList.Items {
			if relName, ok := mwc.Annotations["meta.helm.sh/release-name"]; ok && isInfernexRelease(relName) {
				GinkgoWriter.Printf("清理残留 MutatingWebhookConfiguration: %s (release=%s)\n", mwc.Name, relName)
				_ = clientset.AdmissionregistrationV1().MutatingWebhookConfigurations().Delete(ctx, mwc.Name, metav1.DeleteOptions{})
			}
		}
	}

	// 删除残留的 CRD 资源（ElasticScaler、LeaderWorkerSet、ResourceScalingGroup 等）
	crdGVRs := []schema.GroupVersionResource{
		{Group: "elasticscaler.io", Version: "v1", Resource: "elasticscalers"},
		{Group: "leaderworkerset.x-k8s.io", Version: "v1", Resource: "leaderworkersets"},
		{Group: "autoscaling.openfuyao.com", Version: "v1alpha1", Resource: "resourcescalinggroups"},
		{Group: "networking.istio.io", Version: "v1", Resource: "destinationrules"},
		{Group: "networking.istio.io", Version: "v1", Resource: "envoyfilters"},
		{Group: "gateway.networking.k8s.io", Version: "v1", Resource: "gateways"},
		{Group: "gateway.networking.k8s.io", Version: "v1", Resource: "httproutes"},
		{Group: "inference.networking.k8s.io", Version: "v1", Resource: "inferencepools"},
	}
	for _, gvr := range crdGVRs {
		resources, err := dynamicClient.Resource(gvr).List(ctx, metav1.ListOptions{LabelSelector: "app.kubernetes.io/managed-by=Helm"})
		if err != nil {
			continue
		}
		for _, item := range resources.Items {
			relName, ok := item.GetAnnotations()["meta.helm.sh/release-name"]
			if !ok || !isInfernexRelease(relName) {
				continue
			}
			GinkgoWriter.Printf("清理残留 %s: %s/%s (release=%s)\n", gvr.Resource, item.GetNamespace(), item.GetName(), relName)
			_ = dynamicClient.Resource(gvr).Namespace(item.GetNamespace()).Delete(ctx, item.GetName(), metav1.DeleteOptions{})
		}
	}
}

// infernexDeferCleanupWait 用例结束后等待 helm 卸载触发的资源删除（部分组件较慢）。
const infernexDeferCleanupWait = 1 * time.Minute

// deferCleanupInfernexRelease 用例 DeferCleanup：helm uninstall（若存在）→ 等待 → 清理 RSG 扩缩容残留 → 清理共享命名空间。
// 预计 install 失败的用例通常无 release，helmUninstallIfExists 会直接跳过。
func deferCleanupInfernexRelease(releaseName, namespace string) {
	GinkgoWriter.Printf("DeferCleanup: helm uninstall %s/%s（若存在）\n", namespace, releaseName)
	if err := helmUninstallIfExists(releaseName, namespace); err != nil {
		GinkgoWriter.Printf("DeferCleanup: helm uninstall 失败: %v\n", err)
	}

	GinkgoWriter.Printf("DeferCleanup: 等待 %v，以便组件完成删除...\n", infernexDeferCleanupWait)
	time.Sleep(infernexDeferCleanupWait)

	GinkgoWriter.Println("DeferCleanup: 清理 RSG 扩缩容残留...")
	infernexutils.CleanupRSGScaledResources(clientset, dynamicClient, ctx, namespace)

	// 清理共享命名空间，避免影响后续用例
	for _, ns := range infernexSharedNamespaces {
		if infernexutils.NamespaceExists(ctx, deployK8sAdapter(), ns) && isHelmOwnedNamespace(ns) {
			GinkgoWriter.Printf("DeferCleanup: 删除 helm 管理的共享 namespace: %s\n", ns)
			if delErr := clientset.CoreV1().Namespaces().Delete(ctx, ns, metav1.DeleteOptions{}); delErr != nil {
				GinkgoWriter.Printf("DeferCleanup: 删除 namespace %s 失败（可能已在终止中）: %v\n", ns, delErr)
			}
		}
	}
}

// finalizeInfernexUninstall 等待 release 卸载并清理 infernex 关联资源（仅用于装前/装后清场，不参与 FIT_3 断言）。
func finalizeInfernexUninstall(releaseName, namespace string) {
	Eventually(func() bool {
		return infernexutils.HelmReleaseExists(releaseName, namespace)
	}, 2*time.Minute, 5*time.Second).Should(BeFalse(), "等待 release %s 卸载完成超时", releaseName)

	uninstallOrphanedReleasesOnSharedNamespaces(releaseName)
	cleanupOrphanedInferenceDeployments(namespace)

	// 主动删除仍残留的 helm 管理的共享 namespace
	for _, ns := range infernexSharedNamespaces {
		if infernexutils.NamespaceExists(ctx, deployK8sAdapter(), ns) && isHelmOwnedNamespace(ns) {
			GinkgoWriter.Printf("主动删除 helm 管理的共享 namespace: %s\n", ns)
			if delErr := clientset.CoreV1().Namespaces().Delete(ctx, ns, metav1.DeleteOptions{}); delErr != nil {
				GinkgoWriter.Printf("删除 namespace %s 失败（可能已在终止中）: %v\n", ns, delErr)
			}
		}
	}

	// 等待共享 namespace 消失（仅等待带有 helm release annotation 的，跳过无关 namespace）
	Eventually(func() []string {
		var remaining []string
		for _, ns := range infernexSharedNamespaces {
			if infernexutils.NamespaceExists(ctx, deployK8sAdapter(), ns) && isHelmOwnedNamespace(ns) {
				remaining = append(remaining, ns)
			}
		}
		return remaining
	}, 5*time.Minute, 5*time.Second).Should(BeEmpty(), "以下共享 namespace 未被正确清理")
	GinkgoWriter.Println("初始环境已清理干净")
}

// verifyHelmUninstallResourceCleanup 校验 helm uninstall 后 release 与各共享命名空间无残留（RSG 扩缩容资源在 release ns 可保留）。
func verifyHelmUninstallResourceCleanup(releaseNamespace, helmReleaseName string) error {
	if infernexutils.HelmReleaseExists(helmReleaseName, releaseNamespace) {
		return fmt.Errorf("helm release %s 仍存在", helmReleaseName)
	}
	if err := verifyReleaseNamespaceNoLeftover(releaseNamespace); err != nil {
		return fmt.Errorf("release 命名空间 %s: %w", releaseNamespace, err)
	}
	for _, ns := range infernexSharedNamespaces {
		if !infernexutils.NamespaceExists(ctx, deployK8sAdapter(), ns) {
			continue
		}
		if err := verifySharedNamespaceNoHelmReleaseLeftover(ns, helmReleaseName); err != nil {
			return fmt.Errorf("共享命名空间 %s: %w", ns, err)
		}
	}
	return nil
}

// verifyReleaseNamespaceNoLeftover 检查 release 部署命名空间：除带 RSG 标签的扩缩容资源外不应有工作负载残留。
func verifyReleaseNamespaceNoLeftover(namespace string) error {
	var leftovers []string

	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods: %w", err)
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if infernexutils.WorkloadHasRSGLabels(clientset, ctx, namespace, pod) {
			continue
		}
		leftovers = append(leftovers, fmt.Sprintf("Pod/%s", pod.Name))
	}

	deploys, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list deployments: %w", err)
	}
	for i := range deploys.Items {
		dep := &deploys.Items[i]
		if infernexutils.DeploymentHasRSGLabels(dep) {
			continue
		}
		leftovers = append(leftovers, fmt.Sprintf("Deployment/%s", dep.Name))
	}

	stsList, err := clientset.AppsV1().StatefulSets(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list statefulsets: %w", err)
	}
	for i := range stsList.Items {
		sts := &stsList.Items[i]
		if infernexutils.StatefulSetHasRSGLabels(sts) {
			continue
		}
		leftovers = append(leftovers, fmt.Sprintf("StatefulSet/%s", sts.Name))
	}

	dsList, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list daemonsets: %w", err)
	}
	for i := range dsList.Items {
		ds := &dsList.Items[i]
		if infernexutils.DaemonSetHasRSGLabels(ds) {
			continue
		}
		leftovers = append(leftovers, fmt.Sprintf("DaemonSet/%s", ds.Name))
	}

	svcs, err := clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list services: %w", err)
	}
	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Name == "kubernetes" {
			continue
		}
		if infernexutils.HasRSGLabels(svc.Labels) {
			continue
		}
		leftovers = append(leftovers, fmt.Sprintf("Service/%s", svc.Name))
	}

	if dynamicClient != nil {
		if err := infernexutils.EnsureLeaderWorkerSetAPI(clientset); err == nil {
			lwsList, lwsErr := dynamicClient.Resource(esutils.LeaderWorkerSetGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
			if lwsErr != nil {
				return fmt.Errorf("list LeaderWorkerSet: %w", lwsErr)
			}
			for _, item := range lwsList.Items {
				if infernexutils.HasRSGLabels(item.GetLabels()) {
					continue
				}
				leftovers = append(leftovers, fmt.Sprintf("LeaderWorkerSet/%s", item.GetName()))
			}
		}
	}

	if len(leftovers) > 0 {
		return fmt.Errorf("仍存在残留资源: %v", leftovers)
	}
	return nil
}

// verifySharedNamespaceNoHelmReleaseLeftover 检查共享命名空间内是否仍有本 helm release 安装的资源。
func verifySharedNamespaceNoHelmReleaseLeftover(namespace, helmReleaseName string) error {
	var leftovers []string

	pods, err := clientset.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list pods: %w", err)
	}
	for i := range pods.Items {
		pod := &pods.Items[i]
		if infernexutils.ResourceBelongsToHelmRelease(pod.Labels, pod.Annotations, helmReleaseName) {
			leftovers = append(leftovers, fmt.Sprintf("Pod/%s", pod.Name))
		}
	}

	deploys, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list deployments: %w", err)
	}
	for i := range deploys.Items {
		dep := &deploys.Items[i]
		if infernexutils.ResourceBelongsToHelmRelease(dep.Labels, dep.Annotations, helmReleaseName) {
			leftovers = append(leftovers, fmt.Sprintf("Deployment/%s", dep.Name))
		}
	}

	stsList, err := clientset.AppsV1().StatefulSets(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list statefulsets: %w", err)
	}
	for i := range stsList.Items {
		sts := &stsList.Items[i]
		if infernexutils.ResourceBelongsToHelmRelease(sts.Labels, sts.Annotations, helmReleaseName) {
			leftovers = append(leftovers, fmt.Sprintf("StatefulSet/%s", sts.Name))
		}
	}

	dsList, err := clientset.AppsV1().DaemonSets(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list daemonsets: %w", err)
	}
	for i := range dsList.Items {
		ds := &dsList.Items[i]
		if infernexutils.ResourceBelongsToHelmRelease(ds.Labels, ds.Annotations, helmReleaseName) {
			leftovers = append(leftovers, fmt.Sprintf("DaemonSet/%s", ds.Name))
		}
	}

	svcs, err := clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return fmt.Errorf("list services: %w", err)
	}
	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Name == "kubernetes" {
			continue
		}
		if infernexutils.ResourceBelongsToHelmRelease(svc.Labels, svc.Annotations, helmReleaseName) {
			leftovers = append(leftovers, fmt.Sprintf("Service/%s", svc.Name))
		}
	}

	if dynamicClient != nil {
		rsgList, rsgErr := dynamicClient.Resource(esutils.ResourceScalingGroupGVR).Namespace(namespace).List(ctx, metav1.ListOptions{})
		if rsgErr != nil && !apierrors.IsNotFound(rsgErr) {
			return fmt.Errorf("list ResourceScalingGroup: %w", rsgErr)
		}
		if rsgErr == nil {
			for _, item := range rsgList.Items {
				if infernexutils.ResourceBelongsToHelmRelease(item.GetLabels(), item.GetAnnotations(), helmReleaseName) {
					leftovers = append(leftovers, fmt.Sprintf("ResourceScalingGroup/%s", item.GetName()))
				}
			}
		}
	}

	if len(leftovers) > 0 {
		return fmt.Errorf("仍存在本 release 的残留资源: %v", leftovers)
	}
	return nil
}

// uninstallOrphanedReleasesOnSharedNamespaces 检查共享 namespace 的 helm ownership 注解，
// 若发现由其他 release 管理（前次运行残留），则先卸载该 release。
func uninstallOrphanedReleasesOnSharedNamespaces(currentReleaseName string) {
	seen := map[string]bool{}
	for _, nsName := range infernexSharedNamespaces {
		ns, err := clientset.CoreV1().Namespaces().Get(ctx, nsName, metav1.GetOptions{})
		if err != nil {
			continue
		}
		ann := ns.Annotations
		if ann == nil {
			continue
		}
		ownerRelease := ann["meta.helm.sh/release-name"]
		ownerNS := ann["meta.helm.sh/release-namespace"]
		if ownerRelease == "" || ownerRelease == currentReleaseName {
			continue
		}
		key := ownerNS + "/" + ownerRelease
		if seen[key] {
			continue
		}
		seen[key] = true
		if infernexutils.HelmReleaseExists(ownerRelease, ownerNS) {
			GinkgoWriter.Printf("发现共享 namespace %s 由残留 release %s/%s 管理，正在卸载...\n",
				nsName, ownerNS, ownerRelease)
			_ = helmUninstall(ownerRelease, ownerNS)
		}
	}
}

// isHelmOwnedNamespace 检查 namespace 是否带有 helm release annotation（即由 helm 管理）。
func isHelmOwnedNamespace(name string) bool {
	ns, err := clientset.CoreV1().Namespaces().Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return false
	}
	_, ok := ns.Annotations["meta.helm.sh/release-name"]
	return ok
}

// cleanupOrphanedInferenceDeployments 删除目标 namespace 中由 RSG 控制器创建但未被清理的 Deployment。
// helm uninstall 会同时删除 RSG CR 和 RSG 控制器，控制器可能在处理 CR 删除前就已终止，
// 导致其创建的 Deployment（如 pd-vllm-*）成为孤儿资源。
func cleanupOrphanedInferenceDeployments(namespace string) {
	deploys, err := clientset.AppsV1().Deployments(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=inference-backend",
	})
	if err != nil {
		return
	}
	for i := range deploys.Items {
		d := &deploys.Items[i]
		if deploymentHasRSGLabels(d) {
			GinkgoWriter.Printf("跳过 RSG 扩缩容 deployment: %s\n", d.Name)
			continue
		}
		GinkgoWriter.Printf("清理残留的 inference-backend deployment: %s\n", d.Name)
		propagation := metav1.DeletePropagationBackground
		_ = clientset.AppsV1().Deployments(namespace).Delete(ctx, d.Name, metav1.DeleteOptions{
			PropagationPolicy: &propagation,
		})
	}
}

// --- 推理请求校验辅助函数 ---

// defaultModelName 返回默认推理模型名称，优先从环境变量 HERMES_INFERENCE_MODEL 读取
func defaultModelName() string {
	if m := os.Getenv("HERMES_INFERENCE_MODEL"); m != "" {
		return m
	}
	return "Qwen/Qwen3-8B"
}

// verifyInferenceRequest 在指定命名空间中解析推理服务 URL，发送 /v1/chat/completions 请求并校验响应
func verifyInferenceRequest(targetNS, model string) error {
	baseURL, err := resolveInferenceURL(targetNS)
	if err != nil {
		return fmt.Errorf("解析推理服务 URL 失败: %w", err)
	}

	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"

	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": "Hello! How are you today"},
		},
		"stream": false,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求体失败: %w", err)
	}

	GinkgoWriter.Printf("发送推理请求: POST %s (model=%s)\n", fullURL, model)

	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送请求失败: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	GinkgoWriter.Printf("响应状态码: %d\n", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("期望状态码 200，实际 %d，响应: %s", resp.StatusCode, string(respBody))
	}

	var respJSON map[string]any
	if err := json.Unmarshal(respBody, &respJSON); err != nil {
		return fmt.Errorf("解析响应 JSON 失败: %w，响应: %s", err, string(respBody))
	}
	if _, ok := respJSON["choices"]; !ok {
		return fmt.Errorf("响应不包含 'choices' 字段: %s", string(respBody))
	}

	GinkgoWriter.Println("推理请求校验成功，响应包含有效的 choices")
	return nil
}

// verifyNonStreamInferenceRequest 发送非流式推理请求并详细校验响应内容（choices content、usage、model 字段）
func verifyNonStreamInferenceRequest(targetNS, model string) error {
	baseURL, err := resolveInferenceURL(targetNS)
	if err != nil {
		return fmt.Errorf("解析推理服务 URL 失败: %w", err)
	}

	fullURL := strings.TrimSuffix(baseURL, "/") + "/v1/chat/completions"

	reqBody := map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "user", "content": "Hello! How are you today"},
		},
		"stream": false,
	}

	bodyBytes, err := json.Marshal(reqBody)
	if err != nil {
		return fmt.Errorf("序列化请求体失败: %w", err)
	}

	GinkgoWriter.Printf("发送非流式推理请求: POST %s (model=%s, stream=false)\n", fullURL, model)

	client := &http.Client{Timeout: 120 * time.Second}
	req, err := http.NewRequest(http.MethodPost, fullURL, bytes.NewReader(bodyBytes))
	if err != nil {
		return fmt.Errorf("创建请求失败: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := client.Do(req)
	if err != nil {
		return fmt.Errorf("发送请求失败: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("读取响应失败: %w", err)
	}

	GinkgoWriter.Printf("非流式响应状态码: %d\n", resp.StatusCode)

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("期望状态码 200，实际 %d，响应: %s", resp.StatusCode, string(respBody))
	}

	var respJSON map[string]any
	if err := json.Unmarshal(respBody, &respJSON); err != nil {
		return fmt.Errorf("解析响应 JSON 失败: %w，响应: %s", err, string(respBody))
	}

	// 校验 choices 字段
	choicesRaw, ok := respJSON["choices"]
	if !ok {
		return fmt.Errorf("响应不包含 'choices' 字段: %s", string(respBody))
	}
	choices, ok := choicesRaw.([]any)
	if !ok || len(choices) == 0 {
		return fmt.Errorf("'choices' 为空或格式错误: %v", choicesRaw)
	}

	// 校验 choices[0].message.content 非空
	firstChoice, ok := choices[0].(map[string]any)
	if !ok {
		return fmt.Errorf("choices[0] 格式错误: %v", choices[0])
	}
	msgRaw, ok := firstChoice["message"]
	if !ok {
		return fmt.Errorf("choices[0] 不包含 'message' 字段: %v", firstChoice)
	}
	msg, ok := msgRaw.(map[string]any)
	if !ok {
		return fmt.Errorf("choices[0].message 格式错误: %v", msgRaw)
	}
	content, _ := msg["content"].(string)
	if strings.TrimSpace(content) == "" {
		return fmt.Errorf("choices[0].message.content 为空，响应: %s", string(respBody))
	}

	// 校验 usage 字段
	if _, ok := respJSON["usage"]; !ok {
		return fmt.Errorf("响应不包含 'usage' 字段: %s", string(respBody))
	}
	usage, ok := respJSON["usage"].(map[string]any)
	if !ok {
		return fmt.Errorf("'usage' 格式错误: %v", respJSON["usage"])
	}
	for _, field := range []string{"prompt_tokens", "completion_tokens", "total_tokens"} {
		if _, exists := usage[field]; !exists {
			return fmt.Errorf("usage 不包含 '%s' 字段: %v", field, usage)
		}
	}

	// 校验 id 字段
	if _, ok := respJSON["id"]; !ok {
		return fmt.Errorf("响应不包含 'id' 字段: %s", string(respBody))
	}

	GinkgoWriter.Printf("非流式推理请求校验通过: content长度=%d, usage=%v\n", len(content), usage)
	return nil
}

// resolveInferenceURL 在指定命名空间中解析推理服务的基础 URL（scheme://host:port）
func resolveInferenceURL(namespace string) (string, error) {
	// 优先使用环境变量指定的网关地址
	if gw := os.Getenv("HERMES_GATEWAY_URL"); gw != "" {
		base := gw
		if idx := strings.Index(gw, "/v1/"); idx != -1 {
			base = gw[:idx]
		}
		return strings.TrimSuffix(base, "/"), nil
	}

	// 从 K8s Service 中自动探测
	svcs, err := clientset.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		return "", fmt.Errorf("list services in %s: %w", namespace, err)
	}

	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Name == "inference-gateway-istio" {
			if url := buildServiceURL(svc); url != "" {
				return url, nil
			}
		}
	}

	// 优先查找 inference-gateway Service (通过 label)
	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Labels["app.kubernetes.io/name"] == "inference-gateway" {
			if url := buildServiceURL(svc); url != "" {
				return url, nil
			}
		}
	}

	// 回退查找 inference-backend Service (通过 label)
	for i := range svcs.Items {
		svc := &svcs.Items[i]
		if svc.Labels["app.kubernetes.io/name"] == "inference-backend" {
			if url := buildServiceURL(svc); url != "" {
				return url, nil
			}
		}
	}

	return "", fmt.Errorf("在命名空间 %s 中未找到推理网关或后端 Service", namespace)
}

// buildServiceURL 根据 Service 类型构建访问 URL。
// 优先选择业务端口（排除 status-port、health 等非业务端口）。
func buildServiceURL(svc *corev1.Service) string {
	skipPortNames := map[string]bool{"status-port": true, "health": true, "healthz": true, "readiness": true, "metrics": true, "monitor": true}

	// 第一轮：跳过非业务端口
	for _, sp := range svc.Spec.Ports {
		if skipPortNames[strings.ToLower(sp.Name)] {
			continue
		}
		if url := buildServiceURLForPort(svc, sp); url != "" {
			return url
		}
	}
	// 第二轮：兜底，返回第一个能构建的端口
	for _, sp := range svc.Spec.Ports {
		if url := buildServiceURLForPort(svc, sp); url != "" {
			return url
		}
	}
	return ""
}

func buildServiceURLForPort(svc *corev1.Service, sp corev1.ServicePort) string {
	scheme := "http"
	if sp.Port == 443 || strings.Contains(strings.ToLower(sp.Name), "https") {
		scheme = "https"
	}

	switch svc.Spec.Type {
	case corev1.ServiceTypeNodePort:
		if sp.NodePort != 0 {
			nodeIP, err := infernexutils.ChooseNodeIPForNodePort(ctx, &infernexutils.ClientsetK8SAdapter{Clientset: clientset})
			if err == nil && nodeIP != "" {
				return fmt.Sprintf("%s://%s:%d", scheme, nodeIP, sp.NodePort)
			}
		}
	case corev1.ServiceTypeLoadBalancer:
		if len(svc.Status.LoadBalancer.Ingress) > 0 {
			host := svc.Status.LoadBalancer.Ingress[0].IP
			if host == "" {
				host = svc.Status.LoadBalancer.Ingress[0].Hostname
			}
			if host != "" {
				return fmt.Sprintf("%s://%s:%d", scheme, host, sp.Port)
			}
		}
	default:
		if svc.Spec.ClusterIP != "" && svc.Spec.ClusterIP != "None" {
			return fmt.Sprintf("%s://%s:%d", scheme, svc.Spec.ClusterIP, sp.Port)
		}
	}
	return ""
}

// deploymentHasRSGLabels 判断 Deployment 是否由 RSG (ResourceScalingGroup) 控制器管理。
func deploymentHasRSGLabels(d *appsv1.Deployment) bool {
	if d.Labels == nil {
		return false
	}
	_, ok := d.Labels["autoscaling.rsg.group"]
	return ok
}
