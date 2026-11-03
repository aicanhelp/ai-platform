package infernex_eagle_eye

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	ieutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/eagle-eye/utils"
)

var _ = Describe("Infernex optional eagle-eye observability", Ordered, Label("infernex-eagle-eye", "npu"), func() {
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
		GinkgoWriter.Printf("infernex install: chartURL=%q version=%q release=%q ns=%q\n",
			chartURL, chartVersion, releaseName, namespace)

		err := ieutils.EnsureNamespace(ctx, clientset, namespace)
		Expect(err).NotTo(HaveOccurred(), "确保命名空间 %q 存在", namespace)
	})

	AfterAll(func() {
		if releaseName != "" && namespace != "" {
			_ = ieutils.HelmUninstallIfExists(releaseName, namespace)
		}
		if namespace != "" {
			err := ieutils.DeleteNamespace(ctx, clientset, namespace)
			Expect(err).NotTo(HaveOccurred(), "清理命名空间 %q", namespace)
		}
		// InferNex 子 chart 会创建独立命名空间；只删 ai-inference 会留下 eagle-eye/nats。
		for _, ns := range []string{ieutils.NSEagleEye, ieutils.NSNats} {
			_ = ieutils.DeleteNamespace(ctx, clientset, ns)
		}
	})

	It("不装eagle-eye", func() {
		Expect(clientset).NotTo(BeNil())

		err := ieutils.HelmUninstallIfExists(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred())

		err = ieutils.HelmInstallChart(releaseName, chartURL, chartVersion, namespace, false)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			ok, name, e := ieutils.EagleEyeHardwarePodsAbsent(ctx, clientset)
			if e != nil {
				return e
			}
			if !ok {
				return fmt.Errorf("eagle-eye不应存在hardware-monitor/hardware-diagnosis Pod，但发现 %q", name)
			}
			return nil
		}, 12*time.Minute, 12*time.Second).Should(Succeed(), "eagle-eye命名空间不应存在hardware-monitor/hardware-diagnosis Pod")

		Eventually(func() error {
			ok, name, e := ieutils.NatsPodAbsent(ctx, clientset)
			if e != nil {
				return e
			}
			if !ok {
				return fmt.Errorf("nats命名空间不应存在nats Pod，但发现 %q", name)
			}
			return nil
		}, 12*time.Minute, 12*time.Second).Should(Succeed(), "nats命名空间不应存在nats Pod")

		Eventually(func() error {
			ok, name, e := ieutils.ObservabilityPodsAbsentInAiInference(ctx, clientset)
			if e != nil {
				return e
			}
			if !ok {
				return fmt.Errorf("ai-inference不应有prometheus-operator/node-exporter Pod等，但发现 %q", name)
			}
			return nil
		}, 12*time.Minute, 12*time.Second).Should(Succeed(), "ai-inference不应部署prometheus-operator、node-exporter等组件")

		err = ieutils.HelmUninstallIfExists(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
	})

	It("装eagle-eye", func() {
		Expect(clientset).NotTo(BeNil())

		hadExternalPromStack, err := ieutils.ClusterHasPrometheusStackInMonitoring(ctx, clientset)
		Expect(err).NotTo(HaveOccurred())

		err = ieutils.HelmUninstallIfExists(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred())

		err = ieutils.HelmInstallChart(releaseName, chartURL, chartVersion, namespace, true)
		Expect(err).NotTo(HaveOccurred())

		Eventually(func() error {
			return ieutils.EagleEyeHardwarePodsHealthy(ctx, clientset)
		}, 15*time.Minute, 15*time.Second).Should(Succeed(), "eagle-eye下hardware-monitor/hardware-diagnosis应Running且无CrashLoopBackOff等异常")

		Eventually(func() error {
			return ieutils.NatsPodHealthy(ctx, clientset)
		}, 15*time.Minute, 15*time.Second).Should(Succeed(), "nats命名空间下nats Pod应就绪")

		if hadExternalPromStack {
			Eventually(func() error {
				return ieutils.AiInferencePrometheusStackAbsent(ctx, clientset)
			}, 15*time.Minute, 15*time.Second).Should(Succeed(), "集群已有prometheus栈时，ai-inference不应再装prometheus-operator、node-exporter等组件")
		} else {
			Eventually(func() error {
				return ieutils.AiInferencePrometheusStackHealthy(ctx, clientset)
			}, 15*time.Minute, 15*time.Second).Should(Succeed(), "集群未预装prometheus栈时，ai-inference应部署prometheus-operator、node-exporter等组件且健康")
		}

		err = ieutils.HelmUninstallIfExists(releaseName, namespace)
		Expect(err).NotTo(HaveOccurred())
	})
})
