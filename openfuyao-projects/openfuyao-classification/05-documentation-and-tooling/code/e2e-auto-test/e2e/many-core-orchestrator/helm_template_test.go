package mco

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

var _ = Describe("众核 Helm Template 渲染测试", Label("many-core-orchestrator", "helm-template", "virtual-machine"), func() {
	var chartPath string

	BeforeEach(func() {
		chartPath = GetChartPath()
	})

	It("众核-KATA-IT-001 默认值渲染——全量资源", func() {
		args := append([]string{}, KataPhaseHelmSets...)
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template 默认配置（仅 kata 阶段）应正常返回")

		Expect(output).To(ContainSubstring("kind: DaemonSet"), "应包含 DaemonSet")
		Expect(strings.Count(output, "kind: DaemonSet")).To(BeNumerically(">=", 1), "至少 1 个 DaemonSet (kata-deploy)")

		Expect(output).To(ContainSubstring("kind: ConfigMap"), "应包含 ConfigMap (preflight)")
		Expect(strings.Count(output, "kind: ConfigMap")).To(BeNumerically(">=", 1), "至少 1 个 ConfigMap")

		Expect(output).To(ContainSubstring("kind: ServiceAccount"), "应包含 ServiceAccount")
		Expect(output).To(ContainSubstring("kind: ClusterRole"), "应包含 ClusterRole")
		Expect(output).To(ContainSubstring("kind: ClusterRoleBinding"), "应包含 ClusterRoleBinding")

		Expect(output).To(ContainSubstring("kind: RuntimeClass"), "应包含 RuntimeClass")
		Expect(output).To(ContainSubstring("kata-qemu"), "应包含 kata-qemu RuntimeClass")
		Expect(output).To(ContainSubstring("kata-preflight"), "DaemonSet 中应可见 kata-preflight initContainer")
	})

	It("众核-KATA-IT-002 关闭 kata-deploy 子 chart", func() {
		args := append([]string{}, KataPhaseHelmSets...)
		args = append(args, "--set", "kata-deploy.enabled=false")
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template 关闭 kata-deploy 子 chart 应正常返回")

		Expect(output).NotTo(ContainSubstring("kata-deploy"), "不应包含 kata-deploy 关键字")
		Expect(output).NotTo(ContainSubstring("kind: RuntimeClass"), "不应包含 RuntimeClass")
	})

	It("众核-KATA-IT-003 关闭 preflight", func() {
		args := append([]string{}, KataPhaseHelmSets...)
		args = append(args, "--set", "kata-deploy.kataPreflight.enabled=false")
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template 关闭 preflight 应正常返回")

		Expect(output).NotTo(ContainSubstring("kata-preflight"), "不应包含 preflight initContainer 名称")
		Expect(output).NotTo(ContainSubstring("kataPreflight"), "不应包含 kataPreflight 相关名称")
		Expect(output).NotTo(ContainSubstring("/dev/kvm"), "不应包含 /dev/kvm volume 挂载")

		Expect(output).To(ContainSubstring("kind: DaemonSet"), "主容器 DaemonSet 应仍存在")
		Expect(output).To(ContainSubstring("kind: RuntimeClass"), "RuntimeClass 应仍存在")
	})

	It("众核-KATA-IT-004 禁用单个 shim", func() {
		args := append([]string{}, KataPhaseHelmSets...)
		args = append(args, "--set", "kata-deploy.shims.qemu-runtime-rs.enabled=false")
		output, err := HelmTemplateLocal(ctx, chartPath, args...)
		Expect(err).NotTo(HaveOccurred(), "helm template 禁用单个 shim 应正常返回")

		Expect(output).NotTo(ContainSubstring("kata-qemu-runtime-rs"), "不应包含禁用 shim 的 RuntimeClass 名称")
		Expect(output).To(ContainSubstring("kind: RuntimeClass"), "仍应有 RuntimeClass (其他 shim)")
		Expect(output).To(ContainSubstring("kind: DaemonSet"), "DaemonSet 应仍存在")
	})
})
