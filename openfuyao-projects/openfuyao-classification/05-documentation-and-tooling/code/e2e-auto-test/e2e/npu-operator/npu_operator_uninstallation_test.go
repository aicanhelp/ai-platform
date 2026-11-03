package npu_operator

import (
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
	. "github.com/onsi/ginkgo/v2"
)

func UninstallationPhase() {
	var _ = Describe("Uninstall NPU Operator", Ordered, Label("npu_operator_uninstallation", "npu"), func() {
		BeforeEach(func() {
			// 执行 npu-operator 安装
			EnsureNPUOperatorInstallation(clientset, ctx, nodesNPUSituation)

			GinkgoWriter.Println()
		})

		It("149-npu-operator 插件卸载", func() {
			// 验证 npu-operator 插件卸载
			VerifyNPUOperatorUninstallation(clientset, ctx)
		})
	})
}
