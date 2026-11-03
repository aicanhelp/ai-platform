package npu_dra_plugin

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("NPU DRA Plugin 设备挂载测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P0"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
		})

		It("昇腾-npu-dra-IT-002：基础 NPU 整卡设备挂载", func() {
			By("设置 npu-0 为 full 模式")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-it02"
			podName := "npu-pod02"

			By("创建 ResourceClaimTemplate 和 Pod")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("在 Pod 内执行 npu-smi info 验证设备可见")
			output := verifyNPUSMIInPod(podName)
			Expect(strings.Contains(output, "910") || strings.Contains(output, "310") || strings.Contains(output, "Ascend")).
				To(BeTrue(), "NPU 设备应当在 Pod 内可见")

			By("验证 Pod 使用整卡挂载（无 VNPU）")
			requireWholeCardInPod(podName)
			GinkgoWriter.Printf("IT-002 验证通过: 基础 NPU 整卡挂载正常\n")
		})

		It("昇腾-npu-dra-IT-003：Ascend 310P 芯片适配", func() {
			node := findNPUNode("310P")
			if node == "" {
				Skip("未找到 310P 节点")
			}

			claimTemplate := "npu-claim-template-it03"
			podName := "npu-pod03"

			By("创建 ResourceClaimTemplate 和 Pod")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("验证 npu-smi 输出包含 310P")
			output := verifyNPUSMIInPod(podName)
			Expect(output).To(ContainSubstring("310P"))
			GinkgoWriter.Printf("IT-003 验证通过: Ascend 310P 芯片适配正常\n")
		})

		It("昇腾-npu-dra-IT-004：Ascend 910C 芯片适配", func() {
			node := findNPUNode("910C")
			if node == "" {
				node = findNPUNode("Ascend910")
			}
			if node == "" {
				Skip("未找到 910C/Ascend910 节点")
			}

			claimTemplate := "npu-claim-template-it04"
			podName := "npu-pod04"

			By("创建 ResourceClaimTemplate 和 Pod")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("验证 npu-smi 输出包含 Ascend910")
			output := verifyNPUSMIInPod(podName)
			Expect(output).To(ContainSubstring("Ascend910"), "npu-smi 输出应包含 Ascend910")
			GinkgoWriter.Printf("IT-004 验证通过: Ascend 910C 芯片适配正常\n")
		})
	})
