package npu_dra_plugin

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("NPU DRA Plugin 多模式共存测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P1"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
		})

	It("昇腾-npu-dra-IT-019：三种模式同节点共存验证", func() {
		if findNPUNode("910B") == "" {
			Skip("未找到 910B 节点，本用例需 910B 芯片（hard 模式需 vir05 模板）")
		}
		// 需要 3 张卡分别配置 soft/hard/full
			// 配置: physicalId=0 soft, physicalId=1 hard, physicalId=2 full
			By("配置三卡不同模式: 0=soft, 1=hard, 2=full")
			config := mixedConfig(localNodeName,
				"    - physicalId: 0\n      vnpuMode: soft\n      schedulingPolicy: elastic",
				"    - physicalId: 1\n      vnpuMode: hard",
				"    - physicalId: 2\n      vnpuMode: full",
			)
			setNodeVnpuMode(localNodeName, config)

			By("创建 soft Pod (elastic, physicalID=0)")
			softClaim := "npu-claim-template-it19-soft"
			softPod := "npu-pod19-soft"
			createSoftClaimTemplate(softClaim, memCapacity16GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(softPod, softClaim, buildSoftVNpuPod)

			By("创建 hard Pod (hard, physicalID=1)")
			hardClaim := "npu-claim-template-it19-hard"
			hardPod := "npu-pod19-hard"
			createHardClaimTemplate(hardClaim, 5, 1, celPhysicalID1)
			createPodAndWaitRunning(hardPod, hardClaim, buildHardVNpuPod)

			By("创建 full Pod (full, physicalID=2)")
			fullClaim := "npu-claim-template-it19-full"
			fullPod := "npu-pod19-full"
			createFullClaimTemplate(fullClaim)
			createPodAndWaitRunning(fullPod, fullClaim, buildFullNPUPod)

		By("验证 soft Pod 有 enpu-monitor 配额")
		Expect(verifyEnpuMonitorInPod(softPod)).To(ContainSubstring("Aicore Limit Quota"))

		By("验证 soft Pod 的 ASCEND_VNPU_ID 环境变量")
		softVNpuID, _, _ := execCommandInPod(testNamespace, softPod, "app", "echo $ASCEND_VNPU_ID")
		Expect(strings.TrimSpace(softVNpuID)).NotTo(BeEmpty(), "soft Pod 的 ASCEND_VNPU_ID 应当注入")

		By("验证 hard Pod 有 VNPU 设备")
		vnpu := requireVNPUInPod(hardPod)
		Expect(vnpu).To(ContainSubstring("vir05_1c_8g"))

		By("验证 hard Pod 的 ASCEND_VNPU_SPECS 环境变量")
		hardVnpuSpecs, _, _ := execCommandInPod(testNamespace, hardPod, "app", "echo $ASCEND_VNPU_SPECS")
		Expect(strings.TrimSpace(hardVnpuSpecs)).NotTo(BeEmpty(), "hard Pod 的 ASCEND_VNPU_SPECS 应当注入")

		By("验证 full Pod 使用整卡挂载")
		requireWholeCardInPod(fullPod)

		By("验证 ResourceSlice 中三张卡的 vnpuMode 分别为 soft/hard/full")
		Expect(getDeviceVnpuMode(0)).To(Equal("soft"), "npu-0 的 vnpuMode 应为 soft")
		Expect(getDeviceVnpuMode(1)).To(Equal("hard"), "npu-1 的 vnpuMode 应为 hard")
		Expect(getDeviceVnpuMode(2)).To(Equal("full"), "npu-2 的 vnpuMode 应为 full")

			By("清理资源")
			deletePodAndWait(testNamespace, softPod)
			deletePodAndWait(testNamespace, hardPod)
			deletePodAndWait(testNamespace, fullPod)
			deleteResourceClaimTemplateAndWait(softClaim)
			deleteResourceClaimTemplateAndWait(hardClaim)
			deleteResourceClaimTemplateAndWait(fullClaim)
			GinkgoWriter.Printf("IT-019 验证通过: 三种模式同节点共存正常\n")
		})
	})
