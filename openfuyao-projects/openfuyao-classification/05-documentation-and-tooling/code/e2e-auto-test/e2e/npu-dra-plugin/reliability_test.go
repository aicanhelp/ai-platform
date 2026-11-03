package npu_dra_plugin

import (
	"fmt"
	"os/exec"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("NPU DRA Plugin 可靠性测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P2"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
		})

	It("昇腾-npu-dra-IT-011：硬切分 VNPU 重复创建删除可靠性", func() {
		if findNPUNode("910B") == "" {
			Skip("未找到 910B 节点，本用例需 910B 芯片")
		}
		By("设置 npu-0 为 hard 模式")
			setNodeVnpuMode(localNodeName, hardConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-it11"
			podName := "npu-pod11"

			By("创建硬切分 ResourceClaimTemplate (aiCore=5, aiCPU=1)")
			createHardClaimTemplate(claimTemplate, 5, 1, celPhysicalID0)

			for i := 1; i <= 3; i++ {
				By(fmt.Sprintf("第 %d 轮: 创建 Pod 并验证 VNPU", i))
				createPodAndWaitRunning(podName, claimTemplate, buildHardVNpuPod)
				vnpu := requireVNPUInPod(podName)
				GinkgoWriter.Printf("迭代 %d/3 VNPU: %s\n", i, vnpu)

				By(fmt.Sprintf("第 %d 轮: 删除 Pod 并验证 VNPU 回收", i))
				deletePodAndWait(testNamespace, podName)
				time.Sleep(30 * time.Second)
				verifyVNPUCleanedUp("0", "0")
				GinkgoWriter.Printf("迭代 %d/3 VNPU 回收完成\n", i)
			}

			deleteResourceClaimTemplateAndWait(claimTemplate)
			GinkgoWriter.Printf("IT-011 验证通过: 3 次硬切分 VNPU 创建/删除循环无泄漏\n")
		})

		It("昇腾-npu-dra-IT-012：插件重启后资源上报与挂载自愈验证", func() {
			By("设置 npu-0 为 full 模式")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-it12"
			podName := "npu-pod12"

			By("强制重启 DRA 插件 Pod")
			exec.Command("kubectl", "rollout", "restart", "ds", "npu-dra-plugin", "-n", pluginNamespace).Run()
			time.Sleep(15 * time.Second)

			By("删除现有 ResourceSlice")
			deleteAllResourceSlices()

			By("等待插件自动重新上报 ResourceSlice")
			waitForResourceSlice(localNodeName)

			By("创建 Pod 验证挂载恢复")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)
			output := verifyNPUSMIInPod(podName)
			Expect(output).NotTo(BeEmpty(), "Pod 内应当可见 NPU 设备")

			By("清理资源")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			GinkgoWriter.Printf("IT-012 验证通过: 插件重启后自愈正常\n")
		})
	})
