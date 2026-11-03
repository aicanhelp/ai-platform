package npu_dra_plugin

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("NPU DRA Plugin 硬切分测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P1"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
			By("设置 npu-0 为 hard 模式")
			setNodeVnpuMode(localNodeName, hardConfig(localNodeName, 0))
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-IT-005：单个硬切分 VNPU 创建", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片")
			}
			claimTemplate := "npu-claim-template-it05"
			podName := "npu-pod05"

			By("创建硬切分 ResourceClaimTemplate (aiCore=5, aiCPU=1)")
			createHardClaimTemplate(claimTemplate, 5, 1, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildHardVNpuPod)

			By("验证 Pod 内可见 VNPU 设备")
			vnpu := requireVNPUInPod(podName)
			GinkgoWriter.Printf("VNPU 设备: %s\n", vnpu)
			Expect(vnpu).To(ContainSubstring("vir05_1c_8g"))

			By("删除 Pod 并验证 VNPU 资源回收")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			time.Sleep(30 * time.Second)
			verifyVNPUCleanedUp("0", "0")
			GinkgoWriter.Printf("IT-005 验证通过: 单个硬切分 VNPU 创建与回收正常\n")
		})

		It("昇腾-npu-dra-IT-006：硬切分同卡资源不足场景", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片")
			}
			claim1 := "npu-claim-template-it06-5core"
			claim2 := "npu-claim-template-it06-10core-0"
			claim3 := "npu-claim-template-it06-10core-1"
			pod1 := "npu-pod06-5core"
			pod2 := "npu-pod06-10core-0"
			pod3 := "npu-pod06-10core-1"

			By("创建两个不同规格硬切分 Pod (aiCore=5 → vir05_1c_8g, aiCore=10 → vir10_3c_16g)")
			createHardClaimTemplate(claim1, 5, 1, celPhysicalID0)
			createPodAndWaitRunning(pod1, claim1, buildHardVNpuPod)
			vnpu1 := requireVNPUInPod(pod1)

			createHardClaimTemplate(claim2, 10, 3, celPhysicalID0)
			createPodAndWaitRunning(pod2, claim2, buildHardVNpuPod)
			vnpu2 := requireVNPUInPod(pod2)

			By("验证两个运行 Pod 的 vNPU 规格与申请值匹配")
			Expect(vnpu1).To(ContainSubstring("vir05"), "pod1 vNPU 规格应为 vir05")
			Expect(vnpu2).To(ContainSubstring("vir10"), "pod2 vNPU 规格应为 vir10")

			By("创建第三个 Pod 并验证 Pending (aiCore 总和 5+10+10=25 超过 max=20)")
			createHardClaimTemplate(claim3, 10, 3, celPhysicalID0)
			pod3Obj := buildHardVNpuPod(pod3, claim3)
			_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod3Obj, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
			waitForPodPending(testNamespace, pod3)

			By("验证 Pod Pending 原因是资源不足")
			Eventually(func() string {
				events, err := k8sClient.Clientset.CoreV1().Events(testNamespace).List(ctx, metav1.ListOptions{
					FieldSelector: "involvedObject.name=" + pod3,
				})
				if err != nil || len(events.Items) == 0 {
					return ""
				}
				for _, e := range events.Items {
					if e.Reason == "FailedScheduling" {
						return e.Message
					}
				}
				return ""
			}, 60*time.Second, pollInterval).Should(ContainSubstring("cannot allocate"), "应当包含资源不足信息")

			By("清理资源")
			deletePodAndWait(testNamespace, pod1)
			deletePodAndWait(testNamespace, pod2)
			deletePodAndWait(testNamespace, pod3)
			deleteResourceClaimTemplateAndWait(claim1)
			deleteResourceClaimTemplateAndWait(claim2)
			deleteResourceClaimTemplateAndWait(claim3)
			GinkgoWriter.Printf("IT-006 验证通过: 硬切分同卡资源不足时第三个 Pod Pending\n")
		})

		It("昇腾-npu-dra-IT-007：硬切分同卡资源充足场景", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片")
			}
			claim1 := "npu-claim-template-it07-5core"
			claim2 := "npu-claim-template-it07-10core"
			pod1 := "npu-pod07-5core"
			pod2 := "npu-pod07-10core"

			By("创建两个不同规格硬切分 Pod (aiCore=5 → vir05_1c_8g, aiCore=10 → vir10_3c_16g)")
			createHardClaimTemplate(claim1, 5, 1, celPhysicalID0)
			createPodAndWaitRunning(pod1, claim1, buildHardVNpuPod)
			vnpu1 := requireVNPUInPod(pod1)

			createHardClaimTemplate(claim2, 10, 3, celPhysicalID0)
			createPodAndWaitRunning(pod2, claim2, buildHardVNpuPod)
			vnpu2 := requireVNPUInPod(pod2)

			By("验证两个 Pod 各自 vNPU 规格与申请值匹配")
			Expect(vnpu1).To(ContainSubstring("vir05"), "pod1 vNPU 规格应为 vir05")
			Expect(vnpu2).To(ContainSubstring("vir10"), "pod2 vNPU 规格应为 vir10")

			By("清理资源")
			deletePodAndWait(testNamespace, pod1)
			deletePodAndWait(testNamespace, pod2)
			deleteResourceClaimTemplateAndWait(claim1)
			deleteResourceClaimTemplateAndWait(claim2)
			GinkgoWriter.Printf("IT-007 验证通过: 两个硬切分 VNPU 共存正常\n")
		})

		It("昇腾-npu-dra-IT-008：申请 AICore 低于模板最小值", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片")
			}
			claimTemplate := "npu-claim-template-it08"
			podName := "npu-pod08"

			By("创建硬切分 ResourceClaimTemplate (aiCore=1, 低于最小模板 vir05 的 aiCore=5)")
			createHardClaimTemplate(claimTemplate, 1, 1, "")

			By("创建 Pod 并验证 Pending")
			podObj := buildHardVNpuPod(podName, claimTemplate)
			_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, podObj, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
			waitForPodPending(testNamespace, podName)

			By("验证插件日志包含 per-claim 错误告警")
			pluginPod := getPluginPod()
			Eventually(func() string {
				return getPluginLogs(pluginPod.Name, pluginPod.Spec.Containers[0].Name)
			}, 120*time.Second, pollInterval).Should(
				ContainSubstring("PrepareResourceClaims completed with per-claim errors"),
				"插件日志应包含 per-claim 错误告警")

			By("清理资源")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			GinkgoWriter.Printf("IT-008 验证通过: aiCore=1 低于模板最小值时 Pod Pending\n")
		})

		It("昇腾-npu-dra-IT-009：申请 AICore 超过模板最大值", func() {
			if findNPUNode("910B") == "" {
				Skip("未找到 910B 节点，本用例需 910B 芯片")
			}
			claimTemplate := "npu-claim-template-it09"
			podName := "npu-pod09"

			By("创建硬切分 ResourceClaimTemplate (aiCore=20, 超过 910B4 最大模板 vir10 的 aiCore=10)")
			createHardClaimTemplate(claimTemplate, 20, 1, "")

			By("创建 Pod 并验证 Pending")
			podObj := buildHardVNpuPod(podName, claimTemplate)
			_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, podObj, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
			waitForPodPending(testNamespace, podName)

			By("验证插件日志包含 per-claim 错误告警")
			pluginPod := getPluginPod()
			Eventually(func() string {
				return getPluginLogs(pluginPod.Name, pluginPod.Spec.Containers[0].Name)
			}, 120*time.Second, pollInterval).Should(
				ContainSubstring("PrepareResourceClaims completed with per-claim errors"),
				"插件日志应包含 per-claim 错误告警")

			By("清理资源")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			GinkgoWriter.Printf("IT-009 验证通过: aiCore=20 超过模板最大值时 Pod Pending\n")
		})

		It("昇腾-npu-dra-IT-010：310P 硬切分基础能力验证", func() {
			if findNPUNode("310P") == "" {
				Skip("未找到 310P 节点")
			}

			claimTemplate := "npu-claim-template-it10"
			podName := "npu-pod10"

			By("创建硬切分 ResourceClaimTemplate (aiCore=1, aiCPU=1, 对应 vir01)")
			createHardClaimTemplate(claimTemplate, 1, 1, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildHardVNpuPod)

			By("验证 Pod 内可见 VNPU 设备")
			vnpu := requireVNPUInPod(podName)
			Expect(vnpu).To(ContainSubstring("vir01"))

			By("清理资源")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			time.Sleep(30 * time.Second)
			verifyVNPUCleanedUp("0", "0")
			GinkgoWriter.Printf("IT-010 验证通过: 310P 硬切分 vir01 正常\n")
		})

		It("昇腾-npu-dra-IT-022：910C 硬切分基础能力验证", func() {
			if findNPUNode("910C") == "" && findNPUNode("Ascend910") == "" {
				Skip("未找到 910C 节点")
			}

			claimTemplate := "npu-claim-template-it22"
			podName := "npu-pod22"

			By("创建硬切分 ResourceClaimTemplate (aiCore=5, aiCPU=1)")
			createHardClaimTemplate(claimTemplate, 5, 1, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildHardVNpuPod)

			By("验证 Pod 内可见 VNPU 设备")
			vnpu := requireVNPUInPod(podName)
			GinkgoWriter.Printf("910C VNPU: %s\n", vnpu)

			By("清理资源")
			deletePodAndWait(testNamespace, podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)
			time.Sleep(30 * time.Second)
			verifyVNPUCleanedUp("0", "0")
			GinkgoWriter.Printf("IT-022 验证通过: 910C 硬切分适配正常\n")
		})
	})
