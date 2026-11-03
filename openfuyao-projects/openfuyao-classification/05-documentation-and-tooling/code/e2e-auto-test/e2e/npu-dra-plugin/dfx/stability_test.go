package npu_dra_dfx

import (
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("NPU DRA Plugin 稳定性测试 (DFS)",
	Ordered, Label("npu-dra-plugin", "26.06"), Label("npu"), Label("dfs"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-DFX-001：DaemonSet 滚动重启不影响已有 Pod 的 NPU 设备挂载", Label("P1"), func() {
			By("设置 npu-0 为 full 模式")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-dfx01"
			podName := "npu-pod-dfx01"

			By("创建 full 模式 ResourceClaimTemplate")
			createFullClaimTemplate(claimTemplate)

			By("创建使用该 template 的 Pod，等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("在 Pod 内执行 npu-smi info 验证 NPU 设备可见，记录输出")
			outputBefore := verifyNPUSMIInPod(podName)
			Expect(outputBefore).NotTo(BeEmpty(), "重启前 Pod 内 npu-smi info 输出不应为空")
			Expect(strings.Contains(outputBefore, "910") || strings.Contains(outputBefore, "310") || strings.Contains(outputBefore, "Ascend")).
				To(BeTrue(), "重启前 Pod 内应当可见 NPU 设备信息")
			GinkgoWriter.Printf("重启前 npu-smi 输出:\n%s\n", outputBefore)

			By("滚动重启 DRA 插件 DaemonSet: kubectl rollout restart ds npu-dra-plugin -n npu-dra-driver")
			restartPluginDaemonSet()

			By("在原业务 Pod 内再次执行 npu-smi info")
			outputAfter := verifyNPUSMIInPod(podName)
			Expect(outputAfter).NotTo(BeEmpty(), "滚动重启后业务 Pod 内 npu-smi info 输出不应为空")

			By("验证业务 Pod 内 NPU 设备仍可访问且输出一致")
			Expect(outputAfter).NotTo(BeEmpty(), "滚动重启后业务 Pod 内 npu-smi info 输出不应为空")
			Expect(strings.Contains(outputAfter, "310P3") || strings.Contains(outputAfter, "910") || strings.Contains(outputAfter, "Ascend")).
				To(BeTrue(), "滚动重启后 NPU 设备仍应当可见")
			Expect(parseNpuName(outputBefore)).To(Equal(parseNpuName(outputAfter)),
				"滚动重启前后 NPU 设备型号应当一致")

			By("验证业务 Pod 仍处于 Running 状态")
			pod, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Get(ctx, podName, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "获取业务 Pod 应当成功")
			Expect(pod.Status.Phase).To(Equal(corev1.PodRunning), "业务 Pod 应当仍处于 Running 状态")

			By("验证插件重启后 ResourceSlice 已重新上报")
			Eventually(func() int {
				rs := waitForResourceSlice(localNodeName)
				return len(rs.Spec.Devices)
			}, 60*time.Second, pollInterval).Should(BeNumerically(">", 0),
				"插件重启后 ResourceSlice 应当重新上报")

			GinkgoWriter.Printf("DFX-001 验证通过: DaemonSet 滚动重启期间业务 Pod NPU 设备挂载不受影响\n")
		})

		It("昇腾-npu-dra-DFX-002：插件 checkpoint 在重启后保持已准备 claim 状态", Label("P1"), func() {
			By("设置 npu-0 为 full 模式")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-dfx02"
			podName := "npu-pod-dfx02"

			By("创建 full 模式 ResourceClaimTemplate 和 Pod，等待 Running")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("在插件 Pod 内验证 checkpoint.json 存在并读取内容")
			checkpointBefore := readCheckpointInPlugin()
			Expect(checkpointBefore).NotTo(BeEmpty(), "checkpoint.json 应当存在且可读取")
			Expect(strings.Contains(checkpointBefore, "preparedClaims")).To(BeTrue(),
				"checkpoint.json 应当包含 preparedClaims 字段")
			claimCountBefore := countPreparedClaims(checkpointBefore)
			Expect(claimCountBefore).To(BeNumerically(">=", 1), "checkpoint 中应当至少有 1 个 prepared claim")
			GinkgoWriter.Printf("重启前 checkpoint:\n%s\n", checkpointBefore)

			By("记录 checkpoint 中的 prepared claims 数量")
			GinkgoWriter.Printf("重启前 preparedClaims 数量: %d\n", claimCountBefore)

			By("重启插件 DaemonSet: kubectl rollout restart ds npu-dra-plugin -n npu-dra-driver")
			restartPluginDaemonSet()

			By("再次读取 checkpoint.json")
			checkpointAfter := readCheckpointInPlugin()
			Expect(checkpointAfter).NotTo(BeEmpty(), "重启后 checkpoint.json 应当仍存在且可读取")
			Expect(strings.Contains(checkpointAfter, "preparedClaims")).To(BeTrue(),
				"重启后 checkpoint.json 应当仍包含 preparedClaims 字段")

			By("验证 prepared claims 数量与重启前一致")
			claimCountAfter := countPreparedClaims(checkpointAfter)
			Expect(claimCountAfter).To(Equal(claimCountBefore),
				"重启后 prepared claims 数量应当与重启前一致")

			By("验证业务 Pod 内 NPU 设备仍可用")
			output := verifyNPUSMIInPod(podName)
			Expect(output).NotTo(BeEmpty(), "重启后业务 Pod 内 NPU 设备应当仍可用")

			GinkgoWriter.Printf("DFX-002 验证通过: 插件重启后 checkpoint 保持已准备 claim 状态\n")
		})
	})
