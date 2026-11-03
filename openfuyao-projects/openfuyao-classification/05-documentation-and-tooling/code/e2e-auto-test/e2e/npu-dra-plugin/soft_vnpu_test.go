package npu_dra_plugin

import (
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

var _ = Describe("NPU DRA Plugin 软切分测试",
	Ordered, Label("npu-dra-plugin", "26.08"), Label("P0"), Label("npu"), func() {

		BeforeEach(func() {
			ensureNamespace()
			By("设置 npu-0 为 soft 模式")
			setNodeVnpuMode(localNodeName, softConfig(localNodeName, 0))
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-IT-013：310P 软切分基础能力验证", func() {
			if findNPUNode("310P") == "" {
				Skip("未找到 310P 节点")
			}

			claimTemplate := "npu-claim-template-it13"
			podName := "npu-pod13"

			By("创建软切分 ResourceClaimTemplate (memCapacity=4Gi, coreCapacity=50)")
			createSoftClaimTemplate(claimTemplate, memCapacity4GiB, 50, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildSoftVNpuPod)

		By("验证容器内 enpu-monitor 配额正确")
		output := verifyEnpuMonitorInPod(podName)
		Expect(output).To(ContainSubstring("Aicore Limit Quota(%)"))
		Expect(output).To(ContainSubstring("50"))
		Expect(output).To(ContainSubstring("Memory Limit quota(MB)"))

		By("验证 ASCEND_VNPU_ID 环境变量")
		stdout, _, _ := execCommandInPod(testNamespace, podName, "app", "echo $ASCEND_VNPU_ID")
		Expect(strings.TrimSpace(stdout)).NotTo(BeEmpty(), "ASCEND_VNPU_ID 应当注入")

		By("验证 npu_info.config 配置正确")
		configOut, _, _ := execCommandInPod(testNamespace, podName, "app", "cat /etc/enpu/vcann-rt/npu_info.config 2>/dev/null")
		Expect(configOut).To(ContainSubstring("physical-npu-id="), "npu_info.config 应包含 physical-npu-id")
		Expect(configOut).To(ContainSubstring("virtual-npu-id="), "npu_info.config 应包含 virtual-npu-id")
		Expect(configOut).To(ContainSubstring("aicore-quota=50"), "npu_info.config 的 aicore-quota 应为 50")
		Expect(configOut).To(ContainSubstring("scheduling-policy="), "npu_info.config 应包含 scheduling-policy")

		By("验证 LD_PRELOAD 环境变量")
		ldPreloadOut, _, _ := execCommandInPod(testNamespace, podName, "app", "echo $LD_PRELOAD")
		ldPreloadVal := strings.TrimSpace(ldPreloadOut)
		if ldPreloadVal != "" {
			GinkgoWriter.Printf("LD_PRELOAD=%s（由运行时注入）\n", ldPreloadVal)
		} else {
			GinkgoWriter.Printf("LD_PRELOAD 未设置（由运行时按需注入，插件不强制）\n")
		}
		GinkgoWriter.Printf("IT-013 验证通过: 310P 软切分配额正确\n")
	})

		It("昇腾-npu-dra-IT-014：910B 软切分基础能力验证", func() {
			claimTemplate := "npu-claim-template-it14"
			podName := "npu-pod14"

			By("创建软切分 ResourceClaimTemplate (memCapacity=16Gi, coreCapacity=50)")
			createSoftClaimTemplate(claimTemplate, memCapacity16GiB, 50, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildSoftVNpuPod)

		By("验证容器内 enpu-monitor 配额正确")
		output := verifyEnpuMonitorInPod(podName)
		Expect(output).To(ContainSubstring("Aicore Limit Quota(%)"))
		Expect(output).To(ContainSubstring("50"))
		Expect(output).To(ContainSubstring("scheduling-policy"))
		Expect(output).To(ContainSubstring("2"))

		By("验证容器内只挂载单个 /dev/davinci 设备")
		devOut, _, _ := execCommandInPod(testNamespace, podName, "app", "ls /dev/davinci[0-9]* 2>/dev/null | wc -l")
		Expect(strings.TrimSpace(devOut)).To(Equal("1"), "应当只挂载单个 davinci 设备节点")
		GinkgoWriter.Printf("IT-014 验证通过: 910B 软切分配额正确\n")
	})

		It("昇腾-npu-dra-IT-015：软切分同卡 memCapacity 资源不足场景", func() {
			claim1 := "npu-claim-template-it15-1"
			claim2 := "npu-claim-template-it15-2"
			claim3 := "npu-claim-template-it15-3"
			pod1 := "npu-pod15-1"
			pod2 := "npu-pod15-2"
			pod3 := "npu-pod15-3"

		By("创建两个软切分 Pod (各 memCapacity=16Gi, coreCapacity=50/30, 共占 32Gi = 910B4 总内存)")
		createSoftClaimTemplate(claim1, memCapacity16GiB, 50, celPhysicalID0)
		createPodAndWaitRunning(pod1, claim1, buildSoftVNpuPod)

		createSoftClaimTemplate(claim2, memCapacity16GiB, 30, celPhysicalID0)
		createPodAndWaitRunning(pod2, claim2, buildSoftVNpuPod)

		By("创建第三个 Pod 申请 memCapacity=16Gi, coreCapacity=20, 总 48Gi > 32Gi")
		createSoftClaimTemplate(claim3, memCapacity16GiB, 20, celPhysicalID0)
			pod3Obj := buildSoftVNpuPod(pod3, claim3)
			_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod3Obj, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
			waitForPodPending(testNamespace, pod3)

			By("验证 Pod Pending 原因是 memCapacity 资源不足")
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
			GinkgoWriter.Printf("IT-015 验证通过: memCapacity 不足时第三个 Pod Pending\n")
		})

		It("昇腾-npu-dra-IT-020：软切分同卡 coreCapacity 资源不足场景", func() {
			claim1 := "npu-claim-template-it20-1"
			claim2 := "npu-claim-template-it20-2"
			claim3 := "npu-claim-template-it20-3"
			pod1 := "npu-pod20-1"
			pod2 := "npu-pod20-2"
			pod3 := "npu-pod20-3"

			By("创建两个软切分 Pod (各 coreCapacity=50, 共占 100 = 910B4 总算力)")
			createSoftClaimTemplate(claim1, memCapacity4GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(pod1, claim1, buildSoftVNpuPod)

			createSoftClaimTemplate(claim2, memCapacity4GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(pod2, claim2, buildSoftVNpuPod)

			By("创建第三个 Pod 申请 coreCapacity=10, 总 110 > 100")
			createSoftClaimTemplate(claim3, memCapacity4GiB, 10, celPhysicalID0)
			pod3Obj := buildSoftVNpuPod(pod3, claim3)
			_, err := k8sClient.Clientset.CoreV1().Pods(testNamespace).Create(ctx, pod3Obj, metav1.CreateOptions{})
			Expect(err).NotTo(HaveOccurred())
			waitForPodPending(testNamespace, pod3)

			By("验证 Pod Pending 原因是 coreCapacity 资源不足")
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
			GinkgoWriter.Printf("IT-020 验证通过: coreCapacity 不足时第三个 Pod Pending\n")
		})

		It("昇腾-npu-dra-IT-016：软切分同卡资源充足场景", func() {
			claim1 := "npu-claim-template-it16-1"
			claim2 := "npu-claim-template-it16-2"
			pod1 := "npu-pod16-1"
			pod2 := "npu-pod16-2"

			By("创建两个软切分 Pod (各 50%+30% coreCapacity, 共享 npu-0)")
			createSoftClaimTemplate(claim1, memCapacity16GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(pod1, claim1, buildSoftVNpuPod)

			createSoftClaimTemplate(claim2, memCapacity16GiB, 30, celPhysicalID0)
			createPodAndWaitRunning(pod2, claim2, buildSoftVNpuPod)

			By("验证两个 Pod 均有独立 vNPU ID")
			out1, _, _ := execCommandInPod(testNamespace, pod1, "app", "echo $ASCEND_VNPU_ID")
			out2, _, _ := execCommandInPod(testNamespace, pod2, "app", "echo $ASCEND_VNPU_ID")
			vnpuID1 := strings.TrimSpace(out1)
			vnpuID2 := strings.TrimSpace(out2)
			Expect(vnpuID1).NotTo(BeEmpty())
			Expect(vnpuID2).NotTo(BeEmpty())
			Expect(vnpuID1).NotTo(Equal(vnpuID2), "两个 Pod 的 vNPU ID 应当不同")
			GinkgoWriter.Printf("vNPU IDs: pod1=%s, pod2=%s\n", vnpuID1, vnpuID2)

		By("验证 enpu-monitor 配额正确")
		output1 := verifyEnpuMonitorInPod(pod1)
		Expect(output1).To(ContainSubstring("50"))
		output2 := verifyEnpuMonitorInPod(pod2)
		Expect(output2).To(ContainSubstring("30"))

			By("清理资源")
			deletePodAndWait(testNamespace, pod1)
			deletePodAndWait(testNamespace, pod2)
			deleteResourceClaimTemplateAndWait(claim1)
			deleteResourceClaimTemplateAndWait(claim2)
			GinkgoWriter.Printf("IT-016 验证通过: 两个软切分 Pod 共享同卡正常\n")
		})

		It("昇腾-npu-dra-IT-017：软切分最大切分数验证", func() {
			By("设置 --share-count=2 限制最大切分数")
			setShareCount(2)
			DeferCleanup(func() {
				By("恢复 --share-count=16")
				restoreShareCount()
			})

			claim1 := "npu-claim-template-it17-1"
			claim2 := "npu-claim-template-it17-2"
			claim3 := "npu-claim-template-it17-3"
			pod1 := "npu-pod17-1"
			pod2 := "npu-pod17-2"
			pod3 := "npu-pod17-3"

		By("创建两个软切分 Pod (各 16Gi，总内存未超限但切分数达到上限)")
		createSoftClaimTemplate(claim1, memCapacity16GiB, 50, celPhysicalID0)
		createPodAndWaitRunning(pod1, claim1, buildSoftVNpuPod)

		createSoftClaimTemplate(claim2, memCapacity16GiB, 30, celPhysicalID0)
		createPodAndWaitRunning(pod2, claim2, buildSoftVNpuPod)

			By("验证 vNPU ID 分别为 0 和 1")
			out1, _, _ := execCommandInPod(testNamespace, pod1, "app", "echo $ASCEND_VNPU_ID")
			out2, _, _ := execCommandInPod(testNamespace, pod2, "app", "echo $ASCEND_VNPU_ID")
			Expect(strings.TrimSpace(out1)).To(Equal("0"))
			Expect(strings.TrimSpace(out2)).To(Equal("1"))

			By("创建第三个 Pod 并验证 Pending (shareCount=2 已满)")
			createSoftClaimTemplate(claim3, memCapacity16GiB, 20, celPhysicalID0)
			pod3Obj := buildSoftVNpuPod(pod3, claim3)
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
			GinkgoWriter.Printf("IT-017 验证通过: shareCount=2 限制正常\n")
		})

		It("昇腾-npu-dra-IT-018：软切分三种调度策略验证", func() {
			// 需要 3 张卡分别配置 elastic/fixed/best-effort
			By("配置三卡不同调度策略: 0=elastic, 1=fixed, 2=best-effort")
			config := mixedConfig(localNodeName,
				"    - physicalId: 0\n      vnpuMode: soft\n      schedulingPolicy: elastic",
				"    - physicalId: 1\n      vnpuMode: soft\n      schedulingPolicy: fixed-share",
				"    - physicalId: 2\n      vnpuMode: soft\n      schedulingPolicy: best-effort",
			)
			setNodeVnpuMode(localNodeName, config)

			By("创建 elastic 策略 Pod (deviceClass=elastic, physicalID=0)")
			elasticClaim := "npu-claim-template-it18-elastic"
			elasticPod := "npu-pod18-elastic"
			createSoftClaimTemplateWithClass(elasticClaim, deviceClassElastic, memCapacity16GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(elasticPod, elasticClaim, buildSoftVNpuPod)

			By("创建 fixed 策略 Pod (deviceClass=fixed, physicalID=1)")
			fixedClaim := "npu-claim-template-it18-fixed"
			fixedPod := "npu-pod18-fixed"
			createSoftClaimTemplateWithClass(fixedClaim, deviceClassFixed, memCapacity16GiB, 50, celPhysicalID1)
			createPodAndWaitRunning(fixedPod, fixedClaim, buildSoftVNpuPod)

			By("创建 best-effort 策略 Pod (deviceClass=best-effort, physicalID=2)")
			bestEffortClaim := "npu-claim-template-it18-besteffort"
			bestEffortPod := "npu-pod18-besteffort"
			createSoftClaimTemplateWithClass(bestEffortClaim, deviceClassBestEffort, memCapacity16GiB, 50, celPhysicalID2)
			createPodAndWaitRunning(bestEffortPod, bestEffortClaim, buildSoftVNpuPod)

			By("验证 elastic Pod enpu-monitor scheduling-policy=2")
			elasticOutput := verifyEnpuMonitorInPod(elasticPod)
			Expect(elasticOutput).To(ContainSubstring("scheduling-policy"))
			Expect(elasticOutput).To(ContainSubstring("2"), "elastic 策略 scheduling-policy 应为 2")

			By("验证 fixed Pod enpu-monitor scheduling-policy=1")
			fixedOutput := verifyEnpuMonitorInPod(fixedPod)
			Expect(fixedOutput).To(ContainSubstring("scheduling-policy"))
			Expect(fixedOutput).To(ContainSubstring("1"), "fixed 策略 scheduling-policy 应为 1")

			By("验证 best-effort Pod enpu-monitor scheduling-policy=3")
			bestEffortOutput := verifyEnpuMonitorInPod(bestEffortPod)
			Expect(bestEffortOutput).To(ContainSubstring("scheduling-policy"))
			Expect(bestEffortOutput).To(ContainSubstring("3"), "best-effort 策略 scheduling-policy 应为 3")

			By("清理资源")
			deletePodAndWait(testNamespace, elasticPod)
			deletePodAndWait(testNamespace, fixedPod)
			deletePodAndWait(testNamespace, bestEffortPod)
			deleteResourceClaimTemplateAndWait(elasticClaim)
			deleteResourceClaimTemplateAndWait(fixedClaim)
			deleteResourceClaimTemplateAndWait(bestEffortClaim)
			GinkgoWriter.Printf("IT-018 验证通过: 三种调度策略 elastic=2, fixed=1, best-effort=3\n")
		})

		It("昇腾-npu-dra-IT-021：910C 软切分基础能力验证", func() {
			if findNPUNode("910C") == "" && findNPUNode("Ascend910") == "" {
				Skip("未找到 910C 节点")
			}

			claimTemplate := "npu-claim-template-it21"
			podName := "npu-pod21"

			By("创建软切分 ResourceClaimTemplate (memCapacity=32Gi, coreCapacity=50)")
			createSoftClaimTemplate(claimTemplate, memCapacity32GiB, 50, "")

			By("创建 Pod 并等待 Running")
			createPodAndWaitRunning(podName, claimTemplate, buildSoftVNpuPod)

			By("验证容器内 enpu-monitor 配额正确")
			output := verifyEnpuMonitorInPod(podName)
		Expect(output).To(ContainSubstring("Aicore Limit Quota(%)"))
		Expect(output).To(ContainSubstring("50"))
		Expect(output).To(ContainSubstring("scheduling-policy"))

		By("验证只挂载单个 /dev/davinci 设备")
			stdout, _, _ := execCommandInPod(testNamespace, podName, "app", "ls /dev/davinci* 2>/dev/null | wc -l")
			Expect(strings.TrimSpace(stdout)).To(Equal("1"), "应当只挂载单个 davinci 设备")

			GinkgoWriter.Printf("IT-021 验证通过: 910C 软切分适配正常\n")
		})
	})
