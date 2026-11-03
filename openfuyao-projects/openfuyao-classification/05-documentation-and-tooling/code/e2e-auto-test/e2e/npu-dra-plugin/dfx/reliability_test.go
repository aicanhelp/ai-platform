package npu_dra_dfx

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("NPU DRA Plugin 可靠性测试 (DFR)",
	Ordered, Label("npu-dra-plugin", "26.06"), Label("npu"), Label("dfr"), func() {

		BeforeEach(func() {
			ensureNamespace()
		})

		AfterEach(func() {
			cleanupTestEnvironment()
			time.Sleep(10 * time.Second)
		})

		It("昇腾-npu-dra-DFX-004：软切分 vNPU 删除时资源 ID 回滚与复用", Label("P2"), func() {
			By("设置 npu-0 为 soft 模式")
			setNodeVnpuMode(localNodeName, softConfig(localNodeName, 0))

			By("设置 --share-count=2 限制最大切分数")
			setShareCount(2)
			DeferCleanup(func() {
				By("恢复 --share-count=16")
				restoreShareCount()
			})

			claimTemplate := "npu-claim-template-dfx04"
			pod1 := "npu-pod-dfx04-1"
			pod2 := "npu-pod-dfx04-2"
			pod3 := "npu-pod-dfx04-3"

			By("创建第 1 个软切分 Pod (memCapacity=4Gi, coreCapacity=50)，等待 Running")
			createSoftClaimTemplate(claimTemplate, memCapacity4GiB, 50, celPhysicalID0)
			createPodAndWaitRunning(pod1, claimTemplate, buildSoftVNpuPod)

			By("获取 vNPU ID，预期为 0")
			expectVnpuID(pod1, "0")

			By("创建第 2 个软切分 Pod (同配置)，等待 Running (vNPU 池达到 shareCount=2 上限)")
			createPodAndWaitRunning(pod2, claimTemplate, buildSoftVNpuPod)

			By("获取第 2 个 Pod 的 vNPU ID，预期为 1")
			expectVnpuID(pod2, "1")

			By("删除第 1 个 Pod，等待 vNPU 资源回收 (触发 DeleteVNPU→returnVNPUID 回滚 ID=0)")
			deletePodAndWait(pod1)
			time.Sleep(15 * time.Second)

			By("创建新的软切分 Pod (同配置)，等待 Running")
			createPodAndWaitRunning(pod3, claimTemplate, buildSoftVNpuPod)

			By("获取新 Pod 的 vNPU ID")
			expectVnpuID(pod3, "0")

			By("验证 vNPU ID 被回滚后重新分配，不递增泄漏")
			GinkgoWriter.Printf("DFX-004 验证通过: vNPU ID 在删除时回滚到内存池并可循环复用\n")

			By("清理资源")
			deletePodAndWait(pod2)
			deletePodAndWait(pod3)
			deleteResourceClaimTemplateAndWait(claimTemplate)
		})

		It("昇腾-npu-dra-DFX-006：Pod 删除后 CDI spec 文件清理验证", Label("P1"), func() {
			By("设置 npu-0 为 full 模式")
			setNodeVnpuMode(localNodeName, fullConfig(localNodeName, 0))

			claimTemplate := "npu-claim-template-dfx06"
			podName := "npu-pod-dfx06"

			By("记录 Pod 创建前的 CDI spec 文件列表")
			filesBefore := listCDISpecFiles()
			GinkgoWriter.Printf("创建前 CDI spec 文件: %v\n", filesBefore)

			By("创建 full 模式 ResourceClaimTemplate 和 Pod，等待 Running")
			createFullClaimTemplate(claimTemplate)
			createPodAndWaitRunning(podName, claimTemplate, buildFullNPUPod)

			By("验证 Pod 创建后新增了 CDI spec 文件")
			filesAfterCreate := listCDISpecFiles()
			newFiles := diffFileLists(filesBefore, filesAfterCreate)
			Expect(len(newFiles)).To(BeNumerically(">", 0),
				"Pod 创建后应当新增 CDI spec 文件 (cdi.go:88 CreateClaimSpecFile)")
			GinkgoWriter.Printf("新增 CDI spec 文件: %v\n", newFiles)

			By("删除 Pod，等待 Unprepare 完成 (触发 DeleteClaimSpecFile)")
			deletePodAndWait(podName)
			deleteResourceClaimTemplateAndWait(claimTemplate)

			By("验证 CDI spec 文件已被清理")
			Eventually(func() []string {
				return diffFileLists(filesBefore, listCDISpecFiles())
			}, 60*time.Second, pollInterval).Should(BeEmpty(),
				"Pod 删除后新增的 CDI spec 文件应当被清理 (cdi.go:124 DeleteClaimSpecFile)")

			GinkgoWriter.Printf("DFX-006 验证通过: Pod 删除后 CDI spec 文件被正确清理\n")
		})
	})
