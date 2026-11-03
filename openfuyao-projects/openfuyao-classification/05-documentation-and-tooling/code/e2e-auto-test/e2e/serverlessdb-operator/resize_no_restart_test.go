package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 扩容 Pod 不重启",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-017：原地扩容时 Pod 不重建", func() {
			createPoolAndAwaitWarm("pool-it017", 1)
			appRef := "app-it017"
			allocateAndAwaitRunning(appRef)

			By("记录扩容前 Pod UID、restartCount、containerID")
			uidBefore := podUID(appRef)
			restartBefore := podRestartCount(appRef)
			idBefore := computeContainerID(appRef)
			Expect(uidBefore).NotTo(BeEmpty())
			Expect(idBefore).NotTo(BeEmpty())

			By("发起扩容 cpu=2, memory=512Mi")
			_, status, err := apiClient.Resize(ctx, testNs, appRef, &ResizeInstanceRequest{
				Containers: []ContainerResource{{
					Name:      ContainerCompute,
					Resources: InstanceResources{Cpu: "2", Memory: "512Mi", CpuLimit: "2", MemoryLimit: "512Mi"},
				}},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200), "扩容请求应当返回 200")

			By("等待实例经 Scaling 回到 Running")
			Eventually(func(g Gomega) string {
				return DBInstancePhase(dbInstanceByAppRef(appRef))
			}, reconcileWait, reconcilePoll).Should(Equal("Running"))

			By("校验 Pod 未重建：UID、restartCount、containerID 不变")
			Eventually(func(g Gomega) {
				pods, perr := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
				g.Expect(perr).NotTo(HaveOccurred())
				g.Expect(pods.Items).NotTo(BeEmpty())
				pod := &pods.Items[0]
				g.Expect(string(pod.UID)).To(Equal(uidBefore), "Pod UID 不应变（原地扩容不重建）")
				for _, c := range pod.Status.ContainerStatuses {
					if c.Name == ContainerCompute {
						g.Expect(c.RestartCount).To(Equal(restartBefore), "restartCount 不应增加")
						g.Expect(c.ContainerID).To(Equal(idBefore), "containerID 不应变")
						return
					}
				}
				g.Expect(false).To(BeTrue(), "应当存在 compute 容器")
			}, podResizeWait, podResizePoll).Should(Succeed())
		})
	})
