package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 实例垂直扩缩容",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-007：扩容 cpu 与 memory 后实例规格生效", func() {
			createPoolAndAwaitWarm("pool-it007", 1)
			appRef := "app-it007"
			allocateAndAwaitRunning(appRef)

			By("发起扩容请求 cpu=2, memory=512Mi")
			resp, status, err := apiClient.Resize(ctx, testNs, appRef, &ResizeInstanceRequest{
				Containers: []ContainerResource{{
					Name:      ContainerCompute,
					Resources: InstanceResources{Cpu: "2", Memory: "512Mi", CpuLimit: "2", MemoryLimit: "512Mi"},
				}},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200), "扩容请求应当返回 200")

			By("实例进入 Scaling 后回到 Running")
			Eventually(func(g Gomega) string {
				return DBInstancePhase(dbInstanceByAppRef(appRef))
			}, reconcileWait, reconcilePoll).Should(Equal("Running"))

			By("校验 Pod compute 容器资源已变更")
			Eventually(func(g Gomega) {
				pods, perr := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
				g.Expect(perr).NotTo(HaveOccurred())
				g.Expect(pods.Items).NotTo(BeEmpty())
				for _, c := range pods.Items[0].Spec.Containers {
					if c.Name == ContainerCompute {
						g.Expect(c.Resources.Requests.Cpu().String()).To(Equal("2"))
						return
					}
				}
				g.Expect(false).To(BeTrue(), "应当存在 compute 容器")
			}, podResizeWait, podResizePoll).Should(Succeed())

			Expect(resp.Endpoint).NotTo(BeEmpty(), "扩容响应应包含 endpoint")
		})

		It("serverlessdb-IT-008：仅指定 cpu 部分字段扩容成功", func() {
			createPoolAndAwaitWarm("pool-it008", 1)
			appRef := "app-it008"
			allocateAndAwaitRunning(appRef)

			By("仅扩容 cpu=1（不指定 memory）")
			_, status, err := apiClient.Resize(ctx, testNs, appRef, &ResizeInstanceRequest{
				Containers: []ContainerResource{{
					Name:      ContainerCompute,
					Resources: InstanceResources{Cpu: "1"},
				}},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200), "部分字段扩容请求应当返回 200")

			By("实例回到 Running")
			Eventually(func(g Gomega) string {
				return DBInstancePhase(dbInstanceByAppRef(appRef))
			}, reconcileWait, reconcilePoll).Should(Equal("Running"))
		})
	})
