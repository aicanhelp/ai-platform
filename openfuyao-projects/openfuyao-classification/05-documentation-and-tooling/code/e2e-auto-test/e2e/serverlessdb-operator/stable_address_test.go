package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 连接地址保持",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-011：释放后重新申请同一 app_ref，endpoint 保持不变", func() {
			poolName := "pool-it011"
			createPoolAndAwaitWarm(poolName, 2)
			appRef := "app-it011"

			By("首次申请")
			resp1, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    appRef,
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200))
			Expect(resp1.Endpoint).NotTo(BeEmpty())
			awaitInstancePhase(appRef, "Running")
			DeferCleanup(func() { _, _ = apiClient.Release(ctx, testNs, appRef) })

			By("释放实例（Headless Service 保留）")
			_, err = apiClient.Release(ctx, testNs, appRef)
			Expect(err).NotTo(HaveOccurred())

			By("等待实例被删除，pool 补齐 warm 实例")
			Eventually(func(g Gomega) int {
				list, err := ListDBInstances(ctx, k8sClient, testNs, LabelEq(LabelAppRef, appRef))
				g.Expect(err).NotTo(HaveOccurred())
				return len(list.Items)
			}, reconcileWait, reconcilePoll).Should(Equal(0))

			By("重新申请同一 app_ref")
			resp2, status2, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    appRef,
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status2).To(Equal(200))
			Expect(resp2.Endpoint).NotTo(BeEmpty())
			awaitInstancePhase(appRef, "Running")
			DeferCleanup(func() { _, _ = apiClient.Release(ctx, testNs, appRef) })

			By("校验两次 endpoint 相同（Headless Service 复用）")
			Expect(resp2.Endpoint).To(Equal(resp1.Endpoint), "释放后重新申请的 endpoint 应当保持不变")
		})
	})
