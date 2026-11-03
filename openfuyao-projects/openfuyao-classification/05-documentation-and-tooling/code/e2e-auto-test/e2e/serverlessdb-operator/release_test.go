package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 实例释放",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-009：释放实例后 DBInstance 与 Pod 被删除", func() {
			poolName := "pool-it009"
			createPoolAndAwaitWarm(poolName, 1)
			appRef := "app-it009"

			allocateAndAwaitRunning(appRef)

			By("调用释放接口")
			status, err := apiClient.Release(ctx, testNs, appRef)
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(204), "释放应当返回 204")

			By("DBInstance 被删除")
			Eventually(func(g Gomega) int {
				list, err := ListDBInstances(ctx, k8sClient, testNs, LabelEq(LabelAppRef, appRef))
				g.Expect(err).NotTo(HaveOccurred())
				return len(list.Items)
			}, reconcileWait, reconcilePoll).Should(Equal(0))

			By("Backing Pod 被级联删除")
			Eventually(func(g Gomega) int {
				pods, perr := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
				g.Expect(perr).NotTo(HaveOccurred())
				return len(pods.Items)
			}, reconcileWait, reconcilePoll).Should(Equal(0))
		})

		It("serverlessdb-IT-010：释放后资源池自动补齐 WarmReady 实例", func() {
			poolName := "pool-it010"
			createPoolAndAwaitWarm(poolName, 2)
			appRef := "app-it010"

			allocateAndAwaitRunning(appRef)

			By("释放实例")
			_, _ = apiClient.Release(ctx, testNs, appRef)

			By("等待 pool 重新补齐到 replicas=2")
			awaitPoolInstanceCount(poolName, 2)
			Eventually(func(g Gomega) bool {
				list, err := ListDBInstances(ctx, k8sClient, testNs, LabelEq(LabelDBPool, poolName))
				g.Expect(err).NotTo(HaveOccurred())
				for i := range list.Items {
					if DBInstancePhase(&list.Items[i]) == "WarmReady" {
						return true
					}
				}
				return false
			}, reconcileWait, reconcilePoll).Should(BeTrue(), "应当补齐出 WarmReady 实例")
		})
	})
