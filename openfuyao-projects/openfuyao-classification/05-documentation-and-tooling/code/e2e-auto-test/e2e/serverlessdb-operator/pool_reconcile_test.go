package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 资源池调谐",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-001：创建资源池后调谐出指定数量的 WarmReady 实例", func() {
			poolName := "pool-it001"
			createPoolAndAwaitWarm(poolName, 2)
			awaitPoolInstanceCount(poolName, 2)
		})

		It("serverlessdb-IT-002：扩容 replicas 后实例数量随之增加", func() {
			poolName := "pool-it002"
			createPoolAndAwaitWarm(poolName, 2)

			By("将 replicas 改为 4")
			_, err := PatchPoolReplicas(ctx, k8sClient, testNs, poolName, 4)
			Expect(err).NotTo(HaveOccurred(), "patch replicas 应当成功")

			awaitPoolInstanceCount(poolName, 4)
		})

		It("serverlessdb-IT-003：缩容 replicas 后未分配实例被回收", func() {
			poolName := "pool-it003"
			createPoolAndAwaitWarm(poolName, 3)

			By("将 replicas 改为 1")
			_, err := PatchPoolReplicas(ctx, k8sClient, testNs, poolName, 1)
			Expect(err).NotTo(HaveOccurred(), "patch replicas 应当成功")

			awaitPoolInstanceCount(poolName, 1)
		})
	})
