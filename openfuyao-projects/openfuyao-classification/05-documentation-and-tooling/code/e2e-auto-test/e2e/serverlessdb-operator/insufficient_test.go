package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 资源不足场景",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-016：超出池容量申请返回 404", func() {
			createPoolAndAwaitWarm("pool-it016", 1)

			By("申请第 1 个实例并等待 Running")
			allocateAndAwaitRunning("app-it016-a")

			By("立即申请第 2 个实例（无可用预热实例）")
			_, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    "app-it016-b",
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(404), "无可用预热实例应当返回 404")
		})
	})
