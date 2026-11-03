package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 控制器重启恢复",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-015：控制器重启后资源池状态正常恢复", func() {
			createPoolAndAwaitWarm("pool-it015", 2)
			appRef := "app-it015"
			allocateAndAwaitRunning(appRef)

			By("等待 pool status 同步 allocatedInstances")
			Eventually(func(g Gomega) {
				_, _, allocated := poolStatusFields("pool-it015")
				g.Expect(allocated).To(BeNumerically(">=", int32(1)), "allocatedInstances 应当 >= 1")
			}, reconcileWait, reconcilePoll).Should(Succeed())

			By("记录重启前 pool status 快照")
			totalBefore, warmBefore, allocatedBefore := poolStatusFields("pool-it015")

			By("删除 operator pod 触发重建")
			restartOperatorPod()

			By("重建 port-forward 与 apiClient")
			reestablishPortForward()

			By("轮询 pool status 恢复至重启前一致")
			Eventually(func(g Gomega) {
				total, warm, allocated := poolStatusFields("pool-it015")
				g.Expect(total).To(Equal(totalBefore), "totalInstances 应当恢复")
				g.Expect(allocated).To(Equal(allocatedBefore), "allocatedInstances 应当恢复")
				g.Expect(warm).To(Equal(warmBefore), "warmInstances 应当恢复")
			}, reconcileWait, reconcilePoll).Should(Succeed())

			By("校验已分配实例仍为 Running")
			Expect(DBInstancePhase(dbInstanceByAppRef(appRef))).To(Equal("Running"), "重启后实例应仍为 Running")

			By("调用申请/释放验证 API 可用")
			probe := "app-it015-probe"
			resp, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    probe,
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200), "重启后申请应当返回 200")
			awaitInstancePhase(probe, "Running")
			Expect(resp.Endpoint).NotTo(BeEmpty())
			_, _ = apiClient.Release(ctx, testNs, probe)
		})
	})
