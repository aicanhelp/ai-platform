package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 主备切换",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-020：多副本主故障备节点接管", func() {
			// 用例默认 operator 单副本，临时扩到 2 副本触发选主，用例结束恢复 1 副本。
			originalReplicas := int32(1)
			By("scale operator Deployment 到 2 副本并等待 Ready")
			scaleOperator(2)
			DeferCleanup(func() {
				By("清理：scale operator Deployment 回 1 副本")
				Expect(ScaleDeployment(ctx, k8sClient, opNs, getOperatorDeploymentName(), originalReplicas)).To(Succeed())
				awaitOperatorReadyCount(1)
				By("清理后重建 port-forward 指向幸存 pod")
				reestablishPortForward()
			})

			By("读取当前 leader pod")
			initialLeader := leaderPodName()
			Expect(initialLeader).NotTo(BeEmpty(), "应当存在当前 leader pod")

			By("创建 pool 并申请 1 个实例")
			createPoolAndAwaitWarm("pool-it020", 2)
			appRef := "app-it020"
			allocateAndAwaitRunning(appRef)

			By("删除 leader pod 触发备节点接管")
			deleteLeaderPodAndAwaitTakeover()

			By("等待新 leader pod Ready")
			awaitOperatorReadyCount(2)

			By("重建 port-forward 与 apiClient 指向新 leader")
			reestablishPortForward()

			By("校验已分配实例状态不变")
			Eventually(func(g Gomega) string {
				return DBInstancePhase(dbInstanceByAppRef(appRef))
			}, reconcileWait, reconcilePoll).Should(Equal("Running"), "切换后实例应仍为 Running")

			By("校验 API 可用（申请并释放一个探针实例）")
			probe := "app-it020-probe"
			_, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    probe,
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(200), "切换后申请应当返回 200")
			awaitInstancePhase(probe, "Running")
			_, _ = apiClient.Release(ctx, testNs, probe)

			By("校验 leader 已切换")
			Expect(leaderPodName()).NotTo(Equal(initialLeader), "leader 应当已切换到备节点")
		})
	})
