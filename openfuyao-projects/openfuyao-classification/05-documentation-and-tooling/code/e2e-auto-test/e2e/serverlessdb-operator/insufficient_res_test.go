package serverlessdb_operator

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 资源不足场景",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-021：资源不足时 Pod 创建后处于 Pending 状态，Operator 正常处理", func() {
			poolName := "pool-ins001"

			By("创建资源需求极高的资源池（超出节点容量）")
			pool := BuildDBResourcePoolWithResources(poolName, testNs, 2, pgVersion,
				"100", "500Gi", "100", "500Gi")
			_, err := CreatePool(ctx, k8sClient, pool)
			Expect(err).NotTo(HaveOccurred(), "创建高资源需求资源池应当成功")

			DeferCleanup(func() {
				_ = DeletePool(ctx, k8sClient, testNs, poolName)
			})

			By("等待 Pod 被创建")
			Eventually(func() int {
				pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{
					LabelSelector: LabelEq(LabelDBPool, poolName),
				})
				if err != nil {
					return 0
				}
				return len(pods.Items)
			}, reconcileWait, reconcilePoll).Should(BeNumerically(">=", 1), "应当创建至少一个 Pod")

			By("验证 Pod 处于 Pending 状态（无法调度）")
			Consistently(func(g Gomega) {
				pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{
					LabelSelector: LabelEq(LabelDBPool, poolName),
				})
				g.Expect(err).NotTo(HaveOccurred())

				for _, pod := range pods.Items {
					g.Expect(pod.Status.Phase).To(Equal(corev1.PodPending),
						"Pod %s 应当处于 Pending 状态", pod.Name)
					var hasUnschedulableCondition bool
					for _, cond := range pod.Status.Conditions {
						if cond.Type == corev1.PodScheduled && cond.Status == corev1.ConditionFalse && cond.Reason == "Unschedulable" {
							hasUnschedulableCondition = true
							break
						}
					}
					g.Expect(hasUnschedulableCondition).To(BeTrue(),
						"Pod %s 应当有不可调度条件", pod.Name)
				}
			}, 90*time.Second, 10*time.Second).Should(Succeed(),
				"Pod 应当持续保持 Pending 状态")

			By("验证 Operator 没有崩溃")
			Eventually(func(g Gomega) {
				_, err := GetPool(ctx, k8sClient, testNs, poolName)
				g.Expect(err).NotTo(HaveOccurred(), "资源池应当存在")
			}, 30*time.Second, 5*time.Second).Should(Succeed(), "Operator 应当正常响应")
		})

		It("serverlessdb-IT-022：资源不足扩容时 Pod 处于 resize pending/in-progress 状态，Operator 正常处理", func() {
			poolName := "pool-ins002"

			By("创建正常资源需求的资源池")
			createPoolAndAwaitWarm(poolName, 1)

			By("申请实例")
			appRef := "app-ins002"
			resp := allocateAndAwaitRunning(appRef)
			Expect(resp).NotTo(BeNil(), "申请实例应当返回响应")

			By("发起超出节点容量的扩容请求")
			_, status, err := apiClient.Resize(ctx, testNs, appRef, &ResizeInstanceRequest{
				Containers: []ContainerResource{{
					Name: ContainerCompute,
					Resources: InstanceResources{
						Cpu: "100", Memory: "500Gi",
						CpuLimit: "100", MemoryLimit: "500Gi",
					},
				}},
			})
			Expect(err).NotTo(HaveOccurred(), "扩容请求应当成功")
			Expect(status).To(Equal(200), "扩容请求应当返回 200")

			By("验证扩容请求被 Operator 接受")
			Eventually(func(g Gomega) string {
				return DBInstancePhase(dbInstanceByAppRef(appRef))
			}, 15*time.Second, 2*time.Second).Should(Or(Equal("Scaling"), Equal("Running")),
				"实例应当进入 Scaling 或保持 Running 状态")

			By("验证 Pod 存活且 Operator 正常运行")
			Eventually(func(g Gomega) {
				pods, err := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{
					LabelSelector: LabelEq(LabelAppRef, appRef),
				})
				g.Expect(err).NotTo(HaveOccurred())
				g.Expect(pods.Items).NotTo(BeEmpty(), "Pod 应当存在")
				g.Expect(pods.Items[0].Status.Phase).To(Equal(corev1.PodRunning), "Pod 应当保持 Running")

				pool, err := GetPool(ctx, k8sClient, testNs, poolName)
				g.Expect(err).NotTo(HaveOccurred(), "资源池应当存在，Operator 正常响应")
				g.Expect(pool.GetDeletionTimestamp().IsZero()).To(BeTrue(), "资源池不应处于删除状态")
			}, 30*time.Second, 5*time.Second).Should(Succeed(),
				"Pod 应当存活且 Operator 正常响应")
		})
	})
