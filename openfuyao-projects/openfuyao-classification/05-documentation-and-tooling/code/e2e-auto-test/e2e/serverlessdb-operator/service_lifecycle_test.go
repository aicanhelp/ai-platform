package serverlessdb_operator

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB Service 生命周期管理",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-018：释放实例后 Headless Service 保留", func() {
			createPoolAndAwaitWarm("pool-it018", 1)
			appRef := "app-it018"
			allocateAndAwaitRunning(appRef)

			By("校验 Service 存在")
			_, err := GetService(ctx, k8sClient, testNs, appRef)
			Expect(err).NotTo(HaveOccurred(), "Service 应当存在")

			By("释放实例")
			_, err = apiClient.Release(ctx, testNs, appRef)
			Expect(err).NotTo(HaveOccurred())

			By("等待 DBInstance 与 Pod 删除")
			Eventually(func(g Gomega) int {
				list, lerr := ListDBInstances(ctx, k8sClient, testNs, LabelEq(LabelAppRef, appRef))
				g.Expect(lerr).NotTo(HaveOccurred())
				return len(list.Items)
			}, reconcileWait, reconcilePoll).Should(Equal(0))
			Eventually(func(g Gomega) int {
				pods, perr := k8sClient.ListPods(ctx, testNs, metav1.ListOptions{LabelSelector: LabelEq(LabelAppRef, appRef)})
				g.Expect(perr).NotTo(HaveOccurred())
				return len(pods.Items)
			}, reconcileWait, reconcilePoll).Should(Equal(0))

			By("校验 Service 仍存在（释放后保留以便复用连接地址）")
			svc, err := GetService(ctx, k8sClient, testNs, appRef)
			Expect(err).NotTo(HaveOccurred(), "Headless Service 应当保留")
			Expect(svc.Spec.ClusterIP).To(Equal("None"), "应当仍是 Headless Service")
		})

		It("serverlessdb-IT-019：孤儿 Service 超 TTL 被清理", func() {
			if EnvOr("SERVERLESSDB_SERVICE_TTL_DISABLED", "") != "" {
				Skip("ServiceTTL 已禁用，跳过孤儿 Service 清理用例")
			}
			poolName := "pool-it019"
			createPoolAndAwaitWarm(poolName, 1)

			orphan := "app-it019-orphan"
			By("手动创建无 backing pod 的 managed Headless Service")
			labels := map[string]string{LabelManaged: "true", LabelAppRef: orphan}
			Expect(CreateHeadlessService(ctx, k8sClient, testNs, orphan, labels, PortNamePg, int32(DefaultPgPort))).To(Succeed())
			DeferCleanup(func() {
				_ = DeleteService(ctx, k8sClient, testNs, orphan)
			})

			By("将 pod-absent-since 注解回填为 2 小时前（模拟超 TTL）")
			past := time.Now().Add(-2 * time.Hour).UTC().Format(time.RFC3339)
			Expect(AnnotateService(ctx, k8sClient, testNs, orphan, map[string]string{AnnoPodAbsent: past})).To(Succeed())

			By("触发 pool reconcile（小幅 patch replicas 不变值以触发 reconcile）")
			_, err := PatchPoolReplicas(ctx, k8sClient, testNs, poolName, 1)
			Expect(err).NotTo(HaveOccurred())

			By("轮询校验孤儿 Service 被清理")
			Eventually(func(g Gomega) bool {
				_, gerr := GetService(ctx, k8sClient, testNs, orphan)
				return apierrors.IsNotFound(gerr)
			}, reconcileWait, reconcilePoll).Should(BeTrue(), "超 TTL 的孤儿 Service 应当被清理")
		})
	})
