package serverlessdb_operator

import (
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 认证鉴权",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-012：无 token 或无效 token 返回 401", func() {
			createPoolAndAwaitWarm("pool-it012", 1)

			By("无 Authorization 头调用申请接口")
			noAuthClient := apiClient.WithToken("")
			_, status, err := noAuthClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    "app-it012-noauth",
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(401), "无 token 应当返回 401")

			By("无效 token 调用申请接口")
			badClient := apiClient.WithToken("invalid.token.value")
			_, status2, err := badClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    "app-it012-badauth",
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status2).To(Equal(401), "无效 token 应当返回 401")
		})

		It("serverlessdb-IT-013：无 RBAC 权限的 token 返回 403", func() {
			createPoolAndAwaitWarm("pool-it013", 1)

			By("使用无 RoleBinding 的 SA token 调用申请接口")
			norbacToken, err := CreateNoRBACSA(ctx, k8sClient, testNs)
			Expect(err).NotTo(HaveOccurred(), "创建无 RBAC SA 应当成功")

			norbacClient := apiClient.WithToken(norbacToken)
			_, status, err := norbacClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    "app-it013-norbac",
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(403), "无 RBAC token 应当返回 403")
		})
	})
