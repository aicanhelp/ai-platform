package serverlessdb_operator

import (
	"fmt"
	"os/exec"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var _ = Describe("ServerlessDB 实例申请",
	Label("serverlessdb-operator", "with-workload-cluster", "26.01"), Label("P0"), func() {

		It("serverlessdb-IT-004：申请实例成功，状态与连接地址正确", func() {
			createPoolAndAwaitWarm("pool-it004", 1)
			appRef := "app-it004"

			resp := allocateAndAwaitRunning(appRef)

			By("校验 DBInstance label 与 status")
			ins := dbInstanceByAppRef(appRef)
			Expect(DBInstanceAppRef(ins)).To(Equal(appRef), "DBInstance 应携带 app-ref label")

			By("校验返回的 endpoint 格式")
			Expect(resp.Endpoint).To(ContainSubstring(fmt.Sprintf("%s.%s.svc.cluster.local", appRef, testNs)),
				"endpoint 应为 <app_ref>.<ns>.svc.cluster.local:<port>")
		})

		It("serverlessdb-IT-005：申请后创建 Headless Service，DNS 与端口可连通", func() {
			createPoolAndAwaitWarm("pool-it005", 1)
			appRef := "app-it005"
			allocateAndAwaitRunning(appRef)

			By("校验 Headless Service 已创建")
			svc, err := k8sClient.Clientset.CoreV1().Services(testNs).Get(ctx, appRef, metav1.GetOptions{})
			Expect(err).NotTo(HaveOccurred(), "Headless Service 应当存在")
			Expect(svc.Spec.ClusterIP).To(Equal("None"), "应当是 Headless Service (ClusterIP=None)")

			if EnvOr("SERVERLESSDB_SKIP_DNS", "") != "" {
				Skip("SERVERLESSDB_SKIP_DNS 已设置，跳过 DNS/TCP 连通性检查")
			}
			By("从集群内校验 DNS 解析 + TCP 5432 可连")
			dnsAddr := fmt.Sprintf("%s.%s.svc.cluster.local:%d", appRef, testNs, DefaultPgPort)
			Eventually(func(g Gomega) {
				pods, lerr := k8sClient.ListPods(ctx, opNs, metav1.ListOptions{LabelSelector: OperatorPodLabel})
				g.Expect(lerr).NotTo(HaveOccurred())
				g.Expect(pods.Items).NotTo(BeEmpty())
				opPod := pods.Items[0].Name
				cmd := exec.CommandContext(ctx, "kubectl", "exec", "-n", opNs, opPod, "--",
					"bash", "-c",
					fmt.Sprintf("exec 3<>/dev/tcp/%s/%d && exec 3>&-",
						fmt.Sprintf("%s.%s.svc.cluster.local", appRef, testNs), DefaultPgPort),
				)
				out, derr := cmd.CombinedOutput()
				g.Expect(derr).NotTo(HaveOccurred(), "应当能从集群内 TCP 连通 %s: %s", dnsAddr, string(out))
			}, 2*time.Minute, 10*time.Second).Should(Succeed())
		})

		It("serverlessdb-IT-006：对已分配的 app_ref 重复申请返回 409", func() {
			createPoolAndAwaitWarm("pool-it006", 1)
			appRef := "app-it006"
			allocateAndAwaitRunning(appRef)

			By("用相同 app_ref 再次申请")
			_, status, err := apiClient.Allocate(ctx, testNs, &CreateInstanceRequest{
				AppRef:    appRef,
				PgVersion: pgVersion,
				Resources: InstanceResources{Cpu: allocateCpu, Memory: allocateMemory},
			})
			Expect(err).NotTo(HaveOccurred())
			Expect(status).To(Equal(409), "重复 app_ref 应当返回 409")
		})
	})
