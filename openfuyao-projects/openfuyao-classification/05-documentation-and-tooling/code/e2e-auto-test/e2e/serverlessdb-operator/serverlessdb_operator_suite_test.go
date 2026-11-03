package serverlessdb_operator

import (
	"context"
	"fmt"
	"net/http"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/serverlessdb-operator/utils"
)

var (
	k8sClient    *k8s.K8SClient
	ctx          context.Context
	cancel       context.CancelFunc
	apiClient    *APIClient
	opNs         string
	testNs       string
	tenantToken  string
	curLocalPort int
	pfStop       func()
)

func init() {
	if err := env.LoadEnv(".env"); err != nil {
		GinkgoWriter.Printf("WARNING: failed to load .env: %v\n", err)
	}
}

var _ = BeforeSuite(func() {
	var err error
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Minute)
	DeferCleanup(func() {
		if cancel != nil {
			cancel()
		}
	})

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	// Gate: the operator must already be deployed (pre-deployed verification mode,
	// no install/uninstall in-suite). Skip the whole suite if not found.
	opNs = EnvOr("SERVERLESSDB_OPERATOR_NAMESPACE", DefaultOpNs)
	pods, err := k8sClient.ListPods(ctx, opNs, metav1.ListOptions{LabelSelector: OperatorPodLabel})
	Expect(err).NotTo(HaveOccurred(), "列出 operator pod 应当成功")
	ready := false
	for i := range pods.Items {
		if IsPodReady(&pods.Items[i]) {
			ready = true
			break
		}
	}
	if !ready {
		Skip(fmt.Sprintf("未发现运行中的 serverlessdb-operator (selector %q in %q)，跳过用例", OperatorPodLabel, opNs))
	}

	testNs = TestNamespace
	Expect(k8sClient.CreateNamespaceIfNotExists(ctx, testNs)).To(Succeed(), "创建测试 namespace 应当成功")
	DeferCleanup(func() {
		Expect(k8sClient.DeleteNamespace(ctx, testNs)).To(Succeed(), "删除测试 namespace 应当成功")
	})

	cleanupLeftoverCRs()

	// Port-forward operator gin port to a local random port.
	var localPort int
	localPort, stopPF, err := StartPortForward(k8sClient, ctx, opNs, OperatorPodLabel, OperatorGinPort)
	Expect(err).NotTo(HaveOccurred(), "建立 port-forward 应当成功")
	curLocalPort = localPort
	pfStop = stopPF
	DeferCleanup(func() {
		if pfStop != nil {
			pfStop()
		}
	})

	Eventually(func(g Gomega) {
		resp, err := http.Get(fmt.Sprintf("http://localhost:%d/api/v1/instances", curLocalPort))
		g.Expect(err).NotTo(HaveOccurred())
		if resp != nil {
			resp.Body.Close()
		}
	}, 60*time.Second, 3*time.Second).Should(Succeed(), "operator HTTP API 应当可达")

	// Dedicated tenant SA + Role + token (namespace-scoped, auto-cleaned with testNs).
	token, err := CreateTenantSA(ctx, k8sClient, testNs)
	Expect(err).NotTo(HaveOccurred(), "创建租户 SA 应当成功")
	tenantToken = token

	apiClient = NewAPIClient(fmt.Sprintf("http://localhost:%d", curLocalPort), tenantToken)
	Expect(apiClient).NotTo(BeNil())

	By("serverlessdb-operator suite 就绪")
})

func TestServerlessDBOperator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ServerlessDB Operator Suite")
}
