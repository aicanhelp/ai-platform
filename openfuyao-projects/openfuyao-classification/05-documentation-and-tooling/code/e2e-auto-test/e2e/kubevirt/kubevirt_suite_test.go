package kubevirt

import (
	"context"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

const (
	kubevirtNamespace = "kubevirt"
	vmNamespace       = "default"
)

var (
	k8sClient *k8s.K8SClient
	ctx       context.Context
	cancel    context.CancelFunc
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
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

	_, err = k8sClient.GetNamespace(ctx, kubevirtNamespace)
	if apierrors.IsNotFound(err) {
		Skip("kubevirt namespace 不存在，跳过所有 kubevirt 用例")
	}
	Expect(err).NotTo(HaveOccurred(), "检查 kubevirt namespace 应当成功")
})

func TestKubevirt(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Kubevirt Suite")
}
