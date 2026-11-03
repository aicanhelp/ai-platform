package mco

import (
	"context"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

var (
	k8sClient *k8s.K8SClient
	ctx       context.Context
	cancel    context.CancelFunc
)

var _ = BeforeSuite(func() {
	By("加载环境变量")
	err := env.LoadEnv(".env")
	Expect(err).NotTo(HaveOccurred())

	By("初始化 Kubernetes 客户端")
	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred())

	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Minute)
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func TestMCO(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "MCO Suite")
}
