package confidential_containers

import (
	"context"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

const (
	ccNamespace       = "confidential-containers-system"
	ccDeployName      = "cc-operator-controller-manager"
	trusteeNS         = "trustee-operator-system"
	trusteeDeployName = "trustee-deployment"
	trusteeImage      = "docker.io/library/trustee-operator:v0.3.0"
)

var (
	k8sClient *k8s.K8SClient
	localExec *executor.LocalExecutor
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

	localExec = executor.NewLocalExecutor(5 * time.Minute)

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")
})

func TestConfidentialContainers(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ConfidentialContainers Suite")
}
