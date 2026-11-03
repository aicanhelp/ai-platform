package usermanagement

import (
	"context"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

var (
	k8sClient *k8s.K8SClient
	suiteCtx  context.Context
)

func TestUserManagement(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "User-Management Suite")
}

var _ = BeforeSuite(func() {
	if err := env.LoadEnv(".env"); err != nil {
		log.Printf("load .env: %v (optional)", err)
	}
	suiteCtx = context.Background()
	if reason := skipNoKubeconfigReason(); reason != "" {
		log.Printf("BeforeSuite: %s — cluster tests will Skip", reason)
		return
	}
	var err error
	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "create k8s client")
})

func skipNoKubeconfigReason() string {
	if v := strings.TrimSpace(os.Getenv("KUBECONFIG")); v != "" {
		if _, err := os.Stat(v); err != nil {
			return "KUBECONFIG not readable: " + v
		}
		return ""
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "no home dir for kubeconfig"
	}
	p := filepath.Join(home, ".kube", "config")
	if _, err := os.Stat(p); err != nil {
		return "no kubeconfig at " + p
	}
	return ""
}

func requireCluster() {
	if reason := skipNoKubeconfigReason(); reason != "" {
		Skip("no kubeconfig: " + reason)
	}
	if k8sClient == nil {
		Skip("k8s client not initialized")
	}
}

func oauthSystemNS() string {
	if v := strings.TrimSpace(os.Getenv("OAUTH_SYSTEM_NAMESPACE")); v != "" {
		return v
	}
	return "openfuyao-system"
}

func testUsername() string {
	if v := strings.TrimSpace(os.Getenv("TEST_USERNAME")); v != "" {
		return v
	}
	return "admin"
}

func testPassword() string {
	return os.Getenv("TEST_PASSWORD")
}

func ctxTimeout(d time.Duration) (context.Context, context.CancelFunc) {
	return context.WithTimeout(suiteCtx, d)
}
