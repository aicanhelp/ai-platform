package preflight

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}

func TestPreflight(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Preflight Suite")
}

// 用例之间无顺序依赖（不使用 Ordered），套件级资源统一在 AfterSuite 清理
var _ = AfterSuite(func() {
	runPreflightSuiteCleanup()
})
