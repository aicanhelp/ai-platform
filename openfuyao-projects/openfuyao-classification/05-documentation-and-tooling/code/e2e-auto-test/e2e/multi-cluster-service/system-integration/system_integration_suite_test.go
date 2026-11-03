package system_integration

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
		log.Fatalf("Failed to load env file multi-cluster-config/.env.multi_cluster: %v", err)
	}
}

func TestSystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "SystemIntegration Suite")
}
