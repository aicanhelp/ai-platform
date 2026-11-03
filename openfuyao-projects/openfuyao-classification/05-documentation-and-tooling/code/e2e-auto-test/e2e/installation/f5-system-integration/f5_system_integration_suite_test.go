package f5_system_integration

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	if err := env.LoadEnv(".env"); err != nil {
		log.Fatalf("Failed to load env file: %v", err)
	}
}

func TestF5SystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "F5SystemIntegration Suite")
}
