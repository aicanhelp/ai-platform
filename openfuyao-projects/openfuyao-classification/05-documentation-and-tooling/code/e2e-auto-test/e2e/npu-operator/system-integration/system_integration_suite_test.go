package npu_operator

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func TestNPUOperatorSystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NPU Operator System Integration Suite")
}

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file ../npu-operator-config/.env.npu-operator: %v", err)
	}
}
