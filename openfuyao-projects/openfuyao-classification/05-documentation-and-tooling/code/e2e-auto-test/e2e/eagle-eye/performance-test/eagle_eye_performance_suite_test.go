package eagle_eye_performance_test

import (
	"log"
	"testing"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file eagle-eye-config/.env.eagle-eye: %v", err)
	}
}

func TestEagleEyePerformance(t *testing.T) {
	gomega.RegisterFailHandler(ginkgo.Fail)
	ginkgo.RunSpecs(t, "Eagle Eye Performance Test Suite")
}
