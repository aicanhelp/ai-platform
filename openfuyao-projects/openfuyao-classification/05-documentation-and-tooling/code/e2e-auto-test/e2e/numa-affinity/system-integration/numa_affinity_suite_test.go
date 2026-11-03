package numa_affinity

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
		log.Fatalf("Failed to load env file numa-affinity-config/.env.numa_affinity: %v", err)
	}
}

func TestSystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NumaAffinityPackage Suite")
}
