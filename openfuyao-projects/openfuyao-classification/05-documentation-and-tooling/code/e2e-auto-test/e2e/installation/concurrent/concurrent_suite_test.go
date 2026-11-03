package concurrent

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

func TestConcurrent(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Installation Concurrent Suite")
}
