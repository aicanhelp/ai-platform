package kae_operator

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

func TestKaeOperatorSystemIntegration(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Kae-operator System Integration Suite")
}

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file kae-operator-config/.env.kae-operator: %v", err)
	}
}
