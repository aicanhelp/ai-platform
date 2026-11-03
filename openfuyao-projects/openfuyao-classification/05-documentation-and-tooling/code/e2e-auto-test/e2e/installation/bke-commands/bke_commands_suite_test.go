package bke_commands

import (
	"log"
	"testing"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func init() {
	if err := env.LoadEnv(".env"); err != nil {
		log.Fatalf("Failed to load .env file: %v", err)
	}
}
func TestBKECommands(t *testing.T) { RegisterFailHandler(Fail); RunSpecs(t, "BKE Commands Suite") }
