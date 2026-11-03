package infernex_checker

import (
	"context"
	"log"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

var (
	ctx    context.Context
	cancel context.CancelFunc
)

var _ = BeforeSuite(func() {
	ctx, cancel = context.WithTimeout(context.Background(), 4*time.Hour)
	DeferCleanup(cancel)
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func init() {
	if err := env.LoadEnv(".env"); err != nil {
		log.Fatalf("load .env: %v", err)
	}
}

func TestInfernexChecker(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Infernex Checker Suite")
}
