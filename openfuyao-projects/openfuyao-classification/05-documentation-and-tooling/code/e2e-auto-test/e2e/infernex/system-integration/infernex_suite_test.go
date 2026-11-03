package infernex

import (
	"log"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file infernex-config/.env.infernex: %v", err)
	}
}

var _ = BeforeSuite(func() {
	Expect(utils.EnsureNPUOperatorInstalled()).To(Succeed(), "npu-operator 应当已存在或安装成功")
})

func TestAiInference(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "AiInference Suite")
}
