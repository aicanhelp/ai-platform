package hermesrouter

import (
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var _ = BeforeSuite(func() {
	Expect(utils.EnsureNPUOperatorInstalled()).To(Succeed(), "npu-operator 应当已存在或安装成功")
})

// TestHermesRouter 是 Hermes Router 集成测试套件入口
func TestHermesRouter(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "HermesRouter Suite")
}
