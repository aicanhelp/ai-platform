package local

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	"application-management-service/pkg/helm"
	"application-management-service/test/functional/local/chartgetter"
)

var (
	testEnv *envtest.Environment
	cfg     *rest.Config
)

// ensureAssets gives a clearer error when KUBEBUILDER_ASSETS is not configured.
func ensureAssets() {
	assets := os.Getenv("KUBEBUILDER_ASSETS")
	if assets == "" {
		Fail("envtest 需要 KUBEBUILDER_ASSETS（包含 etcd/kube-apiserver）")
	}
	// Windows/CI 环境下给一个更友好的提示
	for _, bin := range []string{"etcd", "kube-apiserver"} {
		if _, err := os.Stat(filepath.Join(assets, bin+exeSuffix())); err != nil {
			Fail(fmt.Sprintf("KUBEBUILDER_ASSETS=%s 但未找到 %s，可执行文件缺失：%v", assets, bin, err))
		}
	}
}

func exeSuffix() string {
	if os.PathSeparator == '\\' {
		return ".exe"
	}
	return ""
}

var _ = BeforeSuite(func(ctx SpecContext) {
	ensureAssets()
	By("starting envtest control plane")
	testEnv = &envtest.Environment{}

	var err error
	cfg, err = testEnv.Start()
	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	// Windows 上偶发需要稍等 apiserver ready
	time.Sleep(800 * time.Millisecond)
}, NodeTimeout(180*time.Second))

var _ = AfterSuite(func() {
	By("stopping envtest control plane")
	if testEnv != nil {
		_ = testEnv.Stop()
	}
})

func newOpForTest(cg *chartgetter.FakeChartGetter) (helm.Operation, error) {
	return helm.NewHelmOperationWithDeps(cfg, cg)
}
