package local

import (
	"fmt"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/envtest"

	pluginv1beta1 "plugin-management-service/pkg/api/consoleplugin/v1beta1"
	"plugin-management-service/pkg/server/runtime"
)


func ensureAssets() {
	assets := os.Getenv("KUBEBUILDER_ASSETS")
	if assets == "" {
		Fail("envtest 需要 KUBEBUILDER_ASSETS（包含 etcd/kube-apiserver）")
	}
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

// 定义为包级变量，确保同包下的 plugin_test.go 可以访问
var (
	testEnv    *envtest.Environment
	cfg        *rest.Config
	client     dynamic.Interface
	httpServer *httptest.Server
	baseURL    string
)

func TestFunctionalLocal(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "plugin-management-service functional/local suite")
}

var _ = BeforeSuite(func() {
	ensureAssets()
	By("starting envtest")
	crdPath := filepath.Join("..", "..", "..", "charts", "plugin-management-service", "crds")
	testEnv = &envtest.Environment{
		CRDDirectoryPaths: []string{crdPath},
	}

	var err error
	cfg, err = testEnv.Start()
	Expect(err).NotTo(HaveOccurred())
	Expect(cfg).NotTo(BeNil())

	client, err = dynamic.NewForConfig(cfg)
	Expect(err).NotTo(HaveOccurred())

	By("starting http server with real routes")
	ws := runtime.GetPluginWebService()
	pluginv1beta1.BindPluginRoute(ws, cfg)

	container := restful.NewContainer()
	container.Router(restful.CurlyRouter{})
	container.Add(ws)

	httpServer = httptest.NewServer(container)
	baseURL = httpServer.URL
})

var _ = AfterSuite(func() {
	By("stopping http server")
	if httpServer != nil {
		httpServer.Close()
	}

	By("stopping envtest")
	if testEnv != nil {
		_ = testEnv.Stop()
	}
})
