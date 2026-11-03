package infernex_elastic_scaler

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	esutils "gitcode.com/openFuyao/e2e-auto-test/e2e/elastic-scaler/utils"
	infutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var (
	clientset     *kubernetes.Clientset
	dynamicClient dynamic.Interface
	cancel        context.CancelFunc
	restConfig    *rest.Config
	ctx           context.Context
)

var _ = BeforeSuite(func() {
	var err error

	// 三个 Ordered Describe 串行，各自含多段 8～10 分钟等待，45 分钟会被后续场景误伤。
	ctx, cancel = context.WithTimeout(context.Background(), 4*time.Hour)
	DeferCleanup(cancel)

	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")

	restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred())

	clientset, err = kubernetes.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	dynamicClient, err = dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	Expect(infutils.EnsureNPUOperatorInstalled()).To(Succeed(), "npu-operator 应当已存在或安装成功")

	// HPA/APA CPU 扩缩容依赖 metrics.k8s.io
	esutils.EnsureMetricsServerAvailable(clientset, ctx)
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func TestInfernexElasticScaler(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Infernex ElasticScaler Suite")
}
