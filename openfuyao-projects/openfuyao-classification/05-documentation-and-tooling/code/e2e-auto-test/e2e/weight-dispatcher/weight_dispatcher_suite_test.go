package weight_dispatcher

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	testenv "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/weight-dispatcher/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apiextensionsclientset "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"
)

var _ = BeforeSuite(func() {
	var err error

	ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
	DeferCleanup(cancel)

	if loadErr := testenv.LoadEnv(".env"); loadErr != nil {
		GinkgoWriter.Printf("未加载 e2e/.env，使用代码默认值和已有环境变量: %v\n", loadErr)
	}

	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		kubeconfig = filepath.Join(homedir.HomeDir(), ".kube", "config")
	}

	restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred())

	clientset, err = kubernetes.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	dynamicClient, err = dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	apiExtensionsClient, err = apiextensionsclientset.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	runtimeConfig, err := ResolveSuiteRuntimeConfig(clientset, ctx)
	Expect(err).NotTo(HaveOccurred())
	SetSuiteRuntimeConfig(runtimeConfig)
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func TestWeightDispatcher(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "WeightDispatcher Suite")
}
