package infernex

import (
	"context"
	"log"
	"os"
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

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	infutils "gitcode.com/openFuyao/e2e-auto-test/e2e/infernex/utils"
)

var (
	clientset     *kubernetes.Clientset
	dynamicClient dynamic.Interface
	restConfig    *rest.Config
	cancel        context.CancelFunc
	ctx           context.Context
)

var _ = BeforeSuite(func() {
	var err error

	ctx, cancel = context.WithTimeout(context.Background(), 4*time.Hour)
	DeferCleanup(cancel)

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

	Expect(infutils.EnsureNPUOperatorInstalled()).To(Succeed(), "npu-operator 应当已存在或安装成功")
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

func TestInfernexDeploy(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Infernex deploy suite")
}
