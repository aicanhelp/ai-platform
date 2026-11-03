package npu_operator

import (
	"context"
	"errors"
	"log"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/util/homedir"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	. "gitcode.com/openFuyao/e2e-auto-test/e2e/npu-operator/utils"
)

var (
	clientset         *kubernetes.Clientset
	cancel            context.CancelFunc
	dynamicClient     dynamic.Interface
	extensionClient   *apiextensionsclient.Clientset
	restConfig        *rest.Config
	ctx               context.Context
	nodesNPUSituation map[string]string
)

var _ = BeforeSuite(func() {
	var err error

	ctx, cancel = context.WithTimeout(context.Background(), 45*time.Minute)
	DeferCleanup(cancel)

	kubeconfig := filepath.Join(homedir.HomeDir(), ".kube", "config")

	restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred())

	clientset, err = kubernetes.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	dynamicClient, err = dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	extensionClient, err = apiextensionsclient.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	nodesNPUSituation, err = FindNodesNPUSituation(clientset, ctx)
	Expect(err).NotTo(HaveOccurred())
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func init() {
	loadOptionalEnvFile()
}

func loadOptionalEnvFile() {
	var candidatePaths []string
	customPath := strings.TrimSpace(os.Getenv("NPU_OPERATOR_ENV_FILE"))
	if customPath != "" {
		candidatePaths = append(candidatePaths, customPath)
	}
	candidatePaths = append(candidatePaths,
		".env",
		filepath.Join("npu-operator", ".env.npu-operator"),
		filepath.Join("npu-operator", "npu-operator-config", ".env.npu-operator"),
	)

	for _, path := range candidatePaths {
		err := env.LoadEnv(path)
		if err == nil {
			log.Printf("Loaded NPU Operator env file: %s", path)
			return
		}
		if errors.Is(err, os.ErrNotExist) {
			continue
		}

		log.Printf("Failed to load NPU Operator env file %s: %v", path, err)
		return
	}

	log.Printf("No .env.npu-operator found; SSH-dependent tests may skip")
}

func TestNpuOperatorTest(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NpuOperatorTest Suite")
}
