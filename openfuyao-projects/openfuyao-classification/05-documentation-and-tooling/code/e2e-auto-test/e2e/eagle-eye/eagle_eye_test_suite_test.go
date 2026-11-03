package eagle_eye

import (
	"context"
	_ "embed"
	"k8s.io/client-go/util/homedir"
	"log"
	"os"
	"path/filepath"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	apiextensionsclient "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/eagle-eye/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
)

var (
	clientset           *kubernetes.Clientset
	cancel              context.CancelFunc
	dynamicClient       dynamic.Interface
	extensionClient     *apiextensionsclient.Clientset
	restConfig          *rest.Config
	ctx                 context.Context
	nodesNPUSituation   map[string]string // 记录节点是否有npu设备，没有的话标志为"none"
	nodesAscendChipType map[string]string // node name -> Ascend chip type
)

var _ = BeforeSuite(func() {
	var err error

	// 创建带取消的上下文
	ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
	DeferCleanup(cancel)

	kubeconfig := os.Getenv("KUBECONFIG")
	if kubeconfig == "" {
		kubeconfig = filepath.Join(homedir.HomeDir(), ".kube", "config")
	}

	// 初始化 restConfig
	restConfig, err = clientcmd.BuildConfigFromFlags("", kubeconfig)
	Expect(err).NotTo(HaveOccurred())

	// 创建 clientset
	clientset, err = kubernetes.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	// 创建动态客户端
	dynamicClient, err = dynamic.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	// 创建扩展客户端
	extensionClient, err = apiextensionsclient.NewForConfig(restConfig)
	Expect(err).NotTo(HaveOccurred())

	// 初始化node的NPU情况
	nodesNPUSituation, err = FindNodesNPUSituationEagleEye(clientset, ctx)
	Expect(err).NotTo(HaveOccurred())

	// 根据节点 InternalIP 从 AscendChipByIP 填充 nodesAscendChipType
	nodesAscendChipType = make(map[string]string)
	nodeList, err := clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	for i := range nodeList.Items {
		node := &nodeList.Items[i]
		var internalIP string
		for _, addr := range node.Status.Addresses {
			if addr.Type == corev1.NodeInternalIP {
				internalIP = addr.Address
				break
			}
		}
		if internalIP != "" && AscendChipByIP[internalIP] != "" {
			nodesAscendChipType[node.Name] = AscendChipByIP[internalIP]
		}
	}
})

var _ = AfterSuite(func() {
	if cancel != nil {
		cancel()
	}
})

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		log.Fatalf("Failed to load env file .env.eagle-eye: %v", err)
	}
}

func TestEagleEyeTestSuite(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Eagle-Eye Metrics Collection Suite")
}
