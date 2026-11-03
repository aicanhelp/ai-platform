package npu_dra_dfx

import (
	"context"
	"os"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

func init() {
	err := env.LoadEnv(".env")
	if err != nil {
		GinkgoWriter.Printf("WARNING: failed to load .env: %v\n", err)
	}
}

var _ = BeforeSuite(func() {
	var err error
	ctx, cancel = context.WithTimeout(context.Background(), 60*time.Minute)
	DeferCleanup(func() {
		if cancel != nil {
			cancel()
		}
	})

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred(), "k8s client 应当成功创建")

	localNodeName, err = os.Hostname()
	Expect(err).NotTo(HaveOccurred(), "获取本地主机名应当成功")
	Expect(localNodeName).NotTo(BeEmpty(), "本地主机名不应当为空")
	GinkgoWriter.Printf("本地主机名: %s\n", localNodeName)

	// 如果 hostname 不是 K8s 节点名，从 ResourceSlice 中获取
	slices, err := k8sClient.Clientset.ResourceV1().ResourceSlices().List(ctx, metav1.ListOptions{})
	if err == nil {
		for _, rs := range slices.Items {
			if rs.Spec.Driver == driverName && rs.Spec.NodeName != nil {
				localNodeName = *rs.Spec.NodeName
				break
			}
		}
	}
	GinkgoWriter.Printf("DRA 节点: %s\n", localNodeName)

	_, err = k8sClient.GetNamespace(ctx, testNamespace)
	if apierrors.IsNotFound(err) {
		Skip("ascend-dra namespace 不存在，跳过所有 NPU DRA Plugin DFX 用例")
	}
	Expect(err).NotTo(HaveOccurred(), "检查 ascend-dra namespace 应当成功")
	GinkgoWriter.Printf("namespace %s 已就绪\n", testNamespace)
})

func TestNpuDraDfx(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NpuDraPlugin DFX v26.06 Suite")
}
