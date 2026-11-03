package npu_dra_plugin

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

const (
	testNamespace = "ascend-dra"
	driverName    = "npu.huawei.com"
	testPodImage  = "docker.io/library/ubuntu:22.04"
	kubeConfigPath = "/root/.kube/config"

	pluginNamespace = "npu-dra-driver"

	deviceClassFull       = "full.npu.huawei.com"
	deviceClassHard       = "hard.npu.huawei.com"
	deviceClassElastic    = "elastic.npu.huawei.com"
	deviceClassFixed      = "fixed.npu.huawei.com"
	deviceClassBestEffort = "best-effort.npu.huawei.com"

	pollInterval  = 5 * time.Second
	pollTimeout   = 300 * time.Second
	deleteTimeout = 120 * time.Second

	memCapacity4GiB  = 4 * 1024 * 1024 * 1024
	memCapacity8GiB  = 8 * 1024 * 1024 * 1024
	memCapacity16GiB = 16 * 1024 * 1024 * 1024
	memCapacity32GiB = 32 * 1024 * 1024 * 1024

	memCapacity4GiBMB  = 4 * 1024
	memCapacity16GiBMB = 16 * 1024
	memCapacity32GiBMB = 32 * 1024

	celPhysicalID0 = `device.attributes["npu.huawei.com"].physicalID == 0`
	celPhysicalID1 = `device.attributes["npu.huawei.com"].physicalID == 1`
	celPhysicalID2 = `device.attributes["npu.huawei.com"].physicalID == 2`
)

var (
	k8sClient     *k8s.K8SClient
	ctx           context.Context
	cancel        context.CancelFunc
	localNodeName string
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
		Skip("ascend-dra namespace 不存在，跳过所有 NPU DRA Plugin 用例")
	}
	Expect(err).NotTo(HaveOccurred(), "检查 ascend-dra namespace 应当成功")
	GinkgoWriter.Printf("namespace %s 已就绪\n", testNamespace)
})

func TestNpuDraPlugin(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "NpuDraPlugin v26.08 Suite")
}
