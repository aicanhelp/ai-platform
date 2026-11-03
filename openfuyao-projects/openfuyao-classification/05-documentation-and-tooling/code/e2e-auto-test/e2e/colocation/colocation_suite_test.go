package colocation

import (
	"context"
	"fmt"
	"testing"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/colocation/utils"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/env"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/k8s"
)

// helmChartKeyInYAML 与 framework/helm/helm_config.yaml 中的 key 一致；换 chart 时只改这里或改用下方环境变量覆盖。
const helmChartKeyInYAML = "numa-affinity-package"

var (
	k8sClient *k8s.K8SClient
	cmdExec   *executor.LocalExecutor
	ctx       context.Context
	cancel    context.CancelFunc
)

var _ = BeforeSuite(func() {
	By("初始化客户端")
	err := env.LoadEnv(".env")
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to load .env file: %v", err))

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred())

	By("清理历史测试遗留资源")
	cleanupPods := []string{"hls-pod", "ls-pod", "ls-pod-1", "ls-pod-2", "be-pod", "be-pod-1", "be-pod-2", "stress-ng-1"}
	for _, podName := range cleanupPods {
		_ = k8sClient.Clientset.CoreV1().Pods("default").Delete(context.Background(), podName, metav1.DeleteOptions{})
	}
	time.Sleep(2 * time.Second)

	cmdExec = executor.NewLocalExecutor(1 * time.Minute)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to load .env file: %v", err))

	k8sClient, err = k8s.NewK8SClientFromLocalKubeconfig()
	Expect(err).NotTo(HaveOccurred())

	cmdExec = executor.NewLocalExecutor(1 * time.Minute)

	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Minute)

	By("确认Volcano未安装")
	exists, err := HelmReleaseExists(ctx, NUMAAffinityHelmConfig)
	Expect(err).NotTo(HaveOccurred())
	if exists {
		Expect(HelmUninstallLocal(ctx, NUMAAffinityHelmConfig)).To(Succeed())
	}

	By("安装在离线混部")
	exists, err = HelmReleaseExists(ctx, ColocationHelmConfig)
	Expect(err).NotTo(HaveOccurred())
	if exists {
		Expect(HelmUninstallLocal(ctx, ColocationHelmConfig)).To(Succeed())
	}
	Expect(HelmInstallLocal(ctx, ColocationHelmConfig)).To(Succeed())
	Expect(WaitHelmReleaseDeployed(ctx, ColocationHelmConfig["releaseName"], ColocationHelmConfig["namespace"], 90*time.Second, 5*time.Second)).To(Succeed())
	Expect(WaitAllPodsRunningOrSucceeded(k8sClient, ctx, "openfuyao-colocation", 90*time.Second, 5*time.Second)).To(Succeed())

	By("启用节点混部")
	cm, err := GetColocationConfigMap(k8sClient, ctx)
	Expect(err).NotTo(HaveOccurred())
	nodeList, err := k8sClient.Clientset.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
	Expect(err).NotTo(HaveOccurred())
	nodesConfig := ""
	for i, node := range nodeList.Items {
		if i > 0 {
			nodesConfig += ","
		}
		nodesConfig += fmt.Sprintf("{\"name\":\"%s\"}", node.Name)
	}
	cm.Data["colocation-options"] = fmt.Sprintf("{\"enable\":true,\"nodesConfig\":[%s]}", nodesConfig)
	_, err = k8sClient.Clientset.CoreV1().ConfigMaps("openfuyao-colocation").Update(ctx, cm, metav1.UpdateOptions{})
	Expect(err).NotTo(HaveOccurred())

	// Test 2, 3, 4
	By("【在离线混部-002/003/004】验证未安装volcano，HLS/LS/BE业务不能部署成功，调度器使用的还是volcano")
	func() {
		hlsPod, err := InstallHLSPod(k8sClient, ctx, "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(hlsPod.Namespace).Delete(ctx, hlsPod.Name, metav1.DeleteOptions{})
		Expect(WaitPodCondition(k8sClient, ctx, hlsPod.Name, hlsPod.Namespace, "Pending", 30*time.Second)).To(BeTrue())
		Expect(hlsPod.Spec.SchedulerName).To(Equal("volcano"))

		lsPod, err := InstallLSPod(k8sClient, ctx, "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(lsPod.Namespace).Delete(ctx, lsPod.Name, metav1.DeleteOptions{})
		Expect(WaitPodCondition(k8sClient, ctx, lsPod.Name, lsPod.Namespace, "Pending", 30*time.Second)).To(BeTrue())
		Expect(lsPod.Spec.SchedulerName).To(Equal("volcano"))

		bePod, err := InstallBEPod(k8sClient, ctx, "", "")
		Expect(err).NotTo(HaveOccurred())
		defer k8sClient.Clientset.CoreV1().Pods(bePod.Namespace).Delete(ctx, bePod.Name, metav1.DeleteOptions{})
		Expect(WaitPodCondition(k8sClient, ctx, bePod.Name, bePod.Namespace, "Pending", 30*time.Second)).To(BeTrue())
		Expect(bePod.Spec.SchedulerName).To(Equal("volcano"))
	}()

	By("安装Volcano")
	Expect(HelmInstallLocal(ctx, NUMAAffinityHelmConfig)).To(Succeed())
	Expect(WaitHelmReleaseDeployed(ctx, NUMAAffinityHelmConfig["releaseName"], NUMAAffinityHelmConfig["namespace"], 90*time.Second, 5*time.Second)).To(Succeed())
	Expect(WaitAllPodsRunningOrSucceeded(k8sClient, ctx, "volcano-system", 90*time.Second, 5*time.Second)).To(Succeed())

	By("等待Volcano webhook服务就绪")
	Expect(WaitVolcanoWebhookReady(k8sClient, ctx, 60*time.Second)).To(Succeed())
	Expect(WaitVolcanoSchedulerReady(k8sClient, ctx, 60*time.Second)).To(Succeed())
})

var _ = AfterSuite(func() {
	Expect(HelmUninstallLocal(ctx, ColocationHelmConfig)).To(Succeed())
	Expect(HelmUninstallLocal(ctx, NUMAAffinityHelmConfig)).To(Succeed())
	cancel()
})

func TestColocation(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "Colocation Suite")
}
