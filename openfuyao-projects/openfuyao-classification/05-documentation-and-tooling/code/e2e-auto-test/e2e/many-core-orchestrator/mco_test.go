package mco

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"

	. "gitcode.com/openFuyao/e2e-auto-test/e2e/many-core-orchestrator/utils"
)

// 集群级集成测试需要 KVM 物理机环境（Kata Pod 创建要求 /dev/kvm 支持）。
// 在纯虚拟机集群上运行时，IT-006/008~012 均可能无法正常通过。
var _ = Describe("众核 Kata 集群集成测试", Ordered, Label("many-core-orchestrator", "with-workload-cluster", "physical-machine", "skip-temporarily"), func() {
	var kataPodName string

	BeforeAll(func(ctx SpecContext) {
		kataPodName = fmt.Sprintf("kata-test-%d", time.Now().UnixNano())

		By("确认历史 helm release 不存在，若存在则卸载")
		Expect(UninstallExistingReleaseAndWait(ctx, WaitKataDeployResourcesDeleted)).To(Succeed())

		By("清理可能由 filter 测试残留的非 Helm 管理的 RuntimeClass")
		cleanupOrphanedRuntimeClasses(ctx)

		By("众核-KATA-IT-008 Helm 安装 kata-deploy（仅 kata 阶段，关闭 collector/analyzer/mcoPlugin）")
		chartPath := GetChartPath()
		Expect(HelmInstallLocalChart(ctx, MCOHelmConfig, chartPath, KataPhaseHelmSets...)).To(Succeed())
		Expect(WaitHelmReleaseDeployed(ctx, MCOHelmConfig["releaseName"], 120*time.Second, 5*time.Second)).To(Succeed())
		Expect(WaitAllPodsRunningOrSucceeded(k8sClient, ctx, MCODeployNamespace, 120*time.Second, 5*time.Second)).To(Succeed())
	}, NodeTimeout(5*time.Minute))

	It("众核-KATA-IT-009 RuntimeClass 注册验证", Label("physical-machine"), func(ctx SpecContext) {
		By("查询已注册的 RuntimeClass")
		rcList, err := ListRuntimeClasses(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		var foundQemu, foundClh bool
		for _, item := range rcList.Items {
			name := item.GetName()
			if name == KataRuntimeClassQemu {
				foundQemu = true
			}
			if name == KataRuntimeClassClh {
				foundClh = true
			}
		}

		Expect(foundQemu).To(BeTrue(), "kata-qemu RuntimeClass 应存在")
		Expect(foundClh).To(BeTrue(), "kata-clh RuntimeClass 应存在")

		By("验证 kata-qemu RuntimeClass 的 handler 字段")
		rc, err := GetRuntimeClass(k8sClient, ctx, KataRuntimeClassQemu)
		Expect(err).NotTo(HaveOccurred())
		handler, _, _ := unstructured.NestedString(rc.Object, "handler")
		Expect(handler).NotTo(BeEmpty(), "RuntimeClass handler 不应为空")
	}, NodeTimeout(1*time.Minute))

	It("众核-KATA-IT-010 Kata Pod 创建与运行", Label("physical-machine"), func(ctx SpecContext) {
		By("创建 runtimeClassName=kata-qemu 的测试 Pod")
		_, err := CreateKataTestPod(k8sClient, ctx, kataPodName, KataTestNamespace, KataRuntimeClassQemu)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, kataPodName, KataTestNamespace, 30*time.Second)
		})

		By("等待 Kata Pod Running")
		Expect(WaitPodCondition(k8sClient, ctx, kataPodName, KataTestNamespace, "Running", 120*time.Second)).To(BeTrue(), "Kata Pod 应进入 Running 状态")

		By("检查 Pod 事件无 RuntimeClass 相关错误")
		freshPod, err := k8sClient.Clientset.CoreV1().Pods(KataTestNamespace).Get(ctx, kataPodName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())

		events, err := k8sClient.Clientset.CoreV1().Events(KataTestNamespace).List(ctx, metav1.ListOptions{
			FieldSelector: fmt.Sprintf("involvedObject.kind=Pod,involvedObject.name=%s", kataPodName),
		})
		Expect(err).NotTo(HaveOccurred())
		for _, evt := range events.Items {
			Expect(evt.Message).NotTo(ContainSubstring("RuntimeClass"), "Pod 事件不应有 RuntimeClass 相关错误: %s", evt.Message)
			Expect(evt.Message).NotTo(ContainSubstring("runtime handler"), "Pod 事件不应有 runtime handler 相关错误: %s", evt.Message)
		}

		By("验证 Pod 绑定到 kata-ready=true 的节点")
		kataReadyNodes, err := ListNodesWithLabel(k8sClient, ctx, KataReadyLabelKey, "true")
		Expect(err).NotTo(HaveOccurred())
		if len(kataReadyNodes) > 0 {
			Expect(freshPod.Spec.NodeName).To(BeElementOf(kataReadyNodes), "Kata Pod 应绑定到 kata-ready=true 的节点")
		}
	}, NodeTimeout(5*time.Minute))

	It("众核-KATA-IT-011 Kata 运行时实际校验", Label("physical-machine"), func(ctx SpecContext) {
		By("创建 runtimeClassName=kata-qemu 的测试 Pod 用于运行时校验")
		verifyPodName := fmt.Sprintf("kata-verify-%d", time.Now().UnixNano())
		_, err := CreateKataTestPod(k8sClient, ctx, verifyPodName, KataTestNamespace, KataRuntimeClassQemu)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, verifyPodName, KataTestNamespace, 30*time.Second)
		})

		By("等待 Kata Pod Running")
		Expect(WaitPodCondition(k8sClient, ctx, verifyPodName, KataTestNamespace, "Running", 120*time.Second)).To(BeTrue(), "Kata Pod 应进入 Running 状态")

		By("获取 Kata Pod 所在节点")
		nodeName, err := GetPodNodeName(k8sClient, ctx, verifyPodName, KataTestNamespace)
		Expect(err).NotTo(HaveOccurred())
		Expect(nodeName).NotTo(BeEmpty(), "Kata Pod 应已调度到节点")

		By(fmt.Sprintf("通过 SSH 登录节点 %s 校验 Kata 运行时进程", nodeName))
		Expect(VerifyKataRuntimeOnNode(k8sClient, ctx, nodeName, verifyPodName, KataTestNamespace)).To(Succeed(),
			"应在节点 %s 上检测到 kata-runtime/kata-proxy/kata-shim/qemu 等进程，确认 Pod 实际运行在 Kata 虚拟机环境中", nodeName)
	}, NodeTimeout(3*time.Minute))

	It("众核-KATA-IT-006 无 KVM 节点预检失败", Label("physical-machine"), func(ctx SpecContext) {
		By("查找 kata-ready=false 的节点")
		notReadyNodes, err := ListNodesWithLabel(k8sClient, ctx, KataReadyLabelKey, "false")
		Expect(err).NotTo(HaveOccurred())

		if len(notReadyNodes) == 0 {
			Skip("当前集群所有节点均满足 KVM 条件，无法验证无 KVM 节点预检失败场景；此用例需在有非 KVM 节点的物理机集群上手动测试")
		}

		By("查看 kata-ready=false 节点上的 kata-deploy Pod initContainer 日志")
		pods, err := GetKataDeployPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		for _, pod := range pods.Items {
			if pod.Spec.NodeName == "" {
				continue
			}
			val, exists, _ := GetNodeLabelValue(k8sClient, ctx, pod.Spec.NodeName, KataReadyLabelKey)
			if exists && val == "false" {
				logs, err := GetInitContainerLogs(k8sClient, ctx, pod.Name, MCODeployNamespace, KataPreflightContainerName)
				if err == nil {
					Expect(logs).To(ContainSubstring("fail"), "无 KVM 节点的 preflight 日志应提示检查失败")
				}
			}
		}
	}, NodeTimeout(1*time.Minute))

	It("众核-KATA-IT-007 混合节点预检", Label("physical-machine"), func(ctx SpecContext) {
		By("查找 kata-ready=true 和 kata-ready=false 的节点")
		readyNodes, err := ListNodesWithLabel(k8sClient, ctx, KataReadyLabelKey, "true")
		Expect(err).NotTo(HaveOccurred())
		notReadyNodes, err := ListNodesWithLabel(k8sClient, ctx, KataReadyLabelKey, "false")
		Expect(err).NotTo(HaveOccurred())

		if len(readyNodes) == 0 || len(notReadyNodes) == 0 {
			Skip(fmt.Sprintf("当前集群不满足混合节点条件（kata-ready=true: %d, kata-ready=false: %d），此用例需在混合 KVM/非KVM 节点的物理机集群上测试",
				len(readyNodes), len(notReadyNodes)))
		}

		By("查看 kata-deploy DaemonSet 各节点 Pod 状态与 initContainer 日志")
		pods, err := GetKataDeployPods(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		checkedReady, checkedNotReady := 0, 0
		for _, pod := range pods.Items {
			if pod.Spec.NodeName == "" {
				continue
			}
			val, exists, _ := GetNodeLabelValue(k8sClient, ctx, pod.Spec.NodeName, KataReadyLabelKey)
			if !exists {
				continue
			}
			logs, err := GetInitContainerLogs(k8sClient, ctx, pod.Name, MCODeployNamespace, KataPreflightContainerName)
			if err != nil {
				fmt.Fprintf(GinkgoWriter, "WARNING: 获取节点 %s 的 initContainer 日志失败: %v\n", pod.Spec.NodeName, err)
				continue
			}
			if val == "true" {
				By(fmt.Sprintf("验证 KVM 节点 %s: 预检通过", pod.Spec.NodeName))
				Expect(logs).NotTo(ContainSubstring("fail"), "KVM 节点 %s 的 preflight 日志不应提示失败", pod.Spec.NodeName)
				checkedReady++
			} else if val == "false" {
				By(fmt.Sprintf("验证非 KVM 节点 %s: 预检失败", pod.Spec.NodeName))
				Expect(logs).To(ContainSubstring("fail"), "非 KVM 节点 %s 的 preflight 日志应提示检查失败", pod.Spec.NodeName)
				checkedNotReady++
			}
		}
		Expect(checkedReady).To(BeNumerically(">", 0), "应至少验证 1 个 KVM 节点")
		Expect(checkedNotReady).To(BeNumerically(">", 0), "应至少验证 1 个非 KVM 节点")
	}, NodeTimeout(2*time.Minute))

	It("众核-KATA-IT-012 卸载与资源清理", Label("physical-machine"), func(ctx SpecContext) {
		By("删除测试 Kata Pod")
		Expect(DeletePod(k8sClient, ctx, kataPodName, KataTestNamespace, 30*time.Second)).To(Succeed())

		By("记录卸载前存在的 kata 相关资源")
		resourcesBefore, err := GetKataDeployResources(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(resourcesBefore)).To(BeNumerically(">=", 1), "卸载前应存在 kata 相关资源")

		By("执行 helm uninstall")
		Expect(UninstallReleaseAndWait(ctx, WaitKataDeployResourcesDeleted)).To(Succeed())

		By("检查 Helm 管理的资源被正确清理")
		resourcesAfter, err := GetKataDeployResources(k8sClient, ctx)
		Expect(err).NotTo(HaveOccurred())

		helmManagedKeys := []string{
			"DaemonSet/kata-deploy",
		}
		for _, key := range helmManagedKeys {
			Expect(resourcesAfter).NotTo(HaveKey(key), "卸载后 %s 应被删除", key)
		}

		By("验证普通 runc Pod 不受卸载影响")
		runcPodName := fmt.Sprintf("runc-test-%d", time.Now().UnixNano())
		_, err = CreateRuncTestPod(k8sClient, ctx, runcPodName, KataTestNamespace)
		Expect(err).NotTo(HaveOccurred())
		DeferCleanup(func(ctx SpecContext) {
			_ = DeletePod(k8sClient, ctx, runcPodName, KataTestNamespace, 30*time.Second)
		})

		Expect(WaitPodCondition(k8sClient, ctx, runcPodName, KataTestNamespace, "Running", 60*time.Second)).To(BeTrue(), "runc Pod 应正常 Running")

		freshRuncPod, err := k8sClient.Clientset.CoreV1().Pods(KataTestNamespace).Get(ctx, runcPodName, metav1.GetOptions{})
		Expect(err).NotTo(HaveOccurred())
		Expect(freshRuncPod.Status.Phase).To(Equal(corev1.PodRunning), "runc Pod 应不受 Kata 卸载影响")
	}, NodeTimeout(5*time.Minute))
})

func cleanupOrphanedRuntimeClasses(ctx context.Context) {
	for _, rc := range []string{"kata-clh", "kata-qemu"} {
		cmd := exec.CommandContext(ctx, "kubectl", "delete", "runtimeclass", rc, "--ignore-not-found")
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(GinkgoWriter, "WARNING: cleanup RuntimeClass %s failed: %v\n", rc, err)
		}
	}
}
