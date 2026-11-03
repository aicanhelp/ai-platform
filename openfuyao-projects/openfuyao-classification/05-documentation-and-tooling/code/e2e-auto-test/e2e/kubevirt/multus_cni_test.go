package kubevirt

import (
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
)

const multusNamespace = "kube-system"

func labelsToSelector(labels map[string]string) string {
	parts := make([]string, 0, len(labels))
	for k, v := range labels {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ",")
}

func assertDaemonSetHealthy(dsName, ns string) {
	getter := k8sClient.Clientset.AppsV1().DaemonSets(ns)
	Expect(getter).NotTo(BeNil())

	dsObj, err := getter.Get(ctx, dsName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred(), "获取 DaemonSet %s/%s 应当成功", ns, dsName)

	Expect(dsObj.Status.DesiredNumberScheduled).To(BeNumerically(">", 0),
		"DaemonSet %s/%s 应当在至少一个节点上调度", ns, dsName)
	Expect(dsObj.Status.NumberReady).To(Equal(dsObj.Status.DesiredNumberScheduled),
		"DaemonSet %s/%s 应当所有 Pod 均 Ready", ns, dsName)
	Expect(dsObj.Status.UpdatedNumberScheduled).To(Equal(dsObj.Status.DesiredNumberScheduled),
		"DaemonSet %s/%s 应当已完成更新", ns, dsName)
}

func assertDaemonSetPodsRunning(dsName, ns string) {
	var err error
	dsObj, err := k8sClient.Clientset.AppsV1().DaemonSets(ns).Get(ctx, dsName, metav1.GetOptions{})
	Expect(err).NotTo(HaveOccurred())

	selector := labelsToSelector(dsObj.Spec.Selector.MatchLabels)
	podList, err := k8sClient.Clientset.CoreV1().Pods(ns).List(ctx, metav1.ListOptions{
		LabelSelector: selector,
	})
	Expect(err).NotTo(HaveOccurred())
	Expect(podList.Items).NotTo(BeEmpty(), "DaemonSet %s/%s 应当拉起至少一个 Pod", ns, dsName)

	for _, pod := range podList.Items {
		Expect(pod.Status.Phase).To(Equal(corev1.PodRunning),
			"Pod %s/%s 应当处于 Running 状态，当前为 %s", ns, pod.Name, pod.Status.Phase)
		Expect(podFunc.VerifyPodStatus(pod)).To(BeTrue(),
			"Pod %s/%s 的所有容器应当 Ready", ns, pod.Name)
	}
}

var _ = Describe("Kubevirt Multus CNI 厚插件检查",
	Label("kubevirt", "with-workload-cluster", "26.06"), Label("P0"), func() {

		It("鲲鹏-kubevirt-IT-001：检查 Multus CNI 厚插件部署状态", func() {
			By("检查 kube-multus-ds DaemonSet 存在且状态健康")
			assertDaemonSetHealthy("kube-multus-ds", multusNamespace)

			By("检查 kube-multus-ds 拉起的 Pod 均为 Running 状态")
			Eventually(func() bool {
				dsObj, err := k8sClient.Clientset.AppsV1().DaemonSets(multusNamespace).Get(
					ctx, "kube-multus-ds", metav1.GetOptions{},
				)
				if err != nil {
					return false
				}
				selector := labelsToSelector(dsObj.Spec.Selector.MatchLabels)
				list, err := k8sClient.Clientset.CoreV1().Pods(multusNamespace).List(ctx, metav1.ListOptions{
					LabelSelector: selector,
				})
				if err != nil || len(list.Items) == 0 {
					return false
				}
				for _, p := range list.Items {
					if p.Status.Phase != corev1.PodRunning || !podFunc.VerifyPodStatus(p) {
						return false
					}
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
				"kube-multus-ds 的所有 Pod 应当在 5 分钟内均达到 Running 状态")

			By("检查 dynamic-networks-controller-ds DaemonSet 存在且状态健康")
			assertDaemonSetHealthy("dynamic-networks-controller-ds", multusNamespace)

			By("检查 dynamic-networks-controller-ds 拉起的 Pod 均为 Running 状态")
			Eventually(func() bool {
				dsObj, err := k8sClient.Clientset.AppsV1().DaemonSets(multusNamespace).Get(
					ctx, "dynamic-networks-controller-ds", metav1.GetOptions{},
				)
				if err != nil {
					return false
				}
				selector := labelsToSelector(dsObj.Spec.Selector.MatchLabels)
				list, err := k8sClient.Clientset.CoreV1().Pods(multusNamespace).List(ctx, metav1.ListOptions{
					LabelSelector: selector,
				})
				if err != nil || len(list.Items) == 0 {
					return false
				}
				for _, p := range list.Items {
					if p.Status.Phase != corev1.PodRunning || !podFunc.VerifyPodStatus(p) {
						return false
					}
				}
				return true
			}, 5*time.Minute, 10*time.Second).Should(BeTrue(),
				"dynamic-networks-controller-ds 的所有 Pod 应当在 5 分钟内均达到 Running 状态")

			assertDaemonSetPodsRunning("kube-multus-ds", multusNamespace)
			assertDaemonSetPodsRunning("dynamic-networks-controller-ds", multusNamespace)
		})
	})
