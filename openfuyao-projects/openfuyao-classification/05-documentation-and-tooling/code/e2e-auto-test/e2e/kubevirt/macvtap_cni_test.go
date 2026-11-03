package kubevirt

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
)

const macvtapNamespace = "default"
const macvtapDSName = "macvtap-cni"

var _ = Describe("Kubevirt macvtap CNI 检查",
	Label("kubevirt", "with-workload-cluster", "26.06"), Label("P0"), func() {

		It("鲲鹏-kubevirt-IT-003：检查 macvtap-cni DaemonSet 部署状态", func() {
			By("检查 macvtap-cni DaemonSet 存在且状态健康")
			assertDaemonSetHealthy(macvtapDSName, macvtapNamespace)

			By("检查 macvtap-cni 拉起的 Pod 均为 Running 状态")
			Eventually(func() bool {
				dsObj, err := k8sClient.Clientset.AppsV1().DaemonSets(macvtapNamespace).Get(
					ctx, macvtapDSName, metav1.GetOptions{},
				)
				if err != nil {
					return false
				}
				selector := labelsToSelector(dsObj.Spec.Selector.MatchLabels)
				list, err := k8sClient.Clientset.CoreV1().Pods(macvtapNamespace).List(ctx, metav1.ListOptions{
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
				"macvtap-cni 的所有 Pod 应当在 5 分钟内均达到 Running 状态")
		})
	})
