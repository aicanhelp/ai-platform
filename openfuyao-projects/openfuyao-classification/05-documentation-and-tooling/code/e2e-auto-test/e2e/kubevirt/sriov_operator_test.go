package kubevirt

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
)

const sriovNamespace = "sriov-network-operator"

var sriovDaemonSets = []string{"sriov-device-plugin", "sriov-network-config-daemon"}

var _ = Describe("Kubevirt SR-IOV Network Operator 检查",
	Label("kubevirt", "with-workload-cluster", "26.06"), Label("P0"), func() {

		It("鲲鹏-kubevirt-IT-005：检查 sriov-network-operator DaemonSet 部署状态", func() {
			for _, dsName := range sriovDaemonSets {
				By("检查 " + dsName + " DaemonSet 存在且状态健康")
				assertDaemonSetHealthy(dsName, sriovNamespace)

				By("检查 " + dsName + " 拉起的 Pod 均为 Running 状态")
				Eventually(func() bool {
					dsObj, err := k8sClient.Clientset.AppsV1().DaemonSets(sriovNamespace).Get(
						ctx, dsName, metav1.GetOptions{},
					)
					if err != nil {
						return false
					}
					selector := labelsToSelector(dsObj.Spec.Selector.MatchLabels)
					list, err := k8sClient.Clientset.CoreV1().Pods(sriovNamespace).List(ctx, metav1.ListOptions{
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
					dsName+" 的所有 Pod 应当在 5 分钟内均达到 Running 状态")
			}
		})
	})
