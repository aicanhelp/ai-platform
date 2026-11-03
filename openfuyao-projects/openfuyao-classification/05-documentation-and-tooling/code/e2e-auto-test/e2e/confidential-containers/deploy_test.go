package confidential_containers

import (
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	podFunc "gitcode.com/openFuyao/e2e-auto-test/e2e/framework/pod"
)

var _ = Describe("鲲鹏机密容器环境部署",
	Label("confidential-containers", "with-workload-cluster"), Label("P0"), func() {

		It("鲲鹏-机密容器-IT-001：部署鲲鹏机密容器环境", func() {

			By("检查当前节点 containerd 中是否存在 trustee-operator:v0.3.0 镜像")
			result, err := localExec.Exec("ctr -n k8s.io image ls -q")
			Expect(err).NotTo(HaveOccurred(), "ctr image ls 命令应执行成功")

			images := strings.Split(strings.TrimSpace(result.Stdout), "\n")
			imageFound := false
			for _, img := range images {
				if strings.TrimSpace(img) == trusteeImage {
					imageFound = true
					break
				}
			}
			if !imageFound {
				Skip("当前节点 containerd 中未找到 " + trusteeImage + " 镜像，跳过用例")
			}

			startTime := time.Now()

			// ===== Step 1: 部署 confidential-containers operator =====
			By("执行 kubectl apply -k 部署 confidential-containers operator")
			applyResult, err := localExec.Exec(
				"kubectl apply -k github.com/confidential-containers/operator/config/release?ref=v0.13.0")
			Expect(err).NotTo(HaveOccurred(),
				"kubectl apply 应当成功，stdout: %s, stderr: %s",
				applyResult.Stdout, applyResult.Stderr)

			DeferCleanup(func() {
				By("清理：删除 confidential-containers operator")
				localExec.Exec(
					"kubectl delete -k github.com/confidential-containers/operator/config/release?ref=v0.13.0 --ignore-not-found")
			})

			By("检查 Deployment cc-operator-controller-manager 是否存在")
			Eventually(func() error {
				_, getErr := k8sClient.Clientset.AppsV1().
					Deployments(ccNamespace).Get(ctx, ccDeployName, metav1.GetOptions{})
				return getErr
			}, 3*time.Minute, 10*time.Second).Should(Succeed(),
				"Deployment %s/%s 应在 3 分钟内出现", ccNamespace, ccDeployName)

			By("检查 cc-operator-controller-manager 拉起的 Pod 均为 Running 状态")
			Eventually(func() bool {
				deploy, getErr := k8sClient.Clientset.AppsV1().
					Deployments(ccNamespace).Get(ctx, ccDeployName, metav1.GetOptions{})
				if getErr != nil || deploy.Spec.Replicas == nil || *deploy.Spec.Replicas == 0 {
					return false
				}
				selector := labelsToSelector(deploy.Spec.Selector.MatchLabels)
				podList, listErr := k8sClient.Clientset.CoreV1().
					Pods(ccNamespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
				if listErr != nil || len(podList.Items) == 0 {
					return false
				}
				for _, p := range podList.Items {
					if p.Status.Phase != corev1.PodRunning || !podFunc.VerifyPodStatus(p) {
						return false
					}
				}
				return true
			}, 10*time.Minute, 10*time.Second).Should(BeTrue(),
				"cc-operator-controller-manager 的所有 Pod 应在 10 分钟内达到 Running 状态")

			// ===== Step 2: 部署 kata 组件 =====
			By("部署 /root/kata/deploy.yaml")
			deployResult, err := localExec.Exec("kubectl apply -f /root/kata/deploy.yaml")
			Expect(err).NotTo(HaveOccurred(),
				"kubectl apply -f deploy.yaml 应当成功，stdout: %s, stderr: %s",
				deployResult.Stdout, deployResult.Stderr)

			DeferCleanup(func() {
				By("清理：删除 /root/kata/deploy.yaml")
				localExec.Exec("kubectl delete -f /root/kata/deploy.yaml --ignore-not-found")
			})

			By("部署 /root/kata/kbsconfig.yaml")
			kbsResult, err := localExec.Exec("kubectl apply -f /root/kata/kbsconfig.yaml")
			Expect(err).NotTo(HaveOccurred(),
				"kubectl apply -f kbsconfig.yaml 应当成功，stdout: %s, stderr: %s",
				kbsResult.Stdout, kbsResult.Stderr)

			DeferCleanup(func() {
				By("清理：删除 /root/kata/kbsconfig.yaml")
				localExec.Exec("kubectl delete -f /root/kata/kbsconfig.yaml --ignore-not-found")
			})

			By("检查 Deployment trustee-deployment 是否存在")
			Eventually(func() error {
				_, getErr := k8sClient.Clientset.AppsV1().
					Deployments(trusteeNS).Get(ctx, trusteeDeployName, metav1.GetOptions{})
				return getErr
			}, 3*time.Minute, 10*time.Second).Should(Succeed(),
				"Deployment %s/%s 应在 3 分钟内出现", trusteeNS, trusteeDeployName)

			By("检查 trustee-deployment 拉起的 Pod 均为 Running 状态")
			Eventually(func() bool {
				deploy, getErr := k8sClient.Clientset.AppsV1().
					Deployments(trusteeNS).Get(ctx, trusteeDeployName, metav1.GetOptions{})
				if getErr != nil || deploy.Spec.Replicas == nil || *deploy.Spec.Replicas == 0 {
					return false
				}
				selector := labelsToSelector(deploy.Spec.Selector.MatchLabels)
				podList, listErr := k8sClient.Clientset.CoreV1().
					Pods(trusteeNS).List(ctx, metav1.ListOptions{LabelSelector: selector})
				if listErr != nil || len(podList.Items) == 0 {
					return false
				}
				for _, p := range podList.Items {
					if p.Status.Phase != corev1.PodRunning || !podFunc.VerifyPodStatus(p) {
						return false
					}
				}
				return true
			}, 10*time.Minute, 10*time.Second).Should(BeTrue(),
				"trustee-deployment 的所有 Pod 应在 10 分钟内达到 Running 状态")

			// ===== Step 3: 校验总耗时 =====
			By("校验从部署 operator 到所有组件就绪的总耗时小于 10 分钟")
			elapsed := time.Since(startTime)
			Expect(elapsed).To(BeNumerically("<", 10*time.Minute),
				"从部署 operator 到所有组件就绪的总耗时应小于 10 分钟，实际耗时: %v", elapsed)
		})
	})

func labelsToSelector(labels map[string]string) string {
	parts := make([]string, 0, len(labels))
	for k, v := range labels {
		parts = append(parts, k+"="+v)
	}
	return strings.Join(parts, ",")
}
