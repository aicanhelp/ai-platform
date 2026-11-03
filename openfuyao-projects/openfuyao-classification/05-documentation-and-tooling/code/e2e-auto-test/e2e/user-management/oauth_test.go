package usermanagement

import (
	"context"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/auth"
)

var _ = Describe("OAuth server", func() {
	It("【用户管理-007】重启认证服务器用户信息持久化存储", Label("用户管理-007", "P1"), func() {
		requireCluster()
		baseURL := auth.BaseURLFromEnv()
		if baseURL == "" {
			Skip("TEST_FUYAOURL or TEST_NODE2_IP is required")
		}
		password := testPassword()
		if password == "" {
			Skip("TEST_PASSWORD is empty")
		}

		ctx, cancel := ctxTimeout(5 * time.Minute)
		defer cancel()
		ns := oauthSystemNS()

		before, err := listOAuthServerPods(ctx, k8sClient, ns)
		Expect(err).NotTo(HaveOccurred())
		oldNames := runningOAuthPodNames(before)
		Expect(oldNames).NotTo(BeEmpty(), "no running oauth-server pod in ns %s", ns)

		for _, name := range oldNames {
			err := k8sClient.Clientset.CoreV1().Pods(ns).Delete(ctx, name, metav1.DeleteOptions{})
			Expect(err).NotTo(HaveOccurred(), "delete pod %s", name)
		}

		var ready corev1.Pod
		Eventually(func(g Gomega) {
			pods, err := listOAuthServerPods(ctx, k8sClient, ns)
			g.Expect(err).NotTo(HaveOccurred())
			found := false
			for _, p := range pods {
				if !oauthPodReady(p) {
					continue
				}
				replaced := true
				for _, old := range oldNames {
					if p.Name == old {
						replaced = false
						break
					}
				}
				if replaced {
					ready = p
					found = true
					break
				}
			}
			g.Expect(found).To(BeTrue(), "waiting for a new Ready oauth-server pod")
		}).WithContext(ctx).WithTimeout(3 * time.Minute).WithPolling(5 * time.Second).Should(Succeed())
		Expect(ready.Name).NotTo(BeEmpty())

		loginCtx, loginCancel := context.WithTimeout(ctx, 2*time.Minute)
		defer loginCancel()
		Eventually(func() error {
			c, err := auth.NewInsecureClient()
			if err != nil {
				return err
			}
			return auth.Login(c, baseURL, testUsername(), password)
		}).WithContext(loginCtx).WithTimeout(2 * time.Minute).WithPolling(5 * time.Second).Should(Succeed())
	})
})
