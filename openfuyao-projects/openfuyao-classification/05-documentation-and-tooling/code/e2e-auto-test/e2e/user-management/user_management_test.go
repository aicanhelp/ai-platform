package usermanagement

import (
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("User CR", func() {
	It("【用户管理-005】验证用户账号信息安全存储", Label("用户管理-005", "P1"), func() {
		requireCluster()
		password := testPassword()
		if password == "" {
			Skip("TEST_PASSWORD is empty")
		}
		ctx, cancel := ctxTimeout(2 * time.Minute)
		defer cancel()

		enc, err := getUserEncryptedPassword(ctx, k8sClient, testUsername())
		Expect(err).NotTo(HaveOccurred())
		Expect(enc).NotTo(BeEmpty())
		Expect(verifyPBKDF2Password(enc, password)).To(Succeed())
	})
})
