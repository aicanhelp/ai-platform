package bke_commands

import (
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
)

var _ = Describe("BKE 离线引导节点初始化测试", Label("offline", "pre-init"), func() {
	var sshExecutor *executor.SSHExecutor

	BeforeEach(func() {
		guide := config.LoadGuideNodeFromEnv()
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guide.Host, guide.Port, guide.Username, guide.Password)
		Expect(err).NotTo(HaveOccurred(), "连接引导节点失败")
		_, _ = sshExecutor.Exec("echo Y | bke reset --all --mount")
		time.Sleep(10 * time.Second)
	})

	AfterEach(func() {
		_, _ = sshExecutor.Exec("echo Y | bke reset --all --mount")
		time.Sleep(5 * time.Second)
	})

	runOfflineInit := func(args string) (*executor.ExecResult, error) {
		return sshExecutor.Exec(fmt.Sprintf("echo Y | bke init %s", args))
	}

	checkPodsReady := func() {
		Eventually(func() bool {
			result, err := sshExecutor.Exec("kubectl get pod -A --no-headers")
			if err != nil || result.ExitCode != 0 {
				return false
			}
			for _, line := range strings.Split(strings.TrimSpace(result.Stdout), "\n") {
				fields := strings.Fields(line)
				if len(fields) >= 4 && fields[3] != "Running" && fields[3] != "Completed" {
					return false
				}
			}
			return true
		}, 20*time.Minute, 10*time.Second).Should(BeTrue())
	}

	// Author: leoliu600
	It("离线初始化应该成功：指定正确的 oFVersion", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		result, err := runOfflineInit("--oFVersion latest")
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0))
		Expect(result.Stdout).To(ContainSubstring("BKE initialization is complete"))
		checkPodsReady()
	})

	// Author: leoliu600
	It("离线初始化应该失败：指定不存在的 oFVersion", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		result, _ := runOfflineInit("--oFVersion v99.99")
		Expect(result.Stdout + result.Stderr).To(MatchRegexp("(?i)failed to start cluster API"))
	})

	It("离线初始化应该成功：不指定 oFVersion", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		result, err := runOfflineInit("")
		Expect(err).NotTo(HaveOccurred())
		Expect(result.ExitCode).To(Equal(0))
		Expect(result.Stdout).To(ContainSubstring("BKE initialization is complete"))
		checkPodsReady()
	})
})
