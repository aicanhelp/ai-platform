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

var _ = Describe("BKE 引导节点初始化测试", Label("guide-install", "pre-init", "online"), func() {
	var (
		sshExecutor *executor.SSHExecutor
		guideNode   *config.GuideNodeConfig
	)

	BeforeEach(func() {
		// 加载引导节点配置
		guideNode = config.LoadGuideNodeFromEnv()
		var err error
		sshExecutor, err = executor.NewSSHExecutor(guideNode.Host, guideNode.Port, guideNode.Username, guideNode.Password)
		Expect(err).NotTo(HaveOccurred(), "连接引导节点失败")

		By("重置引导节点环境 (bke reset)")

		resetCmd := "echo Y | bke reset --all --mount"
		result, err := sshExecutor.Exec(resetCmd)
		if err != nil {
			GinkgoWriter.Printf("Reset 命令执行返回错误 (可能环境已清理): %v\nStderr: %s\n", err, result.Stderr)
		} else {
			GinkgoWriter.Printf("Reset 命令执行成功: %s\n", result.Stdout)
		}

		time.Sleep(10 * time.Second)
	})

	AfterEach(func() {
		By("测试后清理：重置引导节点环境")
		resetCmd := "echo Y | bke reset --all --mount"
		result, err := sshExecutor.Exec(resetCmd)
		if err != nil {
			GinkgoWriter.Printf("AfterEach Reset 失败: %v\nStderr: %s\n", err, result.Stderr)
		} else {
			GinkgoWriter.Printf("AfterEach Reset 成功: %s\n", result.Stdout)
		}
		time.Sleep(5 * time.Second)
	})

	runInit := func(args string) (*executor.ExecResult, error) {
		// 使用 echo Y 自动确认初始化操作
		cmd := fmt.Sprintf("echo Y | bke init %s", args)
		GinkgoWriter.Printf("执行安装命令: %s\n", cmd)

		res, err := sshExecutor.Exec(cmd)

		if res != nil {
			lines := strings.Split(res.Stdout, "\n")
			start := 0
			if len(lines) > 20 {
				start = len(lines) - 20
			}
			GinkgoWriter.Printf("====== Last 20 lines of Stdout ======\n%s\n", strings.Join(lines[start:], "\n"))

			if res.Stderr != "" {
				GinkgoWriter.Printf("====== Stderr ======\n%s\n", res.Stderr)
			}
			GinkgoWriter.Printf("=====================================\n")
		}
		return res, err
	}

	checkPodsReady := func() {
		By("检查引导集群 Pod 状态")
		Eventually(func() bool {
			// 获取所有 Pod
			cmd := "kubectl get pod -A --no-headers"
			res, err := sshExecutor.Exec(cmd)
			if err != nil || res.ExitCode != 0 {
				return false
			}

			lines := strings.Split(strings.TrimSpace(res.Stdout), "\n")
			if len(lines) == 0 {
				return false
			}

			allReady := true
			for _, line := range lines {
				if strings.TrimSpace(line) == "" {
					continue
				}
				fields := strings.Fields(line)
				if len(fields) < 4 {
					continue
				}
				status := fields[3]
				// 允许的状态: Running 或 Completed
				if status != "Running" && status != "Completed" {
					GinkgoWriter.Printf("Pod 未就绪: %s (Status: %s)\n", fields[1], status)
					allReady = false
				}
			}
			return allReady
		}, 20*time.Minute, 10*time.Second).Should(BeTrue(), "引导集群所有 Pod 应该处于 Running 或 Completed 状态")
	}

	// 用例名称：应该成功初始化：指定正确的 oFVersion 和 versionUrl
	// 用例步骤：1) 通过引导节点执行 `bke init`，传入正确的 oFVersion 与 versionUrl；2) 校验初始化命令退出码为 0；3) 校验日志包含初始化完成信息；4) 等待引导集群 Pod 就绪。
	// 预期结果：初始化成功，`res.Stdout` 包含 `BKE initialization is complete`，且引导集群所有 Pod 处于 Running/Completed。
	// Author: leoliu600
	It("应该成功初始化：指定正确的 oFVersion 和 versionUrl", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		args := "--onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --oFVersion latest --versionUrl https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/version-config/"

		res, err := runInit(args)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.ExitCode).To(Equal(0), "安装命令应该成功退出")
		Expect(res.Stdout).To(ContainSubstring("BKE initialization is complete"), "日志应包含初始化完成信息")

		checkPodsReady()
	})

	// 用例名称：应该成功初始化：指定正确的 oFVersion，使用默认 versionUrl
	// 用例步骤：1) 通过引导节点执行 `bke init`，仅指定正确的 oFVersion；2) 校验命令退出码为 0；3) 校验日志包含初始化完成信息；4) 等待引导集群 Pod 就绪。
	// 预期结果：初始化成功，`BKE initialization is complete` 出现在日志中，且引导集群 Pod 均就绪。
	// Author: leoliu600
	It("应该成功初始化：指定正确的 oFVersion，使用默认 versionUrl", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		args := "--onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --oFVersion latest"

		res, err := runInit(args)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.ExitCode).To(Equal(0))
		Expect(res.Stdout).To(ContainSubstring("BKE initialization is complete"))

		checkPodsReady()
	})

	// 用例名称：应该初始化失败：指定正确的 oFVersion 但 versionUrl 错误
	// 用例步骤：1) 通过引导节点执行 `bke init`，指定 oFVersion 与错误的 versionUrl；2) 执行安装命令；3) 检查输出日志中包含 cluster API 启动失败信息（大小写不敏感）。
	// 预期结果：初始化失败，`failed to start cluster API` 相关错误信息出现在 stdout 或 stdout+stderr。
	// Author: leoliu600
	It("应该初始化失败：指定正确的 oFVersion 但 versionUrl 错误", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		args := "--onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --oFVersion latest --versionUrl https://openfuyao.obs.cn-north-4.myhuaweicloud.com/openFuyao/ext-components/version-config/error/"

		res, err := runInit(args)
		if err == nil && res.ExitCode == 0 {
			Expect(res.Stdout).To(MatchRegexp("(?i)failed to start cluster API"))
		} else {
			output := res.Stdout + res.Stderr
			Expect(output).To(MatchRegexp("(?i)failed to start cluster API"))
		}
	})

	// 用例名称：应该初始化失败：指定不存在的 oFVersion
	// 用例步骤：1) 通过引导节点执行 `bke init`，传入一个不存在的 oFVersion；2) 等待命令结束并收集 stdout/stderr。
	// 预期结果：初始化失败，日志中包含 `failed to start cluster API`，并在必要时包含 `version ... not in released version list`。
	// Author: leoliu600
	It("应该初始化失败：指定不存在的 oFVersion", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		args := "--onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest --oFVersion v25.11"

		res, err := runInit(args)

		if err == nil && res.ExitCode == 0 {
			// 验证日志中确实包含错误信息
			Expect(res.Stdout).To(MatchRegexp("(?i)failed to start cluster API"))
		} else {
			output := res.Stdout + res.Stderr
			Expect(output).To(MatchRegexp("(?i)failed to start cluster API"))
			Expect(output).To(ContainSubstring("version v25.11 not in released version list"))
		}
	})

	// 用例名称：应该成功初始化：不指定 oFVersion（使用默认/latest）
	// 用例步骤：1) 通过引导节点执行 `bke init`，不显式指定 oFVersion；2) 校验初始化命令退出码为 0；3) 校验日志包含初始化完成信息；4) 等待引导集群 Pod 就绪。
	// 预期结果：初始化成功，stdout 包含 `BKE initialization is complete`，且引导集群 Pod 处于 Running/Completed。
	// Author: leoliu600
	It("应该成功初始化：不指定 oFVersion (使用默认/latest)", Label("required_nodes:0"), SpecTimeout(InstallationItTimeout), func(ctx SpecContext) {
		args := "--onlineImage cr.openfuyao.cn/openfuyao/bke-online-installed:latest"

		res, err := runInit(args)
		Expect(err).NotTo(HaveOccurred())
		Expect(res.ExitCode).To(Equal(0))
		Expect(res.Stdout).To(ContainSubstring("BKE initialization is complete"))

		checkPodsReady()
	})
})
