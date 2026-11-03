package infernex_checker

import (
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

var _ = Describe("Infernex software pre-flight checks", Label("infernex_checker", "precheck"), func() {
	Describe("Deploy command modes", Ordered, Label("deploy"), func() {
		It("ic_deploy_001-全量模式（all）终端输出三层结果", Label("P0"), func() {
			result := runInfernexCheckCached("all", true, true)
			output := result.stdout

			Expect(result.err).NotTo(HaveOccurred(), "infernex-check all should run successfully, stderr: %s", result.stderr)
			Expect(result.report).NotTo(BeNil(), "all mode should write JSON report")
			if result.report.HardwareTerminated {
				Skip("current environment has no node passing hardware layer; all mode terminates before K8s and Config-Env layers")
			}
			if result.report.K8s == nil {
				Skip("all mode did not reach K8s layer in the current environment")
			}
			if result.report.ConfigEnv == nil {
				Skip("all mode did not reach Config-Env layer in the current environment; three-layer output requires hardware and K8s prerequisites to pass")
			}

			hardwareIdx := strings.Index(output, "[Hardware Layer")
			k8sIdx := strings.Index(output, "[K8s Layer]")
			configEnvIdx := strings.Index(output, "[Config-Env Layer")

			Expect(hardwareIdx).To(BeNumerically(">=", 0), "terminal output should contain Hardware Layer result")
			Expect(k8sIdx).To(BeNumerically(">=", 0), "terminal output should contain K8s Layer result")
			Expect(configEnvIdx).To(BeNumerically(">=", 0), "terminal output should contain Config-Env Layer result")
			Expect(hardwareIdx).To(BeNumerically("<", k8sIdx), "Hardware Layer should be printed before K8s Layer")
			Expect(k8sIdx).To(BeNumerically("<", configEnvIdx), "K8s Layer should be printed before Config-Env Layer")

			Expect(output).To(MatchRegexp(`(?m)^\s*\S+\s+H-(0[1-9]|10)\s+.+`),
				"terminal output should contain hardware check items in '<icon> <ID> <message>' format")
			Expect(output).To(MatchRegexp(`(?m)^\s*\S+\s+K-0[1-4]\s+.+`),
				"terminal output should contain k8s check items in '<icon> <ID> <message>' format")
			Expect(output).To(MatchRegexp(`(?m)^\s*\S+\s+B-0[1-2]\s+.+`),
				"terminal output should contain config-env check items in '<icon> <ID> <message>' format")

			if hasFailedCheckLine(output) {
				Expect(hasSuggestionLine(output)).To(BeTrue(), "failed check items should print suggestion line")
			}

			Expect(output).To(MatchRegexp(`Result:\s+\d+\s+passed,\s+\d+\s+failed,\s+\d+\s+warning,\s+\d+\s+info`),
				"terminal output should contain final summary")
		})

		It("ic_deploy_002-全量模式所有节点硬件层失败时终端提示并终止后续层", Label("P0"), func() {
			result := runInfernexCheckCached("all", true, true)
			Expect(result.err).NotTo(HaveOccurred(), "infernex-check all should run successfully, stderr: %s", result.stderr)
			Expect(result.report).NotTo(BeNil(), "all mode should write JSON report")

			if !result.report.HardwareTerminated {
				Skip("current environment has at least one node passing hardware layer; all-hardware-failed scenario requires a fault environment or mock test")
			}

			Expect(result.stdout).To(ContainSubstring("All-check terminated"))
			Expect(result.report.Hardware).NotTo(BeNil())
			Expect(result.report.K8s).To(BeNil())
			Expect(result.report.ConfigEnv).To(BeNil())
		})

		It("ic_deploy_003-仅硬件模式（hardware）只执行H-01~H-10", Label("P0"), func() {
			result := runInfernexCheckCached("hardware", false, false)
			Expect(result.err).NotTo(HaveOccurred(), "infernex-check hardware should run successfully, stderr: %s", result.stderr)
			Expect(result.stdout).To(ContainSubstring("[Hardware Layer"))
			Expect(result.stdout).NotTo(ContainSubstring("[K8s Layer]"))
			Expect(result.stdout).NotTo(ContainSubstring("[Config-Env Layer"))
			Expect(result.report).NotTo(BeNil())
			Expect(result.report.Hardware).NotTo(BeNil())
			Expect(result.report.K8s).To(BeNil())
			Expect(result.report.ConfigEnv).To(BeNil())
		})

		It("ic_deploy_004-仅k8s模式（k8s）只执行K-01~K-04", Label("P0"), func() {
			result := runInfernexCheckCached("k8s", false, true)
			Expect(result.err).NotTo(HaveOccurred(), "infernex-check k8s should run successfully, stderr: %s", result.stderr)
			Expect(result.stdout).NotTo(ContainSubstring("[Hardware Layer"))
			Expect(result.stdout).To(ContainSubstring("[K8s Layer]"))
			Expect(result.stdout).NotTo(ContainSubstring("[Config-Env Layer"))
			Expect(result.report).NotTo(BeNil())
			Expect(result.report.Hardware).To(BeNil())
			Expect(result.report.K8s).NotTo(BeNil())
			Expect(result.report.ConfigEnv).To(BeNil())
		})

		It("ic_deploy_005-仅configenv模式（config-env）只执行B-01~B-02", Label("P0"), func() {
			result := runInfernexCheckCached("config-env", true, false)
			Expect(result.err).NotTo(HaveOccurred(), "infernex-check config-env should run successfully, stderr: %s", result.stderr)
			Expect(result.stdout).NotTo(ContainSubstring("[Hardware Layer"))
			Expect(result.stdout).NotTo(ContainSubstring("[K8s Layer]"))
			Expect(result.stdout).To(ContainSubstring("[Config-Env Layer"))
			Expect(result.report).NotTo(BeNil())
			Expect(result.report.Hardware).To(BeNil())
			Expect(result.report.K8s).To(BeNil())
			Expect(result.report.ConfigEnv).NotTo(BeNil())
		})

		It("ic_deploy_006-配置--output路径时生成合法JSON报告", Label("P0"), func() {
			result := runInfernexCheckCached("all", true, true)
			Expect(result.err).NotTo(HaveOccurred(), "infernex-check all should run successfully, stderr: %s", result.stderr)
			Expect(result.report).NotTo(BeNil(), "--output should write a valid JSON report")
			Expect(result.report.Summary.Total).To(Equal(result.report.Summary.Passed + result.report.Summary.Failed + result.report.Summary.Warning + result.report.Summary.Info + result.report.Summary.Skipped))
			Expect(result.stdout).To(ContainSubstring("JSON result written to:"))
		})
	})
})
