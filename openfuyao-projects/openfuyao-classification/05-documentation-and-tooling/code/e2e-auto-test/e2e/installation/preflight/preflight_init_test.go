package preflight

import (
	"fmt"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

var _ = Describe("统一前置检查 init", Label("preflight", "pre-init", "26.09"), func() {
	var (
		ssh *executor.SSHExecutor
		h   *preflightHarness
	)

	// 用例之间无顺序依赖，可乱序执行；套件级资源懒初始化一次，统一由 AfterSuite 清理
	// init 用例在节点一（TEST_NODE1，已创建引导节点）上执行，期望全部通过
	BeforeEach(func() {
		if h == nil {
			ssh = newNodeExecutor(1)
			h = newInitPreflightHarness(ssh)
			Expect(h.hostIP).NotTo(BeEmpty(), "PREFLIGHT_INIT_HOST_IP 或 TEST_NODE1_IP 必须设置")
			h.tmpDir = h.createTempDir()
			registerPreflightSuiteCleanup(func() {
				h.cleanup()
				_ = ssh.Close()
			})
		}

		By("校验节点一上 bke 二进制可用")
		res := h.run("test -x " + shq(h.bke))
		Expect(res.ExitCode).To(Equal(0), "节点一上 bke 二进制应存在且可执行: %s", h.bke)
	})

	// 用例编号：PREFLIGHT-INIT-001（p0）
	// 前置：使用外部策略 preflight.yaml（cpu≥2、memory≥6GiB、systemDisk≥40GiB）
	// 步骤：执行 bke preflight init --preflight-config；读取 JSON 报告
	// 预期：Overall=PASS、exit 0，平台/资源/工具/Workspace/端口检查项全 PASS，报告含四态统计
	It("【PREFLIGHT-INIT-001】init 正常场景全量通过", Label("required_nodes:0", "PREFLIGHT-INIT-001", "P0", "online"),
		SpecTimeout(preflightInitItTimeout), func(ctx SpecContext) {
			By("执行 init 前置检查（外部策略 preflight.yaml）")
			policy := h.ensureDefaultPolicy()
			o := h.runInit("", policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "PASS", 0)

			By("校验报告四态统计与明细一致")
			expectSummaryArithmetic(o.Report)
			s := o.Report.Summary
			Expect(s.Fail).To(Equal(0), "不应有 FAIL 项")
			Expect(s.Warn).To(Equal(0), "不应有 WARN 项")
			Expect(s.Skip).To(Equal(0), "不应有 SKIP 项")
			Expect(s.Total).To(Equal(s.Pass), "全部检查项应为 PASS")

			By("校验平台/资源/工具/Workspace/端口检查项")
			expectAllResultsPass(o.Report)
			resItems := filterResults(o.Report, "INIT-RES")
			Expect(resItems).To(HaveLen(3), "应有 CPU/MEM/DISK 三项资源检查")
			platformItems := filterResults(o.Report, "PLAT")
			Expect(platformItems).NotTo(BeEmpty(), "应有平台发行版检查项")
			toolItems := filterResults(o.Report, "TOOL")
			Expect(toolItems).To(HaveLen(2), "应有 tar/systemctl 两项工具检查")
			Expect(allStatus(toolItems, "PASS")).To(BeTrue(), "工具检查应全部 PASS")
			wsItems := filterResults(o.Report, "WORKSPACE")
			Expect(wsItems).NotTo(BeEmpty(), "应有 Workspace 路径检查项")
			portItems := filterResults(o.Report, "INIT-PORT")
			Expect(portItems).To(HaveLen(5), "默认特性下应有 5 项端口检查(chart/console/http/k8s/registry)")
			Expect(allStatus(portItems, "PASS")).To(BeTrue(), "端口检查应全部 PASS")

			By("校验报告元数据")
			Expect(o.Report.Operation).To(Equal("init"), "operation 应为 init")
			Expect(o.Report.ConfigDigest).NotTo(BeEmpty(), "configDigest 不应为空")
			Expect(o.Report.BundleDigest).NotTo(BeEmpty(), "bundleDigest 不应为空")
		})

	// 用例编号：PREFLIGHT-INIT-002（p1）
	// 前置：同 INIT-001（含外部策略）；chart 端口 38080 空闲
	// 步骤：占用 38080 → 执行 init --preflight-config → 释放端口 → 重新执行
	// 预期：首次 Overall=FAIL、exit 3，仅端口项 FAIL 且 remediation 给出端口与释放建议；释放后重跑 PASS
	It("【PREFLIGHT-INIT-002】init 端口被占用时单项失败并给出释放建议", Label("required_nodes:0", "PREFLIGHT-INIT-002", "P1", "online"),
		SpecTimeout(preflightInitItTimeout), func(ctx SpecContext) {
			const chartPort = 38080
			policy := h.ensureDefaultPolicy()

			By("占用 chart 端口 38080")
			pid := occupyPortOn(ssh, chartPort)
			h.trackedPIDs[pid] = ssh
			Expect(waitPortUsedOn(ssh, chartPort)).To(BeTrue(), "端口 %d 应进入监听状态", chartPort)

			By("执行 init 前置检查，预期端口项 FAIL")
			o := h.runInit("", policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验仅端口单项 FAIL，其余 PASS")
			s := o.Report.Summary
			Expect(s.Fail).To(Equal(1), "应只有 1 项 FAIL，实际 %+v", s)
			portItems := filterResults(o.Report, "INIT-PORT")
			var failed *preflightResult
			for i := range portItems {
				if portItems[i].Status == "FAIL" {
					failed = &portItems[i]
					break
				}
			}
			Expect(failed).NotTo(BeNil(), "FAIL 项应为端口检查")
			Expect(failed.Name).To(ContainSubstring("38080"),
				"失败端口项应指向 38080，实际 %s", failed.Name)
			Expect(failed.Remediation).NotTo(BeEmpty(), "端口项应给出修复建议")
			Expect(failed.Remediation).To(Or(
				ContainSubstring("38080"),
				ContainSubstring("释放"),
				ContainSubstring("kill"),
			), "修复建议应包含端口与释放手段: %s", failed.Remediation)
			Expect(failed.Actual.Value).NotTo(BeNil(), "端口项应给出实际占用状态")

			By("释放端口后重新执行 init，预期 PASS")
			killPIDOn(ssh, pid)
			delete(h.trackedPIDs, pid)
			Expect(waitPortFreeOn(ssh, chartPort)).To(BeTrue(), "端口 %d 应已释放", chartPort)
			o2 := h.runInit("", policy)
			expectPreflightOutcome(o2, "PASS", 0)
			Expect(o2.Report.Summary.Fail).To(Equal(0), "释放后不应再有 FAIL 项")
		})

	// 用例编号：PREFLIGHT-INIT-003（p1）
	// 步骤：分别以 --kubernetesPort abc / 70000 执行 init（不传外部策略，测输入校验）
	// 预期：均 INPUT_ERROR、exit 4，输出"INPUT_ERROR"并定位字段与修复建议，不产生伪造检查项
	It("【PREFLIGHT-INIT-003】init 非法 kubernetesPort 输入返回 INPUT_ERROR", Label("required_nodes:0", "PREFLIGHT-INIT-003", "P1", "online"),
		SpecTimeout(preflightInitItTimeout), func(ctx SpecContext) {
			for _, badValue := range []string{"abc", "70000"} {
				By(fmt.Sprintf("以 --kubernetesPort %s 执行 init", badValue))
				o := h.runInit(badValue, "")

				Expect(o.ExitCode).To(Equal(4), "非法输入 %s 应返回 exit 4，实际 %d\nstdout:\n%s\nstderr:\n%s",
					badValue, o.ExitCode, o.Stdout, o.Stderr)

				By("校验输出提示 INPUT_ERROR 并定位字段")
				Expect(o.CombinedOutput()).To(ContainSubstring("INPUT_ERROR"),
					"输出应包含\"INPUT_ERROR\":\n%s", o.CombinedOutput())
				Expect(o.CombinedOutput()).To(MatchRegexp(`(?i)kubernetes\s*port`),
					"输出应定位到 kubernetesPort 字段:\n%s", o.CombinedOutput())

				By("校验不产生伪造检查项表")
				if o.Report != nil {
					Expect(o.Report.Overall).To(Equal("INPUT_ERROR"),
						"报告 overall 应为 INPUT_ERROR")
					Expect(o.Report.Results).To(BeEmpty(),
						"INPUT_ERROR 不应产生检查项明细，实际 %d 条", len(o.Report.Results))
					if o.Report.Summary.Total != 0 {
						Expect(o.Report.Results).To(BeEmpty(),
							"summary.total=%d 与空明细矛盾", o.Report.Summary.Total)
					}
				} else {
					GinkgoWriter.Printf("badValue=%s: 未生成报告文件（符合预期）\n", badValue)
				}
			}
		})

	// 用例编号：PREFLIGHT-INIT-004（p2）
	// 前置：外部策略——优先 PREFLIGHT_INIT_POLICY_TEMPLATE，否则使用包内 preflight.yaml
	// 步骤：复制策略并将 init 资源 cpu minimum 调整为 99 → 执行 init --preflight-config
	// 预期：Overall=FAIL、exit 3，CPU 项 FAIL，configDigest 与默认策略不同，其余资源项不受影响
	It("【PREFLIGHT-INIT-004】外部策略调整 CPU 阈值后检查失败", Label("required_nodes:0", "PREFLIGHT-INIT-004", "P2", "online"),
		SpecTimeout(preflightInitItTimeout), func(ctx SpecContext) {
			By("准备外部策略（env 覆盖优先，否则使用包内 preflight.yaml）")
			policySrc, usingFixture := h.policyFixturePath(h.policyTemplate,
				"preflight.yaml", preflightPolicyYAML)
			if !usingFixture {
				Expect(h.fileExists(policySrc)).To(BeTrue(), "策略模板不存在: %s", policySrc)
			}

		By("先以默认外部策略执行 init 获取基准 configDigest")
		base := h.runInit("", h.ensureDefaultPolicy())
			expectPreflightOutcome(base, "PASS", 0)

			By("复制外部策略并将 cpu minimum 调整为 99")
			policy := h.copyToTemp(policySrc, "policy-cpu99.yaml")
			applied := h.rewriteKey(policy, "minimum", "2", 1, "99")
			Expect(applied).To(BeTrue(),
				"策略中应存在 \"minimum: 2\"（init cpu 阈值）以支持阈值调整: %s", policySrc)

			By("以外部策略执行 init，预期 CPU 项 FAIL")
			o := h.runInit("", policy)
			h.skipIfPolicyRejected(o, usingFixture)
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 configDigest 与默认策略不同")
			Expect(o.Report.ConfigDigest).NotTo(BeEmpty(), "外部策略报告应包含 configDigest")
			Expect(o.Report.ConfigDigest).NotTo(Equal(base.Report.ConfigDigest),
				"外部策略的 configDigest 应区别于默认策略")

			By("校验仅 CPU 资源项 FAIL，其余不受影响")
			s := o.Report.Summary
			Expect(s.Fail).To(Equal(1), "应只有 CPU 一项 FAIL，实际 %+v", s)
			cpu := findResult(o.Report, "INIT-RES-CPU")
			Expect(cpu).NotTo(BeNil(), "应存在 CPU 资源检查项")
			Expect(cpu.Status).To(Equal("FAIL"), "CPU 资源项应为 FAIL")
			expCPU, ok := rawValueInt64(cpu.Expected.Value)
			Expect(ok).To(BeTrue(), "CPU 期望阈值应为数值: %s", rawValueString(cpu.Expected.Value))
			Expect(expCPU).To(Equal(int64(99)), "CPU 期望阈值应为调整后的 99")
			memDisk := filterResults(o.Report, "INIT-RES")
			for i := range memDisk {
				if strings.Contains(strings.ToUpper(memDisk[i].InstanceID), "CPU") {
					continue
				}
				Expect(memDisk[i].Status).To(Equal("PASS"),
					"资源项 %s 不应受 CPU 阈值调整影响", memDisk[i].InstanceID)
			}
		})

	// 用例编号：PREFLIGHT-INIT-006（p1）
	// 前置：离线物料在用例内生成——合法 tar.gz 制品包、自签 CA、节点一本机 TLS 监听充当真实可达仓库地址；
	//       与 INIT-001 相同传入外部策略 preflight.yaml（阈值放宽，避免环境规格误 FAIL）
	// 步骤：执行 init --preflight-config --imageFilePath/--otherRepo/--installConsole=false/--imageRepoCAFile →
	//       核对离线制品/CA/来源检查项 → 去掉离线三参数重跑核对检查项数量
	// 预期：首次 Overall=PASS、exit 0，INIT-OFFLINE-*/INIT-REPO-CA-*/INIT-SRC-* 全 PASS 且目标列指向实际物料；
	//       重跑后对应检查项为 0 条（条件实例化，不适用不产生 SKIP）
	It("【PREFLIGHT-INIT-006】init 离线模式正向场景与条件实例化验证", Label("required_nodes:0", "PREFLIGHT-INIT-006", "P1", "offline"),
		SpecTimeout(preflightInitItTimeout), func(ctx SpecContext) {
			By("生成离线物料：tar.gz 制品包 + 自签 CA + 本机可达仓库监听")
			policy := h.ensureDefaultPolicy()
			pkg := h.tmpDir + "/pf-e2e-offline.tar.gz"
			h.run(fmt.Sprintf("mkdir -p %s/pf-e2e-bundle && tar czf %s -C %s pf-e2e-bundle",
				shq(h.tmpDir), shq(pkg), shq(h.tmpDir)))
			repo := h.startLocalRepoListener()

			By("以离线参数执行 init 前置检查")
			o := h.runInitArgs(fmt.Sprintf("--preflight-config %s --imageFilePath %s --otherRepo %s --installConsole=false --imageRepoCAFile %s",
				shq(policy), shq(pkg), shq(repo.Addr), shq(repo.CAFile)))
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "PASS", 0)

			By("校验报告四态统计与明细一致")
			expectSummaryArithmetic(o.Report)
			Expect(o.Report.Summary.Fail).To(Equal(0), "不应有 FAIL 项")

			By("校验离线制品三项检查 PASS，目标列指向制品路径")
			offline := filterResults(o.Report, "OFFLINE")
			Expect(offline).To(HaveLen(3), "应有 EXISTS/READ/FORMAT 三项离线制品检查")
			Expect(allStatus(offline, "PASS")).To(BeTrue(), "离线制品检查应全部 PASS")
			for i := range offline {
				Expect(initResultTarget(&offline[i])).To(ContainSubstring(pkg),
					"离线制品项目标列应指向实际制品路径: %s", offline[i].InstanceID)
			}

			By("校验自定义 CA 三项检查 PASS，目标列指向 CA 文件")
			repoCA := filterResults(o.Report, "REPO-CA")
			Expect(repoCA).To(HaveLen(3), "应有 EXISTS/READ/PARSE 三项 CA 检查")
			Expect(allStatus(repoCA, "PASS")).To(BeTrue(), "CA 检查应全部 PASS")
			for i := range repoCA {
				Expect(initResultTarget(&repoCA[i])).To(ContainSubstring(repo.CAFile),
					"CA 项目标列应指向 CA 文件路径: %s", repoCA[i].InstanceID)
			}

			By("校验来源检查项 PASS，目标列指向仓库地址")
			src := filterResults(o.Report, "INIT-SRC")
			Expect(src).NotTo(BeEmpty(), "应存在来源检查项")
			Expect(allStatus(src, "PASS")).To(BeTrue(), "来源检查应全部 PASS")
			for i := range src {
				Expect(initResultTarget(&src[i])).To(ContainSubstring(repo.Addr),
					"来源项目标列应指向仓库地址: %s", src[i].InstanceID)
			}

			By("释放本机仓库监听")
			killPIDOn(h.ssh, repo.PID)
			delete(h.trackedPIDs, repo.PID)
			Expect(waitPortFreeOn(h.ssh, repo.Port)).To(BeTrue(), "仓库监听端口 %d 应已释放", repo.Port)

			By("去掉离线参数重跑，校验条件实例化语义")
			o2 := h.runInitArgs(fmt.Sprintf("--preflight-config %s --installConsole=false", shq(policy)))
			expectPreflightOutcome(o2, "PASS", 0)
			expectSummaryArithmetic(o2.Report)
			Expect(o2.Report.Summary.Fail).To(Equal(0), "重跑不应有 FAIL 项")
			Expect(o2.Report.Summary.Skip).To(Equal(0), "不适用检查项不实例化，不应产生 SKIP")
			Expect(filterResults(o2.Report, "OFFLINE")).To(BeEmpty(),
				"未传离线参数时不应有离线制品检查项")
			Expect(filterResults(o2.Report, "REPO-CA")).To(BeEmpty(),
				"未传 CA 参数时不应有 CA 检查项")
			for i := range o2.Report.Results {
				Expect(initResultTarget(&o2.Report.Results[i])).NotTo(ContainSubstring(repo.Addr),
					"重跑报告不应再出现仓库地址: %s", o2.Report.Results[i].InstanceID)
			}
		})
})
