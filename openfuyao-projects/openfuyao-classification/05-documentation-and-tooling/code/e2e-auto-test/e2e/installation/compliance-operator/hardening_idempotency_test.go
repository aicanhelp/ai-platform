package compliance_operator

import (
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// CISSTIG_DFX_02 已加固集群重复加固幂等零修改（可靠性，挂 E2E-A 加固态之后、E2E-C 之前）。
// CISSTIG_DFX_03 回退后重复回退幂等零副作用（可靠性，挂 E2E-C 统一回退之后）。
// 两个 It 组均规划于 hardening_idempotency 模块（It[repeat-harden] / It[repeat-rollback]）。
func defineIdempotencyHardenSpecs() {
	Describe("[DFX][DFX-02] Repeated hardening idempotency",
		Ordered, Label("DFX", "DFX-02", "hardening_idempotency", "repeat-harden", "P1", "26.09", "online"), func() {

			var (
				fpHardened utils.FingerprintSet
				repeatOut  string
				ts2        string
			)

			BeforeAll(func() {
				requireHardened("CISSTIG_DFX_02")
				By("采集加固态指纹基线")
				if shared.hardenedFingerprints != nil {
					fpHardened = shared.hardenedFingerprints
				} else {
					fpHardened = collectAllFingerprints()
				}
			})

			It("[DFX][DFX-02] should repeat hardening without pending changes or cluster side effects", func() {
				By("再次执行 bash bin/k8s-hardening.sh cluster-harden config/cluster.yaml -y")
				res := utils.ClusterHarden(entry, cfg, "-y")
				repeatOut = res.Stdout + "\n" + res.Stderr
				Expect(res.ExitCode).To(BeZero(), fmt.Sprintf("重复加固应成功退出（exit=0），输出尾部:\n%s", tail(repeatOut)))

				By("断言各节点输出无待整改项跳过加固，不执行任何整改动作")
				Expect(strings.Count(repeatOut, utils.MsgNoPendingNode)).To(BeNumerically(">=", len(cfg.Nodes)),
					"4 节点均应输出「无待整改项（全部已合规或高危跳过），跳过加固执行」")
				Expect(repeatOut).NotTo(ContainSubstring("加固失败"))
				Expect(repeatOut).NotTo(ContainSubstring(utils.MsgSSHFailed))

				By("解析本轮统一时间戳 TS2（供 changes.json 校验）")
				ts2 = utils.ParseUnifiedTimestamp(repeatOut)
				Expect(ts2).NotTo(BeEmpty(), "重复加固亦应下发统一备份时间戳")

				By("读取最新集群统一报告 JSON")
				_, raw, err := utils.LatestReport(entry, cfg.ReportDir)
				Expect(err).NotTo(HaveOccurred())
				report, err := utils.ParseReport(raw)
				Expect(err).NotTo(HaveOccurred())
				states := report.NodeStates()
				for _, n := range cfg.Nodes {
					Expect(states).To(HaveKeyWithValue(n.IP, "no_pending_compliant"),
						fmt.Sprintf("节点 %s state 应为 no_pending_compliant（实际 %s，非 failed/no_result_data）", n.IP, states[n.IP]))
				}

				By("校验每个节点均生成空变更集")
				for _, n := range cfg.Nodes {
					Expect(utils.BackupExists(nodeExecs[n.IP], cfg.BackupDir, ts2)).To(BeTrue(),
						fmt.Sprintf("节点 %s 应生成本轮统一备份 %s（含已合规节点）", n.IP, ts2))
					changes, err := utils.ExecTrim(nodeExecs[n.IP],
						fmt.Sprintf("cat %s/%s/changes.json 2>/dev/null", utils.ShellPath(cfg.BackupDir), ts2))
					Expect(err).NotTo(HaveOccurred())
					Expect(strings.TrimSpace(changes)).To(Equal("[]"),
						fmt.Sprintf("节点 %s 本轮 changes.json 应为空（当前配置已合规，无整改记录）", n.IP))
				}

				By("校验重复加固前后指纹不变")
				Expect(utils.DiffFingerprints(fpHardened, collectAllFingerprints())).To(BeEmpty(),
					"重复加固不应产生任何配置变化")

				By("校验所有节点保持 Ready")
				assertAllNodesReady()
			})
		})
}

func defineIdempotencyRollbackSpecs() {
	Describe("[DFX][DFX-03] Repeated rollback idempotency",
		Ordered, Label("DFX", "DFX-03", "hardening_idempotency", "repeat-rollback", "P2", "26.09", "online"), func() {

			var (
				ts1       string
				fpS0      utils.FingerprintSet
				repeatOut string
			)

			BeforeAll(func() {
				By("解析 TS1 并确认集群处于回退后 S0 态")
				ts1 = shared.unifiedTS
				if ts1 == "" {
					ts1, _ = utils.ReadStateFile(entry, cfg, "harden-ts.txt")
				}
				Expect(ts1).NotTo(BeEmpty(),
					"缺少统一备份时间戳 TS1：请先运行 E2E-A hardening_main_path 与 E2E-C hardening_rollback")
				if shared.s0Fingerprints != nil {
					fpS0 = shared.s0Fingerprints
				} else {
					fpS0 = collectAllFingerprints() // E2E-C 完成后当前态即 S0
				}
			})

			It("[DFX][DFX-03] should repeat rollback as a no-op without cluster side effects", func() {
				By("对已经恢复到 S0 的集群再次执行相同备份回退")
				res := utils.ClusterRollback(entry, cfg, "", "-y", "--backup-subdir", ts1)
				repeatOut = res.Stdout + "\n" + res.Stderr
				Expect(res.ExitCode).To(BeZero(), fmt.Sprintf("重复回退应成功退出（exit=0），输出尾部:\n%s", tail(repeatOut)))

				By("断言所有节点均判定无差异，且不恢复文件、不重启组件")
				Expect(strings.Count(repeatOut, "ROLLBACK_NOOP=1")).To(BeNumerically(">=", len(cfg.Nodes)),
					fmt.Sprintf("%d 个节点都应返回 ROLLBACK_NOOP=1\n输出尾部:\n%s", len(cfg.Nodes), tail(repeatOut)))
				Expect(repeatOut).NotTo(ContainSubstring("ROLLBACK_NOOP=0"),
					"重复回退不应报告任何节点存在配置差异")
				Expect(repeatOut).NotTo(ContainSubstring(utils.MsgFileRestoreDone),
					"重复回退不应复制与备份内容相同的文件")
				idempotentHints := []string{utils.MsgNoDiffRestore, utils.MsgNoModifyRestart, "配置与备份一致（无修改），跳过等待"}
				hit := 0
				for _, hint := range idempotentHints {
					if strings.Contains(repeatOut, hint) {
						hit++
					}
				}
				Expect(hit).To(BeNumerically(">=", 2),
					fmt.Sprintf("输出应含幂等提示（%v）\n输出尾部:\n%s", idempotentHints, tail(repeatOut)))

				By("校验重复回退前后指纹不变且所有节点 Ready")
				Expect(utils.DiffFingerprints(fpS0, collectAllFingerprints())).To(BeEmpty(),
					"重复回退不应产生任何配置变化")
				assertAllNodesReady()
			})
		})
}
