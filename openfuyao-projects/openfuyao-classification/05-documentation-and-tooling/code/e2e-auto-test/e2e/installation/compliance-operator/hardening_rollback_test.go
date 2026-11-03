package compliance_operator

import (
	"fmt"
	"regexp"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// E2E-C implements XLSX FIT-12..FIT-14 with one Ginkgo test point per reviewed case.
//
// It must run after E2E-A because it depends on the unified backup TS1 and hardened state:
// 步骤1 确认块输入 n 取消（结果1）→ 步骤2 非法时间戳预检中止（结果2）
// → 步骤3/4 零修改核验（结果2）→ 步骤5 统一回退 TS1（结果3）→ 步骤6/7 恢复核验（结果3）
// → 步骤8 可选复扫（结果4，降级跳过）→ 步骤9 回退编排日志（结果5）。
func defineRollbackSpecs() {
	Describe("[E2E-C] Rollback Loop",
		Ordered, Label("E2E-C", "hardening_rollback", "P0", "26.09", "online"), func() {

			var (
				ts1           string
				fpHardened    utils.FingerprintSet
				fpS0Reference utils.FingerprintSet // 集合形式 S0 基线（套件内共享或从 entry_node 状态文件恢复）
				rollbackOut   string
			)

			BeforeAll(func() {
				By("解析统一备份时间戳 TS1（本轮套件产出或 entry_node 状态文件）")
				ts1 = shared.unifiedTS
				if ts1 == "" {
					ts1, _ = utils.ReadStateFile(entry, cfg, "harden-ts.txt")
				}
				Expect(ts1).NotTo(BeEmpty(),
					"缺少统一备份时间戳 TS1：请先运行 E2E-A hardening_main_path 再执行回退闭环")

				By("校验 TS1 备份在全部节点存在（前置 1）")
				for _, n := range cfg.Nodes {
					Expect(utils.BackupExists(nodeExecs[n.IP], cfg.BackupDir, ts1)).To(BeTrue(),
						fmt.Sprintf("节点 %s 应存在统一备份 %s", n.IP, ts1))
				}

				By("采集加固态指纹（作为非法回退零副作用的对照基线，前置 2）")
				requireHardened("E2E-C")
				if shared.hardenedFingerprints != nil {
					fpHardened = shared.hardenedFingerprints
				} else {
					fpHardened = collectAllFingerprints()
				}

				By("恢复 S0 指纹参照（E2E-B 持久化的基线）")
				if shared.s0Fingerprints != nil {
					fpS0Reference = shared.s0Fingerprints
				} else if text, _ := utils.ReadStateFile(entry, cfg, "s0-fingerprints.txt"); text != "" {
					fpS0Reference = parseFingerprintText(text)
				}
				Expect(fpS0Reference).NotTo(BeNil(),
					"缺少 S0 指纹参照：请先运行 E2E-B 或 E2E-A，以便在 entry_node 持久化 s0-fingerprints.txt")
			})

			It("[E2E-C][FIT-14] should display each node target backup and cancel without changes", Label("FIT-14"), func() {
				By("执行 cluster-rollback（不加 -y），管道输入 n")
				res := utils.ClusterRollback(entry, cfg, "echo n")
				out := res.Stdout + "\n" + res.Stderr
				Expect(res.ExitCode).To(BeZero(), fmt.Sprintf("取消回退应 exit=0，输出尾部:\n%s", tail(out)))

				By("断言确认块列出 4 个节点将回退到的备份")
				Expect(out).To(ContainSubstring(utils.MsgConfirmRollback), "应出现回退确认块")
				Expect(out).To(ContainSubstring("各节点各自最新备份"), "未指定 --backup-subdir 时目标应为各节点最新备份")
				// 用例基线口径：紧跟 E2E-A 执行时该备份即 TS1；套件内 DFX_02 在 A/C 之间会再产出
				// 一轮统一时间戳备份，故此处断言「4 节点目标备份统一」（最新一轮统一时间戳），不写死 TS1。
				targets := backupTargetRe.FindAllStringSubmatch(out, -1)
				Expect(targets).To(HaveLen(len(cfg.Nodes)), "确认块应逐节点列出将回退到的备份")
				seen := map[string]bool{}
				for _, t := range targets {
					seen[t[1]] = true
				}
				Expect(seen).To(HaveLen(1), "4 个节点的回退目标备份应统一（统一时间戳机制）")

				By("输入 n 后输出「用户取消，未执行任何回退」")
				Expect(out).To(ContainSubstring(utils.MsgRollbackCanceled))
				Expect(out).NotTo(ContainSubstring(utils.MsgWorkerParallel), "取消后不应执行任何节点回退")

				By("集群零修改（指纹与加固态零差异）")
				Expect(utils.DiffFingerprints(fpHardened, collectAllFingerprints())).To(BeEmpty(),
					"取消回退不应产生任何配置变化")
			})

			Context("[E2E-C][FIT-13] Missing backup timestamp rejection", Label("FIT-13"), func() {
				It("[E2E-C][FIT-13] should reject a nonexistent backup timestamp without changing any node", func() {
					By("执行 cluster-rollback -y --backup-subdir 20990101-000000")
					res := utils.ClusterRollback(entry, cfg, "", "-y", "--backup-subdir", "20990101-000000")
					rollbackOut = res.Stdout + "\n" + res.Stderr
					Expect(res.ExitCode).To(Equal(1),
						fmt.Sprintf("非法统一备份应预检中止（exit=1），实际 exit=%d\n输出尾部:\n%s", res.ExitCode, tail(rollbackOut)))

					By("断言逐节点输出「✗ 缺失（可用备份：<TS1>）」")
					Expect(rollbackOut).To(ContainSubstring(utils.MsgBackupMissNode), "应逐节点输出缺失提示")
					Expect(rollbackOut).To(ContainSubstring(ts1), "缺失提示应列出可用备份 TS1")

					By("断言整体中止（未做任何修改）：不出现确认块、不执行任何节点回退")
					Expect(rollbackOut).To(ContainSubstring(fmt.Sprintf(utils.MsgBackupMissing, len(cfg.Nodes))),
						fmt.Sprintf("应输出「统一备份 20990101-000000 在 %d 个节点上不存在，已中止（未做任何修改）」", len(cfg.Nodes)))
					Expect(rollbackOut).NotTo(ContainSubstring(utils.MsgConfirmRollback), "中止前不应出现确认块")
					Expect(rollbackOut).NotTo(ContainSubstring("文件级恢复完成"), "中止后不应执行任何节点回退")

					By("重新采集全节点指纹并与加固态基线 diff")
					Expect(utils.DiffFingerprints(fpHardened, collectAllFingerprints())).To(BeEmpty(),
						"非法回退中止后不应产生任何配置变化")

					By("抽查首台 master 的 apiserver manifest 权限（stat -c %a）")
					m := cfg.Masters()[0]
					perm, err := utils.FilePerm(nodeExecs[m.IP], utils.ManifestAPIServer)
					Expect(err).NotTo(HaveOccurred())
					Expect(perm).To(Equal("600"), fmt.Sprintf("节点 %s apiserver 权限应仍为 600（未被回退改动）", m.IP))
				})
			})

			Context("[E2E-C][FIT-12] Full-cluster rollback restoration", Label("FIT-12"), func() {
				It("[E2E-C][FIT-12] should restore the entire cluster to its pre-hardening state", func() {
					By("执行 cluster-rollback -y --backup-subdir <TS1>")
					res := utils.ClusterRollback(entry, cfg, "", "-y", "--backup-subdir", ts1)
					rollbackOut = res.Stdout + "\n" + res.Stderr
					Expect(res.ExitCode).To(BeZero(), fmt.Sprintf("统一回退应成功（exit=0），输出尾部:\n%s", tail(rollbackOut)))
					Expect(rollbackOut).To(ContainSubstring("=== 全集群回退完成 ==="))

					By("断言 master 串行回退、worker 并行")
					Expect(utils.MasterSerialOrdered(rollbackOut, len(cfg.Masters()))).To(BeTrue(),
						"master 应串行回退且 worker 并行段在其后")

					By("断言差量恢复输出「文件级恢复完成：N 个文件」（N 与节点实际改过内容的文件数对应，参考值以实测为准）")
					restoreCount := strings.Count(rollbackOut, utils.MsgFileRestoreDone)
					Expect(restoreCount).To(BeNumerically(">=", len(cfg.Nodes)),
						fmt.Sprintf("4 节点均应输出文件级恢复（实际 %d 处）\n输出尾部:\n%s", restoreCount, tail(rollbackOut)))

					By("重新采集全节点指纹并与 S0 基线 diff")
					fpAfter := collectAllFingerprints()
					Expect(utils.DiffFingerprints(fpS0Reference, fpAfter)).To(BeEmpty(),
						"回退应完整恢复至 S0\nS0:\n%s\n回退后:\n%s",
						fpS0Reference.FingerprintText(), fpAfter.FingerprintText())

					By("执行 kubectl get nodes 校验 4/4 Ready")
					assertAllNodesReady()

					if shared.cisBaseline == nil && shared.stigBaseline == nil {
						GinkgoWriter.Printf("[降级] FIT-12 回退后复扫跳过：%s\n", shared.scanSkipReason)
					}
					if shared.cisBaseline != nil {
						By("CIS 复扫对比基线")
						after, err := utils.RunKubeBench(entry, cfg)
						Expect(err).NotTo(HaveOccurred())
						GinkgoWriter.Printf("CIS 复扫: FAIL项=%v\n", after.FailItems)
						unexpected := utils.SubsetOf(after.FailItems, shared.cisBaseline.FailItems)
						Expect(unexpected).To(BeEmpty(),
							fmt.Sprintf("回退后 CIS 不应出现基线外的新增 FAIL（新增 %v，基线 %v，实际 %v）",
								unexpected, shared.cisBaseline.FailItems, after.FailItems))
						missing := utils.SubsetOf(shared.cisBaseline.FailItems, after.FailItems)
						Expect(missing).To(BeEmpty(),
							fmt.Sprintf("回退后 CIS 应恢复全部基线 FAIL（未恢复 %v，基线 %v，实际 %v）",
								missing, shared.cisBaseline.FailItems, after.FailItems))
					}
					if shared.stigBaseline != nil {
						By("STIG 复扫对比基线")
						after, err := utils.RunOpenSCAP(entry, cfg)
						Expect(err).NotTo(HaveOccurred())
						GinkgoWriter.Printf("STIG 复扫: FAIL项=%v\n", after.FailItems)
						unexpected := utils.SubsetOf(after.FailItems, shared.stigBaseline.FailItems)
						Expect(unexpected).To(BeEmpty(),
							fmt.Sprintf("回退后 STIG 不应出现基线外的新增 FAIL（新增 %v，基线 %v，实际 %v）",
								unexpected, shared.stigBaseline.FailItems, after.FailItems))
						missing := utils.SubsetOf(shared.stigBaseline.FailItems, after.FailItems)
						Expect(missing).To(BeEmpty(),
							fmt.Sprintf("回退后 STIG 应恢复全部基线 FAIL（未恢复 %v，基线 %v，实际 %v）",
								missing, shared.stigBaseline.FailItems, after.FailItems))
					}
					By("entry_node 回退编排日志存在且含 TS1 与 master 串行段落")
					// 以「本次回退的专属行」做内容级选取：真实回退日志独有「指定回退备份（所有节点统一）：<TS1>」
					// （lib/cluster.sh:765，--backup-subdir 时输出）；E2E_C-2 的中止日志虽含 TS1 子串（可用备份
					// 逐节点枚举），但其该行写的是非法 TS——直接 grep TS1 + head -1 会因文件名字典序命中中止
					// 日志（run9 实测），此处按完整行匹配天然唯一。
					out, err := utils.ExecTrim(entry, fmt.Sprintf(
						`grep -l "指定回退备份（所有节点统一）：%s" %s/%s 2>/dev/null | head -1`, ts1,
						utils.ShellPath(cfg.LogDir), utils.RollbackLogGlob))
					Expect(err).NotTo(HaveOccurred())
					logPath := strings.TrimSpace(out)
					Expect(logPath).NotTo(BeEmpty(), fmt.Sprintf("%s 下应存在含 TS1 的回退编排日志", cfg.LogDir))

					logText, err := utils.ExecTrim(entry, fmt.Sprintf("cat %s", utils.ShellPath(logPath)))
					Expect(err).NotTo(HaveOccurred())
					for i := 1; i <= len(cfg.Masters()); i++ {
						Expect(logText).To(ContainSubstring(fmt.Sprintf("[master %d/%d] 回退", i, len(cfg.Masters()))),
							fmt.Sprintf("回退编排日志应含 [master %d/%d] 回退 段落", i, len(cfg.Masters())))
					}
					By("加固侧编排日志同目录可追溯")
					exist, err := utils.GlobCount(entry, fmt.Sprintf("%s/%s", utils.ShellPath(cfg.LogDir), utils.HardenLogGlob))
					Expect(err).NotTo(HaveOccurred())
					Expect(exist).To(BeNumerically(">", 0), "加固编排日志应存在（与回退侧日志链齐全）")
				})
			})
		})
}

// backupTargetRe 确认块逐节点目标备份行：「  <label> → 备份 <时间戳>」
var backupTargetRe = regexp.MustCompile(`→ 备份 ([0-9]{8}-[0-9]{6})`)

// parseFingerprintText 从持久化文本（FingerprintSet.FingerprintText 格式）恢复指纹集合。
func parseFingerprintText(text string) utils.FingerprintSet {
	set := utils.FingerprintSet{}
	currentIP := ""
	var sb strings.Builder
	flush := func() {
		if currentIP != "" {
			set[currentIP] = strings.TrimSpace(sb.String())
		}
	}
	for _, line := range strings.Split(text, "\n") {
		if m := regexp.MustCompile(`^== (\S+) ==$`).FindStringSubmatch(strings.TrimSpace(line)); m != nil {
			flush()
			currentIP = m[1]
			sb.Reset()
			continue
		}
		if currentIP != "" {
			sb.WriteString(line + "\n")
		}
	}
	flush()
	return set
}
