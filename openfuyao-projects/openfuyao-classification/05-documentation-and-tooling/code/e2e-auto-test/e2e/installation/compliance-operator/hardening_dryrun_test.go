package compliance_operator

import (
	"fmt"
	"regexp"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// E2E-B implements XLSX FIT-10 and FIT-11 with one Ginkgo test point per reviewed case.
//
// It must run before E2E-A because the preview requires an unhardened S0 state:
// 步骤 1 预览输出断言（结果1）→ 步骤 2 预览真实性（结果2）→ 步骤 3/4 零副作用（结果3）。
func defineDryRunSpecs() {
	Describe("[E2E-B] Dry-Run Precheck",
		Ordered, Label("E2E-B", "hardening_dryrun", "P0", "26.09", "online"), func() {

			var (
				fpBefore      utils.FingerprintSet
				backupBefore  map[string]int
				dryrunOut     string
				dryrunExit    int
				summaryLineRe = regexp.MustCompile(`汇总：待整改=\d+\s+跳过\(已合规\)=\d+\s+人工处理\(高危操作\)=\d+`)
			)

			BeforeAll(func() {
				requireUnhardened("E2E-B")
				By("采集 S0 指纹与备份目录数量基线（并持久化供 E2E-C 回退后对比）")
				fpBefore = collectAllFingerprints()
				shared.s0Fingerprints = fpBefore
				Expect(utils.WriteStateFile(entry, cfg, "s0-fingerprints.txt", fpBefore.FingerprintText())).
					NotTo(HaveOccurred(), "持久化 S0 指纹基线应成功")
				backupBefore = map[string]int{}
				for _, n := range cfg.Nodes {
					cnt, err := utils.BackupDirCount(nodeExecs[n.IP], cfg.BackupDir)
					Expect(err).NotTo(HaveOccurred())
					backupBefore[n.IP] = cnt
				}
			})

			Context("[E2E-B][FIT-10] Full-cluster dry-run change preview", Label("FIT-10"), func() {
				It("[E2E-B][FIT-10] should preview the change list by node and report actual current values", func() {
					By("执行 cluster-harden --dry-run")
					res := utils.ClusterHarden(entry, cfg, "--dry-run")
					dryrunOut = res.Stdout + "\n" + res.Stderr
					dryrunExit = res.ExitCode
					Expect(dryrunExit).To(BeZero(), fmt.Sprintf("dry-run 应成功退出（exit=0），输出尾部:\n%s", tail(dryrunOut)))

					By("断言末尾「未执行任何变更」")
					Expect(dryrunOut).To(ContainSubstring(utils.MsgDryRunDone))

					By("断言按节点分组预览：master 段逐条列出待整改规则（rid、旧值→新值），worker 段仅 g6 规则")
					sections := splitNodeSections(dryrunOut)
					Expect(sections).NotTo(BeEmpty(), "预览输出应含按节点分组的段落")
					for _, n := range cfg.Nodes {
						sec, ok := nodeSection(sections, n.IP)
						Expect(ok).To(BeTrue(), fmt.Sprintf("预览输出应含节点段落：%s", labelOf(n.IP)))
						Expect(summaryLineRe.MatchString(sec)).To(BeTrue(),
							fmt.Sprintf("节点 %s 预览段应以「汇总：待整改=N 跳过(已合规)=M 人工处理(高危操作)=K」结尾\n节点输出尾部:\n%s",
								n.IP, tail(sec)))
						if n.Role == utils.RoleMaster {
							Expect(sec).To(ContainSubstring("cis-1.1.1"),
								fmt.Sprintf("master %s 预览应列出 g1 待整改规则（S0 态 apiserver 权限 644→600）", n.IP))
							Expect(sec).To(ContainSubstring("→"), "master 预览应含旧值→新值明细")
						} else {
							Expect(sec).NotTo(ContainSubstring("cis-1.1.1"), "worker 预览不应含 g1 规则")
							Expect(sec).NotTo(ContainSubstring("cis-1.3.7"), "worker 预览不应含 g3 规则")
							Expect(sec).NotTo(ContainSubstring("cis-1.4.2"), "worker 预览不应含 g4 规则")
						}
					}

					By("断言 3 条高危跳过标记可见（cis-1.1.12 / stig-sv242445 / stig-sv242404）")
					for _, rid := range utils.HighRiskSkipRuleIDs {
						Expect(dryrunOut).To(ContainSubstring(rid), fmt.Sprintf("预览应含高危跳过规则 %s", rid))
					}
					Expect(dryrunOut).To(ContainSubstring("高危跳过"), "预览应含「高危跳过」标记")

					By(fmt.Sprintf("抽查首台 master（%s）预览中 cis-1.1.1 的旧值", cfg.Masters()[0].IP))
					m := cfg.Masters()[0]
					sec, ok := nodeSection(splitNodeSections(dryrunOut), m.IP)
					Expect(ok).To(BeTrue(), "应能定位 master 预览段")
					re := regexp.MustCompile(`cis-1\.1\.1\s+待整改\s+(\d+)\s*→`)
					matches := re.FindStringSubmatch(sec)
					Expect(matches).NotTo(BeNil(), fmt.Sprintf("master 预览段应含 cis-1.1.1 待整改行\n段落:\n%s", tail(sec)))

					By("与该节点实际权限对比（stat -c %a；预览为四位补零如 0644，归一化前导零后比较）")
					actual, err := utils.FilePerm(nodeExecs[m.IP], utils.ManifestAPIServer)
					Expect(err).NotTo(HaveOccurred())
					Expect(normalizePerm(matches[1])).To(Equal(normalizePerm(actual)),
						fmt.Sprintf("预览旧值 %s 应与节点实际权限 %s 一致（证明预览基于真实状态检测）", matches[1], actual))
				})
			})

			It("[E2E-B][FIT-11] should complete dry-run without writing files, changing configuration, or creating backups", Label("FIT-11"), func() {
				By("重新采集全节点指纹并与 S0 基线 diff")
				fpAfter := collectAllFingerprints()
				Expect(utils.DiffFingerprints(fpBefore, fpAfter)).To(BeEmpty(),
					"dry-run 不应产生任何配置变化\nbefore:\n%s\nafter:\n%s",
					fpBefore.FingerprintText(), fpAfter.FingerprintText())

				By("统计各节点备份目录数量并与执行前对比")
				for _, n := range cfg.Nodes {
					cnt, err := utils.BackupDirCount(nodeExecs[n.IP], cfg.BackupDir)
					Expect(err).NotTo(HaveOccurred())
					Expect(cnt).To(Equal(backupBefore[n.IP]),
						fmt.Sprintf("节点 %s 备份目录数量不应变化（dry-run 不备份）", n.IP))
				}
			})
		})
}

// splitNodeSections 按编排日志的「--- <节点label> ---」分段标记切 dry-run 输出。
// 段落头均经工具 _log 输出（lib/common.sh），带「[时间戳] [级别] 」前缀；key 为
// cluster_node_label（lib/cluster.sh）的「hostname(ip) [role]」形态，另兼容 SSH
// 连接失败时的「--- <label>：跳过（SSH 连接失败）---」跳过标记（尾随 --- 前无空格）。
func splitNodeSections(out string) map[string]string {
	sections := map[string]string{}
	re := regexp.MustCompile(`(?m)^(?:\[[^\]]*\]\s+)*--- (.+?)(?:：跳过（SSH 连接失败）)?\s*---\s*$`)
	matches := re.FindAllStringSubmatchIndex(out, -1)
	for i, m := range matches {
		label := out[m[2]:m[3]]
		end := len(out)
		if i+1 < len(matches) {
			end = matches[i+1][0]
		}
		sections[label] = out[m[1]:end]
	}
	return sections
}

// nodeSection 在分段结果中定位指定节点（按 ip）的段落：优先精确匹配 labelOf
// （cluster.yaml hostname），否则按「hostname(ip)」前缀或「(ip)」括号段匹配——
// 兼容 cluster.yaml 缺失 hostname 时 labelOf 回退为裸 ip 的情况。
func nodeSection(sections map[string]string, ip string) (string, bool) {
	want := labelOf(ip)
	if sec, ok := sections[want]; ok {
		return sec, true
	}
	for label, sec := range sections {
		if strings.HasPrefix(label, want+"(") || strings.Contains(label, "("+ip+")") {
			return sec, true
		}
	}
	return "", false
}

// normalizePerm 归一化八进制权限串的前导零：dry-run 预览行为四位补零（如 0644），
// 而 stat -c %a 返回无前导零形式（如 644），统一为无前导零形式后再断言。
func normalizePerm(s string) string {
	t := strings.TrimLeft(strings.TrimSpace(s), "0")
	if t == "" {
		return "0"
	}
	return t
}

// labelOf 查询 cluster.yaml 中节点展示名（hostname // ip），与编排日志段落标记一致。
func labelOf(ip string) string {
	if labels == nil {
		labels = map[string]string{}
		if m, err := utils.ClusterYAMLLabelByIP(entry, cfg); err == nil {
			labels = m
		}
	}
	if l, ok := labels[ip]; ok && l != "" {
		return l
	}
	return ip
}

var labels map[string]string

// tail 取文本尾部（失败信息展示用，避免超长输出刷屏）。
func tail(s string) string {
	s = strings.TrimSpace(s)
	if len(s) <= 2000 {
		return s
	}
	return "......" + s[len(s)-2000:]
}
