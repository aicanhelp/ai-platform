package compliance_operator

import (
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// CISSTIG_DFX_01 非法 cluster.yaml 配置即停且零副作用（健壮性）。
//
// 等价类：①不存在的配置路径 ②非法 YAML ③节点未配置密码；
// 三类输入均应在进入预览/确认/执行动作前报错退出（exit≠0），全程零副作用。
// 可任意集群状态执行（建议置于套件最前作冒烟），不依赖真实加固动作。
func defineConfigGuardSpecs() {
	Describe("[DFX][DFX-01] Invalid cluster configuration guard",
		Ordered, Label("DFX", "DFX-01", "hardening_config_guard", "P1", "26.09", "online"), func() {

			var (
				fpBefore     utils.FingerprintSet
				backupBefore map[string]int
			)

			BeforeAll(func() {
				By("记录各节点指纹与备份目录数量基线")
				fpBefore = collectAllFingerprints()
				backupBefore = map[string]int{}
				for _, n := range cfg.Nodes {
					cnt, err := utils.BackupDirCount(nodeExecs[n.IP], cfg.BackupDir)
					Expect(err).NotTo(HaveOccurred(), "统计备份目录数量应成功: "+n.IP)
					backupBefore[n.IP] = cnt
				}
			})

			runGuardCase := func(configPath, expectMsg string) {
				By(fmt.Sprintf("执行 cluster-harden %s --dry-run，预期报错退出", configPath))
				res := utils.ClusterHardenWithConfig(entry, cfg, configPath, "--dry-run")
				Expect(res.ExitCode).NotTo(BeZero(),
					fmt.Sprintf("非法配置应报错退出（exit≠0），实际 exit=%d\nstderr: %s", res.ExitCode, tail(res.Stderr)))
				combined := res.Stdout + "\n" + res.Stderr
				Expect(combined).To(ContainSubstring(expectMsg),
					fmt.Sprintf("错误输出应含「%s」\n输出尾部: %s", expectMsg, tail(combined)))
				Expect(combined).NotTo(ContainSubstring("=== 全集群变更预览 ==="),
					"非法配置不应进入预览/确认/执行动作")
			}

			It("[DFX][DFX-01] should reject every invalid cluster configuration without side effects", func() {
				By("拒绝不存在的 cluster 配置文件")
				runGuardCase(cfg.WorkDir+"/nonexistent-cluster.yaml", utils.MsgConfigNotExist)

				By("拒绝 YAML 语法错误的 cluster 配置文件")
				badYAML := cfg.WorkDir + "/invalid-cluster.yaml"
				// 制表符缩进属非法 YAML 语法，yq eval 解析必失败
				_, err := utils.ExecTrim(entry, fmt.Sprintf(
					`printf 'cluster:\n\tssh:\n\t  user: root\n\tnodes: [\n' > %s`, utils.ShellPath(badYAML)))
				Expect(err).NotTo(HaveOccurred(), "写入非法 YAML 夹具应成功")
				runGuardCase(badYAML, utils.MsgConfigYAMLError)

				By("拒绝既无默认口令也无节点级口令的 cluster 配置")
				noPasswordConfig := cfg.WorkDir + "/no-password-cluster.yaml"
				// 使用真实节点 IP/角色，仅剔除全部口令（预检在 SSH 之前触发，零副作用）
				var sb strings.Builder
				sb.WriteString("cluster:\n  ssh:\n    user: root\n    port: 22\n  nodes:\n")
				for _, n := range cfg.Nodes {
					sb.WriteString(fmt.Sprintf("    - ip: %s\n      role: %s\n", n.IP, n.Role))
				}
				_, err = utils.ExecTrim(entry, fmt.Sprintf("cat > %s <<'E2EOF'\n%sE2EOF",
					utils.ShellPath(noPasswordConfig), sb.String()))
				Expect(err).NotTo(HaveOccurred(), "写入缺密码配置夹具应成功")
				runGuardCase(noPasswordConfig, utils.MsgNoPasswordPrefix)

				By("重新采集指纹与备份目录数量并对比")
				fpAfter := collectAllFingerprints()
				Expect(utils.DiffFingerprints(fpBefore, fpAfter)).To(BeEmpty(),
					"非法配置输入不应产生任何节点配置变化\nbefore:\n%s\nafter:\n%s",
					fpBefore.FingerprintText(), fpAfter.FingerprintText())
				for _, n := range cfg.Nodes {
					cnt, err := utils.BackupDirCount(nodeExecs[n.IP], cfg.BackupDir)
					Expect(err).NotTo(HaveOccurred())
					Expect(cnt).To(Equal(backupBefore[n.IP]),
						fmt.Sprintf("节点 %s 备份目录数量不应变化", n.IP))
				}
			})
		})
}
