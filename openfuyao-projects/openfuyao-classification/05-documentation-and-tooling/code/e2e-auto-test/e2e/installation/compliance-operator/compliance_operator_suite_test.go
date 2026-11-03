// Package compliance_operator 承载 compliance-operator compliance-hardening 组件的 E2E 测试。
//
// Reviewed-case baseline: CIS-STIG安全加固特性级集成测试用例-v0.9.xlsx.
//   - E2E-A: FIT-01..FIT-09 and FIT-15
//   - E2E-B: FIT-10..FIT-11
//   - E2E-C: FIT-12..FIT-14
//   - FIT-16..FIT-17 remain product-side Bats unit tests and are not duplicated here
//   - DFX-01..DFX-04 are an additional set outside the 17 reviewed cases
//
// 执行顺序（与用例基线一致，由下方全链路 Ordered 容器固化）：
//
//	DFX_01 冒烟 → E2E_B(S0 预检) → E2E_A(S0→加固态, 产出 TS1) → DFX_02(加固态幂等)
//	→ E2E_C(加固态→S0 回退闭环) → DFX_03(S0 幂等) → DFX_04(真实产物安全审查)
//
// 节点 IP/凭据/工具路径经 e2e/.env 或系统环境变量注入（见 .env.template 的
// compliance-operator 段），代码与日志中不得出现明文凭据。
package compliance_operator

import (
	"fmt"
	"sort"
	"strings"
	"testing"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	complianceutils "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

func TestComplianceOperator(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "ComplianceOperator Suite")
}

var _ = SIGDescribe("compliance-operator compliance-hardening reviewed E2E and additional DFX flow",
	Ordered, Label("compliance-operator", "compliance-hardening", "post-init", "online", "required_nodes:4"), func() {
		var lifecycle *workloadClusterLifecycle

		BeforeAll(func() {
			var err error
			By("连接并确认 E2E 外已初始化的 management cluster")
			lifecycle, err = newWorkloadClusterLifecycle()
			Expect(err).NotTo(HaveOccurred())
			Expect(lifecycle.verifyManagementClusterReady()).To(Succeed(),
				"management cluster 必须在 compliance E2E 之前完成 bke init 并保持 Ready")

			By("初始化本轮 workload cluster 生命周期")
			Expect(lifecycle.create()).To(Succeed(), "3Master1Worker compliance workload cluster 应创建并达到 Healthy/Ready")

			cfg, err = complianceutils.LoadConfig()
			Expect(err).NotTo(HaveOccurred(), "配置加载失败：请通过 e2e/.env 或系统环境变量提供 compliance-operator 节点 IP/凭据")

			By(fmt.Sprintf("连接 entry_node %s（SSH）", cfg.Entry.IP))
			entry, err = executor.NewSSHExecutor(cfg.Entry.IP, cfg.Entry.Port, cfg.Entry.User, cfg.Entry.Password)
			Expect(err).NotTo(HaveOccurred(), "entry_node SSH 连接失败: "+cfg.Entry.IP)
			Expect(lifecycle.prepareEntryContext(entry)).To(Succeed(),
				"应备份 entry_node 原 kubeconfig 并切换到本轮 workload cluster 的 VIP context")

			// 节点角色判定与集群对账：kubectl 角色标签为权威源，kubectl 不可用时按主机名前缀降级；
			// .env 节点集合与 kubectl get nodes 实际节点不一致即停（口径见 README「配置复用与节点角色判定」）。
			By("判定节点角色并对账实际集群节点")
			roleWarnings, err := cfg.ResolveNodeRoles(entry)
			for _, w := range roleWarnings {
				fmt.Fprintf(GinkgoWriter, "[角色判定] %s\n", w)
			}
			Expect(err).NotTo(HaveOccurred(), "节点角色判定/对账失败：请核对 e2e/.env 节点清单与被测集群实际节点（kubectl get nodes）")

			By(fmt.Sprintf("连接被测集群节点（%d master + %d worker）", len(cfg.Masters()), len(cfg.Workers())))
			nodeExecs = map[string]*executor.SSHExecutor{}
			for _, n := range cfg.Nodes {
				e, err := executor.NewSSHExecutor(n.IP, n.Port, n.User, n.Password)
				Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("节点 %s(%s) SSH 连接失败", n.IP, n.Role))
				nodeExecs[n.IP] = e
			}

			By("审核 compliance-hardening 运行依赖，并仅安装缺失项")
			dependencyAudit, err := complianceutils.EnsureHardeningPrerequisites(entry, nodeExecs, cfg)
			for _, result := range dependencyAudit {
				GinkgoWriter.Printf("[依赖审核] %s\n", result)
			}
			Expect(err).NotTo(HaveOccurred(),
				"所有用例开始前，entry_node 应具备 sshpass/yq v4/jq，全部被测节点应具备 yq v4/jq")

			requireWorkerKubeletHAEndpoint()

			By("确保 compliance-hardening 被测工具已部署到 entry_node")
			deployed, err := complianceutils.EnsureHardeningTool(entry, cfg)
			Expect(err).NotTo(HaveOccurred(), "compliance-hardening 被测工具应能部署并通过完整性检查")
			if deployed {
				GinkgoWriter.Printf("已从 Runner 制品目录部署 compliance-hardening 到 entry_node %s\n", cfg.Entry.IP)
			}

			By("确保 cluster.yaml 就绪，并校验节点清单与 .env 一致")
			created, err := complianceutils.EnsureClusterYAML(entry, cfg)
			Expect(err).NotTo(HaveOccurred(), "cluster.yaml 不存在时应能根据 E2E 环境安全生成")
			if created {
				GinkgoWriter.Printf("已根据 E2E 环境在 entry_node 生成 %s（权限 600，凭据未输出）\n", cfg.ClusterYAML)
			}
			_, err = complianceutils.ExecTrim(entry, fmt.Sprintf("test -f %s && echo ok", complianceutils.ShellPath(cfg.ClusterYAML)))
			Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("entry_node 上的 %s 在保留/自动生成后仍不可用", cfg.ClusterYAML))
			yamlIPs, err := complianceutils.ClusterYAMLNodeIPs(entry, cfg)
			Expect(err).NotTo(HaveOccurred(), "读取 cluster.yaml 节点清单应成功")
			envIPs := make([]string, 0, len(cfg.Nodes))
			for _, n := range cfg.Nodes {
				envIPs = append(envIPs, n.IP)
			}
			sort.Strings(yamlIPs)
			sort.Strings(envIPs)
			Expect(strings.Join(yamlIPs, ",")).To(Equal(strings.Join(envIPs, ",")),
				"cluster.yaml 节点 IP 集合应与环境变量中的节点清单一致（TEST_NODE2~5_IP 或 COMPLIANCE_NODE{2..5}_IP，请核对被测集群配置）")

			Expect(complianceutils.PrepareWorkDir(entry, cfg)).NotTo(HaveOccurred(), "创建 entry_node 测试工作目录应成功")
		})

		AfterAll(func() {
			for _, e := range nodeExecs {
				_ = e.Close()
			}
			if entry != nil {
				if err := lifecycle.restoreEntryContext(entry); err != nil {
					GinkgoWriter.Printf("恢复 entry_node 原 kubeconfig 失败: %v\n", err)
				}
				_ = entry.Close()
			}
			lifecycle.cleanup()
		})

		// —— 阶段顺序 = 用例基线执行顺序；各阶段定义分布在本包对应 *_test.go 文件 —— //

		defineConfigGuardSpecs()         // DFX-01: invalid configuration guard
		defineDryRunSpecs()              // E2E-B: FIT-10..FIT-11, requires S0
		defineMainPathSpecs()            // E2E-A: FIT-01..FIT-09 and FIT-15, transitions S0 to hardened
		defineIdempotencyHardenSpecs()   // DFX-02: repeated hardening
		defineRollbackSpecs()            // E2E-C: FIT-12..FIT-14, restores S0
		defineIdempotencyRollbackSpecs() // DFX-03: repeated rollback
		defineSensitiveAuditSpecs()      // DFX-04: real artifact disclosure audit
	})
