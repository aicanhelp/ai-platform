package preflight

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

var _ = Describe("统一前置检查 cluster-create", Label("preflight", "with-management-cluster", "26.09"), func() {
	var (
		ssh *executor.SSHExecutor
		h   *preflightHarness
	)

	// 用例之间无顺序依赖，可乱序执行；套件级资源懒初始化一次，统一由 AfterSuite 清理
	// cluster-create 用例在节点一（TEST_NODE1，已安装管理集群）上执行；
	// cluster.yaml 取 bke 自带模板，nodes.yaml 由代码按节点二（master）、三（worker）信息渲染
	BeforeEach(func() {
		if h == nil {
			ssh = newNodeExecutor(1)
			h = newClusterPreflightHarness(ssh)
			h.tmpDir = h.createTempDir()
			registerPreflightSuiteCleanup(func() {
				h.cleanup()
				_ = ssh.Close()
			})
			h.loadClusterTemplate()
		}

		By("校验节点一上 bke 二进制与 kubeconfig 可用")
		h.run("test -x " + shq(h.bke))
		Expect(h.fileExists(h.kubeconfig)).To(BeTrue(), "kubeconfig 不存在: %s", h.kubeconfig)
	})

	// freshCluster/freshNodes 每个用例生成一份新输入文件，避免用例间相互污染
	baseNodes := func() []preflightNode {
		return preflightEnvNodes()
	}
	freshCluster := func(name string) string {
		p := h.tmpDir + "/" + name
		h.writeFile(p, h.clusterTemplate)
		return p
	}
	freshNodes := func(name string, nodes []preflightNode) string {
		p := h.tmpDir + "/" + name
		h.writeFile(p, renderPreflightNodesYAML(nodes))
		return p
	}

	// 用例编号：PREFLIGHT-CLUSTER-001（p0）
	// 前置：管理集群可用；外部策略 preflight.yaml（cpu≥4、memory≥6GiB、systemDisk≥40GiB）
	// 步骤：执行 bke preflight cluster-create --preflight-config；读取 JSON 报告
	// 预期：Overall=PASS、exit 0，管理集群与节点检查全 PASS，资源实际值与真实环境一致
	It("【PREFLIGHT-CLUSTER-001】cluster-create 正常场景全量通过", Label("required_nodes:2", "PREFLIGHT-CLUSTER-001", "P0", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("执行 cluster-create 前置检查（外部策略 preflight.yaml）")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-default.yaml"), freshNodes("nodes-default.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "PASS", 0)
			expectSummaryArithmetic(o.Report)

			By("校验管理集群与节点检查全 PASS")
			expectAllResultsPass(o.Report)
			s := o.Report.Summary
			Expect(s.Fail).To(Equal(0), "不应有 FAIL 项")
			Expect(s.Skip).To(Equal(0), "不应有 SKIP 项")

			By("校验管理集群检查链（API/Discovery/CRD/Secret）")
			for _, kw := range []string{"API", "DISCOVERY", "CRD", "SECRET"} {
				Expect(filterResults(o.Report, kw)).NotTo(BeEmpty(),
					"管理集群检查应包含 %s 项", kw)
			}

			By("校验管理集群检查项数量不少于 10（目录含固定模板与按角色实例化项）")
			nodeNames := map[string]bool{}
			for _, hn := range testNodeHostnames() {
				nodeNames[hn] = true
			}
			mgmtCount := 0
			for _, res := range o.Report.Results {
				suffix := instanceSuffix(res)
				if suffix != "" && !nodeNames[suffix] {
					mgmtCount++
				}
			}
			Expect(mgmtCount).To(BeNumerically(">=", 10),
				"管理集群检查项不应少于 10 项，实际 %d 项", mgmtCount)

			By("校验节点 SSH 依赖检查与资源实际值")
			sshItems := filterResults(o.Report, "SSH")
			Expect(sshItems).NotTo(BeEmpty(), "应存在节点 SSH 依赖检查")
			Expect(allStatus(sshItems, "PASS")).To(BeTrue(), "SSH 依赖检查应全部 PASS")
			crossCheckNodeFacts(o.Report)
		})

	// 用例编号：PREFLIGHT-CLUSTER-002（p1）
	// 前置：外部策略——优先 PREFLIGHT_CLUSTER_POLICY_TEMPLATE，否则使用包内 preflight.yaml
	// 步骤：外部策略将节点资源 cpu minimum 提高到 999 → 执行 cluster-create
	// 预期：Overall=FAIL、exit 3，资源项 FAIL 且期望/实际准确，其余不受影响，建议换节点或扩容
	It("【PREFLIGHT-CLUSTER-002】cluster-create 节点资源不足时资源项失败", Label("required_nodes:2", "PREFLIGHT-CLUSTER-002", "P1", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("准备外部策略（env 覆盖优先，否则使用包内 preflight.yaml）")
			policySrc, usingFixture := h.policyFixturePath(h.policyTemplate,
				"preflight.yaml", preflightPolicyYAML)
			if !usingFixture {
				Expect(h.fileExists(policySrc)).To(BeTrue(), "策略模板不存在: %s", policySrc)
			}

			By("复制外部策略并将 clusterCreate 资源 cpu minimum 调整为 999")
			policy := h.copyToTemp(policySrc, "policy-cpu999.yaml")
		applied := h.rewriteKey(policy, "minimum", "4", 1, "999")
		Expect(applied).To(BeTrue(),
			"策略中应存在 \"minimum: 4\"（clusterCreate cpu 阈值）: %s", policySrc)

			By("执行 cluster-create，预期 CPU 资源项 FAIL")
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-res.yaml"), freshNodes("nodes-res.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, usingFixture)
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 CPU 资源项 FAIL 且期望/实际准确")
			cpuItems := filterResults(o.Report, "RES-CPU")
			Expect(cpuItems).NotTo(BeEmpty(), "应存在 CPU 资源检查项")
			failed := countByStatus(cpuItems, "FAIL")
			Expect(failed).To(Equal(len(cpuItems)),
				"阈值 999 下所有节点 CPU 项都应 FAIL: %+v", cpuItems)
			for i := range cpuItems {
				exp, ok := rawValueInt64(cpuItems[i].Expected.Value)
				Expect(ok).To(BeTrue(), "CPU 期望值应为数值: %s", rawValueString(cpuItems[i].Expected.Value))
				Expect(exp).To(Equal(int64(999)), "CPU 期望阈值应为 999")
				act, ok := rawValueInt64(cpuItems[i].Actual.Value)
				Expect(ok).To(BeTrue(), "CPU 实际值应为数值: %s", rawValueString(cpuItems[i].Actual.Value))
				Expect(act).To(BeNumerically(">", 0), "CPU 实际值应为真实核数")
				Expect(cpuItems[i].Remediation).NotTo(BeEmpty(), "CPU 项应给出修复建议")
			}

			By("校验其余检查项不受影响")
			for _, res := range o.Report.Results {
				if res.Status == "FAIL" && !strings.Contains(strings.ToUpper(res.InstanceID), "RES-CPU") {
					Fail(fmt.Sprintf("不应有 CPU 之外的 FAIL 项: %s", res.InstanceID))
				}
			}
			Expect(filterResults(o.Report, "RES-MEM")).NotTo(BeEmpty(), "MEM 检查应正常执行")
			Expect(filterResults(o.Report, "RES-SYSTEM-DISK")).NotTo(BeEmpty(), "DISK 检查应正常执行")

			By("用 nproc 交叉核对实际值准确性")
			crossCheckNodeFacts(o.Report)
		})

	// 用例编号：PREFLIGHT-CLUSTER-003（p1）
	// 步骤：cluster.yaml 中 imageRepo.domain/ip 改为不可达地址 → 执行 → 还原后重跑
	// 预期：不可达时每节点各一条 UNREACHABLE FAIL（不拆分 DNS/TLS/认证），修正后 PASS
	It("【PREFLIGHT-CLUSTER-003】cluster-create 镜像仓库不可达时单项失败", Label("required_nodes:2", "PREFLIGHT-CLUSTER-003", "P1", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("复制 cluster.yaml 并将 imageRepo.domain/ip 指向不可达地址")
			cluster := freshCluster("cluster-badrepo.yaml")
			// bke 自带模板中 imageRepo 为映射结构（domain/ip/port），提取 domain/ip 子键原值后改写
			probe := h.tryRun("awk '/^[[:space:]]*imageRepo:/{f=1;next} f && /^[[:space:]]*domain:/{print $2; exit}' " + shq(cluster))
			oldDomain := ""
			if probe != nil {
				oldDomain = strings.Trim(strings.TrimSpace(probe.Stdout), "\"'")
			}
			if oldDomain == "" {
				Skip("集群模板中 imageRepo 块未找到 domain 子键，无法改写为不可达地址")
			}
			Expect(h.rewriteKey(cluster, "domain", oldDomain, 1, "unreachable.pf-e2e.invalid")).To(BeTrue(),
				"应能改写 imageRepo.domain 字段（原值 %s）", oldDomain)

			// ip 与 domain 同构处理：先取 imageRepo 块内 ip 子键原值，再改写为保留网段不可达地址
			probeIP := h.tryRun("awk '/^[[:space:]]*imageRepo:/{f=1;next} f && /^[[:space:]]*ip:/{print $2; exit}' " + shq(cluster))
			oldIP := ""
			if probeIP != nil {
				oldIP = strings.Trim(strings.TrimSpace(probeIP.Stdout), "\"'")
			}
			if oldIP == "" {
				Skip("集群模板中 imageRepo 块未找到 ip 子键，无法改写为不可达地址")
			}
			Expect(h.rewriteKey(cluster, "ip", oldIP, 1, "192.0.2.1")).To(BeTrue(),
				"应能改写 imageRepo.ip 字段（原值 %s）", oldIP)

			By("执行 cluster-create，预期镜像仓库检查单条 FAIL")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, cluster, freshNodes("nodes-badrepo.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验每 (节点,image-repo) 恰一条 FAIL，不拆分 DNS/TLS/认证")
			// 来源可达检查按节点实例化（instanceID 为 @<node>:<source>），不可达时两节点各产生一条 FAIL，
			// CSV 契约按 (节点,来源) 计数而非全局唯一，故 Summary.Fail 应为 2。
			Expect(o.Report.Summary.Fail).To(Equal(2),
				"两节点各应产生一条镜像仓库 FAIL，实际 %+v", o.Report.Summary)
			failNodes := map[string]int{}
			for i := range o.Report.Results {
				if o.Report.Results[i].Status != "FAIL" {
					continue
				}
				res := o.Report.Results[i]
				up := strings.ToUpper(res.InstanceID + res.RuleID + res.Name)
				Expect(strings.Contains(up, "REACHABLE") || strings.Contains(up, "IMAGE") || strings.Contains(up, "REPO")).To(BeTrue(),
					"不应有镜像仓库之外的 FAIL 项: %s", res.InstanceID)
				// CSV 契约未要求 UNREACHABLE 项携带 remediation（bke 当前对镜像仓库 FAIL 不给出修复建议），
				// 仅校验 FAIL 类型归属与按节点实例化，不在此断言 Remediation。
				failNodes[strings.SplitN(instanceSuffix(res), ":", 2)[0]]++
			}
			Expect(failNodes).To(HaveLen(2), "镜像仓库检查应按节点实例化并覆盖两个节点: %v", failNodes)
			for node, cnt := range failNodes {
				Expect(cnt).To(Equal(1), "节点 %s 应恰一条 image-repo FAIL，实际 %d 条", node, cnt)
			}

			By("还原 imageRepo 后重跑，预期 PASS")
			o2 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-okrepo.yaml"), freshNodes("nodes-okrepo.yaml", baseNodes()), policy)
			expectPreflightOutcome(o2, "PASS", 0)
		})

	// 用例编号：PREFLIGHT-CLUSTER-004（p1）
	// 步骤：nodes.yaml 中第 2 个节点密码改错 → 执行 → 还原后重跑
	// 预期：SSH-TRANSPORT PASS、SSH-AUTH FAIL，下游 systemd/资源/平台/端口/来源全 SKIP，修正后 PASS
	It("【PREFLIGHT-CLUSTER-004】cluster-create 节点 SSH 密码错误时下游跳过", Label("required_nodes:2", "PREFLIGHT-CLUSTER-004", "P1", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("生成 nodes.yaml 并将第 2 个节点密码改错")
			nodes := freshNodes("nodes-badpass.yaml", baseNodes())
			Expect(h.rewriteNthKey(nodes, "password", 2, "Pf-Wrong-Pass-123")).To(BeTrue(),
				"nodes.yaml 应包含第 2 个节点的 password 字段")

			By("执行 cluster-create，预期 SSH 认证失败")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-badpass.yaml"), nodes, policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 SSH-TRANSPORT PASS 而 SSH-AUTH FAIL")
			transport := filterResults(o.Report, "SSH-TRANSPORT")
			Expect(transport).NotTo(BeEmpty(), "应存在 SSH-TRANSPORT 检查")
			Expect(allStatus(transport, "PASS")).To(BeTrue(), "SSH 传输层应可达 PASS")
			auth := filterResults(o.Report, "SSH-AUTH")
			authFail := countByStatus(auth, "FAIL")
			Expect(authFail).To(Equal(1), "应恰好 1 个节点 SSH-AUTH FAIL")
			var failNode string
			for i := range auth {
				if auth[i].Status == "FAIL" {
					failNode = instanceSuffix(auth[i])
					break
				}
			}
			Expect(failNode).NotTo(BeEmpty(), "应能定位 SSH-AUTH 失败节点")

			By("校验失败节点下游检查全 SKIP")
			var downstream []preflightResult
			for _, res := range o.Report.Results {
				if instanceSuffix(res) == failNode && res.Status != "PASS" {
					downstream = append(downstream, res)
				}
			}
			skipCount := countByStatus(downstream, "SKIP")
			Expect(skipCount).To(BeNumerically(">=", 3),
				"失败节点 %s 的 systemd/资源/平台/端口/来源检查应 SKIP，实际 SKIP %d 项", failNode, skipCount)

			By("校验仅 SSH-AUTH 一项 FAIL，其他节点不受影响")
			Expect(o.Report.Summary.Fail).To(Equal(1), "应只有 SSH-AUTH 一项 FAIL")

			By("还原密码后重跑，预期 PASS")
			o2 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-okpass.yaml"), freshNodes("nodes-okpass.yaml", baseNodes()), policy)
			expectPreflightOutcome(o2, "PASS", 0)
		})

	// 用例编号：PREFLIGHT-CLUSTER-005（p2）
	// 步骤：使用无效 kubeconfig 执行 cluster-create
	// 预期：Overall=FAIL、exit 3，管理集群 API 项 FAIL，下游 Discovery/CRD/Secret 链 SKIP，
	// 节点 SSH 检查不受影响，无重复 FAIL
	It("【PREFLIGHT-CLUSTER-005】cluster-create 无效 kubeconfig 时管理集群链路失败", Label("required_nodes:2", "PREFLIGHT-CLUSTER-005", "P2", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("构造无效 kubeconfig")
			badKC := h.tmpDir + "/kubeconfig.invalid"
			h.writeFile(badKC, "not-a-kubeconfig-preflight-e2e")

			By("以无效 kubeconfig 执行 cluster-create")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(badKC, freshCluster("cluster-badkc.yaml"), freshNodes("nodes-badkc.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验管理集群 API 项 FAIL 且无重复 FAIL")
			Expect(o.Report.Summary.Fail).To(Equal(1),
				"应只有 API 一项 FAIL（无重复），实际 %+v", o.Report.Summary)
			apiItems := filterResults(o.Report, "API")
			var apiFail *preflightResult
			for i := range apiItems {
				if apiItems[i].Status == "FAIL" {
					apiFail = &apiItems[i]
					break
				}
			}
			Expect(apiFail).NotTo(BeNil(), "管理集群 API 检查应 FAIL")

			By("校验下游 Discovery/CRD/Secret 链 SKIP")
			nodeNames := map[string]bool{}
			for _, hn := range testNodeHostnames() {
				nodeNames[hn] = true
			}
			mgmtSkip := 0
			for _, res := range o.Report.Results {
				if !nodeNames[instanceSuffix(res)] && res.Status == "SKIP" {
					mgmtSkip++
				}
			}
			Expect(mgmtSkip).To(BeNumerically(">=", 3),
				"API 项失败后 Discovery/CRD/Secret 链应 SKIP，实际管理面 SKIP %d 项", mgmtSkip)

			By("校验节点 SSH 检查不受影响")
			for _, res := range o.Report.Results {
				if nodeNames[instanceSuffix(res)] {
					Expect(res.Status).To(Equal("PASS"),
						"节点 %s 的检查不应受 kubeconfig 影响，实际 %s", res.InstanceID, res.Status)
				}
			}
		})

	// 用例编号：PREFLIGHT-CLUSTER-006（p2）
	// 步骤：cluster.yaml 注入未知字段执行；再改为非法 CIDR podSubnet:not-a-cidr 执行
	// 预期：未知字段 INPUT_ERROR、exit 4 并定位 cluster.unknownField；非法 CIDR 时
	// CLUSTER-NET-POD-CIDR FAIL、exit 3、目标列指向 networking.podSubnet，不阻塞其他检查
	It("【PREFLIGHT-CLUSTER-006】cluster-create 输入校验与非法 CIDR 检查", Label("required_nodes:2", "PREFLIGHT-CLUSTER-006", "P2", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("场景A：cluster.yaml 注入未知字段")
			clusterA := freshCluster("cluster-unknown.yaml")
			h.appendTopLevelKey(clusterA, "unknownField", "pf-e2e-unknown")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, clusterA, freshNodes("nodes-unknown.yaml", baseNodes()), policy)

			Expect(o.ExitCode).To(Equal(4), "未知字段应返回 exit 4，实际 %d\nstdout:\n%s\nstderr:\n%s",
				o.ExitCode, o.Stdout, o.Stderr)

			By("校验 INPUT_ERROR 的 JSON 报告契约（issue 定位未知字段）")
			Expect(o.Report).NotTo(BeNil(),
				"INPUT_ERROR 应输出 JSON 报告:\nstdout:\n%s\nstderr:\n%s", o.Stdout, o.Stderr)
			Expect(o.Report.Overall).To(Equal("INPUT_ERROR"), "报告 overall 应为 INPUT_ERROR")
			Expect(o.Report.Issues).NotTo(BeEmpty(), "INPUT_ERROR 应输出结构化 issue")
			var iss preflightIssue
			Expect(json.Unmarshal(o.Report.Issues[0], &iss)).To(Succeed(),
				"issue 应可解析: %s", string(o.Report.Issues[0]))
			Expect(iss.Effect).To(Equal("BLOCK_START"),
				"未知字段应阻塞启动（effect=BLOCK_START），实际 %+v", iss)
			Expect(iss.Field).To(Equal("cluster.unknownField"),
				"issue 应定位到 cluster.unknownField，实际 %+v", iss)
			Expect(o.Report.Results).To(BeEmpty(), "INPUT_ERROR 不应产生检查项明细")

			By("场景B：podSubnet 配置为非法 CIDR")
			clusterB := freshCluster("cluster-badcidr.yaml")
			if !h.rewriteNthKey(clusterB, "podSubnet", 1, "not-a-cidr") {
				Skip("集群模板需包含 podSubnet 字段以支持场景B")
			}
			o2 := h.runClusterCreate(h.kubeconfig, clusterB, freshNodes("nodes-badcidr.yaml", baseNodes()), policy)
			expectPreflightOutcome(o2, "FAIL", 3)

			By("校验 CLUSTER-NET-POD-CIDR 单项 FAIL 且目标列正确")
			cidr := findResult(o2.Report, "POD-CIDR")
			Expect(cidr).NotTo(BeNil(), "应存在 CLUSTER-NET-POD-CIDR 检查项")
			Expect(cidr.Status).To(Equal("FAIL"), "非法 CIDR 时 POD-CIDR 应 FAIL")
			Expect(cidr.Evidence).NotTo(BeEmpty(), "POD-CIDR 应携带 evidence 定位字段路径")
			Expect(cidr.Evidence[0].Locator).To(ContainSubstring("podSubnet"),
				"目标列应指向 podSubnet 字段，实际 locator=%s subject=%s sourceRefs=%v",
				cidr.Evidence[0].Locator, cidr.Subject, cidr.SourceRefs)

			By("校验非法 CIDR 不阻塞其他检查")
			Expect(o2.Report.Summary.Fail).To(Equal(1), "应只有 CIDR 一项 FAIL")
			Expect(len(o2.Report.Results)).To(BeNumerically(">", 1), "其余检查应正常执行")
			Expect(o2.Report.Summary.Pass).To(BeNumerically(">", 0), "应有 PASS 项")
		})

	// 用例编号：PREFLIGHT-CLUSTER-008（p2）
	// 前置：含自定义 exec 检查的策略——优先 PREFLIGHT_CLUSTER_POLICY_CUSTOM_EXEC，
	//       否则基于包内 preflight.yaml 注入 clusterCreate.customChecks（check-runtime）
	// 步骤：节点一与目标节点（三/四）预置 /opt/openfuyao/check-runtime 输出 fact JSON → 执行 → 移除脚本重跑 → 换非法输出重跑
	// 预期：正常时自定义检查按节点展开 PASS 且 detail 不进报告；脚本缺失/非法输出时 FAIL 且不泄漏敏感信息
	It("【PREFLIGHT-CLUSTER-008】cluster-create 自定义 exec 检查", Label("required_nodes:2", "PREFLIGHT-CLUSTER-008", "P2", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			const (
				scriptDir  = "/opt/openfuyao"
				scriptPath = scriptDir + "/check-runtime"
				detailMark = "pf-e2e-detail-should-not-leak"
			)
			By("准备含自定义 exec 检查的策略（env 覆盖优先，否则基于包内 preflight.yaml）")
			policy, usingFixture := h.policyFixturePath(
				strings.TrimSpace(os.Getenv("PREFLIGHT_CLUSTER_POLICY_CUSTOM_EXEC")),
				"preflight-custom-exec.yaml", preflightPolicyCustomExecContent())
			if !usingFixture {
				Expect(h.fileExists(policy)).To(BeTrue(), "自定义策略不存在: %s", policy)
			}

			// cluster-create 的自定义 exec 检查按 roles 展开到各目标节点并经 SSH 在目标节点上执行
			// （CSV 前置条件：目标节点预置脚本），脚本须同时部署到节点一（bke 执行节点）与目标节点二/三
			deployNodes := []*executor.SSHExecutor{h.ssh}
			for _, idx := range []int{2, 3} {
				nodeExec := newNodeExecutor(idx)
				deployNodes = append(deployNodes, nodeExec)
				// 清理：删除预置脚本（目录仅在为空时移除）并断开连接
				defer func(nodeExec *executor.SSHExecutor) {
					_, _ = nodeExec.Exec(fmt.Sprintf("rm -f %s %s.bak; rmdir %s 2>/dev/null || true",
						shq(scriptPath), shq(scriptPath), shq(scriptDir)))
					_ = nodeExec.Close()
				}(nodeExec)
			}
			runOn := func(exec *executor.SSHExecutor, cmd string) *executor.ExecResult {
				res, err := exec.Exec(cmd)
				Expect(res).NotTo(BeNil(), "命令应返回结果: %s", cmd)
				Expect(err).NotTo(HaveOccurred(), "命令执行应成功: %s\nstderr:\n%s", cmd, res.Stderr)
				Expect(res.ExitCode).To(Equal(0), "命令退出码应为0: %s\nstdout:\n%s\nstderr:\n%s", cmd, res.Stdout, res.Stderr)
				return res
			}
			deployScript := func(body string) {
				for _, exec := range deployNodes {
					runOn(exec, "mkdir -p "+scriptDir)
					runOn(exec, fmt.Sprintf("cat > %s <<'PFEOF'\n%s\nPFEOF", shq(scriptPath), body))
					runOn(exec, "chmod +x "+shq(scriptPath))
				}
			}
			// 清理：节点一上的脚本由 harness 统一清理
			h.addCleanupCmd(fmt.Sprintf("rm -f %s %s.bak; rmdir %s 2>/dev/null || true",
				shq(scriptPath), shq(scriptPath), shq(scriptDir)))

			By("节点侧预置 fact 脚本（输出 fact/v1 契约 JSON，detail 位于 value 内）")
			deployScript(fmt.Sprintf(`#!/bin/bash
cat <<'JSON'
{"apiVersion":"preflight.openfuyao.io/fact/v1","available":true,"type":"object","value":{"runtime":"containerd","healthy":true,"detail":"%s"}}
JSON
`, detailMark))

			By("执行 cluster-create，预期自定义检查 PASS")
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-exec.yaml"), freshNodes("nodes-exec.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, usingFixture)
			expectPreflightOutcome(o, "PASS", 0)

			customItems := filterResults(o.Report, "check-runtime")
			Expect(customItems).NotTo(BeEmpty(), "自定义检查应按节点展开为检查项")
			Expect(allStatus(customItems, "PASS")).To(BeTrue(), "自定义检查应全部 PASS")
			By("校验 detail 不进报告")
			Expect(o.ReportRaw).NotTo(ContainSubstring(detailMark),
				"自定义检查的 detail 字段不应出现在报告中")

			By("移除脚本后重跑，预期自定义检查 FAIL")
			for _, exec := range deployNodes {
				runOn(exec, "mv "+shq(scriptPath)+" "+shq(scriptPath+".bak"))
			}
			o2 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-exec2.yaml"), freshNodes("nodes-exec2.yaml", baseNodes()), policy)
			expectPreflightOutcome(o2, "FAIL", 3)
			custom2 := filterResults(o2.Report, "check-runtime")
			Expect(countByStatus(custom2, "FAIL")).To(BeNumerically(">=", 1),
				"脚本缺失时自定义检查应 FAIL")
			Expect(o2.ReportRaw).NotTo(ContainSubstring(detailMark), "失败时也不应泄漏 detail")

			By("脚本输出非法 JSON 时重跑，预期自定义检查 FAIL")
			deployScript("#!/bin/bash\necho 'not-a-fact-json'\n")
			o3 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-exec3.yaml"), freshNodes("nodes-exec3.yaml", baseNodes()), policy)
			expectPreflightOutcome(o3, "FAIL", 3)
			custom3 := filterResults(o3.Report, "check-runtime")
			Expect(countByStatus(custom3, "FAIL")).To(BeNumerically(">=", 1),
				"脚本输出非法时自定义检查应 FAIL")
			Expect(o3.ReportRaw).NotTo(ContainSubstring(detailMark), "失败时也不应泄漏 detail")
		})

	// 用例编号：PREFLIGHT-CLUSTER-009（p1）
	// 前置：两节点集群（master=TEST_NODE2、worker=TEST_NODE3），master 节点 6443 可占用
	// 步骤：master-01 占用 6443 → 执行 cluster-create → 释放后重跑
	// 预期：CLUSTER-PORT-APISERVER 仅在 master 节点实例化一条 FAIL；KUBELET 端口检查两节点各一条；
	// SRC-REACHABLE 每节点一条；worker 不受影响；summary total=pass+fail+warn+skip；释放后 PASS
	It("【PREFLIGHT-CLUSTER-009】cluster-create 多节点端口检查按节点实例化", Label("required_nodes:2", "PREFLIGHT-CLUSTER-009", "P1", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			const apiPort = 6443
			masterHostname := strings.TrimSpace(os.Getenv("TEST_NODE2_HOSTNAME"))
			if masterHostname == "" {
				Skip("需要设置 TEST_NODE2_HOSTNAME 标识 master 节点（约定节点二为 master）")
			}

			By("在 master 节点(TEST_NODE2)上占用 6443")
			masterExec := newNodeExecutor(2)
			defer masterExec.Close()
			pid := occupyPortOn(masterExec, apiPort)
			defer killPIDOn(masterExec, pid)
			Expect(waitPortUsedOn(masterExec, apiPort)).To(BeTrue(), "master 节点 %d 端口应进入监听", apiPort)

			By("执行 cluster-create，预期 APISERVER 端口项单点 FAIL")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-port.yaml"), freshNodes("nodes-port.yaml", baseNodes()), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 CLUSTER-PORT-APISERVER 仅 master 一条 FAIL")
			apiPortItems := filterResults(o.Report, "PORT-APISERVER")
			Expect(apiPortItems).To(HaveLen(1),
				"APISERVER 端口检查应只在 master 节点实例化一条，实际 %d 条", len(apiPortItems))
			apiRes := apiPortItems[0]
			Expect(apiRes.Status).To(Equal("FAIL"), "master 端口被占用时 APISERVER 应 FAIL")
			Expect(instanceSuffix(apiRes)).To(ContainSubstring(masterHostname),
				"APISERVER 失败项应指向 master 节点 %s，实际 %s", masterHostname, apiRes.InstanceID)
			Expect(rawValueString(apiRes.Expected.Value)).To(Equal("true"),
				"期望值应为 true（端口空闲），实际 %s", rawValueString(apiRes.Expected.Value))
			Expect(rawValueString(apiRes.Actual.Value)).To(Equal("false"),
				"实际值应为 false（端口已被占用），实际 %s", rawValueString(apiRes.Actual.Value))
			Expect(apiRes.Name).To(ContainSubstring("6443"),
				"检查项名称应包含端口 6443，实际 %s", apiRes.Name)

			By("校验 KUBELET 端口检查两节点各一条不重复")
			kubeletItems := filterResults(o.Report, "KUBELET")
			Expect(kubeletItems).To(HaveLen(4),
				"两节点 KUBELET-SECURE/HEALTH 应各一条共 4 条，实际 %d 条", len(kubeletItems))
			suffixes := map[string]int{}
			for _, res := range kubeletItems {
				suffixes[instanceSuffix(res)]++
			}
			Expect(suffixes).To(HaveLen(2), "KUBELET 检查应覆盖两个节点: %v", suffixes)
			for s, n := range suffixes {
				Expect(n).To(Equal(2), "节点 %s 的 KUBELET 检查应恰为 2 条", s)
			}

			By("校验 CLUSTER-SRC-REACHABLE 每节点每来源各一条")
			srcItems := filterResults(o.Report, "SRC-REACHABLE")
			srcNodes := map[string]int{}
			srcSuffixes := map[string]int{}
			for _, res := range srcItems {
				// instanceID 格式为 @<node>:<source>，按节点分组取冒号前段
				node := strings.SplitN(instanceSuffix(res), ":", 2)[0]
				srcNodes[node]++
				srcSuffixes[instanceSuffix(res)]++
			}
			Expect(srcNodes).To(HaveLen(2), "来源可达检查应覆盖两个节点: %v", srcNodes)
			Expect(srcSuffixes).To(HaveLen(6), "来源可达检查应为每 (节点,来源) 各一条: %v", srcSuffixes)
			for s, n := range srcSuffixes {
				Expect(n).To(Equal(1), "(节点,来源) %s 的来源可达检查应恰为 1 条", s)
			}

			By("校验 worker 节点不受影响且 summary 一致")
			expectSummaryArithmetic(o.Report)
			Expect(o.Report.Summary.Fail).To(Equal(1),
				"应只有 master APISERVER 一项 FAIL，worker 不受影响")

			By("释放端口后重跑，预期 PASS")
			killPIDOn(masterExec, pid)
			Expect(waitPortFreeOn(masterExec, apiPort)).To(BeTrue(), "master 节点 6443 应已释放")
			o2 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-port2.yaml"), freshNodes("nodes-port2.yaml", baseNodes()), policy)
			expectPreflightOutcome(o2, "PASS", 0)
		})

	// 用例编号：PREFLIGHT-CLUSTER-010（p2）
	// 步骤：nodes.yaml 中节点二、三均声明 master/etcd 角色（2 个 master，偶数）→ 执行 cluster-create
	// 预期：CLUSTER-TOPO-MASTER-ODD FAIL、exit 3，机器报告以 relation 表达（期望奇数 true、实际 false），subject=cluster，其余检查正常执行
	It("【PREFLIGHT-CLUSTER-010】cluster-create master 偶数拓扑检查失败", Label("required_nodes:2", "PREFLIGHT-CLUSTER-010", "P2", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("构造两 master 节点定义（节点二、三均声明 master/etcd 角色）")
			nodes := baseNodes()
			nodes[1].Roles = []string{"master/node", "etcd"}

			By("以偶数 master 配置执行 cluster-create")
			policy := h.ensureDefaultPolicy()
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-topo.yaml"), freshNodes("nodes-topo.yaml", nodes), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 CLUSTER-TOPO-MASTER-ODD 单项 FAIL")
			topo := findResult(o.Report, "MASTER-ODD")
			Expect(topo).NotTo(BeNil(), "应存在 CLUSTER-TOPO-MASTER-ODD 检查项")
			Expect(topo.Status).To(Equal("FAIL"), "master 偶数时拓扑检查应 FAIL")
			Expect(rawValueString(topo.Expected.Value)).To(Equal("true"),
				"期望值应为 true（master 数为奇数），实际 %s", rawValueString(topo.Expected.Value))
			Expect(rawValueString(topo.Actual.Value)).To(Equal("false"),
				"实际值应为 false（master 数为偶数），实际 %s", rawValueString(topo.Actual.Value))
			Expect(topo.Subject).To(Equal("cluster"),
				"目标应为集群拓扑，实际 subject=%s sourceRefs=%v", topo.Subject, topo.SourceRefs)

			By("校验其余检查正常执行")
			Expect(len(o.Report.Results)).To(BeNumerically(">", 1), "其余检查不应被阻塞")
			Expect(o.Report.Summary.Pass).To(BeNumerically(">", 0), "应有 PASS 项")
		})

	// 用例编号：PREFLIGHT-CLUSTER-011（p2）
	// 前置：管理集群就绪（同 CLUSTER-001，含外部策略 preflight.yaml：cpu≥4、memory≥6GiB、systemDisk≥40GiB）
	// 场景B：两节点同 IP（hostname 不同，指向同一真实机器）→ CLUSTER-UNIQUE-IP 恰一条 FAIL 且 remediation 非空
	//        （bke v26.06 报告不含冲突详情，目标列细节要求经用户裁决回退），两份节点定义的 SSH 采集各自执行不崩溃，Overall=FAIL、exit 3
	// 场景C：另备合法 nodes.yaml，两节点 hostname 为空（metadata.name 保持填充）、IP 不同 →
	//        空 hostname 不参与 hostname 唯一性检查，相关检查项 PASS，不产生 SKIP/FAIL，其余满足则 PASS
	// 场景D：另备合法 nodes.yaml，两节点 hostname 相同、IP 不同 → CLUSTER-UNIQUE-HOSTNAME 恰一条 FAIL 且 remediation 非空
	//        （目标列细节要求经用户裁决回退，同场景B），Overall=FAIL、exit 3
	It("【PREFLIGHT-CLUSTER-011】cluster-create 节点唯一性检查", Label("required_nodes:2", "PREFLIGHT-CLUSTER-011", "P2", "online"),
		SpecTimeout(preflightClusterItTimeout), func(ctx SpecContext) {
			By("场景B：两节点同 IP（hostname 不同）执行 cluster-create")
			policy := h.ensureDefaultPolicy()
			dupNodes := baseNodes()
			dupNodes[1].IP = dupNodes[0].IP
			o := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-uni-b.yaml"), freshNodes("nodes-uni-b.yaml", dupNodes), policy)
			h.skipIfPolicyRejected(o, h.policyTemplate == "")
			expectPreflightOutcome(o, "FAIL", 3)

			By("校验 CLUSTER-UNIQUE-IP 恰一条 FAIL 且 remediation 非空")
			uniq := filterResults(o.Report, "UNIQUE-IP")
			Expect(uniq).To(HaveLen(1), "IP 唯一性检查应恰一条，实际 %d 条", len(uniq))
			Expect(uniq[0].Status).To(Equal("FAIL"), "两节点同 IP 时 UNIQUE-IP 应 FAIL")
			Expect(uniq[0].Remediation).NotTo(BeEmpty(), "UNIQUE-IP FAIL 应给出修复建议")
			// bke 对两份相同 IP 的 BKENode 会同时触发本地输入校验（CLUSTER-MGMT-LOCAL-NODE）FAIL，
			// 属合理的本地输入校验行为；CSV 契约仅要求 UNIQUE-IP 恰一条 FAIL，不约束全局 FAIL 数。

			By("校验 SSH 采集在冲突场景下各自执行不崩溃")
			nodeScoped := 0
			for _, res := range o.Report.Results {
				if strings.Contains(res.InstanceID, "@") {
					nodeScoped++
				}
			}
			Expect(nodeScoped).To(BeNumerically(">=", 2),
				"两节点 SSH 采集应各自执行，实际节点级检查 %d 条", nodeScoped)

			By("场景C：两节点 hostname 为空（metadata.name 保留）、IP 不同，执行 cluster-create")
			emptyNodes := baseNodes()
			// 仅 spec.hostname 置空；metadata.name 来自 Name 字段（构造时取 hostname 原值），
			// bke 以 metadata.name 作为 plan subject，不应为空
			emptyNodes[0].Hostname = ""
			emptyNodes[1].Hostname = ""
			o2 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-uni-c.yaml"), freshNodes("nodes-uni-c.yaml", emptyNodes), policy)
			expectPreflightOutcome(o2, "PASS", 0)

			By("校验空 hostname 不参与 hostname 唯一性检查：相关检查项 PASS，不产生 SKIP/FAIL")
			Expect(o2.Report.Summary.Fail).To(Equal(0), "不应产生 FAIL")
			Expect(o2.Report.Summary.Skip).To(Equal(0), "不应产生 SKIP")
			Expect(allStatus(filterResults(o2.Report, "UNIQUE"), "PASS")).To(BeTrue(),
				"唯一性相关检查项应 PASS: %+v", filterResults(o2.Report, "UNIQUE"))

			By("场景D：两节点 hostname 相同（IP 不同）执行 cluster-create")
			dupHostNodes := baseNodes()
			dupHostNodes[1].Hostname = dupHostNodes[0].Hostname
			o3 := h.runClusterCreate(h.kubeconfig, freshCluster("cluster-uni-d.yaml"), freshNodes("nodes-uni-d.yaml", dupHostNodes), policy)
			h.skipIfPolicyRejected(o3, h.policyTemplate == "")
			expectPreflightOutcome(o3, "FAIL", 3)

			By("校验 CLUSTER-UNIQUE-HOSTNAME 恰一条 FAIL 且 remediation 非空")
			hostUniq := filterResults(o3.Report, "UNIQUE-HOSTNAME")
			Expect(hostUniq).To(HaveLen(1), "hostname 唯一性检查应恰一条，实际 %d 条", len(hostUniq))
			Expect(hostUniq[0].Status).To(Equal("FAIL"), "两节点 hostname 相同时 UNIQUE-HOSTNAME 应 FAIL")
			Expect(hostUniq[0].Remediation).NotTo(BeEmpty(), "UNIQUE-HOSTNAME FAIL 应给出修复建议")
			// CSV 契约要求 UNIQUE-HOSTNAME 恰一条 FAIL，不约束全局 FAIL 数。
		})
})
