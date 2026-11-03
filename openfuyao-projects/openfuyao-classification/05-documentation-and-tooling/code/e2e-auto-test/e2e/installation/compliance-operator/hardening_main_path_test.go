package compliance_operator

import (
	"fmt"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// E2E-A implements XLSX FIT-01..FIT-09 and FIT-15 with one Ginkgo test point per reviewed case.
//
// The Ordered container performs hardening once and keeps dependent assertions traceable:
// S0 基线采集（指纹/业务负载/控制面 Pod/audit md5/节点名/双基线扫描）
// → 步骤1 执行加固 → 步骤2 逐组合规断言 → 步骤3 etcd 高危跳过 → 步骤4 hostname-override
// → 步骤5 audit-policy 零修改 → 步骤6/7 双复扫对比（镜像缺失时降级跳过）
// → 步骤8 统一报告解析 → 步骤9 业务验证 → 步骤10 产物收集。
func defineMainPathSpecs() {
	Describe("[E2E-A] Hardening Main Flow",
		Ordered, Label("E2E-A", "hardening_main_path", "P0", "26.09", "online"), func() {

			var (
				fpS0          utils.FingerprintSet
				bizPods       []PodSnapshot
				cmRestarts    map[string]int
				schedRestarts map[string]int
				auditMD5      map[string]string
				etcdBefore    map[string]utils.EtcdDataDirState
				nodeNames     []string

				hardenOut string
				ts1       string
				reportMD  string
			)

			BeforeAll(func() {
				requireUnhardened("E2E-A")
				requireWorkerKubeletHAEndpoint()
				By("采集 S0 指纹基线")
				fpS0 = collectAllFingerprints()
				shared.s0Fingerprints = fpS0
				Expect(utils.WriteStateFile(entry, cfg, "s0-fingerprints.txt", fpS0.FingerprintText())).
					NotTo(HaveOccurred(), "persist the E2E-A S0 fingerprint baseline for a separate E2E-C run")

				By("校验 cluster.yaml 节点级差异配置形态（前置 3：至少 1 个节点独立 password 覆盖）")
				if cfg.RequireNodeOverride {
					has, err := utils.ClusterYAMLHasNodeOverride(entry, cfg)
					Expect(err).NotTo(HaveOccurred())
					Expect(has).To(BeTrue(),
						"cluster.yaml 应含至少 1 个节点的独立 password 覆盖（混合凭据形态，凭据校验不回显口令）")
				}

				By("准备业务负载并记录 Pod UID/所在节点（前置 5）")
				bizPods = ensureBusinessWorkload()

				By("记录控制面 Pod 重启数与节点注册名（CrashLoop / hostname-override 对照基线）")
				cmRestarts = systemPodRestarts("kube-controller-manager")
				schedRestarts = systemPodRestarts("kube-scheduler")
				nodeNames = registeredNodeNames()

				By("采集 audit-policy md5 基线（3 台 master）")
				auditMD5 = auditPolicyMD5()

				By("capture each master's detected etcd data-dir and owner/group before hardening")
				etcdBefore = map[string]utils.EtcdDataDirState{}
				for _, n := range cfg.Masters() {
					state, err := utils.CaptureEtcdDataDirState(nodeExecs[n.IP])
					Expect(err).NotTo(HaveOccurred(), fmt.Sprintf(
						"node=%s detected-data-dir=<unavailable> before=<unavailable> after=<not-captured>: %v", n.IP, err))
					etcdBefore[n.IP] = state
					GinkgoWriter.Printf("etcd baseline node=%s data-dir=%s owner/group=%s\n", n.IP, state.Path, state.OwnerGroup)
				}

				By("采集 CIS/STIG 双基线扫描结果（复扫镜像不可用时降级跳过并记录，不阻塞其余断言）")
				shared.cisBaseline, shared.stigBaseline = nil, nil
				shared.scanSkipReason = ""
				if cfg.ScanEnabledOn() {
					if !utils.ImageAvailable(entry, cfg.KubeBenchImage) {
						shared.scanSkipReason = fmt.Sprintf("kube-bench 镜像不可用: %s", cfg.KubeBenchImage)
					}
					if !utils.ImageAvailable(entry, cfg.OpenSCAPImage) {
						if shared.scanSkipReason != "" {
							shared.scanSkipReason += "；"
						}
						shared.scanSkipReason += fmt.Sprintf("openscap 镜像不可用: %s", cfg.OpenSCAPImage)
					}
					if shared.scanSkipReason == "" {
						cis, err := utils.RunKubeBench(entry, cfg)
						Expect(err).NotTo(HaveOccurred(), "CIS 基线复扫应成功")
						shared.cisBaseline = cis
						stig, err := utils.RunOpenSCAP(entry, cfg)
						Expect(err).NotTo(HaveOccurred(), "STIG 基线复扫应成功")
						shared.stigBaseline = stig
						GinkgoWriter.Printf("CIS 基线: PASS=%d FAIL=%d WARN=%d FAIL项=%v\n",
							cis.Pass, cis.Fail, cis.Warn, cis.FailItems)
						GinkgoWriter.Printf("STIG 基线: FAIL项=%v NA项=%v\n", stig.FailItems, stig.NAItems)
					}
				} else {
					shared.scanSkipReason = "COMPLIANCE_SCAN_MODE=off"
				}
				if shared.scanSkipReason != "" {
					GinkgoWriter.Printf("[降级] 双基线扫描跳过：%s（结果6 断言块将降级跳过）\n", shared.scanSkipReason)
				}
			})

			Context("[E2E-A][FIT-01] Full-cluster hardening main path", Label("FIT-01"), func() {
				It("[E2E-A][FIT-01] should harden the entire cluster with serial masters, parallel workers, and per-node overrides", func() {
					By("执行 bash bin/k8s-hardening.sh cluster-harden config/cluster.yaml -y")
					res := utils.ClusterHarden(entry, cfg, "-y")
					hardenOut = res.Stdout + "\n" + res.Stderr

					// cluster-harden 在修改任何节点前创建全节点统一备份并打印时间戳。
					// 即使后续健康检查失败，也先尽力保存该时间戳，避免失败断言中止当前
					// It 后丢失人工恢复所需的准确回退目标。
					By("解析并尽力保存统一备份时间戳 TS1（包括失败执行）")
					ts1 = utils.ParseUnifiedTimestamp(hardenOut)
					var stateWriteErr error
					if ts1 != "" {
						shared.unifiedTS = ts1
						stateWriteErr = utils.WriteStateFile(entry, cfg, "harden-ts.txt", ts1)
						GinkgoWriter.Printf("cluster-harden recovery timestamp=%s state-file=%s/harden-ts.txt write-error=%v\n",
							ts1, cfg.WorkDir, stateWriteErr)
					} else {
						GinkgoWriter.Printf("cluster-harden recovery timestamp was not found in command output\n")
					}
					healthReadyRetries := strings.Count(hardenOut, "节点暂未达就绪阈值")
					GinkgoWriter.Printf("cluster health ready-threshold retries=%d\n", healthReadyRetries)

					Expect(res.ExitCode).To(BeZero(), fmt.Sprintf(
						"cluster-harden 应成功（exit=0）；recovery timestamp=%q，state-file write error=%v，health ready-threshold retries=%d；输出尾部:\n%s",
						ts1, stateWriteErr, healthReadyRetries, tail(hardenOut)))
					Expect(hardenOut).To(ContainSubstring(utils.MsgHardenDone))

					Expect(ts1).NotTo(BeEmpty(), "输出应含「本次集群加固备份时间戳（所有节点统一）：<TS1>」")
					Expect(stateWriteErr).NotTo(HaveOccurred(), "应保存统一备份时间戳供后续回退使用")

					By("断言 master 严格串行（[master 1/3] 至 [master 3/3] 依次出现）且 worker 并行段在其后")
					Expect(utils.MasterSerialOrdered(hardenOut, len(cfg.Masters()))).To(BeTrue(),
						"master 应严格串行加固且 worker 并行段在其后")

					By("断言混合凭据全节点可达：无 SSH 报错、无节点加固失败")
					Expect(hardenOut).NotTo(ContainSubstring(utils.MsgSSHFailed), "任何节点都不应出现 SSH 连接失败")
					Expect(hardenOut).NotTo(ContainSubstring("加固失败"), "任何节点都不应加固失败")
					Expect(hardenOut).NotTo(ContainSubstring("回退失败"), "任何节点都不应回退失败")

					By("校验每台 master 的已评审加固值")
					for _, n := range cfg.Masters() {
						assertGroup1ManifestPerms(n)
						assertGroup2APIServer(n)
						assertGroup5Etcd(n)
						assertGroup6Kubelet(n)
					}
				})
			})

			It("[E2E-A][FIT-03] should synchronize liveness probes with controller-manager and scheduler bind addresses", Label("FIT-03"), func() {
				for _, n := range cfg.Masters() {
					assertGroup34ControllerScheduler(n)
				}
				assertControlPlanePodsHealthy("kube-controller-manager", cmRestarts)
				assertControlPlanePodsHealthy("kube-scheduler", schedRestarts)
			})

			It("[E2E-A][FIT-04] should skip etcd ownership remediation when etcd runs as root", Label("FIT-04"), func() {
				By("日志/报告对 cis-1.1.12、stig-sv242445 输出 [SKIP] 高危人工标记及原因")
				Expect(hardenOut).To(MatchRegexp(`\[SKIP\]\s+cis-1\.1\.12`), "编排输出应对 cis-1.1.12 输出 [SKIP] 高危标记及原因")
				Expect(hardenOut).To(MatchRegexp(`\[SKIP\]\s+stig-sv242445`), "编排输出应对 stig-sv242445 输出 [SKIP] 高危标记及原因")

				By("detect each master's etcd data-dir again and compare owner/group with the same node's baseline")
				for _, n := range cfg.Masters() {
					before, ok := etcdBefore[n.IP]
					Expect(ok).To(BeTrue(), fmt.Sprintf(
						"node=%s detected-data-dir=<unavailable> before=<missing> after=<not-captured>: baseline was not recorded", n.IP))
					after, err := utils.CaptureEtcdDataDirState(nodeExecs[n.IP])
					Expect(err).NotTo(HaveOccurred(), fmt.Sprintf(
						"node=%s detected-data-dir=%s before=%s after=<unavailable>: %v", n.IP, before.Path, before.OwnerGroup, err))
					Expect(after.Path).To(Equal(before.Path), fmt.Sprintf(
						"etcd data-dir changed on node %s: before-path=%s after-path=%s before=%s after=%s",
						n.IP, before.Path, after.Path, before.OwnerGroup, after.OwnerGroup))
					Expect(after.OwnerGroup).To(Equal(before.OwnerGroup), fmt.Sprintf(
						"etcd data-dir owner/group changed on node %s: data-dir=%s before=%s after=%s",
						n.IP, after.Path, before.OwnerGroup, after.OwnerGroup))
				}

				By("etcd 容器保持 Running")
				etcdState, err := utils.ExecTrim(entry,
					`kubectl -n kube-system get pods -l component=etcd -o jsonpath='{range .items[*]}{.status.phase}{"\n"}{end}'`)
				Expect(err).NotTo(HaveOccurred())
				Expect(etcdState).To(ContainSubstring("Running"), "etcd 容器应保持 Running")

			})

			It("[E2E-A][FIT-05] should skip an active kubelet hostname override and preserve node identity", Label("FIT-05"), func() {
				By("日志/报告对 stig-sv242404 输出 [SKIP] 高危人工标记及原因")
				Expect(hardenOut).To(MatchRegexp(`\[SKIP\]\s+stig-sv242404`), "编排输出应对 stig-sv242404 输出 [SKIP] 高危标记及原因")

				By("各节点 kubelet 启动参数仍含 --hostname-override")
				for _, n := range cfg.Nodes {
					cmdline, err := utils.KubeletCmdline(nodeExecs[n.IP])
					Expect(err).NotTo(HaveOccurred(), "读取 kubelet 启动参数应成功: "+n.IP)
					Expect(cmdline).To(ContainSubstring("--hostname-override"),
						fmt.Sprintf("节点 %s kubelet 启动参数应保留 --hostname-override（高危跳过语义）", n.IP))
				}

				By("各节点注册名与加固前一致且全部 Ready")
				Expect(registeredNodeNames()).To(Equal(nodeNames), "节点注册名不应变化")
				assertAllNodesReady()
			})

			It("[E2E-A][FIT-06] should apply only the g6 kubelet group on worker nodes", Label("FIT-06"), func() {
				Expect(strings.Count(hardenOut, "groups=g6_kubelet")).To(BeNumerically(">=", len(cfg.Workers())),
					"hardening output should identify g6_kubelet for every worker")
				for _, n := range cfg.Workers() {
					for _, manifest := range []string{utils.ManifestAPIServer, utils.ManifestController, utils.ManifestScheduler, utils.ManifestEtcd} {
						Expect(utils.FileExists(nodeExecs[n.IP], manifest)).To(BeFalse(),
							fmt.Sprintf("worker node %s should not contain control-plane manifest %s", n.IP, manifest))
					}
					assertGroup6Kubelet(n)
				}
			})

			It("[E2E-A][FIT-15] should preserve audit-policy content and report SV-242403 as a manual item", Label("FIT-15"), func() {
				By("对比 3 台 master 加固前后 audit-policy md5（FIT_15 零修改断言）")
				Expect(auditPolicyMD5()).To(Equal(auditMD5), "audit-policy.yaml 加固前后 md5 应完全一致（工具零修改）")

				By("编排输出含 stig-sv242403（审计策略 has-rules）规则行")
				Expect(hardenOut).To(ContainSubstring("stig-sv242403"), "编排输出应含审计策略规则")

				By("统一报告中该规则状态为 skipped（已合规）而非 [SKIP] 高危标记")
				_, raw, err := utils.LatestReport(entry, cfg.ReportDir)
				Expect(err).NotTo(HaveOccurred(), "应能读取集群统一报告")
				report, err := utils.ParseReport(raw)
				Expect(err).NotTo(HaveOccurred())
				statuses := report.RuleStatuses("stig-sv242403")
				Expect(statuses).NotTo(BeEmpty(), "报告应含 stig-sv242403 规则结果")
				for ip, st := range statuses {
					Expect(st).To(Equal("skipped"), fmt.Sprintf("节点 %s 的 stig-sv242403 应为 skipped（实际 %s）", ip, st))
				}
			})

			Context("[E2E-A][FIT-02] Post-hardening CIS and STIG rescans", Label("FIT-02"), func() {
				It("[E2E-A][FIT-02] should reduce CIS and STIG failures to the reviewed manual exception set", func() {
					if shared.cisBaseline == nil || shared.stigBaseline == nil {
						Skip(fmt.Sprintf("复扫降级跳过：%s", shared.scanSkipReason))
					}
					By("执行加固后 CIS 复扫")
					cisAfter, err := utils.RunKubeBench(entry, cfg)
					Expect(err).NotTo(HaveOccurred(), "加固后 CIS 复扫应成功")
					GinkgoWriter.Printf("CIS 复扫: PASS=%d FAIL=%d WARN=%d FAIL项=%v\n", cisAfter.Pass, cisAfter.Fail, cisAfter.Warn, cisAfter.FailItems)

					By("断言 TOTAL 不变、基线 FAIL 项全部转 PASS（环境自适应断言，参考值以实测为准）")
					Expect(cisAfter.Total).To(Equal(shared.cisBaseline.Total), "CIS TOTAL 应与基线一致")
					cisRemaining := []string{}
					for _, item := range cisAfter.FailItems {
						if !utils.ContainsItem(shared.cisBaseline.FailItems, item) {
							cisRemaining = append(cisRemaining, item)
						}
					}
					Expect(cisRemaining).To(BeEmpty(), "加固后不应出现基线之外的新增 FAIL 项")
					cisExcess := utils.SubsetOf(cisAfter.FailItems, utils.AllowedCISRemainAfterHarden)
					Expect(cisExcess).To(BeEmpty(), fmt.Sprintf("加固后 CIS FAIL 应仅剩高危项 %v（实际残留 %v）",
						utils.AllowedCISRemainAfterHarden, cisAfter.FailItems))
					By("执行加固后 STIG 复扫")
					stigAfter, err := utils.RunOpenSCAP(entry, cfg)
					Expect(err).NotTo(HaveOccurred(), "加固后 STIG 复扫应成功")
					GinkgoWriter.Printf("STIG 复扫: FAIL项=%v NA项=%v\n", stigAfter.FailItems, stigAfter.NAItems)

					By("断言基线 FAIL 项全部转 PASS、NA 项不变、FAIL 残留仅高危/人工项")
					Expect(stigAfter.NAItems).To(Equal(shared.stigBaseline.NAItems), "STIG NA 项应与基线一致")
					stigRemaining := []string{}
					for _, item := range stigAfter.FailItems {
						if !utils.ContainsItem(shared.stigBaseline.FailItems, item) {
							stigRemaining = append(stigRemaining, item)
						}
					}
					Expect(stigRemaining).To(BeEmpty(), "加固后不应出现基线之外的新增 FAIL 项")
					stigExcess := utils.SubsetOf(stigAfter.FailItems, utils.AllowedSTIGRemainAfterHarden)
					Expect(stigExcess).To(BeEmpty(), fmt.Sprintf("加固后 STIG FAIL 应仅剩高危/人工项 %v（实际残留 %v）",
						utils.AllowedSTIGRemainAfterHarden, stigAfter.FailItems))
				})
			})

			It("[E2E-A][FIT-07] should generate complete Markdown and JSON cluster reports", Label("FIT-07"), func() {
				By("读取最新集群统一报告 JSON")
				path, raw, err := utils.LatestReport(entry, cfg.ReportDir)
				Expect(err).NotTo(HaveOccurred(), "应能读取集群统一报告 JSON")
				GinkgoWriter.Printf("报告: %s\n", path)
				report, err := utils.ParseReport(raw)
				Expect(err).NotTo(HaveOccurred())

				By("断言各节点 state=hardened（本轮已执行节点）")
				states := report.NodeStates()
				for _, n := range cfg.Nodes {
					Expect(states).To(HaveKeyWithValue(n.IP, "hardened"),
						fmt.Sprintf("节点 %s 报告 state 应为 hardened（实际 %s）", n.IP, states[n.IP]))
				}
				Expect(report.Summary.Error).To(BeZero(), "报告汇总错误数应为 0")

				By("断言 md 报告四部分结构：集群汇总 / 各节点详情 / 逐规则明细（旧值→新值）/ manual 醒目提醒")
				mdGlob := fmt.Sprintf("%s/%s", cfg.ReportDir, strings.Replace(path[strings.LastIndex(path, "/")+1:], ".json", ".md", 1))
				reportMD, err = utils.ExecTrim(entry, fmt.Sprintf("cat %s 2>/dev/null", utils.ShellPath(mdGlob)))
				Expect(err).NotTo(HaveOccurred(), "应能读取集群统一报告 md")
				Expect(reportMD).To(ContainSubstring(utils.RptSectionSummary))
				Expect(reportMD).To(ContainSubstring(utils.RptSectionNodes))
				Expect(reportMD).To(ContainSubstring(utils.RptSectionRules))
				Expect(reportMD).To(ContainSubstring(utils.RptOldNewHeader), "逐规则明细应含旧值→新值列")
				Expect(reportMD).To(ContainSubstring(utils.RptManualWarn), "md 报告应含 manual 醒目提醒")
			})

			It("[E2E-A][FIT-08] should preserve workload availability and Kubernetes API functionality after hardening", Label("FIT-08"), func() {
				assertBusinessWorkloadStable(bizPods)
				assertAllNodesReady()
			})

			It("[E2E-A][FIT-09] should retain correlated hardening logs, reports, and per-node backups", Label("FIT-09"), func() {
				By("entry_node 编排日志 cluster-harden-*.log 存在且含 TS1")
				out, err := utils.ExecTrim(entry, fmt.Sprintf(
					`grep -l "%s" %s/%s 2>/dev/null | head -1`, ts1,
					utils.ShellPath(cfg.LogDir), utils.HardenLogGlob))
				Expect(err).NotTo(HaveOccurred())
				Expect(strings.TrimSpace(out)).NotTo(BeEmpty(),
					fmt.Sprintf("%s 下应存在含 TS1 的编排日志", cfg.LogDir))

				By("各节点加固日志存在且含本轮 TS1（entry_node /tmp/cluster-harden-results-*/harden-<ip>.log 扁平布局）")
				for _, n := range cfg.Nodes {
					Expect(nodeLogExistsOnEntry(utils.ResultsDirGlob, n.IP, ts1)).To(BeTrue(),
						fmt.Sprintf("节点 %s 的加固日志应存在且含本轮 TS1 %s", n.IP, ts1))
				}

				By("统一报告 md+json 已生成（步骤8 已校验内容）")
				Expect(reportMD).NotTo(BeEmpty())

				By("4 节点备份目录统一为 TS1（含已合规节点）")
				for _, n := range cfg.Nodes {
					Expect(utils.BackupExists(nodeExecs[n.IP], cfg.BackupDir, ts1)).To(BeTrue(),
						fmt.Sprintf("节点 %s 应存在统一备份 %s", n.IP, ts1))
				}

				By("记录加固态指纹（供 DFX-02 / E2E-C 对比）")
				shared.hardenedFingerprints = collectAllFingerprints()
				Expect(utils.DiffFingerprints(fpS0, shared.hardenedFingerprints)).NotTo(BeEmpty(),
					"加固态指纹应与 S0 存在差异（集群确实被加固）")
			})
		})
}
