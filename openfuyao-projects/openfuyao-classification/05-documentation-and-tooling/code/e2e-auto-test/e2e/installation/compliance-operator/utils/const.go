// Package utils 为 compliance-operator compliance-hardening E2E 提供配置、SSH 执行、
// 指纹采集、报告解析与复扫等公共能力。节点 IP/凭据经环境变量注入，固定测试参数集中定义于本文件。
package utils

// 环境变量名（见 e2e/.env.template 中 compliance-operator 段）
const (
	EnvEntryHost          = "COMPLIANCE_ENTRY_HOST"
	EnvEntryPort          = "COMPLIANCE_ENTRY_PORT"
	EnvEntryUser          = "COMPLIANCE_ENTRY_USER"
	EnvEntryPassword      = "COMPLIANCE_ENTRY_PASSWORD"
	EnvHardeningSourceDir = "COMPLIANCE_HARDENING_SOURCE_DIR"

	EnvNodeN = "COMPLIANCE_NODE%d_%s" // COMPLIANCE_NODE2_IP / _ROLE / _PORT / _USER / _PASSWORD（下标与 TEST_NODE 槽位对齐）

	// 复用 e2e/.env.template 既有字段（其他 e2e 模块同源使用，语义不变，仅作本套件的取值回退）：
	// entry 侧回退 TEST_NODE2_*（首个业务 master，即 compliance entry_node），集群节点侧回退
	// TEST_NODE2~5_*（IP/HOSTNAME/PORT/USER/PASSWORD）。取值优先级：COMPLIANCE_* 显式值 >
	// TEST_NODE_* 复用值 > 缺配置即停（禁止静默猜测）。GUIDE_NODE_*（=TEST_NODE1，管理/引导面）
	// 与 compliance 被测业务集群无关，不参与读取链。
	EnvTestNodeN = "TEST_NODE%d_%s" // TEST_NODE2_IP / _HOSTNAME / _PORT / _USER / _PASSWORD

	// 业务集群节点槽位（.env.template 通用拓扑口径）：TEST_NODE2~4 = 业务集群 3 master，
	// TEST_NODE5 = 业务集群 worker；TEST_NODE1（=GUIDE_NODE，管理集群与引导节点共部署）不参与。
	FirstClusterNodeSlot = 2
	LastClusterNodeSlot  = 5
)

// CI 与本地共用的固定测试参数。这些值不再从 .env 或系统环境变量读取。
const (
	ComplianceHardeningDir        = "/root/compliance-operator/compliance-hardening"
	ComplianceClusterYAML         = ComplianceHardeningDir + "/config/cluster.yaml"
	ComplianceBackupDir           = "/var/lib/compliance-hardening/backups"
	ComplianceReportDir           = "/var/lib/compliance-hardening/reports"
	ComplianceLogDir              = "/var/log/compliance-hardening"
	ComplianceWorkDir             = "/tmp/compliance-e2e"
	ComplianceKubeletConfig       = "/var/lib/kubelet/config.yaml"
	ComplianceScanMode            = "auto" // 镜像可用时复扫，不可用时跳过对应断言块
	ComplianceKubeBenchImage      = "hub.oepkgs.net/openfuyao/aquasec/kube-bench:v0.15.6"
	ComplianceOpenSCAPImage       = "cr.openfuyao.cn/openfuyao/openscap:1.4.4"
	ComplianceBizNamespace        = "fit-test"
	ComplianceBizDeployment       = "nginx-fit"
	ComplianceBizImage            = "docker.io/library/busybox:latest"
	ComplianceRequireNodeOverride = true

	RoleMaster = "master"
	RoleWorker = "worker"
)

// 被检文件（S0 指纹 / 分组断言口径，与 compliance-hardening g1-g6 规则对应）
const (
	ManifestDir = "/etc/kubernetes/manifests"
	AuditPolicy = "/etc/kubernetes/audit-policy.yaml"
	CACert      = "/etc/kubernetes/pki/ca.crt"

	ManifestAPIServer  = ManifestDir + "/kube-apiserver.yaml"
	ManifestController = ManifestDir + "/kube-controller-manager.yaml"
	ManifestScheduler  = ManifestDir + "/kube-scheduler.yaml"
	ManifestEtcd       = ManifestDir + "/etcd.yaml"

	// 加固产物形态（与 lib/cluster.sh、用户使用指南口径一致）
	HardenOriginMarker  = ".harden-origin" // 备份目录下的加固起点标记（回退后自动清除）
	ResultsDirGlob      = "/tmp/cluster-harden-results-*"
	RollbackResultsGlob = "/tmp/cluster-rollback-results-*"
	HardenLogGlob       = "cluster-harden-*.log"
	RollbackLogGlob     = "cluster-rollback-*.log"
	ReportJSONGlob      = "cluster-harden-report-*.json"
	ReportMDGlob        = "cluster-harden-report-*.md"

	// 关键日志/输出特征串（与源码 lib/cluster.sh、bin/k8s-hardening.sh 逐字对齐）
	MsgUnifiedTS        = "本次集群加固备份时间戳（所有节点统一）："
	MsgHardenDone       = "=== 全集群加固完成 ==="
	MsgDryRunDone       = "=== 全集群预览完成（未执行任何变更）==="
	MsgConfigNotExist   = "集群配置文件不存在"
	MsgConfigYAMLError  = "集群配置 YAML 解析失败"
	MsgNoPasswordPrefix = "以下节点未配置密码（既无节点 password 也无 default_password）"
	MsgSSHFailed        = "SSH 连接失败"
	MsgNoPendingNode    = "无待整改项（全部已合规或高危跳过），跳过加固执行"
	MsgRollbackCanceled = "用户取消，未执行任何回退"
	MsgBackupMissing    = "在 %d 个节点上不存在，已中止（未做任何修改）"
	MsgBackupMissNode   = "✗ 缺失（可用备份："
	// 确认块断言取块内独有的影响提示行（lib/cluster.sh:815）：confirm 的 read -p 问句仅在
	// stdin 为终端时显示，测试经管道注入应答时该问句不出现在输出中（run8 实测对齐）。
	MsgConfirmRollback = "回退会修改各节点配置并触发静态 Pod/kubelet 重建"
	MsgNoDiffRestore   = "当前配置与备份一致（无内容差异），无需恢复文件"
	MsgNoModifyRestart = "本次回退未修改任何配置（kubelet config/manifest 均无变化），无需重启与等待"
	MsgFileRestoreDone = "文件级恢复完成："
	MsgWorkerParallel  = "[worker 并行]"

	// 集群统一报告 md 四部分结构（lib/cluster.sh cluster_gen_report）
	RptSectionSummary = "## 集群汇总"
	RptSectionNodes   = "## 各节点详情"
	RptSectionRules   = "## 各节点详细规则结果"
	RptManualWarn     = "高危项已跳过，需人工评估"
	RptOldNewHeader   = "| 旧值 | 新值 |"
)

// 高危/人工项规则 ID（环境自适应断言的允许残留集合，取自 config/hardening-rules.yaml）
var (
	AllowedCISRemainAfterHarden  = []string{"1.1.12"}                                                                        // cis-1.1.12 etcd 属主（高危，etcd 以 root 运行时跳过）
	AllowedSTIGRemainAfterHarden = []string{"SV-242403", "SV-242404", "SV-242445"}                                           // 审计策略人工项 + hostname-override + etcd 属主
	HighRiskSkipRuleIDs          = []string{"cis-1.1.12", "stig-sv242445", "stig-sv242404"}                                  // dry-run 预览中应可见的高危跳过标记
	SecretRuleIDs                = []string{"cis-1.2.5", "stig-sv242449", "stig-sv242450", "stig-sv242424", "stig-sv242425"} // DFX-04 涉密规则抽查面
)
