package preflight

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// preflight.yaml：完整外部策略（resource 阈值 memory≥6GiB、systemDisk≥40GiB），
// 由用例通过 --preflight-config 显式传入；与内嵌默认策略解耦，避免环境误 FAIL。
//
//go:embed preflight.yaml
var preflightPolicyYAML string

const (
	// init 前置检查用例超时
	preflightInitItTimeout = 15 * time.Minute
	// cluster-create 前置检查用例超时
	preflightClusterItTimeout = 45 * time.Minute
)

// preflight JSON 报告结构，字段定义见《openFuyao统一前置检查提案》。
// exit code 约定：PASS=0 WARN=2 FAIL=3 INPUT_ERROR/BUNDLE_ERROR=4。
type preflightValue struct {
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

type preflightEvidence struct {
	Kind     string          `json:"kind"`
	Locator  string          `json:"locator"`
	Observed json.RawMessage `json:"observed"`
}

type preflightResult struct {
	InstanceID  string              `json:"instanceID"`
	RuleID      string              `json:"ruleID"`
	Name        string              `json:"name"`
	Status      string              `json:"status"`
	Subject     string              `json:"subject"`
	Expected    preflightValue      `json:"expected"`
	Actual      preflightValue      `json:"actual"`
	Evidence    []preflightEvidence `json:"evidence"`
	SourceRefs  []string            `json:"sourceRefs"`
	Remediation string              `json:"remediation"`
	CauseCode   string              `json:"causeCode"`
}

type preflightSummary struct {
	Total int `json:"total"`
	Pass  int `json:"pass"`
	Fail  int `json:"fail"`
	Warn  int `json:"warn"`
	Skip  int `json:"skip"`
}

type preflightReport struct {
	SchemaVersion string            `json:"schemaVersion"`
	Operation     string            `json:"operation"`
	PlanHash      string            `json:"planHash"`
	ConfigDigest  string            `json:"configDigest"`
	BundleDigest  string            `json:"bundleDigest"`
	Overall       string            `json:"overall"`
	ExitCode      int               `json:"exitCode"`
	Summary       preflightSummary  `json:"summary"`
	Issues        []json.RawMessage `json:"issues"`
	Results       []preflightResult `json:"results"`
}

// preflightIssue 表示报告 issues[] 中的输入校验问题项（INPUT_ERROR 场景）
type preflightIssue struct {
	Code   string `json:"code"`
	Effect string `json:"effect"`
	Field  string `json:"field"`
}

// preflightIssueCodes 汇总报告 issues[] 的 code 列表（用于 Skip/Fail 诊断消息）
func preflightIssueCodes(r *preflightReport) string {
	if r == nil {
		return ""
	}
	codes := make([]string, 0, len(r.Issues))
	for _, raw := range r.Issues {
		var iss preflightIssue
		if err := json.Unmarshal(raw, &iss); err == nil && iss.Code != "" {
			codes = append(codes, iss.Code)
		}
	}
	return strings.Join(codes, ",")
}

// preflightRunOutcome 记录一次 preflight 命令的执行产物
type preflightRunOutcome struct {
	Report    *preflightReport
	ReportRaw string
	Stdout    string
	Stderr    string
	ExitCode  int
}

func (o preflightRunOutcome) CombinedOutput() string {
	return o.Stdout + "\n" + o.Stderr
}

// ---- 报告结果筛选 ----

// filterResults 按 instanceID / ruleID / name 子串（大小写不敏感）筛选检查项
func filterResults(r *preflightReport, substr string) []preflightResult {
	var out []preflightResult
	if r == nil {
		return out
	}
	upper := strings.ToUpper(substr)
	for _, res := range r.Results {
		if strings.Contains(strings.ToUpper(res.InstanceID), upper) ||
			strings.Contains(strings.ToUpper(res.RuleID), upper) ||
			strings.Contains(strings.ToUpper(res.Name), upper) {
			out = append(out, res)
		}
	}
	return out
}

func findResult(r *preflightReport, substr string) *preflightResult {
	rs := filterResults(r, substr)
	if len(rs) == 0 {
		return nil
	}
	return &rs[0]
}

func countByStatus(rs []preflightResult, status string) int {
	n := 0
	for _, res := range rs {
		if res.Status == status {
			n++
		}
	}
	return n
}

func allStatus(rs []preflightResult, status string) bool {
	if len(rs) == 0 {
		return false
	}
	for _, res := range rs {
		if res.Status != status {
			return false
		}
	}
	return true
}

// instanceSuffix 取 instanceID 中最后一个 "@" 之后的部分（检查对象标识）
func instanceSuffix(res preflightResult) string {
	if idx := strings.LastIndex(res.InstanceID, "@"); idx >= 0 {
		return res.InstanceID[idx+1:]
	}
	return res.InstanceID
}

// rawValueString 将 expected/actual 的 value 原样转为字符串（去引号）
func rawValueString(raw json.RawMessage) string {
	s := strings.TrimSpace(string(raw))
	if len(s) >= 2 && s[0] == '"' {
		var str string
		if err := json.Unmarshal(raw, &str); err == nil {
			return str
		}
	}
	return s
}

func rawValueInt64(raw json.RawMessage) (int64, bool) {
	s := strings.ReplaceAll(rawValueString(raw), ",", "")
	v, err := strconv.ParseInt(strings.TrimSpace(s), 10, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

// ---- 断言辅助 ----

// expectPreflightOutcome 校验进程退出码、报告 overall 与报告内 exitCode 三者一致
func expectPreflightOutcome(o preflightRunOutcome, overall string, exitCode int) {
	Expect(o.ExitCode).To(Equal(exitCode), "preflight 进程退出码应为 %d，实际 %d\nstdout:\n%s\nstderr:\n%s",
		exitCode, o.ExitCode, o.Stdout, o.Stderr)
	Expect(o.Report).NotTo(BeNil(), "应生成 JSON 报告文件")
	Expect(o.Report.Overall).To(Equal(overall), "报告 overall 应为 %s", overall)
	Expect(o.Report.ExitCode).To(Equal(exitCode), "报告 exitCode 应与进程退出码一致")
}

// expectSummaryArithmetic 校验 summary.total == pass+fail+warn+skip 且与明细条数一致
func expectSummaryArithmetic(report *preflightReport) {
	s := report.Summary
	Expect(s.Total).To(Equal(s.Pass+s.Fail+s.Warn+s.Skip),
		"summary.total 应等于 pass+fail+warn+skip: %+v", s)
	Expect(s.Total).To(Equal(len(report.Results)), "summary.total 应等于 results 明细条数")
}

func expectAllResultsPass(report *preflightReport) {
	Expect(report.Results).NotTo(BeEmpty(), "报告应包含检查明细")
	for _, res := range report.Results {
		Expect(res.Status).To(Equal("PASS"), "检查项 %s (%s) 应为 PASS，实际 %s",
			res.InstanceID, res.Name, res.Status)
	}
}

// ---- 环境与执行基座 ----

func envOr(key, def string) string {
	if v := strings.TrimSpace(os.Getenv(key)); v != "" {
		return v
	}
	return def
}

// shq 单引号包裹的 shell 参数
func shq(s string) string {
	return "'" + strings.ReplaceAll(s, "'", "'\\''") + "'"
}

// newNodeExecutor 按 TEST_NODE<i>_* 环境变量连接测试节点
func newNodeExecutor(nodeIdx int) *executor.SSHExecutor {
	host := os.Getenv(fmt.Sprintf("TEST_NODE%d_IP", nodeIdx))
	Expect(host).NotTo(BeEmpty(), "TEST_NODE%d_IP 环境变量必须设置", nodeIdx)
	port := 22
	if p := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_PORT", nodeIdx))); p != "" {
		var err error
		port, err = strconv.Atoi(p)
		Expect(err).NotTo(HaveOccurred(), "TEST_NODE%d_PORT 应为数字", nodeIdx)
	}
	user := envOr(fmt.Sprintf("TEST_NODE%d_USER", nodeIdx), "root")
	pass := os.Getenv(fmt.Sprintf("TEST_NODE%d_PASSWORD", nodeIdx))
	Expect(pass).NotTo(BeEmpty(), "TEST_NODE%d_PASSWORD 环境变量必须设置", nodeIdx)
	exec, err := executor.NewSSHExecutor(host, port, user, pass)
	Expect(err).NotTo(HaveOccurred(), "应该成功连接到测试节点 %d", nodeIdx)
	return exec
}

// preflightHarness 封装 preflight 用例的远程执行环境
// （init 用例 → 节点一 TEST_NODE1 引导节点；cluster-create 用例 → 节点一 TEST_NODE1 管理集群节点）
type preflightHarness struct {
	ssh             *executor.SSHExecutor
	bke             string // bke 二进制路径（所在节点上）
	hostIP          string // init --hostIP（仅 init harness）
	kubeconfig      string // 管理集群 kubeconfig（节点一上，仅 cluster harness）
	clusterTemplate string // 集群模板内容（bke 自带模板，路径见 preflightClusterTemplatePath）
	policyTemplate  string // preflight 策略模板（所在节点上，可选 env 覆盖）
	defaultPolicy   string // 已落盘到远程临时目录的默认策略路径（懒写入）
	tmpDir          string
	createdPaths    []string
	cleanupCmds     []string
	trackedPIDs     map[string]*executor.SSHExecutor // 后台占用进程 pid -> 所在节点 executor
	reportSeq       int
}

// newInitPreflightHarness 构造 init 用例 harness（在节点一 TEST_NODE1 引导节点上执行）
func newInitPreflightHarness(ssh *executor.SSHExecutor) *preflightHarness {
	return &preflightHarness{
		ssh:            ssh,
		bke:            envOr("PREFLIGHT_BKE_BIN", "/usr/local/bin/bke"),
		hostIP:         envOr("PREFLIGHT_INIT_HOST_IP", os.Getenv("TEST_NODE1_IP")),
		policyTemplate: strings.TrimSpace(os.Getenv("PREFLIGHT_INIT_POLICY_TEMPLATE")),
		trackedPIDs:    map[string]*executor.SSHExecutor{},
	}
}

// newClusterPreflightHarness 构造 cluster-create 用例 harness（在节点一 TEST_NODE1 管理集群节点上执行）
func newClusterPreflightHarness(ssh *executor.SSHExecutor) *preflightHarness {
	return &preflightHarness{
		ssh:            ssh,
		bke:            envOr("PREFLIGHT_BKE_BIN", "/usr/local/bin/bke"),
		kubeconfig:     envOr("KUBECONFIG", "/root/.kube/config"),
		policyTemplate: strings.TrimSpace(os.Getenv("PREFLIGHT_CLUSTER_POLICY_TEMPLATE")),
		trackedPIDs:    map[string]*executor.SSHExecutor{},
	}
}

// ---- 集群/节点输入模板 ----

const (
	preflightClusterName         = "pf-e2e"
	preflightClusterTemplatePath = "/bke/cluster/1master1node-cluster.yaml"
)

// ---- 外部策略（preflight.yaml）----
// 外部 --preflight-config 是全量替换、不与内嵌默认策略 merge，省略固定策略键会得到 INPUT_ERROR。
// metadata.release 须与部署 bke 的 release 一致；当前为 v26.06，bke 升级后需同步更新 preflight.yaml。
// 策略被当前 bke 版本拒绝时（INPUT_ERROR）对应用例 Skip，详见 skipIfPolicyRejected。

// preflightPolicyCustomExecContent 在 preflight.yaml 基础上将 clusterCreate 的 customChecks
// 替换为 check-runtime 自定义 exec 检查（与 CLUSTER-008 预置脚本输出对齐）
func preflightPolicyCustomExecContent() string {
	const marker = "customChecks: []"
	const block = `customChecks:
      - key: check-runtime
        description: 节点容器运行时状态检查
        expected: runtime 为 containerd 且 healthy 为 true
        collector:
          type: exec
          path: /opt/openfuyao/check-runtime
          args: [--format, json]
          resultType: object
          publicFields: [runtime, healthy]
        expression: actual.healthy && actual.runtime == 'containerd'
        failure: fail
        unknown: fail
        remediation: 检查节点容器运行时状态后重试`
	idx := strings.LastIndex(preflightPolicyYAML, marker)
	if idx < 0 {
		panic("preflight.yaml 缺少 clusterCreate customChecks 占位")
	}
	return preflightPolicyYAML[:idx] + block + preflightPolicyYAML[idx+len(marker):]
}

// preflightNode nodes.yaml 中单个 BKENode 的字段
type preflightNode struct {
	// Name BKENode metadata.name；为空时渲染回退使用 Hostname
	Name     string
	Hostname string
	IP       string
	Port     string
	Username string
	Password string
	Roles    []string
}

// preflightEnvNodes 从 TEST_NODE2/TEST_NODE3 环境变量构造两节点定义
// （约定节点二为 master、节点三为 worker）
func preflightEnvNodes() []preflightNode {
	master := preflightNode{
		Hostname: strings.TrimSpace(os.Getenv("TEST_NODE2_HOSTNAME")),
		IP:       strings.TrimSpace(os.Getenv("TEST_NODE2_IP")),
		Port:     envOr("TEST_NODE2_PORT", "22"),
		Username: envOr("TEST_NODE2_USER", "root"),
		Password: os.Getenv("TEST_NODE2_PASSWORD"),
		Roles:    []string{"master/node", "etcd"},
	}
	worker := preflightNode{
		Hostname: strings.TrimSpace(os.Getenv("TEST_NODE3_HOSTNAME")),
		IP:       strings.TrimSpace(os.Getenv("TEST_NODE3_IP")),
		Port:     envOr("TEST_NODE3_PORT", "22"),
		Username: envOr("TEST_NODE3_USER", "root"),
		Password: os.Getenv("TEST_NODE3_PASSWORD"),
		Roles:    []string{"node"},
	}
	Expect(master.Hostname).NotTo(BeEmpty(), "TEST_NODE2_HOSTNAME 必须设置（节点二为 master）")
	Expect(master.IP).NotTo(BeEmpty(), "TEST_NODE2_IP 必须设置")
	Expect(master.Password).NotTo(BeEmpty(), "TEST_NODE2_PASSWORD 必须设置")
	Expect(worker.Hostname).NotTo(BeEmpty(), "TEST_NODE3_HOSTNAME 必须设置（节点三为 worker）")
	Expect(worker.IP).NotTo(BeEmpty(), "TEST_NODE3_IP 必须设置")
	Expect(worker.Password).NotTo(BeEmpty(), "TEST_NODE3_PASSWORD 必须设置")
	master.Name = master.Hostname
	worker.Name = worker.Hostname
	return []preflightNode{master, worker}
}

// renderPreflightNodesYAML 按 BKENode 多文档格式渲染 nodes.yaml（与 bke 集群创建输入一致）
func renderPreflightNodesYAML(nodes []preflightNode) string {
	var sb strings.Builder
	for i, node := range nodes {
		if i > 0 {
			sb.WriteString("---\n")
		}
		// metadata.name 独立于 spec.hostname：hostname 可为空，name 需保持填充
		name := node.Name
		if strings.TrimSpace(name) == "" {
			name = node.Hostname
		}
		sb.WriteString("apiVersion: bke.bocloud.com/v1beta1\n")
		sb.WriteString("kind: BKENode\n")
		sb.WriteString("metadata:\n")
		sb.WriteString(fmt.Sprintf("  name: %s\n", name))
		sb.WriteString(fmt.Sprintf("  namespace: bke-%s\n", preflightClusterName))
		sb.WriteString("  labels:\n")
		sb.WriteString(fmt.Sprintf("    cluster.x-k8s.io/cluster-name: bke-%s\n", preflightClusterName))
		sb.WriteString("spec:\n")
		sb.WriteString(fmt.Sprintf("  hostname: %s\n", node.Hostname))
		sb.WriteString(fmt.Sprintf("  ip: %s\n", node.IP))
		sb.WriteString(fmt.Sprintf("  password: '%s'\n", node.Password))
		sb.WriteString(fmt.Sprintf("  port: \"%s\"\n", node.Port))
		sb.WriteString(fmt.Sprintf("  username: %s\n", node.Username))
		sb.WriteString("  role:\n")
		for _, r := range node.Roles {
			sb.WriteString(fmt.Sprintf("  - %s\n", r))
		}
		sb.WriteString("status: {}\n")
	}
	return sb.String()
}

// loadClusterTemplate 加载 bke 自带集群模板内容：优先当前节点（节点一），未找到时回退引导节点（TEST_NODE1）
func (h *preflightHarness) loadClusterTemplate() {
	if strings.TrimSpace(h.clusterTemplate) != "" {
		return
	}
	cmd := "cat " + shq(preflightClusterTemplatePath)
	if res, _ := h.ssh.Exec(cmd); res != nil && res.ExitCode == 0 && strings.TrimSpace(res.Stdout) != "" {
		h.clusterTemplate = res.Stdout
		return
	}
	bootstrap := newNodeExecutor(1)
	defer func() { _ = bootstrap.Close() }()
	res, _ := bootstrap.Exec(cmd)
	Expect(res).NotTo(BeNil(), "读取集群模板应返回结果")
	Expect(res.ExitCode).To(Equal(0), "节点一上未找到集群模板 %s", preflightClusterTemplatePath)
	h.clusterTemplate = res.Stdout
}

// run 执行命令并断言退出码为 0
func (h *preflightHarness) run(cmd string) *executor.ExecResult {
	res, err := h.ssh.Exec(cmd)
	if res != nil {
		GinkgoWriter.Printf("cmd: %s\nexit=%d\nstdout:\n%s\nstderr:\n%s\n", cmd, res.ExitCode, res.Stdout, res.Stderr)
	}
	Expect(res).NotTo(BeNil(), "命令应返回结果: %s", cmd)
	Expect(err).NotTo(HaveOccurred(), "命令执行应成功(exit 0): %s\nstderr:\n%s", cmd, res.Stderr)
	Expect(res.ExitCode).To(Equal(0), "命令退出码应为0: %s\nstdout:\n%s\nstderr:\n%s", cmd, res.Stdout, res.Stderr)
	return res
}

// tryRun 执行命令但不校验退出码
func (h *preflightHarness) tryRun(cmd string) *executor.ExecResult {
	res, _ := h.ssh.Exec(cmd)
	if res != nil {
		GinkgoWriter.Printf("cmd(try): %s\nexit=%d\nstdout:\n%s\nstderr:\n%s\n", cmd, res.ExitCode, res.Stdout, res.Stderr)
	}
	return res
}

func (h *preflightHarness) trackPath(path string) {
	if strings.TrimSpace(path) == "" {
		return
	}
	h.createdPaths = append(h.createdPaths, path)
}

// addCleanupCmd 追加自定义清理命令（cleanup 中逆序执行）
func (h *preflightHarness) addCleanupCmd(cmd string) {
	h.cleanupCmds = append(h.cleanupCmds, cmd)
}

func (h *preflightHarness) createTempDir() string {
	res := h.run("mktemp -d /tmp/preflight-e2e-XXXXXX")
	dir := strings.TrimSpace(res.Stdout)
	h.trackPath(dir)
	return dir
}

func (h *preflightHarness) writeFile(path, content string) {
	cmd := fmt.Sprintf("cat > %s <<'PFEOF'\n%s\nPFEOF", shq(path), content)
	h.run(cmd)
	h.trackPath(path)
}

// copyToTemp 将模板文件复制到临时目录并返回新路径
func (h *preflightHarness) copyToTemp(src, name string) string {
	Expect(src).NotTo(BeEmpty(), "源文件路径不能为空")
	dst := h.tmpDir + "/" + name
	h.run(fmt.Sprintf("cp %s %s", shq(src), shq(dst)))
	h.trackPath(dst)
	return dst
}

func (h *preflightHarness) fileExists(path string) bool {
	res := h.tryRun("test -e " + shq(path))
	return res != nil && res.ExitCode == 0
}

func (h *preflightHarness) nextReportPath() string {
	h.reportSeq++
	return fmt.Sprintf("%s/report-%d.json", h.tmpDir, h.reportSeq)
}

// policyFixturePath 返回外部策略文件来源：env 提供的路径优先（使用方自行校验存在性），
// 否则将 content 写入临时目录；返回值第二个 bool 表示是否使用了包内策略内容
func (h *preflightHarness) policyFixturePath(envPath, name, content string) (string, bool) {
	if envPath != "" {
		return envPath, false
	}
	p := h.tmpDir + "/" + name
	h.writeFile(p, content)
	GinkgoWriter.Printf("使用包内策略文件内容落盘: %s\n", p)
	return p, true
}

// ensureDefaultPolicy 返回本用例应使用的默认外部策略路径：
// 优先 PREFLIGHT_*_POLICY_TEMPLATE env，否则将 preflight.yaml 落盘到远程临时目录。
func (h *preflightHarness) ensureDefaultPolicy() string {
	if h.policyTemplate != "" {
		return h.policyTemplate
	}
	if h.defaultPolicy != "" {
		return h.defaultPolicy
	}
	p, _ := h.policyFixturePath("", "preflight.yaml", preflightPolicyYAML)
	h.defaultPolicy = p
	return p
}

// skipIfPolicyRejected 使用包内策略且 bke 以 INPUT_ERROR 拒绝该策略时 Skip
// （策略与当前 bke 版本不兼容，需在环境变量提供与版本匹配的完整策略文件）；
// 外部提供的策略被拒绝时照常向下执行（由后续断言 Fail），不吞失败
func (h *preflightHarness) skipIfPolicyRejected(o preflightRunOutcome, usingFixture bool) {
	if !usingFixture || o.Report == nil || o.Report.Overall != "INPUT_ERROR" {
		return
	}
	Skip(fmt.Sprintf("包内 preflight.yaml 被当前 bke 版本拒绝（INPUT_ERROR，issues: %s），"+
		"请在 PREFLIGHT_*_POLICY_* 环境变量提供与版本匹配的完整策略文件",
		preflightIssueCodes(o.Report)))
}

// preflightLocalRepo 用例内生成的本机仓库物料
type preflightLocalRepo struct {
	Addr   string // 仓库地址 host:port（--otherRepo）
	CAFile string // 签发本仓库服务端证书的自签 CA（--imageRepoCAFile）
	Port   int    // 监听端口
	PID    string // 监听进程 pid（已登记 trackedPIDs，断言失败时由套件清理兜底）
}

// startLocalRepoListener 在节点一上生成自签 CA 与其签发的服务端证书，并用 openssl s_server
// 起一个 TLS 监听充当真实可达的私有仓库地址（与 --otherRepo/--imageRepoCAFile 配对使用）
func (h *preflightHarness) startLocalRepoListener() preflightLocalRepo {
	dir := h.tmpDir
	caFile := dir + "/pf-e2e-ca.crt"
	caKey := dir + "/pf-e2e-ca.key"
	srvKey := dir + "/pf-e2e-srv.key"
	srvCsr := dir + "/pf-e2e-srv.csr"
	srvCrt := dir + "/pf-e2e-srv.crt"
	sanFile := dir + "/pf-e2e-san.cnf"

	portRes := h.run(`python3 -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()'`)
	port, err := strconv.Atoi(strings.TrimSpace(portRes.Stdout))
	Expect(err).NotTo(HaveOccurred(), "应能获取空闲端口，实际输出: %s", portRes.Stdout)

	h.writeFile(sanFile, "subjectAltName=IP:"+h.hostIP+",DNS:localhost,DNS:127.0.0.1")
	h.run(fmt.Sprintf("openssl req -x509 -newkey rsa:2048 -nodes -keyout %s -out %s -days 2 -subj /CN=pf-e2e-ca",
		shq(caKey), shq(caFile)))
	h.run(fmt.Sprintf("openssl req -newkey rsa:2048 -nodes -keyout %s -out %s -subj /CN=%s",
		shq(srvKey), shq(srvCsr), shq(h.hostIP)))
	h.run(fmt.Sprintf("openssl x509 -req -in %s -CA %s -CAkey %s -CAcreateserial -out %s -days 2 -extfile %s",
		shq(srvCsr), shq(caFile), shq(caKey), shq(srvCrt), shq(sanFile)))

	res := h.tryRun(fmt.Sprintf("nohup openssl s_server -accept %d -cert %s -key %s -www >/dev/null 2>&1 & echo $!",
		port, shq(srvCrt), shq(srvKey)))
	Expect(res).NotTo(BeNil(), "应能启动本机 TLS 仓库监听")
	Expect(res.ExitCode).To(Equal(0), "应能启动本机 TLS 仓库监听: %s", res.Stderr)
	pid := strings.TrimSpace(res.Stdout)
	h.trackedPIDs[pid] = h.ssh
	Expect(waitPortUsedOn(h.ssh, port)).To(BeTrue(), "本机仓库监听端口 %d 应进入监听状态", port)
	return preflightLocalRepo{Addr: fmt.Sprintf("%s:%d", h.hostIP, port), CAFile: caFile, Port: port, PID: pid}
}

// initResultTarget 返回检查项的目标列内容（subject + evidence locator，
// 不同规则的目标信息落点不同）
func initResultTarget(r *preflightResult) string {
	parts := []string{r.Subject}
	for _, e := range r.Evidence {
		parts = append(parts, e.Locator)
	}
	return strings.Join(parts, " ")
}

// runPreflight 执行 preflight 命令并加载 JSON 报告（报告缺失时 Report 为 nil）
func (h *preflightHarness) runPreflight(cmd, reportPath string) preflightRunOutcome {
	GinkgoWriter.Printf("preflight cmd: %s\n", cmd)
	_, _ = h.ssh.Exec("rm -f " + shq(reportPath))
	res, _ := h.ssh.Exec(cmd)
	Expect(res).NotTo(BeNil(), "preflight 命令应返回结果: %s", cmd)
	GinkgoWriter.Printf("preflight exit=%d\nstdout:\n%s\nstderr:\n%s\n", res.ExitCode, res.Stdout, res.Stderr)

	out := preflightRunOutcome{
		Stdout:   res.Stdout,
		Stderr:   res.Stderr,
		ExitCode: res.ExitCode,
	}
	catRes, _ := h.ssh.Exec("cat " + shq(reportPath))
	if catRes != nil && catRes.ExitCode == 0 && strings.TrimSpace(catRes.Stdout) != "" {
		var rpt preflightReport
		if err := json.Unmarshal([]byte(catRes.Stdout), &rpt); err != nil {
			Fail(fmt.Sprintf("JSON 报告解析失败: %v\nraw:\n%s", err, catRes.Stdout))
		}
		out.Report = &rpt
		out.ReportRaw = catRes.Stdout
		GinkgoWriter.Printf("report loaded: overall=%s summary=%+v results=%d\n",
			rpt.Overall, rpt.Summary, len(rpt.Results))
	} else {
		GinkgoWriter.Printf("report file not generated: %s\n", reportPath)
	}
	return out
}

// runInitArgs 执行 bke preflight init 并附加额外参数（extra 为拼接好的参数片段，调用方负责转义）
func (h *preflightHarness) runInitArgs(extra string) preflightRunOutcome {
	reportPath := h.nextReportPath()
	cmd := fmt.Sprintf("%s preflight init --hostIP %s --output json --output-file %s",
		shq(h.bke), shq(h.hostIP), shq(reportPath))
	if extra != "" {
		cmd += " " + extra
	}
	return h.runPreflight(cmd, reportPath)
}

// runInit 执行 bke preflight init；kubernetesPort / policyPath 可为空
func (h *preflightHarness) runInit(kubernetesPort, policyPath string) preflightRunOutcome {
	extra := ""
	if policyPath != "" {
		extra += " --preflight-config " + shq(policyPath)
	}
	if kubernetesPort != "" {
		extra += " --kubernetesPort " + shq(kubernetesPort)
	}
	return h.runInitArgs(extra)
}

// runClusterCreate 执行 bke preflight cluster-create；policyPath 可为空
func (h *preflightHarness) runClusterCreate(kubeconfig, clusterYAML, nodesYAML, policyPath string) preflightRunOutcome {
	reportPath := h.nextReportPath()
	cmd := fmt.Sprintf("%s --kubeconfig %s preflight cluster-create -f %s -n %s --output json --output-file %s",
		shq(h.bke), shq(kubeconfig), shq(clusterYAML), shq(nodesYAML), shq(reportPath))
	if policyPath != "" {
		cmd += " --preflight-config " + shq(policyPath)
	}
	return h.runPreflight(cmd, reportPath)
}

func (h *preflightHarness) cleanup() {
	for pid, exec := range h.trackedPIDs {
		if exec != nil {
			_, _ = exec.Exec(fmt.Sprintf("kill -9 %s 2>/dev/null || true", pid))
		}
	}
	for i := len(h.cleanupCmds) - 1; i >= 0; i-- {
		_, _ = h.ssh.Exec(h.cleanupCmds[i])
	}
	if h.ssh != nil {
		for i := len(h.createdPaths) - 1; i >= 0; i-- {
			p := strings.TrimSpace(h.createdPaths[i])
			if p == "" {
				continue
			}
			_, _ = h.ssh.Exec(fmt.Sprintf("if [ -e %s ]; then rm -rf %s; fi", shq(p), shq(p)))
		}
	}
}

// ---- YAML 模板编辑 ----

// rewriteNthKey 将文件中第 nth 个 "key:" 行的值替换为 value；未匹配返回 false
func (h *preflightHarness) rewriteNthKey(file, key string, nth int, value string) bool {
	return h.rewriteKey(file, key, "", nth, value)
}

// rewriteKey 按 key（可叠加旧值精确匹配）改写 YAML 行，返回是否命中
func (h *preflightHarness) rewriteKey(file, key, oldValue string, nth int, value string) bool {
	script := h.tmpDir + "/rewrite_key.sh"
	body := fmt.Sprintf(`#!/bin/bash
set -euo pipefail
file=%s
awk -v key='%s' -v old='%s' -v nth='%d' -v val='%s' '
  BEGIN { c = 0; done = 0 }
  {
    if (!done && $0 ~ "^[ \t]*" key "[ \t]*:") {
      if (old == "" || $0 ~ "^[ \t]*" key "[ \t]*:[ \t]*" old "([ \t]|$)") {
        c++
        if (c == nth) { sub(/:.*/, ": " val); done = 1 }
      }
    }
    print
  }
  END { if (!done) { exit 3 } }
' "$file" > "$file.pfnew" && mv "$file.pfnew" "$file"
`, shq(file), key, oldValue, nth, value)
	h.writeFile(script, body)
	h.run("chmod +x " + shq(script))
	res := h.tryRun("bash " + shq(script))
	return res != nil && res.ExitCode == 0
}

// appendTopLevelKey 在 YAML 文件末尾追加一个顶层键（用于未知字段注入）
func (h *preflightHarness) appendTopLevelKey(file, key, value string) {
	h.run(fmt.Sprintf("printf '\\n%s: %s\\n' >> %s", key, value, shq(file)))
}

// ---- 端口占用 ----

// occupyPortOn 在目标节点上占用指定 TCP 端口，返回占用进程 PID
func occupyPortOn(exec *executor.SSHExecutor, port int) string {
	cmd := fmt.Sprintf(
		"if command -v python3 >/dev/null 2>&1; then nohup python3 -m http.server %d >/dev/null 2>&1 & else nohup nc -lk -p %d >/dev/null 2>&1 & fi; echo $!",
		port, port)
	res, _ := exec.Exec(cmd)
	Expect(res).NotTo(BeNil(), "端口占用命令应返回结果")
	Expect(res.ExitCode).To(Equal(0), "端口占用命令应成功")
	pid := strings.TrimSpace(res.Stdout)
	Expect(pid).NotTo(BeEmpty(), "应返回占用进程 PID")
	GinkgoWriter.Printf("port %d occupied by pid %s\n", port, pid)
	return pid
}

// waitPortUsedOn 等待端口进入监听状态
func waitPortUsedOn(exec *executor.SSHExecutor, port int) bool {
	check := fmt.Sprintf("(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':%d '", port)
	for i := 0; i < 10; i++ {
		if res, _ := exec.Exec(check); res != nil && res.ExitCode == 0 {
			return true
		}
		time.Sleep(time.Second)
	}
	return false
}

// waitPortFreeOn 等待端口释放
func waitPortFreeOn(exec *executor.SSHExecutor, port int) bool {
	check := fmt.Sprintf("(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -q ':%d '", port)
	for i := 0; i < 15; i++ {
		if res, _ := exec.Exec(check); res != nil && res.ExitCode != 0 {
			return true
		}
		time.Sleep(time.Second)
	}
	return false
}

func killPIDOn(exec *executor.SSHExecutor, pid string) {
	_, _ = exec.Exec(fmt.Sprintf("kill -9 %s 2>/dev/null || true", pid))
}

// ---- 节点事实交叉核对 ----

// testNodeHostnames 收集 .env 中配置的测试节点 hostname
func testNodeHostnames() map[int]string {
	out := map[int]string{}
	for i := 1; i <= 5; i++ {
		if hn := strings.TrimSpace(os.Getenv(fmt.Sprintf("TEST_NODE%d_HOSTNAME", i))); hn != "" {
			out[i] = hn
		}
	}
	return out
}

// crossCheckNodeFacts 用 SSH 采集的 nproc/free -b/df -B1 交叉核对报告中的资源实际值
func crossCheckNodeFacts(report *preflightReport) {
	if report == nil {
		return
	}
	hostnames := testNodeHostnames()
	for idx, hostname := range hostnames {
		var matched []*preflightResult
		for i := range report.Results {
			if instanceSuffix(report.Results[i]) == hostname {
				matched = append(matched, &report.Results[i])
			}
		}
		if len(matched) == 0 {
			continue
		}
		var nodeExec *executor.SSHExecutor
		func() {
			defer func() {
				if r := recover(); r != nil {
					GinkgoWriter.Printf("skip cross-check on node %s: %v\n", hostname, r)
				}
			}()
			nodeExec = newNodeExecutor(idx)
		}()
		if nodeExec == nil {
			continue
		}
		defer nodeExec.Close()

		checkIntFact := func(ruleSubstr, factCmd string, tolerance float64) {
			candidates := filterResults(report, ruleSubstr)
			var hit *preflightResult
			for i := range candidates {
				if instanceSuffix(candidates[i]) == hostname {
					hit = &candidates[i]
					break
				}
			}
			if hit == nil {
				return
			}
			factRes, err := nodeExec.Exec(factCmd)
			if err != nil || factRes.ExitCode != 0 {
				GinkgoWriter.Printf("fact cmd failed on %s: %s\n", hostname, factCmd)
				return
			}
			fact, ok := rawValueInt64([]byte(strings.TrimSpace(factRes.Stdout)))
			if !ok {
				return
			}
			actual, ok := rawValueInt64(hit.Actual.Value)
			if !ok {
				GinkgoWriter.Printf("rule %s actual value not numeric: %s\n",
					hit.InstanceID, rawValueString(hit.Actual.Value))
				return
			}
			if tolerance == 0 {
				Expect(actual).To(Equal(fact), "规则 %s 实际值应与 %s 输出一致", hit.InstanceID, factCmd)
				return
			}
			diff := actual - fact
			if diff < 0 {
				diff = -diff
			}
			Expect(float64(diff) <= tolerance*float64(fact)).To(BeTrue(),
				"规则 %s 实际值 %d 应接近 %s 输出 %d（容差 %.0f%%）",
				hit.InstanceID, actual, factCmd, fact, tolerance*100)
		}

		checkIntFact("RES-CPU", "nproc", 0)
		checkIntFact("RES-MEM", "free -b | awk '/^Mem:/{print $2}'", 0.01)
		checkIntFact("RES-DISK", "df -B1 / | awk 'NR==2{print $2}'", 0.01)
		return // 只核对第一个能对上的节点
	}
	GinkgoWriter.Printf("no matched node for resource cross-check\n")
}

// preflightSuiteCleanupFns 收集各 Describe 的套件级清理函数。
// 用例之间无顺序依赖（不使用 Ordered），套件级资源懒初始化，统一由 AfterSuite 清理。
var preflightSuiteCleanupFns []func()

func registerPreflightSuiteCleanup(fn func()) {
	preflightSuiteCleanupFns = append(preflightSuiteCleanupFns, fn)
}

func runPreflightSuiteCleanup() {
	for i := len(preflightSuiteCleanupFns) - 1; i >= 0; i-- {
		preflightSuiteCleanupFns[i]()
	}
}
