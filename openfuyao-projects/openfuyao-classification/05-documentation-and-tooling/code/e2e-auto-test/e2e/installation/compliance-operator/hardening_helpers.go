package compliance_operator

import (
	"fmt"
	"net"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
	config "gitcode.com/openFuyao/e2e-auto-test/e2e/installation/bke-config"
	"gitcode.com/openFuyao/e2e-auto-test/e2e/installation/compliance-operator/utils"
	. "github.com/onsi/ginkgo/v2"
	. "github.com/onsi/gomega"
)

// 套件级共享状态（声明于非 _test 文件，供包内各用例文件与 helper 共同使用）。
var (
	cfg       *utils.Config
	entry     *executor.SSHExecutor // entry_node（编排节点，工具部署处）
	nodeExecs map[string]*executor.SSHExecutor
	shared    = &suiteState{}
)

// suiteState 跨阶段共享状态（全链路 Ordered 容器内串行传递，无并发访问）。
type suiteState struct {
	// s0Fingerprints 未加固态（S0）全节点指纹（E2E-B 采集并持久化到 entry_node，供 E2E-C 回退后对比）
	s0Fingerprints utils.FingerprintSet
	// hardenedFingerprints 加固态全节点指纹（E2E-A 末采集，供 DFX-02/E2E-C 对比）
	hardenedFingerprints utils.FingerprintSet
	// unifiedTS E2E-A 产出的统一备份时间戳 TS1（同时持久化到 entry_node 状态文件）
	unifiedTS string
	// cisBaseline / stigBaseline S0 双基线扫描结果（E2E-A BeforeAll 采集；镜像缺失时为 nil）
	cisBaseline  *utils.CISResult
	stigBaseline *utils.STIGResult
	// scanSkipReason 复扫不可用原因（降级记录）
	scanSkipReason string
}

// collectAllFingerprints 采集全节点指纹（失败即 Fatal 于当前 It）。
func collectAllFingerprints() utils.FingerprintSet {
	By("采集全节点配置指纹（md5+权限+属主）")
	set, err := utils.CollectAllFingerprints(cfg, nodeExecs)
	Expect(err).NotTo(HaveOccurred(), "全节点指纹采集应成功")
	return set
}

// requireUnhardened 前置态校验：集群处于未加固态（S0）。
// 判据：各 master apiserver manifest 权限非 600，且无 .harden-origin 加固起点标记残留。
func requireUnhardened(reason string) {
	By(fmt.Sprintf("校验集群处于未加固态（%s）", reason))
	for _, n := range cfg.Masters() {
		e := nodeExecs[n.IP]
		perm, err := utils.FilePerm(e, utils.ManifestAPIServer)
		Expect(err).NotTo(HaveOccurred(), "读取 apiserver manifest 权限应成功: "+n.IP)
		Expect(perm).NotTo(Equal("600"),
			fmt.Sprintf("节点 %s 已处于加固态（apiserver 权限 600），不满足 %s 前置；请先执行 cluster-rollback 统一回退恢复 S0", n.IP, reason))
		Expect(utils.HardenedOriginMarkerExists(e, cfg.BackupDir)).To(BeFalse(),
			fmt.Sprintf("节点 %s 存在加固起点标记（%s/%s），说明上一轮加固未回退；请先回退再执行 %s",
				n.IP, cfg.BackupDir, utils.HardenOriginMarker, reason))
	}
}

// requireHardened 前置态校验：集群处于加固态（各 master apiserver manifest 权限 600）。
func requireHardened(reason string) {
	By(fmt.Sprintf("校验集群处于加固态（%s）", reason))
	for _, n := range cfg.Masters() {
		perm, err := utils.FilePerm(nodeExecs[n.IP], utils.ManifestAPIServer)
		Expect(err).NotTo(HaveOccurred(), "读取 apiserver manifest 权限应成功: "+n.IP)
		Expect(perm).To(Equal("600"),
			fmt.Sprintf("节点 %s 不处于加固态（apiserver 权限 %s，期望 600），不满足 %s 前置；请先执行 E2E-A hardening_main_path", n.IP, perm, reason))
	}
}

type kubeletAPIConfig struct {
	Kubeconfig string
	Context    string
	Server     string
}

// detectKubeletAPIConfig follows the kubeconfig used by the running process.
// The service may use a path other than /etc/kubernetes/kubelet.conf, so a
// fixed path can report an endpoint that kubelet never reads.
func detectKubeletAPIConfig(e *executor.SSHExecutor) (kubeletAPIConfig, error) {
	out, err := utils.ExecTrim(e, `
pid=$(pidof kubelet 2>/dev/null | awk '{print $1}')
test -n "$pid" || { echo "kubelet process not found" >&2; exit 1; }
args=$(tr '\000' '\n' < "/proc/$pid/cmdline")
kubeconfig=$(printf '%s\n' "$args" | sed -n 's/^--kubeconfig=//p' | head -1)
if [ -z "$kubeconfig" ]; then
  kubeconfig=$(printf '%s\n' "$args" | awk '$0 == "--kubeconfig" {getline; print; exit}')
fi
test -n "$kubeconfig" || { echo "--kubeconfig is absent from kubelet cmdline" >&2; exit 1; }
test -r "$kubeconfig" || { echo "kubelet kubeconfig is unreadable: $kubeconfig" >&2; exit 1; }
context=$(kubectl --kubeconfig="$kubeconfig" config current-context 2>/dev/null) || exit 1
server=$(kubectl --kubeconfig="$kubeconfig" config view --minify -o jsonpath='{.clusters[0].cluster.server}' 2>/dev/null) || exit 1
test -n "$server" || { echo "active API server is empty in $kubeconfig" >&2; exit 1; }
printf '%s\n%s\n%s\n' "$kubeconfig" "$context" "$server"
`)
	if err != nil {
		return kubeletAPIConfig{}, err
	}
	parts := strings.Split(strings.TrimSpace(out), "\n")
	if len(parts) != 3 {
		return kubeletAPIConfig{}, fmt.Errorf("unexpected kubelet API config output: %q", out)
	}
	return kubeletAPIConfig{Kubeconfig: parts[0], Context: parts[1], Server: parts[2]}, nil
}

func resolveEndpointIPs(e *executor.SSHExecutor, host string) ([]string, error) {
	if net.ParseIP(host) != nil {
		return []string{host}, nil
	}
	out, err := utils.ExecTrim(e, fmt.Sprintf(
		`getent ahostsv4 %s | awk '{print $1}' | sort -u`, utils.ShellPath(host)))
	if err != nil {
		return nil, fmt.Errorf("resolve %s on worker: %w", host, err)
	}
	ips := nonEmptyLines(out)
	if len(ips) == 0 {
		return nil, fmt.Errorf("resolve %s on worker: no IPv4 address returned", host)
	}
	return ips, nil
}

// requireWorkerKubeletHAEndpoint enforces the FIT-08 environment prerequisite
// before hardening changes cluster state. BKE may write the HA domain (for
// example master.bocloud.com) instead of the literal VIP into kubelet.conf, so
// domain endpoints are resolved on the worker where kubelet uses them.
func requireWorkerKubeletHAEndpoint() {
	By("校验 worker kubelet 实际使用高可用 API endpoint（FIT-08 前置）")
	vipHost, vipPort := config.LoadWorkloadClusterVIPFromEnv()
	Expect(vipHost).NotTo(BeEmpty(), "WORKLOAD_CLUSTER_VIP_HOST 必须配置")
	Expect(vipPort).NotTo(BeEmpty(), "WORKLOAD_CLUSTER_VIP_PORT 必须配置")
	masterIPs := make([]string, 0, len(cfg.Masters()))
	for _, master := range cfg.Masters() {
		masterIPs = append(masterIPs, master.IP)
	}
	for _, worker := range cfg.Workers() {
		actual, err := detectKubeletAPIConfig(nodeExecs[worker.IP])
		Expect(err).NotTo(HaveOccurred(), "动态读取 worker kubelet 的 --kubeconfig 和活动 API server 应成功: "+worker.IP)
		parsed, err := url.Parse(actual.Server)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("worker %s 的 kubelet API server 无效: %s", worker.IP, actual.Server))
		Expect(parsed.Hostname()).NotTo(BeEmpty(), fmt.Sprintf("worker %s 的 kubelet API server 缺少主机名: %s", worker.IP, actual.Server))
		Expect(parsed.Port()).To(Equal(vipPort), fmt.Sprintf(
			"worker %s 的 kubelet VIP 端口不匹配：kubeconfig=%s context=%s server=%s",
			worker.IP, actual.Kubeconfig, actual.Context, actual.Server))

		resolvedIPs, err := resolveEndpointIPs(nodeExecs[worker.IP], parsed.Hostname())
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf(
			"worker %s 应能解析 kubelet API endpoint：kubeconfig=%s context=%s server=%s",
			worker.IP, actual.Kubeconfig, actual.Context, actual.Server))
		Expect(resolvedIPs).To(ContainElement(vipHost), fmt.Sprintf(
			"worker %s 的 kubelet API endpoint 必须解析到 workload VIP %s：server=%s resolved=%v",
			worker.IP, vipHost, actual.Server, resolvedIPs))
		for _, masterIP := range masterIPs {
			Expect(resolvedIPs).NotTo(ContainElement(masterIP), fmt.Sprintf(
				"worker %s 的 kubelet API endpoint 不得解析到 master %s：server=%s resolved=%v",
				worker.IP, masterIP, actual.Server, resolvedIPs))
		}
		GinkgoWriter.Printf("worker kubelet API node=%s kubeconfig=%s context=%s server=%s resolved=%v\n",
			worker.IP, actual.Kubeconfig, actual.Context, actual.Server, resolvedIPs)
	}
}

// assertAllNodesReady 断言 kubectl get nodes 全部 Ready。
func assertAllNodesReady() {
	By("执行 kubectl get nodes 校验全节点 Ready")
	out, err := utils.ExecTrim(entry, `kubectl get nodes --no-headers | awk '{print $1, $2}'`)
	Expect(err).NotTo(HaveOccurred(), "kubectl get nodes 应成功（entry_node 需具备 kubectl）")
	lines := nonEmptyLines(out)
	Expect(lines).To(HaveLen(len(cfg.Nodes)), "集群节点数应与 .env 配置一致")
	for _, l := range lines {
		fields := strings.Fields(l)
		Expect(fields).To(HaveLen(2))
		Expect(fields[1]).To(Equal("Ready"), fmt.Sprintf("节点 %s 应 Ready（实际 %s）", fields[0], fields[1]))
	}
}

// ---- g1-g6 逐组合规断言（与 config/hardening-rules.yaml 期望值逐字对应）----

// manifestFlagValue 提取 manifest 中 --flag 的值（支持 --flag=value 与 --flag value 两种形态）。
func manifestFlagValue(e *executor.SSHExecutor, manifest, flag string) (string, bool) {
	cmd := fmt.Sprintf(`grep -oE -- "--%s(=[^\"[:space:]]+|[[:space:]]+[^\"[:space:]]+)" %s | head -1`,
		regexp.QuoteMeta(flag), utils.ShellPath(manifest))
	res := utils.Run(e, cmd)
	if res.ExitCode != 0 || strings.TrimSpace(res.Stdout) == "" {
		return "", false
	}
	v := strings.TrimSpace(res.Stdout)
	v = strings.TrimPrefix(v, "--"+flag)
	v = strings.TrimSpace(strings.TrimPrefix(v, "="))
	return v, true
}

// expectManifestFlag 断言 manifest 中 --flag 存在且值符合期望（exact 或 contains）。
func expectManifestFlag(e *executor.SSHExecutor, manifest, flag, want string, contains bool, ctx string) {
	v, ok := manifestFlagValue(e, manifest, flag)
	Expect(ok).To(BeTrue(), fmt.Sprintf("%s：manifest %s 应含 --%s", ctx, manifest, flag))
	if contains {
		Expect(v).To(ContainSubstring(want), fmt.Sprintf("%s：--%s 应含 %s（实际 %s）", ctx, flag, want, v))
	} else {
		Expect(v).To(Equal(want), fmt.Sprintf("%s：--%s 应为 %s（实际 %s）", ctx, flag, want, v))
	}
}

// expectFilePerm 断言远端文件权限。
func expectFilePerm(e *executor.SSHExecutor, path, want, ctx string) {
	perm, err := utils.FilePerm(e, path)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("%s：stat %s 应成功", ctx, path))
	Expect(perm).To(Equal(want), fmt.Sprintf("%s：%s 权限应为 %s（实际 %s）", ctx, path, want, perm))
}

func expectFileOwner(e *executor.SSHExecutor, path, want, ctx string) {
	owner, err := utils.FileOwner(e, path)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("%s：stat %s 应成功", ctx, path))
	Expect(owner).To(Equal(want), fmt.Sprintf("%s：%s 属主应为 %s（实际 %s）", ctx, path, want, owner))
}

// kubeletFieldValue 使用 yq 读取 kubelet YAML 字段；字段不存在或值为 null 时返回空字符串。
func kubeletFieldValue(e *executor.SSHExecutor, configPath, field string) (string, error) {
	return utils.ExecTrim(e, fmt.Sprintf(`yq eval -r '.%s // ""' %s`, field, utils.ShellPath(configPath)))
}

// assertGroup1ManifestPerms g1 静态 Pod manifest 权限 600（仅 master）。
func assertGroup1ManifestPerms(n utils.NodeConfig) {
	By(fmt.Sprintf("g1 manifest 权限断言（%s）", n.IP))
	e := nodeExecs[n.IP]
	for _, m := range []string{utils.ManifestAPIServer, utils.ManifestController, utils.ManifestScheduler, utils.ManifestEtcd} {
		expectFilePerm(e, m, "600", fmt.Sprintf("g1(%s)", n.IP))
	}
}

// assertGroup2APIServer g2 kube-apiserver 参数 + audit-policy 含 rules（仅 master）。
func assertGroup2APIServer(n utils.NodeConfig) {
	By(fmt.Sprintf("g2 apiserver 参数断言（%s）", n.IP))
	e := nodeExecs[n.IP]
	ctx := fmt.Sprintf("g2(%s)", n.IP)
	expectManifestFlag(e, utils.ManifestAPIServer, "kubelet-certificate-authority", "/etc/kubernetes/pki/ca.crt", false, ctx)
	expectManifestFlag(e, utils.ManifestAPIServer, "enable-admission-plugins", "NodeRestriction", true, ctx)
	expectManifestFlag(e, utils.ManifestAPIServer, "service-account-extend-token-expiration", "false", false, ctx)
	expectManifestFlag(e, utils.ManifestAPIServer, "tls-min-version", "VersionTLS12", false, ctx)
	expectManifestFlag(e, utils.ManifestAPIServer, "tls-cipher-suites", "TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256", true, ctx)
	expectManifestFlag(e, utils.ManifestAPIServer, "request-timeout", "1m", false, ctx)

	out, err := utils.ExecTrim(e, fmt.Sprintf("grep -E '^rules:' %s | wc -l", utils.ShellPath(utils.AuditPolicy)))
	Expect(err).NotTo(HaveOccurred(), ctx+"：读取 audit-policy 应成功")
	Expect(strings.TrimSpace(out)).To(Equal("1"), ctx+"：audit-policy.yaml 应含 rules:")
}

// assertGroup34ControllerScheduler g3/g4 controller-manager 与 scheduler 参数（仅 master）。
func assertGroup34ControllerScheduler(n utils.NodeConfig) {
	By(fmt.Sprintf("g3/g4 controller-manager 与 scheduler 参数断言（%s）", n.IP))
	e := nodeExecs[n.IP]
	expectManifestFlag(e, utils.ManifestController, "bind-address", "127.0.0.1", false, fmt.Sprintf("g3(%s)", n.IP))
	expectManifestFlag(e, utils.ManifestController, "tls-min-version", "VersionTLS12", false, fmt.Sprintf("g3(%s)", n.IP))
	// sync_probe：bind-address 改造同步修改 livenessProbe.httpGet.host
	out, err := utils.ExecTrim(e, fmt.Sprintf("grep -cE 'host:[[:space:]]*127\\.0\\.0\\.1' %s", utils.ShellPath(utils.ManifestController)))
	Expect(err).NotTo(HaveOccurred())
	Expect(strings.TrimSpace(out)).NotTo(Equal("0"), fmt.Sprintf("g3(%s)：controller-manager livenessProbe host 应为 127.0.0.1", n.IP))

	expectManifestFlag(e, utils.ManifestScheduler, "bind-address", "127.0.0.1", false, fmt.Sprintf("g4(%s)", n.IP))
	expectManifestFlag(e, utils.ManifestScheduler, "tls-min-version", "VersionTLS12", false, fmt.Sprintf("g4(%s)", n.IP))
	out, err = utils.ExecTrim(e, fmt.Sprintf("grep -cE 'host:[[:space:]]*127\\.0\\.0\\.1' %s", utils.ShellPath(utils.ManifestScheduler)))
	Expect(err).NotTo(HaveOccurred())
	Expect(strings.TrimSpace(out)).NotTo(Equal("0"), fmt.Sprintf("g4(%s)：scheduler livenessProbe host 应为 127.0.0.1", n.IP))
}

// assertGroup5Etcd g5 etcd 数据目录与参数（仅 master）。
func assertGroup5Etcd(n utils.NodeConfig) {
	By(fmt.Sprintf("g5 etcd 断言（%s）", n.IP))
	e := nodeExecs[n.IP]
	ctx := fmt.Sprintf("g5(%s)", n.IP)
	etcdDataDir, err := utils.DetectEtcdDataDir(e)
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("%s: detect etcd data-dir from %s", n.IP, utils.ManifestEtcd))
	expectFilePerm(e, etcdDataDir, "700", ctx)
	// 数据目录内文件权限 600（stig-sv242459：STIG 要求 644 或更严格，工具取 600）
	out, err := utils.ExecTrim(e, fmt.Sprintf(
		"find %s -maxdepth 1 -type f ! -perm 600 2>/dev/null | wc -l", utils.ShellPath(etcdDataDir)))
	Expect(err).NotTo(HaveOccurred(), ctx+"：读取 etcd 数据目录文件权限应成功")
	Expect(strings.TrimSpace(out)).To(Equal("0"), ctx+"：etcd 数据目录内文件权限应全部 600")

	expectManifestFlag(e, utils.ManifestEtcd, "auto-tls", "false", false, ctx)
	expectManifestFlag(e, utils.ManifestEtcd, "peer-auto-tls", "false", false, ctx)
}

// assertGroup6Kubelet g6 kubelet 配置与文件权限（master/worker 均断言，worker 仅此组）。
func assertGroup6Kubelet(n utils.NodeConfig) {
	By(fmt.Sprintf("g6 kubelet 断言（%s）", n.IP))
	e := nodeExecs[n.IP]
	ctx := fmt.Sprintf("g6(%s)", n.IP)
	expectFileOwner(e, cfg.KubeletConfig, "root:root", ctx)

	for field, want := range map[string]string{
		"authentication.x509.clientCAFile": "/etc/kubernetes/pki/ca.crt",
		"protectKernelDefaults":            "true",
		"streamingConnectionIdleTimeout":   "5m",
	} {
		actual, err := kubeletFieldValue(e, cfg.KubeletConfig, field)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("%s：读取 kubelet config 字段 %s 应成功", ctx, field))
		Expect(actual).To(Equal(want), fmt.Sprintf("%s：%s 应为 %s（实际 %s）", ctx, field, want, actual))
	}

	// 证书与私钥路径由加固工具在运行时检测，可能是默认路径，也可能是集群已有的
	// 自定义路径；这里不硬编码路径，但要求字段值非空且指向的文件真实存在。
	for _, field := range []string{"tlsCertFile", "tlsPrivateKeyFile"} {
		actual, err := kubeletFieldValue(e, cfg.KubeletConfig, field)
		Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("%s：读取 kubelet config 字段 %s 应成功", ctx, field))
		Expect(actual).NotTo(BeEmpty(), fmt.Sprintf("%s：%s 应设置为非空的运行时路径", ctx, field))
		Expect(utils.FileExists(e, actual)).To(BeTrue(), fmt.Sprintf("%s：%s 指向的文件应存在（实际 %s）", ctx, field, actual))
	}
	expectFilePerm(e, utils.CACert, "644", ctx)
	expectFileOwner(e, utils.CACert, "root:root", ctx)
}

// ---- 业务负载与系统 Pod 快照 ----

// PodSnapshot 业务 Pod 快照（name/uid/node，UID 对比口径与 FIT 自验手册 0.5 一致）。
type PodSnapshot struct {
	Name string
	UID  string
	Node string
}

// podSnapshotViaKubectl 按 selector 采集业务 Pod 快照（name/uid/node，UID 对比口径与
// FIT 自验手册 0.5 一致）。selector 传 "-l k1=v1,k2=v2" 形式（业务快照统一经
// deploymentPodSnapshots 以 Deployment 实际 selector 圈定范围，勿用空 selector 列全 ns）。
// 输出第 4 字段为 deletionTimestamp：非空＝Pod 正在删除（Terminating），不入快照与 UID 对比，
// 避免滚动期瞬态 Pod 混入基线（run4 的 len(before)=4 vs readyReplicas=2 即此因）。
func podSnapshotViaKubectl(ns, selector string) []PodSnapshot {
	out, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n %s get pods %s -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.uid}{" "}{.spec.nodeName}{" "}{.metadata.deletionTimestamp}{"\n"}{end}'`,
		ns, selector))
	Expect(err).NotTo(HaveOccurred(), "查询 Pod 快照应成功: ns="+ns)
	var out2 []PodSnapshot
	for _, l := range nonEmptyLines(out) {
		f := strings.Fields(l)
		if len(f) >= 4 && f[3] != "" {
			continue
		}
		if len(f) >= 3 {
			out2 = append(out2, PodSnapshot{Name: f[0], UID: f[1], Node: f[2]})
		}
	}
	return out2
}

// deploymentPodSelector 返回业务 Deployment 的实际 Pod selector（-l 参数形式，取
// .spec.selector.matchLabels，以 Deployment 实际 selector 为准，勿假设 app=<dep>）。
// 解析用 jq（entry_node 既定依赖）而非 jsonpath：jsonpath 的 map 迭代 {@.key}={@.value}
// 对字符串值取出的 key/value 恒为空（run5 实测输出恒为 "="，-l = 被 API 以 BadRequest 拒绝），
// 两变量 range（{range $k, $v := ...}）在本 kubectl 亦不被支持（unrecognized character U+002C）。
// 解析结果整体校验「非空k=非空v」逗号分隔形态，不合法即失败并附原始输出，杜绝 "=" 静默下传。
func deploymentPodSelector(ns, dep string) string {
	out, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n %s get deploy %s -o json | jq -r '.spec.selector.matchLabels | to_entries | map("\(.key)=\(.value)") | join(",")'`,
		ns, dep))
	Expect(err).NotTo(HaveOccurred(), "查询业务 Deployment selector 应成功: "+dep)
	Expect(out).To(MatchRegexp(`^[^=,\s]+=[^=,\s]+(,[^=,\s]+=[^=,\s]+)*$`),
		fmt.Sprintf("业务 Deployment %s 的 selector 解析结果应为「非空k=非空v」逗号分隔形态（原始输出 %q）", dep, out))
	return out
}

// waitForSteadyPods 秒级短轮询等待快照范围收敛到稳态（存活 Pod 数 == spec.replicas 且无
// Terminating）——rollout status 返回时旧 RS 的 Pod 可能仍在 Terminating，立即采集会把瞬态
// Pod 计入基线。
func waitForSteadyPods(ns, dep, sel string) {
	repOut, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n %s get deploy %s -o jsonpath='{.spec.replicas}'`, ns, dep))
	Expect(err).NotTo(HaveOccurred(), "查询业务 Deployment 副本数应成功")
	var replicas int
	Expect(fmt.Sscanf(strings.TrimSpace(repOut), "%d", &replicas)).NotTo(BeZero(),
		fmt.Sprintf("业务 Deployment spec.replicas 应为数字（实际 %q）", repOut))
	deadline := time.Now().Add(30 * time.Second)
	for {
		out, err := utils.ExecTrim(entry, fmt.Sprintf(
			`kubectl -n %s get pods -l %s -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.metadata.deletionTimestamp}{"\n"}{end}'`, ns, sel))
		Expect(err).NotTo(HaveOccurred(), "查询业务 Pod 稳态应成功")
		total, terminating := 0, 0
		for _, l := range nonEmptyLines(out) {
			total++
			if len(strings.Fields(l)) >= 2 {
				terminating++
			}
		}
		if total-terminating == replicas {
			return
		}
		if time.Now().After(deadline) {
			Expect(total-terminating).To(Equal(replicas),
				fmt.Sprintf("业务 Pod 应在 30s 内收敛到稳态（存活 %d 台且无 Terminating，期望 %d；Terminating %d 台）",
					total-terminating, replicas, terminating))
		}
		time.Sleep(2 * time.Second)
	}
}

// deploymentPodSnapshots 业务 Deployment 的稳态 Pod 快照：以 Deployment 实际 selector 圈定
// 采集范围，短轮询等滚动收敛后采集；setup 基线与加固后对比共用本实现，两侧口径一致。
func deploymentPodSnapshots(ns, dep string) []PodSnapshot {
	sel := deploymentPodSelector(ns, dep)
	waitForSteadyPods(ns, dep, sel)
	return podSnapshotViaKubectl(ns, "-l "+sel)
}

var k8sNodeIPMap map[string]string // 集群「节点名 → InternalIP」映射（一次查询缓存）

// k8sNodeIPMapping 返回集群「节点名 → InternalIP」映射（一次查询并缓存）。
// Pod 快照的 Node 字段是 .spec.nodeName（k8s 节点名，如 m1/n1），而 .env 节点清单以 IP
// 为键，角色判定前须经此映射转换；节点名本身即 IP 的集群映射值等于自身，行为不变。
func k8sNodeIPMapping() map[string]string {
	if k8sNodeIPMap != nil {
		return k8sNodeIPMap
	}
	out, err := utils.ExecTrim(entry, `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{" "}{.status.addresses[?(@.type=="InternalIP")].address}{"\n"}{end}'`)
	Expect(err).NotTo(HaveOccurred(), "查询集群节点名→IP 映射应成功")
	m := map[string]string{}
	for _, l := range nonEmptyLines(out) {
		f := strings.Fields(l)
		if len(f) >= 2 { // 节点名 + InternalIP（多 IP 时取第一个）
			m[f[0]] = f[1]
		}
	}
	k8sNodeIPMap = m
	return k8sNodeIPMap
}

// nodeRole 返回节点角色：入参为 Pod 快照的 .spec.nodeName（k8s 节点名），先经集群实测
// 「节点名→IP」映射解析为 IP（节点名即 IP 时解析结果不变），再查 .env 节点清单（IP 为键）。
func nodeRole(node string) string {
	ip := node
	if resolved, ok := k8sNodeIPMapping()[node]; ok {
		ip = resolved
	}
	for _, n := range cfg.Nodes {
		if n.IP == ip {
			return n.Role
		}
	}
	return "unknown"
}

// workerNodeNames 返回 .env worker 节点对应的 k8s 节点名：复用 k8sNodeIPMapping 的同一
// 映射源反查（角色判定唯一来源仍是 .env 清单），供业务负载 worker 节点亲和使用。
func workerNodeNames() []string {
	byName := k8sNodeIPMapping()
	var names []string
	for _, w := range cfg.Workers() {
		for name, ip := range byName {
			if ip == w.IP {
				names = append(names, name)
				break
			}
		}
	}
	sort.Strings(names)
	return names
}

// assertBusinessWorkloadStable 加固后业务验证（结果8）：
// 副本终态一致且无 CrashLoopBackOff；worker 节点 Pod UID 不变；master 节点 UID 允许变化
// （apiserver 重建窗口内驱逐重建属 manifest 级加固固有代价，非工具缺陷）；API 读写链路可用。
func assertBusinessWorkloadStable(before []PodSnapshot) {
	ns := cfg.BizNamespace
	dep := cfg.BizDeployment

	By("校验 Deployment 终态副本与可用性")
	ready, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n %s get deploy %s -o jsonpath='{.status.readyReplicas}'`, ns, dep))
	Expect(err).NotTo(HaveOccurred(), "查询 Deployment 终态应成功")
	Expect(strings.TrimSpace(ready)).To(Equal(fmt.Sprintf("%d", len(before))),
		"Deployment Ready 副本数应与加固前一致")

	current := deploymentPodSnapshots(ns, dep)
	By("校验 worker 节点业务 Pod UID 不变、master 节点无 CrashLoopBackOff")
	beforeWorker := filterSnapshots(before, utils.RoleWorker)
	afterWorker := filterSnapshots(current, utils.RoleWorker)
	beforeWorkerUIDs := podUIDMap(beforeWorker)
	afterWorkerUIDs := podUIDMap(afterWorker)
	diagnostic := "worker 节点业务 Pod UID 应不变（kubelet 重启不应驱逐业务 Pod）"
	if !stringMapEqual(afterWorkerUIDs, beforeWorkerUIDs) {
		diagnostic = workerPodReplacementDiagnostics(beforeWorker, afterWorker)
	}
	Expect(afterWorkerUIDs).To(Equal(beforeWorkerUIDs), diagnostic)
	Expect(len(afterWorker)).To(BeNumerically(">", 0), "worker 节点应有业务 Pod")
	for _, snap := range current {
		phase, err := utils.ExecTrim(entry, fmt.Sprintf(
			`kubectl -n %s get pod %s -o jsonpath='{.status.phase}{" "}{range .status.containerStatuses[*]}{.state.waiting.reason}{end}'`, ns, snap.Name))
		Expect(err).NotTo(HaveOccurred())
		Expect(phase).To(ContainSubstring("Running"), fmt.Sprintf("业务 Pod %s 应 Running", snap.Name))
		Expect(phase).NotTo(ContainSubstring("CrashLoopBackOff"), fmt.Sprintf("业务 Pod %s 不应 CrashLoopBackOff", snap.Name))
	}

	By("校验 ConfigMap 读写删链路（API 可用性）")
	cm := utils.RunID()
	_, err = utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s create configmap %s --from-literal=e2e=k8s`, ns, cm))
	Expect(err).NotTo(HaveOccurred(), "创建 ConfigMap 应成功")
	DeferCleanup(func() {
		_, _ = utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s delete configmap %s --ignore-not-found=true`, ns, cm))
	})
	val, err := utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s get cm %s -o jsonpath='{.data.e2e}'`, ns, cm))
	Expect(err).NotTo(HaveOccurred(), "读取 ConfigMap 应成功")
	Expect(strings.TrimSpace(val)).To(Equal("k8s"), "ConfigMap 键值应可读回")
	_, err = utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s delete configmap %s`, ns, cm))
	Expect(err).NotTo(HaveOccurred(), "删除 ConfigMap 应成功")
}

// workerPodReplacementDiagnostics 在 FIT-08 UID 断言失败时补充只读诊断证据。
// worker kubelet 若直连某台 master 的 apiserver，该 master 的 manifest 重建可能让
// worker 进入 NotReady 并触发 TaintManagerEviction；报告 endpoint 和近期事件可区分
// 真实驱逐、Deployment 滚动与 E2E 快照问题。该 helper 不改变判定结果。
func workerPodReplacementDiagnostics(before, after []PodSnapshot) string {
	endpoints := map[string]string{}
	for _, worker := range cfg.Workers() {
		actual, err := detectKubeletAPIConfig(nodeExecs[worker.IP])
		if err != nil {
			endpoints[worker.IP] = fmt.Sprintf("<unavailable: %v>", err)
		} else {
			endpoints[worker.IP] = fmt.Sprintf("kubeconfig=%s context=%s server=%s",
				actual.Kubeconfig, actual.Context, actual.Server)
		}
	}
	events, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n %s get events --sort-by=.metadata.creationTimestamp --no-headers 2>/dev/null | grep -E 'NodeNotReady|TaintManagerEviction|SuccessfulCreate' | tail -20`,
		cfg.BizNamespace))
	if err != nil || strings.TrimSpace(events) == "" {
		events = fmt.Sprintf("<unavailable: %v>", err)
	}
	return fmt.Sprintf(
		"worker 节点业务 Pod UID 应不变（kubelet 重启不应驱逐业务 Pod）；before=%v after=%v worker-kubelet-api=%v recent-events:\n%s",
		before, after, endpoints, events)
}

// workerAffinityPatch 生成业务 Deployment 的 worker 节点亲和 strategic-merge patch
// （JSON 单行，nodeSelectorTerms 无合并键、整体替换为目标亲和）。
func workerAffinityPatch(workers []string) string {
	vals := make([]string, len(workers))
	for i, n := range workers {
		vals[i] = fmt.Sprintf("%q", n)
	}
	return fmt.Sprintf(`{"spec":{"template":{"spec":{"affinity":{"nodeAffinity":{"requiredDuringSchedulingIgnoredDuringExecution":{"nodeSelectorTerms":[{"matchExpressions":[{"key":"kubernetes.io/hostname","operator":"In","values":[%s]}]}]}}}}}}}`,
		strings.Join(vals, ","))
}

// ensureBusinessWorkload 确保业务负载就绪（前置 5；不存在时按 FIT 手册 0.5 口径创建，2 副本 busybox sleep 86400）。
// 业务负载带 worker 节点亲和（requiredDuringSchedulingIgnoredDuringExecution）：「worker 应有
// 业务 Pod」的前置由测试 setup 自身保证，不依赖调度器落点；创建与存量两种形态都覆盖——存量
// Deployment 亲和缺失或不一致时以 patch 补齐并滚动，快照在其后采集，UID 对比口径不变。
func ensureBusinessWorkload() []PodSnapshot {
	ns, dep := cfg.BizNamespace, cfg.BizDeployment
	By(fmt.Sprintf("确保业务负载就绪（ns=%s deploy=%s）", ns, dep))
	_, err := utils.ExecTrim(entry, fmt.Sprintf(`kubectl get ns %s >/dev/null 2>&1 || kubectl create ns %s`, ns, ns))
	Expect(err).NotTo(HaveOccurred())
	// 亲和目标节点名复用 k8sNodeIPMapping 的同一映射源解析（禁止另建节点角色判定）
	workers := workerNodeNames()
	Expect(workers).NotTo(BeEmpty(),
		"应能从集群节点映射解析出 worker 节点名（.env worker IP 须与集群节点 InternalIP 一致）")
	GinkgoWriter.Printf("业务负载 worker 亲和目标节点: %v\n", workers)
	if _, err := utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s get deploy %s -o name`, ns, dep)); err != nil {
		// 创建形态：容器命令时长 86400s 须覆盖用例全程，避免容器到期原地重建（RESTARTS+1）误判为加固影响
		yaml := fmt.Sprintf(`apiVersion: apps/v1
kind: Deployment
metadata:
  name: %s
  namespace: %s
spec:
  replicas: 2
  selector:
    matchLabels: {app: %s}
  template:
    metadata:
      labels: {app: %s}
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: kubernetes.io/hostname
                operator: In
                values: [%s]
      containers:
      - name: busybox
        image: %s
        command: ["sh", "-c", "sleep 86400"]`, dep, ns, dep, dep, strings.Join(workers, ", "), cfg.BizImage)
		_, err = utils.ExecTrim(entry, fmt.Sprintf(`cat <<'E2EOF' | kubectl apply -f -
%s
E2EOF`, yaml))
		Expect(err).NotTo(HaveOccurred(), "创建业务 Deployment 应成功")
	} else {
		// 存量形态：亲和缺失或与目标不一致时 patch 补齐（strategic merge 整体替换 nodeSelectorTerms）
		cur, err := utils.ExecTrim(entry, fmt.Sprintf(
			`kubectl -n %s get deploy %s -o jsonpath='{.spec.template.spec.affinity.nodeAffinity.requiredDuringSchedulingIgnoredDuringExecution.nodeSelectorTerms[*].matchExpressions[?(@.key=="kubernetes.io/hostname")].values[*]}'`,
			ns, dep))
		Expect(err).NotTo(HaveOccurred(), "查询业务 Deployment 亲和配置应成功")
		got := strings.Fields(cur)
		sort.Strings(got)
		if strings.Join(got, ",") != strings.Join(workers, ",") {
			_, err = utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s patch deploy %s -p '%s'`, ns, dep, workerAffinityPatch(workers)))
			Expect(err).NotTo(HaveOccurred(), "为存量业务 Deployment 补 worker 节点亲和应成功")
		}
	}
	_, err = utils.ExecTrim(entry, fmt.Sprintf(`kubectl -n %s rollout status deploy/%s --timeout=180s`, ns, dep))
	Expect(err).NotTo(HaveOccurred(), "业务 Deployment 应 rollout 完成（2/2 Running）")
	return deploymentPodSnapshots(ns, dep)
}

// systemPodRestarts 记录 kube-system 控制面组件 Pod（uid → restartCount），CrashLoop 判定用。
func systemPodRestarts(component string) map[string]int {
	out, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n kube-system get pods -l component=%s -o jsonpath='{range .items[*]}{.metadata.uid}{" "}{range .status.containerStatuses[*]}{.restartCount}{" "}{end}{"\n"}{end}'`,
		component))
	Expect(err).NotTo(HaveOccurred(), "查询 kube-system 组件 Pod 应成功: "+component)
	m := map[string]int{}
	for _, l := range nonEmptyLines(out) {
		f := strings.Fields(l)
		if len(f) >= 2 {
			rc := 0
			_, _ = fmt.Sscanf(f[1], "%d", &rc)
			m[f[0]] = rc
		}
	}
	return m
}

// assertControlPlanePodsHealthy 结果3 前半：cm/scheduler 两 Pod 重建后 Running 且 RESTARTS 无增长（无 CrashLoop）。
func assertControlPlanePodsHealthy(component string, before map[string]int) {
	By(fmt.Sprintf("校验 %s Pod 重建后健康（无 CrashLoop）", component))
	after := systemPodRestarts(component)
	Expect(after).NotTo(BeEmpty(), fmt.Sprintf("kube-system 应存在 component=%s 的 Pod", component))
	for uid, rc := range after {
		Expect(rc).To(BeNumerically("<=", restartOf(before, uid)+1),
			fmt.Sprintf("%s Pod RESTARTS 无异常增长（无 CrashLoop）", component))
	}
	state, err := utils.ExecTrim(entry, fmt.Sprintf(
		`kubectl -n kube-system get pods -l component=%s -o jsonpath='{range .items[*]}{.status.phase}{" "}{range .status.containerStatuses[*]}{.state.waiting.reason}{end}{"\n"}{end}'`, component))
	Expect(err).NotTo(HaveOccurred())
	Expect(state).To(ContainSubstring("Running"), fmt.Sprintf("%s Pod 应处于 Running", component))
	Expect(state).NotTo(ContainSubstring("CrashLoopBackOff"), fmt.Sprintf("%s 不应 CrashLoopBackOff", component))
}

// auditPolicyMD5 各 master audit-policy.yaml md5（FIT_15 零修改断言）。
func auditPolicyMD5() map[string]string {
	m := map[string]string{}
	for _, n := range cfg.Masters() {
		out, err := utils.ExecTrim(nodeExecs[n.IP], fmt.Sprintf(`md5sum %s | cut -d' ' -f1`, utils.ShellPath(utils.AuditPolicy)))
		Expect(err).NotTo(HaveOccurred(), "读取 audit-policy md5 应成功: "+n.IP)
		m[n.IP] = strings.TrimSpace(out)
	}
	return m
}

// registeredNodeNames 集群注册节点名（hostname-override 一致性断言）。
func registeredNodeNames() []string {
	out, err := utils.ExecTrim(entry, `kubectl get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}'`)
	Expect(err).NotTo(HaveOccurred(), "kubectl get nodes 应成功")
	names := nonEmptyLines(out)
	sort.Strings(names)
	return names
}

// nodeLogExistsOnEntry 判断 entry_node 是否存在该节点的本轮加固日志：
// 工具实际为扁平布局 /tmp/cluster-harden-results-*/harden-<ip>.log（run6 实测，一层目录；
// 此前实现多写一层 */ 与布局不符而永不匹配），并要求文件内容含本轮统一备份时间戳 ts——
// /tmp 下可能残留多轮历史结果目录，仅按文件名存在会造成假匹配。
// glob 无匹配时 shell 将字面量传给 grep（exit=2）、有匹配但均不含 ts 时 grep -l 输出为空
// （exit=1），两种情况均正确返回 false。
func nodeLogExistsOnEntry(globDir, ip, ts string) bool {
	res := utils.Run(entry, fmt.Sprintf(
		"grep -l '%s' %s/harden-%s.log 2>/dev/null | grep .", ts, globDir, ip))
	return res.ExitCode == 0 && strings.TrimSpace(res.Stdout) != ""
}

func restartOf(m map[string]int, uid string) int {
	if rc, ok := m[uid]; ok {
		return rc
	}
	return 0
}

func podUIDMap(snaps []PodSnapshot) map[string]string {
	m := map[string]string{}
	for _, s := range snaps {
		m[s.Name] = s.UID
	}
	return m
}

func stringMapEqual(a, b map[string]string) bool {
	if len(a) != len(b) {
		return false
	}
	for key, value := range a {
		other, ok := b[key]
		if !ok || other != value {
			return false
		}
	}
	return true
}

func filterSnapshots(snaps []PodSnapshot, role string) []PodSnapshot {
	var out []PodSnapshot
	for _, s := range snaps {
		if role == "" || nodeRole(s.Node) == role {
			out = append(out, s)
		}
	}
	return out
}

func nonEmptyLines(s string) []string {
	var out []string
	for _, l := range strings.Split(s, "\n") {
		if l = strings.TrimSpace(l); l != "" {
			out = append(out, l)
		}
	}
	return out
}
