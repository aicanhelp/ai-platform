package utils

import (
	"fmt"
	"regexp"
	"sort"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// Run 在目标节点执行命令并返回完整结果（不因非零退出视为异常，由调用方按用例口径断言）。
func Run(e *executor.SSHExecutor, cmd string) *executor.ExecResult {
	res, err := e.Exec(cmd)
	if err != nil && res == nil {
		// SSH 通道级失败（连接中断等）：返回带错误描述的结果，保证调用方拿到非 nil
		return &executor.ExecResult{Command: cmd, ExitCode: -1, Stderr: err.Error()}
	}
	return res
}

// ExecTrim 执行命令并返回去空白 stdout；非零退出视为错误（带 stderr）。
func ExecTrim(e *executor.SSHExecutor, cmd string) (string, error) {
	res, err := e.Exec(cmd)
	if err != nil {
		if res != nil {
			return "", fmt.Errorf("命令失败（exit=%d）: %s\nstderr: %s", res.ExitCode, cmd, strings.TrimSpace(res.Stderr))
		}
		return "", fmt.Errorf("命令执行通道错误: %s: %w", cmd, err)
	}
	return strings.TrimSpace(res.Stdout), nil
}

// FilePerm 返回文件权限（stat -c %a，如 644/600）。
func FilePerm(e *executor.SSHExecutor, path string) (string, error) {
	return ExecTrim(e, fmt.Sprintf("stat -c %%a %s", shellPath(path)))
}

// FileOwner 返回文件属主（stat -c %U:%G，如 root:root）。
func FileOwner(e *executor.SSHExecutor, path string) (string, error) {
	return ExecTrim(e, fmt.Sprintf("stat -c %%U:%%G %s", shellPath(path)))
}

// EtcdDataDirState is the per-node state used to prove that hardening preserves
// the owner and group of the actual etcd data directory.
type EtcdDataDirState struct {
	Path       string
	OwnerGroup string
}

// DetectEtcdDataDir resolves --data-dir from the etcd static Pod manifest.
// Both --data-dir=/path and the split --data-dir /path forms are supported.
// Missing manifests, missing arguments, non-absolute paths, and missing
// directories are errors so an E2E assertion can never silently skip a node.
func DetectEtcdDataDir(e *executor.SSHExecutor) (string, error) {
	cmd := fmt.Sprintf(`set -o pipefail
manifest=%s
test -f "$manifest" || { echo "etcd manifest is missing: $manifest" >&2; exit 11; }
data_dir="$(yq eval -r '.spec.containers[] | select(.name == "etcd") | (.command[]?, .args[]?)' "$manifest" 2>/dev/null | awk '
  /^--data-dir=/ { sub(/^--data-dir=/, ""); print; exit }
  take_next { print; exit }
  $0 == "--data-dir" { take_next=1 }
')"
test -n "$data_dir" || { echo "--data-dir was not found in $manifest" >&2; exit 12; }
case "$data_dir" in /*) ;; *) echo "detected etcd data-dir is not absolute: $data_dir" >&2; exit 13 ;; esac
test -d "$data_dir" || { echo "detected etcd data-dir does not exist: $data_dir" >&2; exit 14; }
printf '%%s\n' "$data_dir"`, shellPath(ManifestEtcd))
	out, err := ExecTrim(e, cmd)
	if err != nil {
		return "", fmt.Errorf("detect etcd data-dir from %s: %w", ManifestEtcd, err)
	}
	path := strings.TrimSpace(out)
	if path == "" {
		return "", fmt.Errorf("detect etcd data-dir from %s: empty result", ManifestEtcd)
	}
	return path, nil
}

// CaptureEtcdDataDirState detects the real path and reads its owner/group.
func CaptureEtcdDataDirState(e *executor.SSHExecutor) (EtcdDataDirState, error) {
	path, err := DetectEtcdDataDir(e)
	if err != nil {
		return EtcdDataDirState{}, err
	}
	ownerGroup, err := FileOwner(e, path)
	if err != nil {
		return EtcdDataDirState{Path: path}, fmt.Errorf("stat etcd data-dir owner/group at %s: %w", path, err)
	}
	return EtcdDataDirState{Path: path, OwnerGroup: ownerGroup}, nil
}

// FileExists 判断文件/目录是否存在。
func FileExists(e *executor.SSHExecutor, path string) bool {
	res := Run(e, fmt.Sprintf("test -e %s && echo yes", shellPath(path)))
	return res.ExitCode == 0 && strings.TrimSpace(res.Stdout) == "yes"
}

// GlobCount 统计远端 glob 展开的文件数量（0 个时返回 0 而非报错）。
func GlobCount(e *executor.SSHExecutor, pattern string) (int, error) {
	out, err := ExecTrim(e, fmt.Sprintf("ls -1 %s 2>/dev/null | wc -l", pattern))
	if err != nil {
		return 0, err
	}
	return parseInt(out), nil
}

// BackupDirCount 统计备份目录下的时间戳子目录数量。
func BackupDirCount(e *executor.SSHExecutor, backupDir string) (int, error) {
	out, err := ExecTrim(e, fmt.Sprintf("find %s -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l", shellPath(backupDir)))
	if err != nil {
		return 0, err
	}
	return parseInt(out), nil
}

// BackupExists 判断节点上是否存在指定统一备份（以 changes.json 为准，与工具预检口径一致）。
func BackupExists(e *executor.SSHExecutor, backupDir, ts string) bool {
	return FileExists(e, fmt.Sprintf("%s/%s/changes.json", backupDir, ts))
}

// HardenedOriginMarkerExists 检测备份目录下的加固起点标记（.harden-origin，回退成功后自动清除）。
func HardenedOriginMarkerExists(e *executor.SSHExecutor, backupDir string) bool {
	return FileExists(e, fmt.Sprintf("%s/%s", backupDir, HardenOriginMarker))
}

// KubeletCmdline 返回运行中 kubelet 的完整启动参数（hostname-override 保留性断言用）。
func KubeletCmdline(e *executor.SSHExecutor) (string, error) {
	return ExecTrim(e, `cat /proc/$(pidof kubelet | awk '{print $1}')/cmdline | tr '\0' ' '`)
}

// CollectNodeFingerprint 采集单节点配置指纹（md5+权限+属主），口径与 FIT 自验手册 0.2 一致：
// master 8 行（7 文件 + etcd 数据目录）、worker 2 行（ca.crt + kubelet config）；已排序便于 diff。
func CollectNodeFingerprint(e *executor.SSHExecutor, node NodeConfig, cfg *Config) (string, error) {
	files := []string{ManifestAPIServer, ManifestController, ManifestScheduler, ManifestEtcd, cfg.KubeletConfig, AuditPolicy, CACert}
	quoted := make([]string, 0, len(files))
	for _, f := range files {
		quoted = append(quoted, shellQuote(f))
	}
	etcdDataDir := ""
	if node.Role == RoleMaster {
		var err error
		etcdDataDir, err = DetectEtcdDataDir(e)
		if err != nil {
			return "", fmt.Errorf("node %s: %w", node.IP, err)
		}
	}
	script := fmt.Sprintf(`
for f in %s; do
  [ -f "$f" ] && printf '%%s %%s %%s %%s %%s\n' %s "$f" "$(md5sum "$f" | cut -d' ' -f1)" "$(stat -c %%a "$f")" "$(stat -c %%U:%%G "$f")"
done
[ -d %s ] && printf '%%s etcd-datadir - %%s %%s\n' %s "$(stat -c %%a %s)" "$(stat -c %%U:%%G %s)"
exit 0`,
		strings.Join(quoted, " "), shellQuote(node.IP),
		shellQuote(etcdDataDir), shellQuote(node.IP),
		shellQuote(etcdDataDir), shellQuote(etcdDataDir))
	out, err := ExecTrim(e, script)
	if err != nil {
		return "", fmt.Errorf("节点 %s 指纹采集失败: %w", node.IP, err)
	}
	lines := strings.Split(out, "\n")
	sort.Strings(lines)
	return strings.TrimSpace(strings.Join(lines, "\n")), nil
}

// FingerprintSet 全节点指纹集合（key=节点 IP，value=该节点排序后的指纹文本）。
type FingerprintSet map[string]string

// CollectAllFingerprints 并发采集全部节点指纹（任一节点失败即返回错误）。
func CollectAllFingerprints(cfg *Config, execs map[string]*executor.SSHExecutor) (FingerprintSet, error) {
	set := FingerprintSet{}
	type result struct {
		ip          string
		fingerprint string
		err         error
	}
	resultCh := make(chan result, len(cfg.Nodes))
	for _, n := range cfg.Nodes {
		go func(n NodeConfig) {
			e, ok := execs[n.IP]
			if !ok {
				resultCh <- result{err: fmt.Errorf("节点 %s 无可用 SSH 执行器", n.IP)}
				return
			}
			fp, err := CollectNodeFingerprint(e, n, cfg)
			if err != nil {
				resultCh <- result{err: err}
				return
			}
			resultCh <- result{ip: n.IP, fingerprint: fp}
		}(n)
	}
	for range cfg.Nodes {
		result := <-resultCh
		if result.err != nil {
			return nil, result.err
		}
		set[result.ip] = result.fingerprint
	}
	return set, nil
}

// DiffFingerprints 对比两份全节点指纹，返回差异行（空切片=零差异）。
func DiffFingerprints(before, after FingerprintSet) []string {
	var diffs []string
	ips := make([]string, 0, len(before))
	for ip := range before {
		ips = append(ips, ip)
	}
	sort.Strings(ips)
	for _, ip := range ips {
		bLines := splitLines(before[ip])
		aSet := map[string]bool{}
		for _, l := range splitLines(after[ip]) {
			aSet[l] = true
		}
		bSet := map[string]bool{}
		for _, l := range bLines {
			bSet[l] = true
			if !aSet[l] {
				diffs = append(diffs, fmt.Sprintf("%s 变更前: %s", ip, l))
			}
		}
		for _, l := range splitLines(after[ip]) {
			if !bSet[l] {
				diffs = append(diffs, fmt.Sprintf("%s 变更后: %s", ip, l))
			}
		}
	}
	return diffs
}

// FingerprintText 展开为多行文本（失败信息展示用）。
func (s FingerprintSet) FingerprintText() string {
	ips := make([]string, 0, len(s))
	for ip := range s {
		ips = append(ips, ip)
	}
	sort.Strings(ips)
	var sb strings.Builder
	for _, ip := range ips {
		sb.WriteString(fmt.Sprintf("== %s ==\n%s\n", ip, s[ip]))
	}
	return sb.String()
}

// APIServerPermFromFingerprint 从指纹中提取该节点 apiserver manifest 权限（S0/加固态判定用）。
func APIServerPermFromFingerprint(fp string) string {
	re := regexp.MustCompile(`^(\S+) /etc/kubernetes/manifests/kube-apiserver\.yaml \S+ (\d+) `)
	for _, line := range splitLines(fp) {
		if m := re.FindStringSubmatch(line); m != nil {
			return m[2]
		}
	}
	return ""
}

func splitLines(s string) []string {
	out := []string{}
	for _, l := range strings.Split(s, "\n") {
		l = strings.TrimSpace(l)
		if l != "" {
			out = append(out, l)
		}
	}
	return out
}

func parseInt(s string) int {
	n := 0
	_, _ = fmt.Sscanf(strings.TrimSpace(s), "%d", &n)
	return n
}

// ShellPath 供测试侧拼装远端命令使用：安全路径原样返回，其余单引号包裹。
func ShellPath(s string) string {
	return shellPath(s)
}

// shellQuote 单引号包裹远端字符串（IP/路径均来自 .env，此处兜底防注入）。
func shellQuote(s string) string {
	return "'" + strings.ReplaceAll(s, "'", `'\''`) + "'"
}

func shellPath(s string) string {
	if regexp.MustCompile(`^[A-Za-z0-9_./:-]+$`).MatchString(s) {
		return s
	}
	return shellQuote(s)
}
