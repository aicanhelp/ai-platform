package utils

import (
	"encoding/json"
	"fmt"
	"regexp"
	"sort"
	"strings"

	"gitcode.com/openFuyao/e2e-auto-test/e2e/framework/executor"
)

// CISResult kube-bench（cis-1.12）复扫结果摘要。
type CISResult struct {
	Pass      int
	Fail      int
	Warn      int
	Total     int
	FailItems []string // 失败项编号（如 1.1.1 / 1.2.5）
}

// STIGResult oscap（MAC-3_Public）复扫结果摘要。
type STIGResult struct {
	Counts    map[string]int // 结果态计数（pass/fail/notapplicable/notchecked...）
	FailItems []string       // FAIL 项（SV-xxxxxxrxxxxxx）
	NAItems   []string       // notapplicable 项（如控制面节点恒 NA 的 SV-242393）
}

// ImageAvailable 探测 entry_node 上扫描镜像是否可用（不可用时先尝试拉取一次）。
// E2E-A 结果6 断言块依据此结果降级跳过（用例基线已确认口径）。
func ImageAvailable(e *executor.SSHExecutor, image string) bool {
	res := Run(e, fmt.Sprintf("nerdctl image inspect %s >/dev/null 2>&1 || nerdctl pull %s >/dev/null 2>&1", image, image))
	return res.ExitCode == 0
}

// RunKubeBench 在 entry_node 用 nerdctl 直跑 kube-bench（挂载口径与 FIT 自验手册 0.3 一致：
// --net=host 绕开 CNI 限制，只读挂载被检路径），输出 JSON 落 <workDir> 并解析摘要。
// 退出码口径：kube-bench run 未指定 --exit-code 时即使存在 fail 项也返回 0（与 oscap 的
// exit=2 语义不同），故 ExecTrim 非零即错成立；已在 run2-run4 真实集群采集验证
// （CIS 基线含 FAIL=10 仍 exit=0 成功采集）。
func RunKubeBench(e *executor.SSHExecutor, cfg *Config) (*CISResult, error) {
	etcdDataDir, err := DetectEtcdDataDir(e)
	if err != nil {
		return nil, err
	}
	outPath := fmt.Sprintf("%s/kube-bench-%d.json", cfg.WorkDir, timeNowUnix())
	cmd := fmt.Sprintf(
		"nerdctl run --rm --net=host --pid=host"+
			" -v /etc/kubernetes:/etc/kubernetes:ro"+
			" -v /var/lib/kubelet:/var/lib/kubelet:ro"+
			" -v %s:%s:ro"+
			" -v /etc/systemd:/etc/systemd:ro"+
			" -v /usr/bin:/usr/local/bin/host:ro"+
			" %s run --targets master,controlplane,etcd,policies --benchmark cis-1.12 --json > %s 2>/dev/null",
		shellPath(etcdDataDir), shellPath(etcdDataDir), cfg.KubeBenchImage, shellPath(outPath))
	if _, err := ExecTrim(e, cmd); err != nil {
		return nil, fmt.Errorf("kube-bench 复扫运行失败: %w", err)
	}
	raw, err := ExecTrim(e, fmt.Sprintf("cat %s", shellPath(outPath)))
	if err != nil {
		return nil, err
	}
	return ParseKubeBench(raw)
}

var kbJSONStart = regexp.MustCompile(`^\{"Controls"`)

// ParseKubeBench 从输出（含前后噪音）解析 kube-bench JSON：定位 {"Controls" 起点，逐项汇总。
func ParseKubeBench(raw string) (*CISResult, error) {
	idx := strings.Index(raw, `{"Controls"`)
	if idx < 0 {
		return nil, fmt.Errorf("kube-bench 输出中未找到 JSON 数据（前 200 字符：%s）", truncate(raw, 200))
	}
	dec := json.NewDecoder(strings.NewReader(raw[idx:]))
	var payload struct {
		Controls []struct {
			Tests []struct {
				Pass    int `json:"pass"`
				Fail    int `json:"fail"`
				Warn    int `json:"warn"`
				Results []struct {
					Status     string `json:"status"`
					TestNumber string `json:"test_number"`
				} `json:"results"`
			} `json:"tests"`
		} `json:"Controls"`
	}
	if err := dec.Decode(&payload); err != nil {
		return nil, fmt.Errorf("kube-bench JSON 解析失败: %w", err)
	}
	r := &CISResult{}
	seen := map[string]bool{}
	for _, c := range payload.Controls {
		for _, t := range c.Tests {
			r.Pass += t.Pass
			r.Fail += t.Fail
			r.Warn += t.Warn
			for _, item := range t.Results {
				if item.Status == "FAIL" && item.TestNumber != "" && !seen[item.TestNumber] {
					seen[item.TestNumber] = true
					r.FailItems = append(r.FailItems, item.TestNumber)
				}
			}
		}
	}
	r.Total = r.Pass + r.Fail + r.Warn
	sort.Strings(r.FailItems)
	return r, nil
}

// RunOpenSCAP 在 entry_node 用 nerdctl 直跑 oscap（xccdf eval，MAC-3_Public profile，
// 挂载/USER 0:0 口径与 FIT 自验手册 0.3b 一致），解析 eval.log 产出摘要。
// 退出码 0/2 均视为评估完成（2=存在 fail 规则的正常返回），并以 results.xml/arf.xml
// 非空落盘为「评估完成」证据；其余退出码视为执行错误。
func RunOpenSCAP(e *executor.SSHExecutor, cfg *Config) (*STIGResult, error) {
	etcdDataDir, err := DetectEtcdDataDir(e)
	if err != nil {
		return nil, err
	}
	outDir := fmt.Sprintf("%s/oscap-%d", cfg.WorkDir, timeNowUnix())
	setup := fmt.Sprintf("mkdir -p %s/results && rm -f %s/results/*", shellPath(outDir), shellPath(outDir))
	if _, err := ExecTrim(e, setup); err != nil {
		return nil, err
	}
	cmd := fmt.Sprintf(
		"nerdctl run --rm --net=host --pid=host --user 0:0"+
			" -v /etc/kubernetes:/etc/kubernetes:ro"+
			" -v /var/lib/kubelet:/var/lib/kubelet:ro"+
			" -v %s:/var/lib/etcd:ro"+
			" -v /etc/systemd:/etc/systemd:ro"+
			" -v /var/log:/var/log:ro"+
			" -v /etc/passwd:/etc/passwd:ro"+
			" -v /etc/group:/etc/group:ro"+
			" -v %s/results:/results"+
			" %s xccdf eval --profile xccdf_mil.disa.stig_profile_MAC-3_Public"+
			" --results-arf /results/arf.xml --results /results/results.xml"+
			" /scap-content/scap-rule.xml > %s/eval.log 2>&1",
		shellPath(etcdDataDir), shellPath(outDir), cfg.OpenSCAPImage, shellPath(outDir))
	// oscap xccdf eval 退出码口径（OpenSCAP 约定）：0=评估完成且全部规则 pass；
	// 2=评估完成且存在 fail 规则（S0 未加固基线的正常形态，评估本身成功）；
	// 其余非零才是执行错误。故不能用 ExecTrim 一律按非零报错。
	res := Run(e, cmd)
	if res.ExitCode != 0 && res.ExitCode != 2 {
		logTail, _ := ExecTrim(e, fmt.Sprintf("tail -c 2000 %s/eval.log 2>/dev/null || true", shellPath(outDir)))
		return nil, fmt.Errorf("oscap 复扫运行失败: 命令失败（exit=%d）: %s\neval.log 尾部:\n%s",
			res.ExitCode, cmd, logTail)
	}
	// 「评估完成」证据校验：结果文件须完整落盘且非空，防止真实执行错误被 0/2 误放行。
	if chk := Run(e, fmt.Sprintf("test -s %s/results/results.xml && test -s %s/results/arf.xml",
		shellPath(outDir), shellPath(outDir))); chk.ExitCode != 0 {
		return nil, fmt.Errorf("oscap 评估产物缺失或为空（%s/results/results.xml、arf.xml），评估可能未完成（exit=%d）",
			outDir, res.ExitCode)
	}
	logText, err := ExecTrim(e, fmt.Sprintf("cat %s/eval.log", shellPath(outDir)))
	if err != nil {
		return nil, err
	}
	return ParseOpenSCAPLog(logText)
}

var (
	// oscap CLI 实测输出为 TAB 分隔且字段名后带 \r（如 "Title\r\t<文本>"、"Result\r\tfail"，
	// 见 run2 基线 eval.log），兼容 "\r\t" 与 ": " 两种分隔形态。
	oscapRuleRe   = regexp.MustCompile(`SV-[0-9]+`) // 规则 ID 归一化为裸 SV 编号，与 AllowedSTIGRemainAfterHarden 断言清单同口径
	oscapTitleRe  = regexp.MustCompile(`^Title\s*:?\s*(.*)$`)
	oscapRuleLine = regexp.MustCompile(`^Rule\s*:?\s*(\S+)`) // Rule 行携带 xccdf_mil.disa.stig_rule_SV-…_rule 全名
	oscapResultRe = regexp.MustCompile(`^Result\s*:?\s*(\S+)`)
)

// ParseOpenSCAPLog 解析 oscap eval.log：按 Title/Rule/Result 三元组逐规则汇总
// （口径与 FIT 自验手册「FAIL 项/NA 项」提取一致）。规则 ID 优先取 Rule 行
// （实测 Title 文本不含 SV 编号），归一化为裸 SV-xxxxxx 形式。
func ParseOpenSCAPLog(logText string) (*STIGResult, error) {
	r := &STIGResult{Counts: map[string]int{}}
	currentRule := ""
	for _, line := range strings.Split(logText, "\n") {
		if m := oscapRuleLine.FindStringSubmatch(line); m != nil {
			if rule := oscapRuleRe.FindString(m[1]); rule != "" {
				currentRule = rule
			}
			continue
		}
		if m := oscapTitleRe.FindStringSubmatch(line); m != nil {
			if rule := oscapRuleRe.FindString(m[1]); rule != "" {
				currentRule = rule
			}
			continue
		}
		if m := oscapResultRe.FindStringSubmatch(line); m != nil {
			status := m[1]
			r.Counts[status]++
			if currentRule != "" {
				switch status {
				case "fail":
					r.FailItems = append(r.FailItems, currentRule)
				case "notapplicable":
					r.NAItems = append(r.NAItems, currentRule)
				}
			}
			currentRule = ""
		}
	}
	if len(r.Counts) == 0 {
		return nil, fmt.Errorf("oscap eval.log 中未解析到任何规则结果（前 200 字符：%s）", truncate(logText, 200))
	}
	sort.Strings(r.FailItems)
	sort.Strings(r.NAItems)
	return r, nil
}

// ContainsItem 判断列表是否包含指定项（精确匹配）。
func ContainsItem(items []string, target string) bool {
	for _, it := range items {
		if it == target {
			return true
		}
	}
	return false
}

// SubsetOf 判断 items 的每一项都在 allowed 中（复扫残留项口径断言用）。
func SubsetOf(items, allowed []string) []string {
	var excess []string
	for _, it := range items {
		if !ContainsItem(allowed, it) {
			excess = append(excess, it)
		}
	}
	return excess
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "..."
}
