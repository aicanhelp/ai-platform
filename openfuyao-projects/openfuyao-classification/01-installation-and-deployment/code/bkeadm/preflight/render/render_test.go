/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package render

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"unicode"

	prettytext "github.com/jedib0t/go-pretty/v6/text"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	clusteradapter "gopkg.openfuyao.cn/preflight/adapter/openfuyao/clustercreate"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
)

func sampleReport() api.Report {
	return api.Report{
		SchemaVersion: api.SchemaVersion,
		Operation:     api.OperationInit,
		Overall:       api.OverallWarn,
		ExitCode:      api.ExitCode(api.OverallWarn),
		Summary:       api.Summary{Total: 2, Pass: 1, Warn: 1},
		Results: []api.Result{
			{InstanceID: "INIT-PLAT-OS@bootstrap", RuleID: "INIT-PLAT-OS", Name: "Operating system [uname -s]",
				Status: api.StatusPass, Subject: "bootstrap",
				Expected: api.DisplayValue{Value: "linux"}, Actual: api.DisplayValue{Value: "linux"}},
			{InstanceID: "INIT-COLOCATE@bootstrap", RuleID: "INIT-COLOCATE", Name: "Colocation risk",
				Status: api.StatusWarn, Subject: "bootstrap",
				Expected: api.DisplayValue{Value: "not-colocated"}, Actual: api.DisplayValue{Value: "same-node"}, Remediation: "confirm capacity and failure-domain risk"},
		},
	}
}

func TestTextRendersAllResults(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, sampleReport()))
	out := buf.String()
	assert.Contains(t, out, "openFuyao init 前置检查报告")
	assert.Contains(t, out, "结论：警告（可以继续）")
	assert.Contains(t, out, "适用：2  通过：1  失败：0  警告：1")
	assert.Contains(t, out, "检查结果")
	assert.Contains(t, out, "主机操作系统")
	assert.Contains(t, out, "节点：bootstrap")
	assert.Contains(t, out, "位置：本机")
	assert.Contains(t, out, "uname -s")
	assert.Contains(t, out, "Linux")
	assert.Contains(t, out, "共部署风险")
	rows := resultRows(sampleReport().Results)
	assert.Equal(t, "存在共部署风险；确认容量和故障域满足要求", rows[1].CauseFix)
	assert.NotContains(t, out, "INIT-PLAT-OS")
	assert.NotContains(t, out, "DOC-BEFORE")
}

func TestJSONIsCanonical(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, JSON(&buf, sampleReport()))
	out := buf.String()
	assert.Contains(t, out, `"overall": "WARN"`)
	assert.Contains(t, out, `"exitCode": 2`)
	assert.Contains(t, out, `"ruleID": "INIT-PLAT-OS"`)
	assert.Contains(t, out, `"status": "PASS"`)
	assert.NotContains(t, out, `"diagnostics"`)
	// HTML escaping disabled: no & etc.
	assert.NotContains(t, out, `&`)
	// A passing result must not carry remediation; only the WARN entry does.
	assert.Contains(t, out, `"remediation": "confirm capacity and failure-domain risk"`)
	assert.NotRegexp(t, `"status": "PASS"[\s\S]*?"remediation"`, out)
}

func TestJSONRoundTrip(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, JSON(&buf, sampleReport()))
	var r api.Report
	require.NoError(t, json.Unmarshal(buf.Bytes(), &r))
	assert.Equal(t, api.OverallWarn, r.Overall)
	assert.Len(t, r.Results, 2)
}

func TestCustomCheckRowIsReadableAndRedacted(t *testing.T) {
	result := api.Result{
		RuleID: "CONFIG-CLUSTER-CREATE-CUSTOM-RUNTIME-HEALTH", Name: "容器运行时健康状态",
		Status: api.StatusFail, Subject: "worker-1",
		Expected: api.DisplayValue{Type: api.ValueTypeString, Value: "运行时为 containerd 且健康"},
		Actual: api.DisplayValue{Type: api.ValueTypeObject, Value: map[string]any{
			"runtime": "containerd", "healthy": false,
		}},
		Evidence:    []api.Evidence{{Kind: "command", Locator: "/opt/openfuyao/check-runtime", Observed: "sha256:abc"}},
		Remediation: "启动 containerd 后重试",
	}
	row := resultRows([]api.Result{result})[0]
	assert.Equal(t, "节点：worker-1\n/opt/openfuyao/check-runtime", row.Target)
	assert.Equal(t, "运行时为 containerd 且健康", row.Expected)
	assert.Equal(t, `{"healthy":false,"runtime":"containerd"}`, row.Actual)
	assert.Equal(t, "启动 containerd 后重试", row.CauseFix)
	assert.NotContains(t, row.Actual, "sha256")
}

func TestTextEmptyReport(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, api.Report{Operation: api.OperationInit, Overall: api.OverallSkip, Summary: api.Summary{Total: 0}}))
	out := buf.String()
	assert.Contains(t, out, "结论：跳过（没有适用检查）")
	assert.Contains(t, out, "适用：0  通过：0  失败：0  警告：0")
	assert.False(t, strings.Contains(out, "[PASS]"))
}

func TestSummaryShowsOnlyActionableResults(t *testing.T) {
	report := api.Report{
		Operation: api.OperationClusterCreate,
		Overall:   api.OverallFail,
		Summary:   api.Summary{Total: 5, Pass: 1, Fail: 1, Warn: 1, Skip: 2},
		Results: []api.Result{
			{InstanceID: "pass", RuleID: "PASS", Name: "Healthy check", Status: api.StatusPass},
			{InstanceID: "warn", RuleID: "WARN", Name: "Warning check", Status: api.StatusWarn, Remediation: "review the warning"},
			{InstanceID: "fail", RuleID: "FAIL", Name: "Failed check", Status: api.StatusFail, Remediation: "fix the failure"},
			{InstanceID: "not-applicable", RuleID: "SKIP-NA", Name: "Disabled check", Status: api.StatusSkip, CauseCode: api.CauseNotApplicable},
			{InstanceID: "blocked", RuleID: "SKIP-BLOCKED", Name: "Blocked check", Status: api.StatusSkip, CauseCode: api.CauseDependencyFailed, DependsOn: "FAIL"},
		},
	}
	var buf bytes.Buffer
	require.NoError(t, Summary(&buf, report, "/tmp/report.txt"))
	out := buf.String()
	assert.Contains(t, out, "需要处理")
	assert.Contains(t, out, "状态")
	assert.Contains(t, out, "详细报告：/tmp/report.txt")
	assert.Contains(t, out, "适用：3  通过：1  失败：1  警告：1")
	assert.Contains(t, out, "依赖阻塞或跳过：2（详见报告）")
	assert.NotContains(t, out, "Healthy check")
	assert.NotContains(t, out, "Disabled check")
	assert.NotContains(t, out, "Blocked check")
	assert.NotContains(t, out, "SKIP")
	assert.Less(t, strings.Index(out, "Failed check"), strings.Index(out, "Warning check"))
}

func TestSummaryShowsRunIssues(t *testing.T) {
	report := api.Report{
		Operation: api.OperationClusterCreate,
		Overall:   api.OverallFail,
		Summary:   api.Summary{Total: 1, Pass: 1},
		Issues: []api.RunIssue{
			{Code: "SETUP", Severity: api.IssueError, Stage: api.IssueStageSetup, Effect: api.IssueContinue, Cause: "provider defaults unavailable", Action: "restore API access"},
			{Code: "NOTICE", Severity: api.IssueWarn, Stage: api.IssueStageSetup, Effect: api.IssueContinue, Cause: "static defaults used"},
		},
	}
	var buf bytes.Buffer
	require.NoError(t, Summary(&buf, report, ""))
	out := buf.String()
	assert.Contains(t, out, "运行问题")
	assert.Equal(t, "provider defaults unavailable; restore API access", issueRows(report.Issues, true)[0].CauseFix)
	assert.NotContains(t, out, "diagnostic")
}

func TestSummarySaysChecksDidNotStart(t *testing.T) {
	report := api.Report{Operation: api.OperationClusterCreate, Overall: api.OverallInputError,
		Issues: []api.RunIssue{{Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
			Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Field: "nodes[0].spec.ip",
			Cause: "invalid BKENode YAML", Action: "correct the file and rerun"}}}
	var buf bytes.Buffer
	require.NoError(t, Summary(&buf, report, ""))
	out := buf.String()
	assert.Contains(t, out, "状态：未启动")
	assert.Contains(t, out, "检查：未启动（已执行 0 项）")
	assert.Contains(t, out, "阶段：输入")
	assert.Contains(t, issueRows(report.Issues, true)[0].Target, "对象：nodes[0].spec.ip")
}

func TestRunIssueMissingFileIsConciseAndKeepsLocator(t *testing.T) {
	issue := api.RunIssue{Code: "CLUSTER-INPUT-SCHEMA", Severity: api.IssueError,
		Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Field: "cluster",
		Cause:  "read cluster file /etc/openfuyao/cluster.yaml: stat /etc/openfuyao/cluster.yaml: no such file or directory",
		Action: "correct the cluster file, then run preflight again"}
	row := issueRows([]api.RunIssue{issue}, true)[0]
	assert.Equal(t, "输入校验", row.Check)
	assert.Contains(t, row.Target, "/etc/openfuyao/cluster.yaml")
	assert.Equal(t, "文件不存在；确认目标路径并补齐文件后重试", row.CauseFix)
	assert.NotContains(t, row.CauseFix, "stat")
}

func TestDetailedIssueKeepsChineseRepairAndRawTechnicalCause(t *testing.T) {
	issue := api.RunIssue{Code: "PREFLIGHT-OPTIONS", Severity: api.IssueError,
		Stage: api.IssueStageInput, Effect: api.IssueBlockStart, Field: "timeout",
		Cause: "--timeout must be greater than zero", Action: "set --timeout and rerun"}
	row := issueRows([]api.RunIssue{issue}, false)[0]
	assert.Contains(t, row.CauseFix, "--timeout 必须大于 0")
	assert.Contains(t, row.CauseFix, "详情：--timeout must be greater than zero")
}

func TestOrderedReportPlacesPassBeforeSkip(t *testing.T) {
	report := api.Report{Results: []api.Result{
		{InstanceID: "skip", RuleID: "SKIP", Status: api.StatusSkip},
		{InstanceID: "pass", RuleID: "PASS", Status: api.StatusPass},
		{InstanceID: "warn", RuleID: "WARN", Status: api.StatusWarn},
		{InstanceID: "fail", RuleID: "FAIL", Status: api.StatusFail},
	}}
	var buf bytes.Buffer
	require.NoError(t, JSON(&buf, report))
	out := buf.String()
	assert.Less(t, strings.Index(out, `"instanceID": "fail"`), strings.Index(out, `"instanceID": "warn"`))
	assert.Less(t, strings.Index(out, `"instanceID": "warn"`), strings.Index(out, `"instanceID": "pass"`))
	assert.Less(t, strings.Index(out, `"instanceID": "pass"`), strings.Index(out, `"instanceID": "skip"`))
}

func TestTextFormatsByteValuesWithHumanUnits(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{
		{InstanceID: "bytes", RuleID: "CHECK", Name: "Storage", Status: api.StatusPass,
			Expected: api.DisplayValue{Type: api.ValueTypeBytes, Value: int64(1536)},
			Actual:   api.DisplayValue{Type: api.ValueTypeBytes, Value: int64(3 * 1024 * 1024)}},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	assert.Contains(t, out, "1.5 KiB")
	assert.Contains(t, out, "3 MiB")
}

func TestTextDoesNotFormatCPUCountsAsBytes(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{
		{InstanceID: "cpu", RuleID: "INIT-RES-CPU", Name: "Host CPU capacity", Status: api.StatusPass,
			Expected: api.DisplayValue{Value: 2}, Actual: api.DisplayValue{Value: 14}},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	assert.Contains(t, out, " 2 ")
	assert.Contains(t, out, " 14 ")
	assert.NotContains(t, out, "2 B")
	assert.NotContains(t, out, "14 B")
}

func TestTextDoesNotFormatPlainIntegerAsBytes(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{
		{InstanceID: "plain", RuleID: "CHECK", Name: "Plain integer", Status: api.StatusPass,
			Expected: api.DisplayValue{Value: 1}, Actual: api.DisplayValue{Value: 1}},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	assert.Contains(t, out, " 1 ")
	assert.NotContains(t, out, "1 B")
}

func TestTextExplainsRootUserFailure(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{
		{InstanceID: "INIT-EXEC-001@bootstrap", RuleID: "INIT-EXEC-001", Status: api.StatusFail,
			Expected: api.DisplayValue{Value: 0}, Actual: api.DisplayValue{Value: 501}},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	row := resultRows(report.Results)[0]
	assert.Equal(t, "root 用户权限", row.Check)
	assert.Equal(t, "root（UID 0）", row.Expected)
	assert.Equal(t, "UID 501（非 root）", row.Actual)
	assert.Equal(t, "当前用户不是 root；使用 root 重新执行", row.CauseFix)
}

func TestTextAggregatesSourceProbeFailure(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{
		{InstanceID: "src-reachable@bootstrap", RuleID: "INIT-SRC-REACHABLE", Name: "Image repository is accessible (mirror.invalid.example:5443)", Status: api.StatusFail,
			Expected: api.DisplayValue{Value: true}, Actual: api.DisplayValue{}, CauseCode: api.FactErrUnreachable,
			Message: "EOF", Remediation: "verify the TLS certificate for the image repository is trusted"},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	row := resultRows(report.Results)[0]
	assert.Equal(t, "来源不可达；检查地址、DNS、代理和防火墙后重试", row.CauseFix)
	assert.NotContains(t, out, "verify the TLS certificate for the image repository is trusted")
}

func TestTextRendersKernelEvidenceInsteadOfBoolean(t *testing.T) {
	report := api.Report{Operation: api.OperationClusterCreate, Results: []api.Result{{
		RuleID: "CLUSTER-PLAT-KERNEL", Name: "Node kernel compatibility [uname -r]", Subject: "aio-1", Status: api.StatusFail,
		Expected: api.DisplayValue{Type: api.ValueTypeBool, Value: true}, Actual: api.DisplayValue{Type: api.ValueTypeBool, Value: false},
		Evidence:    []api.Evidence{{Kind: "command", Locator: "aio-1", Observed: "distribution=openeuler version=24.03 arch=arm64 kernel=7.0.14-orbstack"}},
		Remediation: "boot a node kernel covered by the release compatibility matrix",
	}}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	row := resultRows(report.Results)[0]
	assert.Equal(t, "受支持的内核", row.Expected)
	assert.Equal(t, "发行版=openeuler 版本=24.03 架构=arm64 内核=7.0.14-orbstack", row.Actual)
	assert.Equal(t, "内核不在版本兼容矩阵中；切换受支持内核后重试", row.CauseFix)
	assert.NotContains(t, out, " yes ")
}

func TestResultTargetShowsInvalidClusterYAMLFieldPath(t *testing.T) {
	result := api.Result{
		RuleID: "CLUSTER-MGMT-LOCAL-CLUSTER", Subject: "cluster", Status: api.StatusFail,
		Evidence: []api.Evidence{{
			Kind: "plan", Locator: "cluster.spec.clusterConfig.cluster.containerRuntime.cri",
			Observed: "The cri only support docker or containerd.",
		}},
	}
	row := resultRows([]api.Result{result})[0]
	assert.Contains(t, row.Target, "cluster.spec.clusterConfig.cluster.containerRuntime.cri")
	assert.Contains(t, row.CauseFix, "The cri only support docker or containerd")
}

func TestInputFieldLocatorOverridesGenericBindingTarget(t *testing.T) {
	result := api.Result{
		RuleID:  "CLUSTER-MGMT-LOCAL-CLUSTER",
		Name:    "BKECluster local validation [management cluster API/kubeconfig]",
		Subject: "cluster", Status: api.StatusFail,
		Evidence: []api.Evidence{{
			Kind: "plan", Locator: "cluster.spec.clusterConfig.cluster.networking.podSubnet",
			Observed: "Cluster podSubnet invalid CIDR address: not-a-cidr",
		}},
	}
	row := resultRows([]api.Result{result})[0]
	assert.Contains(t, row.Target, "cluster.spec.clusterConfig.cluster.networking.podSubnet")
	assert.NotContains(t, row.Target, "management cluster API/kubeconfig")
	assert.Contains(t, row.CauseFix, "not-a-cidr")
}

func TestTextGivesDependencySpecificRecoveryActions(t *testing.T) {
	tests := []struct {
		result api.Result
		want   string
	}{
		{api.Result{RuleID: "CLUSTER-SSH-TRANSPORT", Name: "SSH transport to node", Status: api.StatusFail,
			CauseCode: api.FactErrUnreachable, Message: "dial tcp 192.0.2.10:22: connection refused"}, "检查节点地址、端口、防火墙和 sshd"},
		{api.Result{RuleID: "CLUSTER-MGMT-AGENT-KUBECONFIG-GET", Name: "kube-system/localkubeconfig Secret is readable", Status: api.StatusFail,
			CauseCode: api.FactErrNotFound, Message: "Secret kube-system/localkubeconfig was not found"}, "创建 kube-system/localkubeconfig"},
	}
	for _, tt := range tests {
		rows := resultRows([]api.Result{tt.result})
		assert.Contains(t, rows[0].CauseFix, tt.want)
	}
}

func TestTextMakesBlockedChecksRepairable(t *testing.T) {
	report := api.Report{Operation: api.OperationClusterCreate, Results: []api.Result{{
		RuleID: "CLUSTER-RES-CPU", Name: "Node CPU capacity", Status: api.StatusSkip,
		CauseCode: api.CauseDependencyFailed, DependsOn: "CLUSTER-SSH-PRIVILEGE",
	}}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	assert.Equal(t, "前置依赖失败；修复根因后重试", resultRows(report.Results)[0].CauseFix)
}

func TestTextAppendsRerunWhenActionOnlyMentionsPolicy(t *testing.T) {
	report := api.Report{Operation: api.OperationInit, Results: []api.Result{{
		RuleID: "INIT-PLAT-DISTRO", Name: "Host Linux distribution", Status: api.StatusFail,
		Expected: api.DisplayValue{Value: []any{"openeuler"}}, Actual: api.DisplayValue{Value: []any{"debian"}},
		Remediation: "use a distribution and version listed in the preflight policy",
	}}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	assert.Equal(t, "发行版或版本不受支持；改用策略清单内的版本后重试", resultRows(report.Results)[0].CauseFix)
}

func TestTextTableAlignsUnicodeAndMakesTargetExplicit(t *testing.T) {
	report := api.Report{Operation: api.OperationClusterCreate, Results: []api.Result{
		{RuleID: "CLUSTER-SSH-TRANSPORT", Name: "SSH 连接 [[2001:db8:10:20::11]:22]", Subject: "master-1", Status: api.StatusFail,
			Expected: api.DisplayValue{Value: true}, Actual: api.DisplayValue{Value: false}, Remediation: "检查网络和 sshd"},
		{RuleID: "CLUSTER-PATH-WRITE", Name: "工作目录可写 [/var/lib/openfuyao/cluster-create/workspace]", Subject: "worker-1", Status: api.StatusFail,
			Expected: api.DisplayValue{Value: true}, Actual: api.DisplayValue{Value: false}, Remediation: "修正目录权限后重试"},
	}}
	var buf bytes.Buffer
	require.NoError(t, Text(&buf, report))
	out := buf.String()
	rows := resultRows(report.Results)
	assert.Equal(t, "节点：master-1\n[2001:db8:10:20::11]:22", rows[0].Target)
	assert.Equal(t, "节点：worker-1\n/var/lib/openfuyao/cluster-create/workspace", rows[1].Target)
	var tableWidths []int
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "│") {
			tableWidths = append(tableWidths, prettytext.StringWidthWithoutEscSequences(line))
		}
	}
	require.NotEmpty(t, tableWidths)
	for _, width := range tableWidths[1:] {
		assert.Equal(t, tableWidths[0], width)
	}
	assert.NotContains(t, out, "\x1b[")
}

func TestTableFillsRequestedWindowWidth(t *testing.T) {
	rows := []displayRow{{
		Status: "FAIL", Check: "工作目录可写", Target: "node: worker-1\n/var/lib/openfuyao/workspace",
		Expected: "可创建且可写", Actual: "权限不足", CauseFix: "目录权限不足；修正权限后重试",
	}}
	for _, width := range []int{80, 100, 140, 180, 220} {
		rendered := tableOutputAtWidth(rows, false, width)
		firstLine := strings.Split(rendered, "\n")[0]
		assert.Equal(t, width, prettytext.StringWidthWithoutEscSequences(firstLine), "window width %d", width)
	}
}

func TestTableDoesNotOverflowVeryNarrowWindow(t *testing.T) {
	for _, width := range []int{15, 60} {
		rendered := tableOutputAtWidth([]displayRow{{Status: "FAIL", Check: "检查", CauseFix: "修复后重试"}}, false, width)
		for _, line := range strings.Split(rendered, "\n") {
			assert.LessOrEqual(t, prettytext.StringWidthWithoutEscSequences(line), width, "width=%d line=%q", width, line)
		}
	}
}

func TestCurrentCatalogRulesHaveChineseOperatorText(t *testing.T) {
	rules := append(initadapter.CatalogRules(), clusteradapter.CatalogRules()...)
	for _, rule := range rules {
		row := resultRows([]api.Result{{RuleID: rule.ID, Name: rule.Name, Status: api.StatusFail,
			Expected: api.DisplayValue{Value: rule.Expected}, Actual: api.DisplayValue{Value: false},
			Remediation: rule.Fix}})[0]
		assert.True(t, containsHan(row.Check), "check name for %s: %q", rule.ID, row.Check)
		assert.True(t, containsHan(row.CauseFix), "repair text for %s: %q", rule.ID, row.CauseFix)
	}
}

func containsHan(value string) bool {
	for _, r := range value {
		if unicode.Is(unicode.Han, r) {
			return true
		}
	}
	return false
}

func TestColoredWrappedRowColorsEveryPhysicalLine(t *testing.T) {
	prettytext.EnableColors()
	defer prettytext.DisableColors()
	rendered := tableOutput(&bytes.Buffer{}, []displayRow{{
		Status: "FAIL", Check: "工作目录可写", Target: "node: worker-1\nobject: /var/lib/openfuyao/cluster-create/workspace",
		Expected: "可创建且可写", Actual: "权限不足", CauseFix: "目录权限不足；修正权限后重试",
	}}, true)
	coloredLines := 0
	for _, line := range strings.Split(rendered, "\n") {
		plain := prettytext.StripEscape(line)
		if strings.HasPrefix(plain, "│") && (strings.Contains(plain, "工作目录") || strings.Contains(plain, "workspace")) {
			coloredLines++
			assert.Contains(t, line, "\x1b[31m")
		}
	}
	assert.GreaterOrEqual(t, coloredLines, 2)
}
