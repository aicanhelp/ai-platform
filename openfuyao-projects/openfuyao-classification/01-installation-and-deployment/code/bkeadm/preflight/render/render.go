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

// Package render produces the human-readable and machine-readable preflight
// report. Both renderers consume the SAME api.Report; neither recomputes
// status, severity or overall. stdout is reserved for the report; logs and
// progress go to stderr (written by the host).
package render

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"

	"github.com/jedib0t/go-pretty/v6/table"
	"github.com/jedib0t/go-pretty/v6/text"
	"golang.org/x/term"

	"gopkg.openfuyao.cn/preflight/api"
)

const (
	detailedReportWidth = 180
	minimumTableWidth   = 72
	defaultWriterWidth  = 140

	tableFrameWidth      = 19
	columnWeightTotal    = 100
	checkColumnWeight    = 10
	targetColumnWeight   = 14
	expectedColumnWeight = 12
	actualColumnWeight   = 12

	targetContentOffset = 2
	targetClosingWidth  = 1
)

const (
	statusColumn = iota + 1
	checkColumn
	targetColumn
	expectedColumn
	actualColumn
	causeColumn
)

// Text renders a Report for installation operators. Internal rule IDs,
// collector/source identifiers and subject IDs are deliberately hidden from
// this default view; they remain available in the machine report and debug
// logs. The saved document is UTF-8 and never contains terminal color codes.
func Text(w io.Writer, r api.Report) error {
	r = orderedReport(r)
	var b strings.Builder
	fmt.Fprintf(&b, "openFuyao %s 前置检查报告\n", strings.ToLower(string(r.Operation)))
	writeRunOverview(&b, r)
	if len(r.Issues) > 0 {
		b.WriteString("\n运行问题\n")
		b.WriteString(tableOutputAtWidth(issueRows(r.Issues, false), false, detailedReportWidth))
	}
	if len(r.Results) > 0 {
		b.WriteString("\n检查结果\n")
		b.WriteString(tableOutputAtWidth(resultRows(r.Results), false, detailedReportWidth))
	}
	_, err := io.WriteString(w, b.String())
	return err
}

// Summary renders the operator-facing cluster-create overview. Detailed
// results remain in the report file; failures and warnings are shown so a
// large node result set is actionable from the terminal. SKIP is never
// rendered to the console.
func Summary(w io.Writer, r api.Report, detailPath string) error {
	r = orderedReport(r)
	var b strings.Builder
	fmt.Fprintf(&b, "openFuyao %s 前置检查\n", strings.ToLower(string(r.Operation)))
	writeRunOverview(&b, r)
	actionableIssues := visibleIssues(r.Issues)
	if len(actionableIssues) > 0 {
		b.WriteString("\n运行问题\n")
		b.WriteString(tableOutput(w, issueRows(actionableIssues, true), useColor(w)))
	}
	actionableResults := resultsWithStatus(r.Results, api.StatusFail, api.StatusWarn)
	if len(actionableResults) > 0 {
		b.WriteString("\n需要处理\n")
		b.WriteString(tableOutput(w, resultRows(actionableResults), useColor(w)))
	}
	if r.Summary.Fail == 0 && r.Summary.Warn == 0 && !api.HasErrorIssue(r.Issues) &&
		(r.Overall == api.OverallPass || r.Overall == api.OverallSkip) {
		b.WriteString("\n没有失败或警告项。\n")
	}
	if detailPath != "" {
		fmt.Fprintf(&b, "\n详细报告：%s\n", detailPath)
	}
	_, err := io.WriteString(w, b.String())
	return err
}

type displayRow struct {
	Status   string
	Check    string
	Target   string
	Expected string
	Actual   string
	CauseFix string
}

func resultRows(results []api.Result) []displayRow {
	rows := make([]displayRow, 0, len(results))
	for _, res := range results {
		name := res.Name
		if name == "" {
			name = friendlyName(res.RuleID)
		}
		check, object := splitNameTarget(name)
		check = stripDuplicatedObject(check, object)
		expected, actual := displayExpectedActual(res, check)
		cause, action := causeAndAction(res, check)
		check = localizedCheckName(res.RuleID, check)
		rows = append(rows, displayRow{
			Status: statusLabel(res.Status), Check: check,
			Target: resultTarget(res, object), Expected: expected, Actual: actual,
			CauseFix: compactCauseAction(res, cause, action),
		})
	}
	return rows
}

func issueRows(issues []api.RunIssue, concise bool) []displayRow {
	rows := make([]displayRow, 0, len(issues))
	for _, issue := range issues {
		status, expected, actual := issueDisplay(issue)
		target := "阶段：" + issueStageLabel(issue.Stage)
		if issue.Field != "" {
			target += "\n对象：" + localizedIssueField(issue.Field)
		}
		if locator := issueFileLocator(issue.Cause); locator != "" && !strings.Contains(target, locator) {
			target += "\n" + locator
		}
		causeFix := detailedIssueCauseAction(issue)
		if concise {
			causeFix = compactIssueCauseAction(issue)
		}
		rows = append(rows, displayRow{
			Status: status, Check: friendlyIssueName(issue.Code), Target: target,
			Expected: expected, Actual: actual,
			CauseFix: causeFix,
		})
	}
	return rows
}

func detailedIssueCauseAction(issue api.RunIssue) string {
	localized := compactIssueCauseAction(issue)
	raw := combineCauseAction(issue.Cause, issue.Action)
	if localized == raw || strings.TrimSpace(issue.Cause) == "" {
		return localized
	}
	return localized + "\n详情：" + oneLine(issue.Cause)
}

func tableOutput(w io.Writer, rows []displayRow, color bool) string {
	return tableOutputAtWidth(rows, color, writerWidth(w))
}

func tableOutputAtWidth(rows []displayRow, color bool, width int) string {
	if width > 0 && width < minimumTableWidth {
		return displayNarrowRows(rows, color, width)
	}
	t := table.NewWriter()
	t.SetStyle(table.StyleLight)
	t.Style().Options.SeparateRows = true
	t.AppendHeader(rowValues("状态", "检查项", "目标", "期望", "实际", "原因及建议"))
	for _, row := range rows {
		t.AppendRow(rowValues(row.Status, row.Check, row.Target, row.Expected, row.Actual, row.CauseFix))
	}
	t.SetColumnConfigs(columnConfigs(width))
	if color {
		t.SetRowPainter(func(row table.Row) text.Colors {
			if len(row) == 0 {
				return nil
			}
			return rowColors(fmt.Sprint(row[0]))
		})
	}
	return t.Render() + "\n"
}

func rowValues(values ...any) table.Row {
	return table.Row(values)
}

func displayNarrowRows(rows []displayRow, color bool, width int) string {
	if width < 1 {
		width = 1
	}
	var b strings.Builder
	for i, row := range rows {
		if i > 0 {
			b.WriteByte('\n')
		}
		fields := [...]string{
			"[" + row.Status + "] " + row.Check,
			"目标：" + row.Target,
			"期望：" + row.Expected,
			"实际：" + row.Actual,
			"原因及建议：" + row.CauseFix,
		}
		for _, field := range fields {
			writeNarrowField(&b, field, row.Status, color, width)
		}
	}
	return b.String()
}

func writeNarrowField(b *strings.Builder, field, status string, color bool, width int) {
	// go-pretty's wrapper can exceed the requested display width by one cell
	// when a line mixes ASCII and wide CJK runes. Reserve one cell in the
	// narrow fallback so the terminal never performs a second wrap.
	wrapWidth := width
	if wrapWidth > 1 {
		wrapWidth--
	}
	for _, wrappedLine := range strings.Split(text.WrapText(field, wrapWidth), "\n") {
		line := wrappedLine
		if color {
			line = rowColors(status).Sprint(line)
		}
		b.WriteString(line)
		b.WriteByte('\n')
	}
}

func rowColors(status string) text.Colors {
	switch status {
	case "失败", "阻塞", "错误", "中断", string(api.StatusFail), "BLOCK", "ERROR", "STOP":
		return colorSet(text.FgRed)
	case "警告", string(api.StatusWarn):
		return colorSet(text.FgYellow)
	case "通过", string(api.StatusPass):
		return colorSet(text.FgGreen)
	default:
		return colorSet(text.FgHiBlack)
	}
}

func colorSet(color text.Color) text.Colors {
	return append(text.Colors(nil), color)
}

func columnConfigs(width int) []table.ColumnConfig {
	statusWidth, checkWidth, targetWidth := 6, 7, 9
	expectedWidth, actualWidth, causeWidth := 7, 7, 17

	// A six-column table has 19 non-content cells: seven borders and two
	// padding cells per column. Assign every remaining terminal cell so the
	// table fills the current window instead of shrinking to natural content.
	extra := width - tableFrameWidth -
		(statusWidth + checkWidth + targetWidth + expectedWidth + actualWidth + causeWidth)
	checkExtra := extra * checkColumnWeight / columnWeightTotal
	targetExtra := extra * targetColumnWeight / columnWeightTotal
	expectedExtra := extra * expectedColumnWeight / columnWeightTotal
	actualExtra := extra * actualColumnWeight / columnWeightTotal
	checkWidth += checkExtra
	targetWidth += targetExtra
	expectedWidth += expectedExtra
	actualWidth += actualExtra
	causeWidth += extra - checkExtra - targetExtra - expectedExtra - actualExtra

	return []table.ColumnConfig{
		fixedColumn(statusColumn, statusWidth, text.WrapSoft, text.AlignCenter),
		fixedColumn(checkColumn, checkWidth, text.WrapSoft, text.AlignDefault),
		fixedColumn(targetColumn, targetWidth, text.WrapText, text.AlignDefault),
		fixedColumn(expectedColumn, expectedWidth, text.WrapSoft, text.AlignDefault),
		fixedColumn(actualColumn, actualWidth, text.WrapSoft, text.AlignDefault),
		fixedColumn(causeColumn, causeWidth, text.WrapSoft, text.AlignDefault),
	}
}

func fixedColumn(number, width int, enforcer table.WidthEnforcer, align text.Align) table.ColumnConfig {
	return table.ColumnConfig{
		Number: number, WidthMin: width, WidthMax: width,
		Align: align, AlignHeader: align, VAlign: text.VAlignTop, WidthMaxEnforcer: enforcer,
	}
}

func writerWidth(w io.Writer) int {
	f, ok := w.(*os.File)
	if !ok {
		return defaultWriterWidth
	}
	width, _, err := term.GetSize(int(f.Fd()))
	if err != nil {
		return defaultWriterWidth
	}
	return width
}

func useColor(w io.Writer) bool {
	if noColor := os.Getenv("NO_COLOR"); noColor != "" && noColor != "0" {
		return false
	}
	f, ok := w.(*os.File)
	return ok && term.IsTerminal(int(f.Fd()))
}

func resultsWithStatus(results []api.Result, statuses ...api.Status) []api.Result {
	wanted := make(map[api.Status]struct{}, len(statuses))
	for _, status := range statuses {
		wanted[status] = struct{}{}
	}
	out := make([]api.Result, 0, len(results))
	for _, result := range results {
		if _, ok := wanted[result.Status]; ok {
			out = append(out, result)
		}
	}
	return out
}

func visibleIssues(issues []api.RunIssue) []api.RunIssue {
	out := make([]api.RunIssue, 0, len(issues))
	for _, issue := range issues {
		if issue.Severity == api.IssueError || issue.Severity == api.IssueWarn {
			out = append(out, issue)
		}
	}
	return out
}

func splitNameTarget(name string) (string, string) {
	name = oneLine(name)
	if !strings.HasSuffix(name, "]") {
		return name, ""
	}
	if start := strings.LastIndex(name, " ["); start >= 0 {
		return strings.TrimSpace(name[:start]), strings.TrimSpace(
			name[start+targetContentOffset : len(name)-targetClosingWidth],
		)
	}
	return name, ""
}

func statusLabel(status api.Status) string {
	switch status {
	case api.StatusPass:
		return "通过"
	case api.StatusFail:
		return "失败"
	case api.StatusWarn:
		return "警告"
	case api.StatusSkip:
		return "跳过"
	default:
		return string(status)
	}
}

var localizedCheckNames = map[string]string{
	"INIT-EXEC-001": "root 用户权限", "INIT-PLAT-OS": "主机操作系统",
	"INIT-PLAT-DISTRO": "主机发行版及版本", "INIT-PLAT-KERNEL": "内核兼容性",
	"INIT-PLAT-ARCH": "主机架构", "INIT-RES-CPU": "主机 CPU 资源",
	"INIT-RES-MEMORY": "主机内存资源", "INIT-RES-SYSTEM-DISK": "主机系统盘容量",
	"INIT-TOOL-TAR": "tar 命令", "INIT-TOOL-SYSTEMCTL": "systemctl 命令",
	"INIT-PATH-CREATE": "工作目录可创建", "INIT-PATH-MOUNT": "工作目录可写",
	"INIT-PORT-K8S": "Kubernetes API 端口", "INIT-PORT-REGISTRY": "镜像仓库端口",
	"INIT-PORT-CHART": "Chart 仓库端口", "INIT-PORT-HTTP": "软件包仓库端口",
	"INIT-PORT-CONSOLE": "BKE Console 端口", "INIT-PORT-NTP": "本地 NTP 端口",
	"INIT-REPO-URL": "自定义镜像仓库配置", "INIT-REPO-CA-EXISTS": "仓库 CA 文件存在",
	"INIT-REPO-CA-READ": "仓库 CA 文件可读", "INIT-REPO-CA-PARSE": "仓库 CA 文件有效",
	"INIT-OFFLINE-EXISTS": "离线包存在", "INIT-OFFLINE-READ": "离线包可读",
	"INIT-OFFLINE-FORMAT": "离线包格式", "INIT-COLOCATE": "共部署风险",
	"CLUSTER-UNIQUE-IP": "节点 IP 唯一", "CLUSTER-UNIQUE-HOSTNAME": "非空主机名唯一",
	"CLUSTER-TOPO-MASTER": "Master 节点数量", "CLUSTER-TOPO-MASTER-ODD": "Master 数量为奇数",
	"CLUSTER-TOPO-ETCD": "etcd 节点数量", "CLUSTER-TOPO-WORKER": "Worker 节点数量",
	"CLUSTER-MGMT-API": "管理集群 API 连通性", "CLUSTER-MGMT-DISCOVERY": "管理集群 API 发现",
	"CLUSTER-MGMT-CRD-CLUSTER": "管理集群 BKECluster CRD", "CLUSTER-MGMT-CRD-NODE": "管理集群 BKENode CRD",
	"CLUSTER-MGMT-AGENT-KUBECONFIG-GET":    "localkubeconfig Secret 可读",
	"CLUSTER-MGMT-AGENT-KUBECONFIG-DATA":   "localkubeconfig 数据非空",
	"CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE":  "localkubeconfig 格式有效",
	"CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE": "localkubeconfig 服务地址",
	"CLUSTER-MGMT-LOCAL-CLUSTER":           "cluster.yaml 本地校验",
	"CLUSTER-MGMT-LOCAL-NODE":              "nodes.yaml 本地校验",
	"CLUSTER-LOCAL-PATH-CREATE":            "集群输出目录可创建",
	"CLUSTER-LOCAL-PATH-ACCESS":            "集群输出目录可访问",
	"CLUSTER-LOCAL-PATH-MOUNT":             "集群输出目录可写", "CLUSTER-NET-POD-CIDR": "Pod CIDR 配置",
	"CLUSTER-NET-SERVICE-CIDR": "Service CIDR 配置", "CLUSTER-HA-COLOCATE": "HA 节点共部署风险",
	"CLUSTER-REF-VERSION": "集群配置版本", "CLUSTER-ADDON-CLOSURE": "Addon 依赖完整性",
	"CLUSTER-ADDON-SOURCE": "Addon 来源配置", "CLUSTER-SSH-TRANSPORT": "SSH 连通性",
	"CLUSTER-SSH-AUTH": "SSH 认证", "CLUSTER-SSH-PRIVILEGE": "节点 root 权限",
	"CLUSTER-AGENT-SYSTEMD": "节点 systemd", "CLUSTER-RES-CPU": "节点 CPU 资源",
	"CLUSTER-RES-MEMORY": "节点内存资源", "CLUSTER-RES-SYSTEM-DISK": "节点系统盘容量",
	"CLUSTER-PLAT-OS": "节点操作系统", "CLUSTER-PLAT-DISTRO": "节点发行版及版本",
	"CLUSTER-PLAT-KERNEL": "节点内核兼容性", "CLUSTER-PORT-APISERVER": "Kubernetes API 端口",
	"CLUSTER-PORT-ETCD-CLIENT": "etcd 客户端端口", "CLUSTER-PORT-ETCD-PEER": "etcd Peer 端口",
	"CLUSTER-PORT-ETCD-METRICS": "etcd 指标端口", "CLUSTER-PORT-KUBELET-SECURE": "Kubelet API 端口",
	"CLUSTER-PORT-KUBELET-HEALTH": "Kubelet 健康端口", "CLUSTER-PORT-CONTROLLER": "Controller Manager 端口",
	"CLUSTER-PORT-SCHEDULER": "Scheduler 端口", "CLUSTER-PORT-LOCAL-HA": "本地 HA 端口",
	"CLUSTER-PORT-BKEAGENT": "BKE Agent 端口",
}

func localizedCheckName(ruleID, fallback string) string {
	if name, ok := localizedCheckNames[ruleID]; ok {
		return name
	}
	if ruleID == "INIT-SRC-REACHABLE" || ruleID == "CLUSTER-SRC-REACHABLE" {
		lower := strings.ToLower(fallback)
		switch {
		case strings.Contains(lower, "online image"):
			return "在线镜像可达性"
		case strings.Contains(lower, "image repo"):
			return "镜像仓库可达性"
		case strings.Contains(lower, "package repo"), strings.Contains(lower, "yum repo"):
			return "软件仓库可达性"
		case strings.Contains(lower, "chart repo"):
			return "Chart 来源可达"
		default:
			return "安装来源可达性"
		}
	}
	return fallback
}

func stripDuplicatedObject(check, object string) string {
	if object == "" || !strings.HasSuffix(check, ")") {
		return check
	}
	start := strings.LastIndex(check, " (")
	if start < 0 {
		return check
	}
	label := strings.TrimSpace(check[start+targetContentOffset : len(check)-targetClosingWidth])
	normalizedObject := strings.TrimPrefix(strings.TrimPrefix(object, "https://"), "http://")
	if (label != "" && strings.Contains(normalizedObject, label)) || strings.HasPrefix(label, "--") {
		return strings.TrimSpace(check[:start])
	}
	return check
}

func resultTarget(res api.Result, object string) string {
	if field := inputFieldLocator(res.Evidence); field != "" {
		object = inputFieldTarget(field)
	}
	if object == "" {
		for _, evidence := range res.Evidence {
			if locator := oneLine(evidence.Locator); locator != "" && locator != res.Subject {
				object = locator
				break
			}
		}
	}
	subject := valueOrDash(res.Subject)
	prefix := "节点："
	switch subject {
	case "cluster":
		prefix = "范围："
		subject = "集群"
	case "bootstrap":
		subject = "bootstrap\n位置：本机"
	default:
		prefix = "节点："
	}
	target := prefix + subject
	if object != "" && object != subject {
		target += "\n" + object
	}
	return target
}

func inputFieldTarget(field string) string {
	leaf := field
	if index := strings.LastIndex(field, "."); index >= 0 && index+1 < len(field) {
		leaf = field[index+1:]
	}
	return "字段：" + leaf + "\n路径：" + field
}

func inputFieldLocator(evidence []api.Evidence) string {
	for _, item := range evidence {
		locator := oneLine(item.Locator)
		if strings.HasPrefix(locator, "cluster.") || strings.HasPrefix(locator, "nodes[") {
			return locator
		}
	}
	return ""
}

func localizedIssueField(field string) string {
	switch field {
	case "cluster":
		return "cluster.yaml"
	case "nodes":
		return "nodes.yaml"
	case "clusterConfig":
		return "spec.clusterConfig"
	default:
		return field
	}
}

func combineCauseAction(cause, action string) string {
	cause = strings.TrimRight(oneLine(cause), ".; ")
	action = strings.TrimRight(oneLine(action), ".; ")
	switch {
	case cause == "" && action == "":
		return "-"
	case cause == "":
		return action
	case action == "":
		return cause
	default:
		return cause + "; " + action
	}
}

func compactCauseAction(res api.Result, cause, action string) string {
	switch {
	case res.Status == api.StatusPass:
		return "-"
	case res.RuleID == "INIT-EXEC-001":
		return "当前用户不是 root；使用 root 重新执行"
	case isKernelCompatibilityRule(res.RuleID):
		return "内核不在版本兼容矩阵中；切换受支持内核后重试"
	case isSourceAccessCheck(res.Name):
		return "来源不可达；检查地址、DNS、代理和防火墙后重试"
	case res.RuleID == "CLUSTER-SSH-TRANSPORT":
		return "SSH 不可达；检查节点地址、端口、防火墙和 sshd 后重试"
	case res.RuleID == "CLUSTER-SSH-AUTH":
		return "SSH 认证失败；检查用户名和凭据后重试"
	case res.RuleID == "CLUSTER-SSH-PRIVILEGE":
		return "SSH 权限不足；配置 root 或免密 sudo 后重试"
	case res.CauseCode == api.CauseDependencyFailed:
		return "前置依赖失败；修复根因后重试"
	default:
		return fallbackResultRepair(res, cause, action)
	}

}

func fallbackResultRepair(res api.Result, cause, action string) string {
	if localized := localizedRuleRepair(res); localized != "" {
		return localized
	}
	cause = strings.TrimSpace(cause)
	if cause == "The observed value does not meet the requirement." {
		cause = ""
	}
	action = strings.ReplaceAll(action, "then run preflight again", "then rerun")
	action = strings.ReplaceAll(action, "run preflight again", "rerun")
	return combineCauseAction(cause, action)
}

func localizedRuleRepair(res api.Result) string {
	ruleID := res.RuleID
	expected := formatValue(ruleID, res.Name, res.Expected)
	if message := localizedInputValidationRepair(res); message != "" {
		return message
	}
	if message := localizedResourceRepair(ruleID, expected); message != "" {
		return message
	}
	if message := localizedHostRepair(ruleID); message != "" {
		return message
	}
	if message := localizedTopologyRepair(ruleID); message != "" {
		return message
	}
	if message := localizedManagementRepair(ruleID); message != "" {
		return message
	}
	return localizedClusterConfigRepair(ruleID)
}

func localizedInputValidationRepair(res api.Result) string {
	observed := evidenceObserved(res.Evidence)
	switch res.RuleID {
	case "CLUSTER-MGMT-LOCAL-CLUSTER":
		if observed != "" {
			return "BKECluster 字段校验失败：" + observed + "；修正目标字段后重试"
		}
	case "CLUSTER-MGMT-LOCAL-NODE":
		if observed != "" {
			return "BKENode 字段校验失败：" + observed + "；修正目标字段后重试"
		}
	case "CLUSTER-NET-POD-CIDR":
		if observed != "" {
			return "Pod CIDR 值 " + observed + " 无效；修正目标字段后重试"
		}
	case "CLUSTER-NET-SERVICE-CIDR":
		if observed != "" {
			return "Service CIDR 值 " + observed + " 无效；修正目标字段后重试"
		}
	default:
		return ""
	}
	return ""
}

func localizedResourceRepair(ruleID, expected string) string {
	if strings.HasPrefix(ruleID, "INIT-PORT-") || strings.HasPrefix(ruleID, "CLUSTER-PORT-") {
		return "端口已占用或不可用；释放目标端口或修改对应端口配置后重试"
	}
	switch ruleID {
	case "INIT-RES-CPU":
		return "CPU 不足；将主机逻辑 CPU 增加到至少 " + expected + " 核后重试"
	case "CLUSTER-RES-CPU":
		return "CPU 不足；将节点逻辑 CPU 增加到至少 " + expected + " 核后重试"
	case "INIT-RES-MEMORY":
		return "内存不足；将主机内存增加到至少 " + expected + " 后重试"
	case "CLUSTER-RES-MEMORY":
		return "内存不足；将节点内存增加到至少 " + expected + " 后重试"
	case "INIT-RES-SYSTEM-DISK", "CLUSTER-RES-SYSTEM-DISK":
		return "系统盘容量不足；扩容到至少 " + expected + " 后重试"
	default:
		return ""
	}
}

func localizedHostRepair(ruleID string) string {
	switch ruleID {
	case "INIT-PLAT-OS", "CLUSTER-PLAT-OS":
		return "操作系统不受支持；改用受支持的 Linux 系统后重试"
	case "INIT-PLAT-DISTRO", "CLUSTER-PLAT-DISTRO":
		return "发行版或版本不受支持；改用策略清单内的版本后重试"
	case "INIT-PLAT-ARCH":
		return "主机架构不受支持；改用 amd64 或 arm64 后重试"
	case "INIT-TOOL-TAR":
		return "tar 不可执行；安装 tar 并加入 PATH 后重试"
	case "INIT-TOOL-SYSTEMCTL", "CLUSTER-AGENT-SYSTEMD":
		return "systemd 工具不可用；安装并启用 systemd 后重试"
	case "INIT-PATH-CREATE", "CLUSTER-LOCAL-PATH-CREATE":
		return "目录不可创建；修正父目录权限或更换工作目录后重试"
	case "INIT-PATH-MOUNT", "CLUSTER-LOCAL-PATH-MOUNT":
		return "目录所在文件系统只读；重新以读写方式挂载或更换目录后重试"
	case "CLUSTER-LOCAL-PATH-ACCESS":
		return "目录不可访问；修正工作目录权限后重试"
	case "INIT-OFFLINE-EXISTS":
		return "离线包不存在；确认文件路径并补齐离线包后重试"
	case "INIT-OFFLINE-READ":
		return "离线包不可读；修正文件权限后重试"
	case "INIT-OFFLINE-FORMAT":
		return "离线包格式不受支持；改用 tar、tar.gz 或 tgz 后重试"
	case "INIT-REPO-URL":
		return "镜像仓库配置无效；修正 --otherRepo 后重试"
	case "INIT-REPO-CA-EXISTS":
		return "CA 文件不存在；修正 --imageRepoCAFile 后重试"
	case "INIT-REPO-CA-READ":
		return "CA 文件不可读；修正文件权限后重试"
	case "INIT-REPO-CA-PARSE":
		return "CA 文件不是有效 PEM 证书；替换证书后重试"
	case "INIT-COLOCATE", "CLUSTER-HA-COLOCATE":
		return "存在共部署风险；确认容量和故障域满足要求"
	default:
		return ""
	}
}

func localizedTopologyRepair(ruleID string) string {
	switch ruleID {
	case "CLUSTER-UNIQUE-IP":
		return "节点 IP 重复；为每个节点配置唯一的 spec.ip 后重试"
	case "CLUSTER-UNIQUE-HOSTNAME":
		return "非空主机名重复；为已配置主机名的节点设置唯一值后重试"
	case "CLUSTER-TOPO-MASTER":
		return "缺少 Master 节点；在 nodes.yaml 中增加 Master 角色后重试"
	case "CLUSTER-TOPO-MASTER-ODD":
		return "Master 数量不是奇数；调整为 1、3、5 等奇数后重试"
	case "CLUSTER-TOPO-ETCD":
		return "缺少 etcd 节点；在 nodes.yaml 中增加 etcd 角色后重试"
	case "CLUSTER-TOPO-WORKER":
		return "缺少 Worker 节点；在 nodes.yaml 中增加 Worker 角色后重试"
	default:
		return ""
	}
}

func localizedManagementRepair(ruleID string) string {
	switch ruleID {
	case "CLUSTER-MGMT-API":
		return "管理集群 API 不可达；检查 --kubeconfig、网络和权限后重试"
	case "CLUSTER-MGMT-DISCOVERY":
		return "API 发现失败；检查凭据和 Discovery 权限后重试"
	case "CLUSTER-MGMT-CRD-CLUSTER":
		return "缺少 BKECluster CRD；在管理集群安装 CRD 后重试"
	case "CLUSTER-MGMT-CRD-NODE":
		return "缺少 BKENode CRD；在管理集群安装 CRD 后重试"
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-GET":
		return "Secret 不存在或不可读；创建 kube-system/localkubeconfig 并授权后重试"
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-DATA":
		return "Secret 数据为空；写入预期的 kubeconfig 数据后重试"
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-PARSE":
		return "kubeconfig 格式无效；替换为有效配置后重试"
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-SOURCE":
		return "kubeconfig 缺少服务地址；修正 server 字段后重试"
	case "CLUSTER-MGMT-LOCAL-CLUSTER":
		return "BKECluster 本地校验失败；修正 cluster.yaml 对应字段后重试"
	case "CLUSTER-MGMT-LOCAL-NODE":
		return "BKENode 本地校验失败；修正 nodes.yaml 对应字段后重试"
	default:
		return ""
	}
}

func localizedClusterConfigRepair(ruleID string) string {
	switch ruleID {
	case "CLUSTER-NET-POD-CIDR":
		return "Pod CIDR 无效；修正 cluster.yaml 网络配置后重试"
	case "CLUSTER-NET-SERVICE-CIDR":
		return "Service CIDR 无效；修正 cluster.yaml 网络配置后重试"
	case "CLUSTER-REF-VERSION":
		return "配置版本不受支持；改用受支持版本后重试"
	case "CLUSTER-ADDON-CLOSURE":
		return "Addon 依赖不完整；补齐依赖或禁用对应 Addon 后重试"
	case "CLUSTER-ADDON-SOURCE":
		return "Addon 来源配置不完整；补齐 Chart 仓库配置后重试"
	default:
		return ""
	}
}

func issueDisplay(issue api.RunIssue) (status, expected, actual string) {
	switch issue.Effect {
	case api.IssueBlockStart:
		return "阻塞", "检查可启动", "未启动"
	case api.IssueInterrupt:
		return "中断", "检查可完成", "已中断"
	default:
		if issue.Severity == api.IssueInfo {
			return "提示", "-", "信息"
		}
		if issue.Severity == api.IssueWarn {
			return "警告", "准备成功", "降级"
		}
		return "错误", "准备成功", "失败"
	}
}

func friendlyIssueName(code string) string {
	if strings.Contains(code, "OPTIONS") {
		return "命令参数"
	}
	if strings.Contains(code, "TIMEOUT") {
		return "检查超时"
	}
	if strings.Contains(code, "CANCELED") {
		return "检查已取消"
	}
	if strings.Contains(code, "BUNDLE") {
		return "检查策略包"
	}
	if strings.Contains(code, "PROVIDER-DEFAULTS") {
		return "Provider 默认配置"
	}
	if strings.Contains(code, "SCHEMA") || strings.Contains(code, "INPUT") || strings.Contains(code, "REQUIRED") {
		return "输入校验"
	}
	return "检查准备"
}

func issueStageLabel(stage api.IssueStage) string {
	switch stage {
	case api.IssueStageInput:
		return "输入"
	case api.IssueStagePlan:
		return "计划"
	case api.IssueStageSetup:
		return "准备"
	case api.IssueStageExecution:
		return "执行"
	default:
		return strings.ToLower(string(stage))
	}
}

func compactIssueCauseAction(issue api.RunIssue) string {
	if issueFileLocator(issue.Cause) != "" && strings.Contains(issue.Cause, "no such file or directory") {
		return "文件不存在；确认目标路径并补齐文件后重试"
	}
	switch issue.Code {
	case "PREFLIGHT-OPTIONS":
		if message := compactOptionIssue(issue.Field); message != "" {
			return message
		}
		return compactGenericIssue(issue)
	case "PREFLIGHT-TIMEOUT":
		return "前置检查超时；恢复管理集群连接或增大 --timeout 后重试"
	case "PREFLIGHT-CANCELED":
		return "前置检查被取消；环境可用后重新执行"
	case "PREFLIGHT-CONFIG-INVALID":
		return "前置检查配置无效；修正 --preflight-config 指定的 YAML 后重试"
	case "BUNDLE-INVALID":
		return "检查策略包无效；恢复受信任的策略包后重试"
	default:
		return compactGenericIssue(issue)
	}
}

func compactGenericIssue(issue api.RunIssue) string {
	if strings.Contains(issue.Code, "INPUT-SCHEMA") || strings.Contains(issue.Code, "INPUT-INVALID") {
		return "输入格式或字段无效；修正对应配置后重试"
	}
	if strings.Contains(issue.Code, "INPUT-REQUIRED") {
		return "缺少必填输入；补齐对应配置后重试"
	}
	if strings.Contains(issue.Code, "PLAN-HASH") || strings.Contains(issue.Code, "CONFIG-CONDITION") {
		return "无法生成检查计划；修正对应配置条件后重试"
	}
	return combineCauseAction(issue.Cause, issue.Action)
}

func compactOptionIssue(field string) string {
	switch field {
	case "timeout":
		return "--timeout 必须大于 0；设置有效总超时后重试"
	case "fact-timeout":
		return "--fact-timeout 必须大于 0；设置有效单项超时后重试"
	case "output":
		return "输出格式无效；将 --output 设置为 text 或 json 后重试"
	case "max-concurrency":
		return "并发数超出范围；将 --max-concurrency 设置为 0 到 64 后重试"
	default:
		return "命令参数无效；修正参数后重试"
	}
}

func issueFileLocator(cause string) string {
	for _, marker := range []string{"read cluster file ", "read nodes file ", "stat request file ", "read request file "} {
		start := strings.Index(cause, marker)
		if start < 0 {
			continue
		}
		remainder := cause[start+len(marker):]
		if end := strings.Index(remainder, ": "); end > 0 {
			return remainder[:end]
		}
	}
	return ""
}

func writeRunOverview(b *strings.Builder, r api.Report) {
	blockers, interruptions := 0, 0
	for _, issue := range r.Issues {
		switch issue.Effect {
		case api.IssueBlockStart:
			blockers++
		case api.IssueInterrupt:
			interruptions++
		default:
			continue
		}
	}
	switch {
	case blockers > 0:
		fmt.Fprintf(
			b,
			"状态：未启动\n原因：%d 个启动阻塞项导致检查计划无法生成\n"+
				"检查：未启动（已执行 0 项）\n",
			blockers,
		)
	case interruptions > 0:
		fmt.Fprintf(b, "状态：未完成\n原因：%d 个运行问题中断了检查\n", interruptions)
		writeCounts(b, r.Summary)
	default:
		fmt.Fprintf(b, "结论：%s\n", overallLabel(r.Overall))
		writeCounts(b, r.Summary)
	}
}

func displayExpectedActual(res api.Result, name string) (string, string) {
	if isKernelCompatibilityRule(res.RuleID) {
		expected := "受支持的内核"
		if observed := evidenceObserved(res.Evidence); observed != "" {
			return expected, localizedKernelObservation(observed)
		}
		return expected, formatValue(res.RuleID, name, res.Actual)
	}
	return formatValue(res.RuleID, name, res.Expected), formatValue(res.RuleID, name, res.Actual)
}

func localizedKernelObservation(observed string) string {
	replacer := strings.NewReplacer(
		"distribution=", "发行版=", "version=", "版本=",
		"arch=", "架构=", "kernel=", "内核=",
	)
	return replacer.Replace(observed)
}

func isKernelCompatibilityRule(ruleID string) bool {
	return ruleID == "INIT-PLAT-KERNEL" || ruleID == "CLUSTER-PLAT-KERNEL"
}

func evidenceObserved(evidence []api.Evidence) string {
	for _, item := range evidence {
		if observed := oneLine(item.Observed); observed != "" {
			return observed
		}
	}
	return ""
}

func writeCounts(b *strings.Builder, summary api.Summary) {
	applicable := summary.Pass + summary.Fail + summary.Warn
	fmt.Fprintf(b, "适用：%d  通过：%d  失败：%d  警告：%d\n",
		applicable, summary.Pass, summary.Fail, summary.Warn)
	if summary.Skip > 0 {
		fmt.Fprintf(b, "依赖阻塞或跳过：%d（详见报告）\n", summary.Skip)
	}
}

// JSON renders a Report as canonical JSON to w. The structure is the api.Report
// itself; no re-computation of status/overall. HTML escaping is disabled for
// readability.
func JSON(w io.Writer, r api.Report) error {
	r = orderedReport(r)
	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)
	enc.SetIndent("", "  ")
	return enc.Encode(r)
}

func orderedReport(r api.Report) api.Report {
	r.Results = append([]api.Result(nil), r.Results...)
	sort.SliceStable(r.Results, func(i, j int) bool {
		pi, pj := statusPriority(r.Results[i].Status), statusPriority(r.Results[j].Status)
		if pi != pj {
			return pi < pj
		}
		if r.Results[i].RuleID != r.Results[j].RuleID {
			return r.Results[i].RuleID < r.Results[j].RuleID
		}
		if r.Results[i].Subject != r.Results[j].Subject {
			return r.Results[i].Subject < r.Results[j].Subject
		}
		return r.Results[i].InstanceID < r.Results[j].InstanceID
	})
	return r
}

func statusPriority(status api.Status) int {
	const (
		failPriority = iota
		warnPriority
		passPriority
		skipPriority
		unknownPriority
	)
	switch status {
	case api.StatusFail:
		return failPriority
	case api.StatusWarn:
		return warnPriority
	case api.StatusPass:
		return passPriority
	case api.StatusSkip:
		return skipPriority
	default:
		return unknownPriority
	}
}

func overallLabel(o api.Overall) string {
	switch o {
	case api.OverallPass:
		return "通过（可以继续）"
	case api.OverallWarn:
		return "警告（可以继续）"
	case api.OverallFail:
		return "失败（已阻塞）"
	case api.OverallSkip:
		return "跳过（没有适用检查）"
	case api.OverallInputError:
		return "未启动（输入无效）"
	case api.OverallBundleError:
		return "未启动（检查策略无效）"
	case api.OverallInternalError:
		return "未启动（内部错误）"
	case api.OverallTimedOut:
		return "未完成（检查超时）"
	case api.OverallCanceled:
		return "未完成（检查已取消）"
	default:
		return string(o)
	}
}

func formatValue(ruleID, name string, v api.DisplayValue) string {
	if v.Value == nil {
		return "-"
	}
	if ruleID == "INIT-EXEC-001" {
		if n, ok := number(v.Value); ok {
			if n == 0 {
				return "root（UID 0）"
			}
			return fmt.Sprintf("UID %.0f（非 root）", n)
		}
	}
	if v.Type == api.ValueTypeBytes || isByteValue(ruleID, name) {
		if n, ok := number(v.Value); ok {
			return formatBytes(n)
		}
	}
	if value, ok := v.Value.(bool); ok {
		return formatBoolValue(name, value)
	}
	if formatted, ok := formatCompositeValue(v.Value); ok {
		return formatted
	}
	if value, ok := v.Value.(string); ok {
		switch value {
		case "linux":
			return "Linux"
		case "not-colocated":
			return "未共部署"
		case "same-node":
			return "同一节点"
		default:
			return fmt.Sprintf("%v", v.Value)
		}
	}
	return fmt.Sprintf("%v", v.Value)
}

func formatBoolValue(name string, value bool) string {
	if strings.Contains(strings.ToLower(name), "port") {
		if value {
			return "可用"
		}
		return "已占用或不可用"
	}
	if value {
		return "是"
	}
	return "否"
}

func formatCompositeValue(value any) (string, bool) {
	switch typed := value.(type) {
	case []any:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			parts = append(parts, fmt.Sprint(item))
		}
		return strings.Join(parts, ", "), true
	case map[string]any:
		encoded, err := json.Marshal(typed)
		return string(encoded), err == nil
	default:
		return "", false
	}
}

func isByteValue(ruleID, name string) bool {
	// A rule name containing "memory", "capacity" or "bytes" is not enough
	// to infer units: kernel counters such as vm.overcommit_memory and
	// kernel.keys.root_maxbytes are integers. Typed DisplayValue values remain
	// authoritative; these ID hints cover the legacy engine results that do
	// not yet preserve the fact type in Result.Actual.
	switch strings.ToUpper(ruleID) {
	case "INIT-RES-MEMORY", "INIT-RES-SYSTEM-DISK", "CLUSTER-RES-MEMORY",
		"CLUSTER-RES-SYSTEM-DISK":
		return true
	default:
		return false
	}
}

func number(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case int32:
		return float64(n), true
	case uint64:
		return float64(n), true
	case float64:
		return n, true
	default:
		return 0, false
	}
}

func formatBytes(n float64) string {
	if n < 0 {
		return strconv.FormatFloat(n, 'f', -1, 64) + " B"
	}
	units := [...]string{"B", "KiB", "MiB", "GiB", "TiB", "PiB"}
	value := n
	unit := 0
	for value >= 1024 && unit < len(units)-1 {
		value /= 1024
		unit++
	}
	if unit == 0 {
		return fmt.Sprintf("%.0f %s", value, units[unit])
	}
	formatted := strconv.FormatFloat(value, 'f', 1, 64)
	formatted = strings.TrimSuffix(formatted, ".0")
	return formatted + " " + units[unit]
}

func causeAndAction(res api.Result, name string) (string, string) {
	if res.Status == api.StatusPass {
		return "", ""
	}
	message := resultCause(res, name)
	recommendation := resultRecommendation(res, name)
	return oneLine(message), oneLine(recommendation)
}

func resultCause(res api.Result, name string) string {
	message := strings.TrimSpace(res.Message)
	if res.RuleID == "INIT-EXEC-001" && res.Status == api.StatusFail {
		message = "The command is running as " + formatValue(
			res.RuleID,
			name,
			res.Actual,
		) + "; installation requires the root user."
	}
	if isKernelCompatibilityRule(res.RuleID) && res.Status == api.StatusFail {
		if observed := evidenceObserved(res.Evidence); observed != "" {
			message = "The observed platform and kernel are not covered by the release compatibility matrix: " + observed + "."
		}
	}
	if isFactCollectionFailure(res) {
		message = "Could not verify this check: " + collectionFailureCause(name, message)
	}
	if message == "" {
		message = defaultResultCause(res, name)
	}
	return message
}

func collectionFailureCause(name, message string) string {
	if isSourceAccessCheck(name) {
		// Keep collector-level errors at the source-access level; do not expose
		// a protocol-specific root cause as a separate installation failure.
		return "the source address is not accessible"
	}
	if message == "" {
		return "the fact required for this check could not be collected"
	}
	return message
}

func defaultResultCause(res api.Result, name string) string {
	switch {
	case res.CauseCode == api.CauseDependencyFailed:
		return "A prerequisite check failed, so this check was not run."
	case res.Status == api.StatusSkip:
		return "This check does not apply to the current configuration."
	case strings.Contains(strings.ToLower(name), "port"):
		return "The required port is occupied or could not be verified."
	default:
		return "The observed value does not meet the requirement."
	}
}

func resultRecommendation(res api.Result, name string) string {
	recommendation := strings.TrimSpace(res.Remediation)
	if isFactCollectionFailure(res) {
		// Do not present the rule's normal fix when the observed value is
		// unavailable.  For example, EOF from an endpoint probe does not prove
		// that TLS is the problem; it may be a proxy, DNS, route or auth issue.
		recommendation = neutralCollectionRecommendation(res, name)
	}
	if res.RuleID == "INIT-EXEC-001" && recommendation == "" {
		recommendation = "run bke as root (UID 0); for example, start it from a root shell or use sudo"
	}
	if res.Status == api.StatusSkip && strings.Contains(strings.ToLower(name), "offline") {
		recommendation = "No action is required unless offline installation is selected."
	}
	if res.Status == api.StatusSkip && res.CauseCode == api.CauseDependencyFailed {
		recommendation = "Fix the prerequisite failure reported above, then run preflight again."
	}
	if needsGenericRecommendation(recommendation) {
		recommendation = "Review the installation inputs and run preflight again."
	}
	lowerRecommendation := strings.ToLower(recommendation)
	if res.Status == api.StatusFail && !strings.Contains(lowerRecommendation, "run preflight again") &&
		!strings.Contains(lowerRecommendation, "rerun preflight") && !strings.Contains(recommendation, "重试") {
		recommendation = strings.TrimRight(recommendation, ".; ") + "; then run preflight again."
	}
	return recommendation
}

func needsGenericRecommendation(recommendation string) bool {
	lower := strings.ToLower(recommendation)
	return recommendation == "" || strings.Contains(lower, "see provider") ||
		strings.Contains(lower, "see the installation manual")
}

func oneLine(value string) string {
	return strings.Join(strings.Fields(value), " ")
}

func valueOrDash(value string) string {
	if value = oneLine(value); value == "" {
		return "-"
	}
	return value
}

func isSourceAccessCheck(name string) bool {
	lower := strings.ToLower(strings.TrimSpace(name))
	return strings.Contains(lower, "is accessible") || strings.Contains(lower, "source access")
}

// isFactCollectionFailure distinguishes an unavailable observation from a
// real rule mismatch.  Actual.Value is intentionally nil for an unavailable
// fact, while a successfully collected false/empty/zero value remains a real
// value and keeps the rule-specific remediation.
func isFactCollectionFailure(res api.Result) bool {
	if res.Status == api.StatusSkip || res.Actual.Value != nil {
		return false
	}
	switch res.CauseCode {
	case api.CauseUnknownFact, api.FactErrNotFound, api.FactErrPermissionDenied,
		api.FactErrUnreachable, api.FactErrTimeout, api.FactErrUnsupported,
		api.FactErrParseError, api.FrameworkErrInternal:
		return true
	default:
		return strings.TrimSpace(res.Message) != ""
	}
}

func neutralCollectionRecommendation(res api.Result, name string) string {
	switch res.RuleID {
	case "CLUSTER-SSH-TRANSPORT":
		return "Check the node IP and SSH port in nodes.yaml, network/firewall reachability, " +
			"and sshd status, then run preflight again."
	case "CLUSTER-SSH-AUTH":
		return "Check the SSH username and credential configured for this node, then run preflight again."
	case "CLUSTER-SSH-PRIVILEGE":
		return "Configure root SSH access or non-interactive sudo for this node, then run preflight again."
	case "CLUSTER-MGMT-AGENT-KUBECONFIG-GET":
		if res.CauseCode == api.FactErrNotFound {
			return "Create Secret kube-system/localkubeconfig with the expected kubeconfig data, " +
				"then run preflight again."
		}
		return "Check --kubeconfig API access and permission to get Secret " +
			"kube-system/localkubeconfig, then run preflight again."
	case "CLUSTER-MGMT-API":
		return "Check --kubeconfig, management API reachability, and access permissions, then run preflight again."
	case "CLUSTER-MGMT-DISCOVERY", "CLUSTER-MGMT-CRD-BKECLUSTER", "CLUSTER-MGMT-CRD-BKENODE":
		return "Check management API discovery access and the required BKE CRDs, then run preflight again."
	default:
		return neutralCollectionFallback(name)
	}
}

func neutralCollectionFallback(name string) string {
	lower := strings.ToLower(name)
	switch {
	case strings.Contains(lower, "is accessible"), strings.Contains(lower, "source access"):
		return "Unable to access this source; check network connectivity, the endpoint address, " +
			"proxy/NO_PROXY, and firewall settings, then run preflight again."
	case strings.Contains(lower, "ssh"), strings.Contains(lower, "node"):
		return "Unable to verify this node check; inspect node reachability, SSH configuration, " +
			"and permissions, then run preflight again."
	default:
		return "Unable to collect the required fact; inspect the host configuration and permissions, " +
			"then run preflight again."
	}
}

func friendlyName(id string) string {
	known := map[string]string{
		"INIT-COLOCATE":           "Colocation risk",
		"INIT-EXEC-001":           "Root user required",
		"INIT-PATH-MOUNT":         "Workspace mount mode",
		"INIT-TOOL-TAR":           "tar executable",
		"INIT-TOOL-SYSTEMCTL":     "systemctl executable",
		"INIT-REPO-URL":           "Custom repository URL",
		"INIT-REPO-CA-EXISTS":     "Custom repository CA certificate",
		"INIT-REPO-CA-READ":       "Custom repository CA readability",
		"INIT-REPO-CA-PARSE":      "Custom repository CA format",
		"INIT-PLAT-OS":            "Operating system",
		"INIT-PLAT-DISTRO":        "Linux distribution",
		"INIT-PLAT-KERNEL":        "Kernel version",
		"INIT-PLAT-ARCH":          "Host architecture",
		"INIT-RES-CPU":            "CPU capacity",
		"INIT-RES-MEMORY":         "Memory capacity",
		"INIT-RES-SYSTEM-DISK":    "System disk capacity",
		"INIT-PATH-CREATE":        "Workspace path",
		"INIT-PORT-K8S":           "Kubernetes API port",
		"INIT-PORT-REGISTRY":      "Image repository port",
		"INIT-PORT-CHART":         "Chart repository port",
		"INIT-PORT-HTTP":          "Package repository port",
		"INIT-PORT-CONSOLE":       "BKE Console port",
		"INIT-PORT-NTP":           "NTP port",
		"INIT-OFFLINE-EXISTS":     "Offline artifact availability",
		"INIT-OFFLINE-READ":       "Offline artifact readability",
		"INIT-OFFLINE-FORMAT":     "Offline artifact format",
		"CLUSTER-RES-CPU":         "Node CPU capacity",
		"CLUSTER-RES-MEMORY":      "Node memory capacity",
		"CLUSTER-RES-SYSTEM-DISK": "Node system disk capacity",
	}
	if name, ok := known[id]; ok {
		return name
	}
	if strings.HasPrefix(id, "INIT-OFFLINE-") {
		return "Offline artifact " + strings.ToLower(strings.TrimPrefix(id, "INIT-OFFLINE-"))
	}
	words := strings.Fields(
		strings.ReplaceAll(strings.TrimPrefix(strings.TrimPrefix(id, "INIT-"), "CLUSTER-"), "-", " "),
	)
	for i := range words {
		words[i] = strings.ToLower(words[i])
	}
	name := strings.Join(words, " ")
	if name == "" {
		return ""
	}
	return strings.ToUpper(name[:1]) + name[1:]
}
