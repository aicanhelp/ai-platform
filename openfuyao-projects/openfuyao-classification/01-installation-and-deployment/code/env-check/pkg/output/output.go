/******************************************************************
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

// Package output handles formatted output
package output

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/liushuochen/gotable"

	"env-check/pkg/config"
)

const (
	dispatchErrorPreviewMaxLen = 30
	bytesPerGB                 = 1 << 30 // 1073741824, converts bytes to gibibytes
)

// Formatter output formatter
type Formatter struct{}

// NewFormatter creates a new formatter
func NewFormatter() *Formatter {
	return &Formatter{}
}

// FormatQuery formats query result
func (f *Formatter) FormatQuery(result *config.QueryResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatQueryJSON(result)
	default:
		return f.formatQueryText(result)
	}
}

// FormatClean formats clean result
func (f *Formatter) FormatClean(result *config.CleanResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatCleanJSON(result)
	default:
		return f.formatCleanText(result)
	}
}

// FormatProgramCheck formats program check result
func (f *Formatter) FormatProgramCheck(result *config.ProgramCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatProgramCheckJSON(result)
	default:
		return f.formatProgramCheckText(result)
	}
}

// FormatClockCheck formats clock check result
func (f *Formatter) FormatClockCheck(result *config.ClockCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatClockCheckJSON(result)
	default:
		return f.formatClockCheckText(result)
	}
}

// FormatRouteCheck formats route check result
func (f *Formatter) FormatRouteCheck(result *config.RouteCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatRouteCheckJSON(result)
	default:
		return f.formatRouteCheckText(result)
	}
}

// FormatIptablesCheck formats iptables check result
func (f *Formatter) FormatIptablesCheck(result *config.IptablesCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatIptablesCheckJSON(result)
	default:
		return f.formatIptablesCheckText(result)
	}
}

// FormatDispatch formats dispatch result
func (f *Formatter) FormatDispatch(result *config.DispatchResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatDispatchJSON(result)
	default:
		return f.formatDispatchText(result)
	}
}

func (f *Formatter) formatDispatchJSON(result *config.DispatchResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatDispatchText(result *config.DispatchResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Dispatch Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("Total Duration: %s\n", result.Duration))
	builder.WriteString(fmt.Sprintf("Overall Result: %s\n\n", result.Summary.Result))

	// Nodes table
	table, err := gotable.Create("IP", "Role", "Status", "Duration", "Error")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	for _, node := range result.Nodes {
		role := strings.Join(node.Role, ",")
		duration := node.EndTime.Sub(node.StartTime).String()
		errStr := "-"
		if node.Error != "" {
			errStr = truncateString(node.Error, dispatchErrorPreviewMaxLen)
		}
		table.AddRow([]string{node.IP, role, node.Status, duration, errStr})
	}

	builder.WriteString(table.String())
	builder.WriteString("\n")

	// Summary
	summaryTable, _ := gotable.Create("Summary", "Count")
	summaryTable.AddRow([]string{"Total Nodes", fmt.Sprintf("%d", result.Summary.TotalNodes)})
	summaryTable.AddRow([]string{"Success", fmt.Sprintf("%d", result.Summary.SuccessNodes)})
	summaryTable.AddRow([]string{"Failed", fmt.Sprintf("%d", result.Summary.FailedNodes)})
	summaryTable.AddRow([]string{"Running", fmt.Sprintf("%d", result.Summary.RunningNodes)})
	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	// Detailed results per node
	details, err := f.formatDispatchDetails(result.Nodes)
	if err != nil {
		return "", err
	}
	if details != "" {
		builder.WriteString("\n")
		builder.WriteString("Detailed Results:\n")
		builder.WriteString(details)
	}

	return builder.String(), nil
}

// formatDispatchDetails builds per-node check detail tables for text output.
// It mirrors the HTML "Detailed Results" section: a per-check summary table
// followed by detail sub-tables for checks that carry structured detail data.
func (f *Formatter) formatDispatchDetails(nodes []config.NodeResult) (string, error) {
	var builder strings.Builder
	for _, node := range nodes {
		if len(node.Results) == 0 {
			continue
		}
		role := strings.Join(node.Role, ",")
		builder.WriteString(fmt.Sprintf("Node: %s (%s)\n", node.IP, role))

		// Per-check summary table
		summaryTable, err := gotable.Create("Check Type", "Status", "Detail")
		if err != nil {
			return "", fmt.Errorf("failed to create table: %v", err)
		}
		for _, r := range node.Results {
			detail := r.Detail
			if detail == "" {
				detail = "-"
			}
			summaryTable.AddRow([]string{r.CheckType, strings.ToUpper(r.Status), detail})
		}
		builder.WriteString(summaryTable.String())

		// Per-check detail sub-tables
		for _, r := range node.Results {
			if s := f.formatCheckSubTable(r); s != "" {
				builder.WriteString(s)
			}
		}
		builder.WriteString("\n")
	}
	return builder.String(), nil
}

// formatCheckSubTable renders a detail sub-table for a single check result.
// Checks without structured detail data (clock, kernel) return an empty string.
func (f *Formatter) formatCheckSubTable(r config.CheckResult) string {
	switch r.CheckType {
	case "port":
		return f.buildPortSubTable(getPortDetails(r))
	case "disk":
		return f.buildDiskSubTable(getDiskDetails(r))
	case "fileQuery":
		return f.buildFileQuerySubTable(getFileQueryDetails(r))
	case "programCheck":
		return f.buildProgramSubTable(getProgramCheckDetails(r))
	case "route":
		return f.buildRouteSubTable(getRouteDetails(r))
	case "iptables":
		return f.buildIptablesSubTable(getIptablesPolicy(r))
	default:
		return ""
	}
}

func (f *Formatter) buildPortSubTable(ports []config.PortCheckInfo) string {
	if len(ports) == 0 {
		return ""
	}
	table, err := gotable.Create("Port", "Protocol", "Status", "Process")
	if err != nil {
		return ""
	}
	for _, p := range ports {
		status := "Free"
		if p.IsUsed {
			status = "Occupied"
		}
		process := p.Process
		if process == "" {
			process = "-"
		}
		table.AddRow([]string{p.Port, p.Protocol, status, process})
	}
	return table.String()
}

func (f *Formatter) buildDiskSubTable(spaces []config.DiskSpace) string {
	if len(spaces) == 0 {
		return ""
	}
	table, err := gotable.Create("Path", "Total(GB)", "Free(GB)", "Used%", "Required(GB)", "Status")
	if err != nil {
		return ""
	}
	for _, s := range spaces {
		status := "PASS"
		if !s.IsSufficient {
			status = "FAIL"
		}
		if s.Error != "" {
			status = "ERROR"
		}
		table.AddRow([]string{
			s.Path,
			fmt.Sprintf("%.1f", divideFloat(s.Total, bytesPerGB)),
			fmt.Sprintf("%.1f", divideFloat(s.Free, bytesPerGB)),
			fmt.Sprintf("%.1f%%", s.UsedPercent),
			fmt.Sprintf("%.1f", divideFloat(s.MinFree, bytesPerGB)),
			status,
		})
	}
	return table.String()
}

func (f *Formatter) buildFileQuerySubTable(files []config.FileInfo) string {
	if len(files) == 0 {
		return ""
	}
	table, err := gotable.Create("Path", "Exists", "Type", "Owner", "Group", "Permissions", "Error")
	if err != nil {
		return ""
	}
	for _, file := range files {
		exists := "No"
		if file.Exists {
			exists = "Yes"
		}
		table.AddRow([]string{
			file.Path,
			exists,
			f.getTypeString(file),
			orDash(file.Owner),
			orDash(file.Group),
			orDash(file.Permissions),
			orDash(file.Error),
		})
	}
	return table.String()
}

func (f *Formatter) buildProgramSubTable(programs []config.ProgramInfo) string {
	if len(programs) == 0 {
		return ""
	}
	table, err := gotable.Create("Program", "Installed", "Version", "Path")
	if err != nil {
		return ""
	}
	for _, p := range programs {
		installed := "No"
		if p.Installed {
			installed = "Yes"
		}
		version := p.Version
		if version == "" {
			version = "-"
		}
		path := p.Path
		if path == "" {
			path = "-"
		}
		table.AddRow([]string{p.Name, installed, version, path})
	}
	return table.String()
}

func (f *Formatter) buildRouteSubTable(routes []config.RouteInfo) string {
	if len(routes) == 0 {
		return ""
	}
	table, err := gotable.Create("Destination", "Gateway", "Interface", "Metric")
	if err != nil {
		return ""
	}
	for _, r := range routes {
		table.AddRow([]string{r.Destination, r.Gateway, r.Interface, fmt.Sprintf("%d", r.Metric)})
	}
	return table.String()
}

func (f *Formatter) buildIptablesSubTable(policy config.ForwardPolicy) string {
	if policy.Chain == "" {
		return ""
	}
	table, err := gotable.Create("Chain", "Policy", "Is Accept")
	if err != nil {
		return ""
	}
	table.AddRow([]string{policy.Chain, policy.Policy, fmt.Sprintf("%v", policy.IsAccept)})
	return table.String()
}

// truncateString truncates string to max length
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// orDash returns s when non-empty, otherwise "-" for table display.
func orDash(s string) string {
	if s == "" {
		return "-"
	}
	return s
}

// Query result text format using gotable
func (f *Formatter) formatQueryText(result *config.QueryResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Query Time: %s\n\n", result.Timestamp.Format("2006-01-02 15:04:05")))

	tableStr, err := f.buildQueryFilesTable(result.Files)
	if err != nil {
		return "", err
	}
	builder.WriteString(tableStr)
	builder.WriteString("\n")

	summaryStr, err := f.buildQuerySummary(result)
	if err != nil {
		return "", err
	}
	builder.WriteString(summaryStr)
	builder.WriteString("\n")

	return builder.String(), nil
}

// buildQueryFilesTable creates the files table for query output
func (f *Formatter) buildQueryFilesTable(files []config.FileInfo) (string, error) {
	table, err := gotable.Create("Path", "Exists", "Type", "Owner", "Group", "Permissions")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	for _, file := range files {
		status := "Missing"
		if file.Exists {
			status = "Exists"
		}

		if err := table.AddRow([]string{
			file.Path,
			status,
			f.getTypeString(file),
			orDash(file.Owner),
			orDash(file.Group),
			orDash(file.Permissions),
		}); err != nil {
			return "", fmt.Errorf("failed to add row: %v", err)
		}
	}

	return table.String(), nil
}

// buildQuerySummary creates the summary table for query output
func (f *Formatter) buildQuerySummary(result *config.QueryResult) (string, error) {
	summaryTable, err := gotable.Create("Summary", "Count")
	if err != nil {
		return "", fmt.Errorf("failed to create summary table: %v", err)
	}

	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Total Exists", fmt.Sprintf("%d", result.Summary.TotalExists)})
	summaryTable.AddRow([]string{"Total Missing", fmt.Sprintf("%d", result.Summary.TotalMissing)})
	summaryTable.AddRow([]string{"Total Directories", fmt.Sprintf("%d", result.Summary.TotalDirs)})
	summaryTable.AddRow([]string{"Total Files", fmt.Sprintf("%d", result.Summary.TotalFiles)})

	return summaryTable.String(), nil
}

// Clean result text format using gotable
func (f *Formatter) formatCleanText(result *config.CleanResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Clean Time: %s\n\n", result.Timestamp.Format("2006-01-02 15:04:05")))

	if s, err := f.buildCleanDeletedSection(result.Deleted); err != nil {
		return "", err
	} else if s != "" {
		builder.WriteString(s)
		builder.WriteString("\n")
	} else {
		// empty section, nothing to output
	}

	if s, err := f.buildCleanFailedSection(result.Failed); err != nil {
		return "", err
	} else if s != "" {
		builder.WriteString(s)
		builder.WriteString("\n")
	} else {
		// empty section, nothing to output
	}

	if s, err := f.buildCleanSkippedSection(result.Skipped); err != nil {
		return "", err
	} else if s != "" {
		builder.WriteString(s)
		builder.WriteString("\n")
	} else {
		// empty section, nothing to output
	}

	summaryStr, err := f.buildCleanSummary(result)
	if err != nil {
		return "", err
	}
	builder.WriteString(summaryStr)
	builder.WriteString("\n")

	return builder.String(), nil
}

func (f *Formatter) buildCleanDeletedSection(deleted []config.FileInfo) (string, error) {
	if len(deleted) == 0 {
		return "", nil
	}
	var b strings.Builder
	b.WriteString("Deleted Files:\n")
	table, err := gotable.Create("Path", "Type")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}
	for _, file := range deleted {
		typeStr := "File"
		if file.IsDir {
			typeStr = "Directory"
		}
		table.AddRow([]string{file.Path, typeStr})
	}
	b.WriteString(table.String())
	return b.String(), nil
}

func (f *Formatter) buildCleanFailedSection(failed []config.FileInfo) (string, error) {
	if len(failed) == 0 {
		return "", nil
	}
	var b strings.Builder
	b.WriteString("Failed Files:\n")
	table, err := gotable.Create("Path", "Error")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}
	for _, file := range failed {
		errorMsg := file.Error
		if errorMsg == "" {
			errorMsg = "Unknown error"
		}
		table.AddRow([]string{file.Path, errorMsg})
	}
	b.WriteString(table.String())
	return b.String(), nil
}

func (f *Formatter) buildCleanSkippedSection(skipped []config.FileInfo) (string, error) {
	if len(skipped) == 0 {
		return "", nil
	}
	var b strings.Builder
	b.WriteString("Skipped Files:\n")
	table, err := gotable.Create("Path", "Reason")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}
	for _, file := range skipped {
		reason := "User canceled"
		if file.Error != "" {
			reason = file.Error
		}
		table.AddRow([]string{file.Path, reason})
	}
	b.WriteString(table.String())
	return b.String(), nil
}

func (f *Formatter) buildCleanSummary(result *config.CleanResult) (string, error) {
	summaryTable, err := gotable.Create("Summary", "Count")
	if err != nil {
		return "", fmt.Errorf("failed to create summary table: %v", err)
	}
	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Total Deleted", fmt.Sprintf("%d", result.Summary.TotalDeleted)})
	summaryTable.AddRow([]string{"Total Failed", fmt.Sprintf("%d", result.Summary.TotalFailed)})
	summaryTable.AddRow([]string{"Total Skipped", fmt.Sprintf("%d", result.Summary.TotalSkipped)})
	return summaryTable.String(), nil
}

// Program check result text format using gotable
func (f *Formatter) formatProgramCheckText(result *config.ProgramCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Check Time: %s\n\n", result.Timestamp.Format("2006-01-02 15:04:05")))

	// Create table for program information
	table, err := gotable.Create("Program", "Expected", "Status", "Result", "Version", "Path")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	for _, program := range result.Programs {
		expected := "Should Exist"
		if !program.ShouldExist {
			expected = "Should Not Exist"
		}

		status := "Not Installed"
		if program.Installed {
			status = "Installed"
		}

		resultStatus := "Passed"
		if !program.Passed {
			resultStatus = "Failed"
		}

		version := program.Version
		if version == "" {
			version = "Unknown"
		}

		path := program.Path
		if path == "" {
			path = "Not found"
		}

		table.AddRow([]string{
			program.Name,
			expected,
			status,
			resultStatus,
			version,
			path,
		})
	}

	builder.WriteString(table.String())
	builder.WriteString("\n")

	// Add summary information
	summaryTable, err := gotable.Create("Summary", "Count")
	if err != nil {
		return "", fmt.Errorf("failed to create summary table: %v", err)
	}

	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Total Installed", fmt.Sprintf("%d", result.Summary.TotalInstalled)})
	summaryTable.AddRow([]string{"Total Missing", fmt.Sprintf("%d", result.Summary.TotalMissing)})
	summaryTable.AddRow([]string{"Total Passed", fmt.Sprintf("%d", result.Summary.TotalPassed)})
	summaryTable.AddRow([]string{"Total Failed", fmt.Sprintf("%d", result.Summary.TotalFailed)})

	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	return builder.String(), nil
}

// JSON format functions
func (f *Formatter) formatQueryJSON(result *config.QueryResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatCleanJSON(result *config.CleanResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatProgramCheckJSON(result *config.ProgramCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// Utility functions
func (f *Formatter) getTypeString(file config.FileInfo) string {
	if !file.Exists {
		return "Missing"
	}
	if file.IsDir {
		return "Directory"
	}
	return "File"
}

func (f *Formatter) formatClockCheckJSON(result *config.ClockCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatClockCheckText(result *config.ClockCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Clock Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("Clock Check Result: %s\n\n", result.Result))
	table, err := gotable.Create("Host", "Role", "Synced", "Time Diff (s)", "Local", "Error")
	if err != nil {
		fmt.Printf("Error creating table: %v\n", err)
		return "", err
	}

	for _, r := range result.Clocks {
		var syncedStr, diffStr, localStr, errStr string
		syncedStr = fmt.Sprintf("%v", r.IsSynced)
		if r.TimeDiff >= 0 {
			diffStr = fmt.Sprintf("%d", r.TimeDiff)
		} else {
			diffStr = "ERROR"
		}
		localStr = fmt.Sprintf("%v", r.IsLocal)
		errStr = r.Error
		err1 := table.AddRow([]string{r.Host, strings.Join(r.Role, ","), syncedStr, diffStr, localStr, errStr})
		if err1 != nil {
			return "", fmt.Errorf("failed to add row: %v", err1)
		}
	}
	builder.WriteString(table.String())
	builder.WriteString("\n")
	return builder.String(), nil
}

func (f *Formatter) formatRouteCheckJSON(result *config.RouteCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ========== Kernel Check Formatting ==========

// FormatKernelCheck formats kernel check result
func (f *Formatter) FormatKernelCheck(result *config.KernelCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatKernelCheckJSON(result)
	default:
		return f.formatKernelCheckText(result)
	}
}

func (f *Formatter) formatKernelCheckJSON(result *config.KernelCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatRouteCheckText(result *config.RouteCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Route Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("Check Passed: %v\n", result.Summary.CheckPassed))
	builder.WriteString(fmt.Sprintf("Message: %s\n\n", result.Summary.Message))

	if len(result.DefaultRoutes) > 0 {
		table, err := gotable.Create("Destination", "Gateway", "Interface", "Metric")
		if err != nil {
			return "", fmt.Errorf("failed to create table: %v", err)
		}

		for _, route := range result.DefaultRoutes {
			table.AddRow([]string{
				route.Destination,
				route.Gateway,
				route.Interface,
				fmt.Sprintf("%d", route.Metric),
			})
		}
		builder.WriteString("Default Routes:\n")
		builder.WriteString(table.String())
		builder.WriteString("\n")
	}

	summaryTable, err := gotable.Create("Summary", "Value")
	if err != nil {
		return "", fmt.Errorf("failed to create summary table: %v", err)
	}
	summaryTable.AddRow([]string{"Total Routes", fmt.Sprintf("%d", result.Summary.TotalRoutes)})
	summaryTable.AddRow([]string{"Check Passed", fmt.Sprintf("%v", result.Summary.CheckPassed)})
	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	return builder.String(), nil
}

func (f *Formatter) formatKernelCheckText(result *config.KernelCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Kernel Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString("\n")

	table, err := gotable.Create("Item", "Value")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	info := result.KernelInfo
	table.AddRow([]string{"Current Version", info.Version})
	table.AddRow([]string{"OS", info.OS})
	table.AddRow([]string{"Architecture", info.Arch})
	table.AddRow([]string{"Expected Version", fmt.Sprintf("%s %s", info.Operator, info.ExpectedVer)})

	status := "PASS"
	if !info.IsValid {
		status = "FAIL"
	}
	table.AddRow([]string{"Status", status})

	if info.Error != "" {
		table.AddRow([]string{"Error", info.Error})
	}

	builder.WriteString(table.String())
	builder.WriteString("\n")

	// Summary
	summaryTable, _ := gotable.Create("Summary", "Count")
	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Passed", fmt.Sprintf("%d", result.Summary.Passed)})
	summaryTable.AddRow([]string{"Failed", fmt.Sprintf("%d", result.Summary.Failed)})
	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	return builder.String(), nil
}

func (f *Formatter) formatIptablesCheckJSON(result *config.IptablesCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

// ========== Port Check Formatting ==========

// FormatPortCheck formats port check result
func (f *Formatter) FormatPortCheck(result *config.PortCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatPortCheckJSON(result)
	default:
		return f.formatPortCheckText(result)
	}
}

func (f *Formatter) formatPortCheckJSON(result *config.PortCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatIptablesCheckText(result *config.IptablesCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Iptables Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString(fmt.Sprintf("Check Passed: %v\n", result.Summary.CheckPassed))
	builder.WriteString(fmt.Sprintf("Message: %s\n\n", result.Summary.Message))

	table, err := gotable.Create("Chain", "Policy", "Is Accept")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	table.AddRow([]string{
		result.ForwardPolicy.Chain,
		result.ForwardPolicy.Policy,
		fmt.Sprintf("%v", result.ForwardPolicy.IsAccept),
	})
	builder.WriteString("Forward Chain Policy:\n")
	builder.WriteString(table.String())
	builder.WriteString("\n")

	return builder.String(), nil
}

func (f *Formatter) formatPortCheckText(result *config.PortCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Port Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString("\n")

	table, err := gotable.Create("Port", "Protocol", "Status", "Process")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	for _, port := range result.Ports {
		status := "Free"
		if port.IsUsed {
			status = "Occupied"
		}
		process := port.Process
		if process == "" {
			process = "-"
		}
		table.AddRow([]string{port.Port, port.Protocol, status, process})
	}

	builder.WriteString(table.String())
	builder.WriteString("\n")

	// Summary
	summaryTable, _ := gotable.Create("Summary", "Count")
	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Occupied", fmt.Sprintf("%d", result.Summary.Used)})
	summaryTable.AddRow([]string{"Free", fmt.Sprintf("%d", result.Summary.Free)})
	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	return builder.String(), nil
}

// ========== Disk Check Formatting ==========

// FormatDiskCheck formats disk check result
func (f *Formatter) FormatDiskCheck(result *config.DiskCheckResult, format string) (string, error) {
	switch format {
	case "json":
		return f.formatDiskCheckJSON(result)
	default:
		return f.formatDiskCheckText(result)
	}
}

func (f *Formatter) formatDiskCheckJSON(result *config.DiskCheckResult) (string, error) {
	data, err := json.MarshalIndent(result, "", "  ")
	if err != nil {
		return "", err
	}
	return string(data), nil
}

func (f *Formatter) formatDiskCheckText(result *config.DiskCheckResult) (string, error) {
	var builder strings.Builder

	builder.WriteString("\n")
	builder.WriteString(fmt.Sprintf("Disk Check Time: %s\n", result.Timestamp.Format("2006-01-02 15:04:05")))
	builder.WriteString("\n")

	table, err := gotable.Create("Path", "Filesystem", "Total(GB)", "Free(GB)", "Used(GB)", "Used%", "Status")
	if err != nil {
		return "", fmt.Errorf("failed to create table: %v", err)
	}

	for _, space := range result.Spaces {
		status := "PASS"
		if !space.IsSufficient {
			status = "FAIL"
		}
		if space.Error != "" {
			status = "ERROR"
		}

		totalGB := float64(space.Total) / (1024 * 1024 * 1024)
		freeGB := float64(space.Free) / (1024 * 1024 * 1024)
		usedGB := float64(space.Used) / (1024 * 1024 * 1024)

		fs := space.Filesystem
		if fs == "" {
			fs = "-"
		}

		table.AddRow([]string{
			space.Path,
			fs,
			fmt.Sprintf("%.1f", totalGB),
			fmt.Sprintf("%.1f", freeGB),
			fmt.Sprintf("%.1f", usedGB),
			fmt.Sprintf("%.1f%%", space.UsedPercent),
			status,
		})
	}

	builder.WriteString(table.String())
	builder.WriteString("\n")

	// Summary
	summaryTable, _ := gotable.Create("Summary", "Count")
	summaryTable.AddRow([]string{"Total Checked", fmt.Sprintf("%d", result.Summary.TotalChecked)})
	summaryTable.AddRow([]string{"Sufficient", fmt.Sprintf("%d", result.Summary.SufficientPath)})
	summaryTable.AddRow([]string{"Insufficient", fmt.Sprintf("%d", result.Summary.InsufficientPath)})
	builder.WriteString(summaryTable.String())
	builder.WriteString("\n")

	return builder.String(), nil
}
