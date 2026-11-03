/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package output formats and renders check results, printing a human-readable
// terminal report and optionally writing a structured JSON result file.
package output

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strings"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

const (
	iconPass       = "✅"
	iconFail       = "❌"
	iconWarn       = "⚠️"
	iconInfo       = "ℹ️"
	outputFileMode = 0644 // owner read/write, group and others read-only
	iconSkip       = "⏭️"
	iconBlock      = "⛔"
	separator      = "────────────────────────────────"
)

// PrintReport prints the terminal report
func PrintReport(report *types.Report) {
	fmt.Println("\n=== InferNex Checker ===")

	if report.Hardware != nil {
		printHardware(report.Hardware)
	}

	// all-check mode: hardware layer failed, K8s and Config-Env layers were skipped
	if report.HardwareTerminated {
		fmt.Printf("\n%s All-check terminated: no nodes passed hardware layer, "+
			"skipping K8s and Config-Env layers\n", iconBlock)
	}

	if report.K8s != nil {
		printK8s(report.K8s)
	}
	if report.ConfigEnv != nil {
		printConfigEnv(report.ConfigEnv)
	}

	fmt.Println("\n" + separator)
	if report.Summary.Skipped > 0 {
		fmt.Printf("Result: %d passed, %d failed, %d warning, %d info, %d skipped\n",
			report.Summary.Passed, report.Summary.Failed, report.Summary.Warning, report.Summary.Info, report.Summary.Skipped)
	} else {
		fmt.Printf("Result: %d passed, %d failed, %d warning, %d info\n",
			report.Summary.Passed, report.Summary.Failed, report.Summary.Warning, report.Summary.Info)
	}

	// collect warning items
	var warnItems []string
	collectWarningItems(report, &warnItems)
	if len(warnItems) > 0 {
		fmt.Printf("\n%s Warning items:\n", iconWarn)
		for _, item := range warnItems {
			fmt.Printf("  %s\n", item)
		}
	}

	// collect info items
	var infoItems []string
	collectInfoItems(report, &infoItems)
	if len(infoItems) > 0 {
		fmt.Printf("\n%s Info items:\n", iconInfo)
		for _, item := range infoItems {
			fmt.Printf("  %s\n", item)
		}
	}

	// collect skipped items
	var skippedItems []string
	collectSkippedItems(report, &skippedItems)
	if len(skippedItems) > 0 {
		fmt.Printf("\n%s Skipped checks:\n", iconSkip)
		for _, item := range skippedItems {
			fmt.Printf("  %s\n", item)
		}
	}
}

func printCheck(check types.CheckResult) {
	fmt.Printf("  %s %-4s  %s\n", statusIcon(check.Status), check.ID, check.Message)
	if (check.Status == types.StatusFailed || check.Status == types.StatusWarning ||
		check.Status == types.StatusInfo) && check.Suggestion != "" {
		fmt.Printf("    → %s\n", check.Suggestion)
	}
}

func printHardware(hw *types.HardwareResult) {
	for _, node := range hw.Nodes {
		fmt.Printf("\n[Hardware Layer - Single-Node Check: %s]\n", node.Name)
		for _, check := range node.Checks {
			printCheck(check)
		}
		if node.SkippedHCCS {
			fmt.Printf("  %s H-06~H-10 skipped  %s\n", iconSkip, node.SkipReason)
		}
	}

	if hw.CrossNodes != nil {
		// collect 910-series node names that passed single-node checks
		var passedNames []string
		for _, node := range hw.Nodes {
			if node.Is910Series && node.Status == types.StatusPassed {
				passedNames = append(passedNames, node.Name)
			}
		}
		title := "[Hardware Layer - Cross-Node Check]"
		if len(passedNames) > 0 {
			title += fmt.Sprintf(" (910-series nodes: %s)", strings.Join(passedNames, ", "))
		}
		fmt.Printf("\n%s\n", title)
		for _, check := range hw.CrossNodes {
			printCheck(check)
		}
	}
}

func printK8s(k8s *types.K8sResult) {
	fmt.Printf("\n[K8s Layer]\n")
	for _, check := range k8s.Checks {
		printCheck(check)
	}
	if k8s.Status == types.StatusFailed {
		// find the first failed check to identify what triggered termination
		terminatedBy := ""
		for _, check := range k8s.Checks {
			if check.Status == types.StatusFailed {
				terminatedBy = check.ID
				break
			}
		}
		fmt.Printf("  %s K8s layer terminated: %s failed, skipping subsequent checks\n", iconBlock, terminatedBy)
	}
}

func printConfigEnv(biz *types.ConfigEnvResult) {
	// group output by node
	nodeChecks := map[string][]types.CheckResult{}
	var nodeOrder []string
	for _, check := range biz.Checks {
		node := check.Node
		if _, exists := nodeChecks[node]; !exists {
			nodeOrder = append(nodeOrder, node)
		}
		nodeChecks[node] = append(nodeChecks[node], check)
	}

	for _, node := range nodeOrder {
		if node != "" {
			fmt.Printf("\n[Config-Env Layer: %s]\n", node)
		} else {
			fmt.Printf("\n[Config-Env Layer]\n")
		}
		for _, check := range nodeChecks[node] {
			printCheck(check)
		}
	}
}

func collectWarningItems(report *types.Report, items *[]string) {
	if report.Hardware != nil {
		collectHardwareWarningItems(report.Hardware, items)
	}
	if report.K8s != nil {
		collectK8sWarningItems(report.K8s, items)
	}
	if report.ConfigEnv != nil {
		collectConfigEnvWarningItems(report.ConfigEnv, items)
	}
}

func collectHardwareWarningItems(hw *types.HardwareResult, items *[]string) {
	for _, node := range hw.Nodes {
		for _, check := range node.Checks {
			if check.Status == types.StatusWarning {
				*items = append(*items, fmt.Sprintf("%s  %s  %s", check.ID, node.Name, check.Message))
			}
		}
	}
}

func collectK8sWarningItems(k8s *types.K8sResult, items *[]string) {
	for _, check := range k8s.Checks {
		if check.Status == types.StatusWarning {
			*items = append(*items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
}

func collectConfigEnvWarningItems(configEnv *types.ConfigEnvResult, items *[]string) {
	for _, check := range configEnv.Checks {
		if check.Status != types.StatusWarning {
			continue
		}
		if check.Node != "" {
			*items = append(*items, fmt.Sprintf("%s  %s  %s", check.ID, check.Node, check.Message))
		} else {
			*items = append(*items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
}

func collectInfoItems(report *types.Report, items *[]string) {
	if report.Hardware != nil {
		*items = append(*items, collectHardwareInfoItems(report.Hardware)...)
	}
	if report.K8s != nil {
		*items = append(*items, collectK8sInfoItems(report.K8s)...)
	}
	if report.ConfigEnv != nil {
		*items = append(*items, collectConfigEnvInfoItems(report.ConfigEnv)...)
	}
}

func collectHardwareInfoItems(hw *types.HardwareResult) []string {
	var items []string
	for _, node := range hw.Nodes {
		for _, check := range node.Checks {
			if check.Status == types.StatusInfo {
				items = append(items, fmt.Sprintf("%s  %s  %s", check.ID, node.Name, check.Message))
			}
		}
	}
	return items
}

func collectK8sInfoItems(k8s *types.K8sResult) []string {
	var items []string
	for _, check := range k8s.Checks {
		if check.Status == types.StatusInfo {
			items = append(items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
	return items
}

func collectConfigEnvInfoItems(cfg *types.ConfigEnvResult) []string {
	var items []string
	for _, check := range cfg.Checks {
		if check.Status != types.StatusInfo {
			continue
		}
		if check.Node != "" {
			items = append(items, fmt.Sprintf("%s  %s  %s", check.ID, check.Node, check.Message))
		} else {
			items = append(items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
	return items
}

func collectSkippedItems(report *types.Report, items *[]string) {
	if report.Hardware != nil {
		collectHardwareSkippedItems(report.Hardware, items)
	}
	if report.K8s != nil {
		collectK8sSkippedItems(report.K8s, items)
	}
	if report.ConfigEnv != nil {
		collectConfigEnvSkippedItems(report.ConfigEnv, items)
	}
}

func collectHardwareSkippedItems(hw *types.HardwareResult, items *[]string) {
	collectSingleNodeSkippedItems(hw.Nodes, items)
	collectCrossNodeSkippedItems(hw.CrossNodes, items)
}

// groupSkippedNodeNames collects skipped check IDs -> node names across all nodes.
func groupSkippedNodeNames(nodes []types.NodeResult) map[string][]string {
	result := make(map[string][]string)
	for _, node := range nodes {
		for _, check := range node.Checks {
			if check.Status == types.StatusSkipped {
				result[check.ID] = append(result[check.ID], node.Name)
			}
		}
	}
	return result
}

// groupSkippedMessages collects the first-seen message per skipped check ID across all nodes.
func groupSkippedMessages(nodes []types.NodeResult) map[string]string {
	result := make(map[string]string)
	for _, node := range nodes {
		for _, check := range node.Checks {
			if check.Status != types.StatusSkipped {
				continue
			}
			if _, ok := result[check.ID]; !ok {
				result[check.ID] = check.Message
			}
		}
	}
	return result
}

func collectSingleNodeSkippedItems(nodes []types.NodeResult, items *[]string) {
	nodeSkipped := groupSkippedNodeNames(nodes)
	nodeSkippedMsg := groupSkippedMessages(nodes)
	checkIDs := make([]string, 0, len(nodeSkipped))
	for checkID := range nodeSkipped {
		checkIDs = append(checkIDs, checkID)
	}
	sort.Strings(checkIDs)
	for _, checkID := range checkIDs {
		msg := nodeSkippedMsg[checkID]
		if msg == "" {
			msg = checkID
		}
		nodeList := strings.Join(nodeSkipped[checkID], ", ")
		*items = append(*items, fmt.Sprintf("%s [%s]  %s", checkID, nodeList, msg))
	}
}

func collectCrossNodeSkippedItems(crossNodes []types.CheckResult, items *[]string) {
	for _, check := range crossNodes {
		if check.Status != types.StatusSkipped {
			continue
		}
		msg := check.Message
		if msg == "" {
			msg = check.ID
		}
		*items = append(*items, fmt.Sprintf("%s  %s", check.ID, msg))
	}
}

func collectK8sSkippedItems(k8s *types.K8sResult, items *[]string) {
	for _, check := range k8s.Checks {
		if check.Status == types.StatusSkipped {
			*items = append(*items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
}

func collectConfigEnvSkippedItems(configEnv *types.ConfigEnvResult, items *[]string) {
	for _, check := range configEnv.Checks {
		if check.Status != types.StatusSkipped {
			continue
		}
		if check.Node != "" {
			*items = append(*items, fmt.Sprintf("%s  %s  %s", check.ID, check.Node, check.Message))
		} else {
			*items = append(*items, fmt.Sprintf("%s  %s", check.ID, check.Message))
		}
	}
}

// WriteJSON writes the report to a JSON file
func WriteJSON(report *types.Report, path string) error {
	data, err := json.MarshalIndent(report, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to serialize report: %w", err)
	}
	if err := os.WriteFile(path, data, outputFileMode); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}
	return nil
}

// BuildSummary counts check results
func BuildSummary(report *types.Report) types.Summary {
	var s types.Summary
	if report.Hardware != nil {
		for _, node := range report.Hardware.Nodes {
			for _, check := range node.Checks {
				countCheck(check.Status, &s)
			}
		}
		if report.Hardware.CrossNodes != nil {
			for _, check := range report.Hardware.CrossNodes {
				countCheck(check.Status, &s)
			}
		}
	}
	if report.K8s != nil {
		for _, check := range report.K8s.Checks {
			countCheck(check.Status, &s)
		}
	}
	if report.ConfigEnv != nil {
		for _, check := range report.ConfigEnv.Checks {
			countCheck(check.Status, &s)
		}
	}
	s.Total = s.Passed + s.Failed + s.Warning + s.Info + s.Skipped
	return s
}

func countCheck(status types.CheckStatus, s *types.Summary) {
	switch status {
	case types.StatusPassed:
		s.Passed++
	case types.StatusFailed:
		s.Failed++
	case types.StatusWarning:
		s.Warning++
	case types.StatusInfo:
		s.Info++
	case types.StatusSkipped:
		s.Skipped++
	default:
	}
}

func statusIcon(status types.CheckStatus) string {
	switch status {
	case types.StatusPassed:
		return iconPass
	case types.StatusFailed:
		return iconFail
	case types.StatusWarning:
		return iconWarn
	case types.StatusInfo:
		return iconInfo
	case types.StatusSkipped:
		return iconSkip
	default:
		return iconInfo
	}
}
