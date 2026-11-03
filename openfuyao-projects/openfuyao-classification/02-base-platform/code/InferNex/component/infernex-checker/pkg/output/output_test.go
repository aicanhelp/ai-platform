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

package output

import (
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// captureStdout replaces os.Stdout with a pipe, runs f, then returns the captured output.
func captureStdout(f func()) string {
	r, w, _ := os.Pipe()
	old := os.Stdout
	os.Stdout = w

	f()

	w.Close()
	os.Stdout = old
	out, _ := io.ReadAll(r)
	return string(out)
}

// ---------------------------------------------------------------------------
// statusIcon
// ---------------------------------------------------------------------------

func TestStatusIcon(t *testing.T) {
	tests := []struct {
		status types.CheckStatus
		want   string
	}{
		{types.StatusPassed, iconPass},
		{types.StatusFailed, iconFail},
		{types.StatusWarning, iconWarn},
		{types.StatusInfo, iconInfo},
		{types.StatusSkipped, iconSkip},
		{"unknown_status", iconInfo}, // default branch
	}
	for _, tt := range tests {
		got := statusIcon(tt.status)
		if got != tt.want {
			t.Errorf("statusIcon(%q) = %q, want %q", tt.status, got, tt.want)
		}
	}
}

// ---------------------------------------------------------------------------
// countCheck / BuildSummary
// ---------------------------------------------------------------------------

func TestCountCheck(t *testing.T) {
	s := &types.Summary{}
	countCheck(types.StatusPassed, s)
	countCheck(types.StatusFailed, s)
	countCheck(types.StatusInfo, s)
	countCheck(types.StatusWarning, s)
	countCheck(types.StatusSkipped, s) // default — no increment
	if s.Passed != 1 {
		t.Errorf("Passed = %d, want 1", s.Passed)
	}
	if s.Failed != 1 {
		t.Errorf("Failed = %d, want 1", s.Failed)
	}
	if s.Warning != 1 {
		t.Errorf("Warning = %d, want 1", s.Warning)
	}
	if s.Info != 1 {
		t.Errorf("Info = %d, want 1", s.Info)
	}
}

func TestBuildSummaryEmpty(t *testing.T) {
	s := BuildSummary(&types.Report{})
	if s.Total != 0 || s.Passed != 0 || s.Failed != 0 || s.Info != 0 {
		t.Errorf("expected all-zero summary, got %+v", s)
	}
}

func TestBuildSummaryHardwareOnly(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Checks: []types.CheckResult{
						{Status: types.StatusPassed},
						{Status: types.StatusFailed},
						{Status: types.StatusInfo},
					},
				},
			},
			CrossNodes: []types.CheckResult{
				{Status: types.StatusPassed},
				{Status: types.StatusWarning},
			},
		},
	}
	s := BuildSummary(report)
	if s.Passed != 2 {
		t.Errorf("Passed = %d, want 2", s.Passed)
	}
	if s.Failed != 1 {
		t.Errorf("Failed = %d, want 1", s.Failed)
	}
	if s.Warning != 1 {
		t.Errorf("Warning = %d, want 1", s.Warning)
	}
	if s.Info != 1 {
		t.Errorf("Info = %d, want 1", s.Info)
	}
	if s.Total != 5 {
		t.Errorf("Total = %d, want 5", s.Total)
	}
}

func TestBuildSummaryK8sOnly(t *testing.T) {
	report := &types.Report{
		K8s: &types.K8sResult{
			Checks: []types.CheckResult{
				{Status: types.StatusPassed},
				{Status: types.StatusFailed},
			},
		},
	}
	s := BuildSummary(report)
	if s.Passed != 1 || s.Failed != 1 || s.Total != 2 {
		t.Errorf("unexpected summary: %+v", s)
	}
}

func TestBuildSummaryConfigEnvOnly(t *testing.T) {
	report := &types.Report{
		ConfigEnv: &types.ConfigEnvResult{
			Checks: []types.CheckResult{
				{Status: types.StatusPassed},
				{Status: types.StatusInfo},
				{Status: types.StatusSkipped},
			},
		},
	}
	s := BuildSummary(report)
	if s.Passed != 1 {
		t.Errorf("Passed = %d, want 1", s.Passed)
	}
	if s.Info != 1 {
		t.Errorf("Info = %d, want 1", s.Info)
	}
	// skipped does count in Total now
	if s.Total != 3 {
		t.Errorf("Total = %d, want 3", s.Total)
	}
	if s.Skipped != 1 {
		t.Errorf("Skipped = %d, want 1", s.Skipped)
	}
}

func TestBuildSummaryAllLayers(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{Checks: []types.CheckResult{{Status: types.StatusPassed}}},
			},
		},
		K8s: &types.K8sResult{
			Checks: []types.CheckResult{{Status: types.StatusFailed}},
		},
		ConfigEnv: &types.ConfigEnvResult{
			Checks: []types.CheckResult{{Status: types.StatusInfo}},
		},
	}
	s := BuildSummary(report)
	if s.Passed != 1 || s.Failed != 1 || s.Info != 1 || s.Total != 3 {
		t.Errorf("unexpected summary: %+v", s)
	}
}

func TestBuildSummaryNilCrossNodes(t *testing.T) {
	// CrossNodes == nil should not panic
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{Checks: []types.CheckResult{{Status: types.StatusPassed}}},
			},
			CrossNodes: nil,
		},
	}
	s := BuildSummary(report)
	if s.Passed != 1 {
		t.Errorf("Passed = %d, want 1", s.Passed)
	}
}

// ---------------------------------------------------------------------------
// collectInfoItems helpers
// ---------------------------------------------------------------------------

func TestCollectHardwareInfoItems(t *testing.T) {
	hw := &types.HardwareResult{
		Nodes: []types.NodeResult{
			{
				Name: "node1",
				Checks: []types.CheckResult{
					{ID: "H-04", Status: types.StatusInfo, Message: "2/8 NPUs available"},
					{ID: "H-01", Status: types.StatusPassed, Message: "driver loaded"},
				},
			},
		},
	}
	items := collectHardwareInfoItems(hw)
	if len(items) != 1 {
		t.Fatalf("expected 1 info item, got %d", len(items))
	}
	if !strings.Contains(items[0], "H-04") || !strings.Contains(items[0], "node1") {
		t.Errorf("unexpected info item: %s", items[0])
	}
}

func TestCollectK8sInfoItems(t *testing.T) {
	k8s := &types.K8sResult{
		Checks: []types.CheckResult{
			{ID: "K-01", Status: types.StatusInfo, Message: "cluster version 1.27"},
			{ID: "K-02", Status: types.StatusPassed, Message: "nodes ready"},
		},
	}
	items := collectK8sInfoItems(k8s)
	if len(items) != 1 {
		t.Fatalf("expected 1 info item, got %d", len(items))
	}
	if !strings.Contains(items[0], "K-01") {
		t.Errorf("unexpected item: %s", items[0])
	}
}

func TestCollectConfigEnvInfoItemsWithNode(t *testing.T) {
	cfg := &types.ConfigEnvResult{
		Checks: []types.CheckResult{
			{ID: "C-01", Status: types.StatusInfo, Node: "node1", Message: "driver version ok"},
			{ID: "C-02", Status: types.StatusInfo, Node: "", Message: "global info"},
			{ID: "C-03", Status: types.StatusPassed, Node: "node1", Message: "passed"},
		},
	}
	items := collectConfigEnvInfoItems(cfg)
	if len(items) != 2 {
		t.Fatalf("expected 2 info items, got %d: %v", len(items), items)
	}
	// first item has node name
	if !strings.Contains(items[0], "node1") {
		t.Errorf("expected node1 in item: %s", items[0])
	}
	// second item has no node name
	if strings.Contains(items[1], "node1") {
		t.Errorf("unexpected node1 in item: %s", items[1])
	}
}

func TestCollectInfoItemsAllNil(t *testing.T) {
	var items []string
	collectInfoItems(&types.Report{}, &items)
	if len(items) != 0 {
		t.Errorf("expected no items, got %v", items)
	}
}

func TestCollectInfoItemsAllLayers(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{Name: "n1", Checks: []types.CheckResult{{ID: "H-04", Status: types.StatusInfo, Message: "x"}}},
			},
		},
		K8s: &types.K8sResult{
			Checks: []types.CheckResult{{ID: "K-01", Status: types.StatusInfo, Message: "y"}},
		},
		ConfigEnv: &types.ConfigEnvResult{
			Checks: []types.CheckResult{{ID: "C-01", Status: types.StatusInfo, Node: "n1", Message: "z"}},
		},
	}
	var items []string
	collectInfoItems(report, &items)
	if len(items) != 3 {
		t.Errorf("expected 3 info items, got %d: %v", len(items), items)
	}
}

// ---------------------------------------------------------------------------
// PrintReport output capture
// ---------------------------------------------------------------------------

func TestPrintReportMinimal(t *testing.T) {
	report := &types.Report{
		Summary: types.Summary{Total: 0},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "InferNex Checker") {
		t.Errorf("expected header in output, got: %s", out)
	}
	if !strings.Contains(out, "passed") {
		t.Errorf("expected 'passed' in summary line, got: %s", out)
	}
}

func TestPrintReportHardwareTerminated(t *testing.T) {
	report := &types.Report{
		HardwareTerminated: true,
		Summary:            types.Summary{},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "All-check terminated") {
		t.Errorf("expected termination message, got: %s", out)
	}
}

func TestPrintReportWithHardware(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Name: "node1",
					Checks: []types.CheckResult{
						{ID: "H-01", Status: types.StatusPassed, Message: "driver loaded"},
						{ID: "H-02", Status: types.StatusFailed, Message: "plugin missing", Suggestion: "install plugin"},
					},
				},
			},
		},
		Summary: types.Summary{Passed: 1, Failed: 1, Total: 2},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "node1") {
		t.Errorf("expected node name in output")
	}
	if !strings.Contains(out, "H-01") {
		t.Errorf("expected H-01 in output")
	}
	if !strings.Contains(out, "install plugin") {
		t.Errorf("expected suggestion in output")
	}
}

func TestPrintReportHardwareSkippedHCCS(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Name:        "node1",
					SkippedHCCS: true,
					SkipReason:  "non-910 series",
					Checks:      []types.CheckResult{{ID: "H-01", Status: types.StatusPassed, Message: "ok"}},
				},
			},
		},
		Summary: types.Summary{Passed: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "H-06~H-10 skipped") {
		t.Errorf("expected HCCS skip message, got: %s", out)
	}
}

func TestPrintReportCrossNodes(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{Name: "node1", Status: types.StatusPassed, Is910Series: true,
					Checks: []types.CheckResult{{ID: "H-01", Status: types.StatusPassed, Message: "ok"}}},
			},
			CrossNodes: []types.CheckResult{
				{ID: "H-08", Status: types.StatusPassed, Message: "cross-node ok"},
			},
		},
		Summary: types.Summary{Passed: 2, Total: 2},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "Cross-Node Check") {
		t.Errorf("expected cross-node section, got: %s", out)
	}
	if !strings.Contains(out, "node1") {
		t.Errorf("expected node1 in cross-node title, got: %s", out)
	}
}

func TestPrintReportCrossNodesNoPassedNodes(t *testing.T) {
	// CrossNodes present but no 910-series passed nodes → title has no node list
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{Name: "node1", Status: types.StatusFailed, Is910Series: false,
					Checks: []types.CheckResult{{ID: "H-01", Status: types.StatusFailed, Message: "fail"}}},
			},
			CrossNodes: []types.CheckResult{
				{ID: "H-08", Status: types.StatusSkipped, Message: "skipped"},
			},
		},
		Summary: types.Summary{Failed: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "Cross-Node Check") {
		t.Errorf("expected cross-node section")
	}
}

func TestPrintReportWithK8s(t *testing.T) {
	report := &types.Report{
		K8s: &types.K8sResult{
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{ID: "K-01", Status: types.StatusPassed, Message: "cluster healthy"},
			},
		},
		Summary: types.Summary{Passed: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "K8s Layer") {
		t.Errorf("expected K8s layer, got: %s", out)
	}
	if !strings.Contains(out, "K-01") {
		t.Errorf("expected K-01 in output")
	}
}

func TestPrintReportK8sTerminated(t *testing.T) {
	report := &types.Report{
		K8s: &types.K8sResult{
			Status: types.StatusFailed,
			Checks: []types.CheckResult{
				{ID: "K-02", Status: types.StatusFailed, Message: "no nodes ready", Suggestion: "check nodes"},
			},
		},
		Summary: types.Summary{Failed: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "K8s layer terminated") {
		t.Errorf("expected K8s termination message, got: %s", out)
	}
	if !strings.Contains(out, "K-02") {
		t.Errorf("expected terminated-by ID in output")
	}
}

func TestPrintReportK8sFailedNoChecks(t *testing.T) {
	// Failed K8s with no checks — terminatedBy should be empty string
	report := &types.Report{
		K8s: &types.K8sResult{
			Status: types.StatusFailed,
			Checks: []types.CheckResult{},
		},
		Summary: types.Summary{},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "K8s layer terminated") {
		t.Errorf("expected K8s termination message, got: %s", out)
	}
}

func TestPrintReportWithConfigEnv(t *testing.T) {
	report := &types.Report{
		ConfigEnv: &types.ConfigEnvResult{
			Status: types.StatusPassed,
			Checks: []types.CheckResult{
				{ID: "C-01", Status: types.StatusPassed, Node: "node1", Message: "config ok"},
				{ID: "C-02", Status: types.StatusPassed, Node: "", Message: "global check ok"},
			},
		},
		Summary: types.Summary{Passed: 2, Total: 2},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "Config-Env Layer") {
		t.Errorf("expected Config-Env section, got: %s", out)
	}
	if !strings.Contains(out, "node1") {
		t.Errorf("expected node1 in Config-Env output")
	}
	// global check group (node == "") should show Config-Env Layer without node name
	if !strings.Contains(out, "[Config-Env Layer]") {
		t.Errorf("expected unnamed Config-Env Layer group, got: %s", out)
	}
}

func TestPrintReportInfoItemsAppear(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Name: "node1",
					Checks: []types.CheckResult{
						{ID: "H-04", Status: types.StatusInfo, Message: "4/8 available"},
					},
				},
			},
		},
		Summary: types.Summary{Info: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "Info items") {
		t.Errorf("expected Info items section, got: %s", out)
	}
	if !strings.Contains(out, "H-04") {
		t.Errorf("expected H-04 in info items, got: %s", out)
	}
}

func TestPrintReportFailedWithSuggestion(t *testing.T) {
	report := &types.Report{
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Name: "node1",
					Checks: []types.CheckResult{
						{ID: "H-01", Status: types.StatusFailed, Message: "driver missing", Suggestion: "install driver"},
					},
				},
			},
		},
		Summary: types.Summary{Failed: 1, Total: 1},
	}
	out := captureStdout(func() { PrintReport(report) })
	if !strings.Contains(out, "install driver") {
		t.Errorf("expected suggestion in output, got: %s", out)
	}
	if !strings.Contains(out, "→") {
		t.Errorf("expected arrow prefix for suggestion, got: %s", out)
	}
}

func TestPrintCheckNoSuggestionOnPass(t *testing.T) {
	check := types.CheckResult{
		ID:         "H-01",
		Status:     types.StatusPassed,
		Message:    "ok",
		Suggestion: "this should not appear",
	}
	out := captureStdout(func() { printCheck(check) })
	if strings.Contains(out, "this should not appear") {
		t.Errorf("suggestion should not appear for passed check, got: %s", out)
	}
}

// ---------------------------------------------------------------------------
// WriteJSON
// ---------------------------------------------------------------------------

func TestWriteJSONSuccess(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")

	report := &types.Report{
		Summary: types.Summary{Total: 1, Passed: 1},
		K8s: &types.K8sResult{
			Status: types.StatusPassed,
			Checks: []types.CheckResult{{ID: "K-01", Status: types.StatusPassed, Message: "ok"}},
		},
	}

	if err := WriteJSON(report, path); err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read output file: %v", err)
	}

	var parsed types.Report
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("output is not valid JSON: %v", err)
	}
	if parsed.Summary.Passed != 1 {
		t.Errorf("Summary.Passed = %d, want 1", parsed.Summary.Passed)
	}
}

func TestWriteJSONInvalidPath(t *testing.T) {
	report := &types.Report{}
	err := WriteJSON(report, "/nonexistent/dir/report.json")
	if err == nil {
		t.Error("expected error for invalid path, got nil")
	}
}

func TestWriteJSONFileExists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")

	if err := WriteJSON(&types.Report{}, path); err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}

	if _, err := os.Stat(path); err != nil {
		t.Fatalf("expected file to exist: %v", err)
	}
}

func TestWriteJSONRoundTrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "report.json")

	original := &types.Report{
		HardwareTerminated: true,
		Summary:            types.Summary{Total: 3, Passed: 1, Failed: 1, Info: 1},
		Hardware: &types.HardwareResult{
			Nodes: []types.NodeResult{
				{
					Name:   "node1",
					Status: types.StatusPassed,
					Checks: []types.CheckResult{
						{ID: "H-01", Status: types.StatusPassed, Message: "ok"},
					},
				},
			},
		},
		K8s: &types.K8sResult{
			Status: types.StatusFailed,
			Checks: []types.CheckResult{{ID: "K-01", Status: types.StatusFailed, Message: "fail"}},
		},
		ConfigEnv: &types.ConfigEnvResult{
			Status: types.StatusInfo,
			Checks: []types.CheckResult{{ID: "C-01", Status: types.StatusInfo, Node: "node1", Message: "info"}},
		},
	}

	if err := WriteJSON(original, path); err != nil {
		t.Fatalf("WriteJSON failed: %v", err)
	}

	data, _ := os.ReadFile(path)
	var parsed types.Report
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if !parsed.HardwareTerminated {
		t.Error("HardwareTerminated not preserved")
	}
	if parsed.Hardware.Nodes[0].Name != "node1" {
		t.Errorf("node name not preserved: %s", parsed.Hardware.Nodes[0].Name)
	}
}
