/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
)

// createTestQueryResult 创建测试查询结果
func createTestQueryResult() *config.QueryResult {
	return &config.QueryResult{
		Timestamp: time.Now(),
		Files: []config.FileInfo{
			{
				Path:        "/tmp/test.txt",
				Exists:      true,
				IsDir:       false,
				Owner:       "user1",
				Group:       "group1",
				Permissions: "-rw-r--r--",
			},
			{
				Path:        "/tmp/dir",
				Exists:      true,
				IsDir:       true,
				Owner:       "user2",
				Group:       "group2",
				Permissions: "drwxr-xr-x",
			},
			{
				Path:   "/tmp/missing",
				Exists: false,
				Error:  "路径不存在",
			},
		},
		Summary: config.QuerySummary{
			TotalChecked: 3,
			TotalExists:  2,
			TotalMissing: 1,
			TotalDirs:    1,
			TotalFiles:   1,
		},
	}
}

// createTestCleanResult 创建测试清理结果
func createTestCleanResult() *config.CleanResult {
	return &config.CleanResult{
		Timestamp: time.Now(),
		Deleted: []config.FileInfo{
			{
				Path:   "/tmp/deleted1.txt",
				Exists: false,
				IsDir:  false,
			},
			{
				Path:   "/tmp/deleted2",
				Exists: false,
				IsDir:  true,
			},
		},
		Failed: []config.FileInfo{
			{
				Path:  "/tmp/failed.txt",
				Error: "权限不足",
			},
		},
		Skipped: []config.FileInfo{
			{
				Path:  "/tmp/skipped.txt",
				Error: "用户取消",
			},
		},
		Summary: config.CleanSummary{
			TotalChecked: 4,
			TotalDeleted: 2,
			TotalFailed:  1,
			TotalSkipped: 1,
		},
	}
}

// createTestProgramCheckResult 创建测试程序检查结果
func createTestProgramCheckResult() *config.ProgramCheckResult {
	return &config.ProgramCheckResult{
		Timestamp: time.Now(),
		Programs: []config.ProgramInfo{
			{
				Name:        "docker",
				ShouldExist: false,
				Installed:   true,
				Passed:      false,
				Version:     "24.0.6",
				Path:        "/usr/bin/docker",
			},
			{
				Name:        "git",
				ShouldExist: true,
				Installed:   true,
				Passed:      true,
				Version:     "2.39.2",
				Path:        "/usr/bin/git",
			},
			{
				Name:        "nonexistent",
				ShouldExist: true,
				Installed:   false,
				Passed:      false,
				Error:       "程序未找到",
			},
		},
		Summary: config.ProgramCheckSummary{
			TotalChecked:   3,
			TotalInstalled: 2,
			TotalMissing:   1,
			TotalPassed:    1,
			TotalFailed:    2,
		},
	}
}

// createTestClockCheckResult 创建测试时钟检查结果
func createTestClockCheckResult() *config.ClockCheckResult {
	result := &config.ClockCheckResult{
		Timestamp: time.Now(),
		Result:    "fail",
		Clocks: []config.ClockCheckInfo{
			{
				Host:     "*",
				Role:     []string{"bootstrap"},
				IsSynced: true,
				TimeDiff: 0,
				IsLocal:  true,
				Error:    "",
			},
			{
				Host:     "**",
				Role:     []string{"worker"},
				IsSynced: false,
				TimeDiff: -1,
				IsLocal:  false,
				Error:    "failed to connect",
			},
			{
				Host:     "***",
				Role:     []string{"master"},
				IsSynced: true,
				TimeDiff: 2,
				IsLocal:  false,
				Error:    "",
			},
		},
	}
	return result
}

func TestNewFormatter(t *testing.T) {
	formatter := NewFormatter()

	if formatter == nil {
		t.Error("Expected Formatter instance, got nil")
	}
}

func TestFormatQueryText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestQueryResult()

	output, err := formatter.FormatQuery(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, output, "Query Time")
	assert.Contains(t, output, "/tmp/test.txt")
	assert.Contains(t, output, "/tmp/dir")
	assert.Contains(t, output, "/tmp/missing")
	assert.Contains(t, output, "Summary")
}

func TestFormatQueryJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestQueryResult()

	output, err := formatter.FormatQuery(result, "json")

	assert.NoError(t, err)
	assert.Contains(t, output, "timestamp")
	assert.Contains(t, output, "files")
	assert.Contains(t, output, "summary")
	// 验证为有效 JSON
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(output), &js))

	// 验证具体内容
	assert.Contains(t, output, "/tmp/test.txt")
	assert.Contains(t, output, "user1")
	assert.Contains(t, output, "group1")
}

func TestFormatCleanText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestCleanResult()

	output, err := formatter.FormatClean(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, output, "Clean Time")
	assert.Contains(t, output, "Deleted Files")
	assert.Contains(t, output, "Failed Files")
	assert.Contains(t, output, "Skipped Files")
	assert.Contains(t, output, "Summary")
}

func TestFormatCleanJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestCleanResult()

	output, err := formatter.FormatClean(result, "json")

	assert.NoError(t, err)
	assert.Contains(t, output, "timestamp")
	assert.Contains(t, output, "deleted")
	assert.Contains(t, output, "failed")
	assert.Contains(t, output, "skipped")
	assert.Contains(t, output, "summary")
	var js2 interface{}
	assert.NoError(t, json.Unmarshal([]byte(output), &js2))
}

func TestFormatProgramCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestProgramCheckResult()

	output, err := formatter.FormatProgramCheck(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, output, "Check Time")
	assert.Contains(t, output, "docker")
	assert.Contains(t, output, "git")
	assert.Contains(t, output, "nonexistent")
	assert.Contains(t, output, "Summary")
}

func TestFormatProgramCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestProgramCheckResult()

	output, err := formatter.FormatProgramCheck(result, "json")

	assert.NoError(t, err)
	assert.Contains(t, output, "timestamp")
	assert.Contains(t, output, "programs")
	assert.Contains(t, output, "summary")
	assert.Contains(t, output, "docker")
	assert.Contains(t, output, "24.0.6")
	var js3 interface{}
	assert.NoError(t, json.Unmarshal([]byte(output), &js3))
}

func TestFormatClockCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestClockCheckResult()

	out, err := formatter.FormatClockCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "Clock Check Time")
	assert.Contains(t, out, "Clock Check Result")
	assert.Contains(t, out, "ERROR") // negative TimeDiff rendered as ERROR
	assert.Contains(t, out, "failed to connect")
}

func TestFormatClockCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestClockCheckResult()

	out, err := formatter.FormatClockCheck(result, "json")
	assert.NoError(t, err)
	assert.Contains(t, out, "timestamp")
	assert.Contains(t, out, "clocks")
	assert.Contains(t, out, "Result")
	var js4 interface{}
	assert.NoError(t, json.Unmarshal([]byte(out), &js4))
}

// createTestRouteCheckResult creates test route check result
func createTestRouteCheckResult() *config.RouteCheckResult {
	return &config.RouteCheckResult{
		Timestamp: time.Now(),
		DefaultRoutes: []config.RouteInfo{
			{
				Exists:      true,
				Destination: "default",
				Gateway:     "192.168.1.1",
				Interface:   "eth0",
				Metric:      100,
			},
			{
				Exists:      true,
				Destination: "default",
				Gateway:     "192.168.2.1",
				Interface:   "eth1",
				Metric:      200,
			},
		},
		Summary: config.RouteSummary{
			TotalRoutes: 2,
			CheckPassed: true,
			Message:     "default route exists",
		},
	}
}

// createTestIptablesCheckResult creates test iptables check result
func createTestIptablesCheckResult() *config.IptablesCheckResult {
	return &config.IptablesCheckResult{
		Timestamp: time.Now(),
		ForwardPolicy: config.ForwardPolicy{
			Chain:    "FORWARD",
			Policy:   "ACCEPT",
			IsAccept: true,
		},
		Summary: config.IptablesSummary{
			CheckPassed: true,
			Message:     "FORWARD chain default policy is ACCEPT",
		},
	}
}

func TestFormatRouteCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestRouteCheckResult()

	output, err := formatter.FormatRouteCheck(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, output, "Route Check Time")
	assert.Contains(t, output, "Check Passed")
	assert.Contains(t, output, "192.168.1.1")
	assert.Contains(t, output, "eth0")
	assert.Contains(t, output, "Default Routes")
}

func TestFormatRouteCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestRouteCheckResult()

	output, err := formatter.FormatRouteCheck(result, "json")

	assert.NoError(t, err)
	assert.Contains(t, output, "timestamp")
	assert.Contains(t, output, "default_routes")
	assert.Contains(t, output, "summary")
	assert.Contains(t, output, "192.168.1.1")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(output), &js))
}

func TestFormatIptablesCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := createTestIptablesCheckResult()

	output, err := formatter.FormatIptablesCheck(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, output, "Iptables Check Time")
	assert.Contains(t, output, "Check Passed")
	assert.Contains(t, output, "FORWARD")
	assert.Contains(t, output, "ACCEPT")
	assert.Contains(t, output, "Forward Chain Policy")
}

func TestFormatIptablesCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestIptablesCheckResult()

	output, err := formatter.FormatIptablesCheck(result, "json")

	assert.NoError(t, err)
	assert.Contains(t, output, "timestamp")
	assert.Contains(t, output, "forward_policy")
	assert.Contains(t, output, "summary")
	assert.Contains(t, output, "ACCEPT")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(output), &js))
}

// createTestDispatchResult builds a dispatch result with structured detail data
// for port/route checks on one node and a failed node without results.
func createTestDispatchResult() *config.DispatchResult {
	portData, _ := json.Marshal(config.PortCheckResult{
		Timestamp: time.Now(),
		Ports: []config.PortCheckInfo{
			{Port: "6443", Protocol: "tcp", IsUsed: false},
			{Port: "8080", Protocol: "tcp", IsUsed: true, Process: "java"},
		},
		Summary: config.PortSummary{TotalChecked: 2, Used: 1, Free: 1},
	})
	routeData, _ := json.Marshal(config.RouteCheckResult{
		Timestamp: time.Now(),
		DefaultRoutes: []config.RouteInfo{
			{Exists: true, Destination: "default", Gateway: "192.168.1.1", Interface: "eth0", Metric: 100},
		},
		Summary: config.RouteSummary{TotalRoutes: 1, CheckPassed: true, Message: "default route exists"},
	})
	fileQueryData, _ := json.Marshal(config.QueryResult{
		Timestamp: time.Now(),
		Files: []config.FileInfo{
			{Path: "/root/.kube/config", Exists: true, IsDir: false, Owner: "root", Group: "root", Permissions: "-rw-------"},
			{Path: "/var/log/*.log", Exists: false, Error: "no files matched the wildcard pattern"},
		},
		Summary: config.QuerySummary{TotalChecked: 2, TotalExists: 1, TotalMissing: 1, TotalFiles: 1},
	})

	return &config.DispatchResult{
		Timestamp: time.Now(),
		Duration:  "5s",
		Nodes: []config.NodeResult{
			{
				IP:     "10.0.0.1",
				Role:   []string{"master"},
				Status: "success",
				Results: []config.CheckResult{
					{CheckType: "port", Status: "fail", Detail: "1 of 2 ports are occupied", DetailData: portData},
					{CheckType: "route", Status: "pass", Detail: "default route exists", DetailData: routeData},
					{CheckType: "fileQuery", Status: "fail", Detail: "1 of 2 residual paths exist", DetailData: fileQueryData},
					{CheckType: "clock", Status: "pass", Detail: "local time: 2026-06-24 10:00:00"},
				},
			},
			{
				IP:     "10.0.0.2",
				Role:   []string{"worker"},
				Status: "failed",
				Error:  "timeout waiting for result",
			},
		},
		Summary: config.DispatchSummary{TotalNodes: 2, SuccessNodes: 1, FailedNodes: 1, Result: "FAIL"},
	}
}

func TestFormatDispatchTextDetails(t *testing.T) {
	formatter := NewFormatter()
	result := createTestDispatchResult()

	out, err := formatter.FormatDispatch(result, "text")

	assert.NoError(t, err)
	// top-level summary
	assert.Contains(t, out, "Overall Result: FAIL")
	assert.Contains(t, out, "Detailed Results:")
	// per-node detail section header
	assert.Contains(t, out, "Node: 10.0.0.1 (master)")
	// per-check summary table
	assert.Contains(t, out, "Check Type")
	assert.Contains(t, out, "FAIL") // port fail uppercased
	// port detail sub-table
	assert.Contains(t, out, "6443")
	assert.Contains(t, out, "8080")
	assert.Contains(t, out, "Occupied")
	assert.Contains(t, out, "java")
	// route detail sub-table
	assert.Contains(t, out, "192.168.1.1")
	assert.Contains(t, out, "eth0")
	// file query detail sub-table (with expansion error column)
	assert.Contains(t, out, "/root/.kube/config")
	assert.Contains(t, out, "no files matched the wildcard pattern")
	// failed node appears in node table but has no detail section
	assert.Contains(t, out, "10.0.0.2")
	assert.NotContains(t, out, "Node: 10.0.0.2")
}

func TestFormatDispatchTextNoDetails(t *testing.T) {
	formatter := NewFormatter()
	result := &config.DispatchResult{
		Timestamp: time.Now(),
		Duration:  "1s",
		Nodes: []config.NodeResult{
			{IP: "10.0.0.1", Role: []string{"master"}, Status: "failed", Error: "timeout"},
		},
		Summary: config.DispatchSummary{TotalNodes: 1, FailedNodes: 1, Result: "FAIL"},
	}

	out, err := formatter.FormatDispatch(result, "text")

	assert.NoError(t, err)
	assert.Contains(t, out, "Overall Result: FAIL")
	assert.NotContains(t, out, "Detailed Results:")
}
