/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package output

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
)

func TestGenerateQueryHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "query.html")
	result := config.QueryResult{
		Timestamp: time.Now(),
		Files: []config.FileInfo{
			{Path: "/tmp/test.txt", Exists: true, IsDir: false, Owner: "root", Group: "root", Permissions: "-rw-r--r--"},
		},
		Summary: config.QuerySummary{TotalChecked: 1, TotalExists: 1, TotalFiles: 1},
	}
	err := GenerateQueryHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateCleanHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "clean.html")
	result := config.CleanResult{
		Timestamp: time.Now(),
		Deleted: []config.FileInfo{
			{Path: "/tmp/deleted.txt", IsDir: false},
		},
		Summary: config.CleanSummary{TotalChecked: 1, TotalDeleted: 1},
	}
	err := GenerateCleanHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateCheckHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "check.html")
	result := config.ProgramCheckResult{
		Timestamp: time.Now(),
		Programs: []config.ProgramInfo{
			{Name: "docker", ShouldExist: false, Installed: true, Passed: false, Version: "24.0.6", Path: "/usr/bin/docker"},
		},
		Summary: config.ProgramCheckSummary{TotalChecked: 1, TotalInstalled: 1, TotalFailed: 1},
	}
	err := GenerateCheckHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateClockHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "clock.html")
	result := config.ClockCheckResult{
		Timestamp: time.Now(),
		Result:    "pass",
		Clocks: []config.ClockCheckInfo{
			{Host: "10.0.0.1", Role: []string{"bootstrap"}, IsSynced: true, TimeDiff: 0, IsLocal: true},
		},
	}
	err := GenerateClockHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateRouteHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "route.html")
	result := config.RouteCheckResult{
		Timestamp: time.Now(),
		DefaultRoutes: []config.RouteInfo{
			{Exists: true, Destination: "default", Gateway: "192.168.1.1", Interface: "eth0", Metric: 100},
		},
		Summary: config.RouteSummary{TotalRoutes: 1, CheckPassed: true, Message: "default route exists"},
	}
	err := GenerateRouteHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateIptablesHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "iptables.html")
	result := config.IptablesCheckResult{
		Timestamp:     time.Now(),
		ForwardPolicy: config.ForwardPolicy{Chain: "FORWARD", Policy: "ACCEPT", IsAccept: true},
		Summary:       config.IptablesSummary{CheckPassed: true, Message: "FORWARD chain default policy is ACCEPT"},
	}
	err := GenerateIptablesHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateKernelHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "kernel.html")
	result := config.KernelCheckResult{
		Timestamp:  time.Now(),
		KernelInfo: config.KernelInfo{Version: "5.15.0", OS: "ubuntu", Arch: "amd64", ExpectedVer: "4.19", Operator: ">=", IsValid: true},
		Summary:    config.KernelSummary{TotalChecked: 1, Passed: 1},
	}
	err := GenerateKernelHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGeneratePortHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "port.html")
	result := config.PortCheckResult{
		Timestamp: time.Now(),
		Ports: []config.PortCheckInfo{
			{Port: "6443", Protocol: "tcp", IsUsed: false},
		},
		Summary: config.PortSummary{TotalChecked: 1, Free: 1},
	}
	err := GeneratePortHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateDiskHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "disk.html")
	result := config.DiskCheckResult{
		Timestamp: time.Now(),
		Spaces: []config.DiskSpace{
			{Path: "/", Total: 100 * (1 << 30), Free: 50 * (1 << 30), Used: 50 * (1 << 30), UsedPercent: 50.0, MinFree: 20 * (1 << 30), IsSufficient: true, Filesystem: "/dev/sda1"},
		},
		Summary: config.DiskSummary{TotalChecked: 1, SufficientPath: 1},
	}
	err := GenerateDiskHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateDispatchHTML(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "dispatch.html")
	result := config.DispatchResult{
		Timestamp: time.Now(),
		Duration:  "5s",
		Nodes: []config.NodeResult{
			{IP: "10.0.0.1", Role: []string{"master"}, Status: "success", StartTime: time.Now(), EndTime: time.Now()},
		},
		Summary: config.DispatchSummary{TotalNodes: 1, SuccessNodes: 1, Result: "PASS"},
	}
	err := GenerateDispatchHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}

func TestGenerateHTMLCreatesParentDir(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "sub", "deep", "query.html")
	result := config.QueryResult{Timestamp: time.Now()}
	err := GenerateQueryHTML(result, outPath)
	assert.NoError(t, err)
	_, err = os.Stat(outPath)
	assert.NoError(t, err)
}

func TestGenerateQueryHTMLEmptyResult(t *testing.T) {
	dir := t.TempDir()
	outPath := filepath.Join(dir, "empty.html")
	result := config.QueryResult{}
	err := GenerateQueryHTML(result, outPath)
	assert.NoError(t, err)
	info, err := os.Stat(outPath)
	assert.NoError(t, err)
	assert.Greater(t, info.Size(), int64(0))
}
