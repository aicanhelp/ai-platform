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

func TestFormatKernelCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := &config.KernelCheckResult{
		Timestamp: time.Now(),
		KernelInfo: config.KernelInfo{
			Version:     "5.15.0-101-generic",
			OS:          "ubuntu",
			Arch:        "amd64",
			ExpectedVer: "4.19",
			Operator:    ">=",
			IsValid:     true,
		},
		Summary: config.KernelSummary{TotalChecked: 1, Passed: 1, Failed: 0},
	}

	out, err := formatter.FormatKernelCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "Kernel Check Time")
	assert.Contains(t, out, "Current Version")
	assert.Contains(t, out, "5.15.0-101-generic")
	assert.Contains(t, out, "ubuntu")
	assert.Contains(t, out, "amd64")
	assert.Contains(t, out, "PASS")
}

func TestFormatKernelCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := &config.KernelCheckResult{
		Timestamp: time.Now(),
		KernelInfo: config.KernelInfo{
			Version:     "5.15.0-101-generic",
			OS:          "ubuntu",
			Arch:        "amd64",
			ExpectedVer: "4.19",
			Operator:    ">=",
			IsValid:     true,
		},
		Summary: config.KernelSummary{TotalChecked: 1, Passed: 1, Failed: 0},
	}

	out, err := formatter.FormatKernelCheck(result, "json")
	assert.NoError(t, err)
	assert.Contains(t, out, "kernel_info")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(out), &js))
}

func TestFormatKernelCheckTextFail(t *testing.T) {
	formatter := NewFormatter()
	result := &config.KernelCheckResult{
		Timestamp: time.Now(),
		KernelInfo: config.KernelInfo{
			Version:     "3.10.0-1160.el7.x86_64",
			OS:          "centos",
			Arch:        "amd64",
			ExpectedVer: "4.19",
			Operator:    ">=",
			IsValid:     false,
		},
		Summary: config.KernelSummary{TotalChecked: 1, Passed: 0, Failed: 1},
	}

	out, err := formatter.FormatKernelCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "FAIL")
	assert.Contains(t, out, "3.10.0-1160.el7.x86_64")
}

func TestFormatKernelCheckTextWithError(t *testing.T) {
	formatter := NewFormatter()
	result := &config.KernelCheckResult{
		Timestamp: time.Now(),
		KernelInfo: config.KernelInfo{
			Version:     "",
			OS:          "ubuntu",
			Arch:        "amd64",
			ExpectedVer: "4.19",
			Operator:    ">=",
			IsValid:     false,
			Error:       "failed to get kernel version",
		},
		Summary: config.KernelSummary{TotalChecked: 1, Passed: 0, Failed: 1},
	}

	out, err := formatter.FormatKernelCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "FAIL")
	assert.Contains(t, out, "Error")
	assert.Contains(t, out, "failed to get kernel version")
}

func TestFormatPortCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := &config.PortCheckResult{
		Timestamp: time.Now(),
		Ports: []config.PortCheckInfo{
			{Port: "6443", Protocol: "tcp", IsUsed: false},
			{Port: "8080", Protocol: "tcp", IsUsed: true, Process: "java"},
		},
		Summary: config.PortSummary{TotalChecked: 2, Used: 1, Free: 1},
	}

	out, err := formatter.FormatPortCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "Port Check Time")
	assert.Contains(t, out, "6443")
	assert.Contains(t, out, "8080")
	assert.Contains(t, out, "Free")
	assert.Contains(t, out, "Occupied")
	assert.Contains(t, out, "java")
}

func TestFormatPortCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := &config.PortCheckResult{
		Timestamp: time.Now(),
		Ports: []config.PortCheckInfo{
			{Port: "6443", Protocol: "tcp", IsUsed: false},
			{Port: "8080", Protocol: "tcp", IsUsed: true, Process: "java"},
		},
		Summary: config.PortSummary{TotalChecked: 2, Used: 1, Free: 1},
	}

	out, err := formatter.FormatPortCheck(result, "json")
	assert.NoError(t, err)
	assert.Contains(t, out, "ports")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(out), &js))
}

func TestFormatDiskCheckText(t *testing.T) {
	formatter := NewFormatter()
	result := &config.DiskCheckResult{
		Timestamp: time.Now(),
		Spaces: []config.DiskSpace{
			{
				Path:         "/",
				Total:        100 * bytesPerGB,
				Free:         50 * bytesPerGB,
				Used:         50 * bytesPerGB,
				UsedPercent:  50.0,
				MinFree:      20 * bytesPerGB,
				IsSufficient: true,
				Filesystem:   "/dev/sda1",
			},
			{
				Path:         "/var",
				Total:        50 * bytesPerGB,
				Free:         5 * bytesPerGB,
				Used:         45 * bytesPerGB,
				UsedPercent:  90.0,
				MinFree:      30 * bytesPerGB,
				IsSufficient: false,
				Filesystem:   "/dev/sda2",
			},
		},
		Summary: config.DiskSummary{TotalChecked: 2, SufficientPath: 1, InsufficientPath: 1},
	}

	out, err := formatter.FormatDiskCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "Disk Check Time")
	assert.Contains(t, out, "/dev/sda1")
	assert.Contains(t, out, "/dev/sda2")
	assert.Contains(t, out, "PASS")
	assert.Contains(t, out, "FAIL")
}

func TestFormatDiskCheckJSON(t *testing.T) {
	formatter := NewFormatter()
	result := &config.DiskCheckResult{
		Timestamp: time.Now(),
		Spaces: []config.DiskSpace{
			{
				Path:         "/",
				Total:        100 * bytesPerGB,
				Free:         50 * bytesPerGB,
				Used:         50 * bytesPerGB,
				UsedPercent:  50.0,
				MinFree:      20 * bytesPerGB,
				IsSufficient: true,
				Filesystem:   "/dev/sda1",
			},
		},
		Summary: config.DiskSummary{TotalChecked: 1, SufficientPath: 1, InsufficientPath: 0},
	}

	out, err := formatter.FormatDiskCheck(result, "json")
	assert.NoError(t, err)
	assert.Contains(t, out, "spaces")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(out), &js))
}

func TestFormatDiskCheckTextWithErrorAndEmptyFS(t *testing.T) {
	formatter := NewFormatter()
	result := &config.DiskCheckResult{
		Timestamp: time.Now(),
		Spaces: []config.DiskSpace{
			{
				Path:         "/opt",
				Total:        0,
				Free:         0,
				Used:         0,
				UsedPercent:  0,
				MinFree:      10 * bytesPerGB,
				IsSufficient: false,
				Filesystem:   "",
				Error:        "statfs failed",
			},
		},
		Summary: config.DiskSummary{TotalChecked: 1, SufficientPath: 0, InsufficientPath: 1},
	}

	out, err := formatter.FormatDiskCheck(result, "text")
	assert.NoError(t, err)
	assert.Contains(t, out, "ERROR")
	assert.NotContains(t, out, "/dev/")
}

func TestFormatDispatchJSON(t *testing.T) {
	formatter := NewFormatter()
	result := createTestDispatchResult()

	out, err := formatter.FormatDispatch(result, "json")
	assert.NoError(t, err)
	assert.Contains(t, out, "timestamp")
	var js interface{}
	assert.NoError(t, json.Unmarshal([]byte(out), &js))
}

func TestTruncateString(t *testing.T) {
	assert.Equal(t, "hello", truncateString("hello", 10))
	assert.Equal(t, "hel...", truncateString("hello world", 3))
	assert.Equal(t, "", truncateString("", 5))
	assert.Equal(t, "abc", truncateString("abc", 3))
}

func TestOrDash(t *testing.T) {
	assert.Equal(t, "-", orDash(""))
	assert.Equal(t, "value", orDash("value"))
}

func TestDivideFloat(t *testing.T) {
	assert.Equal(t, 25.0, divideFloat(100, 4))
	assert.Equal(t, 0.0, divideFloat(100, 0))
	assert.Equal(t, 0.0, divideFloat(0, 0))
}

func TestBuildDiskSubTable(t *testing.T) {
	formatter := NewFormatter()

	result := formatter.buildDiskSubTable(nil)
	assert.Equal(t, "", result)

	result = formatter.buildDiskSubTable([]config.DiskSpace{
		{Path: "/", Total: 100 * bytesPerGB, Free: 50 * bytesPerGB, Used: 50 * bytesPerGB, UsedPercent: 50.0, MinFree: 20 * bytesPerGB, IsSufficient: true, Filesystem: "/dev/sda1"},
	})
	assert.Contains(t, result, "/")
	assert.Contains(t, result, "PASS")
}

func TestBuildProgramSubTable(t *testing.T) {
	formatter := NewFormatter()

	result := formatter.buildProgramSubTable(nil)
	assert.Equal(t, "", result)

	result = formatter.buildProgramSubTable([]config.ProgramInfo{
		{Name: "docker", ShouldExist: false, Installed: true, Passed: false, Version: "24.0.6", Path: "/usr/bin/docker"},
	})
	assert.Contains(t, result, "docker")
	assert.Contains(t, result, "Yes")
	assert.Contains(t, result, "24.0.6")
}

func TestBuildIptablesSubTable(t *testing.T) {
	formatter := NewFormatter()

	result := formatter.buildIptablesSubTable(config.ForwardPolicy{})
	assert.Equal(t, "", result)

	result = formatter.buildIptablesSubTable(config.ForwardPolicy{
		Chain:    "FORWARD",
		Policy:   "ACCEPT",
		IsAccept: true,
	})
	assert.Contains(t, result, "FORWARD")
	assert.Contains(t, result, "ACCEPT")
	assert.Contains(t, result, "true")
}

func TestGetDiskDetails(t *testing.T) {
	var result config.CheckResult
	assert.Nil(t, getDiskDetails(result))

	diskData, _ := json.Marshal(config.DiskCheckResult{
		Spaces: []config.DiskSpace{{Path: "/", Total: 100, IsSufficient: true}},
	})
	result = config.CheckResult{CheckType: "disk", DetailData: diskData}
	spaces := getDiskDetails(result)
	assert.Len(t, spaces, 1)
	assert.Equal(t, "/", spaces[0].Path)

	result = config.CheckResult{CheckType: "disk", DetailData: json.RawMessage("invalid json")}
	assert.Nil(t, getDiskDetails(result))
}

func TestGetDiskSummary(t *testing.T) {
	var result config.CheckResult
	summary := getDiskSummary(result)
	assert.Equal(t, config.DiskSummary{}, summary)

	diskData, _ := json.Marshal(config.DiskCheckResult{
		Summary: config.DiskSummary{TotalChecked: 2, SufficientPath: 1, InsufficientPath: 1},
	})
	result = config.CheckResult{CheckType: "disk", DetailData: diskData}
	summary = getDiskSummary(result)
	assert.Equal(t, 2, summary.TotalChecked)
	assert.Equal(t, 1, summary.SufficientPath)
	assert.Equal(t, 1, summary.InsufficientPath)

	result = config.CheckResult{CheckType: "disk", DetailData: json.RawMessage("invalid json")}
	summary = getDiskSummary(result)
	assert.Equal(t, config.DiskSummary{}, summary)
}

func TestGetProgramCheckDetails(t *testing.T) {
	var result config.CheckResult
	assert.Nil(t, getProgramCheckDetails(result))

	progData, _ := json.Marshal(config.ProgramCheckResult{
		Programs: []config.ProgramInfo{{Name: "docker", Installed: true}},
	})
	result = config.CheckResult{CheckType: "programCheck", DetailData: progData}
	programs := getProgramCheckDetails(result)
	assert.Len(t, programs, 1)
	assert.Equal(t, "docker", programs[0].Name)

	result = config.CheckResult{CheckType: "programCheck", DetailData: json.RawMessage("invalid json")}
	assert.Nil(t, getProgramCheckDetails(result))
}

func TestGetIptablesPolicy(t *testing.T) {
	var result config.CheckResult
	policy := getIptablesPolicy(result)
	assert.Equal(t, config.ForwardPolicy{}, policy)

	iptData, _ := json.Marshal(config.IptablesCheckResult{
		ForwardPolicy: config.ForwardPolicy{Chain: "FORWARD", Policy: "ACCEPT", IsAccept: true},
	})
	result = config.CheckResult{CheckType: "iptables", DetailData: iptData}
	policy = getIptablesPolicy(result)
	assert.Equal(t, "FORWARD", policy.Chain)
	assert.Equal(t, "ACCEPT", policy.Policy)
	assert.True(t, policy.IsAccept)

	result = config.CheckResult{CheckType: "iptables", DetailData: json.RawMessage("invalid json")}
	policy = getIptablesPolicy(result)
	assert.Equal(t, config.ForwardPolicy{}, policy)
}

func TestFormatCheckSubTable(t *testing.T) {
	formatter := NewFormatter()

	diskData, _ := json.Marshal(config.DiskCheckResult{
		Spaces: []config.DiskSpace{{Path: "/", Total: 100, IsSufficient: true}},
	})
	diskResult := config.CheckResult{CheckType: "disk", DetailData: diskData}
	out := formatter.formatCheckSubTable(diskResult)
	assert.NotEmpty(t, out)

	progData, _ := json.Marshal(config.ProgramCheckResult{
		Programs: []config.ProgramInfo{{Name: "docker", Installed: true}},
	})
	progResult := config.CheckResult{CheckType: "programCheck", DetailData: progData}
	out = formatter.formatCheckSubTable(progResult)
	assert.NotEmpty(t, out)

	iptData, _ := json.Marshal(config.IptablesCheckResult{
		ForwardPolicy: config.ForwardPolicy{Chain: "FORWARD", Policy: "ACCEPT", IsAccept: true},
	})
	iptResult := config.CheckResult{CheckType: "iptables", DetailData: iptData}
	out = formatter.formatCheckSubTable(iptResult)
	assert.NotEmpty(t, out)

	unknownResult := config.CheckResult{CheckType: "unknown", DetailData: nil}
	out = formatter.formatCheckSubTable(unknownResult)
	assert.Equal(t, "", out)
}
