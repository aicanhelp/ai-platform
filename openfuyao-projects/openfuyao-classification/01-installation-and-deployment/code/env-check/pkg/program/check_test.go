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

package program

import (
	"testing"
	
	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

const defaultProgramsLen = 3 // 默认检查程序个数

// createTestConfig 创建测试配置
func createTestConfig() *config.AppConfig {
	return &config.AppConfig{
		ProgramList: []config.ProgramConfig{
			{Name: "go", ShouldExist: false},
			{Name: "docker", ShouldExist: false},
			{Name: "nonexistent", ShouldExist: true},
		},
	}
}

// createTestProgramInfo 创建测试程序信息
func createTestProgramInfo() config.ProgramInfo {
	return config.ProgramInfo{
		Name:        "test-program",
		ShouldExist: false,
		Installed:   true,
		Passed:     true,
		Version:    "1.0.0",
		Path:       "/usr/bin/test-program",
	}
}

func TestNewProgramChecker(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")

	checker := NewProgramChecker(cfg, log)

	if checker == nil {
		t.Error("Expected ApplicationChecker instance, got nil")
	}
	if checker.config != cfg {
		t.Error("Config not set correctly")
	}
}

func TestProgramCheckerExecuteSuccess(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, defaultProgramsLen, len(result.Programs))
	assert.Equal(t, defaultProgramsLen, result.Summary.TotalChecked)
}

func TestProgramCheckerExecuteEmptyProgramList(t *testing.T) {
	cfg := &config.AppConfig{ProgramList: []config.ProgramConfig{}}
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, 0, len(result.Programs))
	assert.Equal(t, 0, result.Summary.TotalChecked)
}

func TestCheckProgramInstalled(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	programInfo := checker.checkProgram(config.ProgramConfig{Name: "go", ShouldExist: false})

	assert.Equal(t, "go", programInfo.Name)
	// 我们不关心具体安装状态，只关心函数正常执行
	if programInfo.Installed {
		t.Logf("Go is installed: %s", programInfo.Version)
	} else {
		t.Log("Go is not installed")
	}
}

func TestCheckProgramNotInstalled(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	programInfo := checker.checkProgram(config.ProgramConfig{Name: "nonexistent-program-xyz", ShouldExist: true})

	assert.Equal(t, "nonexistent-program-xyz", programInfo.Name)
	assert.Equal(t, false, programInfo.Installed)
	assert.NotEqual(t, "", programInfo.Error)
}

func TestGetProgramVersionSuccess(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	version, err := checker.getProgramVersion("go")

	if err == nil {
		assert.NotEqual(t, "", version)
		t.Logf("Got version: %s", version)
	} else {
		t.Logf("Cannot get version: %v", err)
	}
}

func TestGetProgramVersionCommandError(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	_, err := checker.getProgramVersion("nonexistent-command")

	assert.Error(t, err)
}

func TestUpdateProgramSummaryInstalled(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	result := &config.ProgramCheckResult{
		Summary: config.ProgramCheckSummary{},
	}
	programInfo := createTestProgramInfo()

	checker.updateProgramSummary(programInfo, result)

	assert.Equal(t, 1, result.Summary.TotalChecked)
	assert.Equal(t, 1, result.Summary.TotalInstalled)
	assert.Equal(t, 0, result.Summary.TotalMissing)
	assert.Equal(t, 1, result.Summary.TotalPassed)
	assert.Equal(t, 0, result.Summary.TotalFailed)
}

func TestUpdateProgramSummaryNotInstalled(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	result := &config.ProgramCheckResult{
		Summary: config.ProgramCheckSummary{},
	}
	programInfo := config.ProgramInfo{
		Name:        "missing-program",
		Installed:   false,
		ShouldExist: true,
		Passed:      false,
		Error:       "not found",
	}

	checker.updateProgramSummary(programInfo, result)

	assert.Equal(t, 1, result.Summary.TotalChecked)
	assert.Equal(t, 0, result.Summary.TotalInstalled)
	assert.Equal(t, 1, result.Summary.TotalMissing)
	assert.Equal(t, 0, result.Summary.TotalPassed)
	assert.Equal(t, 1, result.Summary.TotalFailed)
}

func TestGetSystemInfo(t *testing.T) {
	cfg := createTestConfig()
	log := logger.NewLogger("")
	checker := NewProgramChecker(cfg, log)

	sysInfo := checker.getSystemInfo()

	assert.NotNil(t, sysInfo)
	assert.NotEqual(t, "", sysInfo["os"])
	assert.NotEqual(t, "", sysInfo["arch"])

	t.Logf("OS: %s, Arch: %s", sysInfo["os"], sysInfo["arch"])
}
