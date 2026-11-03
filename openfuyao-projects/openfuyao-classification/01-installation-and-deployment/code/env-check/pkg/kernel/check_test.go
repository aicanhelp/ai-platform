/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain n copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kernel

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func TestNewChecker(t *testing.T) {
	cfg := &config.KernelCheckConfig{MinVersion: "4.18", Operator: ">="}
	log := logger.NewLogger("")
	checker := NewChecker(cfg, log)
	assert.NotNil(t, checker)
	assert.Equal(t, cfg, checker.cfg)
	assert.Equal(t, log, checker.logger)
}

func TestCompareKernelVersion(t *testing.T) {
	tests := []struct {
		name     string
		current  string
		expected string
		operator string
		want     bool
	}{
		{"geNormalPass", "5.4.0", "4.18", ">=", true},
		{"geNormalFail", "4.15.0", "4.18", ">=", false},
		{"gtNormalPass", "5.4.0", "4.18", ">", true},
		{"gtNormalFail", "4.18.0", "4.18", ">", true},
		{"eqPass", "4.18.0", "4.18.0", "==", true},
		{"eqFail", "4.15.0", "4.18.0", "==", false},
		{"lePass", "4.15.0", "4.18", "<=", true},
		{"leFail", "5.4.0", "4.18", "<=", false},
		{"ltPass", "4.15.0", "4.18", "<", true},
		{"ltFail", "4.18.0", "4.18", "<", false},
		{"oldBuildGe", "3.10.0-1.2.el7.x86_64", "3.10.0", ">=", false},
		{"oldBuildGt", "3.10.0-1.2.el7.x86_64", "3.10.0", ">", false},
		{"oldBuildLe", "3.10.0-1.2.el7.x86_64", "4.18", "<=", true},
		{"oldBuildLt", "3.10.0-1.2.el7.x86_64", "4.18", "<", true},
		{"oldBuildEq", "3.10.0-1.2.el7.x86_64", "3.10.0-1.2.el7.x86_64", "==", true},
		{"notOldBuildGe", "3.10.0-1127.el7.x86_64", "3.10.0", ">=", true},
		{"defaultOperatorPass", "5.4.0", "4.18", "unknown", true},
		{"defaultOperatorFail", "4.15.0", "4.18", "unknown", false},
		{"oldBuildDefault", "3.10.0-1.2.el7.x86_64", "4.18", "unknown", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := compareKernelVersion(tt.current, tt.expected, tt.operator)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestCheckerExecute(t *testing.T) {
	cfg := &config.KernelCheckConfig{MinVersion: "1.0.0", Operator: ">="}
	log := logger.NewLogger("")
	checker := NewChecker(cfg, log)
	result, err := checker.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, 1, result.Summary.TotalChecked)
	assert.NotEmpty(t, result.KernelInfo.Version)
	assert.Equal(t, "1.0.0", result.KernelInfo.ExpectedVer)
	assert.Equal(t, ">=", result.KernelInfo.Operator)
	assert.Equal(t, runtime.GOOS, result.KernelInfo.OS)
	assert.Equal(t, runtime.GOARCH, result.KernelInfo.Arch)
	assert.True(t, result.KernelInfo.IsValid)
}

func TestCheckerExecuteNoVersionReq(t *testing.T) {
	cfg := &config.KernelCheckConfig{MinVersion: "", Operator: ">="}
	log := logger.NewLogger("")
	checker := NewChecker(cfg, log)
	result, err := checker.Execute()
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.True(t, result.KernelInfo.IsValid)
	assert.Equal(t, "no version requirement configured", result.KernelInfo.Error)
}
