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

package iptables

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

func createTestIptablesConfig() *config.AppConfig {
	return &config.AppConfig{}
}

func TestNewIptablesChecker(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")

	checker := NewIptablesChecker(cfg, log)

	assert.NotNil(t, checker)
	assert.Equal(t, cfg, checker.config)
}

func TestParseIptablesOutputAccept(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	output := `Chain FORWARD (policy ACCEPT)
target     prot opt source               destination
ACCEPT     all  --  anywhere             anywhere`

	policy := checker.parseIptablesOutput(output)

	assert.Equal(t, "ACCEPT", policy)
}

func TestParseIptablesOutputDrop(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	output := `Chain FORWARD (policy DROP)
target     prot opt source               destination`

	policy := checker.parseIptablesOutput(output)

	assert.Equal(t, "DROP", policy)
}

func TestParseIptablesOutputEmpty(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	output := `Chain INPUT (policy ACCEPT)
target     prot opt source               destination`

	policy := checker.parseIptablesOutput(output)

	assert.Equal(t, "", policy)
}

func TestIptablesCheckerExecute(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	if result.ForwardPolicy.Chain == "FORWARD" {
		// 命令可用
		if result.ForwardPolicy.IsAccept {
			assert.True(t, result.Summary.CheckPassed)
			assert.Equal(t, "FORWARD chain default policy is ACCEPT", result.Summary.Message)
		} else {
			assert.False(t, result.Summary.CheckPassed)
			assert.Contains(t, result.Summary.Message, "expected ACCEPT")
		}
	} else {
		// 命令不可用
		assert.False(t, result.Summary.CheckPassed)
		assert.NotEmpty(t, result.Summary.Message)
	}
}

func TestForwardPolicyIsAccept(t *testing.T) {
	policy := config.ForwardPolicy{
		Chain:    "FORWARD",
		Policy:   "ACCEPT",
		IsAccept: true,
	}

	assert.True(t, policy.IsAccept)
	assert.Equal(t, "ACCEPT", policy.Policy)
}

func TestForwardPolicyIsDrop(t *testing.T) {
	policy := config.ForwardPolicy{
		Chain:    "FORWARD",
		Policy:   "DROP",
		IsAccept: false,
	}

	assert.False(t, policy.IsAccept)
	assert.Equal(t, "DROP", policy.Policy)
}

func TestIptablesCheckResultSummary(t *testing.T) {
	cfg := createTestIptablesConfig()
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	if result.ForwardPolicy.Chain == "FORWARD" {
		if result.ForwardPolicy.IsAccept {
			assert.True(t, result.Summary.CheckPassed)
			assert.Equal(t, "FORWARD chain default policy is ACCEPT", result.Summary.Message)
		} else {
			assert.False(t, result.Summary.CheckPassed)
			assert.Contains(t, result.Summary.Message, "expected ACCEPT")
		}
	} else {
		assert.False(t, result.Summary.CheckPassed)
		assert.NotEmpty(t, result.Summary.Message)
	}
}