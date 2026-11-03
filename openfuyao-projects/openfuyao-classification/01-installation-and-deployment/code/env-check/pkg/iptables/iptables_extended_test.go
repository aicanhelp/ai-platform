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

func TestExecuteIptablesCommand(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	_, err := checker.executeIptablesCommand()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no iptables command available")
}

func TestGetForwardChainPolicyNoCommand(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	_, err := checker.getForwardChainPolicy()
	assert.Error(t, err)
}

func TestExecuteNoCommand(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	result, err := checker.Execute()

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.False(t, result.Summary.CheckPassed)
	assert.NotEmpty(t, result.Summary.Message)
}

func TestParseIptablesOutputNft(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	output := `table ip filter {
    chain FORWARD {
        type filter hook forward priority 0; policy accept;
    }
}`

	policy := checker.parseIptablesOutput(output)

	assert.Equal(t, "", policy)
}

func TestParseIptablesOutputMultipleChains(t *testing.T) {
	cfg := &config.AppConfig{}
	log := logger.NewLogger("")
	checker := NewIptablesChecker(cfg, log)

	output := `Chain INPUT (policy ACCEPT)
Chain FORWARD (policy DROP)
Chain OUTPUT (policy ACCEPT)`

	policy := checker.parseIptablesOutput(output)

	assert.Equal(t, "DROP", policy)
}
