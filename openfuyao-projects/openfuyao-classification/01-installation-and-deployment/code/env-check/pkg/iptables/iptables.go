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
	"bytes"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

type IptablesChecker struct {
	config *config.AppConfig
	logger logger.Logger
}

func NewIptablesChecker(cfg *config.AppConfig, log logger.Logger) *IptablesChecker {
	return &IptablesChecker{
		config: cfg,
		logger: log,
	}
}

func (i *IptablesChecker) Execute() (*config.IptablesCheckResult, error) {
	i.logger.Info("iptables check start......")

	result := &config.IptablesCheckResult{
		Timestamp: time.Now(),
		Summary:   config.IptablesSummary{},
	}

	policy, err := i.getForwardChainPolicy()
	if err != nil {
		result.Summary.CheckPassed = false
		result.Summary.Message = err.Error()
		i.logger.Error("failed to get FORWARD chain policy: " + err.Error())
		return result, nil
	}

	result.ForwardPolicy = config.ForwardPolicy{
		Chain:    "FORWARD",
		Policy:   policy,
		IsAccept: policy == "ACCEPT",
	}

	result.Summary.CheckPassed = result.ForwardPolicy.IsAccept
	if result.ForwardPolicy.IsAccept {
		result.Summary.Message = "FORWARD chain default policy is ACCEPT"
		i.logger.Info(result.Summary.Message)
	} else {
		result.Summary.Message = "FORWARD chain default policy is " + policy + ", expected ACCEPT"
		i.logger.Warning(result.Summary.Message)
	}

	i.logger.Info("iptables check completed")
	return result, nil
}

func (i *IptablesChecker) getForwardChainPolicy() (string, error) {
	output, err := i.executeIptablesCommand()
	if err != nil {
		return "", err
	}

	policy := i.parseIptablesOutput(output)
	if policy == "" {
		return "", fmt.Errorf("failed to parse FORWARD chain policy")
	}

	return policy, nil
}

func (i *IptablesChecker) executeIptablesCommand() (string, error) {
	if _, err := exec.LookPath("iptables"); err == nil {
		cmd := exec.Command("iptables", "-L", "FORWARD")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if err != nil {
			return "", fmt.Errorf("failed to execute iptables: %v, stderr: %s", err, stderr.String())
		}
		return stdout.String(), nil
	}

	if _, err := exec.LookPath("nft"); err == nil {
		cmd := exec.Command("nft", "list", "table", "ip", "filter")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if err != nil {
			return "", fmt.Errorf("failed to execute nft: %v, stderr: %s", err, stderr.String())
		}
		return stdout.String(), nil
	}

	return "", fmt.Errorf("no iptables command available (iptables or nft)")
}

func (i *IptablesChecker) parseIptablesOutput(output string) string {
	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "Chain") {
			re := regexp.MustCompile(`Chain\s+FORWARD\s*\(policy\s+(\w+)`)
			matches := re.FindStringSubmatch(line)
			if len(matches) > 1 {
				return matches[1]
			}
		}
	}

	return ""
}
