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

// Package kernel performs kernel version check
package kernel

import (
	"fmt"
	"runtime"
	"strings"
	"time"

	"github.com/shirou/gopsutil/v3/host"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

const (
	kernelVersionMinLen = 12
	kernelPrefixLen     = 6
	kernelBuildOffset   = 7
	kernelBuildLen      = 4
	kernelPrefix        = "3.10.0"
)

// Checker kernel checker
type Checker struct {
	cfg    *config.KernelCheckConfig
	logger logger.Logger
}

// NewChecker creates a new kernel checker
func NewChecker(cfg *config.KernelCheckConfig, log logger.Logger) *Checker {
	return &Checker{
		cfg:    cfg,
		logger: log,
	}
}

// Execute executes kernel version check
func (c *Checker) Execute() (*config.KernelCheckResult, error) {
	c.logger.Info("kernel check start......")

	result := &config.KernelCheckResult{
		Timestamp: time.Now(),
		Summary:   config.KernelSummary{},
	}

	kernelInfo := c.checkVersion()
	result.KernelInfo = kernelInfo
	result.Summary.TotalChecked = 1

	if kernelInfo.IsValid {
		result.Summary.Passed = 1
		c.logger.Info(fmt.Sprintf("kernel version check passed: %s", kernelInfo.Version))
	} else {
		result.Summary.Failed = 1
		c.logger.Warning(fmt.Sprintf("kernel version check failed: current=%s, expected=%s %s",
			kernelInfo.Version, kernelInfo.Operator, kernelInfo.ExpectedVer))
	}

	c.logger.Info("kernel check completed")
	return result, nil
}

func (c *Checker) checkVersion() config.KernelInfo {
	h, _ := host.Info()

	info := config.KernelInfo{
		Version:     h.KernelVersion,
		OS:          runtime.GOOS,
		Arch:        runtime.GOARCH,
		ExpectedVer: c.cfg.MinVersion,
		Operator:    c.cfg.Operator,
	}

	if c.cfg.MinVersion == "" {
		info.IsValid = true
		info.Error = "no version requirement configured"
		return info
	}

	info.IsValid = compareKernelVersion(info.Version, c.cfg.MinVersion, c.cfg.Operator)
	return info
}

func compareKernelVersion(current, expected, operator string) bool {
	hasOldBuildNumber := len(current) > kernelVersionMinLen &&
		current[0:kernelPrefixLen] == kernelPrefix &&
		strings.Contains(current[kernelBuildOffset:kernelBuildOffset+kernelBuildLen], ".")

	switch operator {
	case ">=":
		return !hasOldBuildNumber && current >= expected
	case ">":
		return !hasOldBuildNumber && current > expected
	case "==", "=":
		return current == expected
	case "<=":
		return hasOldBuildNumber || current <= expected
	case "<":
		return hasOldBuildNumber || current < expected
	default:
		return !hasOldBuildNumber && current >= expected
	}
}
