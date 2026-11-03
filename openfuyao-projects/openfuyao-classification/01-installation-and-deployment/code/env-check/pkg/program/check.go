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

// Package program handles application-related operations, currently only includes existence detection
package program

import (
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"runtime"
	"strings"
	"time"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

// ApplicationChecker program checker
type ApplicationChecker struct {
	config *config.AppConfig
	logger logger.Logger
}

// NewProgramChecker creates a new program checker
func NewProgramChecker(cfg *config.AppConfig, log logger.Logger) *ApplicationChecker {
	return &ApplicationChecker{
		config: cfg,
		logger: log,
	}
}

// Execute executes program check
func (p *ApplicationChecker) Execute() (*config.ProgramCheckResult, error) {
	p.logger.Info("program check start......")
	sysInfo := p.getSystemInfo()
	p.logger.Info(fmt.Sprintf("os: %s, arch: %s", sysInfo["os"], sysInfo["arch"]))
	p.logger.Info(fmt.Sprintf("program list to check: %v", p.config.ProgramList))

	result := &config.ProgramCheckResult{
		Timestamp: time.Now(),
		Programs:  make([]config.ProgramInfo, 0),
		Summary:   config.ProgramCheckSummary{},
	}

	// Check all programs
	for _, program := range p.config.ProgramList {
		programInfo := p.checkProgram(program)
		result.Programs = append(result.Programs, programInfo)
		p.updateProgramSummary(programInfo, result)
	}

	p.logger.Info("program check completed")
	return result, nil
}

// checkProgram checks a single program
func (p *ApplicationChecker) checkProgram(program config.ProgramConfig) config.ProgramInfo {
	p.logger.Info(fmt.Sprintf("check program: %s (should_exist: %v)", program.Name, program.ShouldExist))

	info := config.ProgramInfo{
		Name:        program.Name,
		ShouldExist: program.ShouldExist,
	}

	// Find program path
	path, err := exec.LookPath(program.Name)
	if err != nil {
		info.Installed = false
		info.Error = fmt.Sprintf("not found: %v", err)
		info.Passed = !program.ShouldExist
		p.logger.Info(fmt.Sprintf("not install: %s, passed: %v", program.Name, info.Passed))
		return info
	}

	info.Installed = true
	info.Path = path

	// Try to get version information
	version, err := p.getProgramVersion(program.Name)
	if err == nil {
		info.Version = version
		p.logger.Info(fmt.Sprintf("%s installed - version: %s", program.Name, version))
	} else {
		p.logger.Info(fmt.Sprintf("%s installed - path: %s", program.Name, path))
	}

	info.Passed = program.ShouldExist
	p.logger.Info(fmt.Sprintf("%s check passed: %v", program.Name, info.Passed))

	return info
}

// getProgramVersion gets program version
func (p *ApplicationChecker) getProgramVersion(program string) (string, error) {
	cmd := exec.Command(program, "--version")
	switch program {
	case "go":
		cmd = exec.Command("go", "version")
	case "java":
		cmd = exec.Command("java", "-version")
	case "kubectl":
		cmd = exec.Command("kubectl", "version", "--client")
	case "helm":
		cmd = exec.Command("helm", "version", "--short")
	default:
		// cmd already initialized to the default "--version" invocation
	}

	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}

	return p.extractVersion(string(output)), nil
}

// extractVersion extracts version number from output
func (p *ApplicationChecker) extractVersion(output string) string {
	output = strings.TrimSpace(output)

	patterns := []string{
		`v?(\d+\.\d+\.\d+[a-zA-Z0-9\-\.]*)`,
		`v?(\d+\.\d+\.\d+)`,
		`v?(\d+\.\d+)`,
	}

	for _, pattern := range patterns {
		re := regexp.MustCompile(pattern)
		matches := re.FindStringSubmatch(output)
		if len(matches) > 1 {
			return matches[1]
		}
	}

	return output
}

// updateProgramSummary updates program check summary information
func (p *ApplicationChecker) updateProgramSummary(programInfo config.ProgramInfo, result *config.ProgramCheckResult) {
	result.Summary.TotalChecked++

	if programInfo.Installed {
		result.Summary.TotalInstalled++
	} else {
		result.Summary.TotalMissing++
	}

	if programInfo.Passed {
		result.Summary.TotalPassed++
	} else {
		result.Summary.TotalFailed++
	}
}

// getSystemInfo gets system information
func (p *ApplicationChecker) getSystemInfo() map[string]string {
	info := make(map[string]string)

	info["os"] = runtime.GOOS
	info["arch"] = runtime.GOARCH

	// Get other system information
	if hostname, err := os.Hostname(); err == nil {
		info["hostname"] = hostname
	}

	return info
}
