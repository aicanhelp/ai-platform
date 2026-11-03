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

package main

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/spf13/cobra"

	"env-check/pkg/config"
	"env-check/pkg/disk"
	"env-check/pkg/dispatch"
	"env-check/pkg/iptables"
	"env-check/pkg/kernel"
	"env-check/pkg/logger"
	"env-check/pkg/output"
	"env-check/pkg/port"
	"env-check/pkg/program"
	"env-check/pkg/query"
	"env-check/pkg/route"
)

const (
	defaultPollInterval    = 15
	defaultConcurrentLimit = 10
)

var (
	checkList string
	skipList  string
	timeout   int
	roles     string
	runLocal  bool
)

// runCmd run subcommand
var runCmd = &cobra.Command{
	Use:   "run",
	Short: "Dispatch and run checks on remote nodes",
	Long:  `Dispatch the envCheck binary to remote nodes and execute checks, then collect results.`,
	RunE:  runDispatch,
}

// runLocalCmd run-local subcommand (for internal use on remote nodes)
var runLocalCmd = &cobra.Command{
	Use:    "run-local",
	Short:  "Run checks locally (internal use)",
	Long:   `Run specified checks on the local node and save results.`,
	RunE:   runLocalChecks,
	Hidden: true,
}

// runDispatch executes dispatch operation
func runDispatch(cmd *cobra.Command, args []string) error {
	return runAction(cmd, args, config.OperationMode("dispatch"), actionHandler{
		execute:      doDispatch,
		generateHTML: generateDispatchHTML,
		format:       formatDispatchResult,
		baseName:     "dispatch-result",
	})
}

// doDispatch performs the actual dispatch execution
func doDispatch(cmd *cobra.Command, cfg *config.AppConfig, log logger.Logger) (interface{}, error) {
	if len(cfg.Hosts) == 0 {
		return nil, fmt.Errorf("no hosts configured in config file")
	}

	dispatchCfg := buildDispatchConfig(cfg)
	targetHosts := getTargetHosts(cfg)
	if len(targetHosts) == 0 {
		return nil, fmt.Errorf("no hosts match the specified roles")
	}

	runCfg := *cfg
	runCfg.Hosts = targetHosts

	d := dispatch.NewDispatcher(dispatchCfg, &runCfg, log)
	result, err := d.Execute()
	if err != nil {
		log.Error("dispatch failed: " + err.Error())
		return nil, err
	}

	log.Info(fmt.Sprintf("dispatch completed: total=%d, success=%d, failed=%d",
		result.Summary.TotalNodes, result.Summary.SuccessNodes, result.Summary.FailedNodes))

	return result, nil
}

// buildDispatchConfig builds dispatch config from app config and flags
func buildDispatchConfig(cfg *config.AppConfig) *config.DispatchConfig {
	dispatchCfg := cfg.Dispatch
	if dispatchCfg == nil {
		dispatchCfg = &config.DispatchConfig{
			Timeout:         timeout,
			PollInterval:    defaultPollInterval,
			WorkDir:         "/tmp/envcheck",
			ConcurrentLimit: defaultConcurrentLimit,
		}
	}

	if timeout > 0 {
		dispatchCfg.Timeout = timeout
	}
	if checkList != "" {
		dispatchCfg.Checks = splitAndTrim(checkList)
	}
	if skipList != "" {
		dispatchCfg.SkipChecks = splitAndTrim(skipList)
	}

	return dispatchCfg
}

// getTargetHosts returns target hosts filtered by roles
func getTargetHosts(cfg *config.AppConfig) []config.Host {
	if roles == "" {
		return cfg.Hosts
	}

	roleList := splitAndTrim(roles)
	filtered := filterHostsByRole(cfg.Hosts, roleList)
	return filtered
}

// generateDispatchHTML generates HTML report for dispatch result
func generateDispatchHTML(res interface{}) error {
	return output.GenerateDispatchHTML(*(res.(*config.DispatchResult)), "dispatch.html")
}

// formatDispatchResult formats dispatch result for output
func formatDispatchResult(res interface{}, outputFormat string) (string, error) {
	formatter := output.NewFormatter()
	return formatter.FormatDispatch(res.(*config.DispatchResult), outputFormat)
}

// runLocalChecks runs checks locally and saves results
func runLocalChecks(cmd *cobra.Command, args []string) error {
	cfg, err := loadLocalConfig()
	if err != nil {
		return fmt.Errorf("failed to load config: %v", err)
	}

	log := logger.NewLogger(cfg.LogFile)

	// Remote nodes receive roles via --roles flag from dispatcher
	nodeRoles := splitAndTrim(roles)
	log.Info(fmt.Sprintf("node roles: %v", nodeRoles))

	checks := splitAndTrim(checkList)
	if len(checks) == 0 {
		checks = []string{"kernel", "port", "disk", "clock", "fileQuery", "programCheck", "route", "iptables"}
	}

	log.Info(fmt.Sprintf("running local checks: %v", checks))

	results := runChecks(cfg, log, nodeRoles, checks)

	workDir := "/tmp/envcheck"
	if cfg.Dispatch != nil && cfg.Dispatch.WorkDir != "" {
		workDir = cfg.Dispatch.WorkDir
	}

	resultPath := workDir + "/result.json"
	if err := dispatch.SaveResults(results, resultPath); err != nil {
		errorPath := workDir + "/error.log"
		if writeErr := os.WriteFile(errorPath, []byte(err.Error()), config.FileMode); writeErr != nil {
			log.Error(fmt.Sprintf("failed to write error log: %v", writeErr))
		}
		return err
	}

	log.Info(fmt.Sprintf("results saved to %s", resultPath))
	return nil
}

// runChecks executes all specified checks and returns results
func runChecks(cfg *config.AppConfig, log logger.Logger, nodeRoles []string, checks []string) []config.CheckResult {
	var results []config.CheckResult
	for _, check := range checks {
		result := config.CheckResult{CheckType: check}

		switch check {
		case "kernel":
			result = runKernelCheck(cfg, log)
		case "port":
			result = runPortCheck(cfg, log, nodeRoles)
		case "disk":
			result, nodeRoles = runDiskCheck(cfg, log, nodeRoles)
		case "clock":
			result = runClockCheck()
		case "fileQuery":
			result = runFileQuery(cfg, log)
		case "programCheck":
			result = runProgramCheck(cfg, log)
		case "route":
			result = runRouteCheck(cfg, log)
		case "iptables":
			result = runIptablesCheck(cfg, log)
		default:
			result.Status = "skip"
			result.Detail = "unknown check type"
		}

		results = append(results, result)
		log.Info(fmt.Sprintf("check %s: %s", check, result.Status))
	}
	return results
}

// runKernelCheck runs kernel version check
func runKernelCheck(cfg *config.AppConfig, log logger.Logger) config.CheckResult {
	result := config.CheckResult{CheckType: "kernel"}
	if cfg.KernelCheck == nil {
		result.Status = "skip"
		result.Detail = "kernel check not configured"
		return result
	}

	k := kernel.NewChecker(cfg.KernelCheck, log)
	kernelResult, err := k.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result
	}

	if kernelResult.KernelInfo.IsValid {
		result.Status = "pass"
		result.Detail = fmt.Sprintf("kernel version %s meets requirement", kernelResult.KernelInfo.Version)
	} else {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("kernel version %s does not meet requirement %s %s",
			kernelResult.KernelInfo.Version, kernelResult.KernelInfo.Operator, kernelResult.KernelInfo.ExpectedVer)
	}
	return result
}

// runPortCheck runs port availability check
func runPortCheck(cfg *config.AppConfig, log logger.Logger, nodeRoles []string) config.CheckResult {
	result := config.CheckResult{CheckType: "port"}
	if cfg.PortCheck == nil {
		result.Status = "skip"
		result.Detail = "port check not configured"
		return result
	}

	p := port.NewChecker(cfg.PortCheck, log)
	portResult := p.Execute(nodeRoles)

	detailData, err := json.Marshal(portResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal port check result: %v", err)
		return result
	}
	result.DetailData = detailData

	if portResult.Summary.Used == 0 {
		result.Status = "pass"
		result.Detail = fmt.Sprintf("all %d ports are free", portResult.Summary.TotalChecked)
	} else {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("%d of %d ports are occupied", portResult.Summary.Used, portResult.Summary.TotalChecked)
	}
	return result
}

// runDiskCheck runs disk space check
func runDiskCheck(cfg *config.AppConfig, log logger.Logger, nodeRoles []string) (config.CheckResult, []string) {
	result := config.CheckResult{CheckType: "disk"}
	if cfg.DiskCheck == nil {
		result.Status = "skip"
		result.Detail = "disk check not configured"
		return result, nodeRoles
	}

	if len(nodeRoles) == 0 {
		for _, host := range cfg.Hosts {
			nodeRoles = append(nodeRoles, host.Role...)
		}
	}

	d := disk.NewChecker(cfg.DiskCheck, log, nodeRoles)
	diskResult, err := d.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result, nodeRoles
	}

	detailData, err := json.Marshal(diskResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal disk check result: %v", err)
		return result, nodeRoles
	}
	result.DetailData = detailData

	if diskResult.Summary.InsufficientPath == 0 {
		result.Status = "pass"
		result.Detail = fmt.Sprintf("all %d paths have sufficient space", diskResult.Summary.TotalChecked)
	} else {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("%d of %d paths have insufficient space",
			diskResult.Summary.InsufficientPath, diskResult.Summary.TotalChecked)
	}
	return result, nodeRoles
}

// runClockCheck runs clock check
func runClockCheck() config.CheckResult {
	return config.CheckResult{
		CheckType: "clock",
		Status:    "pass",
		Detail:    fmt.Sprintf("local time: %s", time.Now().Format("2006-01-02 15:04:05")),
	}
}

// runFileQuery runs file query check
func runFileQuery(cfg *config.AppConfig, log logger.Logger) config.CheckResult {
	result := config.CheckResult{CheckType: "fileQuery"}
	if len(cfg.Paths) == 0 {
		result.Status = "skip"
		result.Detail = "file query not configured (no paths)"
		return result
	}

	q := query.NewQuery(cfg, log)
	queryResult, err := q.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result
	}

	detailData, err := json.Marshal(queryResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal file query result: %v", err)
		return result
	}
	result.DetailData = detailData

	if queryResult.Summary.TotalExists == 0 {
		result.Status = "pass"
		result.Detail = fmt.Sprintf("all %d residual paths are absent", queryResult.Summary.TotalChecked)
	} else {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("%d of %d residual paths exist",
			queryResult.Summary.TotalExists, queryResult.Summary.TotalChecked)
	}
	return result
}

// runProgramCheck runs program existence check
func runProgramCheck(cfg *config.AppConfig, log logger.Logger) config.CheckResult {
	result := config.CheckResult{CheckType: "programCheck"}
	if len(cfg.ProgramList) == 0 {
		result.Status = "skip"
		result.Detail = "program check not configured (no program list)"
		return result
	}

	p := program.NewProgramChecker(cfg, log)
	programResult, err := p.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result
	}

	detailData, err := json.Marshal(programResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal program check result: %v", err)
		return result
	}
	result.DetailData = detailData

	if programResult.Summary.TotalFailed == 0 {
		result.Status = "pass"
		result.Detail = fmt.Sprintf("all %d programs match expected state", programResult.Summary.TotalChecked)
	} else {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("%d of %d programs do not match expected state",
			programResult.Summary.TotalFailed, programResult.Summary.TotalChecked)
	}
	return result
}

// runRouteCheck runs default route check
func runRouteCheck(cfg *config.AppConfig, log logger.Logger) config.CheckResult {
	result := config.CheckResult{CheckType: "route"}

	r := route.NewRouteChecker(cfg, log)
	routeResult, err := r.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result
	}

	detailData, err := json.Marshal(routeResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal route check result: %v", err)
		return result
	}
	result.DetailData = detailData

	if routeResult.Summary.CheckPassed {
		result.Status = "pass"
	} else {
		result.Status = "fail"
	}
	result.Detail = routeResult.Summary.Message
	return result
}

// runIptablesCheck runs iptables FORWARD chain policy check
func runIptablesCheck(cfg *config.AppConfig, log logger.Logger) config.CheckResult {
	result := config.CheckResult{CheckType: "iptables"}

	i := iptables.NewIptablesChecker(cfg, log)
	iptablesResult, err := i.Execute()
	if err != nil {
		result.Status = "fail"
		result.Detail = err.Error()
		return result
	}

	detailData, err := json.Marshal(iptablesResult)
	if err != nil {
		result.Status = "fail"
		result.Detail = fmt.Sprintf("failed to marshal iptables check result: %v", err)
		return result
	}
	result.DetailData = detailData

	if iptablesResult.Summary.CheckPassed {
		result.Status = "pass"
	} else {
		result.Status = "fail"
	}
	result.Detail = iptablesResult.Summary.Message
	return result
}

// splitAndTrim splits comma-separated string and trims spaces
func splitAndTrim(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	var result []string
	for _, p := range parts {
		trimmed := strings.TrimSpace(p)
		if trimmed != "" {
			result = append(result, trimmed)
		}
	}
	return result
}

// filterHostsByRole filters hosts by role
func filterHostsByRole(hosts []config.Host, roles []string) []config.Host {
	roleMap := make(map[string]bool)
	for _, r := range roles {
		roleMap[r] = true
	}

	var filtered []config.Host
	for _, host := range hosts {
		for _, hostRole := range host.Role {
			if roleMap[hostRole] {
				filtered = append(filtered, host)
				break
			}
		}
	}
	return filtered
}

// loadLocalConfig loads config from local file
func loadLocalConfig() (*config.AppConfig, error) {
	cfg := &config.AppConfig{}

	// Try to load from config.json
	if err := config.LoadConfigFromFile("config.json", cfg); err != nil {
		// Use default config
		cfg.LogFile = "./envCheck.log"
		cfg.OutputFormat = "text"
	}

	return cfg, nil
}
