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

// Package output handles formatted output
package output

import (
	"embed"
	"encoding/json"
	"html/template"
	"os"
	"path/filepath"

	"env-check/pkg/config"
)

// divideFloat divides two float64 numbers
func divideFloat(a, b uint64) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

// getPortDetails extracts port details from CheckResult
func getPortDetails(result config.CheckResult) []config.PortCheckInfo {
	if len(result.DetailData) == 0 {
		return nil
	}
	var portResult config.PortCheckResult
	if err := json.Unmarshal(result.DetailData, &portResult); err != nil {
		return nil
	}
	return portResult.Ports
}

// getDiskDetails extracts disk details from CheckResult
func getDiskDetails(result config.CheckResult) []config.DiskSpace {
	if len(result.DetailData) == 0 {
		return nil
	}
	var diskResult config.DiskCheckResult
	if err := json.Unmarshal(result.DetailData, &diskResult); err != nil {
		return nil
	}
	return diskResult.Spaces
}

// getDiskSummary extracts disk summary from CheckResult
func getDiskSummary(result config.CheckResult) config.DiskSummary {
	if len(result.DetailData) == 0 {
		return config.DiskSummary{}
	}
	var diskResult config.DiskCheckResult
	if err := json.Unmarshal(result.DetailData, &diskResult); err != nil {
		return config.DiskSummary{}
	}
	return diskResult.Summary
}

// getFileQueryDetails extracts file query details from CheckResult
func getFileQueryDetails(result config.CheckResult) []config.FileInfo {
	if len(result.DetailData) == 0 {
		return nil
	}
	var queryResult config.QueryResult
	if err := json.Unmarshal(result.DetailData, &queryResult); err != nil {
		return nil
	}
	return queryResult.Files
}

// getProgramCheckDetails extracts program check details from CheckResult
func getProgramCheckDetails(result config.CheckResult) []config.ProgramInfo {
	if len(result.DetailData) == 0 {
		return nil
	}
	var programResult config.ProgramCheckResult
	if err := json.Unmarshal(result.DetailData, &programResult); err != nil {
		return nil
	}
	return programResult.Programs
}

// getRouteDetails extracts route details from CheckResult
func getRouteDetails(result config.CheckResult) []config.RouteInfo {
	if len(result.DetailData) == 0 {
		return nil
	}
	var routeResult config.RouteCheckResult
	if err := json.Unmarshal(result.DetailData, &routeResult); err != nil {
		return nil
	}
	return routeResult.DefaultRoutes
}

// getIptablesPolicy extracts iptables FORWARD policy from CheckResult
func getIptablesPolicy(result config.CheckResult) config.ForwardPolicy {
	if len(result.DetailData) == 0 {
		return config.ForwardPolicy{}
	}
	var iptablesResult config.IptablesCheckResult
	if err := json.Unmarshal(result.DetailData, &iptablesResult); err != nil {
		return config.ForwardPolicy{}
	}
	return iptablesResult.ForwardPolicy
}

// templateFuncs contains custom template functions
var templateFuncs = template.FuncMap{
	"divideFloat":            divideFloat,
	"getPortDetails":         getPortDetails,
	"getDiskDetails":         getDiskDetails,
	"getDiskSummary":         getDiskSummary,
	"getFileQueryDetails":    getFileQueryDetails,
	"getProgramCheckDetails": getProgramCheckDetails,
	"getRouteDetails":        getRouteDetails,
	"getIptablesPolicy":      getIptablesPolicy,
}

//go:embed query.tpl
var queryTemplateFS embed.FS

//go:embed clean.tpl
var cleanTemplateFS embed.FS

//go:embed check.tpl
var checkTemplateFS embed.FS

//go:embed clock.tpl
var clockTemplateFS embed.FS

//go:embed route.tpl
var routeTemplateFS embed.FS

//go:embed iptables.tpl
var iptablesTemplateFS embed.FS

// generateHTML is a small helper to parse a template from an embed.FS,
// create the output file and execute the template with provided data.
func generateHTML(tfs embed.FS, tplName string, data interface{}, outputPath string) error {
	tmpl, err := template.New(tplName).Funcs(templateFuncs).ParseFS(tfs, tplName)
	if err != nil {
		return err
	}

	// Ensure parent directory exists and create file with explicit permissions
	dir := filepath.Dir(outputPath)
	if dir != "." && dir != "" {
		if err := os.MkdirAll(dir, config.DirectoryMode); err != nil {
			return err
		}
	}

	f, err := os.OpenFile(outputPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, config.FileMode)
	if err != nil {
		return err
	}
	defer f.Close()

	return tmpl.Execute(f, data)
}

// GenerateQueryHTML 生成QueryHTML文件
func GenerateQueryHTML(result config.QueryResult, outputPath string) error {
	return generateHTML(queryTemplateFS, "query.tpl", result, outputPath)
}

// GenerateCleanHTML 生成CleanHTML文件
func GenerateCleanHTML(result config.CleanResult, outputPath string) error {
	return generateHTML(cleanTemplateFS, "clean.tpl", result, outputPath)
}

// GenerateCheckHTML 生成CheckHTML文件
func GenerateCheckHTML(result config.ProgramCheckResult, outputPath string) error {
	return generateHTML(checkTemplateFS, "check.tpl", result, outputPath)
}

// GenerateClockHTML 生成ClockHTML文件
func GenerateClockHTML(result config.ClockCheckResult, outputPath string) error {
	return generateHTML(clockTemplateFS, "clock.tpl", result, outputPath)
}

// GenerateRouteHTML 生成RouteHTML文件
func GenerateRouteHTML(result config.RouteCheckResult, outputPath string) error {
	return generateHTML(routeTemplateFS, "route.tpl", result, outputPath)
}

// GenerateIptablesHTML 生成IptablesHTML文件
func GenerateIptablesHTML(result config.IptablesCheckResult, outputPath string) error {
	return generateHTML(iptablesTemplateFS, "iptables.tpl", result, outputPath)
}

//go:embed kernel.tpl
var kernelTemplateFS embed.FS

//go:embed port.tpl
var portTemplateFS embed.FS

//go:embed disk.tpl
var diskTemplateFS embed.FS

// GenerateKernelHTML 生成KernelHTML文件
func GenerateKernelHTML(result config.KernelCheckResult, outputPath string) error {
	return generateHTML(kernelTemplateFS, "kernel.tpl", result, outputPath)
}

// GeneratePortHTML 生成PortHTML文件
func GeneratePortHTML(result config.PortCheckResult, outputPath string) error {
	return generateHTML(portTemplateFS, "port.tpl", result, outputPath)
}

// GenerateDiskHTML 生成DiskHTML文件
func GenerateDiskHTML(result config.DiskCheckResult, outputPath string) error {
	return generateHTML(diskTemplateFS, "disk.tpl", result, outputPath)
}

//go:embed dispatch.tpl
var dispatchTemplateFS embed.FS

// GenerateDispatchHTML 生成DispatchHTML文件
func GenerateDispatchHTML(result config.DispatchResult, outputPath string) error {
	return generateHTML(dispatchTemplateFS, "dispatch.tpl", result, outputPath)
}
