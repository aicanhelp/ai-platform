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

package route

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

const (
	decimalBase       = 10 // base for manual decimal parsing of the metric value
	minRouteNParts    = 4  // minimum field count of a `route -n` line
	routeNMetricParts = 5  // field count required to read the metric column
	routeNIfaceParts  = 8  // field count required to read the interface column
)

type RouteChecker struct {
	config *config.AppConfig
	logger logger.Logger
}

func NewRouteChecker(cfg *config.AppConfig, log logger.Logger) *RouteChecker {
	return &RouteChecker{
		config: cfg,
		logger: log,
	}
}

func (r *RouteChecker) Execute() (*config.RouteCheckResult, error) {
	r.logger.Info("route check start......")

	result := &config.RouteCheckResult{
		Timestamp:     time.Now(),
		DefaultRoutes: make([]config.RouteInfo, 0),
		Summary:       config.RouteSummary{},
	}

	routes, err := r.getDefaultRoutes()
	if err != nil {
		result.Summary.CheckPassed = false
		result.Summary.Message = err.Error()
		r.logger.Error("failed to get default routes: " + err.Error())
		return result, nil
	}

	for _, route := range routes {
		result.DefaultRoutes = append(result.DefaultRoutes, route)
	}

	result.Summary.TotalRoutes = len(result.DefaultRoutes)
	if result.Summary.TotalRoutes > 0 {
		result.Summary.CheckPassed = true
		result.Summary.Message = "default route exists"
		r.logger.Info("found " + fmt.Sprintf("%d", result.Summary.TotalRoutes) + " default routes")
	} else {
		result.Summary.CheckPassed = false
		result.Summary.Message = "no default route found"
		r.logger.Warning(result.Summary.Message)
	}

	r.logger.Info("route check completed")
	return result, nil
}

func (r *RouteChecker) getDefaultRoutes() ([]config.RouteInfo, error) {
	var routes []config.RouteInfo

	output, err := r.executeRouteCommand()
	if err != nil {
		return nil, err
	}

	routes = r.parseRouteOutput(output)
	return routes, nil
}

func (r *RouteChecker) executeRouteCommand() (string, error) {
	if _, err := exec.LookPath("ip"); err == nil {
		cmd := exec.Command("ip", "route", "show", "default")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if err != nil {
			return "", fmt.Errorf("failed to execute ip route: %v, stderr: %s", err, stderr.String())
		}
		return stdout.String(), nil
	}

	if _, err := exec.LookPath("route"); err == nil {
		cmd := exec.Command("route", "-n")
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		err := cmd.Run()
		if err != nil {
			return "", fmt.Errorf("failed to execute route: %v, stderr: %s", err, stderr.String())
		}
		return stdout.String(), nil
	}

	return "", fmt.Errorf("no route command available (ip or route)")
}

func (r *RouteChecker) parseRouteOutput(output string) []config.RouteInfo {
	var routes []config.RouteInfo

	lines := strings.Split(output, "\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "default") {
			route := r.parseIPRouteLine(line)
			if route.Exists && route.Gateway != "" {
				routes = append(routes, route)
			}
		}

		if strings.HasPrefix(line, "0.0.0.0") && strings.Contains(line, "UG") {
			route := r.parseRouteNLine(line)
			if route.Exists && route.Gateway != "" {
				routes = append(routes, route)
			}
		}
	}

	return routes
}

// parseMetric extracts the leading decimal integer from s (e.g. "100" -> 100).
func parseMetric(s string) int {
	re := regexp.MustCompile(`\d+`)
	match := re.FindString(s)
	if match == "" {
		return 0
	}
	metric := 0
	for _, c := range match {
		metric = metric*decimalBase + int(c-'0')
	}
	return metric
}

func (r *RouteChecker) parseIPRouteLine(line string) config.RouteInfo {
	route := config.RouteInfo{
		Exists:      true,
		Destination: "default",
	}

	parts := strings.Fields(line)
	for i, part := range parts {
		if part == "via" && i+1 < len(parts) {
			route.Gateway = parts[i+1]
		}
		if part == "dev" && i+1 < len(parts) {
			route.Interface = parts[i+1]
		}
		if part == "metric" && i+1 < len(parts) {
			route.Metric = parseMetric(parts[i+1])
		}
	}

	return route
}

func (r *RouteChecker) parseRouteNLine(line string) config.RouteInfo {
	route := config.RouteInfo{
		Exists:      true,
		Destination: "default",
	}

	parts := strings.Fields(line)
	if len(parts) < minRouteNParts {
		return route
	}
	if parts[0] != "0.0.0.0" && parts[0] != "default" {
		return route
	}

	if parts[1] != "0.0.0.0" && parts[1] != "*" {
		route.Gateway = parts[1]
	}
	if len(parts) >= routeNIfaceParts {
		route.Interface = parts[7]
	}
	if len(parts) >= routeNMetricParts {
		route.Metric = parseMetric(parts[4])
	}

	return route
}
