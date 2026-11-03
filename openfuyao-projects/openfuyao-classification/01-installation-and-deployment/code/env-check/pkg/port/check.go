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

// Package port performs port check
package port

import (
	"fmt"
	"net"
	"strconv"
	"time"

	gopsnet "github.com/shirou/gopsutil/v3/net"
	gopsproc "github.com/shirou/gopsutil/v3/process"

	"env-check/pkg/config"
	"env-check/pkg/logger"
)

const (
	numZero           = 0
	numOne            = 1
	numThree          = 3
	numFive           = 5
	numOneHundred     = 100
	numTwentyFour     = 24
	defaultCheckLevel = numFive
	defaultDialSec    = numThree

	ipv4LoopbackSegmentA = numOneHundred + numTwentyFour + numThree // 127
	ipv4LoopbackSegmentB = numZero
	ipv4LoopbackSegmentC = numZero
	ipv4LoopbackSegmentD = numOne
)

var loopbackIPv4 = net.IPv4(ipv4LoopbackSegmentA, ipv4LoopbackSegmentB, ipv4LoopbackSegmentC, ipv4LoopbackSegmentD)

// CheckItem defines a port check item with role-based assignment
type CheckItem struct {
	name    string
	version []string
	level   int
	ports   []string
}

// NewCheckItem creates a new port check item
func NewCheckItem(name string, version []string, level int, ports []string) *CheckItem {
	return &CheckItem{
		name:    name,
		version: version,
		level:   level,
		ports:   ports,
	}
}

// Name returns the check item name
func (c *CheckItem) Name() string {
	return c.name
}

// Targets returns the ports to check
func (c *CheckItem) Targets() []string {
	return c.ports
}

// Checker port checker
type Checker struct {
	cfg    *config.PortCheckConfig
	logger logger.Logger
	items  []*CheckItem
}

// NewChecker creates a new port checker
func NewChecker(cfg *config.PortCheckConfig, log logger.Logger) *Checker {
	return &Checker{
		cfg:    cfg,
		logger: log,
		items:  buildCheckItems(cfg),
	}
}

// buildCheckItems builds check items from config
func buildCheckItems(cfg *config.PortCheckConfig) []*CheckItem {
	var items []*CheckItem
	if ports, ok := cfg.Ports["bootstrap"]; ok && len(ports) > 0 {
		items = append(items, NewCheckItem("引导节点端口检查", []string{"v4.0", "all"}, defaultCheckLevel, ports))
	}
	if ports, ok := cfg.Ports["master"]; ok && len(ports) > 0 {
		items = append(items, NewCheckItem("Master节点端口检查", []string{"v4.0"}, defaultCheckLevel, ports))
	}
	if ports, ok := cfg.Ports["worker"]; ok && len(ports) > 0 {
		items = append(items, NewCheckItem("Worker节点端口检查", []string{"v4.0"}, defaultCheckLevel, ports))
	}
	return items
}

// Execute executes port check for given roles
func (c *Checker) Execute(roles []string) *config.PortCheckResult {
	c.logger.Info("port check start......")
	c.logger.Info(fmt.Sprintf("node roles: %v", roles))

	result := &config.PortCheckResult{
		Timestamp: time.Now(),
		Ports:     make([]config.PortCheckInfo, 0),
		Summary:   config.PortSummary{},
	}

	for _, item := range c.items {
		if !c.assign(item, roles) {
			continue
		}
		c.logger.Info(fmt.Sprintf("running check: %s, ports: %v", item.Name(), item.Targets()))
		c.runCheckItem(item, result)
	}

	c.logger.Info("port check completed")
	return result
}

// assign determines if the check item should run for given roles
func (c *Checker) assign(item *CheckItem, roles []string) bool {
	if len(roles) == 0 {
		return true
	}

	for _, role := range roles {
		switch role {
		case "bootstrap":
			if item.Name() == "引导节点端口检查" {
				return true
			}
		case "master":
			if item.Name() == "Master节点端口检查" {
				return true
			}
		case "worker":
			if item.Name() == "Worker节点端口检查" {
				return true
			}
		default:
			continue
		}
	}

	return false
}

// runCheckItem executes a single check item
func (c *Checker) runCheckItem(item *CheckItem, result *config.PortCheckResult) {
	for _, port := range item.Targets() {
		portInfo := c.checkPort(port)
		result.Ports = append(result.Ports, portInfo)
		result.Summary.TotalChecked++

		if portInfo.IsUsed {
			result.Summary.Used++
			c.logger.Warning(fmt.Sprintf("port %s is used", port))
		} else {
			result.Summary.Free++
			c.logger.Info(fmt.Sprintf("port %s is free", port))
		}
	}
}

// checkPort checks if a port is in use
func (c *Checker) checkPort(port string) config.PortCheckInfo {
	info := config.PortCheckInfo{
		Port:     port,
		Protocol: "tcp",
	}

	timeout := c.cfg.Timeout
	if timeout == 0 {
		timeout = defaultDialSec
	}

	addr := net.JoinHostPort(loopbackIPv4.String(), port)
	conn, err := net.DialTimeout("tcp", addr, time.Duration(timeout)*time.Second)
	if err == nil {
		if closeErr := conn.Close(); closeErr != nil {
			c.logger.Warning(fmt.Sprintf("failed to close port check connection for %s: %v", port, closeErr))
		}
		info.IsUsed = true
		info.Process = c.findOccupyingProcess(port)
	}

	return info
}

// findOccupyingProcess returns the "name(pid)" of the process listening on the
// given port, or an empty string when it cannot be determined (insufficient
// permissions, no matching listener, or unavailable proc info). The port
// occupation verdict itself (IsUsed) is unaffected by this lookup failing.
func (c *Checker) findOccupyingProcess(port string) string {
	portNum, err := strconv.Atoi(port)
	if err != nil {
		return ""
	}

	conns, err := gopsnet.Connections("tcp")
	if err != nil {
		c.logger.Warning(fmt.Sprintf("failed to list tcp connections for port %s: %v", port, err))
		return ""
	}

	for _, conn := range conns {
		if conn.Status != "LISTEN" || conn.Laddr.Port != uint32(portNum) || conn.Pid <= 0 {
			continue
		}
		name := "unknown"
		if proc, err := gopsproc.NewProcess(conn.Pid); err == nil {
			if procName, err := proc.Name(); err == nil && procName != "" {
				name = procName
			}
		}
		return fmt.Sprintf("%s(%d)", name, conn.Pid)
	}
	return ""
}
