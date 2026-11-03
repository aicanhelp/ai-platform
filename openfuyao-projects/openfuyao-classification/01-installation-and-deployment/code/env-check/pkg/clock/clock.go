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

// Package clock handles clock operations, currently only includes clock synchronized
package clock

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"
	"time"

	"golang.org/x/crypto/ssh"

	"env-check/pkg/config"
	"env-check/pkg/logger"
	"env-check/pkg/sshutil"
)

var timeoutSec = 5 // ssh timeout

// CheckerClock clock checker
type CheckerClock struct {
	config *config.AppConfig
	logger logger.Logger
}

// HostResult clock info of each host
type HostResult struct {
	host string
	role []string
	time int64
	err  error
}

// NewClockChecker creates a new clock checker
func NewClockChecker(cfg *config.AppConfig, log logger.Logger) *CheckerClock {
	return &CheckerClock{
		config: cfg,
		logger: log,
	}
}

// Execute executes program check
func (c *CheckerClock) Execute() (*config.ClockCheckResult, error) {
	c.logger.Info("clock check start......")
	result := &config.ClockCheckResult{
		Timestamp: time.Now(),
		Clocks:    make([]config.ClockCheckInfo, 0),
		Result:    "pass",
	}

	bootstrapHosts, otherHosts, err := c.getHostSets()
	if err != nil {
		return nil, err
	}

	// Get bootstrap reference time (use first bootstrap host)
	refHost := bootstrapHosts[0]
	c.logger.Info("bootstrap hosts: " + refHost.IP)
	refTime, err := c.getHostTime(refHost)
	if err != nil {
		return nil, fmt.Errorf("failed to get reference time from bootstrap host %s: %v", refHost.IP, err)
	}

	refClocks := config.ClockCheckInfo{
		Host:          refHost.IP,
		Role:          refHost.Role,
		IsSynced:      true,
		TimeDiff:      0,
		RemoteTime:    refTime,
		ReferenceTime: refTime,
		IsLocal:       isLocalHost(refHost.IP),
	}
	result.Clocks = append(result.Clocks, refClocks)

	var wg sync.WaitGroup
	allResults := make([]HostResult, len(otherHosts))

	for i, otherHost := range otherHosts {
		wg.Add(1)
		go func(idx int, h config.Host) {
			defer wg.Done()
			nodeTime, err1 := c.getHostTime(h)
			allResults[idx] = HostResult{host: h.IP, role: h.Role, time: nodeTime, err: err1}
		}(i, otherHost)
	}

	wg.Wait()

	c.updateCheckResult(allResults, refTime, result)

	c.logger.Info("clock check completed")
	return result, nil
}

func (c *CheckerClock) updateCheckResult(resultChan []HostResult, refTime int64, result *config.ClockCheckResult) {
	for _, clockRes := range resultChan {
		c.logger.Info("other hosts: " + clockRes.host)
		checkResult := config.ClockCheckInfo{
			Host:          clockRes.host,
			Role:          clockRes.role,
			RemoteTime:    clockRes.time,
			ReferenceTime: refTime,
			IsLocal:       isLocalHost(clockRes.host),
		}
		if clockRes.err != nil {
			checkResult.IsSynced = false
			checkResult.TimeDiff = -1
			checkResult.Error = clockRes.err.Error()
			result.Result = "fail"
		} else {
			timeDiff := abs(clockRes.time - refTime)
			checkResult.TimeDiff = timeDiff
			checkResult.IsSynced = timeDiff <= c.config.ClockThreshold
			if !checkResult.IsSynced {
				result.Result = "fail"
			}
		}
		result.Clocks = append(result.Clocks, checkResult)
	}
}

func (c *CheckerClock) getHostSets() ([]config.Host, []config.Host, error) {
	// Separate bootstrap and non-bootstrap hosts
	var bootstrapHosts []config.Host
	var otherHosts []config.Host

	for _, host := range c.config.Hosts {
		if c.isBootstrap(host) {
			bootstrapHosts = append(bootstrapHosts, host)
		} else {
			otherHosts = append(otherHosts, host)
		}
	}

	if len(bootstrapHosts) == 0 {
		return nil, nil, fmt.Errorf("no bootstrap host found in config")
	}

	if len(otherHosts) == 0 {
		return nil, nil, fmt.Errorf("no non-bootstrap host found in config")
	}
	return bootstrapHosts, otherHosts, nil
}
func (c *CheckerClock) isBootstrap(host config.Host) bool {
	for _, role := range host.Role {
		if role == "bootstrap" {
			return true
		}
	}
	return false
}

func (c *CheckerClock) getHostTime(host config.Host) (int64, error) {
	// Check if host is local machine
	if isLocalHost(host.IP) {
		// Get local time
		return c.getLocalTime()
	}

	// For remote host, use SSH
	return c.getRemoteTime(host)
}

func (c *CheckerClock) getLocalTime() (int64, error) {
	// Get current Unix timestamp
	return time.Now().Unix(), nil
}

func (c *CheckerClock) getRemoteTime(host config.Host) (int64, error) {
	sshConfig := sshutil.BuildClientConfig(host, time.Duration(timeoutSec)*time.Second)
	address := sshutil.Address(host)
	client, err := ssh.Dial("tcp", address, sshConfig)
	if err != nil {
		return 0, fmt.Errorf("failed to connect to %s: %v", address, err)
	}

	// Create a session
	session, err := client.NewSession()
	if err != nil {
		client.Close()
		return 0, fmt.Errorf("failed to create SSH session: %v", err)
	}
	defer func() {
		session.Close()
		client.Close()
	}()

	// Execute the command to get time
	output, err := session.Output("date +%s")
	if err != nil {
		return 0, fmt.Errorf("failed to execute command: %v", err)
	}

	// Parse the output
	timeStr := strings.TrimSpace(string(output))
	timeInt, err := strconv.ParseInt(timeStr, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse time output '%s': %v", timeStr, err)
	}

	return timeInt, nil
}

func isLocalHost(ip string) bool {
	// Check if IP is localhost
	if ip == "127.0.0.1" || ip == "localhost" {
		return true
	}

	// Get local IP addresses
	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return false
	}

	for _, addr := range addrs {
		var checkIP net.IP
		switch v := addr.(type) {
		case *net.IPNet:
			checkIP = v.IP
		case *net.IPAddr:
			checkIP = v.IP
		default:
			continue
		}

		if checkIP != nil && checkIP.String() == ip {
			return true
		}
	}

	return false
}

func abs(x int64) int64 {
	if x < 0 {
		return -x
	}
	return x
}
