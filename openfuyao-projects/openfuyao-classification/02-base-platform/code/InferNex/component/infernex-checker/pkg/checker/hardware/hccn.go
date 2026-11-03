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

package hardware

import (
	"fmt"
	"net"
	"regexp"
	"strconv"
	"strings"
	"time"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// getPhyIDs returns the list of NPU physical IDs from /dev/davinci* devices,
// excluding /dev/davinci_manager and similar non-numeric entries.
func getPhyIDs(ssh executor.CommandRunner) ([]int, error) {
	stdout, _, err := ssh.Run("ls /dev/davinci*")
	if err != nil {
		return nil, fmt.Errorf("failed to list /dev/davinci*: %w", err)
	}
	re := regexp.MustCompile(`/dev/davinci(\d+)$`)
	var ids []int
	for _, field := range strings.Fields(stdout) {
		m := re.FindStringSubmatch(field)
		if m != nil {
			id, err := strconv.Atoi(m[1])
			if err != nil {
				continue
			}
			ids = append(ids, id)
		}
	}
	return ids, nil
}

// checkH06 checks hccn.conf configuration correctness; returns check result and parsed IP map
func checkH06(ssh executor.CommandRunner, phyIDs []int) (types.CheckResult, map[string]string) {
	stdout, _, err := ssh.Run("cat /etc/hccn.conf")
	if err != nil {
		return types.CheckResult{
			ID:         "H-06",
			Status:     types.StatusFailed,
			Message:    "Failed to read /etc/hccn.conf",
			Suggestion: "Please confirm the file exists and is readable",
		}, nil
	}

	ipMap, maskMap := parseHccnConf(stdout)

	// step 1: IP address completeness — each NPU ID must have a corresponding address entry
	if r := checkH06MissingIPs(phyIDs, ipMap); r.Status == types.StatusFailed {
		return r, nil
	}

	// step 2: netmask completeness — each NPU ID must have a corresponding netmask entry
	if r := checkH06MissingMasks(phyIDs, maskMap); r.Status == types.StatusFailed {
		return r, nil
	}

	// step 3: IP format and uniqueness
	if r := checkH06IPFormatAndUniqueness(ipMap); r.Status == types.StatusFailed {
		return r, nil
	}

	return types.CheckResult{
		ID:      "H-06",
		Status:  types.StatusPassed,
		Message: "hccn.conf configuration is correct",
	}, ipMap
}

// parseHccnConf parses hccn.conf and returns a NIC id -> IP map and a NIC id -> netmask map
func parseHccnConf(content string) (ipMap, maskMap map[string]string) {
	ipMap = map[string]string{}
	maskMap = map[string]string{}
	for _, line := range strings.Split(content, "\n") {
		line = strings.TrimSpace(line)
		parts := strings.SplitN(line, "=", hccnConfSplitN)
		if len(parts) != hccnConfSplitN {
			continue
		}
		val := strings.TrimSpace(parts[1])
		if strings.HasPrefix(line, "address_") {
			id := strings.TrimPrefix(parts[0], "address_")
			ipMap[id] = val
		} else if strings.HasPrefix(line, "netmask_") {
			id := strings.TrimPrefix(parts[0], "netmask_")
			maskMap[id] = val
		} else {
			// Ignore other lines
		}
	}
	return
}

func checkH06MissingIPs(phyIDs []int, ipMap map[string]string) types.CheckResult {
	var missingIPs []string
	for _, i := range phyIDs {
		id := strconv.Itoa(i)
		if ip, ok := ipMap[id]; !ok || ip == "" {
			missingIPs = append(missingIPs, fmt.Sprintf("address_%d", i))
		}
	}
	if len(missingIPs) > 0 {
		return types.CheckResult{
			ID:     "H-06",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("Missing IP configuration for: %s",
				strings.Join(missingIPs, ", ")),
			Suggestion: "Add missing IP configuration for the listed NICs",
		}
	}
	return types.CheckResult{}
}

func checkH06MissingMasks(phyIDs []int, maskMap map[string]string) types.CheckResult {
	var missingMasks []string
	for _, i := range phyIDs {
		id := strconv.Itoa(i)
		if mask, ok := maskMap[id]; !ok || mask == "" {
			missingMasks = append(missingMasks, fmt.Sprintf("netmask_%d", i))
		}
	}
	if len(missingMasks) > 0 {
		return types.CheckResult{
			ID:     "H-06",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("Missing netmask configuration for: %s",
				strings.Join(missingMasks, ", ")),
			Suggestion: "Add missing netmask configuration for the listed NICs",
		}
	}
	return types.CheckResult{}
}

func checkH06IPFormatAndUniqueness(ipMap map[string]string) types.CheckResult {
	seen := map[string]string{}
	for id, ip := range ipMap {
		if net.ParseIP(ip) == nil {
			return types.CheckResult{
				ID:         "H-06",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("NIC address_%s has an invalid IP format: %s", id, ip),
				Suggestion: "Fix the malformed IP address",
			}
		}
		if prev, dup := seen[ip]; dup {
			return types.CheckResult{
				ID:         "H-06",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("address_%s and address_%s have duplicate IP: %s", id, prev, ip),
				Suggestion: "Resolve NIC IP conflict and ensure each NIC has a unique IP",
			}
		}
		seen[ip] = id
	}
	return types.CheckResult{}
}

// checkH07 checks single-node HCCS communication; returns check results and TLS state map
func checkH07(ssh executor.CommandRunner,
	phyIDs []int,
	ipMap map[string]string, enableConnectivityCheck bool) ([]types.CheckResult, map[string]string) {
	// step 1: TLS switch state consistency
	r1, tlsStates := checkH07TLSConsistency(ssh, phyIDs)
	if r1.Status == types.StatusFailed {
		return []types.CheckResult{r1}, tlsStates
	}

	// step 2: NIC-to-NIC connectivity (conditional based on enableConnectivityCheck)
	if !enableConnectivityCheck {
		r2 := types.CheckResult{
			ID:      "H-07",
			Status:  types.StatusSkipped,
			Message: "Step 2: Connectivity check skipped (use --enable-connectivity-check to enable)",
		}
		return []types.CheckResult{r1, r2}, tlsStates
	}

	r2 := checkH07Connectivity(ssh, phyIDs, ipMap)
	return []types.CheckResult{r1, r2}, tlsStates
}

func checkH07TLSConsistency(ssh executor.CommandRunner, phyIDs []int) (types.CheckResult, map[string]string) {
	tlsStates := map[string]string{}

	firstTLS := ""
	var inconsistent []string
	for _, i := range phyIDs {
		out, stderr, err := ssh.Run(fmt.Sprintf("hccn_tool -i %d -tls -g", i))
		if err != nil && !strings.Contains(strings.ToLower(stderr), "no certificate found") {
			return types.CheckResult{
				ID:         "H-07",
				Status:     types.StatusFailed,
				Message:    fmt.Sprintf("Step 1: Failed to get TLS state for NPU%d: %v", i, err),
				Suggestion: "Please check hccn_tool availability and NPU status",
			}, tlsStates
		}
		state := parseTLSState(out + stderr)
		tlsStates[strconv.Itoa(i)] = state
		if firstTLS == "" {
			firstTLS = state
		} else if normalizeTLS(state) != normalizeTLS(firstTLS) {
			inconsistent = append(inconsistent, fmt.Sprintf("NPU%d=%s", i, state))
		} else {
			// TLS state is consistent, no action needed
		}
	}

	if len(inconsistent) > 0 {
		return types.CheckResult{
			ID:     "H-07",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("NIC TLS switch states are inconsistent: %s",
				strings.Join(inconsistent, ", ")),
			Suggestion: "Unify TLS switch configuration across all NICs",
		}, tlsStates
	}

	return types.CheckResult{
		ID:      "H-07",
		Status:  types.StatusPassed,
		Message: fmt.Sprintf("Step 1: NIC TLS switch states are consistent (%s)", firstTLS),
	}, tlsStates
}

func parseTLSState(output string) string {
	lower := strings.ToLower(output)
	if strings.Contains(lower, "no certificate found") {
		return "no_cert"
	}
	re := regexp.MustCompile(`tls switch\[(\d+)]`)
	m := re.FindStringSubmatch(output)
	if len(m) >= regexMinMatchLen {
		if m[1] == "1" {
			return "enable"
		}
		return "disable"
	}
	return "unknown"
}

// hccnPing executes a single hccn_tool ping and returns true if successful.
func hccnPing(ssh executor.CommandRunner, phyID int, targetIP string) bool {
	stdout, _, err := ssh.Run(fmt.Sprintf("hccn_tool -i %d -ping -g address %s pkt %d",
		phyID, targetIP, hccnPingPacketCount))
	return err == nil &&
		!strings.Contains(stdout, "Command execute failed") &&
		!strings.Contains(stdout, "100.00% packet loss")
}

func pingWithRetry(ssh executor.CommandRunner, phyID int, targetIP string) bool {
	for attempt := 0; attempt < hccnPingMaxRetries; attempt++ {
		if hccnPing(ssh, phyID, targetIP) {
			return true
		}
		if attempt < hccnPingMaxRetries-1 {
			time.Sleep(hccnPingRetryInterval)
		}
	}
	return false
}

func checkH07Connectivity(ssh executor.CommandRunner, phyIDs []int, ipMap map[string]string) types.CheckResult {
	var pingFailed []string
	for _, i := range phyIDs {
		for nic, ip := range ipMap {
			if ip == "" || nic == strconv.Itoa(i) {
				continue
			}
			if !pingWithRetry(ssh, i, ip) {
				pingFailed = append(pingFailed, fmt.Sprintf("NPU%d->%s(%s)", i, nic, ip))
			}
		}
	}

	if len(pingFailed) > 0 {
		return types.CheckResult{
			ID:     "H-07",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("Step 2: The following NIC pairs failed ping: %s",
				strings.Join(pingFailed, ", ")),
			Suggestion: "Please check: whether port link is UP, whether optical modules are present, " +
				"whether all NIC IPs are on the same subnet",
		}
	}

	return types.CheckResult{
		ID:      "H-07",
		Status:  types.StatusPassed,
		Message: "Step 2: Single-node NIC-to-NIC connectivity is normal",
	}
}

func normalizeTLS(state string) string {
	if state == "no_cert" {
		return "disable"
	}
	return state
}
