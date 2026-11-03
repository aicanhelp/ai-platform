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

// Package hardware implements the Hardware Layer checks (H-01 through H-10),
// covering NPU driver and firmware status, Ascend Device Plugin readiness, NPU
// availability, host file integrity, hccn.conf correctness, intra-node HCCS
// communication, single-node slow card detection, cross-node slow card detection,
// and cross-node RoCE RDMA connectivity.
package hardware

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
	"sync"
	"time"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

const (
	regexMinMatchLen      = 2               // minimum length of a regex submatch result (full match + 1 group)
	hccnPingPacketCount   = 128             // number of packets used in hccn_tool ping checks
	hccnPingMaxRetries    = 3               // max retry attempts for hccn_tool ping to tolerate transient failures
	hccnPingRetryInterval = 1 * time.Second // wait between ping retries to tolerate transient cold-start failures
	npuStatusGroupIdx     = 2               // regex capture group index for NPU health status field
	npuIDGroupIdx         = 1               // regex capture group index for NPU ID field
	hccnConfSplitN        = 2               // max parts when splitting hccn.conf key=value lines
	crossNodeMinCount     = 2               // minimum number of 910-series nodes required for cross-node check
)

// newSSHRunner is the default factory for creating a CommandRunner from a NodeInfo.
// Tests override this variable to inject a fake runner without a real SSH connection.
var newSSHRunner = executor.NewSSHRunner

// ansiEscape matches ANSI terminal escape sequences for stripping colored output.
var ansiEscape = regexp.MustCompile(`\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])`)

// npuModelRe matches NPU model rows in npu-smi output: | <id>  <model> | OK/Warning/... |
var npuModelRe = regexp.MustCompile(`\|\s+\d+\s+(\S+)\s+\|\s+(OK|Warning|Alarm|Critical|UNKNOWN)`)

// npuDeviceRe matches NPU device rows in npu-smi output for counting: | <id>  <model> | <status> |
var npuDeviceRe = regexp.MustCompile(`\|\s+(\d+)\s+\S+\s+\|\s+(OK|Warning|Alarm|Critical|UNKNOWN)`)

// npuProcessRe matches process rows in npu-smi output: | <npu> <chip> | <pid> | <name> | <mem> |
var npuProcessRe = regexp.MustCompile(`\|\s+(\d+)\s+\d+\s+\|\s+(\d+)\s+\|\s+\S+\s+\|\s+\d+\s+\|`)

// node910CheckInput holds configuration inputs for 910-series node checks
type node910CheckInput struct {
	NodeName                string
	NPUTotal                int
	EnableConnectivityCheck bool
}

// NodeCheckContext holds per-node check context, storing data collected during checks for reuse
type NodeCheckContext struct {
	Node        types.NodeInfo
	Is910Series bool
	PhyIDs      []int             // physical NPU IDs from /dev/davinci*
	HccnIPs     map[string]string // NIC name -> IP
	TLSStates   map[string]string // NIC name -> TLS state
}

type podListOutput struct {
	Items []struct {
		Metadata struct {
			Name string `json:"name"`
		} `json:"metadata"`
		Spec struct {
			NodeName string `json:"nodeName"`
		} `json:"spec"`
		Status struct {
			Phase string `json:"phase"`
		} `json:"status"`
	} `json:"items"`
}

func parsePodList(stdout string) (podListOutput, error) {
	var pods podListOutput
	return pods, json.Unmarshal([]byte(stdout), &pods)
}

// CheckHardware runs full Hardware Layer checks and returns results for all nodes
func CheckHardware(nodes []types.NodeInfo, enableConnectivityCheck bool) *types.HardwareResult {
	result := &types.HardwareResult{}
	var mu sync.Mutex
	var wg sync.WaitGroup

	ctxMap := make(map[string]*NodeCheckContext)
	var ctxMu sync.Mutex

	for _, node := range nodes {
		wg.Add(1)
		go func(n types.NodeInfo) {
			defer wg.Done()
			nodeResult, ctx := checkSingleNode(n, enableConnectivityCheck)

			mu.Lock()
			result.Nodes = append(result.Nodes, nodeResult)
			mu.Unlock()

			if ctx != nil {
				ctxMu.Lock()
				ctxMap[n.Name] = ctx
				ctxMu.Unlock()
			}
		}(node)
	}
	wg.Wait()

	// Collect 910-series nodes that passed single-node checks
	// no lock needed here: all goroutines have finished, ctxMap is no longer written to
	var passedNodes []types.NodeResult
	var passedCtxs []*NodeCheckContext
	for _, nodeResult := range result.Nodes {
		if nodeResult.Status == types.StatusPassed && nodeResult.Is910Series {
			passedNodes = append(passedNodes, nodeResult)
			if ctx, ok := ctxMap[nodeResult.Name]; ok {
				passedCtxs = append(passedCtxs, ctx)
			}
		}
	}

	// H-09: cross-node slow card detection (reuse H-08 results)
	h09Result := checkH09(passedNodes)
	result.CrossNodes = append(result.CrossNodes, h09Result)

	// H-10: cross-node HCCN/RoCE check
	result.CrossNodes = append(result.CrossNodes, checkH10(passedCtxs, enableConnectivityCheck)...)

	return result
}

// checkSingleNode runs H-01~H-05 on all nodes, H-06~H-10 on 910-series only
func checkSingleNode(node types.NodeInfo, enableConnectivityCheck bool) (types.NodeResult, *NodeCheckContext) {
	result := types.NodeResult{
		Name:   node.Name,
		Status: types.StatusPassed,
	}

	runner, cleanup, err := newSSHRunner(node)
	if err != nil {
		log.Error(fmt.Sprintf("node %s SSH connection failed: %v", node.Name, err))
		result.Status = types.StatusFailed
		result.Checks = append(result.Checks, types.CheckResult{
			ID:         "SSH",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("SSH connection failed: %v", err),
			Suggestion: "Please check node IP, port, username, and key/password configuration",
		})
		return result, nil
	}
	defer cleanup()

	ctx := &NodeCheckContext{
		Node:      node,
		HccnIPs:   make(map[string]string),
		TLSStates: make(map[string]string),
	}

	return runNodeChecks(runner, node, result, ctx, enableConnectivityCheck)
}

// runNodeChecks executes H-01~H-05, then H-06~H-10 for 910-series
func runNodeChecks(
	runner executor.CommandRunner, node types.NodeInfo,
	result types.NodeResult, ctx *NodeCheckContext, enableConnectivityCheck bool,
) (types.NodeResult, *NodeCheckContext) {
	// H-01
	h01 := checkH01(runner)
	result.Checks = append(result.Checks, h01)
	if h01.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}

	// H-02
	h02 := checkH02(runner, node.Name)
	result.Checks = append(result.Checks, h02)
	if h02.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}

	// H-03
	h03, is910, npuSmiOut := checkH03(runner)
	result.Checks = append(result.Checks, h03)
	result.Is910Series = is910
	ctx.Is910Series = is910
	if h03.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}

	// H-04
	// Reuse the npu-smi output captured by H-03 to avoid a duplicate SSH round-trip.
	if npuSmiOut == "" {
		result.Status = types.StatusFailed
		result.Checks = append(result.Checks, types.CheckResult{
			ID:         "H-04",
			Status:     types.StatusFailed,
			Message:    "npu-smi info execution failed",
			Suggestion: "Please confirm npu-smi tool is available",
		})
		return result, nil
	}

	h04 := checkH04(npuSmiOut)
	result.Checks = append(result.Checks, h04)
	npuTotal, _, _ := parseNPUCount(npuSmiOut)

	// H-05
	h05 := checkH05(runner)
	result.Checks = append(result.Checks, h05)
	if h05.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}

	// skip H-06~H-10 for non-910 series
	if !is910 {
		result.SkippedHCCS = true
		result.SkipReason = "non-910 series, H-06~H-10 will be skipped"
		return result, ctx
	}
	return check910SeriesNode(runner, node910CheckInput{
		NodeName:                node.Name,
		NPUTotal:                npuTotal,
		EnableConnectivityCheck: enableConnectivityCheck,
	}, result, ctx)
}

// checkH01 checks NPU driver and firmware installation status
func checkH01(ssh executor.CommandRunner) types.CheckResult {
	stdout, _, err := ssh.Run("lsmod | grep drv_pcie_host")
	if err != nil || strings.TrimSpace(stdout) == "" {
		log.Info(fmt.Sprintf("node %s H-01: no lsmod output", ssh.NodeName()))
		return types.CheckResult{
			ID:      "H-01",
			Status:  types.StatusFailed,
			Message: "NPU driver module not detected (no lsmod output)",
			Suggestion: "Please confirm whether the node has an NPU card; if so, " +
				"refer to the Ascend driver/firmware installation guide",
		}
	}

	_, _, err = ssh.Run("npu-smi info")
	if err != nil {
		log.Info(fmt.Sprintf("node %s H-01: npu-smi info failed", ssh.NodeName()))
		return types.CheckResult{
			ID:         "H-01",
			Status:     types.StatusFailed,
			Message:    "Driver module loaded but npu-smi info execution failed",
			Suggestion: "Driver module load anomaly, recommend reinstalling the driver/firmware package",
		}
	}

	return types.CheckResult{
		ID:      "H-01",
		Status:  types.StatusPassed,
		Message: "NPU driver and firmware installed",
	}
}

// checkH02 checks Ascend Device Plugin installation status
func checkH02(ssh executor.CommandRunner, nodeName string) types.CheckResult {
	stdout, _, err := ssh.Run("kubectl get pods -A -o json")
	if err != nil {
		return types.CheckResult{
			ID:         "H-02",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Failed to execute kubectl on node %s: %v", nodeName, err),
			Suggestion: "Please confirm kubectl is installed and kubeconfig is properly configured on this node",
		}
	}

	podList, err := parsePodList(stdout)
	if err != nil {
		return types.CheckResult{
			ID:      "H-02",
			Status:  types.StatusFailed,
			Message: fmt.Sprintf("Failed to parse kubectl JSON output for node %s: %v", nodeName, err),
		}
	}

	pluginRunning := false
	for _, pod := range podList.Items {
		if !strings.Contains(pod.Metadata.Name, "ascend-device-plugin") {
			continue
		}
		if pod.Spec.NodeName == nodeName {
			if pod.Status.Phase == "Running" {
				pluginRunning = true
			} else {
				return types.CheckResult{
					ID:         "H-02",
					Status:     types.StatusFailed,
					Message:    "Ascend Device Plugin Pod is not in Running state",
					Suggestion: "Refer to the Ascend Device Plugin installation guide and confirm the DaemonSet is correctly deployed",
				}
			}
		}
	}

	if !pluginRunning {
		return types.CheckResult{
			ID:         "H-02",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Ascend Device Plugin Pod not found on node %s", nodeName),
			Suggestion: "Refer to the Ascend Device Plugin installation guide and confirm the DaemonSet is correctly deployed",
		}
	}

	// check if NPU resource is registered
	stdout, _, err = ssh.Run(fmt.Sprintf("kubectl get node %s -o jsonpath='{.status.capacity}'", nodeName))
	if err != nil || !strings.Contains(stdout, "huawei.com/Ascend") {
		return types.CheckResult{
			ID:     "H-02",
			Status: types.StatusFailed,
			Message: "Ascend Device Plugin is Running but NPU resource not registered " +
				"(no huawei.com/Ascend field in capacity)",
			Suggestion: "Refer to the Ascend Device Plugin installation guide " +
				"and confirm the DaemonSet is correctly deployed",
		}
	}

	return types.CheckResult{
		ID:      "H-02",
		Status:  types.StatusPassed,
		Message: "Ascend Device Plugin Running and NPU resource registered",
	}
}

// checkH03 checks NPU model; returns check result, whether it is 910 series, and the raw
// npu-smi stdout so callers (H-04) can reuse the same output instead of issuing another
// remote command. stdout is empty only when the command failed or output was unparsable.
func checkH03(ssh executor.CommandRunner) (types.CheckResult, bool, string) {
	stdout, _, err := ssh.Run("npu-smi info")
	if err != nil {
		return types.CheckResult{
			ID:         "H-03",
			Status:     types.StatusFailed,
			Message:    "npu-smi info execution failed",
			Suggestion: "Please confirm npu-smi tool is available",
		}, false, ""
	}

	model := parseNPUModel(stdout)
	if model == "" {
		return types.CheckResult{
			ID:         "H-03",
			Status:     types.StatusFailed,
			Message:    "Unable to parse NPU model from npu-smi info output",
			Suggestion: "Please confirm npu-smi tool is available and output format is correct",
		}, false, ""
	}

	is910 := strings.Contains(model, "910")
	if !is910 {
		return types.CheckResult{
			ID:      "H-03",
			Status:  types.StatusWarning,
			Message: fmt.Sprintf("NPU model is non-910 series: %s", model),
			Suggestion: "The pre-flight checks are designed for 910 series; " +
				"non-910 series may have different check items and H-06~H-10 will be skipped",
		}, false, stdout
	}

	return types.CheckResult{
		ID:      "H-03",
		Status:  types.StatusPassed,
		Message: fmt.Sprintf("NPU model: %s", model),
	}, true, stdout
}

// parseNPUModel extracts the NPU model name from npu-smi info output.
// Matches lines like: | 0     910B4               | OK  ...
// Returns the model name of the first card found.
func parseNPUModel(output string) string {
	// strip ANSI escape sequences
	clean := ansiEscape.ReplaceAllString(output, "")

	// match "| <id>  <model> | OK/Warning/Alarm/Critical/UNKNOWN ..."
	m := npuModelRe.FindStringSubmatch(clean)
	if len(m) >= regexMinMatchLen {
		return m[npuIDGroupIdx]
	}
	return ""
}

// checkH04 reports NPU available count (info only) using pre-fetched npu-smi output
func checkH04(stdout string) types.CheckResult {
	if stdout == "" {
		return types.CheckResult{
			ID:      "H-04",
			Status:  types.StatusInfo,
			Message: "Unable to get NPU available count",
		}
	}

	total, available, unhealthy := parseNPUCount(stdout)
	msg := fmt.Sprintf("NPU available: %d/%d", available, total)
	if len(unhealthy) > 0 {
		msg += fmt.Sprintf(", unhealthy: %s", strings.Join(unhealthy, ", "))
	}
	detail := map[string]interface{}{
		"available": available,
		"total":     total,
		"unhealthy": unhealthy,
	}

	if available == 0 {
		return types.CheckResult{
			ID:         "H-04",
			Status:     types.StatusWarning,
			Message:    msg + ", no NPU cards currently available",
			Detail:     detail,
			Suggestion: "Consider releasing occupied resources or adjusting workloads if insufficient",
		}
	}

	return types.CheckResult{
		ID:      "H-04",
		Status:  types.StatusInfo,
		Message: msg,
		Detail:  detail,
	}
}

// parseNPUCount parses total and available NPU card counts from npu-smi info output.
// Total is counted from device rows: | <id>  <model> | OK/Warning/Alarm/Critical/UNKNOWN ... |
// Occupied is determined from process rows: | <npu> <chip> | <pid> | <name> | <mem> |
func parseNPUCount(output string) (total, available int, unhealthy []string) {
	// strip ANSI escape sequences
	clean := ansiEscape.ReplaceAllString(output, "")

	// count total NPUs and detect unhealthy ones
	unhealthyIDs := map[string]bool{}
	deviceMatches := npuDeviceRe.FindAllStringSubmatch(clean, -1)
	for _, m := range deviceMatches {
		total++
		if m[npuStatusGroupIdx] != "OK" {
			unhealthy = append(unhealthy, fmt.Sprintf("NPU%s(%s)", m[npuIDGroupIdx], m[npuStatusGroupIdx]))
			unhealthyIDs[m[npuIDGroupIdx]] = true
		}
	}

	// count occupied NPUs from process table: "| <npu> <chip> | <pid> | <name> | <mem> |"
	matches := npuProcessRe.FindAllStringSubmatch(clean, -1)
	occupiedIDs := map[string]bool{}
	for _, m := range matches {
		occupiedIDs[m[1]] = true
	}

	// unavailable is the union of unhealthy and occupied (avoid double-counting)
	unavailable := map[string]bool{}
	for id := range unhealthyIDs {
		unavailable[id] = true
	}
	for id := range occupiedIDs {
		unavailable[id] = true
	}

	available = total - len(unavailable)
	if available < 0 {
		available = 0
	}
	return
}

// checkH05 checks whether key host files and directories exist
func checkH05(ssh executor.CommandRunner) types.CheckResult {
	paths := []string{
		"/usr/local/dcmi",
		"/usr/local/Ascend/driver/lib64",
		"/usr/local/bin/npu-smi",
		"/usr/local/Ascend/driver/version.info",
		"/etc/ascend_install.info",
		"/etc/hccn.conf",
	}

	var missing []string
	for _, p := range paths {
		_, _, err := ssh.Run(fmt.Sprintf("ls %s", p))
		if err != nil {
			missing = append(missing, p)
		}
	}

	if len(missing) > 0 {
		return types.CheckResult{
			ID:         "H-05",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Missing paths or files: %s", strings.Join(missing, ", ")),
			Suggestion: "Please verify environment completeness and supply the missing files or directories",
		}
	}

	return types.CheckResult{
		ID:      "H-05",
		Status:  types.StatusPassed,
		Message: "Key host files and directories are complete",
	}
}

func check910SeriesNode(ssh executor.CommandRunner,
	input node910CheckInput,
	result types.NodeResult, ctx *NodeCheckContext) (types.NodeResult, *NodeCheckContext) {
	phyIDs, err := getPhyIDs(ssh)
	if err != nil || len(phyIDs) == 0 {
		log.Error(fmt.Sprintf("node %s failed to get NPU IDs: %v", input.NodeName, err))
		result.Status = types.StatusFailed
		result.Checks = append(result.Checks, types.CheckResult{
			ID:         "H-06",
			Status:     types.StatusFailed,
			Message:    "Failed to get NPU physical IDs from /dev/davinci*",
			Suggestion: "Please confirm NPU devices are present and driver is loaded",
		})
		return result, nil
	}
	if input.NPUTotal > 0 && len(phyIDs) != input.NPUTotal {
		result.Status = types.StatusFailed
		result.Checks = append(result.Checks, types.CheckResult{
			ID:     "H-06",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("NPU device file count mismatch: "+
				"npu-smi reports %d cards but /dev/davinci* found %d IDs %v",
				input.NPUTotal, len(phyIDs), phyIDs),
			Suggestion: "Please check driver installation and device file integrity",
		})
		return result, nil
	}
	ctx.PhyIDs = phyIDs

	// H-06
	h06, hccnIPs := checkH06(ssh, phyIDs)
	result.Checks = append(result.Checks, h06)
	if h06.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}
	ctx.HccnIPs = hccnIPs

	// H-07
	h07, tlsStates := checkH07(ssh, phyIDs, hccnIPs, input.EnableConnectivityCheck)
	result.Checks = append(result.Checks, h07...)
	ctx.TLSStates = tlsStates
	for _, c := range h07 {
		if c.Status == types.StatusFailed {
			result.Status = types.StatusFailed
			return result, nil
		}
	}

	// H-08
	h08 := checkH08(ssh)
	result.Checks = append(result.Checks, h08)
	if h08.Status == types.StatusFailed {
		result.Status = types.StatusFailed
		return result, nil
	}

	return result, ctx
}
