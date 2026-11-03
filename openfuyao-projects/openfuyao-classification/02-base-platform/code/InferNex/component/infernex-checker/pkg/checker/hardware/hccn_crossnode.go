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
	"strings"
	"time"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/log"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// checkH10 H-10 cross-node NIC-side RoCE RDMA communication
func checkH10(ctxs []*NodeCheckContext, enableConnectivityCheck bool) []types.CheckResult {
	if len(ctxs) < crossNodeMinCount {
		return []types.CheckResult{{
			ID:     "H-10",
			Status: types.StatusSkipped,
			Message: fmt.Sprintf("Only %d 910-series node(s) passed single-node checks, "+
				"skipping cross-node check (at least %d required)", len(ctxs), crossNodeMinCount),
		}}
	}

	// step 1: cross-node IP uniqueness
	r1 := crossNodeIPUniqueness(ctxs)
	if r1.Status == types.StatusFailed {
		return []types.CheckResult{r1}
	}

	// step 2: cross-node TLS consistency
	r2 := crossNodeTLSConsistency(ctxs)
	if r2.Status == types.StatusFailed {
		return []types.CheckResult{r1, r2}
	}

	// step 3: cross-node connectivity (conditional based on enableConnectivityCheck)
	if !enableConnectivityCheck {
		r3 := types.CheckResult{
			ID:      "H-10",
			Status:  types.StatusSkipped,
			Message: "Step 3: Cross-node connectivity check skipped (use --enable-connectivity-check to enable)",
		}
		return []types.CheckResult{r1, r2, r3}
	}

	return []types.CheckResult{r1, r2, crossNodeConnectivity(ctxs)}
}

func crossNodeIPUniqueness(ctxs []*NodeCheckContext) types.CheckResult {
	globalIPs := map[string]string{} // ip -> "nodeName NICid"
	var dupIPs []string
	for _, ctx := range ctxs {
		for nic, ip := range ctx.HccnIPs {
			key := fmt.Sprintf("%s NIC%s", ctx.Node.Name, nic)
			if prev, dup := globalIPs[ip]; dup {
				dupIPs = append(dupIPs, fmt.Sprintf("%s and %s share IP %s", prev, key, ip))
			} else {
				globalIPs[ip] = key
			}
		}
	}
	if len(dupIPs) > 0 {
		return types.CheckResult{
			ID:         "H-10",
			Status:     types.StatusFailed,
			Message:    fmt.Sprintf("Step 1: Duplicate IPs across nodes: %s", strings.Join(dupIPs, "; ")),
			Suggestion: "Re-plan NIC IPs across all nodes to ensure global uniqueness",
		}
	}
	return types.CheckResult{
		ID:      "H-10",
		Status:  types.StatusPassed,
		Message: "Step 1: Cross-node NIC IPs are unique",
	}
}

func crossNodeTLSConsistency(ctxs []*NodeCheckContext) types.CheckResult {
	firstTLS := ""
	firstTLSLabel := ""
	var inconsistentTLS []string
	for _, ctx := range ctxs {
		for id, state := range ctx.TLSStates {
			label := fmt.Sprintf("%s NPU%s=%s", ctx.Node.Name, id, state)
			if firstTLS == "" {
				firstTLS = state
				firstTLSLabel = label
			} else if normalizeTLS(state) != normalizeTLS(firstTLS) {
				inconsistentTLS = append(inconsistentTLS, fmt.Sprintf("%s (expected %s)", label, firstTLSLabel))
			} else {
				// TLS state is consistent, no action needed
			}
		}
	}
	if len(inconsistentTLS) > 0 {
		return types.CheckResult{
			ID:     "H-10",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("Step 2: Cross-node TLS configuration inconsistent: %s",
				strings.Join(inconsistentTLS, "; ")),
			Suggestion: "Unify TLS switch configuration across all nodes",
		}
	}
	return types.CheckResult{
		ID:      "H-10",
		Status:  types.StatusPassed,
		Message: fmt.Sprintf("Step 2: Cross-node TLS configuration is consistent (%s)", firstTLS),
	}
}

func crossNodePingOne(src *NodeCheckContext, ctxs []*NodeCheckContext) []string {
	runner, cleanup, err := newSSHRunner(src.Node)
	if err != nil {
		log.Error(fmt.Sprintf("H-10 failed to connect to node %s: %v", src.Node.Name, err))
		return nil
	}
	defer cleanup()

	var failed []string
	for _, dstCtx := range ctxs {
		if dstCtx.Node.Name == src.Node.Name {
			continue
		}
		failed = append(failed, pingToDestination(runner, src, dstCtx)...)
	}
	return failed
}

type pingPath struct {
	srcName  string
	srcPhyID int
	dstName  string
	dstNic   string
	dstIP    string
}

func pingToDestination(ssh executor.CommandRunner, src *NodeCheckContext, dstCtx *NodeCheckContext) []string {
	var failed []string
	for dstNic, dstIP := range dstCtx.HccnIPs {
		for _, i := range src.PhyIDs {
			if result := pingOnePath(ssh, pingPath{
				srcName:  src.Node.Name,
				srcPhyID: i,
				dstName:  dstCtx.Node.Name,
				dstNic:   dstNic,
				dstIP:    dstIP,
			}); result != "" {
				failed = append(failed, result)
			}
		}
	}
	return failed
}

func pingOnePath(ssh executor.CommandRunner, path pingPath) string {
	for attempt := 0; attempt < hccnPingMaxRetries; attempt++ {
		if hccnPing(ssh, path.srcPhyID, path.dstIP) {
			return ""
		}
		if attempt < hccnPingMaxRetries-1 {
			time.Sleep(hccnPingRetryInterval)
		}
	}
	return fmt.Sprintf("%s:NPU%d -> %s:%s(%s)",
		path.srcName, path.srcPhyID, path.dstName, path.dstNic, path.dstIP)
}

func crossNodeConnectivity(ctxs []*NodeCheckContext) types.CheckResult {
	var pingFailed []string
	for _, srcCtx := range ctxs {
		pingFailed = append(pingFailed, crossNodePingOne(srcCtx, ctxs)...)
	}
	if len(pingFailed) > 0 {
		return types.CheckResult{
			ID:     "H-10",
			Status: types.StatusFailed,
			Message: fmt.Sprintf("Step 3: Cross-node connectivity anomaly: "+
				"%s", strings.Join(pingFailed, "; ")),
			Suggestion: "Please check RoCE network configuration, physical links, and switch settings",
		}
	}
	return types.CheckResult{
		ID:      "H-10",
		Status:  types.StatusPassed,
		Message: "Step 3: Communication between all nodes is normal",
	}
}
