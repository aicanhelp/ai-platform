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
	"errors"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---- parseHccnConf ----

func TestParseHccnConf_Normal(t *testing.T) {
	// No spaces around '=' so parts[0] == "address_0" and TrimPrefix gives "0"
	content := "address_0=192.168.1.1\nnetmask_0=255.255.255.0\naddress_1=192.168.1.2\nnetmask_1=255.255.255.0\nother_key=value\n"
	ipMap, maskMap := parseHccnConf(content)
	if ipMap["0"] != "192.168.1.1" {
		t.Errorf("expected 192.168.1.1 for address_0, got %q", ipMap["0"])
	}
	if maskMap["0"] != "255.255.255.0" {
		t.Errorf("expected 255.255.255.0 for netmask_0, got %q", maskMap["0"])
	}
	if ipMap["1"] != "192.168.1.2" {
		t.Errorf("expected 192.168.1.2 for address_1, got %q", ipMap["1"])
	}
}

func TestParseHccnConf_Empty(t *testing.T) {
	ipMap, maskMap := parseHccnConf("")
	if len(ipMap) != 0 || len(maskMap) != 0 {
		t.Error("expected empty maps for empty content")
	}
}

func TestParseHccnConf_NoEquals(t *testing.T) {
	content := "address_0 192.168.1.1\nnetmask_0 255.255.255.0\n"
	ipMap, maskMap := parseHccnConf(content)
	if len(ipMap) != 0 || len(maskMap) != 0 {
		t.Error("expected empty maps when no = sign")
	}
}

// ---- checkH06MissingIPs ----

func TestCheckH06MissingIPs_AllPresent(t *testing.T) {
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	r := checkH06MissingIPs([]int{0, 1}, ipMap)
	if r.Status == types.StatusFailed {
		t.Errorf("expected no failure, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH06MissingIPs_Missing(t *testing.T) {
	ipMap := map[string]string{"0": "192.168.1.1"}
	r := checkH06MissingIPs([]int{0, 1}, ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH06MissingIPs_EmptyValue(t *testing.T) {
	ipMap := map[string]string{"0": "192.168.1.1", "1": ""}
	r := checkH06MissingIPs([]int{0, 1}, ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed for empty IP, got %v", r.Status)
	}
}

// ---- checkH06MissingMasks ----

func TestCheckH06MissingMasks_AllPresent(t *testing.T) {
	maskMap := map[string]string{"0": "255.255.255.0", "1": "255.255.255.0"}
	r := checkH06MissingMasks([]int{0, 1}, maskMap)
	if r.Status == types.StatusFailed {
		t.Errorf("expected no failure, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH06MissingMasks_Missing(t *testing.T) {
	maskMap := map[string]string{"0": "255.255.255.0"}
	r := checkH06MissingMasks([]int{0, 1}, maskMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- checkH06IPFormatAndUniqueness ----

func TestCheckH06IPFormatAndUniqueness_Valid(t *testing.T) {
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	r := checkH06IPFormatAndUniqueness(ipMap)
	if r.Status == types.StatusFailed {
		t.Errorf("expected pass, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH06IPFormatAndUniqueness_Invalid(t *testing.T) {
	ipMap := map[string]string{"0": "not-an-ip"}
	r := checkH06IPFormatAndUniqueness(ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed for invalid IP, got %v", r.Status)
	}
}

func TestCheckH06IPFormatAndUniqueness_Duplicate(t *testing.T) {
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.1"}
	r := checkH06IPFormatAndUniqueness(ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed for duplicate IP, got %v", r.Status)
	}
}

// ---- parseTLSState ----

func TestParseTLSState_Enable(t *testing.T) {
	out := "tls switch[1]"
	if parseTLSState(out) != "enable" {
		t.Errorf("expected enable")
	}
}

func TestParseTLSState_Disable(t *testing.T) {
	out := "tls switch[0]"
	if parseTLSState(out) != "disable" {
		t.Errorf("expected disable")
	}
}

func TestParseTLSState_NoCert(t *testing.T) {
	out := "No Certificate Found"
	if parseTLSState(out) != "no_cert" {
		t.Errorf("expected no_cert")
	}
}

func TestParseTLSState_Unknown(t *testing.T) {
	out := "some random output"
	if parseTLSState(out) != "unknown" {
		t.Errorf("expected unknown")
	}
}

// ---- normalizeTLS ----

func TestNormalizeTLS(t *testing.T) {
	if normalizeTLS("no_cert") != "disable" {
		t.Error("no_cert should normalize to disable")
	}
	if normalizeTLS("enable") != "enable" {
		t.Error("enable should stay enable")
	}
	if normalizeTLS("disable") != "disable" {
		t.Error("disable should stay disable")
	}
}

// ---- checkH06 ----

func goodHccnConf() string {
	return `address_0=192.168.1.1
netmask_0=255.255.255.0
address_1=192.168.1.2
netmask_1=255.255.255.0
`
}

func TestCheckH06_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat /etc/hccn.conf": {stdout: goodHccnConf()},
		},
	}
	r, ipMap := checkH06(m, []int{0, 1})
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
	if ipMap["0"] != "192.168.1.1" {
		t.Errorf("unexpected ipMap: %v", ipMap)
	}
}

func TestCheckH06_ReadFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat /etc/hccn.conf": {err: errors.New("read error")},
		},
	}
	r, _ := checkH06(m, []int{0, 1})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH06_MissingIP(t *testing.T) {
	conf := `address_0 = 192.168.1.1
netmask_0 = 255.255.255.0
netmask_1 = 255.255.255.0
`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat /etc/hccn.conf": {stdout: conf},
		},
	}
	r, _ := checkH06(m, []int{0, 1})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH06_MissingMask(t *testing.T) {
	conf := `address_0 = 192.168.1.1
address_1 = 192.168.1.2
netmask_0 = 255.255.255.0
`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat /etc/hccn.conf": {stdout: conf},
		},
	}
	r, _ := checkH06(m, []int{0, 1})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH06_BadIP(t *testing.T) {
	conf := `address_0 = bad-ip
netmask_0 = 255.255.255.0
address_1 = 192.168.1.2
netmask_1 = 255.255.255.0
`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"cat /etc/hccn.conf": {stdout: conf},
		},
	}
	r, _ := checkH06(m, []int{0, 1})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed for bad IP, got %v", r.Status)
	}
}

// ---- checkH07TLSConsistency ----

func TestCheckH07TLSConsistency_Consistent(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[1]"},
		},
	}
	r, tlsStates := checkH07TLSConsistency(m, []int{0, 1})
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
	if tlsStates["0"] != "enable" || tlsStates["1"] != "enable" {
		t.Errorf("unexpected tls states: %v", tlsStates)
	}
}

func TestCheckH07TLSConsistency_Inconsistent(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[0]"},
		},
	}
	r, _ := checkH07TLSConsistency(m, []int{0, 1})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH07TLSConsistency_NoCertTreatedAsDisable(t *testing.T) {
	// no_cert normalizes to disable, so disable + no_cert should be consistent
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[0]"},
			"hccn_tool -i 1 -tls -g": {stderr: "No Certificate Found"},
		},
	}
	r, _ := checkH07TLSConsistency(m, []int{0, 1})
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed (no_cert==disable), got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH07TLSConsistency_Error(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {err: errors.New("command failed"), stderr: "error"},
		},
	}
	r, _ := checkH07TLSConsistency(m, []int{0})
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- checkH07Connectivity ----

func TestCheckH07Connectivity_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "ping success, 0.00% packet loss"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	r := checkH07Connectivity(m, []int{0, 1}, ipMap)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH07Connectivity_PingFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "100.00% packet loss"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	r := checkH07Connectivity(m, []int{0, 1}, ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH07Connectivity_CommandExecuteFailed(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "Command execute failed"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	r := checkH07Connectivity(m, []int{0}, ipMap)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- getPhyIDs ----

func TestGetPhyIDs_Normal(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*": {stdout: "/dev/davinci0 /dev/davinci1 /dev/davinci_manager"},
		},
	}
	ids, err := getPhyIDs(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ids) != 2 {
		t.Errorf("expected 2 ids, got %v", ids)
	}
}

func TestGetPhyIDs_Error(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*": {err: errors.New("no such file")},
		},
	}
	ids, err := getPhyIDs(m)
	if err == nil {
		t.Error("expected error")
	}
	if len(ids) != 0 {
		t.Errorf("expected empty ids, got %v", ids)
	}
}

func TestGetPhyIDs_OnlyManager(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*": {stdout: "/dev/davinci_manager"},
		},
	}
	ids, err := getPhyIDs(m)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(ids) != 0 {
		t.Errorf("expected empty ids, got %v", ids)
	}
}

// ---- checkH07 ----

func TestCheckH07_TLSFailEarlyReturn(t *testing.T) {
	// TLS inconsistency should cause early return (only 1 result)
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[0]"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	results, tlsStates := checkH07(m, []int{0, 1}, ipMap, true)
	if len(results) != 1 {
		t.Errorf("expected 1 result on TLS fail, got %d", len(results))
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", results[0].Status)
	}
	_ = tlsStates
}

func TestCheckH07_TLSPassConnectivityPass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool":              {stdout: "0.00% packet loss"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	results, _ := checkH07(m, []int{0, 1}, ipMap, true)
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected TLS passed, got %v", results[0].Status)
	}
}

func TestCheckH07_TLSPassConnectivityFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool":              {stdout: "100.00% packet loss"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	results, _ := checkH07(m, []int{0, 1}, ipMap, true)
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	if results[1].Status != types.StatusFailed {
		t.Errorf("expected connectivity failed, got %v", results[1].Status)
	}
}

// TestCheckH07_ConnectivityDisabled tests that Step 2 is skipped when enableConnectivityCheck=false
func TestCheckH07_ConnectivityDisabled(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[1]"},
		},
	}
	ipMap := map[string]string{"0": "192.168.1.1", "1": "192.168.1.2"}
	results, _ := checkH07(m, []int{0, 1}, ipMap, false)
	if len(results) != 2 {
		t.Errorf("expected 2 results, got %d", len(results))
	}
	if results[0].Status != types.StatusPassed {
		t.Errorf("expected TLS passed, got %v", results[0].Status)
	}
	if results[1].Status != types.StatusSkipped {
		t.Errorf("expected Step 2 skipped, got %v", results[1].Status)
	}
	if !strings.Contains(results[1].Message, "skipped") {
		t.Errorf("expected skip message, got: %s", results[1].Message)
	}
}

// ---- check910SeriesNode ----

func TestCheck910SeriesNode_GetPhyIDsFail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*": {err: errors.New("no such file")},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	result, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 2, EnableConnectivityCheck: true}, nodeResult, ctx)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", result.Status)
	}
	if retCtx != nil {
		t.Error("expected nil ctx on failure")
	}
}

func TestCheck910SeriesNode_PhyIDCountMismatch(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			// Only 1 device but npuTotal=2
			"ls /dev/davinci*": {stdout: "/dev/davinci0"},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	result, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 2, EnableConnectivityCheck: true}, nodeResult, ctx)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed for count mismatch, got %v", result.Status)
	}
	if retCtx != nil {
		t.Error("expected nil ctx on mismatch")
	}
}

func TestCheck910SeriesNode_H06Fail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*":   {stdout: "/dev/davinci0 /dev/davinci1"},
			"cat /etc/hccn.conf": {err: errors.New("read error")},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	result, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 2, EnableConnectivityCheck: true}, nodeResult, ctx)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed on H06 fail, got %v", result.Status)
	}
	if retCtx != nil {
		t.Error("expected nil ctx when H06 fails")
	}
}

func TestCheck910SeriesNode_H07Fail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*":   {stdout: "/dev/davinci0 /dev/davinci1"},
			"cat /etc/hccn.conf": {stdout: "address_0=192.168.1.1\nnetmask_0=255.255.255.0\naddress_1=192.168.1.2\nnetmask_1=255.255.255.0\n"},
			// TLS inconsistent -> H07 fails
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[0]"},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	result, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 2, EnableConnectivityCheck: true}, nodeResult, ctx)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed on H07 fail, got %v", result.Status)
	}
	if retCtx != nil {
		t.Error("expected nil ctx when H07 fails")
	}
}

func TestCheck910SeriesNode_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*":       {stdout: "/dev/davinci0 /dev/davinci1"},
			"cat /etc/hccn.conf":     {stdout: "address_0=192.168.1.1\nnetmask_0=255.255.255.0\naddress_1=192.168.1.2\nnetmask_1=255.255.255.0\n"},
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool -i 1 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool":              {stdout: "0.00% packet loss"},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	result, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 2, EnableConnectivityCheck: true}, nodeResult, ctx)
	if result.Status == types.StatusFailed {
		t.Errorf("expected not failed, got %v: checking checks", result.Status)
		for _, c := range result.Checks {
			t.Logf("  check %s: %v - %s", c.ID, c.Status, c.Message)
		}
	}
	if retCtx == nil {
		t.Error("expected non-nil ctx on success")
	}
}

func TestCheck910SeriesNode_NpuTotalZeroNoMismatch(t *testing.T) {
	// npuTotal=0 means skip count check
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"ls /dev/davinci*":       {stdout: "/dev/davinci0"},
			"cat /etc/hccn.conf":     {stdout: "address_0=192.168.1.1\nnetmask_0=255.255.255.0\n"},
			"hccn_tool -i 0 -tls -g": {stdout: "tls switch[1]"},
			"hccn_tool":              {stdout: "0.00% packet loss"},
		},
	}
	nodeResult := types.NodeResult{Name: "node1", Status: types.StatusPassed}
	ctx := &NodeCheckContext{Node: types.NodeInfo{Name: "node1"}}
	// npuTotal=0 -> no mismatch check
	_, retCtx := check910SeriesNode(m, node910CheckInput{NodeName: "node1", NPUTotal: 0, EnableConnectivityCheck: true}, nodeResult, ctx)
	if retCtx == nil {
		t.Error("expected non-nil ctx when npuTotal=0 (no mismatch)")
	}
}

// ---- pingToDestination and pingOnePath ----

func TestPingOnePath_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "0.00% packet loss"},
		},
	}
	path := pingPath{srcName: "node1", srcPhyID: 0, dstName: "node2", dstNic: "0", dstIP: "192.168.1.2"}
	result := pingOnePath(m, path)
	if result != "" {
		t.Errorf("expected empty string on pass, got %q", result)
	}
}

func TestPingOnePath_PacketLoss(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "100.00% packet loss"},
		},
	}
	path := pingPath{srcName: "node1", srcPhyID: 0, dstName: "node2", dstNic: "0", dstIP: "192.168.1.2"}
	result := pingOnePath(m, path)
	if result == "" {
		t.Error("expected failure string, got empty")
	}
}

func TestPingOnePath_CommandFailed(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "Command execute failed"},
		},
	}
	path := pingPath{srcName: "node1", srcPhyID: 0, dstName: "node2", dstNic: "0", dstIP: "192.168.1.2"}
	result := pingOnePath(m, path)
	if result == "" {
		t.Error("expected failure string, got empty")
	}
}

func TestPingOnePath_Error(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {err: errors.New("connection failed")},
		},
	}
	path := pingPath{srcName: "node1", srcPhyID: 0, dstName: "node2", dstNic: "0", dstIP: "192.168.1.2"}
	result := pingOnePath(m, path)
	if result == "" {
		t.Error("expected failure string on error, got empty")
	}
}

func TestPingToDestination_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "0.00% packet loss"},
		},
	}
	src := &NodeCheckContext{
		Node:   types.NodeInfo{Name: "node1"},
		PhyIDs: []int{0},
	}
	dst := &NodeCheckContext{
		Node:    types.NodeInfo{Name: "node2"},
		HccnIPs: map[string]string{"0": "192.168.1.2"},
	}
	failed := pingToDestination(m, src, dst)
	if len(failed) != 0 {
		t.Errorf("expected no failures, got %v", failed)
	}
}

func TestPingToDestination_Fail(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"hccn_tool": {stdout: "100.00% packet loss"},
		},
	}
	src := &NodeCheckContext{
		Node:   types.NodeInfo{Name: "node1"},
		PhyIDs: []int{0},
	}
	dst := &NodeCheckContext{
		Node:    types.NodeInfo{Name: "node2"},
		HccnIPs: map[string]string{"0": "192.168.1.2"},
	}
	failed := pingToDestination(m, src, dst)
	if len(failed) == 0 {
		t.Error("expected failures, got none")
	}
}
