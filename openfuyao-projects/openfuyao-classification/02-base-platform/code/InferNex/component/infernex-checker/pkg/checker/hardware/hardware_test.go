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
	"fmt"
	"strings"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// mockRunner is a fake CommandRunner for unit tests.
type mockRunner struct {
	nodeName string
	// responses maps command substring -> (stdout, stderr, err)
	responses map[string]mockResponse
}

type mockResponse struct {
	stdout string
	stderr string
	err    error
}

func (m *mockRunner) Run(cmd string) (string, string, error) {
	for key, resp := range m.responses {
		if key == cmd {
			return resp.stdout, resp.stderr, resp.err
		}
	}
	// fallback: check if cmd contains key as substring
	for key, resp := range m.responses {
		if len(key) > 0 && contains(cmd, key) {
			return resp.stdout, resp.stderr, resp.err
		}
	}
	return "", "", nil
}

func (m *mockRunner) NodeName() string {
	return m.nodeName
}

func contains(s, sub string) bool {
	return len(sub) > 0 && len(s) >= len(sub) && (s == sub || findSubstring(s, sub))
}

func findSubstring(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

// npuSmiOutput is a realistic npu-smi info output snippet used across tests.
const npuSmiOutput = `+-------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc2                 Version: 23.0.rc2                                       |
+-------------------------------------------------------------------------------------------+
| NPU   Name                  Health   Power(W)   Temp(C)   Hugepages-Usage(page)   AICore |
| Chip                         Bus-Id        AIServer(MHz)                                  |
+==========================================================================================+
| 0     910B4               | OK      | 67.3      | 43       | 0      / 0              | 0  |
| 0     0x0000:C1:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
| 1     910B4               | OK      | 65.1      | 42       | 0      / 0              | 0  |
| 0     0x0000:81:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
+-------------------------------------------------------------------------------------------+
| NPU     Chip   Process id   Process name         Device memory(MB)                        |
+==========================================================================================+
+-------------------------------------------------------------------------------------------+
`

const npuSmiOccupiedOutput = `+-------------------------------------------------------------------------------------------+
| 0     910B4               | OK      | 67.3      | 43       | 0      / 0              | 0  |
| 0     0x0000:C1:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
| 1     910B4               | OK      | 65.1      | 42       | 0      / 0              | 0  |
| 0     0x0000:81:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
+-------------------------------------------------------------------------------------------+
| NPU     Chip   Process id   Process name         Device memory(MB)                        |
+==========================================================================================+
| 0     0      | 12345     | python              | 4096                                    |
+-------------------------------------------------------------------------------------------+
`

const npuSmiUnhealthyOutput = `+-------------------------------------------------------------------------------------------+
| 0     910B4               | Warning | 67.3      | 43       | 0      / 0              | 0  |
| 0     0x0000:C1:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
| 1     910B4               | OK      | 65.1      | 42       | 0      / 0              | 0  |
| 0     0x0000:81:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
`

// ---- parseNPUModel ----

func TestParseNPUModel_Found(t *testing.T) {
	model := parseNPUModel(npuSmiOutput)
	if model != "910B4" {
		t.Errorf("expected 910B4, got %q", model)
	}
}

func TestParseNPUModel_Empty(t *testing.T) {
	model := parseNPUModel("no match here")
	if model != "" {
		t.Errorf("expected empty, got %q", model)
	}
}

func TestParseNPUModel_AnsiStripped(t *testing.T) {
	ansiOutput := "\x1B[0m| 0     910B4               | OK  |\x1B[0m"
	model := parseNPUModel(ansiOutput)
	if model != "910B4" {
		t.Errorf("expected 910B4 after ANSI strip, got %q", model)
	}
}

// ---- parseNPUCount ----

func TestParseNPUCount_Normal(t *testing.T) {
	total, available, unhealthy := parseNPUCount(npuSmiOutput)
	if total != 2 {
		t.Errorf("expected total=2, got %d", total)
	}
	if available != 2 {
		t.Errorf("expected available=2, got %d", available)
	}
	if len(unhealthy) != 0 {
		t.Errorf("expected no unhealthy, got %v", unhealthy)
	}
}

func TestParseNPUCount_Occupied(t *testing.T) {
	total, available, unhealthy := parseNPUCount(npuSmiOccupiedOutput)
	if total != 2 {
		t.Errorf("expected total=2, got %d", total)
	}
	if available != 1 {
		t.Errorf("expected available=1, got %d", available)
	}
	if len(unhealthy) != 0 {
		t.Errorf("expected no unhealthy, got %v", unhealthy)
	}
}

func TestParseNPUCount_Unhealthy(t *testing.T) {
	total, available, unhealthy := parseNPUCount(npuSmiUnhealthyOutput)
	if total != 2 {
		t.Errorf("expected total=2, got %d", total)
	}
	if len(unhealthy) != 1 {
		t.Errorf("expected 1 unhealthy, got %v", unhealthy)
	}
	// available = total - occupied - unhealthy = 2 - 0 - 1 = 1
	if available != 1 {
		t.Errorf("expected available=1, got %d", available)
	}
}

func TestParseNPUCount_Empty(t *testing.T) {
	total, available, unhealthy := parseNPUCount("")
	if total != 0 || available != 0 || len(unhealthy) != 0 {
		t.Errorf("expected all zero/empty for empty input, got %d %d %v", total, available, unhealthy)
	}
}

// ---- checkH01 ----

func TestCheckH01_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"lsmod | grep drv_pcie_host": {stdout: "drv_pcie_host 12345 0"},
			"npu-smi info":               {stdout: npuSmiOutput},
		},
	}
	r := checkH01(m)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH01_NoLsmod(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"lsmod | grep drv_pcie_host": {stdout: ""},
		},
	}
	r := checkH01(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH01_LsmodErr(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"lsmod | grep drv_pcie_host": {err: errors.New("cmd error")},
		},
	}
	r := checkH01(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH01_NpuSmiErr(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"lsmod | grep drv_pcie_host": {stdout: "drv_pcie_host"},
			"npu-smi info":               {err: errors.New("not found")},
		},
	}
	r := checkH01(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- checkH02 ----

func devicePluginPodListJSON(nodeName, phase string) string {
	return fmt.Sprintf(`{"items":[{"metadata":{"name":"ascend-device-plugin-%s"},`+
		`"spec":{"nodeName":"%s"},"status":{"phase":"%s"}}]}`, nodeName, nodeName, phase)
}

func TestCheckH02_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: devicePluginPodListJSON("node1", "Running"),
			},
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `{"cpu":"96","huawei.com/Ascend910B4":"8"}`,
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH02_JSONParseError(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {stdout: "not-json"},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
	if !contains(r.Message, "Failed to parse kubectl JSON output for node node1:") {
		t.Errorf("expected parse error with node name, got %q", r.Message)
	}
}

func TestCheckH02_PluginNotFound(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: `{"items":[{"metadata":{"name":"coredns-abc123"},` +
					`"spec":{"nodeName":"node1"},"status":{"phase":"Running"}}]}`,
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH02_PluginNotRunning(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: devicePluginPodListJSON("node1", "Pending"),
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH02_ResourceNotRegistered(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: devicePluginPodListJSON("node1", "Running"),
			},
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `{"cpu":"96"}`,
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

func TestCheckH02_DoesNotMatchNodeNamePrefix(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: devicePluginPodListJSON("node10", "Running"),
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed when only node10 has a plugin pod, got %v", r.Status)
	}
}

func TestCheckH02_BothNodesRunning(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"kubectl get pods -A -o json": {
				stdout: `{"items":[` +
					`{"metadata":{"name":"ascend-device-plugin-node1"},"spec":{"nodeName":"node1"},"status":{"phase":"Running"}},` +
					`{"metadata":{"name":"ascend-device-plugin-node10"},"spec":{"nodeName":"node10"},"status":{"phase":"Running"}}]}`,
			},
			"kubectl get node node1 -o jsonpath='{.status.capacity}'": {
				stdout: `{"huawei.com/Ascend910B4":"8"}`,
			},
		},
	}
	r := checkH02(m, "node1")
	if r.Status != types.StatusPassed {
		t.Errorf("expected node1 check to pass, got %v: %s", r.Status, r.Message)
	}
}

// ---- checkH03 ----

func TestCheckH03_910Series(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: npuSmiOutput},
		},
	}
	r, is910, _ := checkH03(m)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
	if !is910 {
		t.Error("expected is910=true")
	}
}

func TestCheckH03_NonSeries(t *testing.T) {
	nonSeries := `| 0     310P3               | OK      | 67.3      | 43       | 0      / 0              | 0  |`
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: nonSeries},
		},
	}
	r, is910, _ := checkH03(m)
	if r.Status != types.StatusWarning {
		t.Errorf("expected warning for non-910, got %v", r.Status)
	}
	if is910 {
		t.Error("expected is910=false")
	}
}

func TestCheckH03_NpuSmiErr(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {err: errors.New("not found")},
		},
	}
	r, is910, _ := checkH03(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
	if is910 {
		t.Error("expected is910=false")
	}
}

func TestCheckH03_NoModel(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"npu-smi info": {stdout: "no model here"},
		},
	}
	r, is910, _ := checkH03(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
	if is910 {
		t.Error("expected is910=false")
	}
}

// ---- checkH04 ----

func TestCheckH04_Normal(t *testing.T) {
	r := checkH04(npuSmiOutput)
	if r.Status != types.StatusInfo {
		t.Errorf("expected info, got %v", r.Status)
	}
}

func TestCheckH04_Empty(t *testing.T) {
	r := checkH04("")
	if r.Status != types.StatusInfo {
		t.Errorf("expected info, got %v", r.Status)
	}
}

func TestCheckH04_NoAvailable(t *testing.T) {
	// All NPUs occupied
	allOccupied := `| 0     910B4               | OK      | 67.3      | 43       | 0      / 0              | 0  |
| 0     0      | 12345     | python              | 4096                                    |
`
	r := checkH04(allOccupied)
	if r.Status != types.StatusWarning {
		t.Errorf("expected warning when no NPU available, got %v", r.Status)
	}
}

// ---- checkH05 ----

func TestCheckH05_Pass(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"/usr/local/dcmi":                       {stdout: "ok"},
			"/usr/local/Ascend/driver/lib64":        {stdout: "ok"},
			"/usr/local/bin/npu-smi":                {stdout: "ok"},
			"/usr/local/Ascend/driver/version.info": {stdout: "ok"},
			"/etc/ascend_install.info":              {stdout: "ok"},
			"/etc/hccn.conf":                        {stdout: "ok"},
		},
	}
	r := checkH05(m)
	if r.Status != types.StatusPassed {
		t.Errorf("expected passed, got %v: %s", r.Status, r.Message)
	}
}

func TestCheckH05_MissingFile(t *testing.T) {
	m := &mockRunner{
		nodeName: "node1",
		responses: map[string]mockResponse{
			"/etc/hccn.conf": {err: errors.New("no such file")},
		},
	}
	r := checkH05(m)
	if r.Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", r.Status)
	}
}

// ---- CheckHardware (integration with mock SSHRunner) ----

func TestCheckHardware_SSHFail(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		return nil, func() {}, errors.New("ssh connection refused")
	}

	nodes := []types.NodeInfo{{Name: "node1", IP: "1.2.3.4", Port: 22}}
	result := CheckHardware(nodes, true)
	if len(result.Nodes) != 1 {
		t.Fatalf("expected 1 node result, got %d", len(result.Nodes))
	}
	if result.Nodes[0].Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", result.Nodes[0].Status)
	}
}

func TestCheckHardware_H01Fail(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		m := &mockRunner{
			nodeName: node.Name,
			responses: map[string]mockResponse{
				"lsmod | grep drv_pcie_host": {stdout: ""},
			},
		}
		return m, func() {}, nil
	}

	nodes := []types.NodeInfo{{Name: "node1"}}
	result := CheckHardware(nodes, true)
	if result.Nodes[0].Status != types.StatusFailed {
		t.Errorf("expected failed, got %v", result.Nodes[0].Status)
	}
}

func TestCheckHardware_NonSeries_SkipsHCCS(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	nonSeriesNpuSmi := `| 0     310P3               | OK      | 67.3      | 43       | 0      / 0              | 0  |`

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		m := &mockRunner{
			nodeName: node.Name,
			responses: map[string]mockResponse{
				"lsmod | grep drv_pcie_host": {stdout: "drv_pcie_host"},
				"npu-smi info":               {stdout: nonSeriesNpuSmi},
				"kubectl get pods -A -o json": {
					stdout: devicePluginPodListJSON(node.Name, "Running"),
				},
				fmt.Sprintf("kubectl get node %s -o jsonpath='{.status.capacity}'", node.Name): {
					stdout: `{"huawei.com/Ascend310P3":"4"}`,
				},
				"ls /usr/local/dcmi":                       {stdout: "ok"},
				"ls /usr/local/Ascend/driver/lib64":        {stdout: "ok"},
				"ls /usr/local/bin/npu-smi":                {stdout: "ok"},
				"ls /usr/local/Ascend/driver/version.info": {stdout: "ok"},
				"ls /etc/ascend_install.info":              {stdout: "ok"},
				"ls /etc/hccn.conf":                        {stdout: "ok"},
			},
		}
		return m, func() {}, nil
	}

	nodes := []types.NodeInfo{{Name: "node1"}}
	result := CheckHardware(nodes, true)
	if len(result.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(result.Nodes))
	}
	if !result.Nodes[0].SkippedHCCS {
		t.Error("expected HCCS skipped for non-910 series")
	}
}

func TestCheckHardware_ZeroNodes(t *testing.T) {
	result := CheckHardware([]types.NodeInfo{}, true)
	if result == nil {
		t.Error("expected non-nil result")
	}
	// H-09 returns info (metrics summary), H-10 returns skipped when no nodes are available
	if len(result.CrossNodes) < 2 {
		t.Errorf("expected at least 2 cross-node results (H-09 + H-10), got %d", len(result.CrossNodes))
	}
	for _, r := range result.CrossNodes {
		if r.Status != types.StatusSkipped && r.Status != types.StatusInfo {
			t.Errorf("expected skipped or info cross-node result, got %v: %s", r.Status, r.Message)
		}
	}
}

// ---- CheckHardware full pass integration ----

func TestCheckHardware_FullPass(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		m := &mockRunner{
			nodeName: node.Name,
			responses: map[string]mockResponse{
				"lsmod | grep drv_pcie_host": {stdout: "drv_pcie_host 12345 0"},
				"npu-smi info":               {stdout: npuSmiOutput},
				"kubectl get pods -A -o json": {
					stdout: devicePluginPodListJSON(node.Name, "Running"),
				},
				fmt.Sprintf("kubectl get node %s -o jsonpath='{.status.capacity}'", node.Name): {
					stdout: `{"huawei.com/Ascend910B4":"2"}`,
				},
				"ls /usr/local/dcmi":                       {stdout: "ok"},
				"ls /usr/local/Ascend/driver/lib64":        {stdout: "ok"},
				"ls /usr/local/bin/npu-smi":                {stdout: "ok"},
				"ls /usr/local/Ascend/driver/version.info": {stdout: "ok"},
				"ls /etc/ascend_install.info":              {stdout: "ok"},
				"ls /etc/hccn.conf":                        {stdout: "ok"},
				"ls /dev/davinci*":                         {stdout: "/dev/davinci0 /dev/davinci1"},
				"cat /etc/hccn.conf":                       {stdout: "address_0 = 192.168.1.1\nnetmask_0 = 255.255.255.0\naddress_1 = 192.168.1.2\nnetmask_1 = 255.255.255.0\n"},
				"hccn_tool -i 0 -tls -g":                   {stdout: "tls switch[1]"},
				"hccn_tool -i 1 -tls -g":                   {stdout: "tls switch[1]"},
				"hccn_tool":                                {stdout: "0.00% packet loss"},
			},
		}
		return m, func() {}, nil
	}

	nodes := []types.NodeInfo{{Name: "node1"}}
	result := CheckHardware(nodes, true)
	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Nodes) != 1 {
		t.Fatalf("expected 1 node, got %d", len(result.Nodes))
	}
}

// TestCheckHardware_ConnectivityDisabled tests that connectivity checks are skipped when disabled
func TestCheckHardware_ConnectivityDisabled(t *testing.T) {
	origFactory := newSSHRunner
	defer func() { newSSHRunner = origFactory }()

	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		// Create npu-smi output with only 1 NPU to match /dev/davinci0
		npuSmiSingleOutput := `+-------------------------------------------------------------------------------------------+
| npu-smi 23.0.rc2                 Version: 23.0.rc2                                       |
+-------------------------------------------------------------------------------------------+
| NPU   Name                  Health   Power(W)   Temp(C)   Hugepages-Usage(page)   AICore |
| Chip                         Bus-Id        AIServer(MHz)                                  |
+==========================================================================================+
| 0     910B4               | OK      | 67.3      | 43       | 0      / 0              | 0  |
| 0     0x0000:C1:00.0       | 1800                                                        |
+-------------------------------------------------------------------------------------------+
+-------------------------------------------------------------------------------------------+
| NPU     Chip   Process id   Process name         Device memory(MB)                        |
+==========================================================================================+
+-------------------------------------------------------------------------------------------+
`
		m := &mockRunner{
			nodeName: node.Name,
			responses: map[string]mockResponse{
				"lsmod | grep drv_pcie_host":               {stdout: "drv_pcie_host"},
				"npu-smi info":                             {stdout: npuSmiSingleOutput},
				"kubectl get pods -A -o wide":              {stdout: "kube-system   ascend-device-plugin-abc   1/1   Running   0   1d   192.168.1.1   test-node"},
				"kubectl get node test-node -o jsonpath":   {stdout: "'huawei.com/Ascend910: 8'"},
				"ls /usr/local/dcmi":                       {stdout: "ok"},
				"ls /usr/local/Ascend/driver/lib64":        {stdout: "ok"},
				"ls /usr/local/bin/npu-smi":                {stdout: "ok"},
				"ls /usr/local/Ascend/driver/version.info": {stdout: "ok"},
				"ls /etc/ascend_install.info":              {stdout: "ok"},
				"ls /etc/hccn.conf":                        {stdout: "ok"},
				"cat /etc/hccn.conf":                       {stdout: "address_0=192.168.1.1\nnetmask_0=255.255.255.0\n"},
				"ls /dev/davinci*":                         {stdout: "/dev/davinci0"},
				"hccn_tool -i 0 -tls -g":                   {stdout: "tls switch[1]"},
			},
		}
		return m, func() {}, nil
	}

	nodes := []types.NodeInfo{
		{Name: "test-node", IP: "192.0.2.1", Port: 22, User: "root", Password: "pass"},
	}
	result := CheckHardware(nodes, false)

	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.Nodes) != 1 {
		t.Fatalf("expected 1 node result, got %d", len(result.Nodes))
	}

	// Check that H-07 Step 2 is skipped
	foundH07Skip := false
	for _, check := range result.Nodes[0].Checks {
		t.Logf("Check: ID=%s, Status=%s, Message=%s", check.ID, check.Status, check.Message)
		if check.ID == "H-07" && check.Status == types.StatusSkipped {
			foundH07Skip = true
			if !strings.Contains(check.Message, "skipped") {
				t.Errorf("expected skip message in H-07, got: %s", check.Message)
			}
		}
	}
	if !foundH07Skip {
		t.Error("expected H-07 Step 2 to be skipped when connectivity check is disabled")
	}
}
