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

package configenv

import (
	"fmt"
	"testing"

	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/executor"
	"gitcode.com/openFuyao/InferNex/component/infernex-checker/pkg/types"
)

// ---------------------------------------------------------------------------
// parseDriverVersion
// ---------------------------------------------------------------------------

func TestParseDriverVersion(t *testing.T) {
	tests := []struct {
		name    string
		content string
		want    string
	}{
		{
			name:    "normal version line",
			content: "Version=25.5.0.1\nother=foo\n",
			want:    "25.5.0.1",
		},
		{
			name:    "lowercase prefix",
			content: "version=25.3.1.0\n",
			want:    "25.3.1.0",
		},
		{
			name:    "mixed case prefix",
			content: "VERSION=24.1.0.0\n",
			want:    "24.1.0.0",
		},
		{
			name:    "leading whitespace on version line",
			content: "  version=25.0.RC1\n",
			want:    "25.0.RC1",
		},
		{
			name:    "version line not first",
			content: "chip=Ascend910B\nversion=25.2.0.0\nbuild=20250101\n",
			want:    "25.2.0.0",
		},
		{
			name:    "no version line returns empty string",
			content: "chip=Ascend910B\nbuild=20250101\n",
			want:    "",
		},
		{
			name:    "empty content returns empty string",
			content: "",
			want:    "",
		},
		{
			name: "real version.info format (3-segment version, multiple fields)",
			content: "Version=25.5.0\nascendhal_version=7.35.23\naicpu_version=1.0\n" +
				"Innerversion=V100R001C23SPC005B219\npackage_version=25.5.0\n",
			want: "25.5.0",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := parseDriverVersion(tc.content)
			if got != tc.want {
				t.Errorf("parseDriverVersion(%q) = %q, want %q", tc.content, got, tc.want)
			}
		})
	}
}

// ---------------------------------------------------------------------------
// checkDriverCompatibility
// ---------------------------------------------------------------------------

func TestCheckDriverCompatibility(t *testing.T) {
	const nodeName = "node-01"

	tests := []struct {
		name              string
		driverVersion     string
		imageTag          string
		wantStatus        types.CheckStatus
		wantSuggestionSet bool // true when a non-empty Suggestion is expected
	}{
		// image tag empty
		{
			name:              "empty imageTag fails",
			driverVersion:     "25.5.0.1",
			imageTag:          "",
			wantStatus:        types.StatusFailed,
			wantSuggestionSet: true,
		},
		// unknown image tag (warning level — user should verify against official matrix)
		{
			name:              "unknown imageTag returns warning",
			driverVersion:     "25.5.0.1",
			imageTag:          "v99.0.0",
			wantStatus:        types.StatusWarning,
			wantSuggestionSet: true,
		},
		// v0.13.0 compatible drivers
		{
			name:          "v0.13.0 compatible with 25.5.x",
			driverVersion: "25.5.0.1",
			imageTag:      "v0.13.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.13.0 compatible with 25.3.x",
			driverVersion: "25.3.1.0",
			imageTag:      "v0.13.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.13.0 compatible with 25.2.x",
			driverVersion: "25.2.0.0",
			imageTag:      "v0.13.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.13.0 compatible with 25.0.x",
			driverVersion: "25.0.0.SPC100",
			imageTag:      "v0.13.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:              "v0.13.0 incompatible with 24.1.x returns warning",
			driverVersion:     "24.1.0.0",
			imageTag:          "v0.13.0",
			wantStatus:        types.StatusWarning,
			wantSuggestionSet: true,
		},
		// v0.18.0 compatible drivers
		{
			name:          "v0.18.0 compatible with 26.0.RC1 uppercase",
			driverVersion: "26.0.RC1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 26.0.rc1 lowercase",
			driverVersion: "26.0.rc1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 26.0.Rc1 mixed case",
			driverVersion: "26.0.Rc1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 25.7.RC1 uppercase",
			driverVersion: "25.7.RC1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 25.7.rc1 lowercase",
			driverVersion: "25.7.rc1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 25.5.x",
			driverVersion: "25.5.0.1",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:          "v0.18.0 compatible with 25.3.x",
			driverVersion: "25.3.0.0",
			imageTag:      "v0.18.0",
			wantStatus:    types.StatusPassed,
		},
		{
			name:              "v0.18.0 incompatible with 24.1.x returns warning",
			driverVersion:     "24.1.0.0",
			imageTag:          "v0.18.0",
			wantStatus:        types.StatusWarning,
			wantSuggestionSet: true,
		},
		// driver version detail preserved in result
		{
			name:          "passed result carries driver_version detail",
			driverVersion: "25.5.0.1",
			imageTag:      "v0.13.0",
			wantStatus:    types.StatusPassed,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			result := checkDriverCompatibility(nodeName, tc.driverVersion, tc.imageTag)
			if result.Status != tc.wantStatus {
				t.Errorf("checkDriverCompatibility(%q, %q, %q): status = %q, want %q; message: %s",
					nodeName, tc.driverVersion, tc.imageTag, result.Status, tc.wantStatus, result.Message)
			}
			if result.Node != nodeName {
				t.Errorf("result.Node = %q, want %q", result.Node, nodeName)
			}
			if result.ID != "B-02" {
				t.Errorf("result.ID = %q, want B-02", result.ID)
			}
			if tc.wantSuggestionSet && result.Suggestion == "" {
				t.Errorf("expected non-empty Suggestion for case %q", tc.name)
			}
			// Detail["driver_version"] is always set when driverVersion is non-empty,
			// regardless of imageTag value (empty, unknown, or known).
			if tc.driverVersion != "" {
				if result.Detail == nil {
					t.Errorf("expected result.Detail to be non-nil")
				} else if v, ok := result.Detail["driver_version"]; !ok || v != tc.driverVersion {
					t.Errorf("result.Detail[driver_version] = %v, want %q", v, tc.driverVersion)
				}
			}
		})
	}
}

// ---------------------------------------------------------------------------
// fakeRunner — test double for executor.CommandRunner
// ---------------------------------------------------------------------------

type fakeRunner struct {
	responses map[string]struct {
		stdout string
		stderr string
		err    error
	}
}

func (f *fakeRunner) Run(cmd string) (string, string, error) {
	if r, ok := f.responses[cmd]; ok {
		return r.stdout, r.stderr, r.err
	}
	// default: command succeeds with empty output
	return "", "", nil
}

func (f *fakeRunner) NodeName() string { return "fake-node" }

func newFakeRunner(pairs ...interface{}) *fakeRunner {
	f := &fakeRunner{
		responses: make(map[string]struct {
			stdout string
			stderr string
			err    error
		}),
	}
	for i := 0; i+2 < len(pairs); i += 3 {
		cmd := pairs[i].(string)
		stdout := pairs[i+1].(string)
		var runErr error
		if pairs[i+2] != nil {
			runErr = pairs[i+2].(error)
		}
		f.responses[cmd] = struct {
			stdout string
			stderr string
			err    error
		}{stdout, "", runErr}
	}
	return f
}

// ---------------------------------------------------------------------------
// checkB01
// ---------------------------------------------------------------------------

func TestCheckB01_EmptyCachePath(t *testing.T) {
	runner := &fakeRunner{}
	result := checkB01(runner, "node-1", "")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed for empty cachePath, got %q", result.Status)
	}
	if result.ID != "B-01" {
		t.Errorf("expected ID B-01, got %q", result.ID)
	}
	if result.Suggestion == "" {
		t.Error("expected non-empty suggestion")
	}
}

func TestCheckB01_DirectoryNotExist(t *testing.T) {
	runner := newFakeRunner(
		"ls /model/cache", "", fmt.Errorf("ls: /model/cache: No such file or directory"),
	)
	result := checkB01(runner, "node-1", "/model/cache")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when directory missing, got %q", result.Status)
	}
	if result.ID != "B-01" {
		t.Errorf("expected ID B-01, got %q", result.ID)
	}
	if result.Node != "node-1" {
		t.Errorf("expected Node node-1, got %q", result.Node)
	}
}

func TestCheckB01_DirectoryExistsNotWritable(t *testing.T) {
	runner := newFakeRunner(
		"ls /model/cache", "", nil,
		"touch /model/cache/.infernex_write_test && rm /model/cache/.infernex_write_test", "",
		fmt.Errorf("permission denied"),
	)
	result := checkB01(runner, "node-1", "/model/cache")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed for non-writable directory, got %q", result.Status)
	}
}

func TestCheckB01_DirectoryExistsAndWritable(t *testing.T) {
	runner := newFakeRunner(
		"ls /model/cache", "", nil,
		"touch /model/cache/.infernex_write_test && rm /model/cache/.infernex_write_test", "", nil,
	)
	result := checkB01(runner, "node-1", "/model/cache")
	if result.Status != types.StatusPassed {
		t.Errorf("expected passed for writable directory, got %q: %s", result.Status, result.Message)
	}
	if result.Node != "node-1" {
		t.Errorf("expected Node node-1, got %q", result.Node)
	}
}

// ---------------------------------------------------------------------------
// checkB02 / readDriverVersion
// ---------------------------------------------------------------------------

func TestCheckB02_VersionInfoReadFails(t *testing.T) {
	runner := newFakeRunner(
		"cat /usr/local/Ascend/driver/version.info", "", fmt.Errorf("no such file"),
	)
	result := checkB02(runner, "node-1", "v0.13.0")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when version.info unreadable, got %q", result.Status)
	}
	if result.ID != "B-02" {
		t.Errorf("expected ID B-02, got %q", result.ID)
	}
}

func TestCheckB02_VersionInfoUnparseable(t *testing.T) {
	runner := newFakeRunner(
		"cat /usr/local/Ascend/driver/version.info", "chip=Ascend910B\nbuild=20250101\n", nil,
	)
	result := checkB02(runner, "node-1", "v0.13.0")
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when version cannot be parsed, got %q", result.Status)
	}
}

func TestCheckB02_CompatibleDriver(t *testing.T) {
	runner := newFakeRunner(
		"cat /usr/local/Ascend/driver/version.info", "Version=25.5.0.1\n", nil,
	)
	result := checkB02(runner, "node-1", "v0.13.0")
	if result.Status != types.StatusPassed {
		t.Errorf("expected passed for compatible driver, got %q: %s", result.Status, result.Message)
	}
}

func TestCheckB02_IncompatibleDriver(t *testing.T) {
	runner := newFakeRunner(
		"cat /usr/local/Ascend/driver/version.info", "Version=24.1.0.0\n", nil,
	)
	result := checkB02(runner, "node-1", "v0.13.0")
	if result.Status != types.StatusWarning {
		t.Errorf("expected warning for incompatible driver, got %q", result.Status)
	}
}

// ---------------------------------------------------------------------------
// checkSingleNode (via newSSHRunner injection)
// ---------------------------------------------------------------------------

// injectFakeRunner overrides newSSHRunner for the duration of a test,
// restoring the original when done.
func injectFakeRunner(runner *fakeRunner) func() {
	orig := newSSHRunner
	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		return runner, func() {}, nil
	}
	return func() { newSSHRunner = orig }
}

func TestCheckSingleNode_SSHFails(t *testing.T) {
	orig := newSSHRunner
	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		return nil, func() {}, fmt.Errorf("connection refused")
	}
	defer func() { newSSHRunner = orig }()

	results := checkSingleNode(types.NodeInfo{Name: "node-1"}, "/cache", "v0.13.0")
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if results[0].ID != "B-00" {
		t.Errorf("expected ID B-00 for SSH failure, got %q", results[0].ID)
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed status, got %q", results[0].Status)
	}
}

func TestCheckSingleNode_B01Fails_StopsEarly(t *testing.T) {
	// B-01 fails (empty cachePath) → B-02 should not run
	runner := &fakeRunner{responses: map[string]struct {
		stdout string
		stderr string
		err    error
	}{}}
	restore := injectFakeRunner(runner)
	defer restore()

	results := checkSingleNode(types.NodeInfo{Name: "node-1"}, "", "v0.13.0")
	if len(results) != 1 {
		t.Fatalf("expected 1 result (B-01 only), got %d", len(results))
	}
	if results[0].ID != "B-01" {
		t.Errorf("expected ID B-01, got %q", results[0].ID)
	}
	if results[0].Status != types.StatusFailed {
		t.Errorf("expected failed, got %q", results[0].Status)
	}
}

func TestCheckSingleNode_BothPass(t *testing.T) {
	runner := newFakeRunner(
		"ls /model/cache", "", nil,
		"touch /model/cache/.infernex_write_test && rm /model/cache/.infernex_write_test", "", nil,
		"cat /usr/local/Ascend/driver/version.info", "Version=25.5.0.1\n", nil,
	)
	restore := injectFakeRunner(runner)
	defer restore()

	results := checkSingleNode(types.NodeInfo{Name: "node-1"}, "/model/cache", "v0.13.0")
	if len(results) != 2 {
		t.Fatalf("expected 2 results (B-01 + B-02), got %d", len(results))
	}
	for _, r := range results {
		if r.Status != types.StatusPassed {
			t.Errorf("expected passed for %s, got %q: %s", r.ID, r.Status, r.Message)
		}
	}
}

// ---------------------------------------------------------------------------
// CheckConfigEnv
// ---------------------------------------------------------------------------

func TestCheckConfigEnv_NoNodes(t *testing.T) {
	result := CheckConfigEnv([]types.NodeInfo{}, map[string]interface{}{})
	if result.Status != types.StatusPassed {
		t.Errorf("expected passed for empty node list, got %q", result.Status)
	}
	if len(result.Checks) != 0 {
		t.Errorf("expected 0 checks, got %d", len(result.Checks))
	}
}

func TestCheckConfigEnv_SSHFails_OverallFailed(t *testing.T) {
	orig := newSSHRunner
	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		return nil, func() {}, fmt.Errorf("refused")
	}
	defer func() { newSSHRunner = orig }()

	nodes := []types.NodeInfo{{Name: "node-1"}}
	values := map[string]interface{}{
		"global": map[string]interface{}{"cachePath": "/cache"},
		"inference-backend": map[string]interface{}{
			"images": map[string]interface{}{
				"inferenceEngine": map[string]interface{}{"tag": "v0.13.0"},
			},
		},
	}
	result := CheckConfigEnv(nodes, values)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when SSH fails, got %q", result.Status)
	}
	if len(result.Checks) != 1 || result.Checks[0].ID != "B-00" {
		t.Errorf("expected 1 B-00 check, got %v", result.Checks)
	}
}

func TestCheckConfigEnv_AllPassed(t *testing.T) {
	runner := newFakeRunner(
		"ls /model/cache", "", nil,
		"touch /model/cache/.infernex_write_test && rm /model/cache/.infernex_write_test", "", nil,
		"cat /usr/local/Ascend/driver/version.info", "Version=25.5.0.1\n", nil,
	)
	restore := injectFakeRunner(runner)
	defer restore()

	nodes := []types.NodeInfo{{Name: "node-1"}}
	values := map[string]interface{}{
		"global": map[string]interface{}{"cachePath": "/model/cache"},
		"inference-backend": map[string]interface{}{
			"images": map[string]interface{}{
				"inferenceEngine": map[string]interface{}{"tag": "v0.13.0"},
			},
		},
	}
	result := CheckConfigEnv(nodes, values)
	if result.Status != types.StatusPassed {
		t.Errorf("expected passed, got %q", result.Status)
	}
	if len(result.Checks) != 2 {
		t.Errorf("expected 2 checks (B-01+B-02), got %d", len(result.Checks))
	}
}

func TestCheckConfigEnv_MultipleNodes_OneFails(t *testing.T) {
	callCount := 0
	orig := newSSHRunner
	newSSHRunner = func(node types.NodeInfo) (executor.CommandRunner, func(), error) {
		callCount++
		if node.Name == "node-2" {
			return nil, func() {}, fmt.Errorf("refused")
		}
		runner := newFakeRunner(
			"ls /model/cache", "", nil,
			"touch /model/cache/.infernex_write_test && rm /model/cache/.infernex_write_test", "", nil,
			"cat /usr/local/Ascend/driver/version.info", "Version=25.5.0.1\n", nil,
		)
		return runner, func() {}, nil
	}
	defer func() { newSSHRunner = orig }()

	nodes := []types.NodeInfo{{Name: "node-1"}, {Name: "node-2"}}
	values := map[string]interface{}{
		"global": map[string]interface{}{"cachePath": "/model/cache"},
		"inference-backend": map[string]interface{}{
			"images": map[string]interface{}{
				"inferenceEngine": map[string]interface{}{"tag": "v0.13.0"},
			},
		},
	}
	result := CheckConfigEnv(nodes, values)
	if result.Status != types.StatusFailed {
		t.Errorf("expected failed when one node fails, got %q", result.Status)
	}
}
