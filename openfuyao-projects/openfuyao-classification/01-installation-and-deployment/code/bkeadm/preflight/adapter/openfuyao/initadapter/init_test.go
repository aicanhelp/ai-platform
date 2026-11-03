/******************************************************************
 * Copyright (c) 2026 Bocloud Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package initadapter

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/preflight/api"
)

func TestResolveBuildsPlan(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "deploy.bocloud.k8s",
		KubernetesPort: 36443, ConsolePort: 30010, ImageRepoPort: 40443,
		ChartRepoPort: 38080, YumRepoPort: 40080, Runtime: "containerd",
		InstallConsole: true, EnableNTP: true, NtpServer: "local",
		OnlineImage: "cr.openfuyao.cn/openfuyao/bke-online:latest",
	}
	plan, diags := Resolve(in)
	require.Empty(t, diags)
	assert.Equal(t, api.OperationInit, plan.Operation)
	assert.NotEmpty(t, plan.PlanHash)
	assert.True(t, strings.HasPrefix(plan.PlanHash.String(), "sha256:"))
	assert.Len(t, plan.Subjects, 1)
	assert.Equal(t, "bootstrap", plan.Subjects[0].ID)
	// Sources projected.
	assert.Len(t, plan.Sources, 1)
	assert.Equal(t, "online-image", plan.Sources[0].ID)
	assert.Equal(t, "/v2/openfuyao/bke-online/manifests/latest", plan.Sources[0].ObjectPath)
	// Values include conditional console + ntp ports.
	_, hasConsole := plan.Values["port.console"]
	assert.True(t, hasConsole)
	_, hasNTP := plan.Values["port.ntp"]
	assert.True(t, hasNTP)
	assert.Equal(t, true, plan.Values["feature.localNTP"].Value)
	assert.NotContains(t, plan.Values, "feature.enableNTP")
}

func TestResolveConsoleDisabledSkipsPort(t *testing.T) {
	in := &Input{
		HostIP: "1.2.3.4", Domain: "x", InstallConsole: false, EnableNTP: false,
	}
	plan, _ := Resolve(in)
	_, hasConsole := plan.Values["port.console"]
	assert.False(t, hasConsole)
	_, hasNTP := plan.Values["port.ntp"]
	assert.False(t, hasNTP)
}

func TestResolveNilInput(t *testing.T) {
	plan, diags := Resolve(nil)
	assert.Empty(t, plan.Subjects)
	require.Len(t, diags, 1)
	assert.Equal(t, api.IssueError, diags[0].Severity)
}

func TestResolveDeterministicPlanHash(t *testing.T) {
	in := &Input{HostIP: "1.2.3.4", Domain: "x", InstallConsole: true}
	p1, _ := Resolve(in)
	p2, _ := Resolve(in)
	assert.Equal(t, p1.PlanHash, p2.PlanHash)
}

func TestInstancesIncludesConditionalPorts(t *testing.T) {
	in := &Input{HostIP: "1.2.3.4", Domain: "x", InstallConsole: true, EnableNTP: true, NtpServer: "local"}
	plan, _ := Resolve(in)
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
		ids[i.InstanceID] = true
	}
	// Fixed ports always present.
	for _, id := range []string{"INIT-PORT-K8S", "INIT-PORT-REGISTRY", "INIT-PORT-CHART", "INIT-PORT-HTTP"} {
		assert.True(t, ids[id], "missing %s", id)
	}
	// Conditional console present when enabled.
	assert.True(t, ids["INIT-PORT-CONSOLE"])
	assert.True(t, ids["INIT-PORT-NTP"])
	// Platform + colocate.
	assert.True(t, ids["INIT-PLAT-OS"])
	assert.True(t, ids["INIT-COLOCATE"])
}

func TestSystemDiskThresholdUses100GiB(t *testing.T) {
	plan, diags := Resolve(&Input{HostIP: "1.2.3.4", Domain: "x"})
	require.Empty(t, diags)
	for _, instance := range Instances(plan) {
		if instance.Rule.ID == "INIT-RES-SYSTEM-DISK" {
			assert.Equal(t, int64(100*1024*1024*1024), instance.Rule.Expected)
			return
		}
	}
	t.Fatal("system disk check missing")
}

func TestRootExecutionCheckIsNamedAndActionable(t *testing.T) {
	plan, diags := Resolve(&Input{HostIP: "192.0.2.10", Domain: "x"})
	require.Empty(t, diags)
	for _, instance := range Instances(plan) {
		if instance.Rule.ID != "INIT-EXEC-001" {
			continue
		}
		assert.Equal(t, "Root user required", instance.Rule.Name)
		assert.Contains(t, instance.Rule.Fix, "run bke as root")
		return
	}
	t.Fatal("root execution check missing")
}

func TestEveryInitInstanceHasActionableMetadata(t *testing.T) {
	plan, diags := Resolve(&Input{
		HostIP: "192.0.2.10", Domain: "x", InstallConsole: true, EnableNTP: true, NtpServer: "local",
		OnlineImage: "registry.example.com/openfuyao/bke-online:latest", ImageFilePath: "/tmp/offline.tar.gz",
	})
	require.Empty(t, diags)
	for _, instance := range Instances(plan) {
		assert.NotEmpty(t, instance.Rule.Name, instance.Rule.ID)
		assert.NotEmpty(t, instance.Rule.Fix, instance.Rule.ID)
		assert.NotContains(t, strings.ToLower(instance.Rule.Fix), "see the installation manual", instance.Rule.ID)
	}
}

func TestSourceInstancesHaveReadableUniqueNames(t *testing.T) {
	plan, diags := Resolve(&Input{
		HostIP: "192.0.2.10", Domain: "x",
		OnlineImage: "registry.example.com/openfuyao/bke-online:latest",
		OtherRepo:   "https://images.example.com:5443",
	})
	require.Empty(t, diags)
	seen := map[string]bool{}
	for _, instance := range Instances(plan) {
		if !strings.HasPrefix(instance.Rule.ID, "INIT-SRC-") {
			continue
		}
		assert.NotEmpty(t, instance.Rule.Name)
		assert.NotContains(t, instance.Rule.Name, "Src ")
		assert.Contains(t, instance.Rule.Name, "(")
		assert.NotContains(t, instance.Rule.Fix, "node resources")
		assert.False(t, seen[instance.Rule.Name], "duplicate source check name %q", instance.Rule.Name)
		seen[instance.Rule.Name] = true
	}
	assert.Contains(t, seen, "Online image source is accessible (registry.example.com:443)")
	assert.Contains(t, seen, "Image repository is accessible (images.example.com:5443)")
}

func TestInstancesUseResolvedCustomPorts(t *testing.T) {
	plan, diags := Resolve(&Input{
		HostIP: "192.0.2.10", KubernetesPort: 36444, ImageRepoPort: 40444,
		ChartRepoPort: 38081, YumRepoPort: 40081, ConsolePort: 30011,
		InstallConsole: true,
	})
	require.False(t, api.HasStartBlocker(diags))
	facts := map[string]bool{}
	for _, instance := range Instances(plan) {
		facts[instance.FactKey] = true
	}
	for _, expected := range []string{
		"port.36444.free", "port.40444.free", "port.38081.free",
		"port.40081.free", "port.30011.free",
	} {
		assert.True(t, facts[expected], "missing dynamic fact %s", expected)
	}
	for _, old := range []string{"port.36443.free", "port.40443.free", "port.38080.free", "port.40080.free", "port.30010.free"} {
		assert.False(t, facts[old], "default port must not be checked when overridden: %s", old)
	}
}

func TestInstancesExcludesConditionalWhenDisabled(t *testing.T) {
	in := &Input{HostIP: "1.2.3.4", Domain: "x", InstallConsole: false, EnableNTP: false}
	plan, _ := Resolve(in)
	insts := Instances(plan)
	applicable := make(map[string]bool)
	for _, i := range insts {
		applicable[i.Rule.ID] = i.Applicable
	}
	assert.False(t, applicable["INIT-PORT-CONSOLE"])
	assert.False(t, applicable["INIT-PORT-NTP"])
}

func TestExternalNTPDoesNotInstantiateLocalPort(t *testing.T) {
	plan, _ := Resolve(&Input{HostIP: "1.2.3.4", EnableNTP: true, NtpServer: "time.example.com"})
	assert.Equal(t, false, plan.Values["feature.localNTP"].Value)
	assert.NotContains(t, plan.Values, "port.ntp")
	assert.False(t, instanceApplicable(Instances(plan), "INIT-PORT-NTP"))
}

func TestFileChecksUseRootCauseDependencies(t *testing.T) {
	plan, issues := Resolve(&Input{HostIP: "1.2.3.4", ImageFilePath: "/tmp/offline.tar.gz", ImageRepoCAFile: "/tmp/ca.crt"})
	require.Empty(t, issues)
	dependencies := map[string][]string{}
	for _, instance := range Instances(plan) {
		dependencies[instance.Rule.ID] = instance.Rule.DependsOn
	}
	assert.Equal(t, []string{"INIT-OFFLINE-EXISTS"}, dependencies["INIT-OFFLINE-READ"])
	assert.Equal(t, []string{"INIT-OFFLINE-READ"}, dependencies["INIT-OFFLINE-FORMAT"])
	assert.Equal(t, []string{"INIT-REPO-CA-EXISTS"}, dependencies["INIT-REPO-CA-READ"])
	assert.Equal(t, []string{"INIT-REPO-CA-READ"}, dependencies["INIT-REPO-CA-PARSE"])
}

// TestInstancesCoreDomains freezes the intentionally small init check set.
// This is the "every check item has UT coverage" guarantee.
func TestInstancesAll26Domains(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "deploy.bocloud.k8s",
		KubernetesPort: 36443, ConsolePort: 30010, ImageRepoPort: 40443,
		ChartRepoPort: 38080, YumRepoPort: 40080, Runtime: "containerd",
		InstallConsole: true, EnableNTP: true, NtpServer: "local",
		OnlineImage: "cr.openfuyao.cn/openfuyao/bke-online:latest",
	}
	plan, _ := Resolve(in)
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
		ids[i.InstanceID] = true
	}

	// The 26 logical check items from the design, mapped to representative
	// RuleIDs. Each must be present.
	required := []string{
		// 1. Input params (validated by Resolve/Validate, not an instance)
		// 2. Exec identity
		"INIT-EXEC-001",
		// 3. OS type
		"INIT-PLAT-OS",
		// 4. Distribution
		"INIT-PLAT-DISTRO",
		// 5. Kernel
		"INIT-PLAT-KERNEL",
		// 6. Architecture
		"INIT-PLAT-ARCH",
		// 7. CPU
		"INIT-RES-CPU",
		// 8. Memory
		"INIT-RES-MEMORY",
		// 9. System disk
		"INIT-RES-SYSTEM-DISK",
		// 10. Required tools used by the containerd init path
		"INIT-TOOL-TAR", "INIT-TOOL-SYSTEMCTL",
		// 11. Workspace path (only creation and mount mode)
		"INIT-PATH-CREATE", "INIT-PATH-MOUNT",
		// 12-15. Fixed ports
		"INIT-PORT-K8S", "INIT-PORT-REGISTRY", "INIT-PORT-CHART", "INIT-PORT-HTTP",
		// 18. Conditional console port
		"INIT-PORT-CONSOLE",
		// 19. Conditional NTP port
		"INIT-PORT-NTP",
		// 16. Sources (one public access check per source)
		"INIT-SRC-REACHABLE@online-image",
		// Offline artifacts and custom repository/CA are conditional.
		// Colocate
		"INIT-COLOCATE",
	}
	for _, id := range required {
		assert.True(t, ids[id], "missing required check: %s", id)
	}
}

// TestInstancesConditionalOffline verifies offline artifact checks appear only
// when the plan has an offline artifact (ImageFilePath set → offline mode).
func TestInstancesConditionalOffline(t *testing.T) {
	// Online mode (no ImageFilePath): offline rules should NOT appear.
	in := &Input{HostIP: "1.2.3.4", Domain: "x", OnlineImage: "cr.openfuyao.cn/openfuyao/bke:latest"}
	plan, _ := Resolve(in)
	insts := Instances(plan)
	for _, i := range insts {
		if i.Rule.ID == "INIT-OFFLINE-EXISTS" {
			assert.False(t, i.Applicable)
		}
	}

	// Offline mode (ImageFilePath set): offline rules SHOULD appear.
	in2 := &Input{HostIP: "1.2.3.4", Domain: "x", ImageFilePath: "/data/bke.tar.gz", OtherRepo: "registry.example.com"}
	plan2, _ := Resolve(in2)
	insts2 := Instances(plan2)
	ids := make(map[string]bool)
	for _, i := range insts2 {
		ids[i.Rule.ID] = true
	}
	assert.True(t, ids["INIT-OFFLINE-EXISTS"], "offline EXISTS should appear when ImageFilePath set")
	assert.True(t, ids["INIT-OFFLINE-READ"])
	assert.True(t, ids["INIT-OFFLINE-FORMAT"])
}

// TestInstallModeDetection freezes the online/offline/hybrid mode detection.
func TestInstallModeDetection(t *testing.T) {
	tests := []struct {
		name     string
		input    *Input
		wantMode string
	}{
		{
			name:     "online (OnlineImage only)",
			input:    &Input{HostIP: "1.2.3.4", Domain: "x", OnlineImage: "cr.openfuyao.cn/openfuyao/bke:latest"},
			wantMode: "online",
		},
		{
			name:     "offline (ImageFilePath set)",
			input:    &Input{HostIP: "1.2.3.4", Domain: "x", ImageFilePath: "/data/bke.tar.gz", OtherRepo: "registry.example.com"},
			wantMode: "offline",
		},
		{
			name:     "hybrid (OnlineImage + OtherRepo)",
			input:    &Input{HostIP: "1.2.3.4", Domain: "x", OnlineImage: "cr.openfuyao.cn/openfuyao/bke:latest", OtherRepo: "registry.example.com"},
			wantMode: "hybrid",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan, _ := Resolve(tt.input)
			mode, ok := plan.Values["init.mode"]
			require.True(t, ok, "init.mode must be projected")
			assert.Equal(t, tt.wantMode, mode.Value)
		})
	}
}

// TestOnlineModeSources verifies that online mode projects the OnlineImage source.
func TestOnlineModeSources(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "x",
		OnlineImage: "cr.openfuyao.cn/openfuyao/bke-online:latest",
	}
	plan, _ := Resolve(in)
	assert.Len(t, plan.Sources, 1)
	assert.Equal(t, "online-image", plan.Sources[0].ID)
	assert.Equal(t, "cr.openfuyao.cn:443", plan.Sources[0].Endpoint)

	// Sources get one public access rule.
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
	}
	for _, i := range insts {
		ids[i.InstanceID] = true
	}
	assert.True(t, ids["INIT-SRC-REACHABLE@online-image"], "online source should get an access check")
	for _, i := range insts {
		if i.InstanceID == "INIT-SRC-REACHABLE@online-image" {
			assert.Contains(t, i.FactKey, "endpoint.tcp.")
		}
	}
	// No offline rules.
	assert.False(t, instanceApplicable(insts, "INIT-OFFLINE-EXISTS"))
}

// TestOfflineModeSources verifies that offline mode does NOT project online
// sources (no OnlineImage → no online source access check).
func TestOfflineModeSources(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "x",
		ImageFilePath: "/data/bke.tar.gz",
		OtherRepo:     "registry.example.com",
	}
	plan, _ := Resolve(in)
	// OtherRepo is a source; OnlineImage is not (offline mode).
	assert.Len(t, plan.Sources, 1)
	assert.Equal(t, "other-repo", plan.Sources[0].ID)

	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
	}
	for _, i := range insts {
		ids[i.InstanceID] = true
	}
	// Other-repo source gets one source access check.
	assert.True(t, ids["INIT-SRC-REACHABLE@other-repo"])
	// Offline artifact checks appear.
	assert.True(t, ids["INIT-OFFLINE-EXISTS"])
	assert.True(t, ids["INIT-OFFLINE-READ"])
}

// TestHybridModeSources verifies that hybrid mode (OnlineImage + OtherRepo)
// projects BOTH sources and gets source checks for each.
func TestHybridModeSources(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "x",
		OnlineImage: "cr.openfuyao.cn/openfuyao/bke:latest",
		OtherRepo:   "registry.example.com",
	}
	plan, _ := Resolve(in)
	assert.Len(t, plan.Sources, 2)

	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
	}
	for _, i := range insts {
		ids[i.InstanceID] = true
	}
	// Both sources get one access check.
	assert.True(t, ids["INIT-SRC-REACHABLE@online-image"])
	assert.True(t, ids["INIT-SRC-REACHABLE@other-repo"])
	// No offline artifact rules (no ImageFilePath in hybrid).
	assert.False(t, instanceApplicable(insts, "INIT-OFFLINE-EXISTS"))
}

// TestInstancesConditionalCustomRepo verifies custom repo checks appear only
// when feature.customRepo is enabled.
func TestInstancesConditionalCustomRepo(t *testing.T) {
	// Enable custom repo by adding the feature flag to plan values.
	in := &Input{HostIP: "1.2.3.4", Domain: "x"}
	plan, _ := Resolve(in)
	plan.Values["feature.customRepo"] = api.Value{Key: "feature.customRepo", Type: api.ValueTypeBool, Value: true}
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
	}
	assert.True(t, ids["INIT-REPO-URL"])
	assert.False(t, instanceApplicable(insts, "INIT-REPO-CA-EXISTS"))
}

func instanceApplicable(instances []InitInstance, id string) bool {
	for _, instance := range instances {
		if instance.Rule.ID == id {
			return instance.Applicable
		}
	}
	return false
}

// TestInstancesConditionalCustomCA verifies repository CA checks appear only
// when the repository CA reference is enabled.
func TestInstancesConditionalCustomCA(t *testing.T) {
	in := &Input{HostIP: "1.2.3.4", Domain: "x"}
	plan, _ := Resolve(in)
	plan.Values["feature.repoCA"] = api.Value{Key: "feature.repoCA", Type: api.ValueTypeBool, Value: true}
	insts := Instances(plan)
	ids := make(map[string]bool)
	for _, i := range insts {
		ids[i.Rule.ID] = true
	}
	assert.True(t, ids["INIT-REPO-CA-EXISTS"])
	assert.True(t, ids["INIT-REPO-CA-READ"])
	assert.True(t, ids["INIT-REPO-CA-PARSE"])
}

// TestInstancesRuleCount freezes the minimum rule count: with all supported
// features enabled, the init plan must produce at least 30 atomic rules.
func TestInstancesRuleCount(t *testing.T) {
	in := &Input{
		HostIP: "192.0.2.10", Domain: "x", InstallConsole: true, EnableNTP: true, NtpServer: "local",
		OnlineImage: "cr.openfuyao.cn/openfuyao/bke-online:latest",
	}
	plan, _ := Resolve(in)
	insts := Instances(plan)
	assert.GreaterOrEqual(t, len(insts), 27, "init with all features should produce >=27 retained rules, got %d", len(insts))
}

func TestCatalogRulesAreValidAndUnique(t *testing.T) {
rules := CatalogRules()
require.NotEmpty(t, rules)
seen := map[string]bool{}
for _, rule := range rules {
assert.Nil(t, rule.Validate())
assert.False(t, seen[rule.ID], "duplicate rule %s", rule.ID)
seen[rule.ID] = true
assert.Equal(t, api.OperationInit, rule.Operation)
}
}
