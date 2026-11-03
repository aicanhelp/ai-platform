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

package preflightbridge

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightbundlecompiler"
	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/adapter/openfuyao/initadapter"
	"gopkg.openfuyao.cn/preflight/api"
	"gopkg.openfuyao.cn/preflight/bundle"
	"gopkg.openfuyao.cn/preflight/normalize"
)

func TestDynamicSourceBindingHasReadableNameAndRemediation(t *testing.T) {
	result := resolveInit(&initadapter.Input{
		HostIP:      "192.0.2.1",
		OnlineImage: "registry.example.com/openfuyao/bke-online:latest",
		OtherRepo:   "https://images.example.com:5443",
	})
	var names []string
	var sourceFixes []string
	for _, binding := range result.Bindings {
		if !strings.HasPrefix(binding.TemplateID, "INIT-SRC-") {
			continue
		}
		names = append(names, binding.Name)
		if strings.HasSuffix(binding.TemplateID, "REACHABLE") {
			assert.Contains(t, binding.Name, "is accessible")
			assert.Contains(t, binding.Fact, "endpoint.tcp.")
			sourceFixes = append(sourceFixes, binding.Fix)
		}
	}
	assert.True(t, hasNamePrefix(names, "Online image source is accessible (registry.example.com:443)"))
	assert.True(t, hasNamePrefix(names, "Image repository is accessible (images.example.com:5443)"))
	assert.Contains(t, sourceFixes, "unable to access Online image source (registry.example.com:443); check network connectivity, the endpoint address, proxy/NO_PROXY, and firewall settings, then run preflight again")
	assert.Contains(t, sourceFixes, "unable to access Image repository (images.example.com:5443); check network connectivity, the endpoint address, proxy/NO_PROXY, and firewall settings, then run preflight again")
}

func hasNamePrefix(names []string, prefix string) bool {
	for _, name := range names {
		if strings.HasPrefix(name, prefix) {
			return true
		}
	}
	return false
}

func runtimePolicyConfig(initPolicy preflightconfig.OperationPolicy) preflightconfig.Config {
	return preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
		Operations: preflightconfig.Operations{Init: initPolicy},
	}
}

func TestBindingNamesIncludeConcreteTargets(t *testing.T) {
	result := resolveInit(&initadapter.Input{
		Workspace:      "/var/lib/openfuyao",
		RuntimeStorage: "/var/lib/containerd",
	})
	find := func(id string) string {
		for _, binding := range result.Bindings {
			if binding.TemplateID == id {
				return binding.Name
			}
		}
		return ""
	}
	assert.Contains(t, find("INIT-PATH-CREATE"), "/var/lib/openfuyao")
	assert.Empty(t, find("INIT-RUNTIME-DATAROOT"))
	assert.Empty(t, find("INIT-RUNTIME-SOCKET"))
}

func TestBuildBundleUsesExternalThresholds(t *testing.T) {
	c := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
		Operations: preflightconfig.Operations{
			Init: preflightconfig.OperationPolicy{Resources: []preflightconfig.ResourceCheck{{
				CheckCommon: preflightconfig.CheckCommon{Description: "cpu"}, Key: "cpu", Minimum: 7,
			}}},
			ClusterCreate: preflightconfig.OperationPolicy{},
		},
	}
	raw, err := BuildBundle(c)
	require.NoError(t, err)
	b, err := bundle.Load(raw)
	require.NoError(t, err)
	var found bool
	for _, rule := range b.Rules {
		if rule.ID == "INIT-RES-CPU" {
			found = true
			require.Equal(t, 7, rule.Expected)
		}
	}
	require.True(t, found)
}

func TestBuildBundleAppliesRuntimeThresholdsAndTopology(t *testing.T) {
	runtimeConfig := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
	}
	runtimeConfig.Operations.Init.Resources = []preflightconfig.ResourceCheck{{
		CheckCommon: preflightconfig.CheckCommon{Description: "custom cpu minimum"},
		Key:         "cpu",
		Minimum:     11,
	}}
	runtimeConfig.Operations.ClusterCreate.Topology.Masters.Minimum = 3

	runtimeRaw, err := BuildBundle(runtimeConfig)
	require.NoError(t, err)
	runtimeBundle, err := bundle.Load(runtimeRaw)
	require.NoError(t, err)
	foundCPU, foundMasters := false, false
	for _, rule := range runtimeBundle.Rules {
		if rule.ID == "INIT-RES-CPU" {
			foundCPU = true
			assert.Equal(t, 11, rule.Expected)
			assert.Equal(t, "custom cpu minimum", rule.Name)
		}
		if rule.ID == "CLUSTER-TOPO-MASTER" {
			foundMasters = true
			assert.Equal(t, 3, rule.Expected)
		}
	}
	assert.True(t, foundCPU, "INIT-RES-CPU is missing from runtime Bundle")
	assert.True(t, foundMasters, "CLUSTER-TOPO-MASTER is missing from runtime Bundle")
}

func TestBuildBundleAddsCustomCELRuleAndSource(t *testing.T) {
	runtimeConfig := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	runtimeConfig.Operations.Init.CustomChecks = []preflightconfig.CustomCheck{{
		CheckCommon: preflightconfig.CheckCommon{
			Description: "hosts loopback", Remediation: "add a loopback mapping",
		},
		Key: "hosts-loopback", Expected: "contains a loopback mapping",
		Collector:  preflightconfig.CustomCollector{Type: "fileContent", Path: "/etc/hosts"},
		Expression: `actual.contains("127.0.0.1") || actual.contains("::1")`,
	}}
	raw, err := BuildBundle(runtimeConfig)
	require.NoError(t, err)
	compiled, err := bundle.Load(raw)
	require.NoError(t, err)
	rule := ruleByID(t, compiled.Rules, "CONFIG-INIT-CUSTOM-HOSTS-LOOPBACK")
	assert.Equal(t, api.OpCEL, rule.Op)
	assert.Equal(t, "contains a loopback mapping", rule.ExpectedText)
}

func TestAppendCustomBindingsScopesOperationAndRoles(t *testing.T) {
	result := api.ResolveResult{Plan: api.OperationPlan{
		Operation: api.OperationClusterCreate,
		Subjects: []api.Subject{
			{ID: "cluster", Role: []string{"cluster"}},
			{ID: "master-1", Role: []string{"master"}},
			{ID: "worker-1", Role: []string{"worker"}},
		},
		Values: map[string]api.Value{"feature.custom": {Type: api.ValueTypeBool, Value: true}},
	}}
	policy := preflightconfig.OperationPolicy{CustomChecks: []preflightconfig.CustomCheck{{
		CheckCommon: preflightconfig.CheckCommon{
			Roles: []string{"master"}, When: "feature.custom", Description: "health", Remediation: "fix",
		},
		Key: "health",
	}}}
	appendCustomBindings(&result, policy)
	require.Len(t, result.Bindings, 1)
	assert.Equal(t, "master-1", result.Bindings[0].Subject.ID)
	assert.Equal(t, "custom.cluster-create.health", result.Bindings[0].Fact)

	result.Bindings = nil
	result.Plan.Values["feature.custom"] = api.Value{Type: api.ValueTypeBool, Value: false}
	appendCustomBindings(&result, policy)
	assert.Empty(t, result.Bindings)
}

func TestClusterCustomRuleKeepsSSHRootDependency(t *testing.T) {
	config := preflightconfig.Config{}
	config.Operations.ClusterCreate.CustomChecks = []preflightconfig.CustomCheck{{
		CheckCommon: preflightconfig.CheckCommon{Description: "health", Remediation: "fix"},
		Key:         "health", Expected: "healthy", Expression: "actual == true",
		Collector: preflightconfig.CustomCollector{Type: "exec", Path: "/opt/check", ResultType: "bool"},
	}}
	rules := preflightbundlecompiler.CustomRules(config)
	require.Len(t, rules, 1)
	assert.Equal(t, []string{"CLUSTER-SSH-PRIVILEGE"}, rules[0].DependsOn)
}

func ruleByID(t *testing.T, rules []api.RuleTemplate, id string) api.RuleTemplate {
	t.Helper()
	for _, rule := range rules {
		if rule.ID == id {
			return rule
		}
	}
	t.Fatalf("rule %s not found", id)
	return api.RuleTemplate{}
}

func TestConfigPortConditionProducesSkipBinding(t *testing.T) {
	c := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
		Operations: preflightconfig.Operations{Init: preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
			CheckCommon: preflightconfig.CheckCommon{Description: "console", When: "feature.installConsole"}, Key: "console", DefaultPort: 30010,
			PortFrom: "port.console",
		}}}},
	}
	result := resolveInitWithPolicy(&initadapter.Input{HostIP: "192.0.2.1", InstallConsole: false}, &c)
	var console *api.RuleBinding
	for i := range result.Bindings {
		if result.Bindings[i].TemplateID == "INIT-PORT-CONSOLE" {
			console = &result.Bindings[i]
			break
		}
	}
	require.NotNil(t, console)
	require.False(t, console.Applicable)
}

func TestRuntimePolicyChangesInitNTPPortWithoutInvalidatingPlanHash(t *testing.T) {
	config := runtimePolicyConfig(preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
		CheckCommon: preflightconfig.CheckCommon{Description: "custom ntp"},
		Key:         "ntp",
		DefaultPort: 124,
		Protocol:    "udp",
	}}})
	result := resolveInitWithPolicy(
		&initadapter.Input{HostIP: "192.0.2.1", EnableNTP: true, NtpServer: "local"},
		&config,
	)
	value := result.Plan.Values["port.ntp"]
	assert.Equal(t, 124, value.Value)
	recomputed, err := normalize.ComputePlanHash(&result.Plan)
	require.NoError(t, err)
	assert.Equal(t, recomputed, result.Plan.PlanHash)
	for _, binding := range result.Bindings {
		if binding.TemplateID == "INIT-PORT-NTP" {
			assert.Equal(t, "port.124.udp.free", binding.Fact)
			return
		}
	}
	t.Fatal("INIT-PORT-NTP binding is missing")
}

func TestRuntimePolicyPortDefaultYieldsToExplicitInitPort(t *testing.T) {
	config := runtimePolicyConfig(preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
		CheckCommon: preflightconfig.CheckCommon{Description: "custom api port"},
		Key:         "kubernetes",
		DefaultPort: 45678,
		PortFrom:    "port.kubernetes",
	}}})
	defaulted := resolveInitWithPolicy(
		&initadapter.Input{HostIP: "192.0.2.1", KubernetesPort: 36443},
		&config,
	)
	assert.Equal(t, 45678, defaulted.Plan.Values["port.kubernetes"].Value)

	explicit := resolveInitWithPolicy(
		&initadapter.Input{
			HostIP: "192.0.2.1", KubernetesPort: 36443,
			ExplicitPorts: map[string]bool{"kubernetes": true},
		},
		&config,
	)
	assert.Equal(t, 36443, explicit.Plan.Values["port.kubernetes"].Value)
}

func TestRuntimePolicyControlsDeclarativeApplicability(t *testing.T) {
	result := api.ResolveResult{
		Plan: api.OperationPlan{Operation: api.OperationClusterCreate},
		Bindings: []api.RuleBinding{
			{TemplateID: "CLUSTER-TOPO-MASTER-ODD", Applicable: true},
			{TemplateID: "CLUSTER-UNIQUE-IP", Applicable: true},
			{TemplateID: "CLUSTER-UNIQUE-HOSTNAME", Applicable: true},
			{TemplateID: "CLUSTER-MGMT-DISCOVERY", Applicable: true},
		},
	}
	policy := preflightconfig.OperationPolicy{
		Topology:          preflightconfig.TopologyPolicy{Unique: []string{"ip"}},
		ManagementCluster: preflightconfig.ManagementClusterPolicy{API: true, Discovery: false},
	}
	applyPolicyBindings(&result, policy)
	assert.False(t, result.Bindings[0].Applicable, "odd=false must not instantiate the odd-count rule")
	assert.True(t, result.Bindings[1].Applicable)
	assert.False(t, result.Bindings[2].Applicable)
	assert.False(t, result.Bindings[3].Applicable)
}

func TestRuntimePolicyPresentationReachesBindings(t *testing.T) {
	config := runtimePolicyConfig(preflightconfig.OperationPolicy{Resources: []preflightconfig.ResourceCheck{{
		CheckCommon: preflightconfig.CheckCommon{
			Description: "custom cpu check",
			Remediation: "allocate more custom CPU",
		},
		Key: "cpu", Minimum: 5,
	}}})
	result := resolveInitWithPolicy(&initadapter.Input{HostIP: "192.0.2.1"}, &config)
	for _, binding := range result.Bindings {
		if binding.TemplateID == "INIT-RES-CPU" {
			assert.Contains(t, binding.Name, "custom cpu check")
			assert.Equal(t, "allocate more custom CPU", binding.Fix)
			return
		}
	}
	t.Fatal("INIT-RES-CPU binding is missing")
}

func TestRuntimePolicyFailureSkipAndToolNameTakeEffect(t *testing.T) {
	runtimeConfig := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
	}
	runtimeConfig.Operations.Init.Tools = []preflightconfig.ToolCheck{{
		CheckCommon: preflightconfig.CheckCommon{Description: "bsdtar executable", Failure: "skip"},
		Key:         "tar",
		Names:       []string{"bsdtar"},
	}}
	runtimeRaw, err := BuildBundle(runtimeConfig)
	require.NoError(t, err)
	runtimeBundle, err := bundle.Load(runtimeRaw)
	require.NoError(t, err)
	foundRule := false
	for _, rule := range runtimeBundle.Rules {
		if rule.ID == "INIT-TOOL-TAR" {
			foundRule = true
			assert.Equal(t, api.SeverityInfo, rule.Level)
			assert.Equal(t, "bsdtar executable", rule.Name)
			break
		}
	}
	require.True(t, foundRule, "INIT-TOOL-TAR is missing from runtime Bundle")

	resolved := resolveInitWithPolicy(&initadapter.Input{HostIP: "192.0.2.1"}, &runtimeConfig)
	for _, binding := range resolved.Bindings {
		if binding.TemplateID == "INIT-TOOL-TAR" {
			assert.Equal(t, "tool.bsdtar", binding.Fact)
			return
		}
	}
	t.Fatal("INIT-TOOL-TAR binding is missing")
}

func TestConfigCannotInventPortWithoutReviewedRuleSource(t *testing.T) {
	c := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion,
		Kind:       preflightconfig.Kind,
		Operations: preflightconfig.Operations{Init: preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
			CheckCommon: preflightconfig.CheckCommon{Description: "new service"}, Key: "newService", DefaultPort: 4567,
		}}}},
	}
	require.Error(t, preflightconfig.Validate(c))
	_, err := BuildBundle(c)
	require.ErrorContains(t, err, "has no reviewed rule/source")
	result := resolveInitWithPolicy(&initadapter.Input{HostIP: "192.0.2.1"}, &c)
	id := preflightconfig.CheckID(api.OperationInit, "port", "newService")
	for _, binding := range result.Bindings {
		require.NotEqual(t, id, binding.TemplateID)
	}
}

func TestAllRolePolicyKeepsBKEAgentOnEtcdOnlyNode(t *testing.T) {
	result := api.ResolveResult{
		Plan: api.OperationPlan{Operation: api.OperationClusterCreate},
		Bindings: []api.RuleBinding{{
			TemplateID: "CLUSTER-PORT-BKEAGENT", Applicable: true,
			Subject: api.Subject{ID: "etcd-1", Role: []string{"etcd"}},
		}},
	}
	policy := preflightconfig.OperationPolicy{Ports: []preflightconfig.PortCheck{{
		CheckCommon: preflightconfig.CheckCommon{Roles: []string{"all"}}, Key: "bkeAgent",
	}}}
	applyPolicyBindings(&result, policy)
	require.Len(t, result.Bindings, 1)
	assert.True(t, result.Bindings[0].Applicable)
}

func TestBuildBundleAddsToolAndPathExtensionRules(t *testing.T) {
	runtimeConfig := preflightconfig.Config{
		APIVersion: preflightconfig.APIVersion, Kind: preflightconfig.Kind,
	}
	runtimeConfig.Operations.Init.Tools = append(runtimeConfig.Operations.Init.Tools,
		preflightconfig.ToolCheck{
			CheckCommon: preflightconfig.CheckCommon{
				Description: "docker executable", Remediation: "install docker",
			},
			Key: "docker", Names: []string{"docker"},
		})
	runtimeConfig.Operations.Init.Paths = append(runtimeConfig.Operations.Init.Paths,
		preflightconfig.PathCheck{
			CheckCommon: preflightconfig.CheckCommon{
				Description: "docker config directory", Remediation: "create /etc/docker",
			},
			Key: "docker-config", Path: "/etc/docker",
		})
	raw, err := BuildBundle(runtimeConfig)
	require.NoError(t, err)
	compiled, err := bundle.Load(raw)
	require.NoError(t, err)
	toolRule := ruleByID(t, compiled.Rules, "CONFIG-INIT-TOOL-DOCKER")
	assert.Equal(t, "tool.docker", toolRule.Fact)
	assert.Equal(t, api.OpEq, toolRule.Op)
	// An operator-provided remediation must win over the compiled default.
	assert.Equal(t, "install docker", toolRule.Fix)
	pathRule := ruleByID(t, compiled.Rules, "CONFIG-INIT-PATH-DOCKER-CONFIG")
	assert.Equal(t, "path./etc/docker", pathRule.Fact)
	assert.Equal(t, api.OpCEL, pathRule.Op)
	assert.Equal(t, "create /etc/docker", pathRule.Fix)
}

func TestAppendExtensionBindingsAddsBootstrapToolAndPathChecks(t *testing.T) {
	result := api.ResolveResult{Plan: api.OperationPlan{
		Operation: api.OperationInit,
		Subjects:  []api.Subject{{ID: "bootstrap", Role: []string{"bootstrap"}}},
	}}
	policy := preflightconfig.OperationPolicy{
		Tools: []preflightconfig.ToolCheck{{
			CheckCommon: preflightconfig.CheckCommon{Description: "docker executable", Remediation: "install docker"},
			Key:         "docker", Names: []string{"docker"},
		}},
		Paths: []preflightconfig.PathCheck{{
			CheckCommon: preflightconfig.CheckCommon{Description: "docker config directory", Remediation: "create it"},
			Key:         "docker-config", Path: "/etc/docker",
		}},
	}
	appendExtensionBindings(&result, policy)
	require.Len(t, result.Bindings, 2)
	assert.Equal(t, "CONFIG-INIT-TOOL-DOCKER", result.Bindings[0].TemplateID)
	assert.Equal(t, "tool.docker", result.Bindings[0].Fact)
	assert.Equal(t, "bootstrap", result.Bindings[0].Subject.ID)
	assert.Equal(t, "CONFIG-INIT-PATH-DOCKER-CONFIG", result.Bindings[1].TemplateID)
	assert.Equal(t, "path./etc/docker", result.Bindings[1].Fact)
}
