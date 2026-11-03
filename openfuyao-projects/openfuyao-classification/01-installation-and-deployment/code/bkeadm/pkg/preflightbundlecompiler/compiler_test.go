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

package preflightbundlecompiler

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gopkg.openfuyao.cn/bkeadm/pkg/preflightconfig"
	"gopkg.openfuyao.cn/preflight/api"
)

func TestApplyPolicyUpdatesReviewedPresentation(t *testing.T) {
	config := preflightconfig.Config{Operations: preflightconfig.Operations{
		Init: preflightconfig.OperationPolicy{
			Resources: []preflightconfig.ResourceCheck{{
				CheckCommon: preflightconfig.CheckCommon{
					Description: "CPU available", Failure: "warn", Unknown: "skip", Remediation: "add CPU",
				},
				Key: "cpu", Minimum: 8,
			}},
			Ports: []preflightconfig.PortCheck{{
				CheckCommon: preflightconfig.CheckCommon{Description: "API port", Failure: "fail", Unknown: "fail"},
				Key:         "kubernetes", DefaultPort: 36443,
			}},
			Platform: preflightconfig.PlatformPolicy{Distributions: []preflightconfig.PlatformDistribution{{
				Name: "openeuler", Version: "22.03",
			}}},
		},
	}}
	rules := []api.RuleTemplate{
		{ID: "INIT-RES-CPU", Operation: api.OperationInit},
		{ID: "INIT-PORT-K8S", Operation: api.OperationInit},
		{ID: "INIT-PLAT-DISTRO", Operation: api.OperationInit},
	}
	ApplyPolicy(config, rules)
	assert.Equal(t, "CPU available", rules[0].Name)
	assert.Equal(t, int64(8), rules[0].Expected)
	assert.Equal(t, api.SeverityWarning, rules[0].Level)
	assert.Equal(t, api.UnknownIgnore, rules[0].Unknown)
	assert.Equal(t, "add CPU", rules[0].Fix)
	assert.Equal(t, "API port", rules[1].Name)
	assert.Contains(t, rules[1].Fix, "36443")
	assert.Equal(t, []any{[]any{"openeuler", "22.03"}}, rules[2].Expected)
}

func TestPolicyMatchingHelpers(t *testing.T) {
	assert.True(t, bindingMatchesPort("CLUSTER-PORT-ETCD-PEER", "etcdPeer", api.OperationClusterCreate))
	assert.False(t, bindingMatchesPort("INIT-PORT-K8S", "unknown", api.OperationInit))
	assert.False(t, matchesResource(api.RuleTemplate{ID: "INIT-RES-CPU"}, "unknown"))
}

func TestCustomRules(t *testing.T) {
	config := preflightconfig.Config{Operations: preflightconfig.Operations{
		Init: preflightconfig.OperationPolicy{CustomChecks: []preflightconfig.CustomCheck{{
			CheckCommon: preflightconfig.CheckCommon{
				Description: "hosts mapping", Failure: "warn", Unknown: "warn", Remediation: "fix hosts",
			},
			Key: "hosts", Expected: "loopback exists", Expression: "actual == true",
			Collector: preflightconfig.CustomCollector{Type: "exec", Path: "/opt/check", ResultType: "bool"},
		}}},
		ClusterCreate: preflightconfig.OperationPolicy{CustomChecks: []preflightconfig.CustomCheck{{
			CheckCommon: preflightconfig.CheckCommon{Description: "runtime", Remediation: "fix runtime"},
			Key:         "runtime", Expected: "healthy", Expression: "actual == true",
			Collector: preflightconfig.CustomCollector{Type: "exec", Path: "/opt/runtime", ResultType: "bool"},
		}}},
	}}

	rules := CustomRules(config)
	require.Len(t, rules, 2)
	assert.Equal(t, "CONFIG-INIT-CUSTOM-HOSTS", rules[0].ID)
	assert.Equal(t, "custom.init.hosts", rules[0].Fact)
	assert.Equal(t, api.OpCEL, rules[0].Op)
	assert.Equal(t, "loopback exists", rules[0].ExpectedText)
	assert.Equal(t, api.SeverityWarning, rules[0].Level)
	assert.Equal(t, api.UnknownWarn, rules[0].Unknown)
	assert.Empty(t, rules[0].DependsOn)
	assert.Equal(t, []string{"CLUSTER-SSH-PRIVILEGE"}, rules[1].DependsOn)
	assert.True(t, HasCustomChecks(config))
	assert.False(t, HasCustomChecks(preflightconfig.Config{}))

}

func TestApplyRuntimePolicyUpdatesToolsPathsAndTopology(t *testing.T) {
	config := preflightconfig.Config{Operations: preflightconfig.Operations{
		Init: preflightconfig.OperationPolicy{
			Tools: []preflightconfig.ToolCheck{{
				CheckCommon: preflightconfig.CheckCommon{
					Description: "tar command", Failure: "skip", Unknown: "skip", Remediation: "install tar",
				},
				Key: "tar", Names: []string{"tar"},
			}},
			Paths: []preflightconfig.PathCheck{{
				CheckCommon: preflightconfig.CheckCommon{Description: "workspace", Remediation: "fix workspace"},
				Key:         "workspace", Require: []string{"create"},
			}},
		},
		ClusterCreate: preflightconfig.OperationPolicy{Topology: preflightconfig.TopologyPolicy{
			Masters: preflightconfig.NodeCountPolicy{Minimum: 3},
			Etcd:    preflightconfig.NodeCountPolicy{Minimum: 5},
			Workers: preflightconfig.NodeCountPolicy{Minimum: 2},
		}},
	}}
	rules := []api.RuleTemplate{
		{ID: "INIT-TOOL-TAR", Operation: api.OperationInit},
		{ID: "INIT-PATH-CREATE", Operation: api.OperationInit},
		{ID: "CLUSTER-TOPO-MASTER", Operation: api.OperationClusterCreate},
		{ID: "CLUSTER-TOPO-ETCD", Operation: api.OperationClusterCreate},
		{ID: "CLUSTER-TOPO-WORKER", Operation: api.OperationClusterCreate},
	}

	ApplyRuntimePolicy(config, rules)
	assert.Equal(t, "tar command", rules[0].Name)
	assert.Equal(t, api.SeverityInfo, rules[0].Level)
	assert.Equal(t, api.UnknownIgnore, rules[0].Unknown)
	assert.Equal(t, "workspace", rules[1].Name)
	assert.Equal(t, 3, rules[2].Expected)
	assert.Equal(t, 5, rules[3].Expected)
	assert.Equal(t, 2, rules[4].Expected)
}

func TestResourcePolicyFixTracksThreshold(t *testing.T) {
	config := preflightconfig.Config{Operations: preflightconfig.Operations{
		Init: preflightconfig.OperationPolicy{
			Resources: []preflightconfig.ResourceCheck{
				{
					CheckCommon: preflightconfig.CheckCommon{Description: "memory"},
					Key:         "memory", Minimum: 32 * 1024 * 1024 * 1024,
				},
				{
					CheckCommon: preflightconfig.CheckCommon{Description: "cpu"},
					Key:         "cpu", Minimum: 8,
				},
			},
		},
		ClusterCreate: preflightconfig.OperationPolicy{},
	}}
	rules := []api.RuleTemplate{
		{ID: "INIT-RES-MEMORY", Operation: api.OperationInit, Fix: "use a host with at least 4 GiB memory"},
		{ID: "INIT-RES-CPU", Operation: api.OperationInit, Fix: "use a host with at least 2 logical CPUs"},
	}
	ApplyPolicy(config, rules)
	assert.Equal(t, int64(32*1024*1024*1024), rules[0].Expected)
	assert.Equal(t, "use a host with at least 32 GiB memory", rules[0].Fix)
	assert.Equal(t, int64(8), rules[1].Expected)
	assert.Equal(t, "use a host with at least 8 logical CPUs", rules[1].Fix)
}
