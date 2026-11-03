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

package preflightconfig

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.openfuyao.cn/preflight/api"
)

func TestLoadAndEffectivePort(t *testing.T) {
	raw := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations:
  init:
    ports:
      - key: kubernetes
        description: api port
        defaultPort: 1234
        portFrom: port.kubernetes
      - key: ntp
        description: ntp
        defaultPort: 123
        protocol: udp
  clusterCreate: {}`)
	c, err := Load(raw)
	require.NoError(t, err)
	plan := api.OperationPlan{Values: map[string]api.Value{"port.kubernetes": {Value: 4321}}}
	ports, err := EffectivePorts(c, api.OperationInit, plan)
	require.NoError(t, err)
	require.Equal(t, EffectivePort{Key: "kubernetes", Port: 4321, Protocol: "tcp", ValueSource: "plan"}, ports[0])
	require.Equal(t, EffectivePort{Key: "ntp", Port: 123, Protocol: "udp", ValueSource: "config-default"}, ports[1])
}

func TestLoadRejectsUnknownFieldAndDuplicatePort(t *testing.T) {
	_, err := Load([]byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations: {init: {ports: [{key: kubernetes, description: x, defaultPort: 1, typo: true}]}}`))
	require.Error(t, err)
	_, err = Load([]byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations: {init: {ports: [{key: kubernetes, description: x, defaultPort: 1}, {key: kubernetes, description: y, defaultPort: 2}]}}`))
	require.Error(t, err)
}

func TestLoadRejectsMultiDocument(t *testing.T) {
	_, err := Load([]byte("apiVersion: preflight.openfuyao.io/v1\nkind: PreflightConfig\nmetadata: {}\n---\napiVersion: preflight.openfuyao.io/v1\nkind: PreflightConfig\nmetadata: {}\n"))
	require.Error(t, err)
}

func TestValidateConditionsRequiresBoolPlanValue(t *testing.T) {
	policy := OperationPolicy{Ports: []PortCheck{{CheckCommon: CheckCommon{Description: "x", When: "feature.x"}, Key: "x", DefaultPort: 1}}}
	err := ValidateConditions(policy, api.OperationPlan{Values: map[string]api.Value{"feature.x": {Type: api.ValueTypeString, Value: "true"}}})
	require.Error(t, err)
}

func TestValidateRejectsPortWithoutReviewedRuleSource(t *testing.T) {
	c := Config{APIVersion: APIVersion, Kind: Kind, Operations: Operations{Init: OperationPolicy{Ports: []PortCheck{{
		CheckCommon: CheckCommon{Description: "invented port"}, Key: "invented", DefaultPort: 12345,
	}}}}}
	err := Validate(c)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no reviewed rule/source")
}

func TestValidateRejectsDeadPolicyFields(t *testing.T) {
	base := Config{APIVersion: APIVersion, Kind: Kind}
	withName := base
	withName.Operations.ClusterCreate.Topology.Unique = []string{"hostname", "name", "ip"}
	assert.ErrorContains(t, Validate(withName), "has no reviewed rule")

	withWrite := base
	withWrite.Operations.Init.Paths = []PathCheck{{
		CheckCommon: CheckCommon{Description: "workspace"}, Key: "workspace", Require: []string{"create", "write"},
	}}
	assert.ErrorContains(t, Validate(withWrite), "unsupported requirement")

	withMixedAll := base
	withMixedAll.Operations.ClusterCreate.Ports = []PortCheck{{
		CheckCommon: CheckCommon{Description: "agent", Roles: []string{"all", "worker"}}, Key: "bkeAgent", DefaultPort: 58080,
	}}
	assert.ErrorContains(t, Validate(withMixedAll), "role all cannot be combined")

	withResourceCondition := base
	withResourceCondition.Operations.Init.Resources = []ResourceCheck{{
		CheckCommon: CheckCommon{Description: "cpu", When: "feature.x"}, Key: "cpu", Minimum: 2,
	}}
	assert.ErrorContains(t, Validate(withResourceCondition), "does not support when")

	withToolRoles := base
	withToolRoles.Operations.Init.Tools = []ToolCheck{{
		CheckCommon: CheckCommon{Description: "tar", Roles: []string{"master"}}, Key: "tar", Names: []string{"tar"},
	}}
	assert.ErrorContains(t, Validate(withToolRoles), "does not support roles")
}

func TestValidateRejectsUnsupportedRuntimeProjection(t *testing.T) {
	base := Config{APIVersion: APIVersion, Kind: Kind}
	tests := []struct {
		name   string
		mutate func(*Config)
		part   string
	}{
		{
			name: "port protocol",
			mutate: func(config *Config) {
				config.Operations.Init.Ports = []PortCheck{{
					CheckCommon: CheckCommon{Description: "api"}, Key: "kubernetes", DefaultPort: 36443,
					PortFrom: "port.kubernetes", Protocol: "udp",
				}}
			},
			part: "protocol must be tcp",
		},
		{
			name: "port source",
			mutate: func(config *Config) {
				config.Operations.Init.Ports = []PortCheck{{
					CheckCommon: CheckCommon{Description: "api"}, Key: "kubernetes", DefaultPort: 36443,
					PortFrom: "port.console",
				}}
			},
			part: `portFrom must be "port.kubernetes"`,
		},
		{
			name: "tool path",
			mutate: func(config *Config) {
				config.Operations.Init.Tools = []ToolCheck{{
					CheckCommon: CheckCommon{Description: "tar"}, Key: "tar", Names: []string{"/bin/tar"},
				}}
			},
			part: "requires one executable basename",
		},
		{
			name: "management secret",
			mutate: func(config *Config) {
				config.Operations.ClusterCreate.ManagementCluster.RequiredKubeconfig = "other/secret"
			},
			part: "only supports kube-system/localkubeconfig",
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			config := base
			test.mutate(&config)
			assert.ErrorContains(t, Validate(config), test.part)
		})
	}
}

func TestLoadCustomChecks(t *testing.T) {
	raw := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations:
  init:
    customChecks:
      - key: hosts-loopback
        description: /etc/hosts 回环地址配置
        expected: 包含 127.0.0.1 或 ::1 回环地址映射
        collector: {type: fileContent, path: /etc/hosts}
        expression: actual.matches('(?m)^\\s*(127\\.0\\.0\\.1|::1)(\\s|$)')
        failure: fail
        unknown: fail
        remediation: 补充回环地址映射后重试
  clusterCreate:
    customChecks:
      - key: runtime-health
        description: 容器运行时健康状态
        expected: 运行时为 containerd 且健康
        roles: [master, worker]
        collector:
          type: exec
          path: /opt/openfuyao/check-runtime
          args: [--format, json]
          resultType: object
          publicFields: [runtime, healthy]
        expression: actual.healthy && actual.runtime == 'containerd'
        remediation: 修复容器运行时后重试
`)
	config, err := Load(raw)
	require.NoError(t, err)
	assert.Equal(t, "/etc/hosts", config.Operations.Init.CustomChecks[0].Collector.Path)
	assert.Equal(t, []string{"runtime", "healthy"},
		config.Operations.ClusterCreate.CustomChecks[0].Collector.PublicFields)
}

func TestValidateRejectsInvalidCustomChecks(t *testing.T) {
	valid := CustomCheck{
		CheckCommon: CheckCommon{Description: "check", Remediation: "fix"},
		Key:         "custom-check", Expected: "expected", Expression: "actual == true",
		Collector: CustomCollector{Type: "exec", Path: "/opt/check", ResultType: "bool"},
	}
	tests := []struct {
		name   string
		mutate func(*CustomCheck)
		part   string
	}{
		{"key", func(check *CustomCheck) { check.Key = "Bad_Key" }, "key"},
		{"path", func(check *CustomCheck) { check.Collector.Path = "check.sh" }, "absolute path"},
		{"expression", func(check *CustomCheck) { check.Expression = "actual +" }, "compile CEL"},
		{"type", func(check *CustomCheck) { check.Collector.ResultType = "float" }, "resultType"},
		{"object fields", func(check *CustomCheck) {
			check.Collector.ResultType = "object"
		}, "requires publicFields"},
		{"file extras", func(check *CustomCheck) {
			check.Collector = CustomCollector{Type: "fileContent", Path: "/etc/hosts", Args: []string{"x"}}
		}, "only supports path"},
		{"missing text", func(check *CustomCheck) { check.Expected = " " }, "requires description"},
		{"init roles", func(check *CustomCheck) { check.Roles = []string{"master"} }, "does not support roles"},
		{"collector type", func(check *CustomCheck) { check.Collector.Type = "http" }, "collector type"},
		{"non object fields", func(check *CustomCheck) {
			check.Collector.PublicFields = []string{"value"}
		}, "requires object"},
		{"invalid public field", func(check *CustomCheck) {
			check.Collector.ResultType = "object"
			check.Collector.PublicFields = []string{"bad-field"}
		}, "invalid public field"},
		{"duplicate public field", func(check *CustomCheck) {
			check.Collector.ResultType = "object"
			check.Collector.PublicFields = []string{"value", "value"}
		}, "duplicate public field"},
		{"argument NUL", func(check *CustomCheck) { check.Collector.Args = []string{"bad\x00arg"} }, "contain NUL"},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			check := valid
			test.mutate(&check)
			config := Config{APIVersion: APIVersion, Kind: Kind}
			config.Operations.Init.CustomChecks = []CustomCheck{check}
			assert.ErrorContains(t, Validate(config), test.part)
		})
	}
}

func TestCustomCheckIdentityAndOperation(t *testing.T) {
	config := Config{Operations: Operations{
		Init:          OperationPolicy{CustomChecks: []CustomCheck{{Key: "init"}}},
		ClusterCreate: OperationPolicy{CustomChecks: []CustomCheck{{Key: "cluster"}}},
	}}
	assert.Equal(t, "cluster", config.Operation(api.OperationClusterCreate).CustomChecks[0].Key)
	assert.Equal(t, "init", config.Operation(api.OperationInit).CustomChecks[0].Key)
	assert.Equal(t, "CONFIG-CLUSTER-CREATE-CUSTOM-RUNTIME-HEALTH",
		CheckID(api.OperationClusterCreate, "custom", "runtime.health"))
}

func TestValidateCustomCondition(t *testing.T) {
	policy := OperationPolicy{CustomChecks: []CustomCheck{{
		CheckCommon: CheckCommon{When: "feature.custom"},
	}}}
	err := ValidateConditions(policy, api.OperationPlan{Values: map[string]api.Value{
		"feature.custom": {Type: api.ValueTypeBool, Value: true},
	}})
	assert.NoError(t, err)
	err = ValidateConditions(policy, api.OperationPlan{Values: map[string]api.Value{}})
	assert.ErrorContains(t, err, "not present")
}

// TestValidateCustomCollectorBranches exercises validateCustomCollector with
// platform-neutral absolute paths so every branch is reached on any OS.
func TestValidateCustomCollectorBranches(t *testing.T) {
	script := filepath.Join(t.TempDir(), "check")
	base := func() CustomCheck {
		return CustomCheck{
			CheckCommon: CheckCommon{Description: "check", Remediation: "fix"},
			Key:         "custom-check", Expected: "expected", Expression: "actual == true",
			Collector: CustomCollector{Type: "exec", Path: script, ResultType: "bool"},
		}
	}
	assert.NoError(t, validateCustomCollector("init", base()))

	// fileContent with extras is rejected.
	fileExtra := base()
	fileExtra.Collector = CustomCollector{Type: "fileContent", Path: script, ResultType: "bool"}
	assert.ErrorContains(t, validateCustomCollector("init", fileExtra), "only supports path")

	// publicFields on a non-object resultType is rejected.
	nonObject := base()
	nonObject.Collector.PublicFields = []string{"value"}
	assert.ErrorContains(t, validateCustomCollector("init", nonObject), "requires object")

	// object resultType without publicFields is rejected.
	objectNoFields := base()
	objectNoFields.Collector.ResultType = "object"
	assert.ErrorContains(t, validateCustomCollector("init", objectNoFields), "requires publicFields")

	// validResultType table.
	for _, valid := range []string{"bool", "integer", "string", "stringList", "integerList", "object"} {
		assert.True(t, validResultType(valid), valid)
	}
	assert.False(t, validResultType("float"))
}

func TestLoadResourceMinimumAcceptsUnitSuffixes(t *testing.T) {
	load := func(minimum string) (Config, error) {
		raw := "apiVersion: preflight.openfuyao.io/v1\nkind: PreflightConfig\n" +
			"metadata: {}\noperations:\n  init:\n    resources:\n" +
			"      - key: memory\n        description: memory\n        minimum: " + minimum + "\n  clusterCreate: {}"
		config, err := Load([]byte(raw))
		return config, err
	}

	for _, tc := range []struct {
		minimum string
		want    int64
	}{
		{minimum: "1073741824", want: 1073741824},
		{minimum: `"1G"`, want: 1073741824},
		{minimum: `"4G"`, want: 4294967296},
		{minimum: `"50g"`, want: 53687091200},
		{minimum: `"1T"`, want: 1099511627776},
		{minimum: `"512M"`, want: 536870912},
		{minimum: `"1024K"`, want: 1048576},
		{minimum: `"1.5G"`, want: 1610612736},
		{minimum: `"1073741824B"`, want: 1073741824},
	} {
		config, err := load(tc.minimum)
		require.NoError(t, err, tc.minimum)
		require.Len(t, config.Operations.Init.Resources, 1, tc.minimum)
		assert.Equal(t, tc.want, int64(config.Operations.Init.Resources[0].Minimum), tc.minimum)
	}

	for _, invalid := range []string{`""`, `"4X"`, `"4 GB"`, `"-1G"`, `"abc"`, `"G"`, `-100`, `-1`} {
		_, err := load(invalid)
		assert.Error(t, err, invalid)
	}
}

func TestLoadAcceptsNewToolAndPathKeys(t *testing.T) {
	raw := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations:
  init:
    tools:
      - key: docker
        description: docker executable
        names: [docker]
    paths:
      - key: docker-config
        description: docker config directory
        path: /etc/docker
  clusterCreate: {}`)
	config, err := Load(raw)
	require.NoError(t, err)
	require.Len(t, config.Operations.Init.Tools, 1)
	require.Len(t, config.Operations.Init.Paths, 1)
	assert.Equal(t, "/etc/docker", config.Operations.Init.Paths[0].Path)

	// A path key without a clean absolute path is rejected.
	bad := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations:
  init:
    paths:
      - key: docker-config
        description: docker config directory
        path: etc/docker
  clusterCreate: {}`)
	_, err = Load(bad)
	require.ErrorContains(t, err, "clean absolute")

	// Non-init operations still reject tools and paths.
	cluster := []byte(`apiVersion: preflight.openfuyao.io/v1
kind: PreflightConfig
metadata: {}
operations:
  init: {}
  clusterCreate:
    tools:
      - key: docker
        description: docker executable
        names: [docker]`)
	_, err = Load(cluster)
	require.ErrorContains(t, err, "no supported rule")
}
