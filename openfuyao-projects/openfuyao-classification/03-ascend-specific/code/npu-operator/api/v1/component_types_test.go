// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package v1

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func TestComponentSpecsHaveIndependentTypes(t *testing.T) {
	policy := &NPUClusterPolicy{}
	require.Equal(t, "InferOperatorSpec", reflect.TypeOf(policy.Spec.InferOperator).Name())
	require.Equal(t, "RDMADevicePluginSpec", reflect.TypeOf(policy.Spec.RDMADevicePlugin).Name())
}

func TestComponentSpecsPreserveJSONFields(t *testing.T) {
	const input = `{"managed":true,"imageSpec":{"repository":"test","tag":"v26.1.0"},"env":[{"name":"TEST","value":"original"}],"commandSpec":{"command":["component"],"args":["--test"]},"volumeSpec":{"volumes":[{"name":"data","emptyDir":{}}]},"resources":{"limits":{"cpu":"1"}},"logRotate":{"logLevel":"info"},"config":{"key":"value"},"nodeSelector":{"pool":"npu"},"tolerations":[{"key":"npu","operator":"Exists"}]}`
	policy := &NPUClusterPolicy{}
	for _, spec := range []any{&policy.Spec.InferOperator, &policy.Spec.RDMADevicePlugin} {
		require.NoError(t, json.Unmarshal([]byte(input), spec))
		data, err := json.Marshal(spec)
		require.NoError(t, err)
		var actual, expected map[string]any
		require.NoError(t, json.Unmarshal(data, &actual))
		require.NoError(t, json.Unmarshal([]byte(input), &expected))
		// Compare populated fields; omitted defaults may be emitted by existing types.
		for key := range expected {
			// Field presence at the top level protects the existing CR paths.
			require.Contains(t, actual, key)
		}
		for _, key := range []string{"config", "nodeSelector", "commandSpec", "env", "tolerations"} {
			expectedValue, expectedExists := expected[key]
			actualValue, actualExists := actual[key]
			require.True(t, expectedExists, "missing expected field %s", key)
			require.True(t, actualExists, "missing actual field %s", key)
			require.Equal(t, expectedValue, actualValue)
		}
		require.NotContains(t, actual, "workload")
		require.NotContains(t, actual, "ComponentWorkloadSpec")
	}
}

func TestMindClusterPolicyDeepCopy(t *testing.T) {
	policy := &NPUClusterPolicy{}
	policy.Spec.InferOperator.Config = map[string]string{"fault_handling": "true"}
	policy.Spec.InferOperator.Env = []corev1.EnvVar{{Name: "TEST", Value: "original"}}
	policy.Spec.RDMADevicePlugin.Image.ImagePullSecrets = []string{"original"}
	policy.Spec.RDMADevicePlugin.NodeSelector = map[string]string{"pool": "rdma"}
	policy.Spec.RDMADevicePlugin.Config = map[string]string{"config.json": "original"}
	policy.Spec.NodeD.ContainerSnapshot.Enabled = true
	policyCopy := policy.DeepCopy()
	policyCopy.Spec.InferOperator.Config["fault_handling"] = "false"
	policyCopy.Spec.InferOperator.Env[0].Value = "changed"
	policyCopy.Spec.RDMADevicePlugin.Image.ImagePullSecrets[0] = "changed"
	policyCopy.Spec.RDMADevicePlugin.NodeSelector["pool"] = "changed"
	policyCopy.Spec.RDMADevicePlugin.Config["config.json"] = "changed"
	require.Equal(t, "true", policy.Spec.InferOperator.Config["fault_handling"])
	require.Equal(t, "original", policy.Spec.InferOperator.Env[0].Value)
	require.Equal(t, "original", policy.Spec.RDMADevicePlugin.Image.ImagePullSecrets[0])
	require.Equal(t, "rdma", policy.Spec.RDMADevicePlugin.NodeSelector["pool"])
	require.Equal(t, "original", policy.Spec.RDMADevicePlugin.Config["config.json"])
	require.True(t, policyCopy.Spec.NodeD.ContainerSnapshot.Enabled)
}

func TestMindClusterDefaultsRemainDisabled(t *testing.T) {
	policy := &NPUClusterPolicy{}
	require.NoError(t, json.Unmarshal([]byte(`{"spec":{"nodeD":{"managed":true}}}`), policy))
	require.False(t, policy.Spec.InferOperator.Managed)
	require.False(t, policy.Spec.RDMADevicePlugin.Managed)
	require.False(t, policy.Spec.NodeD.ContainerSnapshot.Enabled)
}
