/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package installer

import (
	"context"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/kubernetes/fake"

	"installer-service/pkg/constant"
)

func newTestBKECluster(openFuyaoVersion string) *unstructured.Unstructured {
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("bke.bocloud.com/v1beta1")
	obj.SetKind("BKECluster")
	obj.SetName("bke-cluster")
	obj.SetNamespace("bke-cluster")

	_ = unstructured.SetNestedField(obj.Object, openFuyaoVersion,
		"spec", "clusterConfig", "cluster", "openFuyaoVersion")

	addons := buildTestAddons(openFuyaoVersion)
	_ = unstructured.SetNestedSlice(obj.Object, addons, "spec", "clusterConfig", "addons")
	return obj
}

func buildTestAddons(openFuyaoVersion string) []interface{} {
	return []interface{}{
		map[string]interface{}{
			"name":    "kubeproxy",
			"version": "v1.33.1",
			"param": map[string]interface{}{
				"clusterNetworkMode": "calico",
			},
		},
		map[string]interface{}{
			"name":    "calico",
			"version": "v3.27.3",
			"param": map[string]interface{}{
				"calicoMode":            "vxlan",
				"ipAutoDetectionMethod": "skip-interface=nerdctl*",
			},
		},
		map[string]interface{}{
			"name":    "coredns",
			"version": "v1.10.1",
		},
		map[string]interface{}{
			"name":    "cluster-api",
			"version": "v1.4.3",
			"block":   true,
			"param": map[string]interface{}{
				"containerdVersion": "v2.1.1",
				"manage":            "true",
				"manifestsVersion":  "latest",
				"offline":           "false",
				"openFuyaoVersion":  openFuyaoVersion,
				"providerVersion":   "latest",
				"replicas":          "1",
				"sandbox":           "cr.openfuyao.cn/openfuyao/kubernetes/pause:3.9",
			},
		},
		map[string]interface{}{
			"name":    "openfuyao-system-controller",
			"version": "latest",
			"param": map[string]interface{}{
				"helmRepo":   "https://helm.openfuyao.cn/_core",
				"tagVersion": "latest",
			},
		},
	}
}

func newPatchConfigMap(name, version, yamlData string) *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: constant.PatchNameSpace,
		},
		Data: map[string]string{
			version: yamlData,
		},
	}
}

func newBKEConfigMap() *corev1.ConfigMap {
	return &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      BKEConfigCmKey().Name,
			Namespace: BKEConfigCmKey().Namespace,
		},
		Data: map[string]string{
			"patch.v1.0.0": "cm.v1.0.0",
			"otherRepo":    "",
			"onlineImage":  "",
		},
	}
}

func TestPatchAddonsInfo(t *testing.T) {
	tests := []struct {
		name                string
		openFuyaoVersion    string
		patchYAML           string
		expectedProviderVer string
		expectedTagVer      string
		expectError         bool
	}{
		{
			name:             "valid patch update",
			openFuyaoVersion: "v1.0.0",
			patchYAML: `addons:
- name: cluster-api
  version: v1.4.3
  param:
    providerVersion: patched-v1.0.0
    manifestsVersion: patched-manifests
- name: openfuyao-system-controller
  version: v1.0.0
  param:
    tagVersion: patched-tag
`,
			expectedProviderVer: "patched-v1.0.0",
			expectedTagVer:      "patched-tag",
			expectError:         false,
		},
		{
			name:             "unsupported version",
			openFuyaoVersion: "v999.0.0",
			expectError:      true,
		},
	}

	for _, tt := range tests {
		tt := tt // capture range variable
		t.Run(tt.name, func(t *testing.T) {
			testPatchAddonsInfoCase(t, tt)
		})
	}
}

// testPatchAddonsInfoCase 执行单个测试用例
func testPatchAddonsInfoCase(t *testing.T, tc struct {
	name                string
	openFuyaoVersion    string
	patchYAML           string
	expectedProviderVer string
	expectedTagVer      string
	expectError         bool
}) {
	fakeClient := fake.NewSimpleClientset()

	// Setup BKE configmap
	bkeCM := newBKEConfigMap()
	_, _ = fakeClient.CoreV1().ConfigMaps(bkeCM.Namespace).Create(context.TODO(), bkeCM, metav1.CreateOptions{})

	// Setup patch configmap if needed
	if !tc.expectError || tc.openFuyaoVersion == "v1.0.0" {
		patchCM := newPatchConfigMap("cm.v1.0.0", "v1.0.0", tc.patchYAML)
		_, _ = fakeClient.CoreV1().ConfigMaps(patchCM.Namespace).Create(context.TODO(), patchCM, metav1.CreateOptions{})
	}

	obj := newTestBKECluster(tc.openFuyaoVersion)
	c := &installerClient{clientset: fakeClient}

	err := patchAddonsInfo(c, obj)
	if tc.expectError {
		require.Error(t, err)
		return
	}
	require.NoError(t, err)

	verifyPatchedAddons(t, obj, tc.expectedProviderVer, tc.expectedTagVer)
}

// verifyPatchedAddons 验证 addons 是否被正确 patch
func verifyPatchedAddons(t *testing.T, obj *unstructured.Unstructured, expectedProviderVer, expectedTagVer string) {
	addonsRaw, _, err := unstructured.NestedSlice(obj.Object, "spec", "clusterConfig", "addons")
	require.NoError(t, err)
	require.NotNil(t, addonsRaw)

	addonMap := make(map[string]map[string]interface{})
	for _, raw := range addonsRaw {
		if addon, ok := raw.(map[string]interface{}); ok {
			if name, ok := addon["name"].(string); ok {
				addonMap[name] = addon
			}
		}
	}

	// Verify cluster-api
	clusterAPI := addonMap["cluster-api"]
	param, ok := clusterAPI["param"].(map[string]interface{})
	require.True(t, ok, "cluster-api.param is not a map[string]interface{}")
	assert.Equal(t, expectedProviderVer, param["providerVersion"])
	assert.Equal(t, "patched-manifests", param["manifestsVersion"])
	assert.Equal(t, "v2.1.1", param["containerdVersion"]) // preserved

	// Verify openfuyao-system-controller
	ofCtrl := addonMap["openfuyao-system-controller"]
	ofParam, ok := ofCtrl["param"].(map[string]interface{})
	require.True(t, ok, "param.param is not a map[string]interface{}")
	assert.Equal(t, expectedTagVer, ofParam["tagVersion"])
	assert.Equal(t, "v1.0.0", ofCtrl["version"])

	// Ensure coredns unchanged
	coreDNS := addonMap["coredns"]
	assert.Equal(t, "v1.10.1", coreDNS["version"])
}

const (
	testNodeIPPrefix    = "192.168.1."
	testSingleNodeCount = 1
	testMultiNodeCount  = 3
	testZeroNodeCount   = 0
)

func TestPatchCoreDNSAntiAffinity(t *testing.T) {
	t.Run("SingleNode_DisableAntiAffinity", testPatchCoreDNSAntiAffinitySingleNode)
	t.Run("MultipleNodes_EnableAntiAffinity", testPatchCoreDNSAntiAffinityMultipleNodes)
	t.Run("NoNodes_Skip", testPatchCoreDNSAntiAffinityNoNodes)
	t.Run("NoAddons_Skip", testPatchCoreDNSAntiAffinityNoAddons)
	t.Run("NoCoreDNS_Skip", testPatchCoreDNSAntiAffinityNoCoreDNS)
}

func testPatchCoreDNSAntiAffinitySingleNode(t *testing.T) {
	obj := newTestBKEClusterWithNodes(testSingleNodeCount)
	err := patchCoreDNSAntiAffinity(obj)
	require.NoError(t, err)

	value := getCoreDNSParamValue(t, obj, "EnableAntiAffinity")
	assert.Equal(t, "false", value)
}

func testPatchCoreDNSAntiAffinityMultipleNodes(t *testing.T) {
	obj := newTestBKEClusterWithNodes(testMultiNodeCount)
	err := patchCoreDNSAntiAffinity(obj)
	require.NoError(t, err)

	value := getCoreDNSParamValue(t, obj, "EnableAntiAffinity")
	assert.Equal(t, "true", value)
}

func testPatchCoreDNSAntiAffinityNoNodes(t *testing.T) {
	obj := newTestBKEClusterWithNodes(testZeroNodeCount)
	err := patchCoreDNSAntiAffinity(obj)
	require.NoError(t, err)
}

func testPatchCoreDNSAntiAffinityNoAddons(t *testing.T) {
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("bke.bocloud.com/v1beta1")
	obj.SetKind("BKECluster")

	nodes := []interface{}{
		map[string]interface{}{"ip": fmt.Sprintf("%s%d", testNodeIPPrefix, 1)},
	}
	_ = unstructured.SetNestedSlice(obj.Object, nodes, "spec", "clusterConfig", "nodes")

	err := patchCoreDNSAntiAffinity(obj)
	require.NoError(t, err)
}

func testPatchCoreDNSAntiAffinityNoCoreDNS(t *testing.T) {
	obj := &unstructured.Unstructured{}
	obj.SetAPIVersion("bke.bocloud.com/v1beta1")
	obj.SetKind("BKECluster")

	nodes := []interface{}{
		map[string]interface{}{"ip": fmt.Sprintf("%s%d", testNodeIPPrefix, 1)},
		map[string]interface{}{"ip": fmt.Sprintf("%s%d", testNodeIPPrefix, 2)},
	}
	_ = unstructured.SetNestedSlice(obj.Object, nodes, "spec", "clusterConfig", "nodes")

	addons := []interface{}{
		map[string]interface{}{
			"name":    "calico",
			"version": "v3.27.3",
		},
	}
	_ = unstructured.SetNestedSlice(obj.Object, addons, "spec", "clusterConfig", "addons")

	err := patchCoreDNSAntiAffinity(obj)
	require.NoError(t, err)
}

func newTestBKEClusterWithNodes(nodeCount int) *unstructured.Unstructured {
	obj := newTestBKECluster("v1.0.0")

	nodes := make([]interface{}, nodeCount)
	for i := 0; i < nodeCount; i++ {
		nodes[i] = map[string]interface{}{
			"hostname": fmt.Sprintf("node-%d", i+1),
			"ip":       fmt.Sprintf("%s%d", testNodeIPPrefix, i+1),
		}
	}
	_ = unstructured.SetNestedSlice(obj.Object, nodes, "spec", "clusterConfig", "nodes")

	return obj
}

func getCoreDNSParamValue(t *testing.T, obj *unstructured.Unstructured, key string) string {
	addons, _, err := unstructured.NestedSlice(obj.Object, "spec", "clusterConfig", "addons")
	require.NoError(t, err)

	for _, addon := range addons {
		addonMap, ok := addon.(map[string]interface{})
		if !ok {
			continue
		}
		name, _ := getString(addonMap, "name")
		if name == "coredns" {
			param, ok := addonMap["param"].(map[string]interface{})
			if !ok {
				return ""
			}
			if v, ok := param[key].(string); ok {
				return v
			}
		}
	}
	return ""
}
