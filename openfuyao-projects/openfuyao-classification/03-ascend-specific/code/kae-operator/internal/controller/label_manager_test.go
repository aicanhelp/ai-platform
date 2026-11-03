/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

// TestHasKAEHPRELabels HPRE 标签测试
func TestHasKAEHPRELabels(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when labels match",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a259.present": "true",
			},
			expected: true,
		},
		{
			name: "should return true when labels match one hpre label",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "false",
				"feature.node.kubernetes.io/pci-1000_19e5_a259.present": "true",
			},
			expected: true,
		},
		{
			name:     "should return false when labels do not match",
			labels:   map[string]string{},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := hasKAEHPRELabels(tc.labels)
			assert.Equal(t, tc.expected, actual, "hasNFDLabels result mismatch")
		})
	}
}

// TestHasKAESECLabels SEC 标签测试
func TestHasKAESECLabels(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when labels match",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a255.present": "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a256.present": "true",
			},
			expected: true,
		},
		{
			name: "should return true when labels match one sec label",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a256.present": "true",
			},
			expected: true,
		},
		{
			name: "should return false when labels do not match",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5.present": "true",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := hasKAESECLabels(tc.labels)
			assert.Equal(t, tc.expected, actual, "hasCommonKAELabel result mismatch")
		})
	}
}

// TestHasKAEZIPLabels ZIP 标签测试
func TestHasKAEZIPLabels(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when labels match zip device label",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1200_19e5_a250.present": "true",
				"feature.node.kubernetes.io/pci-1200_19e5_a251.present": "true",
			},
			expected: true,
		},
		{
			name: "should return true when labels match one zip label",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1200_19e5_a250.present": "true",
			},
			expected: true,
		},
		{
			name: "should return false when labels do not match",
			labels: map[string]string{
				"feature.node.kubernetes.io/present": "true",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := hasKAEZIPLabels(tc.labels)
			assert.Equal(t, tc.expected, actual, "hasKAEZIPLabels result mismatch")
		})
	}
}

// TestGetNodeOSInfo 获取节点操作系统信息标签测试
func TestGetNodeOSInfo(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected struct {
			osName    string
			osVersion string
		}
	}{
		{
			name: "should return correct os info",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: struct {
				osName    string
				osVersion string
			}{
				osName:    "openEuler",
				osVersion: "22.03",
			},
		},
		{
			name: "should return unknown when osName not found",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: struct {
				osName    string
				osVersion string
			}{
				osName:    "unknown",
				osVersion: "",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actualOsName, actualOsVersion := getNodeOSInfo(tc.labels)
			assert.Equal(t, tc.expected.osName, actualOsName, "OS Name mismatch")
			assert.Equal(t, tc.expected.osVersion, actualOsVersion, "OS Version mismatch")
		})
	}
}

// TestIsOSSupportKAE 节点操作系统是否支持KAE
func TestIsOSSupportKAE(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when os is supported",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: true,
		},
		{
			name: "should return false when os is not supported",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "21.09",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isSupport := isOSSupportKAE(tc.labels)
			assert.Equal(t, tc.expected, isSupport, "isOSSupportKAE result mismatch")
		})
	}
}

// TestCanDeployHPRE 节点是否可以部署 HPRE
func TestCanDeployHPRE(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when node can deploy hpre",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a259.present":   "true",
			},
			expected: true,
		},
		{
			name: "should return false when os is not supported",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "21.09",
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a259.present":   "true",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isSupport := canDeployHPRE(tc.labels)
			assert.Equal(t, tc.expected, isSupport, "canDeployHPRE result mismatch")
		})
	}
}

// TestCanDeploySEC 节点是否可以部署 SEC
func TestCanDeploySEC(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when node can deploy sec",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				"feature.node.kubernetes.io/pci-1000_19e5_a255.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a256.present":   "true",
			},
			expected: true,
		},
		{
			name: "should return false when has not sec",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isSupport := canDeploySEC(tc.labels)
			assert.Equal(t, tc.expected, isSupport, "canDeploySEC result mismatch")
		})
	}
}

// TestCanDeployZIP 节点是否可以部署 ZIP
func TestCanDeployZIP(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when node can deploy zip",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "true",
				"feature.node.kubernetes.io/pci-1200_19e5_a251.present":   "true",
			},
			expected: true,
		},
		{
			name: "should return false when has not zip label",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "false",
				"feature.node.kubernetes.io/pci-1200_19e5_a251.present":   "false",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isSupport := canDeployZIP(tc.labels)
			assert.Equal(t, tc.expected, isSupport, "canDeployZIP result mismatch")
		})
	}
}

// TestHasKAELabels 节点是否具有kae设备
func TestHasKAELabels(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when node has hpre/zip/sec",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "true",
				"feature.node.kubernetes.io/pci-1200_19e5_a251.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a259.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a255.present":   "true",
				"feature.node.kubernetes.io/pci-1000_19e5_a256.present":   "true",
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: true,
		},
		{
			name: "should return false when node has no any kae device",
			labels: map[string]string{
				"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
				"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			isSupport := hasKAELabels(tc.labels, true, true, true)
			assert.Equal(t, tc.expected, isSupport, "hasKAELabels result mismatch")
		})
	}
}

// TestHasKAENFDLabels NFD 标签测试
func TestHasKAENFDLabels(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when labels match",
			labels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			expected: true,
		},
		{
			name: "should return false when labels do not match",
			labels: map[string]string{
				"app": "test-test",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := hasNFDLabels(tc.labels)
			if actual != tc.expected {
				t.Errorf("expected %v, but got %v", tc.expected, actual)
			}
		})
	}
}

// TestHasCommonKAELabel common 标签测试
func TestHasCommonKAELabel(t *testing.T) {
	testCases := []struct {
		name     string
		labels   map[string]string
		expected bool
	}{
		{
			name: "should return true when labels match",
			labels: map[string]string{
				"openfuyao.com/kae.present": "true",
			},
			expected: true,
		},
		{
			name:     "should return false when labels do not match",
			labels:   map[string]string{},
			expected: false,
		},
		{
			name: "should return false when labels value do not match",
			labels: map[string]string{
				"openfuyao.com/kae.present": "false",
			},
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := hasCommonKAELabel(tc.labels)
			if actual != tc.expected {
				t.Errorf("expected %v, but got %v", tc.expected, actual)
			}
		})
	}
}

// TestAddDeployLabels 添加部署标签
func TestAddDeployLabels(t *testing.T) {
	hpreSize := 2
	secSize := 3
	zipSize := 4
	// 创建一个空的 labels map
	labels := make(map[string]string)

	// 测试添加 hpreDeployLabels
	modified := addDeployLabels(labels, hpreDeployLabels)
	assert.True(t, modified, "Expected modified to be true for hpreDeployLabels")
	assert.Equal(t, hpreSize, len(labels), "Expected labels map to have %d items for hpreDeployLabels", hpreSize)
	for key, value := range hpreDeployLabels {
		assert.Equal(t, value, labels[key], "Expected label %s to have value %s for hpreDeployLabels", key, value)
	}

	// 测试添加 secDeployLabels
	modified = addDeployLabels(labels, secDeployLabels)
	assert.True(t, modified, "Expected modified to be true for secDeployLabels")
	assert.Equal(t, secSize, len(labels), "Expected labels map to have %d items for secDeployLabels", secSize)
	for key, value := range secDeployLabels {
		assert.Equal(t, value, labels[key], "Expected label %s to have value %s for secDeployLabels", key, value)
	}

	// 测试添加 secDeployLabels
	modified = addDeployLabels(labels, zipDeployLabels)
	assert.True(t, modified, "Expected modified to be true for zipDeployLabels")
	assert.Equal(t, zipSize, len(labels), "Expected labels map to have %d items for zipDeployLabels", zipSize)
	for key, value := range zipDeployLabels {
		assert.Equal(t, value, labels[key], "Expected label %s to have value %s for zipDeployLabels", key, value)
	}
}

// TestRemoveDeployLabels 删除标签测试
func TestRemoveDeployLabels(t *testing.T) {
	// 创建 labels map 包含所有可能的 labels
	labels := map[string]string{
		"openfuyao.com/kae.deploy.driver":             "true",
		"openfuyao.com/kae.hpre.deploy.device-plugin": "true",
		"openfuyao.com/kae.sec.deploy.device-plugin":  "true",
		"openfuyao.com/kae.zip.deploy.device-plugin":  "true",
	}

	// 测试删除 hpreDeployLabels
	modified := removeDeployLabels(labels, hpreDeployLabels)
	assert.True(t, modified, "Expected modified to be true for hpreDeployLabels removal")
	for key := range hpreDeployLabels {
		assert.NotContains(t, labels, key, "Expected label %s to be deleted", key)
	}

	// 测试删除 secDeployLabels
	modified = removeAllDeployLabels(labels)
	assert.True(t, modified, "Expected modified to be true for all deploy labels removal")
	assert.Empty(t, labels, "Expected labels map to be empty")
}

// TestAddNodeDeployLabels 添加部署标签
func TestAddNodeDeployLabels(t *testing.T) {
	// 创建 KAEPolicyManager 实例
	manager := &KAEPolicyManager{}
	// 创建 OperatorSpec 实例
	operatorSpec := kaev1.OperatorSpec{
		// 设置 IsHPREEnabled、IsSECEnabled 和 IsZIPEnabled 字段为 true
		EnabledHPRE: BoolPtr(true),
		EnabledSEC:  BoolPtr(true),
		EnabledZIP:  BoolPtr(true),
	}

	nodeName := "node1"

	// 测试添加成功
	labels := map[string]string{
		"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "true",
		"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
		"feature.node.kubernetes.io/pci-1000_19e5_a255.present":   "true",
		"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
		"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
	}
	updateLabels := manager.addNodeDeployLabels(labels, operatorSpec, nodeName)
	assert.True(t, updateLabels, "Expected updateLabels to be true")
	assert.Equal(t, 1, manager.hpreNodes, "Expected hpreNodes to be 1")

	// 不需要添加，没有hpre设备，且 sec、zip 禁用
	labels = map[string]string{
		"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "true",
		"feature.node.kubernetes.io/pci-1000_19e5_a255.present":   "true",
		"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
		"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
	}
	*operatorSpec.EnabledSEC = false
	*operatorSpec.EnabledZIP = false
	manager.hpreNodes = 0
	updateLabels = manager.addNodeDeployLabels(labels, operatorSpec, nodeName)
	assert.False(t, updateLabels, "Expected updateLabels to be false")
	assert.Equal(t, 0, manager.hpreNodes, "Expected hpreNodes to be 0")
}

// TestRemoveNodeDeployLabels 删除部署标签
func TestRemoveNodeDeployLabels(t *testing.T) {
	// 创建 KAEPolicyManager 实例
	manager := &KAEPolicyManager{}
	// 创建 OperatorSpec 实例
	operatorSpec := kaev1.OperatorSpec{
		// 设置 IsHPREEnabled、IsSECEnabled 和 IsZIPEnabled 字段为 true
		EnabledHPRE: BoolPtr(true),
		EnabledSEC:  BoolPtr(true),
		EnabledZIP:  BoolPtr(true),
	}

	nodeName := "node2"

	// 测试删除成功
	labels := map[string]string{
		"openfuyao.com/kae.deploy.driver":             "true",
		"openfuyao.com/kae.hpre.deploy.device-plugin": "true",
		"openfuyao.com/kae.sec.deploy.device-plugin":  "true",
		"openfuyao.com/kae.zip.deploy.device-plugin":  "true",
	}
	updateLabels := manager.removeNodeDeployLabels(labels, operatorSpec, nodeName)
	assert.True(t, updateLabels, "Expected updateLabels to be true")

	// 不需要删除
	labels = map[string]string{
		"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
		"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
	}
	updateLabels = manager.removeNodeDeployLabels(labels, operatorSpec, nodeName)
	assert.False(t, updateLabels, "Expected updateLabels to be false")
}

func TestUpdateNodeDeployLabels(t *testing.T) {
	operatorSpec := kaev1.OperatorSpec{
		// 设置 IsHPREEnabled、IsSECEnabled 和 IsZIPEnabled 字段为 true
		EnabledHPRE: BoolPtr(true),
		EnabledSEC:  BoolPtr(true),
		EnabledZIP:  BoolPtr(true),
	}
	labels := map[string]string{
		"openfuyao.com/kae.present":                               "true",
		"openfuyao.com/kae.deploy.driver":                         "true",
		"openfuyao.com/kae.sec.deploy.device-plugin":              "true",
		"openfuyao.com/kae.zip.deploy.device-plugin":              "true",
		"feature.node.kubernetes.io/pci-1200_19e5_a250.present":   "true",
		"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
		"feature.node.kubernetes.io/pci-1000_19e5_a255.present":   "true",
		"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
		"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
	}
	*operatorSpec.EnabledHPRE = true
	*operatorSpec.EnabledSEC = false
	*operatorSpec.EnabledZIP = false
	manager := &KAEPolicyManager{}
	updateLabels := manager.updateNodeDeployLabels(labels, operatorSpec, "test")
	assert.True(t, updateLabels, "Expected updateLabels to be true")

	labels = map[string]string{
		"openfuyao.com/kae.present": "false",
	}
	updateLabels = manager.updateNodeDeployLabels(labels, operatorSpec, "test")
	assert.False(t, updateLabels, "Expected updateLabels to be false")
}

var (
	nodes = []*corev1.Node{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node1", Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present":   "true",
					"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "node2", Labels: map[string]string{
					"openfuyao.com/kae.present":                               "true",
					"openfuyao.com/kae.deploy.driver":                         "true",
					"openfuyao.com/kae.sec.deploy.device-plugin":              "true",
					"openfuyao.com/kae.zip.deploy.device-plugin":              "true",
					"feature.node.kubernetes.io/system-os_release.ID":         "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
)

// TestLabelKAENodes KAE 节点标签测试
func TestLabelKAENodes(t *testing.T) {
	fakeClient := fake.NewClientBuilder().Build()
	testManager := &KAEPolicyManager{
		client: fakeClient,
		ctx:    context.TODO(),
		kaeInstance: &kaev1.KAEPolicy{
			// 请替换为实际的类型
			ObjectMeta: metav1.ObjectMeta{
				Name: "kae-test",
			},
			Spec: kaev1.KAEPolicySpec{
				Operator: kaev1.OperatorSpec{
					EnabledHPRE: new(bool),
					EnabledSEC:  new(bool),
					EnabledZIP:  new(bool),
				},
			},
		},
	}

	for _, node := range nodes {
		err := fakeClient.Create(testManager.ctx, node)
		assert.Nil(t, err)
	}

	*testManager.kaeInstance.Spec.Operator.EnabledHPRE = true
	err := testManager.labelKAENodes()
	assert.Nil(t, err)
	assert.Equal(t, true, testManager.hasKAENodes)
}

// TestHasLabelChanged 标签变动测试
func TestHasLabelChanged(t *testing.T) {
	testCases := []struct {
		name         string
		oldLabels    map[string]string
		newLabels    map[string]string
		shouldChange bool
	}{
		{
			name: "No changes",
			oldLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			newLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			shouldChange: false,
		},
		{
			name:      "Key added",
			oldLabels: map[string]string{},
			newLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			shouldChange: true,
		},
		{
			name: "Key removed",
			oldLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			newLabels:    map[string]string{},
			shouldChange: true,
		},
		{
			name: "Key value changed",
			oldLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "false",
			},
			newLabels: map[string]string{
				"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
			},
			shouldChange: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			changed := hasLabelChanged(tc.oldLabels, tc.newLabels)
			assert.Equal(t, tc.shouldChange, changed, "Expected change detection does not match")
		})
	}
}
