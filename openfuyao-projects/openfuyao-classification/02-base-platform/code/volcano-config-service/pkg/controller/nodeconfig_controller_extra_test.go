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

package controller

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDeleteCPUManagerStateFile(t *testing.T) {
	r := &NodeConfigController{}
	err := r.deleteCPUManagerStateFile()
	assert.NoError(t, err)
}

func TestSaveKubeletConfig_Success(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "config.yaml")
	r := &NodeConfigController{}
	config := map[string]interface{}{
		"kind":             "KubeletConfiguration",
		"apiVersion":       "kubelet.config.k8s.io/v1beta1",
		"cpuManagerPolicy": "static",
	}
	err := r.saveKubeletConfig(configPath, config)
	assert.NoError(t, err)

	_, err = os.Stat(configPath)
	assert.NoError(t, err)
}

func TestApplyConfigChanges(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	r := &NodeConfigController{}
	nodeConf := &NodeConfiguration{
		KubeletConfigPath: configPath,
	}
	config := map[string]interface{}{
		"kind":             "KubeletConfiguration",
		"apiVersion":       "kubelet.config.k8s.io/v1beta1",
		"cpuManagerPolicy": "static",
	}
	err := r.applyConfigChanges(nodeConf, config)
	_ = err
}

func TestRestartKubelet(t *testing.T) {
	r := &NodeConfigController{}
	err := r.restartKubelet()
	_ = err
}

func TestRestartBkeKubelet(t *testing.T) {
	r := &NodeConfigController{}
	err := r.restartBkeKubelet()
	_ = err
}

func TestReconcile_NoNodeName(t *testing.T) {
	os.Unsetenv("NODE_NAME")
	r := &NodeConfigController{}
	_, err := r.Reconcile(context.Background(), ctrl.Request{})
	assert.Error(t, err)
}

func TestReconcile_NodeConfigNotFound(t *testing.T) {
	t.Setenv("NODE_NAME", "test-node")
	testScheme := createTestScheme()
	cl := fake.NewClientBuilder().WithScheme(testScheme).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	result, err := r.Reconcile(context.Background(), ctrl.Request{})
	assert.NoError(t, err)
	assert.True(t, result.Requeue)
}

func TestResolveDesiredNodeFlags_Empty(t *testing.T) {
	staticCPU, numa := resolveDesiredNodeFlags([]NodeConfiguration{}, true, false)
	assert.True(t, staticCPU)
	assert.False(t, numa)
}

func TestResolveDesiredNodeFlags_Uniform(t *testing.T) {
	nodes := []NodeConfiguration{
		{EnableStaticCPUManager: true, EnableNUMADistance: true},
		{EnableStaticCPUManager: true, EnableNUMADistance: true},
	}
	staticCPU, numa := resolveDesiredNodeFlags(nodes, false, false)
	assert.True(t, staticCPU)
	assert.True(t, numa)
}

func TestResolveDesiredNodeFlags_NonUniform(t *testing.T) {
	nodes := []NodeConfiguration{
		{EnableStaticCPUManager: true, EnableNUMADistance: true},
		{EnableStaticCPUManager: false, EnableNUMADistance: false},
	}
	staticCPU, numa := resolveDesiredNodeFlags(nodes, false, false)
	assert.False(t, staticCPU)
	assert.False(t, numa)
}

func TestHandleNodeConfigUpdate(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	testScheme := createTestScheme()
	cl := fake.NewClientBuilder().WithScheme(testScheme).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}

	nodeConfig := &NodeConfig{}
	nodeConf := &NodeConfiguration{
		NodeName:               "test-node",
		EnableStaticCPUManager: false,
		EnableNUMADistance:     false,
	}
	result, err := r.handleNodeConfigUpdate(context.Background(), nodeConfig, "test-node", nodeConf)
	// No changes needed, should return requeue after
	_ = err
	_ = result
}

func TestHandleNodeConfigUpdate_NeedsUpdate(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	testScheme := createTestScheme()
	cl := fake.NewClientBuilder().WithScheme(testScheme).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}

	nodeConfig := &NodeConfig{}
	nodeConf := &NodeConfiguration{
		NodeName:               "test-node",
		EnableStaticCPUManager: true,
		EnableNUMADistance:     false,
		KubeletConfigPath:      configPath,
	}
	_, err = r.handleNodeConfigUpdate(context.Background(), nodeConfig, "test-node", nodeConf)
	_ = err
}

func TestUpdateNodeStatusWithRetry_NewNode(t *testing.T) {
	ctx := context.Background()
	testScheme := createTestScheme()
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "kubelet-config"},
	}
	cl := fake.NewClientBuilder().WithScheme(testScheme).WithObjects(nodeConfig).WithStatusSubresource(&NodeConfig{}).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	err := r.updateNodeStatusWithRetry(ctx, nodeConfig, "new-node", NodeConfigStateConfigured, "")
	_ = err
}

func TestUpdateKubeletConfig_EnableStatic(t *testing.T) {
	r := &NodeConfigController{}
	config := map[string]interface{}{
		"cpuManagerPolicy": "none",
	}
	nodeConf := &NodeConfiguration{
		EnableStaticCPUManager: true,
		KubeletConfigPath:      "/var/lib/kubelet/config.yaml",
	}
	err := r.updateKubeletConfig(config, nodeConf)
	_ = err
	assert.Equal(t, "static", config["cpuManagerPolicy"])
}

func TestUpdateKubeletConfig_DisableStatic(t *testing.T) {
	r := &NodeConfigController{}
	config := map[string]interface{}{
		"cpuManagerPolicy": "static",
	}
	nodeConf := &NodeConfiguration{
		EnableStaticCPUManager: false,
		KubeletConfigPath:      "/var/lib/kubelet/config.yaml",
	}
	err := r.updateKubeletConfig(config, nodeConf)
	_ = err
	assert.Equal(t, "none", config["cpuManagerPolicy"])
}

func TestUpdateKubeletConfig_EnableNUMA(t *testing.T) {
	r := &NodeConfigController{}
	config := map[string]interface{}{
		"cpuManagerPolicy": "none",
	}
	nodeConf := &NodeConfiguration{
		EnableStaticCPUManager: false,
		EnableNUMADistance:     true,
		KubeletConfigPath:      "/var/lib/kubelet/config.yaml",
	}
	err := r.updateKubeletConfig(config, nodeConf)
	_ = err
	assert.Equal(t, "static", config["cpuManagerPolicy"])
}

func TestUpdateKubeletConfig_NoChange(t *testing.T) {
	r := &NodeConfigController{}
	config := map[string]interface{}{
		"cpuManagerPolicy": "none",
	}
	nodeConf := &NodeConfiguration{
		EnableStaticCPUManager: false,
		EnableNUMADistance:     false,
		KubeletConfigPath:      "/var/lib/kubelet/config.yaml",
	}
	err := r.updateKubeletConfig(config, nodeConf)
	assert.NoError(t, err)
}

func TestUpdateConfiguration(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	testScheme := createTestScheme()
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "kubelet-config"},
	}
	cl := fake.NewClientBuilder().WithScheme(testScheme).WithObjects(nodeConfig).WithStatusSubresource(&NodeConfig{}).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}

	nodeConf := &NodeConfiguration{
		NodeName:               "test-node",
		EnableStaticCPUManager: true,
		EnableNUMADistance:     false,
		KubeletConfigPath:      configPath,
	}
	currentConfig := map[string]interface{}{
		"cpuManagerPolicy": "none",
	}
	_, err = r.updateConfiguration(context.Background(), nodeConfig, "test-node", nodeConf, currentConfig)
	_ = err
}

func TestReconcile_WithNodeConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	t.Setenv("NODE_NAME", "test-node")
	testScheme := createTestScheme()
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "kubelet-config"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{
					NodeName:               "test-node",
					EnableStaticCPUManager: false,
					EnableNUMADistance:     false,
					KubeletConfigPath:      configPath,
				},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(testScheme).WithObjects(nodeConfig).WithStatusSubresource(&NodeConfig{}).Build()
	r := &NodeConfigController{Client: cl, Scheme: testScheme}
	_, err = r.Reconcile(context.Background(), ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "default", Name: "kubelet-config"}})
	_ = err
}

func TestReadKubeletConfig(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: static\n"), 0644)
	assert.NoError(t, err)

	r := &NodeConfigController{}
	config, err := r.readKubeletConfig(configPath)
	assert.NoError(t, err)
	assert.Equal(t, "static", config["cpuManagerPolicy"])
}

func TestReadKubeletConfig_NotFound(t *testing.T) {
	r := &NodeConfigController{}
	_, err := r.readKubeletConfig("/nonexistent/path")
	assert.Error(t, err)
}

func TestInitializeNodeConfig_NoNodeName(t *testing.T) {
	os.Unsetenv("NODE_NAME")
	testScheme := createTestScheme()
	cl := fake.NewClientBuilder().WithScheme(testScheme).Build()
	err := InitializeNodeConfig(cl, nil)
	assert.Error(t, err)
}

func TestInitializeNodeConfig_WithNodeName(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	t.Setenv("NODE_NAME", "test-node")
	testScheme := createTestScheme()
	cl := fake.NewClientBuilder().WithScheme(testScheme).Build()
	err = InitializeNodeConfig(cl, nil)
	_ = err
}

func TestGetNodeName_Success(t *testing.T) {
	t.Setenv("NODE_NAME", "my-node")
	name, err := getNodeName()
	assert.NoError(t, err)
	assert.Equal(t, "my-node", name)
}

func TestGetNodeName_Empty(t *testing.T) {
	os.Unsetenv("NODE_NAME")
	_, err := getNodeName()
	assert.Error(t, err)
}

func TestReadCurrentKubeletConfig_Success(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: static\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	config, err := readCurrentKubeletConfig()
	assert.NoError(t, err)
	assert.Equal(t, "static", config["cpuManagerPolicy"])
}

func TestReadCurrentKubeletConfig_NotFound(t *testing.T) {
	oldPath := KubeletConfigPath
	KubeletConfigPath = "/nonexistent/path"
	defer func() { KubeletConfigPath = oldPath }()

	_, err := readCurrentKubeletConfig()
	assert.Error(t, err)
}

func TestAnalyzeCurrentConfig_StaticEnabled(t *testing.T) {
	config := map[string]interface{}{
		"cpuManagerPolicy": "static",
	}
	staticCPU, numaEnabled := analyzeCurrentConfig(config)
	assert.True(t, staticCPU)
	assert.False(t, numaEnabled)
}

func TestAnalyzeCurrentConfig_NUMAEnabled(t *testing.T) {
	config := map[string]interface{}{
		"cpuManagerPolicy": "none",
		"topologyManagerPolicyOptions": map[string]interface{}{
			"prefer-closest-numa-nodes": "true",
		},
	}
	staticCPU, numaEnabled := analyzeCurrentConfig(config)
	assert.False(t, staticCPU)
	assert.True(t, numaEnabled)
}

func TestAnalyzeCurrentConfig_BothEnabled(t *testing.T) {
	config := map[string]interface{}{
		"cpuManagerPolicy": "static",
		"topologyManagerPolicyOptions": map[string]interface{}{
			"prefer-closest-numa-nodes": "true",
		},
	}
	staticCPU, numaEnabled := analyzeCurrentConfig(config)
	assert.True(t, staticCPU)
	assert.True(t, numaEnabled)
}

func TestAnalyzeCurrentConfig_Empty(t *testing.T) {
	config := map[string]interface{}{}
	staticCPU, numaEnabled := analyzeCurrentConfig(config)
	assert.False(t, staticCPU)
	assert.False(t, numaEnabled)
}

func TestInitializeNodeConfig_NodeAlreadyExists(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	t.Setenv("NODE_NAME", "existing-node")
	testScheme := createTestScheme()
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "kubelet-config"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "existing-node"},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(testScheme).WithObjects(nodeConfig).Build()
	err = InitializeNodeConfig(cl, nil)
	assert.NoError(t, err)
}

func TestInitializeNodeConfig_AddNewNode(t *testing.T) {
	tempDir := t.TempDir()
	configPath := filepath.Join(tempDir, "kubelet-config.yaml")
	err := os.WriteFile(configPath, []byte("cpuManagerPolicy: none\n"), 0644)
	assert.NoError(t, err)

	oldPath := KubeletConfigPath
	KubeletConfigPath = configPath
	defer func() { KubeletConfigPath = oldPath }()

	t.Setenv("NODE_NAME", "new-node")
	testScheme := createTestScheme()
	nodeConfig := &NodeConfig{
		ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "kubelet-config"},
		Spec: NodeConfigSpec{
			Nodes: []NodeConfiguration{
				{NodeName: "existing-node"},
			},
		},
	}
	cl := fake.NewClientBuilder().WithScheme(testScheme).WithObjects(nodeConfig).Build()
	err = InitializeNodeConfig(cl, nil)
	_ = err
}
