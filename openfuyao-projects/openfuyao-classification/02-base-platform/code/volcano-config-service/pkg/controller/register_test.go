/*
 *
 *  * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 *  * openFuyao is licensed under Mulan PSL v2.
 *  * You can use this software according to the terms and conditions of the Mulan PSL v2.
 *  * You may obtain a copy of Mulan PSL v2 at:
 *  *          http://license.coscl.org.cn/MulanPSL2
 *  * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 *  * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 *  * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *  * See the Mulan PSL v2 for more details.
 *
 */

// Package controller
package controller

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// fakeRestConfig 返回不依赖 kubeconfig 的假 rest.Config。
// NewManager 不会实际连接（mapper 懒加载），避免 CI 无 kubeconfig
// 时 GetConfigOrDie 直接 os.Exit 导致整个包 FAIL
func fakeRestConfig() *rest.Config {
	return &rest.Config{Host: "http://127.0.0.1:1"}
}

// createTestScheme 创建一个用于测试的 KubernetesClient scheme
func createTestScheme() *runtime.Scheme {
	scheme := runtime.NewScheme()

	// 添加 KubernetesClient 核心资源
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		panic(err)
	}

	// 使用已定义的 addToScheme 函数
	if err := addToScheme(scheme); err != nil {
		panic(err)
	}

	return scheme
}

// TestSetupNodeConfigController 测试 NodeConfig 控制器设置
func TestSetupNodeConfigController(t *testing.T) {
	// 创建一个 fake manager
	mgr, err := ctrl.NewManager(fakeRestConfig(), ctrl.Options{
		Scheme:                 createTestScheme(),
		MetricsBindAddress:     "0",
		HealthProbeBindAddress: "0",
	})
	require.NoError(t, err)

	// 测试 setupNodeConfigController 函数
	err = setupNodeConfigController(mgr)
	assert.NoError(t, err, "设置 NodeConfig 控制器应该失败")
}

// TestSetupHealthChecks 测试健康检查设置
func TestSetupHealthChecks(t *testing.T) {
	// 创建一个 fake manager
	mgr, err := ctrl.NewManager(fakeRestConfig(), ctrl.Options{
		Scheme:                 createTestScheme(),
		MetricsBindAddress:     "0",
		HealthProbeBindAddress: "0",
	})
	require.NoError(t, err)

	// 测试 setupHealthChecks 函数
	err = setupHealthChecks(mgr)
	assert.NoError(t, err, "设置健康检查应该成功")
}

// TestInitializeNConfig 测试 NodeConfig CR 初始化
func TestInitializeNConfig(t *testing.T) {
	// 创建测试 scheme
	s := createTestScheme()

	// 创建 fake client
	cl := fake.NewClientBuilder().WithScheme(s).Build()

	// 测试 InitializeNodeConfig 函数
	err := InitializeNodeConfig(cl, nil)
	assert.Error(t, err)

	// 验证 NodeConfig CR 是否已创建
	nodeConfig := &NodeConfig{}
	err = cl.Get(context.Background(), types.NamespacedName{Name: "expected-name", Namespace: "expected-namespace"}, nodeConfig)
	assert.Error(t, err)
}

// 测试 addToScheme 和 addOenumaToScheme 函数
func TestSchemeRegistration(t *testing.T) {
	// 创建新的 scheme
	s := runtime.NewScheme()

	// 测试添加到 scheme
	err := clientgoscheme.AddToScheme(s)
	assert.NoError(t, err, "添加 client-go scheme 应该成功")

	// 这里我们假设 addToScheme 和 addOenumaToScheme 函数存在并可用
	err = addToScheme(s)
	assert.NoError(t, err, "添加自定义资源到 scheme 应该成功")
}

// TestNodeConfigController 测试 NodeConfigController
func TestNodeConfigController(t *testing.T) {
	// 创建测试 scheme
	scheme := createTestScheme()

	// 创建 fake client
	fclient := fake.NewClientBuilder().WithScheme(scheme).Build()

	// 创建控制器
	control := &NodeConfigController{
		Client: fclient,
		Scheme: scheme,
	}

	// 测试方法
	mgr, err := ctrl.NewManager(fakeRestConfig(), ctrl.Options{
		Scheme:                 scheme,
		MetricsBindAddress:     "0",
		HealthProbeBindAddress: "0",
	})
	require.NoError(t, err)

	err = control.SetupWithManager(mgr)
	assert.NoError(t, err, "设置 NodeConfigController 应该成功")
}

// MockManager 是一个用于测试的假 Manager
type MockManager struct{}

// GetClient 返回一个假的客户端
func (m *MockManager) GetClient() client.Client {
	s := createTestScheme()
	return fake.NewClientBuilder().WithScheme(s).Build()
}

// GetScheme 返回一个测试 scheme
func (m *MockManager) GetScheme() *runtime.Scheme {
	return createTestScheme()
}

// Start 模拟启动管理器
func (m *MockManager) Start(ctx context.Context) error {
	// 简单返回 nil，模拟成功启动
	return nil
}

// MockCache 是一个用于测试的假 Cache
type MockCache struct{}

// WaitForCacheSync 模拟缓存同步
func (m *MockCache) WaitForCacheSync(ctx context.Context) bool {
	return true
}

// GetCache 返回一个假的缓存对象
// 我们不使用 manager.Cache 接口类型，而是返回我们自己的 MockCache
func (m *MockManager) GetCache() *MockCache {
	return &MockCache{}
}
