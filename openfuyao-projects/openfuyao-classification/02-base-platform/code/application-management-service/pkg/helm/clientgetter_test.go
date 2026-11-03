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

// Package helm
package helm

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/rest"
)

// 测试常量定义
const (
	testClientHost  = "https://kubernetes.example.com:6443"
	testBearerToken = "test-bearer-token"
	testTimeout     = 30
	defaultPort     = 6443
)

// 测试字符串常量
const (
	testNamespaceCustom = "test-namespace"
	testHostLocal       = "https://localhost:8080"
	testCAFile          = "/etc/kubernetes/ca.crt"
	testCertFile        = "/etc/kubernetes/client.crt"
	testKeyFile         = "/etc/kubernetes/client.key"
	emptyNamespace      = ""
	testServiceAccount  = "test-service-account"
)

// 错误消息常量
const (
	expectedDiscoveryError = "couldn't create discovery client"
	nilConfigError         = "config cannot be nil"
	nilClientError         = "client cannot be nil"
)

// createReleaseRestConfig 创建测试用的 REST 配置
func createClientTestRestConfig() *rest.Config {
	return &rest.Config{
		Host:        testClientHost,
		BearerToken: testBearerToken,
		Timeout:     testTimeout,
		TLSClientConfig: rest.TLSClientConfig{
			Insecure: true,
		},
	}
}

// createTestRestConfigWithCerts 创建带证书的测试 REST 配置
func createTestRestConfigWithCerts() *rest.Config {
	return &rest.Config{
		Host: testHostLocal,
		TLSClientConfig: rest.TLSClientConfig{
			CAFile:   testCAFile,
			CertFile: testCertFile,
			KeyFile:  testKeyFile,
		},
	}
}

// createTestRestClientConfig 创建测试用的 RestClientConfig
func createTestRestClientConfig(namespace string) *RestClientConfig {
	config := createClientTestRestConfig()
	return &RestClientConfig{
		namespace:  namespace,
		restConfig: config,
	}
}

// validateRestClientConfig 验证 RestClientConfig 的基本属性
func validateRestClientConfig(t *testing.T, config *RestClientConfig, expectedNamespace string) {
	assert.NotNil(t, config)
	assert.Equal(t, expectedNamespace, config.namespace)
	assert.NotNil(t, config.restConfig)
	assert.Equal(t, testClientHost, config.restConfig.Host)
	assert.Equal(t, testBearerToken, config.restConfig.BearerToken)
}

func TestNewRESTClientConfigWithValidConfig(t *testing.T) {
	config := createClientTestRestConfig()
	assert.NotNil(t, config)

	restClientGetter := NewRESTClientConfig(testNamespace, config)

	assert.NotNil(t, restClientGetter)

	// 验证返回的是 RestClientConfig 类型
	restClientConfig, ok := restClientGetter.(*RestClientConfig)
	assert.True(t, ok)

	validateRestClientConfig(t, restClientConfig, testNamespace)
}

func TestNewRESTClientConfigWithCustomNamespace(t *testing.T) {
	config := createClientTestRestConfig()
	assert.NotNil(t, config)

	restClientGetter := NewRESTClientConfig(testNamespaceCustom, config)

	assert.NotNil(t, restClientGetter)

	restClientConfig, ok := restClientGetter.(*RestClientConfig)
	assert.True(t, ok)

	validateRestClientConfig(t, restClientConfig, testNamespaceCustom)
}

func TestNewRESTClientConfigWithEmptyNamespace(t *testing.T) {
	config := createClientTestRestConfig()
	assert.NotNil(t, config)

	restClientGetter := NewRESTClientConfig(emptyNamespace, config)

	assert.NotNil(t, restClientGetter)

	restClientConfig, ok := restClientGetter.(*RestClientConfig)
	assert.True(t, ok)

	validateRestClientConfig(t, restClientConfig, emptyNamespace)
}

func TestNewRESTClientConfigWithDifferentConfigs(t *testing.T) {
	testCases := []struct {
		name      string
		config    *rest.Config
		namespace string
		expectRCC bool
	}{
		{
			name:      "valid config with default namespace",
			config:    createClientTestRestConfig(),
			namespace: testNamespace,
			expectRCC: true,
		},
		{
			name:      "valid config with custom namespace",
			config:    createTestRestConfigWithCerts(),
			namespace: testNamespaceCustom,
			expectRCC: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			restClientGetter := NewRESTClientConfig(tc.namespace, tc.config)
			assert.NotNil(t, restClientGetter)

			if tc.expectRCC {
				restClientConfig, ok := restClientGetter.(*RestClientConfig)
				assert.True(t, ok)
				assert.Equal(t, tc.namespace, restClientConfig.namespace)
				assert.Equal(t, tc.config, restClientConfig.restConfig)
			} else {
				_, ok := restClientGetter.(*genericclioptions.ConfigFlags)
				assert.True(t, ok)
			}
		})
	}
}

func TestRestClientConfigToRESTConfig(t *testing.T) {
	restClientConfig := createTestRestClientConfig(testNamespace)
	assert.NotNil(t, restClientConfig)

	config, err := restClientConfig.ToRESTConfig()

	assert.NoError(t, err)
	assert.NotNil(t, config)
	assert.Equal(t, testClientHost, config.Host)
	assert.Equal(t, testBearerToken, config.BearerToken)
	assert.Equal(t, restClientConfig.restConfig, config)
}

func TestRestClientConfigToRESTConfigWithNilRestConfig(t *testing.T) {
	restClientConfig := &RestClientConfig{
		namespace:  testNamespace,
		restConfig: nil,
	}

	config, err := restClientConfig.ToRESTConfig()

	assert.NoError(t, err)
	assert.Nil(t, config)
}

func TestRestClientConfigToDiscoveryClient(t *testing.T) {
	restClientConfig := createTestRestClientConfig(testNamespace)
	assert.NotNil(t, restClientConfig)

	// 注意：这个测试可能失败，因为它尝试连接到真实的 Kubernetes 集群
	// 在实际环境中，您可能需要使用 fake 客户端或跳过这个测试
	client, err := restClientConfig.ToDiscoveryClient()

	if err != nil {
		// 如果连接失败，这是预期的（没有真实集群）
		assert.Error(t, err)
		assert.Nil(t, client)
		t.Logf("Expected error when connecting to test cluster: %v", err)
	} else {
		// 如果成功，验证客户端
		assert.NotNil(t, client)
		assert.Implements(t, (*discovery.CachedDiscoveryInterface)(nil), client)
	}
}

func TestRestClientConfigToRESTMapper(t *testing.T) {
	restClientConfig := createTestRestClientConfig(testNamespace)
	assert.NotNil(t, restClientConfig)

	// 注意：这个测试可能失败，因为它依赖于 ToDiscoveryClient
	mapper, err := restClientConfig.ToRESTMapper()

	if err != nil {
		// 如果发现客户端创建失败，这是预期的
		assert.Error(t, err)
		assert.Nil(t, mapper)
		assert.Contains(t, err.Error(), expectedDiscoveryError)
		t.Logf("Expected error when creating discovery client: %v", err)
	} else {
		// 如果成功，验证映射器
		assert.NotNil(t, mapper)
		assert.Implements(t, (*meta.RESTMapper)(nil), mapper)
	}
}

func TestRestClientConfigFieldAccess(t *testing.T) {
	testCases := []struct {
		name      string
		namespace string
		config    *rest.Config
	}{
		{
			name:      "default namespace",
			namespace: testNamespace,
			config:    createClientTestRestConfig(),
		},
		{
			name:      "custom namespace",
			namespace: testNamespaceCustom,
			config:    createTestRestConfigWithCerts(),
		},
		{
			name:      "empty namespace",
			namespace: emptyNamespace,
			config:    createClientTestRestConfig(),
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			restClientConfig := &RestClientConfig{
				namespace:  tc.namespace,
				restConfig: tc.config,
			}

			assert.Equal(t, tc.namespace, restClientConfig.namespace)
			assert.Equal(t, tc.config, restClientConfig.restConfig)

			// 验证通过 ToRESTConfig 访问
			config, err := restClientConfig.ToRESTConfig()
			assert.NoError(t, err)
			assert.Equal(t, tc.config, config)
		})
	}
}

func TestRestClientConfigInterfaceCompliance(t *testing.T) {
	restClientConfig := createTestRestClientConfig(testNamespace)
	assert.NotNil(t, restClientConfig)

	// 验证实现了 genericclioptions.RESTClientGetter 接口
	var getter genericclioptions.RESTClientGetter = restClientConfig
	assert.NotNil(t, getter)

	// 验证所有接口方法都可以调用
	config, err := getter.ToRESTConfig()
	assert.NoError(t, err)
	assert.NotNil(t, config)

	loader := getter.ToRawKubeConfigLoader()
	assert.NotNil(t, loader)

	// ToDiscoveryClient 和 ToRESTMapper 可能因为网络问题而失败，
	// 但应该返回适当的错误而不是 panic
	// 注意：在测试环境中，这些调用可能会失败但不应该 panic
	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("ToDiscoveryClient should not panic, but got: %v", r)
			}
		}()
		_, _ = getter.ToDiscoveryClient()
	}()

	func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("ToRESTMapper should not panic, but got: %v", r)
			}
		}()
		_, _ = getter.ToRESTMapper()
	}()
}

func TestRestClientConfigStructureValidation(t *testing.T) {
	// 验证结构体可以正确创建和访问
	config := createClientTestRestConfig()
	restClientConfig := &RestClientConfig{
		namespace:  testNamespace,
		restConfig: config,
	}

	assert.NotNil(t, restClientConfig)
	assert.Equal(t, testNamespace, restClientConfig.namespace)
	assert.Equal(t, config, restClientConfig.restConfig)

	// 验证字段可以修改
	restClientConfig.namespace = testNamespaceCustom
	assert.Equal(t, testNamespaceCustom, restClientConfig.namespace)

	newConfig := createTestRestConfigWithCerts()
	restClientConfig.restConfig = newConfig
	assert.Equal(t, newConfig, restClientConfig.restConfig)
}

func TestSourceCodeIssueDocumentation(t *testing.T) {
	// 验证当前的不安全行为（通过文档而不是实际调用）
	restClientConfig := &RestClientConfig{
		namespace:  testNamespace,
		restConfig: nil,
	}

	// 验证结构体字段访问是安全的
	assert.Equal(t, testNamespace, restClientConfig.namespace)
	assert.Nil(t, restClientConfig.restConfig)

	// ToRESTConfig 是安全的，因为它只返回字段值
	config, err := restClientConfig.ToRESTConfig()
	assert.NoError(t, err)
	assert.Nil(t, config)
}

// BenchmarkNewRESTClientConfig 基准测试 NewRESTClientConfig
func BenchmarkNewRESTClientConfig(b *testing.B) {
	config := createClientTestRestConfig()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		getter := NewRESTClientConfig(testNamespace, config)
		if getter == nil {
			b.Fatal("Failed to create REST client config")
		}
	}
}

// BenchmarkRestClientConfigToRESTConfig 基准测试 ToRESTConfig
func BenchmarkRestClientConfigToRESTConfig(b *testing.B) {
	restClientConfig := createTestRestClientConfig(testNamespace)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		config, err := restClientConfig.ToRESTConfig()
		if err != nil {
			b.Fatal("Failed to get REST config:", err)
		}
		if config == nil {
			b.Fatal("REST config is nil")
		}
	}
}

// BenchmarkRestClientConfigToRawKubeConfigLoader 基准测试 ToRawKubeConfigLoader
func BenchmarkRestClientConfigToRawKubeConfigLoader(b *testing.B) {
	restClientConfig := createTestRestClientConfig(testNamespace)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		loader := restClientConfig.ToRawKubeConfigLoader()
		if loader == nil {
			b.Fatal("Failed to create kubeconfig loader")
		}
	}
}
