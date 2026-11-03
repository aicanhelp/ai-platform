/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package k8s

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// 生成标准的测试用 kubeconfig 内容
func generateTestKubeConfigContent() string {
	return `
apiVersion: v1
kind: Config
clusters:
- cluster:
    server: https://localhost:6443
    insecure-skip-tls-verify: true
  name: test-cluster
contexts:
- context:
    cluster: test-cluster
    user: test-user
  name: test-context
current-context: test-context
users:
- name: test-user
  user:
    token: test-token
`
}

// createValidKubeConfigFile 创建有效的 kubeconfig 临时文件
func createValidKubeConfigFile(t *testing.T) (string, func()) {
	tempFile, err := os.CreateTemp("", "kubeconfig-test")
	require.NoError(t, err)

	kubeConfigContent := generateTestKubeConfigContent()
	_, err = tempFile.WriteString(kubeConfigContent)
	require.NoError(t, err)
	err = tempFile.Close()
	require.NoError(t, err)

	cleanup := func() {
		err = os.Remove(tempFile.Name())
		require.NoError(t, err)
	}

	return tempFile.Name(), cleanup
}

// createHomeKubeConfig 创建带 HOME 环境变量的 kubeconfig
func createHomeKubeConfig(t *testing.T) (string, func()) {
	const fileMode1 = 0755
	const fileMode2 = 0644
	// 创建临时home目录和kubeconfig
	tempDir, err := os.MkdirTemp("", "test-home")
	require.NoError(t, err)
	kubeDir := filepath.Join(tempDir, ".kube")
	err = os.MkdirAll(kubeDir, fileMode1)
	require.NoError(t, err)
	configFile := filepath.Join(kubeDir, "config")

	kubeConfigContent := generateTestKubeConfigContent()
	err = os.WriteFile(configFile, []byte(kubeConfigContent), fileMode2)
	require.NoError(t, err)

	// 设置HOME环境变量
	originalHome := os.Getenv("HOME")
	err = os.Setenv("HOME", tempDir)
	require.NoError(t, err)

	cleanup := func() {
		err = os.RemoveAll(tempDir)
		require.NoError(t, err)
		if originalHome != "" {
			err = os.Setenv("HOME", originalHome)
			require.NoError(t, err)
		} else {
			err = os.Unsetenv("HOME")
			require.NoError(t, err)
		}
	}

	return "", cleanup
}

type KubernetesCfgTestCase struct {
	name            string
	kubeCfgFilePath string
	setupFunc       func(*testing.T) (string, func())
	expectedQPS     float32
	expectedBurst   int
	expectValidCfg  bool
}

// createTestCasesForNewKubernetesCfg 创建 TestNewKubernetesCfg 的测试用例
func createTestCasesForNewKubernetesCfg() []KubernetesCfgTestCase {
	return []KubernetesCfgTestCase{
		{
			name:            "with valid kubeconfig file",
			kubeCfgFilePath: "",
			setupFunc:       createValidKubeConfigFile,
			expectedQPS:     1e6,
			expectedBurst:   1e6,
			expectValidCfg:  true,
		},
		{
			name:            "with empty path and home config",
			kubeCfgFilePath: "",
			setupFunc:       createHomeKubeConfig,
			expectedQPS:     1e6,
			expectedBurst:   1e6,
			expectValidCfg:  true,
		},
	}
}

func TestNewKubernetesCfg(t *testing.T) {
	tests := createTestCasesForNewKubernetesCfg()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var cleanup func()
			var actualPath string

			if tt.setupFunc != nil {
				actualPath, cleanup = tt.setupFunc(t)
				defer cleanup()
			}

			// 使用实际路径或测试指定的路径
			testPath := tt.kubeCfgFilePath
			if actualPath != "" {
				testPath = actualPath
			}

			cfg := NewKubernetesCfg(testPath)

			// 验证基本属性
			assert.NotNil(t, cfg)

			// 验证KubeConfigFile设置
			if testPath != "" {
				assert.Equal(t, testPath, cfg.KubeConfigFile)
			} else {
				// 空路径时会查找默认位置
				assert.NotEmpty(t, cfg.KubeConfigFile)
			}

			// 验证KubeConfig是否正确加载
			if tt.expectValidCfg {
				assert.NotNil(t, cfg.KubeConfig)
				if cfg.KubeConfig != nil {
					assert.NotEmpty(t, cfg.KubeConfig.Host)
				}
			}
		})
	}
}

func TestKubernetesCfgIntegration(t *testing.T) {
	t.Run("complete workflow with valid config", func(t *testing.T) {
		// 创建临时kubeconfig文件
		tempFile, err := os.CreateTemp("", "kubeconfig-integration-test")
		require.NoError(t, err)
		defer os.Remove(tempFile.Name())

		kubeConfigContent := generateTestKubeConfigContent()

		_, err = tempFile.WriteString(kubeConfigContent)
		require.NoError(t, err)
		require.NoError(t, tempFile.Close())

		// 测试完整的工作流程
		cfg := NewKubernetesCfg(tempFile.Name())

		// 验证配置对象创建成功
		assert.NotNil(t, cfg)

		// 验证配置验证功能
		errs := cfg.Validate()
		assert.Empty(t, errs, "Valid config should have no validation errors")
	})

	t.Run("workflow with non-existing file", func(t *testing.T) {
		// 测试不存在文件的情况
		cfg := &KubernetesCfg{
			KubeConfigFile: "/non/existing/file",
			KubeConfig:     nil,
			QPS:            1e6,
			Burst:          1e6,
		}

		// 验证配置验证功能
		errs := cfg.Validate()
		const expectedLen = 2
		assert.Len(t, errs, expectedLen, "Should have 2 validation errors: file not found and nil config")
	})
}
