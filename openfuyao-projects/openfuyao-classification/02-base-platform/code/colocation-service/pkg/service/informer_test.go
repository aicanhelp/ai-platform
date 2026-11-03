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

package service

import (
	"os"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestInsertUsagePlugin 测试插入 usage 插件功能
func TestInsertUsagePlugin(t *testing.T) {
	tests := prepareInsertUsagePluginTestCases(t)
	runInsertUsagePluginTests(t, tests)
}

// TestRemoveUsagePlugin 测试移除 usage 插件功能
func TestRemoveUsagePlugin(t *testing.T) {
	tests := prepareRemoveUsagePluginTestCases()
	runRemoveUsagePluginTests(t, tests)
}

// prepareInsertUsagePluginTestCases 准备插入 usage 插件的测试用例
func prepareInsertUsagePluginTestCases(t *testing.T) []insertUsagePluginTestCase {
	defaultPlugin := `    - name: usage
      enablePredicate: false
`
	metricsStr := `metrics:
  - type: Prometheus
    address: "{PROMETHEUS_ENDPOINT}"
`

	return []insertUsagePluginTestCase{
		{
			name:          "usage plugin already exists",
			yamlStr:       createYAMLWithUsagePlugin(),
			defaultPlugin: defaultPlugin,
			metricsStr:    metricsStr,
			setupEnv:      func() { t.Setenv("PROMETHEUS_SERVICE", "http://prometheus:9090") },
			expectedYAML:  createYAMLWithUsagePlugin(),
		},
		{
			name:          "no plugins block",
			yamlStr:       "apiVersion: v1\nkind: Config\n",
			defaultPlugin: defaultPlugin,
			metricsStr:    metricsStr,
			setupEnv:      func() { t.Setenv("PROMETHEUS_SERVICE", "http://prometheus:9090") },
			expectedYAML:  "apiVersion: v1\nkind: Config\n",
		},
		{
			name:          "metrics already exists",
			yamlStr:       createYAMLWithMetrics(),
			defaultPlugin: defaultPlugin,
			metricsStr:    metricsStr,
			setupEnv:      func() { t.Setenv("PROMETHEUS_SERVICE", "http://prometheus:9090") },
			expectedYAML:  createYAMLWithUsagePluginAndMetrics(),
		},
		{
			name:          "prometheus service not provided",
			yamlStr:       createBasicYAML(),
			defaultPlugin: defaultPlugin,
			metricsStr:    metricsStr,
			setupEnv:      func() { os.Unsetenv("PROMETHEUS_SERVICE") },
			expectedYAML:  createYAMLWithUsagePlugin(),
		},
		{
			name:          "insert usage plugin successfully",
			yamlStr:       createBasicYAML(),
			defaultPlugin: defaultPlugin,
			metricsStr:    metricsStr,
			setupEnv:      func() { t.Setenv("PROMETHEUS_SERVICE", "http://prometheus:9090") },
			expectedYAML:  createYAMLWithUsagePluginAndMetrics(),
		},
	}
}

// prepareRemoveUsagePluginTestCases 准备移除 usage 插件的测试用例
func prepareRemoveUsagePluginTestCases() []removeUsagePluginTestCase {
	return []removeUsagePluginTestCase{
		{
			name:         "usage plugin not found",
			yamlStr:      createBasicYAML(),
			expectedYAML: createBasicYAML(),
		},
		{
			name:         "remove usage plugin successfully",
			yamlStr:      createYAMLWithUsagePlugin(),
			expectedYAML: createBasicYAML(),
		},
	}
}

// runInsertUsagePluginTests 运行插入 usage 插件的测试
func runInsertUsagePluginTests(t *testing.T, tests []insertUsagePluginTestCase) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runSingleInsertUsagePluginTest(t, tt)
		})
	}
}

// runRemoveUsagePluginTests 运行移除 usage 插件的测试
func runRemoveUsagePluginTests(t *testing.T, tests []removeUsagePluginTestCase) {
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			runSingleRemoveUsagePluginTest(t, tt)
		})
	}
}

// runSingleInsertUsagePluginTest 运行单个插入 usage 插件测试
func runSingleInsertUsagePluginTest(t *testing.T, tt insertUsagePluginTestCase) {
	// 设置环境变量
	if tt.setupEnv != nil {
		tt.setupEnv()
	}

	result := insertUsagePlugin(tt.yamlStr, tt.defaultPlugin, tt.metricsStr)
	assert.Equal(t, tt.expectedYAML, result)
}

// runSingleRemoveUsagePluginTest 运行单个移除 usage 插件测试
func runSingleRemoveUsagePluginTest(t *testing.T, tt removeUsagePluginTestCase) {
	result := removeUsagePlugin(tt.yamlStr)
	assert.Equal(t, tt.expectedYAML, result)
}

// 测试用例结构体定义
type insertUsagePluginTestCase struct {
	name          string
	yamlStr       string
	defaultPlugin string
	metricsStr    string
	setupEnv      func()
	expectedYAML  string
}

type removeUsagePluginTestCase struct {
	name         string
	yamlStr      string
	expectedYAML string
}

// YAML 字符串创建辅助函数
func createBasicYAML() string {
	return `apiVersion: v1
kind: Config
profiles:
  - plugins:
    - name: prometheus
`
}

func createYAMLWithUsagePlugin() string {
	return `apiVersion: v1
kind: Config
profiles:
  - plugins:
    - name: usage
      enablePredicate: false
    - name: prometheus
`
}

func createYAMLWithMetrics() string {
	return `apiVersion: v1
kind: Config
profiles:
  - plugins:
    - name: prometheus
metrics:
  - type: Prometheus
    address: "http://prometheus:9090"
`
}

func createYAMLWithUsagePluginAndMetrics() string {
	return `apiVersion: v1
kind: Config
profiles:
  - plugins:
    - name: usage
      enablePredicate: false
    - name: prometheus
metrics:
  - type: Prometheus
    address: "http://prometheus:9090"
`
}
