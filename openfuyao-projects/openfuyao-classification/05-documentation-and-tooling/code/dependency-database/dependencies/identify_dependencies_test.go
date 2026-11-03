// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package dependencies

import (
	"bytes"
	"encoding/json"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"gopkg.in/yaml.v2"
)

var validYamlData = `
apiVersion: v1
entries:
  ai-inference-integration:
    - apiVersion: v2
      appVersion: 0.20.0
      created: "2025-12-03T08:07:38.13380973Z"
      dependencies:
        - name: hermes-router
          repository: oci://cr.openfuyao.cn/charts
          version: 0.15.0
        - name: inference-backend
          repository: ""
          version: 0.20.0
        - name: cache-indexer
          repository: ""
          version: 0.15.0
      description: A Helm chart for openFuyao AI inference integration package based on vllm and mooncake.
      digest: 6e51de846489da6e2c719746201dc96d7b4733e2a4c7e5f7fa9d5a96a86791f3
      keywords:
        - artificial-intelligence
      name: ai-inference-integration
      type: application
      urls:
        - oci://cr.openfuyao.cn/charts/ai-inference-integration:0.20.0
      version: 0.20.0
generated: "2025-12-08T11:42:59.797507928Z"
`

var emptyEntriesYamlData = `
apiVersion: v1
entries: 
generated: "2025-12-08T11:42:59.797507928Z"
`

var missingURLsYamlData = `
apiVersion: v1
entries:
  ai-inference-integration:
    - apiVersion: v2
      appVersion: 0.20.0
      created: "2025-12-03T08:07:38.13380973Z"
      dependencies:
        - name: hermes-router
          repository: oci://cr.openfuyao.cn/charts
          version: 0.15.0
        - name: inference-backend
          repository: ""
          version: 0.20.0
        - name: cache-indexer
          repository: ""
          version: 0.15.0
      description: A Helm chart for openFuyao AI inference integration package based on vllm and mooncake.
      digest: 6e51de846489da6e2c719746201dc96d7b4733e2a4c7e5f7fa9d5a96a86791f3
      keywords:
        - artificial-intelligence
      name: ai-inference-integration
      type: application
      version: 0.20.0
generated: "2025-12-08T11:42:59.797507928Z"
`

var multipleYamlData = `
apiVersion: v1
entries:
  ai-inference-integration:
    - apiVersion: v2
      appVersion: 0.20.0
      created: "2025-12-03T08:07:38.13380973Z"
      dependencies:
        - name: hermes-router
          repository: oci://cr.openfuyao.cn/charts
          version: 0.15.0
        - name: inference-backend
          repository: ""
          version: 0.20.0
        - name: cache-indexer
          repository: ""
          version: 0.15.0
      description: A Helm chart for openFuyao AI inference integration package based on vllm and mooncake.
      digest: 6e51de846489da6e2c719746201dc96d7b4733e2a4c7e5f7fa9d5a96a86791f3
      keywords:
        - artificial-intelligence
      name: ai-inference-integration
      type: application
      urls:
        - oci://cr.openfuyao.cn/charts/ai-inference-integration:0.20.0
        - oci://localhost/charts/ai-inference-integration:0.20.0
      version: 0.20.0
generated: "2025-12-08T11:42:59.797507928Z"
`

// TestParseHelmIndexYAML 测试解析Helm索引YAML
func TestParseHelmIndexYAML(t *testing.T) {
	testCases := []struct {
		name        string
		yamlData    string
		wantCharts  []ChartInfo
		wantErr     bool
		errContains string
	}{
		{
			name:     "有效的 index.yaml",
			yamlData: validYamlData,
			wantCharts: []ChartInfo{
				{
					Component:  "ai-inference-integration",
					Version:    "0.20.0",
					URL:        "oci://cr.openfuyao.cn/charts/ai-inference-integration:0.20.0",
					AppVersion: "0.20.0",
				},
			},
			wantErr: false,
		},
		{
			name:       "entries 为空",
			yamlData:   emptyEntriesYamlData,
			wantCharts: []ChartInfo{},
			wantErr:    false,
		},
		{
			name:        "丢失 urls 字段",
			yamlData:    missingURLsYamlData,
			wantErr:     true,
			errContains: "helm包拉取地址不唯一",
		},
		{
			name:        "多个 urls 字段",
			yamlData:    multipleYamlData,
			wantErr:     true,
			errContains: "helm包拉取地址不唯一",
		},
		{
			name:        "无效的 yaml",
			yamlData:    "invalid: yaml: content",
			wantErr:     true,
			errContains: "YAML解析失败",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			charts, err := parseHelmIndexYAML([]byte(tc.yamlData))

			if tc.wantErr {
				require.Error(t, err)
				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, len(tc.wantCharts), len(charts))

				for i, wantChart := range tc.wantCharts {
					if i >= len(charts) {
						continue
					}
					assert.Equal(t, wantChart.Component, charts[i].Component)
					assert.Equal(t, wantChart.Version, charts[i].Version)
					assert.Equal(t, wantChart.URL, charts[i].URL)
					assert.Equal(t, wantChart.AppVersion, charts[i].AppVersion)
				}
			}
		})
	}
}

var singleYamlData = `
apiVersion: v1
entries:
  ai-inference-integration:
    - apiVersion: v2
      appVersion: 0.20.0
      created: "2025-12-03T08:07:38.13380973Z"
      dependencies:
        - name: hermes-router
          repository: oci://cr.openfuyao.cn/charts
          version: 0.15.0
        - name: inference-backend
          repository: ""
          version: 0.20.0
        - name: cache-indexer
          repository: ""
          version: 0.15.0
      description: A Helm chart for openFuyao AI inference integration package based on vllm and mooncake.
      digest: 6e51de846489da6e2c719746201dc96d7b4733e2a4c7e5f7fa9d5a96a86791f3
      keywords:
        - artificial-intelligence
      name: ai-inference-integration
      type: application
      urls:
        - oci://cr.openfuyao.cn/charts/ai-inference-integration:0.20.0
      version: 0.20.0
generated: "2025-12-08T11:42:59.797507928Z"
`

// TestFetchAndParseHelmIndex 测试HTTP请求和解析
func TestFetchAndParseHelmIndex(t *testing.T) {
	// 创建测试HTTP服务器
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/index.yaml" {
			yamlData := singleYamlData
			w.Header().Set("Content-Type", "application/yaml")
			w.WriteHeader(http.StatusOK)
			_, err := w.Write([]byte(yamlData))
			require.NoError(t, err, "撰写yamlData数据应该成功执行")
		} else {
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer testServer.Close()

	t.Run("successful fetch and parse", func(t *testing.T) {
		charts, err := fetchAndParseHelmIndex(testServer.URL + "/index.yaml")
		require.NoError(t, err)
		assert.Len(t, charts, 1)
		assert.Equal(t, "ai-inference-integration", charts[0].Component)
		assert.Equal(t, "0.20.0", charts[0].Version)
		assert.Equal(t, "oci://cr.openfuyao.cn/charts/ai-inference-integration:0.20.0", charts[0].URL)
		assert.Equal(t, "0.20.0", charts[0].AppVersion)
	})

	t.Run("server error", func(t *testing.T) {
		// 创建返回500错误的服务器
		errorServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer errorServer.Close()

		_, err := fetchAndParseHelmIndex(errorServer.URL)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "错误状态码")
	})
}

var validChartYamlContent = `
apiVersion: v2
appVersion: v25.06
dependencies:
- condition: "true"
  name: npu-operator
  repository: https://harbor.openfuyao.com/chartrepo/openfuyao-catalog-pre
  version: 0.14.0
- condition: "true"
  name: kuberay-operator
  repository: https://ray-project.github.io/kuberay-helm/
  version: 1.1.0
description: AI integrated deployment solution, supporting users to deploy custom
  AI inference services in the cluster with just one click
keywords:
- artificial-intelligence
name: aiaio-installer
version: 0.14.0

`

// TestExtractChartInfoFromYAML 测试从Chart.yaml提取信息
func TestExtractChartInfoFromYAML(t *testing.T) {
	// 创建临时目录
	tempDir, err := os.MkdirTemp("", "chart-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	testCases := []struct {
		name        string
		yamlContent string
		wantName    string
		wantVersion string
		wantErr     bool
	}{
		{
			name:        "有效的 chart.yaml",
			yamlContent: validChartYamlContent,
			wantName:    "aiaio-installer",
			wantVersion: "0.14.0",
			wantErr:     false,
		},
		{
			name:        "不存在 name 字段",
			yamlContent: `version: 1.0.0`,
			wantErr:     true,
		},
		{
			name:        "不存在 version 字段",
			yamlContent: `name: my-chart`,
			wantErr:     true,
		},
		{
			name:        "无效的 yaml",
			yamlContent: `name: my-chart: version:`,
			wantErr:     true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// 创建临时Chart.yaml文件
			chartPath := filepath.Join(tempDir, "Chart.yaml")
			var permission = 0644
			err := os.WriteFile(chartPath, []byte(tc.yamlContent), os.FileMode(permission))
			require.NoError(t, err)

			name, version, err := extractChartInfoFromYAML(chartPath)

			if tc.wantErr {
				require.Error(t, err)
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.wantName, name)
				assert.Equal(t, tc.wantVersion, version)
			}
		})
	}

	// 测试不存在的文件
	t.Run("文件不存在", func(t *testing.T) {
		_, _, err := extractChartInfoFromYAML(filepath.Join(tempDir, "nonexistent.yaml"))
		require.Error(t, err)
		assert.Contains(t, err.Error(), "读取 chart.yaml 失败")
	})
}

// TestContainsString 测试字符串包含检查
func TestContainsString(t *testing.T) {
	testCases := []struct {
		name     string
		slice    []string
		target   string
		expected bool
	}{
		{
			name:     "存在该元素",
			slice:    []string{"ai-inference-integration-0.20.0", "ai-inference-integration-0.15.0", "aiaio-installer-0.20.0"},
			target:   "ai-inference-integration-0.20.0",
			expected: true,
		},
		{
			name:     "不存在该元素",
			slice:    []string{"ai-inference-integration-0.20.0", "ai-inference-integration-0.15.0", "aiaio-installer-0.20.0"},
			target:   "aiaio-installer-0.15.0",
			expected: false,
		},
		{
			name:     "空字符串数组",
			slice:    []string{},
			target:   "ai-inference-integration-0.15.0",
			expected: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := containsString(tc.slice, tc.target)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// TestIsDigit 测试数字检测
func TestIsDigit(t *testing.T) {
	testCases := []struct {
		b        byte
		expected bool
	}{
		{'0', true},
		{'5', true},
		{'9', true},
		{'a', false},
		{'A', false},
		{'+', false},
		{'-', false},
		{'.', false},
	}

	for _, tc := range testCases {
		t.Run(string(tc.b), func(t *testing.T) {
			result := isDigit(tc.b)
			assert.Equal(t, tc.expected, result)
		})
	}
}

// TestSplitComponentAndVersionByParts 测试拆分组件名和版本号
func TestSplitComponentAndVersionByParts(t *testing.T) {
	testCases := []struct {
		name          string
		input         string
		wantComponent string
		wantVersion   string
		wantErr       bool
		errContains   string
	}{
		{
			name:          "简单的版本识别",
			input:         "nginx-1.2.3",
			wantComponent: "nginx",
			wantVersion:   "1.2.3",
			wantErr:       false,
		},
		{
			name:          "带v前缀的版本识别",
			input:         "gpu-operator-v25.3.2",
			wantComponent: "gpu-operator",
			wantVersion:   "v25.3.2",
			wantErr:       false,
		},
		{
			name:          "带V前缀的版本识别",
			input:         "gpu-operator-V25.3.2",
			wantComponent: "gpu-operator",
			wantVersion:   "V25.3.2",
			wantErr:       false,
		},
		{
			name:          "带-的组件名的版本识别",
			input:         "ai-inference-integration-0.20.0",
			wantComponent: "ai-inference-integration",
			wantVersion:   "0.20.0",
			wantErr:       false,
		},
		{
			name:          "开发者版本识别",
			input:         "aiaio-installer-0.0.0-latest",
			wantComponent: "aiaio-installer",
			wantVersion:   "0.0.0-latest",
			wantErr:       false,
		},
		{
			name:          "release版本识别",
			input:         "logging-package-0.0.0-release-1.0.x",
			wantComponent: "logging-package",
			wantVersion:   "0.0.0-release-1.0.x",
			wantErr:       false,
		},
		{
			name:        "不带版本号且不带-",
			input:       "nginx",
			wantErr:     true,
			errContains: "字符串格式错误",
		},
		{
			name:        "带-的组件名且不带版本号",
			input:       "colocation-operator",
			wantErr:     true,
			errContains: "无法识别版本号",
		},
		{
			name:        "空字符串",
			input:       "",
			wantErr:     true,
			errContains: "字符串格式错误",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			component, version, err := splitComponentAndVersionByParts(tc.input)

			if tc.wantErr {
				require.Error(t, err)
				if tc.errContains != "" {
					assert.Contains(t, err.Error(), tc.errContains)
				}
			} else {
				require.NoError(t, err)
				assert.Equal(t, tc.wantComponent, component)
				assert.Equal(t, tc.wantVersion, version)
			}
		})
	}
}

// TestEnsureDir 测试目录创建
func TestEnsureDir(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "ensure-dir-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	tests := []struct {
		name      string
		dirPath   string
		wantError bool
	}{
		{
			name:      "创建新目录",
			dirPath:   filepath.Join(tempDir, "newdir"),
			wantError: false,
		},
		{
			name:      "创建嵌套目录",
			dirPath:   filepath.Join(tempDir, "level1", "level2", "level3"),
			wantError: false,
		},
		{
			name:      "目录已经存在",
			dirPath:   tempDir,
			wantError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := ensureDir(tt.dirPath)

			assert.NoError(t, err)
			// 验证目录确实存在
			info, statErr := os.Stat(tt.dirPath)
			assert.NoError(t, statErr)
			assert.True(t, info.IsDir())
		})
	}
}

// TestDependencyMapAdd 测试DependencyMap的Add方法
func TestDependencyMapAdd(t *testing.T) {
	dm := NewDependencyMap([]ChartInfo{
		{Component: "ai-inference-integration", Version: "0.20.0"},
		{Component: "aiaio-installer", Version: "0.15.0"},
	})

	t.Run("向已有组件已有版本添加一个新被依赖项", func(t *testing.T) {
		dm.Add("ai-inference-integration", "0.20.0", "hermes-router-0.15.0")
		assert.Contains(t, dm.Component["ai-inference-integration"]["0.20.0"], "hermes-router-0.15.0")
		assert.Len(t, dm.Component["ai-inference-integration"]["0.20.0"], 1)
	})

	t.Run("向已有组件已有版本（已有被依赖项）添加一个新被依赖项", func(t *testing.T) {
		dm.Add("ai-inference-integration", "0.20.0", "inference-backend-0.20.0")
		assert.Contains(t, dm.Component["ai-inference-integration"]["0.20.0"], "hermes-router-0.15.0")
		assert.Contains(t, dm.Component["ai-inference-integration"]["0.20.0"], "inference-backend-0.20.0")
		assert.Len(t, dm.Component["ai-inference-integration"]["0.20.0"], 2)
	})

	t.Run("add to non-existent component version", func(t *testing.T) {
		dm.Add("logging-package", "0.0.0-latest", "logging-operator-0.0.0-latest")
		assert.Contains(t, dm.Component["logging-package"]["0.0.0-latest"], "logging-operator-0.0.0-latest")
		assert.Len(t, dm.Component["logging-package"]["0.0.0-latest"], 1)
	})
}

// TestFindAllDependencies 测试递归查找依赖
func TestFindAllDependencies(t *testing.T) {
	// 创建测试数据
	charts := []ChartInfo{
		{Component: "app", Version: "1.0.0"},
	}

	dcm := NewDependencyMap(charts)
	cdm := NewDependencyMap(charts)

	// 设置依赖关系: app -> dep1 -> dep2
	//                 -> dep3
	dcm.Add("app", "1.0.0", "dep1-1.0.0")
	dcm.Add("app", "1.0.0", "dep3-1.0.0")
	dcm.Add("dep1", "1.0.0", "dep2-1.0.0")

	// 设置被依赖关系
	cdm.Add("dep1", "1.0.0", "app-1.0.0")
	cdm.Add("dep3", "1.0.0", "app-1.0.0")
	cdm.Add("dep2", "1.0.0", "dep1-1.0.0")

	t.Run("递归依赖解析", func(t *testing.T) {
		updatedDM, updatedCDM, err := findAllDependencies(dcm, cdm, "app", "1.0.0")
		require.NoError(t, err)

		// 检查app应该包含所有依赖
		deps := updatedDM.Component["app"]["1.0.0"]
		assert.Contains(t, deps, "dep1-1.0.0")
		assert.Contains(t, deps, "dep3-1.0.0")
		assert.Contains(t, deps, "dep2-1.0.0") // 递归依赖

		// 检查被依赖关系
		assert.Contains(t, updatedCDM.Component["dep1"]["1.0.0"], "app-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep3"]["1.0.0"], "app-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep2"]["1.0.0"], "dep1-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep2"]["1.0.0"], "app-1.0.0") // 通过dep1间接依赖
		assert.Empty(t, updatedCDM.Component["app"]["1.0.0"])
	})

	// 设置依赖关系: app -> dep1 -> dep2
	//                 -> dep3 ->
	dcm.Add("dep3", "1.0.0", "dep2-1.0.0")

	// 设置被依赖关系
	cdm.Add("dep2", "1.0.0", "dep3-1.0.0")

	t.Run("复杂递归依赖解析", func(t *testing.T) {
		updatedDM, updatedCDM, err := findAllDependencies(dcm, cdm, "app", "1.0.0")
		require.NoError(t, err)

		// 检查app应该包含所有依赖
		deps := updatedDM.Component["app"]["1.0.0"]
		assert.Contains(t, deps, "dep1-1.0.0")
		assert.Contains(t, deps, "dep3-1.0.0")
		assert.Contains(t, deps, "dep2-1.0.0") // 递归依赖

		// 检查被依赖关系
		assert.Contains(t, updatedCDM.Component["dep1"]["1.0.0"], "app-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep3"]["1.0.0"], "app-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep2"]["1.0.0"], "dep1-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep2"]["1.0.0"], "dep3-1.0.0")
		assert.Contains(t, updatedCDM.Component["dep2"]["1.0.0"], "app-1.0.0") // 通过dep1、dep3间接依赖
		assert.Empty(t, updatedCDM.Component["app"]["1.0.0"])
	})

	t.Run("组件无依赖", func(t *testing.T) {
		updatedDCM, updatedCDM, err := findAllDependencies(dcm, cdm, "dep2", "1.0.0")
		require.NoError(t, err)

		assert.Equal(t, updatedDCM, dcm)
		assert.Equal(t, updatedCDM, cdm)
	})
}

// TestExpandMap 测试依赖图扩展
func TestExpandMap(t *testing.T) {
	// 创建测试图表
	charts := []ChartInfo{
		{Component: "app1", Version: "1.0.0"},
		{Component: "app2", Version: "1.0.0"},
	}

	// 创建依赖关系: app1 -> middleware -> base
	//             app2 ->
	dcm := NewDependencyMap(charts)
	cdm := NewDependencyMap(charts)

	dcm.Add("app1", "1.0.0", "middleware-2.0.0")
	dcm.Add("app2", "1.0.0", "middleware-2.0.0")
	dcm.Add("middleware", "2.0.0", "base-3.0.0")

	cdm.Add("middleware", "2.0.0", "app1-1.0.0")
	cdm.Add("middleware", "2.0.0", "app2-1.0.0")
	cdm.Add("base", "3.0.0", "middleware-2.0.0")

	t.Run("expand transitive dependencies", func(t *testing.T) {
		expandedDCM, expandedCDM, err := expandMap(dcm, cdm, charts)
		require.NoError(t, err)

		// 验证app的直接依赖
		assert.Contains(t, expandedDCM.Component["app1"]["1.0.0"], "middleware-2.0.0")
		assert.Contains(t, expandedDCM.Component["app1"]["1.0.0"], "base-3.0.0") // 间接依赖应该被添加
		assert.Contains(t, expandedDCM.Component["app2"]["1.0.0"], "middleware-2.0.0")
		assert.Contains(t, expandedDCM.Component["app2"]["1.0.0"], "base-3.0.0") // 间接依赖应该被添加
		assert.Contains(t, expandedDCM.Component["middleware"]["2.0.0"], "base-3.0.0")

		// 验证被依赖关系
		assert.Contains(t, expandedCDM.Component["base"]["3.0.0"], "middleware-2.0.0")
		assert.Contains(t, expandedCDM.Component["base"]["3.0.0"], "app1-1.0.0") // 间接被依赖
		assert.Contains(t, expandedCDM.Component["base"]["3.0.0"], "app2-1.0.0") // 间接被依赖
		assert.Contains(t, expandedCDM.Component["middleware"]["2.0.0"], "app1-1.0.0")
		assert.Contains(t, expandedCDM.Component["middleware"]["2.0.0"], "app2-1.0.0")
	})
}

// TestConvertToJson 测试JSON转换
func TestConvertToJson(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "json-test-*")
	require.NoError(t, err)
	defer os.RemoveAll(tempDir)

	dm := NewDependencyMap([]ChartInfo{
		{Component: "app1", Version: "1.0.0"},
		{Component: "app2", Version: "2.0.0"},
	})
	dm.Add("app1", "1.0.0", "dep1-1.0.0")
	dm.Add("app1", "1.0.0", "dep2-2.0.0")

	filePath := filepath.Join(tempDir, "test.json")

	t.Run("将 map 转换为 json 文件 - 目录已存在", func(t *testing.T) {
		err := convertToJson(dm, filePath)
		require.NoError(t, err)

		// 验证文件存在
		_, err = os.Stat(filePath)
		assert.NoError(t, err)

		// 读取并解析文件内容
		data, err := os.ReadFile(filePath)
		require.NoError(t, err)

		var parsedDM DependencyMap
		err = json.Unmarshal(data, &parsedDM)
		require.NoError(t, err)

		assert.Equal(t, dm.Component["app1"]["1.0.0"], parsedDM.Component["app1"]["1.0.0"])
		assert.Equal(t, dm.Component["app2"]["2.0.0"], parsedDM.Component["app2"]["2.0.0"])
	})

	t.Run("将 map 转换为 json 文件 - 目录不存在", func(t *testing.T) {
		// 尝试写入到不存在的目录
		err := convertToJson(dm, "/nonexistent/dir/file.json")
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "创建文件失败")
	})
}

// TestDeleteDirectory 测试目录删除
func TestDeleteDirectory(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "delete-test-*")
	require.NoError(t, err)

	// 创建一些测试文件和子目录
	var createPermission = 0755
	err = os.MkdirAll(filepath.Join(tempDir, "subdir"), os.FileMode(createPermission))
	require.NoError(t, err)

	var writePermission = 0644
	err = os.WriteFile(filepath.Join(tempDir, "file1.txt"), []byte("test"), os.FileMode(writePermission))
	require.NoError(t, err)

	err = os.WriteFile(filepath.Join(tempDir, "subdir", "file2.txt"), []byte("test"), os.FileMode(writePermission))
	require.NoError(t, err)

	t.Run("删除存在的目录", func(t *testing.T) {
		err := deleteDirectory(tempDir)
		assert.NoError(t, err)

		// 验证目录已被删除
		_, err = os.Stat(tempDir)
		assert.Error(t, err)
		assert.True(t, os.IsNotExist(err))
	})

	t.Run("删除不存在的目录", func(t *testing.T) {
		err := deleteDirectory("/nonexistent/path")
		assert.NoError(t, err) // 当前实现返回nil
	})
}

// TestPrintChartTable 测试表格打印
func TestPrintChartTable(t *testing.T) {
	charts := []ChartInfo{
		{Component: "app1", Version: "1.0.0", URL: "http://example.com/app1:1.0.0.tgz", AppVersion: "1.0"},
		{Component: "app2", Version: "2.0.0", URL: "http://example.com/app2:2.0.0.tgz", AppVersion: "2.0"},
	}

	// 重定向日志输出
	var buf bytes.Buffer
	log.SetOutput(&buf)
	defer func() {
		log.SetOutput(os.Stderr)
	}()

	t.Run("测试预期输出", func(t *testing.T) {
		printChartTable(charts)

		output := buf.String()
		assert.Contains(t, output, "app1")
		assert.Contains(t, output, "app2")
		assert.Contains(t, output, "1.0.0")
		assert.Contains(t, output, "2.0.0")
		assert.Contains(t, output, "http://example.com/app1:1.0.0.tgz")
		assert.Contains(t, output, "http://example.com/app2:2.0.0.tgz")
		assert.Contains(t, output, "1.0")
		assert.Contains(t, output, "2.0")
	})

	t.Run("测试空图表列表", func(t *testing.T) {
		buf.Reset()
		printChartTable([]ChartInfo{})

		output := buf.String()
		assert.Contains(t, output, "未找到任何组件信息")
	})
}

// 测试ChartInfo结构体
func TestChartInfoCreation(t *testing.T) {
	// 直接创建 ChartInfo 结构体
	chartInfo := ChartInfo{
		Component:  "test-chart",
		Version:    "1.2.3",
		URL:        "http://example.com/test-chart-1.2.3.tgz",
		AppVersion: "0.20.0",
	}

	assert.Equal(t, "test-chart", chartInfo.Component)
	assert.Equal(t, "1.2.3", chartInfo.Version)
	assert.Equal(t, "http://example.com/test-chart-1.2.3.tgz", chartInfo.URL)
	assert.Equal(t, "0.20.0", chartInfo.AppVersion)
}

var chartMetadataYaml = `
name: test-chart
version: 1.2.3
description: A test chart for ChartMetadata
apiVersion: v2
`

// TestChartMetadata 测试ChartMetadata结构体
func TestChartMetadata(t *testing.T) {
	yamlContent := chartMetadataYaml

	var metadata ChartMetadata
	err := yaml.Unmarshal([]byte(yamlContent), &metadata)
	require.NoError(t, err)

	assert.Equal(t, "test-chart", metadata.Name)
	assert.Equal(t, "1.2.3", metadata.Version)
}
