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
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// 测试LoadComponentDependencies - 正常情况
func TestLoadComponentDependenciesSuccess(t *testing.T) {
	// 准备测试数据
	testData := `{
        "Component": {
            "componentA": {
                "1.0.0": ["component1", "component2"],
                "0.15.0": ["component3"]
            },
            "componentB": {
                "1.0.1": ["component4", "component5", "component6"]
            }
        }
    }`

	// 创建临时文件
	tempFile := createTempFile(t, testData)

	// 执行函数
	result, err := LoadComponentDependencies(tempFile)

	// 验证结果
	require.NoError(t, err, "期望没有错误")
	require.NotNil(t, result, "期望返回非nil结果")

	// 验证数据结构
	var expectedComponentNum = 2
	assert.Equal(t, expectedComponentNum, len(result.Component), "期望有2个组件")

	// 验证componentA
	compA, compAExists := result.Component["componentA"]
	require.True(t, compAExists, "期望包含componentA")
	var expectedAVersionNum = 2
	assert.Equal(t, expectedAVersionNum, len(compA), "componentA期望有2个版本")

	// 验证componentA的服务
	assert.ElementsMatch(t, []string{"component1", "component2"}, compA["1.0.0"])
	assert.ElementsMatch(t, []string{"component3"}, compA["0.15.0"])

	// 验证componentB
	compB, compBExists := result.Component["componentB"]
	require.True(t, compBExists, "期望包含componentB")
	var expectedBVersionNum = 1
	assert.Equal(t, expectedBVersionNum, len(compB), "componentB期望有1个版本")
	assert.ElementsMatch(t, []string{"component4", "component5", "component6"}, compB["1.0.1"])
}

// 测试LoadComponentDependencies - 文件不存在
func TestLoadComponentDependenciesFileNotExists(t *testing.T) {
	// 使用一个不存在的文件路径
	nonExistentFile := filepath.Join(t.TempDir(), "nonexistent.json")

	// 执行函数
	result, err := LoadComponentDependencies(nonExistentFile)

	// 验证结果
	assert.Error(t, err, "期望返回错误")
	assert.Nil(t, result, "期望返回nil结果")

	// 验证错误消息包含预期内容
	assert.Contains(t, err.Error(), "组件被依赖情况存储文件不存在")
}

// 测试GetDependencies方法 - 精确查询（有版本号）
func TestDependencyMapGetDependenciesExactVersion(t *testing.T) {
	dm := &DependencyMap{
		Component: map[string]map[string][]string{
			"componentA": {
				"1.0.0":  []string{"component1", "component2"},
				"0.15.0": []string{"component3", "component4", "component5"},
			},
			"componentB": {
				"0.0.0-latest": []string{"component6", "component7"},
				"1.0.1":        []string{"component8"},
			},
		},
	}

	testCases := []struct {
		name                   string
		componentName          string
		version                string
		needDevelopmentVersion bool
		expectedDependencies   []string
		description            string
	}{
		{
			name:                   "精确查询componentA 1.0.0",
			componentName:          "componentA",
			version:                "1.0.0",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component1", "component2"},
			description:            "查询 componentA 特定版本 1.0.0",
		},
		{
			name:                   "精确查询componentA 0.15.0",
			componentName:          "componentA",
			version:                "0.15.0",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component3", "component4", "component5"},
			description:            "查询 componentA 特定版本 0.15.0",
		},
		{
			name:                   "精确查询componentB 1.0.1",
			componentName:          "componentB",
			version:                "1.0.1",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component8"},
			description:            "查询 componentB 特定版本 1.0.1",
		},
		{
			name:                   "精确查询不存在的版本",
			componentName:          "componentA",
			version:                "2.0.0",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "查询不存在的版本应返回nil",
		},
		{
			name:                   "精确查询不存在的组件",
			componentName:          "non-existent",
			version:                "1.0.0",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "查询不存在的组件应返回nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := dm.GetDependencies(tc.componentName, tc.version, tc.needDevelopmentVersion)

			if tc.expectedDependencies == nil {
				assert.Nil(t, result, "%s: 应返回nil", tc.description)
			} else {
				assert.NotNil(t, result, "%s: 应返回非nil", tc.description)
				assert.ElementsMatch(t, tc.expectedDependencies, result, "%s: 依赖列表不匹配", tc.description)
			}
		})
	}
}

// 测试GetDependencies方法 - 模糊查询（无版本号）
func TestDependencyMapGetDependenciesFuzzyQuery(t *testing.T) {
	dm := &DependencyMap{
		Component: map[string]map[string][]string{
			"componentA": {
				"1.0.0":  []string{"component1", "component2"},
				"0.15.0": []string{"component3", "component4", "component5"},
			},
			"componentB": {
				"0.0.0-latest": []string{"component6", "component7"},
				"1.0.1":        []string{"component8"},
			},
			"componentC": {
				"1.0.0": []string{},
			},
		},
	}

	testCases := []struct {
		name                   string
		componentName          string
		version                string
		needDevelopmentVersion bool
		expectedDependencies   []string
		description            string
	}{
		{
			name:                   "模糊查询componentA所有版本",
			componentName:          "componentA",
			version:                "",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component1", "component2", "component3", "component4", "component5"},
			description:            "模糊查询应返回所有版本的依赖",
		},
		{
			name:                   "模糊查询componentC",
			componentName:          "componentC",
			version:                "",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "模糊查询无依赖的服务应返回nil",
		},
		{
			name:                   "模糊查询不存在的组件",
			componentName:          "non-existent",
			version:                "",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "查询不存在的组件应返回nil",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := dm.GetDependencies(tc.componentName, tc.version, tc.needDevelopmentVersion)

			if tc.expectedDependencies == nil {
				assert.Nil(t, result, "%s: 应返回nil", tc.description)
			} else {
				assert.NotNil(t, result, "%s: 应返回非nil", tc.description)
				assert.ElementsMatch(t, tc.expectedDependencies, result, "%s: 依赖列表不匹配", tc.description)
			}
		})
	}
}

// 测试GetDependencies方法 - 开发者版本过滤
func TestDependencyMapGetDependenciesDevelopmentVersionFilter(t *testing.T) {
	dm := &DependencyMap{
		Component: map[string]map[string][]string{
			"componentA": {
				"0.0.0-latest": []string{
					"component1-1.0.1",
					"component2-0.13.0",
					"component3-0.0.0-latest",
					"component4-0.0.0-latest",
				},
				"1.0.0": []string{"component5-0.15.0", "component6-0.14.0"},
				"0.1.0": []string{"component7-0.12.0", "component8-0.0.0-latest"},
			},
		},
	}

	testCases := []struct {
		name                   string
		componentName          string
		version                string
		needDevelopmentVersion bool
		expectedDependencies   []string
		description            string
	}{
		{
			name:                   "精确查询开发者版本-不要开发者版本",
			componentName:          "componentA",
			version:                "0.0.0-latest",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "needDevelopmentVersion=false时应过滤掉0.0.0-latest版本对应的被依赖",
		},
		{
			name:                   "精确查询开发者版本-需要开发者版本",
			componentName:          "componentA",
			version:                "0.0.0-latest",
			needDevelopmentVersion: true,
			expectedDependencies: []string{
				"component1-1.0.1",
				"component2-0.13.0",
				"component3-0.0.0-latest",
				"component4-0.0.0-latest",
			},
			description: "needDevelopmentVersion=true时应包含所有被依赖",
		},
		{
			name:                   "精确查询非开发者版本-不要开发者版本-依赖中无开发者版本",
			componentName:          "componentA",
			version:                "1.0.0",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component5-0.15.0", "component6-0.14.0"},
			description:            "生产版本没有开发者依赖，应返回所有",
		},
		{
			name:                   "精确查询非开发者版本-不要开发者版本-依赖中有开发者版本",
			componentName:          "componentA",
			version:                "0.1.0",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component7-0.12.0"},
			description:            "生产版本有开发者依赖，应进行过滤",
		},
		{
			name:                   "模糊查询-不要开发者版本",
			componentName:          "componentA",
			version:                "",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component5-0.15.0", "component6-0.14.0", "component7-0.12.0"},
			description:            "模糊查询中有开发者依赖，应进行过滤",
		},
		{
			name:                   "模糊查询-需要开发者版本",
			componentName:          "componentA",
			version:                "",
			needDevelopmentVersion: true,
			expectedDependencies: []string{
				"component1-1.0.1",
				"component2-0.13.0",
				"component3-0.0.0-latest",
				"component4-0.0.0-latest",
				"component5-0.15.0",
				"component6-0.14.0",
				"component7-0.12.0",
				"component8-0.0.0-latest",
			},
			description: "模糊查询中应包含所有被依赖",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := dm.GetDependencies(tc.componentName, tc.version, tc.needDevelopmentVersion)

			if tc.expectedDependencies == nil {
				assert.Nil(t, result, "%s: 应返回nil", tc.description)
			} else {
				assert.NotNil(t, result, "%s: 应返回非nil", tc.description)
				assert.ElementsMatch(t, tc.expectedDependencies, result, "%s: 依赖列表不匹配", tc.description)
			}
		})
	}
}

// 测试FindComponentDependency - 成功查询
func TestFindComponentDependencySuccess(t *testing.T) {
	// 设置测试环境
	filePath, cleanup := setupTestEnvironment(t)
	defer cleanup()

	// 创建测试数据文件
	testData := `{
        "Component": {
            "componentA": {
                "1.0.1": ["component1-1.0.1", "component2-0.15.0"],
                "0.13.0": ["component3-1.0.1", "component4-0.12.0", "component5-0.0.0-latest"]
            },
            "componentB": {
				"0.0.0-latest": ["component6-1.0.1", "component7-0.0.0-latest", "component8-0.0.0-latest"],
                "1.1.0": ["component9-1.1.0", "component10-0.13.0"]
            }
        }
    }`

	createTestDataFile(t, filePath, testData)

	testCases := []struct {
		name                   string
		componentName          string
		version                string
		needDevelopmentVersion bool
		expectedDependencies   []string
		description            string
	}{
		{
			name:                   "查询componentA 1.0.1",
			componentName:          "componentA",
			version:                "1.0.1",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component1-1.0.1", "component2-0.15.0"},
			description:            "成功查询特定版本",
		},
		{
			name:                   "查询componentB 0.0.0-latest版本，不要开发者依赖",
			componentName:          "componentB",
			version:                "0.0.0-latest",
			needDevelopmentVersion: false,
			expectedDependencies:   nil,
			description:            "过滤掉开发者依赖",
		},
		{
			name:                   "查询componentB 0.0.0-latest版本，需要开发者依赖",
			componentName:          "componentB",
			version:                "0.0.0-latest",
			needDevelopmentVersion: true,
			expectedDependencies:   []string{"component6-1.0.1", "component7-0.0.0-latest", "component8-0.0.0-latest"},
			description:            "包含开发者依赖",
		},
		{
			name:                   "查询componentB 全部版本，不要开发者依赖",
			componentName:          "componentB",
			version:                "",
			needDevelopmentVersion: false,
			expectedDependencies:   []string{"component9-1.1.0", "component10-0.13.0"},
			description:            "模糊查询，不包含开发者依赖",
		},
		{
			name:                   "查询componentA 全部版本，需要开发者依赖",
			componentName:          "componentA",
			version:                "",
			needDevelopmentVersion: true,
			expectedDependencies: []string{
				"component1-1.0.1",
				"component2-0.15.0",
				"component3-1.0.1",
				"component4-0.12.0",
				"component5-0.0.0-latest",
			},
			description: "模糊查询，包含开发者依赖",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := FindComponentDependency(
				tc.componentName,
				tc.version,
				tc.needDevelopmentVersion,
				filePath,
			)

			assert.NoError(t, err, "%s: 期望没有错误", tc.description)
			if tc.expectedDependencies == nil {
				assert.Nil(t, result, "%s: 应返回nil", tc.description)
			} else {
				assert.NotNil(t, result, "%s: 应返回非nil", tc.description)
				assert.ElementsMatch(t, tc.expectedDependencies, result, "%s: 依赖列表不匹配", tc.description)
			}
		})
	}
}

// 测试辅助函数：创建临时测试文件
func createTempFile(t *testing.T, content string) string {
	t.Helper()

	tempDir := t.TempDir()
	tempFile := filepath.Join(tempDir, "dependencies.json")

	var permission = 0644
	err := os.WriteFile(tempFile, []byte(content), os.FileMode(permission))
	require.NoError(t, err, "创建临时文件失败")

	return tempFile
}

// 测试辅助函数：创建临时测试环境 - 避免使用常量中定义的路径，将各测试环境隔离
func setupTestEnvironment(t *testing.T) (string, func()) {
	t.Helper()

	// 保存当前工作目录
	originalDir, err := os.Getwd()
	require.NoError(t, err, "获取当前工作目录失败")

	// 创建临时目录
	tempDir := t.TempDir()

	// 切换到临时目录
	err = os.Chdir(tempDir)
	require.NoError(t, err, "切换工作目录失败")

	// 创建json目录结构
	jsonDir := "../json" // 避免使用代码文件中定义的常量中的./json路径，而是使用这里mock的./json路径
	var permission = 0755
	err = os.MkdirAll(jsonDir, os.FileMode(permission))
	require.NoError(t, err, "创建json目录失败")

	// 返回清理函数，在测试结束后恢复原始状态
	cleanup := func() {
		// 恢复原始工作目录
		err = os.Chdir(originalDir)
		assert.NoError(t, err, "恢复工作目录失败")
	}

	fileName := "componentDependencyFile.json"
	return filepath.Join(jsonDir, fileName), cleanup
}

// 测试辅助函数：创建测试数据文件
func createTestDataFile(t *testing.T, filePath string, content string) {
	t.Helper()

	// 确保目录存在
	dir := filepath.Dir(filePath)
	var createPermission = 0755
	err := os.MkdirAll(dir, os.FileMode(createPermission))
	require.NoError(t, err, "创建目录失败")

	// 写入文件
	var writePermission = 0644
	err = os.WriteFile(filePath, []byte(content), os.FileMode(writePermission))
	require.NoError(t, err, "写入测试文件失败")
}
