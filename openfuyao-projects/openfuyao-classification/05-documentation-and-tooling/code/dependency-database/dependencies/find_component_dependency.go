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
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

// getAllVersionsDependencies 完成模糊查询
func getAllVersionsDependencies(versions map[string][]string, deps []string, needDevelopmentVersion bool) []string {
	for key, v := range versions {
		// 判断是否为开发者版本
		if !needDevelopmentVersion && key == "0.0.0-latest" {
			continue
		}

		for _, dep := range v {
			// 判断是否为开发者版本
			if !needDevelopmentVersion && strings.HasSuffix(dep, "-0.0.0-latest") {
				continue
			}
			deps = append(deps, dep)
		}
	}
	return deps
}

// GetDependencies 获取指定组件的依赖
// 1. version 不为空，精确查询
// 2. version 为空，模糊查询
func (dm *DependencyMap) GetDependencies(name string, version string, needDevelopmentVersion bool) []string {
	versions, exists := dm.Component[name]
	if !exists {
		return nil
	}

	var deps []string // 初始化nil切片

	if strings.TrimSpace(version) == "" {
		deps = getAllVersionsDependencies(versions, deps, needDevelopmentVersion)
	} else {
		// 判断是否为开发者版本
		if !needDevelopmentVersion && version == "0.0.0-latest" {
			return deps
		}

		for _, dep := range versions[version] {
			// 判断是否为开发者版本
			if !needDevelopmentVersion && strings.HasSuffix(dep, "-0.0.0-latest") {
				continue
			}
			deps = append(deps, dep)
		}
	}
	return deps
}

// LoadComponentDependencies 从文件加载组件依赖
func LoadComponentDependencies(filePath string) (*DependencyMap, error) {
	// 确保文件路径正确
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		return nil, fmt.Errorf("获取绝对路径失败: %w", err)
	}

	// 检查文件是否存在
	if _, err = os.Stat(absPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("组件被依赖情况存储文件不存在: %s", absPath)
	}

	// 读取文件内容
	data, err := os.ReadFile(absPath)
	if err != nil {
		return nil, fmt.Errorf("读取组件被依赖情况存储文件失败: %w", err)
	}

	// 反序列化JSON
	var componentDependencyMap DependencyMap // 不要用指针，需要分配内存
	if err = json.Unmarshal(data, &componentDependencyMap); err != nil {
		return nil, fmt.Errorf("JSON文件反序列化失败: %w", err)
	}

	log.Printf("从文件加载组件被依赖情况成功")
	return &componentDependencyMap, nil
}

// FindComponentDependency 提供查询接口
func FindComponentDependency(name string, version string, needDevelopmentVersion bool,
	filePath string) ([]string, error) {
	// 从文件加载组件依赖
	componentDependencyMap, err := LoadComponentDependencies(filePath)
	if err != nil {
		return nil, fmt.Errorf("获取组件被依赖情况失败: %w", err)
	}

	// 获取指定组件的依赖
	componentDependencies := componentDependencyMap.GetDependencies(name, version, needDevelopmentVersion)

	log.Printf("查询组件被依赖情况成功")
	return componentDependencies, nil
}
