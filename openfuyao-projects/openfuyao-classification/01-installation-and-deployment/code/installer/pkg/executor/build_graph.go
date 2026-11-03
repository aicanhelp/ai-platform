/******************************************************************
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package executor

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"

	"installer/pkg/common"
	"installer/pkg/constants"
	"installer/pkg/executor/core"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/metadata"
	"installer/pkg/zlog"
)

const (
	RightParenthesis = ")"
	Dots             = "."
	Comma            = ","
	VersionLength    = 3
)

// GraphBuilder 构建图
type GraphBuilder struct {
	// 组件名与组件数据map
	NameToComponentMap map[string]fuyaov1beta1.Component
}

// NewGraphBuilder is new GraphBuilder
func NewGraphBuilder() GraphBuilder {
	return GraphBuilder{
		NameToComponentMap: make(map[string]fuyaov1beta1.Component),
	}
}

func (g *GraphBuilder) Builder(components []fuyaov1beta1.Component, destPath string) (*core.DirectedGraph, error) {
	graph := &core.DirectedGraph{}
	if len(components) == 0 {
		zlog.Info("The components is empty.")
		return graph, nil
	}

	for _, component := range components {
		metaDataPath := filepath.Join(destPath, component.Group, component.Name, constants.MetaDataFile)
		metaData := &metadata.MetaData{}
		if err := common.UnmarshalJSON[*metadata.MetaData](metaData, metaDataPath); err != nil {
			return nil, err
		}
		component.BootstrapNode = metaData.BootstrapNode
		component.Version = metaData.Version

		g.NameToComponentMap[component.Name] = component
		var vertex = &core.Vertex{
			Data:      component,
			OutDegree: 0,
			InDegree:  0,
		}
		// 图中加入顶点
		graph.AddVertex(vertex)
	}

	return graph, nil
}

func (g *GraphBuilder) minMaxVersionCheck(versionSlice []string, minVersionSlice []string, maxVersionSlice []string) (bool, string) {
	length := len(versionSlice)
	minLength := len(minVersionSlice)
	maxLength := len(maxVersionSlice)
	if length != VersionLength || minLength != VersionLength || maxLength != VersionLength {
		return false, fmt.Sprintf("Version information incorrect, unable to verify, version %s, min version %s, max version %s", versionSlice, minVersionSlice, maxVersionSlice)
	}

	minPass, maxPass := false, false
	for i := 0; i < length; i++ {
		version, err := strconv.Atoi(strings.TrimSpace(versionSlice[i]))
		if err != nil {
			return false, "component version incorrect"
		}
		minVersion, err := strconv.Atoi(strings.TrimSpace(minVersionSlice[i]))
		if err != nil {
			return false, "component version incorrect"
		}
		maxVersion, err := strconv.Atoi(strings.TrimSpace(maxVersionSlice[i]))
		if err != nil {
			return false, "component version incorrect"
		}

		if !minPass {
			if version == minVersion {
				minPass = false
			} else if version > minVersion {
				minPass = true
			} else {
				return false, "component version mismatch"
			}
		}

		if !maxPass {
			if version == maxVersion {
				maxPass = false
			} else if version < maxVersion {
				maxPass = true
			} else {
				return false, "component version mismatch"
			}
		}
	}

	return true, ""
}

func (g *GraphBuilder) minVersionCheck(versionSlice []string, minVersionSlice []string) (bool, string) {
	// latest版本肯定大于最小依赖版本
	if len(versionSlice) == 1 && versionSlice[0] == constants.LatestVersion {
		return true, ""
	}

	length := len(versionSlice)
	minLength := len(minVersionSlice)
	if length != VersionLength || minLength != VersionLength {
		return false, fmt.Sprintf("Version information incorrect, unable to verify, version %s, min version %s", versionSlice, minVersionSlice)
	}

	for i := 0; i < length; i++ {
		version, err := strconv.Atoi(strings.TrimSpace(versionSlice[i]))
		if err != nil {
			return false, "component version incorrect"
		}
		minVersion, err := strconv.Atoi(strings.TrimSpace(minVersionSlice[i]))
		if err != nil {
			return false, "component version incorrect"
		}
		if version == minVersion {
			continue
		} else if version > minVersion {
			return true, ""
		}
		if version < minVersion {
			return false, "component version mismatch"
		}
	}
	return true, ""
}

// isVersionSatisfy 组件版本是否满足
func (g *GraphBuilder) isVersionSatisfy(depend metadata.Dependency) (bool, string) {
	dependComponent := g.NameToComponentMap[depend.Name]
	curVersionSlice := strings.Split(dependComponent.Version, Dots)

	// depend.Version示例:[1.27.3, 1.30.0] [1.2.0,)，均为标准三位版本号
	versionSlice := strings.Split(depend.Version, Comma)
	minVersionSlice := strings.Split(versionSlice[0][1:], Dots)

	if len(versionSlice) > 1 && strings.Contains(versionSlice[1], RightParenthesis) {
		return g.minVersionCheck(curVersionSlice, minVersionSlice)
	} else {
		maxVersionSlice := strings.Split(versionSlice[1][:len(versionSlice[1])-1], Dots)
		return g.minMaxVersionCheck(curVersionSlice, minVersionSlice, maxVersionSlice)
	}
}

// InstallGraphBuilder 构建安装有向无环图
func (g *GraphBuilder) InstallGraphBuilder(components []fuyaov1beta1.Component,
	node *fuyaov1beta1.Node) (*core.DirectedGraph, error) {

	pkgPath, err := common.GetOptPkgPathByNode(node)
	if err != nil {
		return nil, err
	}

	graph, err := g.Builder(components, pkgPath)
	if err != nil {
		return nil, err
	}

	for _, component := range components {
		metaDataPath := filepath.Join(pkgPath, component.Group, component.Name, constants.MetaDataFile)
		metaData := &metadata.MetaData{}
		if err := common.UnmarshalJSON[*metadata.MetaData](metaData, metaDataPath); err != nil {
			return nil, err
		}
		for _, depend := range metaData.Dependencies {
			if !g.isComponentExit(depend.Name) {
				zlog.Warn("The depend component is not exist: ", depend.Name)
				continue
			}

			ok, msg := g.isVersionSatisfy(depend)
			if !ok {
				zlog.Fatalf(depend.Name, " does not satisfy dependency, error is ", msg)
			}

			dependComponent := g.NameToComponentMap[depend.Name]
			// 增加依赖
			graph.AddEdgeByData(dependComponent.Name, component.Name)
		}
	}

	zlog.Info("Complete the construction of an install directed graph")
	return graph, nil
}

// UninstallGraphBuilder 构建卸载有向无环图
func (g *GraphBuilder) UninstallGraphBuilder(components []fuyaov1beta1.Component,
	node *fuyaov1beta1.Node) (*core.DirectedGraph, error) {

	fmt.Println("[prepare] Generate a directed acyclic graph")
	destPath, err := common.GetOptPkgPathByNode(node)
	if err != nil {
		return nil, err
	}

	graph, err := g.Builder(components, destPath)
	if err != nil {
		return nil, err
	}

	for _, component := range components {
		metaDataPath := filepath.Join(destPath, component.Group, component.Name, constants.MetaDataFile)
		metaData := &metadata.MetaData{}
		if err := common.UnmarshalJSON[*metadata.MetaData](metaData, metaDataPath); err != nil {
			return nil, err
		}
		for _, depend := range metaData.Dependencies {
			if !g.isComponentExit(depend.Name) {
				zlog.Warn("The depend component is not exist: ", depend.Name)
				continue
			}
			dependComponent := g.NameToComponentMap[depend.Name]
			// 增加依赖
			graph.AddEdgeByData(component.Name, dependComponent.Name)
		}
	}

	zlog.Info("Complete the construction of an uninstall directed graph")
	return graph, err
}

// isComponentExit 组件是否存在
func (g *GraphBuilder) isComponentExit(name string) bool {
	for target, _ := range g.NameToComponentMap {
		if target == name {
			return true
		}
	}

	return false
}
