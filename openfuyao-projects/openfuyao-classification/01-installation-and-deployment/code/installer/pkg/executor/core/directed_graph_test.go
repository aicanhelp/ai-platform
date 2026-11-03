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

package core_test

import (
	"fmt"
	"testing"

	"installer/pkg/executor/core"
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/fuyao/v1beta1/metadata"
)

func TestDirectedGraph(t *testing.T) {
	// mock component
	component1 := fuyaov1beta1.Component{
		Name: "1",
	}
	component2 := fuyaov1beta1.Component{
		Name: "2",
	}
	component3 := fuyaov1beta1.Component{
		Name: "3",
	}
	component4 := fuyaov1beta1.Component{
		Name: "4",
	}
	components := []fuyaov1beta1.Component{component1, component2, component3, component4}

	// mock metadata
	depend := metadata.Dependency{
		Name: "4",
	}
	depend2 := metadata.Dependency{
		Name: "2",
	}
	// 1依赖于4
	metaData1 := metadata.MetaData{
		Name:         "1",
		Dependencies: []metadata.Dependency{depend},
	}
	metaData2 := metadata.MetaData{
		Name:         "2",
		Dependencies: []metadata.Dependency{},
	}
	// 1依赖于4
	metaData3 := metadata.MetaData{
		Name:         "3",
		Dependencies: []metadata.Dependency{depend},
	}
	// 4依赖于2
	metaData4 := metadata.MetaData{
		Name:         "4",
		Dependencies: []metadata.Dependency{depend2},
	}
	// 组件的名字不会重复
	name2MetaData := make(map[string]metadata.MetaData)
	name2MetaData[component1.Name] = metaData1
	name2MetaData[component2.Name] = metaData2
	name2MetaData[component3.Name] = metaData3
	name2MetaData[component4.Name] = metaData4

	// 构建图
	graph := &core.DirectedGraph{}
	NameToComponentMap := make(map[string]fuyaov1beta1.Component)

	// 图中加入顶点
	for _, component := range components {
		NameToComponentMap[component.Name] = component
		var vertex = &core.Vertex{
			Data:      component,
			OutDegree: 0,
			InDegree:  0,
		}
		graph.AddVertex(vertex)
	}

	// 构建图的边，1依赖于4，3依赖于4， 4依赖于2
	for _, component := range components {
		metaData := name2MetaData[component.Name]
		for _, depend = range metaData.Dependencies {
			// 增加依赖
			dependComponent := NameToComponentMap[depend.Name]
			graph.AddEdgeByData(component.Name, dependComponent.Name)
		}
	}

	if graph.IsEmpty() {
		t.Error("build graph error")
	}

	// 第一批入度为零的顶点有2个
	zeroVertexList := graph.ZeroInDegreeVertexes()
	fmt.Println(zeroVertexList)
	if len(zeroVertexList) != 2 {
		t.Error("Error in obtaining vertices")
	}

	// 第二批入度为零的顶点有1个
	var vertexList []*core.Vertex
	for _, vertex := range zeroVertexList {
		fmt.Println(vertex.Data.Name)
		vertexes := graph.NextZeroInDegreeVertexByVertex(vertex)
		vertexList = append(vertexList, vertexes...)
	}
	if len(vertexList) != 1 {
		t.Error("Error in obtaining vertices")
	}
	// 顶点为顶点4
	if vertexList[0].Data.Name != component4.Name {
		t.Error("build graph error")
	}

	// 第三批入度为零的顶点有1个
	var vertexList3 []*core.Vertex
	for _, vertex := range vertexList {
		vertexes := graph.NextZeroInDegreeVertexByVertex(vertex)
		vertexList3 = append(vertexList3, vertexes...)
	}
	if len(vertexList3) != 1 {
		t.Error("Error in obtaining vertices")
	}
	// 顶点为顶点2
	if vertexList3[0].Data.Name != component2.Name {
		t.Error("build graph error")
	}

	// 最后一批，没有顶点了
	var vertexList4 []*core.Vertex
	for _, vertex := range vertexList3 {
		vertexes := graph.NextZeroInDegreeVertexByVertex(vertex)
		vertexList4 = append(vertexList4, vertexes...)
	}
	if len(vertexList4) != 0 {
		t.Error("Error in obtaining vertices")
	}
}

func TestMod(t *testing.T) {
	rwx := 0777
	fmt.Println(fmt.Sprintf("chmod %o", rwx))
}
