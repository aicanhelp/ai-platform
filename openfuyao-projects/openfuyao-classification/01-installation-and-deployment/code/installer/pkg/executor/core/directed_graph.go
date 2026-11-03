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

package core

import (
	fuyaov1beta1 "installer/pkg/fuyao/v1beta1/cluster"
	"installer/pkg/zlog"
)

// DirectedGraph 有向图
type DirectedGraph struct {
	// 顶点集
	vertexMap map[string]*Vertex
	// 边集
	edgeSet []*Edge
	// 出边缓存
	outEdgeMap map[*Vertex][]*Edge
	// 入边缓存
	inEdgeMap map[*Vertex][]*Edge
}

// IsEmpty 图是否为空
func (d *DirectedGraph) IsEmpty() bool {
	if len(d.vertexMap) == 0 {
		return true
	}
	return false
}

// GetEdgeCount 获取边的数量
func (d *DirectedGraph) GetEdgeCount() int {
	return len(d.edgeSet)
}

// GetVertex 获取顶点集
func (d *DirectedGraph) GetVertex() []*Vertex {
	var vertexSlice []*Vertex
	for _, value := range d.vertexMap {
		vertexSlice = append(vertexSlice, value)
	}
	return vertexSlice
}

// ZeroInDegreeVertexes 获取所有入度为0的顶点
func (d *DirectedGraph) ZeroInDegreeVertexes() []*Vertex {
	var resultSlice []*Vertex
	for _, vertex := range d.vertexMap {
		if 0 == vertex.InDegree {
			resultSlice = append(resultSlice, vertex)
		}
	}
	return resultSlice
}

// NextZeroInDegreeVertexByData 通过组件获取下一批入度为0的节点，同时删除当前顶点
// name 组件的名字
func (d *DirectedGraph) NextZeroInDegreeVertexByData(name string) []*Vertex {
	vertex, exists := d.vertexMap[name]
	if !exists {
		var slice []*Vertex
		zlog.Info("Not get vertex by component")
		return slice
	}
	return d.NextZeroInDegreeVertexByVertex(vertex)
}

// NextZeroInDegreeVertexByVertex 通过顶点获取下一批入度为0的节点，同时删除当前顶点
func (d *DirectedGraph) NextZeroInDegreeVertexByVertex(vertex *Vertex) []*Vertex {
	var vertexSlice []*Vertex
	if 0 != vertex.InDegree {
		zlog.Error("The vertex inDegree is not zero")
		return vertexSlice
	}

	outEdgeSet, exists := d.outEdgeMap[vertex] // 根据顶点获取数据出边集
	if !exists {
		d.RemoveVertex(vertex)
		return vertexSlice
	}
	for _, edge := range outEdgeSet {
		// 找到与vertx相连且入度为1的节点，当把vertex顶点删除后入度就是1了
		if 1 == edge.inVertex.InDegree {
			vertexSlice = append(vertexSlice, edge.inVertex)
		}
	}
	// 删除当前顶点，使vertexSlice中的顶点入度为0
	d.RemoveVertex(vertex)
	return vertexSlice
}

// RemoveVertex 删除顶点
func (d *DirectedGraph) RemoveVertex(vertex *Vertex) {
	var removeSlice []*Edge
	for _, edge := range d.edgeSet {
		out := edge.outVertex
		in := edge.inVertex

		if vertex.Equals(out) { // 如果待删除的顶点是出顶点，那么就把，对应的入顶点做
			if inEdgeSlice, exists := d.inEdgeMap[in]; exists {
				d.inEdgeMap[in] = d.EdgeSliceRemoveItem(inEdgeSlice, edge) // 入边缓存删除入边
			}
			in.decInDegree()
			out.decOutDegree()
			removeSlice = append(removeSlice, edge)
			continue
		}

		if vertex.Equals(in) {
			if outEdgeSlice, exists := d.outEdgeMap[out]; exists {
				d.outEdgeMap[out] = d.EdgeSliceRemoveItem(outEdgeSlice, edge)
			}
			in.decInDegree()
			out.decOutDegree()
			removeSlice = append(removeSlice, edge)
		}
	}
	// 删除边
	d.edgeSet = d.EdgeSliceRemoveItemBatch(d.edgeSet, removeSlice)

	delete(d.outEdgeMap, vertex)
	delete(d.inEdgeMap, vertex)

	delete(d.vertexMap, vertex.Data.Name)
}

// AddVertex 增加顶点
func (d *DirectedGraph) AddVertex(vertex *Vertex) {
	if d.vertexMap == nil {
		d.vertexMap = make(map[string]*Vertex)
	}

	d.vertexMap[vertex.Data.Name] = vertex
}

// setAppend slice达到集合的效果，slice中存在相同的边，则不加入到slice，不存在，则加入
// slice 边集
// target 目标边
func (d *DirectedGraph) setAppend(slice []*Edge, target *Edge) []*Edge {
	found := false
	for _, item := range slice {
		in := item.inVertex.Data.Name == target.inVertex.Data.Name
		out := item.outVertex.Data.Name == target.outVertex.Data.Name
		if in && out {
			found = true
			break
		}
	}
	if !found {
		slice = append(slice, target)
	}
	return slice
}

// AddEdgeByEdge 通过边增加一条边
func (d *DirectedGraph) AddEdgeByEdge(edge *Edge) {
	d.edgeSet = d.setAppend(d.edgeSet, edge)
	d.AddEdgeMap(edge)
}

// AddEdgeMap 增加出入边缓存
func (d *DirectedGraph) AddEdgeMap(edge *Edge) {
	d.AddOutEdgeMap(edge)
	d.AddInEdgeMap(edge)
}

// AddOutEdgeMap 增加出边缓存
func (d *DirectedGraph) AddOutEdgeMap(edge *Edge) {
	if d.outEdgeMap == nil {
		d.outEdgeMap = make(map[*Vertex][]*Edge)
	}

	outEdgeSet := d.outEdgeMap[edge.outVertex]
	outEdgeSet = d.setAppend(outEdgeSet, edge)

	d.outEdgeMap[edge.outVertex] = outEdgeSet
}

// AddInEdgeMap 增加入边缓存
func (d *DirectedGraph) AddInEdgeMap(edge *Edge) {
	if d.inEdgeMap == nil {
		d.inEdgeMap = make(map[*Vertex][]*Edge)
	}

	inEdgeSet := d.inEdgeMap[edge.inVertex]
	inEdgeSet = d.setAppend(inEdgeSet, edge)

	d.inEdgeMap[edge.inVertex] = inEdgeSet
}

// AddEdgeByData 通过组件数据增加一条边
func (d *DirectedGraph) AddEdgeByData(out string, in string) {
	if d.vertexMap == nil {
		d.vertexMap = make(map[string]*Vertex)
	}
	// 获取表示边的起始顶点
	outVertex := d.vertexMap[out]
	// 出度加1
	outVertex.incOutDegree()
	d.vertexMap[out] = outVertex

	// 获取表示边的终点
	inVertex := d.vertexMap[in]
	// 入度加1
	inVertex.incInDegree()
	d.vertexMap[in] = inVertex

	d.AddEdgeByVertex(outVertex, inVertex)
}

// AddEdgeByVertex 通过顶点增加边
func (d *DirectedGraph) AddEdgeByVertex(out *Vertex, in *Vertex) {
	edge := &Edge{
		inVertex:  in,
		outVertex: out,
	}

	d.AddEdgeByEdge(edge)
}

// RemoveEdgeByData 通过data删除一条边
func (d *DirectedGraph) RemoveEdgeByData(out fuyaov1beta1.Component, in fuyaov1beta1.Component) {
	outVertex, outExists := d.vertexMap[out.Name]
	inVertex, inExists := d.vertexMap[in.Name]
	if outExists && inExists {
		edge := &Edge{
			outVertex: outVertex,
			inVertex:  inVertex,
		}
		d.RemoveEdge(edge)
	}
}

// EdgeSliceRemoveItemBatch edge切片批量移除元素
func (d *DirectedGraph) EdgeSliceRemoveItemBatch(edges []*Edge, targets []*Edge) []*Edge {
	for _, target := range targets {
		edges = d.EdgeSliceRemoveItem(edges, target)
	}
	return edges
}

// EdgeSliceRemoveItem edge切片移除元素
func (d *DirectedGraph) EdgeSliceRemoveItem(edges []*Edge, target *Edge) []*Edge {
	var result []*Edge
	for _, edge := range edges {
		if target.Equals(edge) {
			continue
		}
		result = append(result, edge)
	}
	return result
}

// RemoveEdge 删除一条边
func (d *DirectedGraph) RemoveEdge(edge *Edge) {
	outVertex := edge.outVertex
	outVertex.decOutDegree()
	if edges, ok := d.outEdgeMap[outVertex]; ok {
		d.outEdgeMap[outVertex] = d.EdgeSliceRemoveItem(edges, edge)
	}

	inVertex := edge.inVertex
	inVertex.decInDegree()
	if edges, ok := d.inEdgeMap[inVertex]; ok {
		d.inEdgeMap[outVertex] = d.EdgeSliceRemoveItem(edges, edge)
	}

	d.edgeSet = d.EdgeSliceRemoveItem(d.edgeSet, edge)
}
