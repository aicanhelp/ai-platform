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

// Edge 图的边
type Edge struct {
	// 出节点
	outVertex *Vertex
	// 入节点
	inVertex *Vertex
}

func (e *Edge) GetOutVertex() *Vertex {
	return e.outVertex
}

func (e *Edge) GetInVertex() *Vertex {
	return e.inVertex
}

// RevertEdge 反转边
func (e *Edge) RevertEdge() {
	e.outVertex.incInDegree()
	e.outVertex.decOutDegree()

	e.inVertex.incOutDegree()
	e.inVertex.decInDegree()

	tmp := e.outVertex
	e.outVertex = e.inVertex
	e.inVertex = tmp
}

// Equals 判断两个定点是否相同
func (e *Edge) Equals(other *Edge) bool {
	// 无重复组件名字，使用名字判断
	out := e.outVertex.Equals(other.outVertex)
	in := e.inVertex.Equals(other.inVertex)
	return out && in
}
