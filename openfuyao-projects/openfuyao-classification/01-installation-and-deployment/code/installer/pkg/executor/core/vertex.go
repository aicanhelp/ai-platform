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
)

// Vertex 顶点
type Vertex struct {
	Data      fuyaov1beta1.Component
	OutDegree int
	InDegree  int
}

func (v *Vertex) ResetDegree() {
	v.InDegree = 0
	v.OutDegree = 0
}

func (v *Vertex) incInDegree() {
	v.InDegree += 1
}

func (v *Vertex) decInDegree() {
	v.InDegree -= 1
}

func (v *Vertex) incOutDegree() {
	v.OutDegree += 1
}

func (v *Vertex) decOutDegree() {
	v.OutDegree -= 1
}

// Equals 判断两个组件是否相同
func (v *Vertex) Equals(other *Vertex) bool {
	// 无重复组件名字，使用名字判断
	return other.Data.Name == v.Data.Name
}
