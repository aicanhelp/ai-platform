/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewTopologGraph(t *testing.T) {
	assert.PanicsWithValue(t, "capacity must not be negative", func() { NewTopologyGraph(-1) })

	cases := []struct {
		capacity int
		expect   int
	}{
		{capacity: 0, expect: 0},
		{capacity: 10, expect: 10},
	}

	for _, tt := range cases {
		graph := NewTopologyGraph(tt.capacity)
		assert.Len(t, graph, tt.expect)
	}
}

func TestGetTopologyGraph(t *testing.T) {
	graph := NewTopologyGraph(0)
	assert.Equal(t, "", graph.GetTopologyGraph())

	graph = TopologyGraph([][]int{
		{0, 10, 20},
		{10, 0, 30},
		{20, 30, 1},
	})
	assert.Equal(t, graph.GetTopologyGraph(), "0,10,20;10,0,30;20,30,1")
}
