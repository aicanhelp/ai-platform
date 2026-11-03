/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package core

import (
	"runtime"
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestGetNumaTopologyNoHardware covers the path where the system NUMA
// directory does not exist (typical in CI/containers). In that case Glob
// returns no node directories and we get an empty, error-free topology.
func TestGetNumaTopologyNoHardware(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("not supported on windows")
	}
	topology, err := GetNumaTopology()
	// Either no error (empty topology) or a parse error if /sys exists but
	// is malformed. In test environments it is almost always empty with no
	// error; we assert non-nil topology and that on the happy path the map
	// is initialized.
	assert.NotNil(t, topology)
	if err == nil {
		assert.NotNil(t, topology.Nodes)
	}
}

func TestGetNumaNodeForCPUEmpty(t *testing.T) {
	topology := &NumaTopology{Nodes: map[int]*NumaNode{}}
	id, ok := topology.GetNumaNodeForCPU(0)
	assert.False(t, ok)
	assert.Equal(t, -1, id)
}

func TestGetBestNumaNodeForAffinityExactTie(t *testing.T) {
	topology := &NumaTopology{
		Nodes: map[int]*NumaNode{
			0: {ID: 0, CPUList: []int{0, 1, 2}},
			1: {ID: 1, CPUList: []int{3, 4, 5}},
		},
	}
	// both nodes have exactly 3 CPUs; should return one of them.
	id, err := topology.GetBestNumaNodeForAffinity(3)
	assert.NoError(t, err)
	assert.Contains(t, []int{0, 1}, id)
}

func TestGetBestNumaNodeForAffinityEmpty(t *testing.T) {
	topology := &NumaTopology{Nodes: map[int]*NumaNode{}}
	_, err := topology.GetBestNumaNodeForAffinity(1)
	assert.Error(t, err)
}

func TestGetNumaNodeCPUsReturnsCopy(t *testing.T) {
	topology := &NumaTopology{
		Nodes: map[int]*NumaNode{
			0: {ID: 0, CPUList: []int{0, 1, 2}},
		},
	}
	cpus, err := topology.GetNumaNodeCPUs(0)
	assert.NoError(t, err)
	assert.Equal(t, []int{0, 1, 2}, cpus)

	// mutate the returned copy and ensure the underlying data is untouched
	cpus[0] = 99
	again, err := topology.GetNumaNodeCPUs(0)
	assert.NoError(t, err)
	assert.Equal(t, 0, again[0])
}

func TestCPUListToStringMoreCases(t *testing.T) {
	assert.Equal(t, "", CPUListToString(nil))
	assert.Equal(t, "5", CPUListToString([]int{5}))
	assert.Equal(t, "0-1,3", CPUListToString([]int{3, 0, 1}))
	assert.Equal(t, "100", CPUListToString([]int{100}))
}

func TestParseCPUListSingleCPUErrors(t *testing.T) {
	// single non-numeric value (not a range)
	_, err := parseCPUList("abc")
	assert.Error(t, err)
}
