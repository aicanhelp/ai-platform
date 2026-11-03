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

package plugin

import (
	"testing"

	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

func TestComputeNodeScore_Soft(t *testing.T) {
	devices := map[int]*common.XPUDevice{
		0: {Cores: 100, UsedCores: 30, Memory: 10000, UsedMemory: 2000},
		1: {Cores: 100, UsedCores: 50, Memory: 10000, UsedMemory: 4000},
	}
	score := computeNodeScore("test-node", devices, util.SoftMode)
	expectedCores := float64(30+50) / float64(100+100)
	expectedMem := float64(2000+4000) / float64(10000+10000)
	expected := 100.0 * (expectedCores + expectedMem)
	if vxpuTestAbsFloat(score-expected) > 0.001 {
		t.Errorf("computeNodeScore(soft) = %v, want %v", score, expected)
	}
}

func TestComputeNodeScore_Hard(t *testing.T) {
	devices := map[int]*common.XPUDevice{
		0: {Cores: 100, UsedCores: 30, Memory: 10000},
		1: {Cores: 100, UsedCores: 50, Memory: 10000},
	}
	score := computeNodeScore("test-node", devices, util.HardMode)
	expectedCores := float64(30+50) / float64(100+100)
	expected := 100.0 * expectedCores
	if vxpuTestAbsFloat(score-expected) > 0.001 {
		t.Errorf("computeNodeScore(hard) = %v, want %v", score, expected)
	}
}

func TestComputeNodeScore_Empty(t *testing.T) {
	score := computeNodeScore("test-node", map[int]*common.XPUDevice{}, util.SoftMode)
	if score != 0 {
		t.Errorf("empty devices score = %v, want 0", score)
	}
}
