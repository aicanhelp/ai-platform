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
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
)

func TestScheduleHandler_getNodeXPUDevices(t *testing.T) {
	tests := []struct {
		name           string
		nodeName       string
		setupDevices   map[string]map[int]*common.XPUDevice
		expectedResult map[int]*common.XPUDevice
		expectEmpty    bool
	}{
		{
			name:     "get existing node with single NPU device",
			nodeName: "node1",
			setupDevices: map[string]map[int]*common.XPUDevice{
				"node1": {
					0: {
						PhysicID:   0,
						DieID:      "npu-0",
						NodeID:     "node1",
						Type:       "NPU",
						Health:     true,
						Cores:      8,
						Memory:     16 * 1024 * 1024 * 1024,
						Count:      1,
						UsedCores:  5,
						UsedMemory: 4 * 1024 * 1024 * 1024,
						UsedVids:   0,
						InUse:      true,
						Numa:       0,
					},
				},
			},
			expectedResult: map[int]*common.XPUDevice{
				0: {
					PhysicID:   0,
					DieID:      "npu-0",
					NodeID:     "node1",
					Type:       "NPU",
					Health:     true,
					Cores:      8,
					Memory:     16 * 1024 * 1024 * 1024,
					Count:      1,
					UsedCores:  5,
					UsedMemory: 4 * 1024 * 1024 * 1024,
					UsedVids:   0,
					InUse:      true,
					Numa:       0,
				},
			},
			expectEmpty: false,
		},
		{
			name:     "get existing node with multiple XPU devices",
			nodeName: "node2",
			setupDevices: map[string]map[int]*common.XPUDevice{
				"node2": {
					0: {
						PhysicID:   0,
						DieID:      "npu-0",
						NodeID:     "node2",
						Type:       "NPU",
						Health:     true,
						Cores:      8,
						Memory:     16 * 1024 * 1024 * 1024,
						Count:      1,
						UsedCores:  0,
						UsedMemory: 0,
						UsedVids:   0,
						InUse:      false,
						Numa:       0,
					},
					1: {
						PhysicID:   1,
						DieID:      "npu-1",
						NodeID:     "node2",
						Type:       "NPU",
						Health:     true,
						Cores:      8,
						Memory:     16 * 1024 * 1024 * 1024,
						Count:      1,
						UsedCores:  8,
						UsedMemory: 16 * 1024 * 1024 * 1024,
						UsedVids:   1,
						InUse:      true,
						Numa:       1,
					},
				},
			},
			expectedResult: map[int]*common.XPUDevice{
				0: {
					PhysicID:   0,
					DieID:      "npu-0",
					NodeID:     "node2",
					Type:       "NPU",
					Health:     true,
					Cores:      8,
					Memory:     16 * 1024 * 1024 * 1024,
					Count:      1,
					UsedCores:  0,
					UsedMemory: 0,
					UsedVids:   0,
					InUse:      false,
					Numa:       0,
				},
				1: {
					PhysicID:   1,
					DieID:      "npu-1",
					NodeID:     "node2",
					Type:       "NPU",
					Health:     true,
					Cores:      8,
					Memory:     16 * 1024 * 1024 * 1024,
					Count:      1,
					UsedCores:  8,
					UsedMemory: 16 * 1024 * 1024 * 1024,
					UsedVids:   1,
					InUse:      true,
					Numa:       1,
				},
			},
			expectEmpty: false,
		},
		{
			name:     "get non-existing node returns empty map",
			nodeName: "node-not-exist",
			setupDevices: map[string]map[int]*common.XPUDevice{
				"node1": {
					0: {
						PhysicID: 0,
						DieID:    "npu-0",
						NodeID:   "node1",
					},
				},
			},
			expectedResult: map[int]*common.XPUDevice{},
			expectEmpty:    true,
		},
		{
			name:           "get from empty XPUDevices map",
			nodeName:       "any-node",
			setupDevices:   map[string]map[int]*common.XPUDevice{},
			expectedResult: map[int]*common.XPUDevice{},
			expectEmpty:    true,
		},
		{
			name:     "get node with unhealthy device",
			nodeName: "node3",
			setupDevices: map[string]map[int]*common.XPUDevice{
				"node3": {
					0: {
						PhysicID:   0,
						DieID:      "npu-0",
						NodeID:     "node3",
						Type:       "NPU",
						Health:     false,
						Cores:      8,
						Memory:     16 * 1024 * 1024 * 1024,
						Count:      1,
						UsedCores:  0,
						UsedMemory: 0,
						UsedVids:   0,
						InUse:      false,
						Numa:       0,
					},
				},
			},
			expectedResult: map[int]*common.XPUDevice{
				0: {
					PhysicID:   0,
					DieID:      "npu-0",
					NodeID:     "node3",
					Type:       "NPU",
					Health:     false,
					Cores:      8,
					Memory:     16 * 1024 * 1024 * 1024,
					Count:      1,
					UsedCores:  0,
					UsedMemory: 0,
					UsedVids:   0,
					InUse:      false,
					Numa:       0,
				},
			},
			expectEmpty: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sh := &ScheduleHandler{
				Mutex:      &sync.Mutex{},
				XPUDevices: tt.setupDevices,
				Jobs:       make(map[api.JobID]*SchedulerJob),
			}

			result := sh.getNodeXPUDevices(tt.nodeName)

			if tt.expectEmpty {
				assert.Empty(t, result, "Expected empty map")
				assert.NotNil(t, result, "Result should not be nil, but empty map")
			} else {
				assert.Equal(t, len(tt.expectedResult), len(result), "Device count mismatch")

				for idx, expectedDevice := range tt.expectedResult {
					actualDevice, exists := result[idx]
					assert.True(t, exists, "Device index %d should exist", idx)
					assert.Equal(t, expectedDevice.PhysicID, actualDevice.PhysicID)
					assert.Equal(t, expectedDevice.DieID, actualDevice.DieID)
					assert.Equal(t, expectedDevice.NodeID, actualDevice.NodeID)
					assert.Equal(t, expectedDevice.Type, actualDevice.Type)
					assert.Equal(t, expectedDevice.Health, actualDevice.Health)
					assert.Equal(t, expectedDevice.Cores, actualDevice.Cores)
					assert.Equal(t, expectedDevice.Memory, actualDevice.Memory)
					assert.Equal(t, expectedDevice.Count, actualDevice.Count)
					assert.Equal(t, expectedDevice.UsedCores, actualDevice.UsedCores)
					assert.Equal(t, expectedDevice.UsedMemory, actualDevice.UsedMemory)
					assert.Equal(t, expectedDevice.UsedVids, actualDevice.UsedVids)
					assert.Equal(t, expectedDevice.InUse, actualDevice.InUse)
					assert.Equal(t, expectedDevice.Numa, actualDevice.Numa)
				}
			}

			assert.Equal(t, tt.setupDevices, sh.XPUDevices, "Original XPUDevices should not be modified")
		})
	}
}
