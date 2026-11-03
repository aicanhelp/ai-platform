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

package utils

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNewCPUManagerCheckpoint(t *testing.T) {
	c := NewCPUManagerCheckpoint()
	assert.NotNil(t, c)
	assert.NotNil(t, c.Entries)
	assert.Equal(t, "", c.PolicyName)
	assert.Equal(t, "", c.DefaultCPUSet)
}

func TestNewMemoryManagerCheckpoint(t *testing.T) {
	c := NewMemoryManagerCheckpoint()
	assert.NotNil(t, c)
	assert.NotNil(t, c.Entries)
	assert.Equal(t, "", c.PolicyName)
}

func TestLoadCheckpoint_CPUManager(t *testing.T) {
	data := []byte(`{
		"policyName": "static",
		"defaultCpuSet": "0-3",
		"entries": {
			"container1": {
				"pod1": "0-1"
			}
		},
		"checksum": 12345
	}`)

	cpuCheckpoint := NewCPUManagerCheckpoint()
	err := LoadCheckpoint(data, cpuCheckpoint)
	assert.NoError(t, err)
	assert.Equal(t, "static", cpuCheckpoint.PolicyName)
	assert.Equal(t, "0-3", cpuCheckpoint.DefaultCPUSet)
	assert.Equal(t, uint64(12345), cpuCheckpoint.Checksum)
	assert.Equal(t, "0-1", cpuCheckpoint.Entries["container1"]["pod1"])
}

func TestLoadCheckpoint_MemoryManager(t *testing.T) {
	data := []byte(`{
		"policyName": "Static",
		"machineState": {},
		"entries": {},
		"checksum": 999
	}`)

	memCheckpoint := NewMemoryManagerCheckpoint()
	err := LoadCheckpoint(data, memCheckpoint)
	assert.NoError(t, err)
	assert.Equal(t, "Static", memCheckpoint.PolicyName)
	assert.Equal(t, uint64(999), memCheckpoint.Checksum)
}

func TestLoadCheckpoint_InvalidJSON(t *testing.T) {
	data := []byte(`{invalid json}`)
	cpuCheckpoint := NewCPUManagerCheckpoint()
	err := LoadCheckpoint(data, cpuCheckpoint)
	assert.Error(t, err)
}

func TestLoadCheckpoint_EmptyData(t *testing.T) {
	cpuCheckpoint := NewCPUManagerCheckpoint()
	err := LoadCheckpoint([]byte{}, cpuCheckpoint)
	assert.Error(t, err)
}
