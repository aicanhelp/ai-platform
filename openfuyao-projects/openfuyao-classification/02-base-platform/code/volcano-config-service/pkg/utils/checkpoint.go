/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

// Package utils provides utility functions and types for various operations.
// It includes functions for managing checkpoint data related to CPU and memory,
// converting integer slices to formatted strings and parsing ranges of integers from strings.
package utils

import (
	"encoding/json"
)

type checkpoint interface {
	unmarshalCheckpoint(data []byte) error
}

// CPUManagerCheckpoint represents checkpoint data related to CPU management.
type CPUManagerCheckpoint struct {
	PolicyName    string                       `json:"policyName"`
	DefaultCPUSet string                       `json:"defaultCpuSet"`
	Entries       map[string]map[string]string `json:"entries,omitempty"`
	Checksum      uint64                       `json:"checksum"`
}

// MemoryManagerCheckpoint represents checkpoint data related to memory management.
type MemoryManagerCheckpoint struct {
	PolicyName   string                              `json:"policyName"`
	MachineState map[string]interface{}              `json:"machineState"`
	Entries      map[string]map[string][]MemoryState `json:"entries,omitempty"`
	Checksum     uint64                              `json:"checksum"`
}

// MemoryState represents the state of memory.
type MemoryState struct {
	NumaAffinity []int  `json:"numaAffinity"`
	Type         string `json:"type"`
	Size         int    `json:"size"`
}

// NewCPUManagerCheckpoint creates a new CPUManagerCheckpoint instance.
func NewCPUManagerCheckpoint() *CPUManagerCheckpoint {
	return &CPUManagerCheckpoint{
		Entries: make(map[string]map[string]string),
	}
}

// NewMemoryManagerCheckpoint creates a new MemoryManagerCheckpoint instance.
func NewMemoryManagerCheckpoint() *MemoryManagerCheckpoint {
	return &MemoryManagerCheckpoint{
		Entries: make(map[string]map[string][]MemoryState),
	}
}

func (c *CPUManagerCheckpoint) unmarshalCheckpoint(data []byte) error {
	return json.Unmarshal(data, c)
}

func (m *MemoryManagerCheckpoint) unmarshalCheckpoint(data []byte) error {
	return json.Unmarshal(data, m)
}

// LoadCheckpoint loads checkpoint data into a checkpoint interface.
func LoadCheckpoint(data []byte, checkpoint checkpoint) error {
	if err := checkpoint.unmarshalCheckpoint(data); err != nil {
		return err
	}
	return nil
}
