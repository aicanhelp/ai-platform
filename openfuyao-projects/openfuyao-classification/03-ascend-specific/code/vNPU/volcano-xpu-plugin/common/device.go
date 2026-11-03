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

// Package common defines data structure for plugin and allocator
package common

// XPUDevices for XPU node
type XPUDevice struct {
	PhysicID   int
	DieID      string
	NodeID     string
	Type       string
	Health     bool
	Cores      int
	Memory     uint64
	Count      int
	UsedCores  int
	UsedMemory uint64
	UsedVids   uint
	InUse      bool
	Numa       int
	UsedCpu    int
	Mode       string
	Policy     string
}

// GetUsedVidCount get number of used vids
func (x *XPUDevice) GetUsedVidCount() uint {
	count := uint(0)
	vids := x.UsedVids
	for vids != 0 {
		count += vids & 1
		vids >>= 1
	}
	return count
}

// AllocateVid find an unused vid from XPUDevice, mark it as used and return the vid
func (x *XPUDevice) AllocateVid() uint {
	vid := uint(0)
	vidBit := (x.UsedVids + 1) ^ x.UsedVids
	x.UsedVids |= vidBit
	for vidBit > 1 {
		vid++
		vidBit >>= 1
	}
	return vid
}

// MarkVidUsed mark given vid as used
func (x *XPUDevice) MarkVidUsed(vid uint) bool {
	vidBit := uint(1) << vid
	ok := x.UsedVids&vidBit != 0
	x.UsedVids |= vidBit
	return ok
}

// ContainerDevice description of xpu in the container
type ContainerDevice struct {
	Index      int
	Id         string
	Type       string
	UsedMemory uint64
	UsedCores  int
	Vid        uint
	Template   string
	Policy     string
}
