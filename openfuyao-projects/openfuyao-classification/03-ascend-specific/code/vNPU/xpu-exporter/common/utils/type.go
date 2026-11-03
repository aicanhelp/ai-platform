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

package utils

type XPUDevice struct {
	Index             int32
	Id                string
	Type              string
	Health            bool
	Count             uint32
	Mode              string
	Policy            int32
	MemoryTotal       uint64
	MemoryUsed        uint64
	MemoryUtilization float64
	XpuUtilization    float64
	NodeName          string
	NodeIp            string
	DriverVersion     string
	FrameworkVersion  int
	PowerUsage        int
	Temperature       int
	VxpuDeviceList    VxpuDevices
}

type VxpuDevices []VxpuDevice

type VxpuDevice struct {
	Id                    string
	XpuId                 string
	PodUID                string
	ContainerName         string
	VxpuMemoryUsed        uint64
	VxpuMemoryUtilization float64
	VxpuCoreUtilization   float64
	VxpuMemoryLimit       int64
	VxpuCoreLimit         int64
}
