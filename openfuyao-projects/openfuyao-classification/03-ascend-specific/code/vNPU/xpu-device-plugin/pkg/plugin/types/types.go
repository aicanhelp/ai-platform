/*
Copyright 2023 The Volcano Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package types

const (
	DeviceBindTime = "huawei.com/bind-time"

	DeviceBindPhase = "huawei.com/bind-phase"

	DeviceBindAllocating = "allocating"

	DeviceBindFailed = "failed"

	DeviceBindSuccess = "success"

	VXPULockName = "vnpu/mutex.lock"
)

type ContainerDevice struct {
	Index     int32
	UUID      string
	Type      string
	Usedmem   int32
	Usedcores int32
	Vid       int32
	Template  string
	Policy    int32
}

type ContainerDevices []ContainerDevice

type PodDevices []ContainerDevices

type DeviceInfo struct {
	Index   int32  `protobuf:"varint,1,opt,name=index,proto3" json:"index,omitempty"`
	Id      string `protobuf:"bytes,2,opt,name=id,proto3" json:"id,omitempty"`
	Count   int32  `protobuf:"varint,3,opt,name=count,proto3" json:"count,omitempty"`
	Devmem  int32  `protobuf:"varint,4,opt,name=devmem,proto3" json:"devmem,omitempty"`
	Type    string `protobuf:"bytes,5,opt,name=type,proto3" json:"type,omitempty"`
	Health  bool   `protobuf:"varint,6,opt,name=health,proto3" json:"health,omitempty"`
	Numa    int32  `protobuf:"varint,7,opt,name=numa,proto3" json:"numa,omitempty"`
	Mode    string `protobuf:"bytes,8,opt,name=mode,proto3" json:"mode,omitempty"`
	Corenum int32  `protobuf:"varint,9,opt,name=core_num,json=corenum,proto3" json:"corenum,omitempty"`
}

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
	PowerUsage        uint32
	Temperature       uint32
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
	Policy                int32
}

type ProcessUsage struct {
	ProcessMem             uint64
	ProcessCoreUtilization uint64
}

type DeviceUsageInfo struct {
	CoreUtil    uint32
	MemUtil     uint32
	PowerUsage  uint32
	Temperature uint32
}

type NodeConfig struct {
	Name string `yaml:"name"`
	Mode string `yaml:"vnpu-mode"`
}

type NodeConfigData struct {
	NodeConfigs []NodeConfig `yaml:"nodeConfigs"`
}
