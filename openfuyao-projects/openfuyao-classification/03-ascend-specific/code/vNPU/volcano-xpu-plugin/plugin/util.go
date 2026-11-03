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

// Package plugin implements xpu scheduler plugin
package plugin

import (
	"fmt"
	"strconv"
	"strings"

	"k8s.io/api/core/v1"
	klog "k8s.io/klog/v2"
	"volcano.sh/volcano/pkg/scheduler/api"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/common"
	"volcano.sh/volcano/pkg/scheduler/plugins/volcano-xpu-plugin/util"
)

// DecodeNodeDevices decode string to node's xpus info
func DecodeNodeDevices(str string, nodeId string) map[int]*common.XPUDevice {
	xpuDevices := make(map[int]*common.XPUDevice)
	if !strings.Contains(str, ":") {
		klog.V(util.LogErrorLevel).Infof("Decode node device failed, wrong annotations: %s", str)
		return xpuDevices
	}
	tmp := strings.Split(str, ":")
	for _, val := range tmp {
		if strings.Contains(val, ",") {
			items := strings.Split(val, ",")
			if len(items) != util.XPUDeviceLen {
				klog.V(util.LogErrorLevel).Infof("Decode node device failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}

			index, err := strconv.Atoi(items[0])
			if err != nil {
				klog.V(util.LogErrorLevel).Infof("Decode index failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}

			count, err := strconv.Atoi(items[2])
			if err != nil {
				klog.V(util.LogErrorLevel).Infof("Decode count failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}

			memory, err := strconv.Atoi(items[3])
			if err != nil {
				klog.V(util.LogErrorLevel).Infof("Decode memory failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}

			health, err := strconv.ParseBool(items[5])
			if err != nil {
				klog.V(util.LogErrorLevel).Infof("Decode health failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}

			numa, err := strconv.Atoi(items[6])
			if err != nil {
				klog.V(util.LogErrorLevel).Infof("Decode numa failed, wrong device info: %s", val)
				return map[int]*common.XPUDevice{}
			}
			device := &common.XPUDevice{
				PhysicID:   index,
				DieID:      items[1],
				NodeID:     nodeId,
				Type:       items[4],
				Count:      count,
				Health:     health,
				Cores:      util.Base100,
				Memory:     uint64(memory),
				UsedCores:  0,
				UsedMemory: 0,
				UsedVids:   0,
				InUse:      false,
				Numa:       numa,
				UsedCpu:    0,
				Mode:       items[7],
			}
			xpuDevices[index] = device
		}
	}
	return xpuDevices
}

// EncodeContainerDevices encode vxpu resource request of a container to string
func EncodeContainerDevices(cds ContainerDevices) string {
	var sb strings.Builder
	for _, val := range cds {
		sb.Write([]byte(strconv.Itoa(int(val.Index))))
		sb.Write([]byte(","))
		sb.Write([]byte(val.Id))
		sb.Write([]byte(","))
		valType := val.Type
		if strings.Contains(valType, util.AscendNPUDevice) {
			valType = util.AscendNPUDevice
		}
		sb.Write([]byte(valType))
		sb.Write([]byte(","))
		sb.Write([]byte(strconv.Itoa(int(val.UsedMemory))))
		sb.Write([]byte(","))
		sb.Write([]byte(strconv.Itoa(int(val.UsedCores))))
		sb.Write([]byte(","))
		sb.Write([]byte(strconv.FormatUint(uint64(val.Vid), util.Base10)))
		sb.Write([]byte(","))
		sb.Write([]byte(val.Template))
		sb.Write([]byte(":"))
	}
	klog.V(util.LogDebugLevel).Infof("Encoded container Devices: %s", sb.String())
	return sb.String()
}

// EncodePodDevices encode vxpu resource request of a pod to string
func EncodePodDevices(pd PodDevices) string {
	var sb strings.Builder
	for i, cd := range pd {
		sb.WriteString(EncodeContainerDevices(cd))
		if i < len(pd)-1 {
			sb.WriteString(";")
		}
	}
	return sb.String()
}

// DecodeContainerDevices decode vxpu resource request of a container from string
func DecodeContainerDevices(str string) ContainerDevices {
	if len(str) == 0 {
		return ContainerDevices{}
	}
	cd := strings.Split(str, ":")
	containerDevices := ContainerDevices{}
	for _, val := range cd {
		if strings.Contains(val, ",") == false {
			continue
		}
		fields := strings.Split(val, ",")
		tmpdev := common.ContainerDevice{}
		if len(fields) != util.ContainerLength {
			klog.V(util.LogErrorLevel).Infof("DecodeContainerDevices invalid parameter: %s", str)
			return ContainerDevices{}
		}
		index, err := strconv.Atoi(fields[0])
		if err != nil {
			klog.V(util.LogErrorLevel).Infof("DecodeContainerDevices invalid parameter: %s", str)
			return ContainerDevices{}
		}
		tmpdev.Index = index
		tmpdev.Id = fields[1]
		tmpdev.Type = fields[2]
		mem, err := strconv.Atoi(fields[3])
		if err != nil {
			klog.V(util.LogErrorLevel).Infof("DecodeContainerDevices invalid parameter: %s", str)
			return ContainerDevices{}
		}
		tmpdev.UsedMemory = uint64(mem)
		devcores, err := strconv.Atoi(fields[4])
		if err != nil {
			klog.V(util.LogErrorLevel).Infof("DecodeContainerDevices invalid parameter: %s", str)
			return ContainerDevices{}
		}
		tmpdev.UsedCores = devcores
		vid, err := strconv.ParseUint(fields[5], util.Base10, 0)
		if err != nil {
			klog.V(util.LogErrorLevel).Infof("DecodeContainerDevices invalid parameter: %s", str)
			return ContainerDevices{}
		}
		tmpdev.Vid = uint(vid)
		tmpdev.Template = fields[6]
		containerDevices = append(containerDevices, tmpdev)
	}
	return containerDevices
}

// DecodePodDevices decode vxpu resource request of a pod from string
func DecodePodDevices(str string) PodDevices {
	if len(str) == 0 {
		return PodDevices{}
	}
	var pd PodDevices
	for _, s := range strings.Split(str, ";") {
		cd := DecodeContainerDevices(s)
		pd = append(pd, cd)
	}
	return pd
}

// getContainerDevices get container device info
func getContainerDevices(allocateXPUs []int, xpuDevices map[int]*common.XPUDevice) (ContainerDevices, error) {
	cds := ContainerDevices{}
	for _, v := range allocateXPUs {
		xpuDevice, ok := xpuDevices[v]
		if !ok {
			return nil, fmt.Errorf("getContainerDevices failed, XPU %d does not exist on the node", v)
		}
		cd := common.ContainerDevice{
			Index:      xpuDevice.PhysicID,
			Id:         xpuDevice.DieID,
			Type:       xpuDevice.Type,
			UsedMemory: xpuDevice.Memory,
			UsedCores:  util.Base100,
		}
		cds = append(cds, cd)
	}
	return cds, nil
}

// GetXPUResourceFromTaskInfo get xpu resource info from task info
func GetXPUResourceFromTaskInfo(task *api.TaskInfo, xpuName string) *util.TaskResource {
	if task == nil || task.Pod == nil {
		return nil
	}

	var tr util.TaskResource
	tr.ReqXPUName = xpuName

	for _, c := range task.Pod.Spec.Containers {
		cr := GetXPUResourceFromContainer(&c, xpuName, util.VNPUCore, util.VNPUMemory, util.VNPUType)
		if cr.ReqXPUNum == 0 {
			continue
		}
		tr.ReqXPUNum += cr.ReqXPUNum
		tr.ReqXPUCores += cr.ReqXPUCores * cr.ReqXPUNum
		tr.ReqXPUMem += cr.ReqXPUMem * cr.ReqXPUNum
		tr.ReqXPUMemPercentage += cr.ReqXPUMemPercentage * cr.ReqXPUNum
		if tr.ReqXPUType == "" && cr.ReqXPUType != "" {
			tr.ReqXPUType = cr.ReqXPUType
		}
	}
	return &tr
}

// GetXPUResourceFromContainer get xpu resource info from container
func GetXPUResourceFromContainer(container *v1.Container, xpuName string, xpuCoreName string,
	xpuMemName string, xpuTypeName string) util.ContainerResource {
	var cr util.ContainerResource
	cr.ReqXPUName = xpuName

	vxpuNum := util.GetVXPUResource(container, xpuName)
	vxpuCore := util.GetVXPUResource(container, xpuCoreName)
	vxpuMem := util.GetVXPUResource(container, xpuMemName) * util.Base1024
	cr.ReqXPUNum = vxpuNum
	if vxpuNum < 0 {
		klog.V(util.LogDebugLevel).Infof("Container %s do not apply xpu device, resources limit: %v",
			container.Name, container.Resources.Limits)
		return cr
	}
	cr.ReqXPUCores = vxpuCore
	cr.ReqXPUMem = vxpuMem
	if vxpuCore < 0 {
		cr.ReqXPUCores = util.Base100
	}
	if vxpuMem < 0 {
		cr.ReqXPUMemPercentage = util.Base100
	}

	cr.ReqXPUType = util.GetXPUType(container, xpuTypeName)
	return cr
}
