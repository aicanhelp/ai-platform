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

package xpu

import (
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"ascend-common/common-utils/hwlog"
	"ascend-common/devmanager/common"
	"ascend-common/devmanager/dcmi"
	"huawei.com/vxpu-device-plugin/pkg/graph"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
)

const (
	//
	VxpuNumber = "huawei.com/vnpu-number"

	VxpuCore = "huawei.com/vnpu-cores"

	VxpuMemory          = "huawei.com/vnpu-memory.1Gi"
	healthCheckInterval = 5
	ddrDevcieType       = 1

	VisibleDevices = "ASCEND_VISIBLE_DEVICES"

	VxpuSpecs = "ASCEND_VNPU_SPECS"

	FullCardTemplate = "full"

	FullCardCores = 100

	VxpuInfoFileName = "npu_info.config"

	devShmDir = "/dev/shm"

	DeviceType = "NPU"

	AssignedIDs = "huawei.com/vnpu-ids-new"

	AssignedIDsToAllocate = "huawei.com/vnpu-devices-to-allocate"

	SchedulingPolicy = "huawei.com/vnpu-soft-scheduler-policy"

	PolicyFixedShareStr = "fixed-share"

	PolicyFixedShareInt = 1

	PolicyElasticStr = "elastic"

	PolicyElasticInt = 2

	PolicyBestEffortStr = "best-effort"

	PolicyBestEffortInt = 3

	NodeVXPUHandshake = "huawei.com/node-vnpu-handshake"

	NodeVXPURegister = "huawei.com/node-vnpu-register"

	AssignedNode = "huawei.com/vnpu-node"

	NodeXpuTopology = "huawei.com/node-npu-topology"

	Ascend910 = "910"
)

var (
	dm *dcmi.DcManager = &dcmi.DcManager{}
)

func GetDevShmMount() *v1beta1.Mount {
	return &v1beta1.Mount{
		ContainerPath: filepath.Clean(devShmDir),
		HostPath:      filepath.Clean(devShmDir),
		ReadOnly:      false,
	}
}

func Init() error {
	logConfig := hwlog.LogConfig{OnlyToStdout: true}
	if err := hwlog.InitRunLogger(&logConfig, nil); err != nil {
		return err
	}
	if dm == nil {
		return fmt.Errorf("dcmi.DcManager instance dm initialize failed")
	}
	return dm.DcInit()
}

func Uninit() error {
	if dm == nil {
		return fmt.Errorf("dcmi.DcManager instance dm initialize failed")
	}
	return dm.DcShutDown()
}

// GetCardIDs returns the list of NPU card IDs on this node.
func GetCardIDs() ([]int32, error) {
	if dm == nil {
		return nil, fmt.Errorf("dcmi.DcManager instance dm initialize failed")
	}
	_, cardIDs, err := dm.DcGetCardList()
	if err != nil {
		return nil, fmt.Errorf("get card list failed: %v", err)
	}
	return cardIDs, nil
}

type DeviceManager struct{}

func (*DeviceManager) Devices() []*Device {
	if dm == nil {
		log.Errorf("dcmi.DcManager instance dm initialize failed")
		return nil
	}

	_, logicIDs, err := dm.DcGetLogicIDList()
	if err != nil {
		log.Errorf("DcGetLogicIDList failed: %v", err)
		return nil
	}

	var devs []*Device
	for _, logicID := range logicIDs {
		cardID, deviceID, err := dm.DcGetCardIDDeviceID(logicID)
		if err != nil {
			log.Errorf("DcGetCardIDDeviceID failed: %v", err)
			return nil
		}
		dieID, err := dm.DcGetDieID(cardID, deviceID, dcmi.VDIE)
		if err != nil {
			log.Errorf("DcGetDieID failed: %v", err)
			return nil
		}
		physicID, err := dm.DcGetPhysicIDFromLogicID(logicID)
		if err != nil {
			log.Errorf("DcGetPhysicIDFromLogicID failed: %v", err)
			return nil
		}
		dev := Device{}
		dev.ID = dieID
		dev.Health = v1beta1.Healthy
		dev.LogicID = logicID
		dev.PhysicID = physicID
		devs = append(devs, &dev)
	}
	return devs
}

func (*DeviceManager) CheckHealth(stop <-chan interface{}, devices []*Device, unhealthy chan<- *Device) {
	if dm == nil {
		log.Errorf("dcmi.DcManager instance dm initialize failed")
		return
	}

	for {
		select {
		case <-stop:
			return
		default:
		}

		time.Sleep(healthCheckInterval * time.Second)

		for _, d := range devices {
			cardID, deviceID, err := dm.DcGetCardIDDeviceID(d.LogicID)
			if err != nil {
				log.Warningf("device DcGetCardIDDeviceID failed: %v, mark it unhealthy, logicID: %v, dieID: %v",
					err, d.LogicID, d.ID)
				unhealthy <- d
				continue
			}

			healthCode, err := dm.DcGetDeviceHealth(cardID, deviceID)
			if err != nil {
				log.Warningf("device DcGetDeviceHealth failed: %v, mark it unhealthy, logicID: %v, dieID: %v", err, d.LogicID, d.ID)
				unhealthy <- d
				continue
			}
			if healthCode != 0 {
				log.Warningf("device unhealthy: %v, logicID: %v, dieID: %v", healthCode, d.LogicID, d.ID)
				unhealthy <- d
				continue
			}
		}
	}
}

func isHbmChip(chipName string) bool {
	return strings.Contains(chipName, Ascend910)
}

func GetDeviceInfo(devs []*Device) []*types.DeviceInfo {
	if dm == nil {
		log.Errorf("dcmi.DcManager instance dm initialize failed")
		return nil
	}

	res := make([]*types.DeviceInfo, 0, len(devs))
	for _, dev := range devs {
		cardID, deviceID, err := dm.DcGetCardIDDeviceID(dev.LogicID)
		if err != nil {
			log.Fatalf("dcmi DcGetCardIDDeviceID failed: %v, logicID: %v, dieID: %v", err, dev.LogicID, dev.ID)
		}
		chipInfo, err := dm.DcGetChipInfo(cardID, deviceID)
		if err != nil {
			log.Fatalf("dcmi DcGetChipInfo failed: %v, logicID: %v, dieID: %v", err, dev.LogicID, dev.ID)
		}
		var memSize int32
		if isHbmChip(chipInfo.Name) {
			hbmInfo, err := dcmi.FuncDcmiGetDeviceHbmInfo(cardID, deviceID)
			if err != nil {
				log.Fatalf("dcmi FuncDcmiGetDeviceHbmInfo failed: %v, logicID: %v, dieID: %v", err,
					dev.LogicID, dev.ID)
			}
			memSize = int32(hbmInfo.MemorySize)
		} else {
			memInfo, err := dm.DcGetMemoryInfo(cardID, deviceID)
			if err != nil {
				log.Fatalf("dcmi DcGetMemoryInfo failed: %v, logicID: %v, dieID: %v", err, dev.LogicID, dev.ID)
			}
			memSize = int32(memInfo.MemorySize)
		}
		CgoVDevInfo, err := dm.DcGetVDeviceInfo(dev.LogicID)
		if err != nil {
			log.Fatalf("dcmi DcGetVDeviceInfo failed: %v, logicID: %v", err, dev.LogicID)

		}
		log.Infoln("dcmi registered deviceId:", dev.ID, "memory:", memSize, "name:", chipInfo.Name)

		res = append(res, &types.DeviceInfo{
			Index:   dev.PhysicID,
			Id:      dev.ID,
			Count:   int32(config.DeviceSplitCount),
			Devmem:  int32(memSize),
			Type:    fmt.Sprintf("%v-%v-%v", DeviceType, "ASCEND", strings.ReplaceAll(chipInfo.Name, " ", "")),
			Health:  dev.Health == v1beta1.Healthy,
			Mode:    config.NodeMode,
			Corenum: int32(CgoVDevInfo.TotalResource.Computing.Aic),
		})
	}
	return res
}

func GetVisibleDevices(devReq types.ContainerDevices) string {
	visibleDevices := make([]string, 0)
	for _, dev := range devReq {
		visibleDevices = append(visibleDevices, strconv.Itoa(int(dev.Index)))
	}
	return strings.Join(visibleDevices, ",")
}

func GetXPUUsage(logicID, _ int32) (types.DeviceUsageInfo, map[uint32]*types.ProcessUsage, error) {
	if dm == nil {
		err := fmt.Errorf("dcmi.DcManager instance dm initialize failed")
		log.Errorf("%v", err)
		return types.DeviceUsageInfo{}, nil, err
	}

	cardID, deviceID, err := dm.DcGetCardIDDeviceID(logicID)
	if err != nil {
		log.Errorf("dcmi DcGetCardIDDeviceID failed: %v, logicID: %v", err, logicID)
		return types.DeviceUsageInfo{}, nil, err
	}

	chipInfo, err := dm.DcGetChipInfo(cardID, deviceID)
	if err != nil {
		log.Errorf("dcmi DcGetChipInfo failed: %v, cardID: %v, deviceID: %v", err, cardID, deviceID)
		return types.DeviceUsageInfo{}, nil, err
	}

	var memUtilRate int32
	if isHbmChip(chipInfo.Name) {
		memUtilRate, err = dm.DcGetDeviceUtilizationRate(cardID, deviceID, common.HbmUtilization)
	} else {
		memUtilization := common.DeviceType{Code: ddrDevcieType, Name: "DDR"}
		memUtilRate, err = dm.DcGetDeviceUtilizationRate(cardID, deviceID, memUtilization)
	}
	if err != nil {
		log.Errorf("dcmi DcGetDeviceUtilizationRate Memory failed: %v, cardID: %v, deviceID: %v", err, cardID, deviceID)
		return types.DeviceUsageInfo{}, nil, err
	}

	aiCoreUtilRate, err := dm.DcGetDeviceUtilizationRate(cardID, deviceID, common.AICore)
	if err != nil {
		log.Errorf("dcmi DcGetDeviceUtilizationRate AiCore failed: %v, cardID: %v, deviceID: %v", err, cardID, deviceID)
		return types.DeviceUsageInfo{}, nil, err
	}

	retUtilization := types.DeviceUsageInfo{
		CoreUtil: uint32(aiCoreUtilRate),
		MemUtil:  uint32(memUtilRate),
	}

	devProcInfo, err := dm.DcGetDevProcessInfo(cardID, deviceID)
	if err != nil {
		log.Errorf("dcmi DcGetDevProcessInfo failed: %v, cardID: %v, deviceID: %v", err, cardID, deviceID)
		return retUtilization, nil, err
	}

	processMap := make(map[uint32]*types.ProcessUsage)
	for _, v := range devProcInfo.DevProcArray {
		p := types.ProcessUsage{
			ProcessMem:             uint64(v.MemUsage),
			ProcessCoreUtilization: 0,
		}
		processMap[uint32(v.Pid)] = &p
	}
	return retUtilization, processMap, nil
}

type npuTopologyProvider struct{}

var _ graph.TopologyProvider = (*npuTopologyProvider)(nil)

func NewTopologyProvider() graph.TopologyProvider {
	return &npuTopologyProvider{}
}

func (provider *npuTopologyProvider) Topology() string {
	return ""
}

func GetVersionInfo() (string, int, error) {
	return "", 0, nil
}
