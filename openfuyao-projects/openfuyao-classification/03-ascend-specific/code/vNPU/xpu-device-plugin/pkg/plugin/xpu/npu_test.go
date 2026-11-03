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
	"context"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"ascend-common/common-utils/hwlog"
	"ascend-common/devmanager/common"
	"ascend-common/devmanager/dcmi"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
)

var dc *dcmi.DcManager = &dcmi.DcManager{}

type PatchesManager struct {
	patches []*gomonkey.Patches
}

func NewPatchesManager() *PatchesManager {
	return &PatchesManager{patches: make([]*gomonkey.Patches, 0)}
}

func (pm *PatchesManager) Add(patch *gomonkey.Patches) {
	pm.patches = append(pm.patches, patch)
}

func (pm *PatchesManager) Reset() {
	for _, p := range pm.patches {
		p.Reset()
	}
}

func createPatchDcGetLogicIDList(result []int32, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetLogicIDList",
		func(_ *dcmi.DcManager) (int32, []int32, error) {
			return 0, result, err
		})
}

func createPatchDcGetCardIDDeviceID(cardID, deviceID int32, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetCardIDDeviceID",
		func(_ *dcmi.DcManager, _ int32) (int32, int32, error) {
			return cardID, deviceID, err
		})
}

func createPatchDcGetDieID(result string, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetDieID",
		func(_ *dcmi.DcManager, _, _ int32, _ dcmi.DieType) (string, error) {
			return result, err
		})
}

func createPatchDcGetPhysicIDFromLogicID(result int32, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetPhysicIDFromLogicID",
		func(_ *dcmi.DcManager, _ int32) (int32, error) {
			return result, err
		})
}

func createPatchDcGetMemoryInfo(memorySize uint64, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetMemoryInfo",
		func(_ *dcmi.DcManager, _, _ int32) (*common.MemoryInfo, error) {
			sample := &common.MemoryInfo{
				MemorySize:      memorySize,
				MemoryAvailable: 512,
				Frequency:       2000,
				Utilization:     80,
			}
			return sample, err
		})
}

func createPatchDcGetChipInfo(name string, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetChipInfo",
		func(_ *dcmi.DcManager, _, _ int32) (*common.ChipInfo, error) {
			chip := &common.ChipInfo{
				Name:    name,
				Type:    "ChipType",
				Version: "ChipVersion",
			}
			return chip, err
		})
}

func createPatchDcGetVDeviceInfo(aicCount float32, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetVDeviceInfo",
		func(_ *dcmi.DcManager, _ int32) (common.VirtualDevInfo, error) {
			devInfo := common.VirtualDevInfo{
				TotalResource: common.CgoSocTotalResource{Computing: common.CgoComputingResource{Aic: aicCount}},
			}
			return devInfo, err
		})
}

func createPatchDcGetDeviceUtilizationRate(rate int32, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetDeviceUtilizationRate",
		func(_ *dcmi.DcManager, _, _ int32, _ common.DeviceType) (int32, error) {
			return rate, err
		})
}

func createPatchDcGetDevProcessInfo(pid int32, memUsage float64, err error) *gomonkey.Patches {
	return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetDevProcessInfo",
		func(_ *dcmi.DcManager, _, _ int32) (*common.DevProcessInfo, error) {
			tmp := []common.DevProcInfo{{Pid: pid, MemUsage: memUsage}}
			retValue := &common.DevProcessInfo{
				DevProcArray: tmp,
				ProcNum:      1,
			}
			return retValue, err
		})
}

func TestNpuDevicesError(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetLogicIDList([]int32{}, errors.New("test error")))
	defer pm.Reset()

	d := DeviceManager{}
	if tmp := d.Devices(); tmp == nil {
		fmt.Println("recovering from error")
	} else {
		t.Error("error should be called")
	}
}

func TestNpuDevices(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetLogicIDList([]int32{1, 2, 3}, nil))
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, nil))
	pm.Add(createPatchDcGetDieID("", nil))
	pm.Add(createPatchDcGetPhysicIDFromLogicID(0, nil))
	defer pm.Reset()

	d := DeviceManager{}
	devs := d.Devices()
	expectedDevs := []Device{{v1beta1.Device{ID: "", Health: v1beta1.Healthy}, 1, 0},
		{v1beta1.Device{ID: "", Health: v1beta1.Healthy}, 2, 0},
		{v1beta1.Device{ID: "", Health: v1beta1.Healthy}, 3, 0}}
	reflect.DeepEqual(devs, expectedDevs)
}

type testCase struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected string
}

var testCasesForCheckHealth = []testCase{
	{
		desc: "DcGetCardIDDeviceID returned an error",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetCardIDDeviceID",
					func(_ *dcmi.DcManager, _ int32) (int32, int32, error) {
						return 0, 0, errors.New("error")
					})
			},
		},
		expected: "unhealthy",
	},
	{
		desc: "DcGetDeviceHealth returned an error",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetCardIDDeviceID",
					func(_ *dcmi.DcManager, _ int32) (int32, int32, error) {
						return 0, 0, nil
					})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetDeviceHealth",
					func(_ *dcmi.DcManager, _, _ int32) (int32, error) {
						return 0, errors.New("error")
					})
			},
		},
		expected: "unhealthy",
	},
	{
		desc: "DcGetDeviceHealth reported unhealthy",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetCardIDDeviceID",
					func(_ *dcmi.DcManager, _ int32) (int32, int32, error) {
						return 0, 0, nil
					})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyMethod(reflect.TypeOf(dc), "DcGetDeviceHealth",
					func(_ *dcmi.DcManager, _, _ int32) (int32, error) {
						return 1, nil
					})
			},
		},
		expected: "unhealthy",
	},
}

func sigFuncHelper(t *testing.T, tc testCase, stopCh chan<- interface{}, unhealthy <-chan *Device) {
	select {
	case <-unhealthy:
	case <-time.After(6 * time.Second):
		t.Error("error in test case: ", tc.desc)
	}
	stopCh <- 1
}

func ApplyPatches(patches []func() *gomonkey.Patches) []*gomonkey.Patches {
	var appliedPatches []*gomonkey.Patches
	for _, f := range patches {
		ap := f()
		appliedPatches = append(appliedPatches, ap)
	}
	return appliedPatches
}

func ResetPatches(appliedPatches []*gomonkey.Patches) {
	for _, p := range appliedPatches {
		p.Reset()
	}
}

func TestCheckHealth(t *testing.T) {
	for _, tc := range testCasesForCheckHealth {
		t.Run(tc.desc, func(t *testing.T) {
			stopCh := make(chan interface{})
			unhealthy := make(chan *Device, 1)
			devs := []*Device{{v1beta1.Device{ID: "testID1"}, 1, 1}}
			appliedPatches := ApplyPatches(tc.patches)
			defer ResetPatches(appliedPatches)
			d := DeviceManager{}
			go sigFuncHelper(t, tc, stopCh, unhealthy)
			d.CheckHealth(stopCh, devs, unhealthy)
		})
	}
}

func TestGetDeviceInfo(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, nil))
	pm.Add(createPatchDcGetMemoryInfo(1024, nil))
	pm.Add(createPatchDcGetChipInfo("ChipName", nil))
	pm.Add(createPatchDcGetVDeviceInfo(8.0, nil))
	defer pm.Reset()

	devs := []*Device{{v1beta1.Device{ID: "testID1"}, 1, 1}}
	actual := GetDeviceInfo(devs)
	expected := &types.DeviceInfo{
		Index:   1,
		Id:      "testID1",
		Count:   int32(config.DeviceSplitCount),
		Devmem:  1024,
		Type:    fmt.Sprintf("%v-%v-%v", DeviceType, "ASCEND", strings.ReplaceAll("ChipName", " ", "")),
		Health:  false,
		Corenum: 8,
	}
	assert.Equal(t, expected, actual[0])
}

func TestGetXPUUsage(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, nil))
	pm.Add(createPatchDcGetChipInfo("ChipName", nil))
	pm.Add(createPatchDcGetDeviceUtilizationRate(25, nil))
	pm.Add(createPatchDcGetDevProcessInfo(1234, 55.0, nil))
	defer pm.Reset()

	actualretUtilization, actualprocessMap, _ := GetXPUUsage(0, 0)
	expectedretUtilization := types.DeviceUsageInfo{
		CoreUtil: 25,
		MemUtil:  25,
	}
	expectedprocessMap := map[uint32]*types.ProcessUsage{1234: {ProcessMem: 55,
		ProcessCoreUtilization: 0}}
	assert.Equal(t, expectedprocessMap, actualprocessMap)
	assert.Equal(t, expectedretUtilization, actualretUtilization)
}

func TestGetVisibleDevices(t *testing.T) {
	devReq := types.ContainerDevices{
		{Index: 0},
		{Index: 1},
		{Index: 2},
	}
	result := GetVisibleDevices(devReq)
	if result != "0,1,2" {
		t.Errorf("expected 0,1,2, got %s", result)
	}

	empty := GetVisibleDevices(types.ContainerDevices{})
	if empty != "" {
		t.Errorf("expected empty string, got %s", empty)
	}
}

func TestNewTopologyProvider(t *testing.T) {
	provider := NewTopologyProvider()
	if provider == nil {
		t.Errorf("expected non-nil provider")
	}
	if provider.Topology() != "" {
		t.Errorf("expected empty topology")
	}
}

func TestGetVersionInfo(t *testing.T) {
	version, code, err := GetVersionInfo()
	if err != nil {
		t.Errorf("expected no error")
	}
	if version != "" || code != 0 {
		t.Errorf("expected empty version and 0 code")
	}
}

func TestInit_DmNil(t *testing.T) {
	origDm := dm
	dm = nil
	defer func() { dm = origDm }()

	err := Init()
	if err == nil {
		t.Errorf("expected error when dm is nil")
	}
}

func TestUninit_DmNil(t *testing.T) {
	origDm := dm
	dm = nil
	defer func() { dm = origDm }()

	err := Uninit()
	if err == nil {
		t.Errorf("expected error when dm is nil")
	}
}

func TestGetCardIDs_DmNil(t *testing.T) {
	origDm := dm
	dm = nil
	defer func() { dm = origDm }()

	_, err := GetCardIDs()
	if err == nil {
		t.Errorf("expected error when dm is nil")
	}
}

func TestGetXPUUsage_DmNil(t *testing.T) {
	origDm := dm
	dm = nil
	defer func() { dm = origDm }()

	_, _, err := GetXPUUsage(0, 0)
	if err == nil {
		t.Errorf("expected error when dm is nil")
	}
}

func TestUninit_Success(t *testing.T) {
	origDm := dm
	dm = &dcmi.DcManager{}
	defer func() { dm = origDm }()

	patchShutdown := gomonkey.ApplyMethod(reflect.TypeOf(dm), "DcShutDown", func(_ *dcmi.DcManager) error {
		return nil
	})
	defer patchShutdown.Reset()

	err := Uninit()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestGetCardIDs_Success(t *testing.T) {
	origDm := dm
	dm = &dcmi.DcManager{}
	defer func() { dm = origDm }()

	patch := gomonkey.ApplyMethod(reflect.TypeOf(dm), "DcGetCardList", func(_ *dcmi.DcManager) (int32, []int32, error) {
		return 1, []int32{0, 1}, nil
	})
	defer patch.Reset()

	cardIDs, err := GetCardIDs()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
	if len(cardIDs) != 2 {
		t.Errorf("expected 2 card IDs, got %d", len(cardIDs))
	}
}

func TestGetCardIDs_Error(t *testing.T) {
	origDm := dm
	dm = &dcmi.DcManager{}
	defer func() { dm = origDm }()

	patch := gomonkey.ApplyMethod(reflect.TypeOf(dm), "DcGetCardList", func(_ *dcmi.DcManager) (int32, []int32, error) {
		return 0, nil, fmt.Errorf("get card list error")
	})
	defer patch.Reset()

	_, err := GetCardIDs()
	if err == nil {
		t.Errorf("expected error from GetCardIDs")
	}
}

func TestInit_Success(t *testing.T) {
	origDm := dm
	dm = &dcmi.DcManager{}
	defer func() { dm = origDm }()

	patchLogger := gomonkey.ApplyFunc(hwlog.InitRunLogger, func(config *hwlog.LogConfig, ctx context.Context) error {
		return nil
	})
	defer patchLogger.Reset()

	patchInit := gomonkey.ApplyMethod(reflect.TypeOf(dm), "DcInit", func(_ *dcmi.DcManager) error {
		return nil
	})
	defer patchInit.Reset()

	err := Init()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestInit_DcInitError(t *testing.T) {
	origDm := dm
	dm = &dcmi.DcManager{}
	defer func() { dm = origDm }()

	patchLogger := gomonkey.ApplyFunc(hwlog.InitRunLogger, func(config *hwlog.LogConfig, ctx context.Context) error {
		return nil
	})
	defer patchLogger.Reset()

	patchInit := gomonkey.ApplyMethod(reflect.TypeOf(dm), "DcInit", func(_ *dcmi.DcManager) error {
		return fmt.Errorf("dc init error")
	})
	defer patchInit.Reset()

	err := Init()
	if err == nil {
		t.Errorf("expected error from DcInit")
	}
}

func TestGetXPUUsage_CardError(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, fmt.Errorf("card id error")))
	defer pm.Reset()

	_, _, err := GetXPUUsage(0, 0)
	if err == nil {
		t.Errorf("expected error from GetXPUUsage")
	}
}

func TestGetXPUUsage_ChipInfoError(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, nil))
	pm.Add(createPatchDcGetChipInfo("ChipName", fmt.Errorf("chip info error")))
	defer pm.Reset()

	_, _, err := GetXPUUsage(0, 0)
	if err == nil {
		t.Errorf("expected error from GetXPUUsage")
	}
}

func TestGetXPUUsage_MemUtilError(t *testing.T) {
	pm := NewPatchesManager()
	pm.Add(createPatchDcGetCardIDDeviceID(0, 0, nil))
	pm.Add(createPatchDcGetChipInfo("ChipName", nil))
	pm.Add(createPatchDcGetDeviceUtilizationRate(25, fmt.Errorf("util rate error")))
	defer pm.Reset()

	_, _, err := GetXPUUsage(0, 0)
	if err == nil {
		t.Errorf("expected error from GetXPUUsage")
	}
}
