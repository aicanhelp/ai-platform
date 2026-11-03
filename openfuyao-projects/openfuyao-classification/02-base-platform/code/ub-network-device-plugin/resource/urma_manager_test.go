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

package resource

import (
	"context"
	"errors"
	"testing"
	"time"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"github.com/agiledragon/gomonkey/v2"
	"k8s.io/apimachinery/pkg/util/wait"

	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
)

const (
	testIntervalMs    = 10 * time.Millisecond
	testSleepMs       = 50 * time.Millisecond
	testMaxRetries    = 3
	testExpectedLen   = 1
	testExpectedCount = 2
)

func newTestResourceManager() *UrmaResourceManager {
	return &UrmaResourceManager{
		resourceManager: resourceManager{
			name:      utils.ResourceName,
			resources: Resources{},
			devices:   Devices{},
		},
	}
}

func newTestResourceManagerWithDevices(devices Devices) *UrmaResourceManager {
	return &UrmaResourceManager{
		resourceManager: resourceManager{
			name:      utils.ResourceName,
			resources: Resources{},
			devices:   devices,
		},
	}
}

func mockGetDeviceIds(ids []string, err error) *gomonkey.Patches {
	return gomonkey.ApplyFunc(
		utils.GetAllDeviceIdsFromKubeletCheckpoint,
		func(string) ([]string, error) {
			return ids, err
		},
	)
}

func mockAllocate(success bool) *gomonkey.Patches {
	return gomonkey.ApplyFunc(
		urmasdk.UbsAllocateDevice,
		func(string) (urmasdk.DeviceInfo, error) {
			if success {
				return urmasdk.DeviceInfo{}, nil
			}
			return urmasdk.DeviceInfo{}, errors.New("allocate failed")
		},
	)
}

func mockAllocateWithCondition(condition func(name string) bool) *gomonkey.Patches {
	return gomonkey.ApplyFunc(
		urmasdk.UbsAllocateDevice,
		func(name string) (urmasdk.DeviceInfo, error) {
			if condition(name) {
				return urmasdk.DeviceInfo{}, nil
			}
			return urmasdk.DeviceInfo{}, errors.New("allocate failed")
		},
	)
}

func mockSleep() *gomonkey.Patches {
	return gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {
		// Empty function for testing - skip actual sleep
	})
}

func mockPollUntilContextTimeout() *gomonkey.Patches {
	return gomonkey.ApplyFunc(
		wait.PollUntilContextTimeout,
		func(ctx context.Context, _ time.Duration, _ time.Duration,
			_ bool, condition wait.ConditionWithContextFunc,
		) error {
			done, err := condition(ctx)
			if done {
				return nil
			}
			return err
		},
	)
}

func createStopChannel() <-chan interface{} {
	ch := make(chan interface{})
	go func() {
		time.Sleep(testSleepMs)
		close(ch)
	}()
	return ch
}

func TestSyncDevices(t *testing.T) {
	tests := []struct {
		name    string
		wantErr bool
	}{
		{name: "shared resource", wantErr: false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mock := urma.MockUrma{}
			defer mock.Reset()
			mock.MockUbsGetSharedDevice(tt.name)

			r := newTestResourceManager()
			err := r.SyncDevices(testIntervalMs, createStopChannel(), make(chan struct{}, 1))
			if (err != nil) != tt.wantErr {
				t.Errorf("SyncDevices() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestNewUrmaResourceManagers_Success(t *testing.T) {
	resources := &Resources{SHAREDs: []Resource{{Pattern: Pattern("*"), Name: utils.ResourceName}}}
	mock := urma.MockUrma{}
	defer mock.Reset()
	mock.MockUbsGetSharedDevice("success")

	got, err := NewUrmaResourceManagers(resources)
	if err != nil {
		t.Errorf("NewUrmaResourceManagers() unexpected error: %v", err)
		return
	}
	if len(got) != testExpectedLen {
		t.Errorf("NewUrmaResourceManagers() got %d managers, want %d", len(got), testExpectedLen)
	}
}

func TestNewUrmaResourceManagers_GetSharedDeviceError(t *testing.T) {
	resources := &Resources{SHAREDs: []Resource{{Pattern: Pattern("*"), Name: utils.ResourceName}}}
	mock := urma.MockUrma{}
	defer mock.Reset()
	mock.MockUbsGetSharedDevice("UbsGetSharedDevice err")

	got, err := NewUrmaResourceManagers(resources)
	if err == nil {
		t.Error("NewUrmaResourceManagers() expected error, got nil")
	}
	if got != nil {
		t.Errorf("NewUrmaResourceManagers() expected nil result, got %v", got)
	}
}

func TestNewUrmaResourceManagers_EmptyResources(t *testing.T) {
	resources := &Resources{SHAREDs: []Resource{}}
	mock := urma.MockUrma{}
	defer mock.Reset()
	mock.MockUbsGetSharedDevice("success")

	got, err := NewUrmaResourceManagers(resources)
	if err == nil {
		t.Error("NewUrmaResourceManagers() expected error for empty resources")
	}
	if got != nil {
		t.Errorf("NewUrmaResourceManagers() expected nil result, got %v", got)
	}
}

func TestUpdateDevices_EmptyDevices(t *testing.T) {
	r := newTestResourceManagerWithDevices(Devices{})
	newDevices := Devices{"dev1": {Name: "dev1", Healthy: true, HwResId: 1}}
	r.UpdateDevices(newDevices)
	got := r.Devices()

	if len(got) != 1 {
		t.Errorf("UpdateDevices() got %d devices, want 1", len(got))
		return
	}
	if gotDev, ok := got["dev1"]; !ok || gotDev.Healthy != true {
		t.Errorf("UpdateDevices() device dev1 mismatch")
	}
}

func TestUpdateDevices_ReplaceExisting(t *testing.T) {
	r := newTestResourceManagerWithDevices(Devices{"old": {Name: "old", Healthy: true, HwResId: 1}})
	newDevices := Devices{"new": {Name: "new", Healthy: false, HwResId: 2}}
	r.UpdateDevices(newDevices)
	got := r.Devices()

	if len(got) != 1 {
		t.Errorf("UpdateDevices() got %d devices, want 1", len(got))
		return
	}
	if _, ok := got["old"]; ok {
		t.Error("UpdateDevices() old device should be removed")
	}
	if gotDev, ok := got["new"]; !ok || gotDev.Healthy != false {
		t.Errorf("UpdateDevices() new device mismatch")
	}
}

func TestUpdateDevices_ClearDevices(t *testing.T) {
	r := newTestResourceManagerWithDevices(Devices{"dev1": {Name: "dev1", Healthy: true, HwResId: 1}})
	r.UpdateDevices(Devices{})
	got := r.Devices()

	if len(got) != 0 {
		t.Errorf("UpdateDevices() got %d devices, want 0", len(got))
	}
}

func TestRecover_EmptyDeviceIds(t *testing.T) {
	patchGetIds := mockGetDeviceIds([]string{}, nil)
	defer patchGetIds.Reset()

	r := newTestResourceManager()
	err := r.Recover()

	if err != nil {
		t.Errorf("Recover() unexpected error for empty device ids: %v", err)
	}
}

func TestRecover_Success(t *testing.T) {
	patchGetIds := mockGetDeviceIds([]string{"dev1", "dev2"}, nil)
	defer patchGetIds.Reset()

	patchAllocate := mockAllocateWithCondition(func(string) bool { return true })
	defer patchAllocate.Reset()

	r := newTestResourceManager()
	err := r.Recover()

	if err != nil {
		t.Errorf("Recover() unexpected error: %v", err)
	}
}

func TestRecover_GetDeviceIdsError(t *testing.T) {
	patchGetIds := mockGetDeviceIds(nil, errors.New("read error"))
	defer patchGetIds.Reset()

	patchSleep := mockSleep()
	defer patchSleep.Reset()

	r := newTestResourceManager()
	err := r.Recover()

	if err == nil {
		t.Error("Recover() expected error for get device ids failure")
	}
}

func TestRecover_PartialSuccess(t *testing.T) {
	patchGetIds := mockGetDeviceIds([]string{"dev1"}, nil)
	defer patchGetIds.Reset()

	patchAllocate := mockAllocateWithCondition(func(name string) bool { return name == "dev1" })
	defer patchAllocate.Reset()

	patchSleep := mockSleep()
	defer patchSleep.Reset()

	r := newTestResourceManager()
	err := r.Recover()

	if err != nil {
		t.Errorf("Recover() unexpected error: %v", err)
	}
}

func TestRecoverMaxRetries(t *testing.T) {
	callCount := 0
	patchGetIds := gomonkey.ApplyFunc(
		utils.GetAllDeviceIdsFromKubeletCheckpoint,
		func(string) ([]string, error) {
			callCount++
			if callCount < testMaxRetries {
				return nil, errors.New("get ids error")
			}
			return []string{"dev1"}, nil
		},
	)
	defer patchGetIds.Reset()

	patchAllocate := mockAllocate(true)
	defer patchAllocate.Reset()

	patchSleep := mockSleep()
	defer patchSleep.Reset()

	patchPoll := mockPollUntilContextTimeout()
	defer patchPoll.Reset()

	r := newTestResourceManager()
	err := r.Recover()

	if err != nil {
		t.Errorf("Recover() unexpected error: %v", err)
	}
	if callCount != testMaxRetries {
		t.Errorf("Recover() should call GetAllDeviceIdsFromKubeletCheckpoint %d times, got %d",
			testMaxRetries, callCount)
	}
}
