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
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/sirupsen/logrus"

	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

func TestNewDeviceRegister(t *testing.T) {
	dr := NewDeviceRegister(&DeviceCache{})
	if dr == nil {
		t.Error("new device register creation failed")
	}
}

var annotationMap map[string]string

var registerTestCasesForInAnnotation = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
}{
	{
		desc: "checking error handling from PatchNodeAnnotations()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var dr *DeviceRegister
				return gomonkey.ApplyPrivateMethod(dr, "apiDevices", func() []*types.DeviceInfo {
					return []*types.DeviceInfo{{Index: 0}}
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.EncodeNodeDevices, func([]*types.DeviceInfo) string {
					return "test_encoded_device_string"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PatchNodeAnnotations, func(string, map[string]string) error {
					return fmt.Errorf("test error at patching node annotation")
				})
			},
		},
		expected: 1,
	},
	{
		desc: "happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var dr *DeviceRegister
				return gomonkey.ApplyPrivateMethod(dr, "apiDevices", func() []*types.DeviceInfo {
					return []*types.DeviceInfo{{Index: 0}}
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.EncodeNodeDevices, func([]*types.DeviceInfo) string {
					return "test_encoded_device_string"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(time.Now, func() time.Time {
					return time.Date(2000, 01, 01, 01, 01, 01, 01, time.Local)
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PatchNodeAnnotations, func(_ string, a map[string]string) error {
					annotationMap = a
					return nil
				})
			},
		},
		expected: 0,
	},
}

func TestRegisterInAnnotation(t *testing.T) {
	for i := range registerTestCasesForInAnnotation {
		tc := registerTestCasesForInAnnotation[i]
		t.Run(tc.desc, func(t *testing.T) {
			dr := NewDeviceRegister(&DeviceCache{})
			appliedPatches := ApplyPatches(tc.patches)
			err := dr.registerInAnnotation()
			var status int
			if err != nil {
				status = 1
			}
			if _, ok := annotationMap[xpu.NodeVXPURegister]; !ok && tc.expected == 0 {
				t.Error("error in test case: ", tc.desc)
			}
			if _, ok := annotationMap[xpu.NodeVXPUHandshake]; !ok && tc.expected == 0 {
				t.Error("error in test case: ", tc.desc)
			}
			if tc.expected == 0 && (annotationMap[xpu.NodeVXPURegister] != "test_encoded_device_string" ||
				annotationMap[xpu.NodeVXPUHandshake] != "Reported_2000.01.01 01:01:01") {
				t.Error("error in test case: ", tc.desc)
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

var registerTestCasesForWatchAndRegister = []struct {
	desc         string
	patches      []func() *gomonkey.Patches
	expectedStop bool
}{
	{
		desc: "checking error handling from registerInAnnotation()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var r *DeviceRegister
				return gomonkey.ApplyPrivateMethod(r, "registerInAnnotation", func(*DeviceRegister) error {
					return fmt.Errorf("test error from registerInAnnotation()")
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
			},
			func() *gomonkey.Patches {
				var l *logrus.Logger
				return gomonkey.ApplyMethod(l, "Fatalln", func(_ *logrus.Logger, args ...interface{}) {
					fmt.Println("patched Fatalln")
					return
				})
			},
		},
		expectedStop: true,
	},
	{
		desc: "checking success then failure sequence",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var r *DeviceRegister
				count := 0
				return gomonkey.ApplyPrivateMethod(r, "registerInAnnotation", func(*DeviceRegister) error {
					count++
					if count == 1 {
						return nil
					}
					return fmt.Errorf("test error after success")
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(time.Sleep, func(time.Duration) {})
			},
			func() *gomonkey.Patches {
				var l *logrus.Logger
				return gomonkey.ApplyMethod(l, "Fatalln", func(_ *logrus.Logger, args ...interface{}) {
					fmt.Println("patched Fatalln")
					return
				})
			},
		},
		expectedStop: true,
	},
}

func watchAndRegisterHelper(r *DeviceRegister, ev chan<- int, t *testing.T) {
	if r == nil {
		t.Error("internal error")
		return
	}
	r.watchAndRegister()
	ev <- 1
}

func TestWatchAndRegister(t *testing.T) {
	for i := range registerTestCasesForWatchAndRegister {
		tc := registerTestCasesForWatchAndRegister[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			defer ResetPatches(appliedPatches)
			r := &DeviceRegister{}
			evDone := make(chan int)
			event := false
			go watchAndRegisterHelper(r, evDone, t)
			select {
			case <-evDone:
				event = true
			case <-time.After(2 * time.Second):
				t.Error("goroutine not finished by timeout")
			}
			if event == true && tc.expectedStop != true {
				t.Error("goroutine shouldn't finished")
			} else if event == false && tc.expectedStop == true {
				t.Error("goroutine should finished")
			} else {
				// event matches expectedStop; no assertion needed
			}
		})
	}
}

func TestRegisterStart(t *testing.T) {
	dr := NewDeviceRegister(&DeviceCache{})
	started := false
	patch := gomonkey.ApplyPrivateMethod(dr, "watchAndRegister", func() {
		started = true
	})
	defer patch.Reset()
	dr.Start()
	time.Sleep(100 * time.Millisecond)
	if !started {
		t.Errorf("expected watchAndRegister to be called")
	}
}

func TestRegisterApiDevices(t *testing.T) {
	dc := &DeviceCache{}
	dr := &DeviceRegister{deviceCache: dc}
	devs := []*xpu.Device{{}}
	patchGet := gomonkey.ApplyMethodFunc(dc, "GetCache", func() []*xpu.Device {
		return devs
	})
	defer patchGet.Reset()
	patchGetInfo := gomonkey.ApplyFunc(xpu.GetDeviceInfo, func(_ []*xpu.Device) []*types.DeviceInfo {
		return []*types.DeviceInfo{{Index: 0}}
	})
	defer patchGetInfo.Reset()
	result := dr.apiDevices()
	if len(result) != 1 {
		t.Errorf("expected 1 device, got %d", len(result))
	}
}
