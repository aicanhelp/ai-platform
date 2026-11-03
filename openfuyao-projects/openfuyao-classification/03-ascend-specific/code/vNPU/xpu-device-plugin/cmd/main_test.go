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

package main

import (
	"flag"
	"fmt"
	"os"
	"reflect"
	"syscall"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/fsnotify/fsnotify"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"huawei.com/vxpu-device-plugin/pkg/api/runtime/service"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
	"huawei.com/vxpu-device-plugin/watchers"
)

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

func basePatches() []func() *gomonkey.Patches {
	return []func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(syscall.Umask, func(_ int) int { return 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(xpu.Init, func() error { return nil })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(xpu.Uninit, func() error { return nil })
		},
	}
}

type eventTestCase struct {
	desc     string
	expected bool
	send     func(w *fsnotify.Watcher, s chan os.Signal)
}

func runEventTest(t *testing.T, tc eventTestCase) {
	t.Helper()
	w := &fsnotify.Watcher{Events: make(chan fsnotify.Event), Errors: make(chan error)}
	s := make(chan os.Signal)
	pi := &plugin.DevicePlugin{}
	var res bool
	done := make(chan bool)
	go func() {
		res = events(w, s, pi)
		done <- true
	}()
	tc.send(w, s)
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("event loop should exit, but timeout")
		return
	}
	if res != tc.expected {
		t.Errorf("expected %v, got %v", tc.expected, res)
	}
}

func devicesPatch() func() *gomonkey.Patches {
	return func() *gomonkey.Patches {
		return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DevicePlugin{}), "Devices", func(_ *plugin.DevicePlugin) []*xpu.Device {
			return []*xpu.Device{{Device: v1beta1.Device{ID: "test-device", Health: v1beta1.Healthy}}}
		})
	}
}

var testCasesForEventsWatcher = []struct {
	desc     string
	signal   string
	expected bool
}{
	{
		desc:     "test for checking fsnotify event",
		signal:   "fsnotify",
		expected: true,
	},
	{
		desc:     "test for checking sighup event",
		signal:   "sighup",
		expected: true,
	},
	{
		desc:     "test for checking sighup event",
		signal:   "sigusr",
		expected: false,
	},
}

func TestEventsWatcher(t *testing.T) {
	for i := range testCasesForEventsWatcher {
		tc := testCasesForEventsWatcher[i]
		t.Run(tc.desc, func(t *testing.T) {
			w := &fsnotify.Watcher{Events: make(chan fsnotify.Event)}
			s := make(chan os.Signal)
			pi := &plugin.DevicePlugin{}
			var res bool
			eventDoneCh := make(chan bool)
			go func() {
				res = events(w, s, pi)
				eventDoneCh <- true
			}()
			switch tc.signal {
			case "fsnotify":
				w.Events <- fsnotify.Event{Name: v1beta1.KubeletSocket, Op: fsnotify.Create}
			case "sighup":
				s <- syscall.SIGHUP
			default:
				s <- syscall.SIGUSR1
			}
			select {
			case <-eventDoneCh:
			case <-time.After(2 * time.Second):
				t.Error("event loop should exit, but timeout")
				return
			}
			if res != tc.expected {
				t.Error("event loop should with true status")
			}
		})
	}
}

func TestEventsWatcherNegativeCases(t *testing.T) {
	cases := []eventTestCase{
		{
			desc:     "test watcher error",
			expected: false,
			send: func(w *fsnotify.Watcher, s chan os.Signal) {
				if s == nil {
					return
				}
				w.Errors <- fmt.Errorf("test watcher error")
				s <- syscall.SIGTERM
			},
		},
		{
			desc:     "test non-kubelet socket",
			expected: false,
			send: func(w *fsnotify.Watcher, s chan os.Signal) {
				if s == nil {
					return
				}
				w.Events <- fsnotify.Event{Name: "/tmp/other.sock", Op: fsnotify.Create}
				s <- syscall.SIGTERM
			},
		},
		{
			desc:     "test non-create op on kubelet socket",
			expected: false,
			send: func(w *fsnotify.Watcher, s chan os.Signal) {
				if s == nil {
					return
				}
				w.Events <- fsnotify.Event{Name: v1beta1.KubeletSocket, Op: fsnotify.Remove}
				s <- syscall.SIGTERM
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.desc, func(t *testing.T) {
			runEventTest(t, tc)
		})
	}
}

func TestStartXpuInitError(t *testing.T) {
	patches := ApplyPatches([]func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(syscall.Umask, func(_ int) int { return 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(log.InitLogging, func(_ string) error { return nil })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(xpu.Init, func() error {
				return fmt.Errorf("mock init error")
			})
		},
	})
	defer ResetPatches(patches)

	err := start()
	if err == nil {
		t.Errorf("expected error from xpu.Init, got nil")
	}
}

func TestStartFSWatcherError(t *testing.T) {
	patches := ApplyPatches(append(basePatches(), []func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(watchers.NewFSWatcher, func(_ ...string) (*fsnotify.Watcher, error) {
				return nil, fmt.Errorf("mock fs watcher error")
			})
		},
	}...))
	defer ResetPatches(patches)

	err := start()
	if err == nil {
		t.Errorf("expected error from NewFSWatcher, got nil")
	}
}

func startCommonPatches(sigs chan os.Signal) []func() *gomonkey.Patches {
	return append(basePatches(), []func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(watchers.NewFSWatcher, func(_ ...string) (*fsnotify.Watcher, error) {
				return &fsnotify.Watcher{Events: make(chan fsnotify.Event), Errors: make(chan error)}, nil
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&fsnotify.Watcher{}), "Close", func(_ *fsnotify.Watcher) error {
				return nil
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(watchers.NewOSWatcher, func(_ ...os.Signal) chan os.Signal {
				return sigs
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(util.SetNodeConfig, func() { _ = 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(plugin.NewDeviceCache, func() *plugin.DeviceCache {
				return &plugin.DeviceCache{}
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DeviceCache{}), "Start", func(_ *plugin.DeviceCache) { _ = 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DeviceCache{}), "Stop", func(_ *plugin.DeviceCache) { _ = 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(plugin.NewDeviceRegister, func(_ *plugin.DeviceCache) *plugin.DeviceRegister {
				return &plugin.DeviceRegister{}
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DeviceRegister{}), "Start", func(_ *plugin.DeviceRegister) { _ = 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(service.Start, func() { _ = 0 })
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(plugin.NewDevicePlugin, func(_ string, _ *plugin.DeviceCache, _ string) *plugin.DevicePlugin {
				return &plugin.DevicePlugin{}
			})
		},
	}...)
}

func TestStartNoDevices(t *testing.T) {
	sigs := make(chan os.Signal, 1)
	commonPatches := startCommonPatches(sigs)
	allPatches := append(commonPatches, []func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DevicePlugin{}), "Devices", func(_ *plugin.DevicePlugin) []*xpu.Device {
				return []*xpu.Device{}
			})
		},
	}...)
	patches := ApplyPatches(allPatches)
	defer ResetPatches(patches)

	err := start()
	if err == nil {
		t.Errorf("expected error for no devices, got nil")
	}
}

func TestStartPluginStartError(t *testing.T) {
	sigs := make(chan os.Signal, 1)
	commonPatches := startCommonPatches(sigs)
	allPatches := append(commonPatches, []func() *gomonkey.Patches{
		devicesPatch(),
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DevicePlugin{}), "Start", func(_ *plugin.DevicePlugin) error {
				return fmt.Errorf("mock plugin start error")
			})
		},
	}...)
	patches := ApplyPatches(allPatches)
	defer ResetPatches(patches)

	err := start()
	if err == nil {
		t.Errorf("expected error from pluginInst.Start, got nil")
	}
}

func TestStartSuccessThenShutdown(t *testing.T) {
	sigs := make(chan os.Signal, 1)
	commonPatches := startCommonPatches(sigs)
	allPatches := append(commonPatches, []func() *gomonkey.Patches{
		devicesPatch(),
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DevicePlugin{}), "Start", func(_ *plugin.DevicePlugin) error {
				return nil
			})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyMethod(reflect.TypeOf(&plugin.DevicePlugin{}), "Stop", func(_ *plugin.DevicePlugin) { _ = 0 })
		},
	}...)
	patches := ApplyPatches(allPatches)
	defer ResetPatches(patches)

	go func() {
		sigs <- syscall.SIGTERM
	}()

	err := start()
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}
}

func TestMainSuccess(t *testing.T) {
	originalResourceName := resourceName
	defer func() { resourceName = originalResourceName }()

	flag.CommandLine = flag.NewFlagSet("test", flag.ExitOnError)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(flag.Parse, func() { _ = 0 })
	patches.ApplyFunc(start, func() error { return nil })
	patches.ApplyFunc(log.Fatalln, func(_ ...interface{}) { t.Fatal("Fatalln should not be called") })

	main()

	if resourceName != xpu.VxpuNumber {
		t.Errorf("expected resourceName %s, got %s", xpu.VxpuNumber, resourceName)
	}
}

func TestStartLogInitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(syscall.Umask, func(_ int) int { return 0 })
	patches.ApplyFunc(log.InitLogging, func(_ string) error { return fmt.Errorf("mock log init error") })
	exitCalled := false
	patches.ApplyFunc(os.Exit, func(_ int) { exitCalled = true })

	start()
	if !exitCalled {
		t.Errorf("expected os.Exit to be called")
	}
}

func TestMainStartError(t *testing.T) {
	fatallnCalled := false

	flag.CommandLine = flag.NewFlagSet("test", flag.ExitOnError)

	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(flag.Parse, func() { _ = 0 })
	patches.ApplyFunc(start, func() error { return fmt.Errorf("mock start error") })
	patches.ApplyFunc(log.Fatalln, func(_ ...interface{}) { fatallnCalled = true })

	main()

	if !fatallnCalled {
		t.Errorf("expected Fatalln to be called when start returns error")
	}
}
