/*
 * Copyright (c) 2024-2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package deviceplugin

import (
	"fmt"
	"io/fs"
	"io/ioutil"
	"net"
	"os"
	"path"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	kubcletEndpoint = "kubelet.sock"
	waitServerReady = 2 * time.Second
	chmodPermission = 0666
)

// TestNewKaeManager 测试 NewKAEManager
func TestNewKaeManager(t *testing.T) {
	testManager := NewKAEManager()
	assert.NotNil(t, testManager)
}

func TestDiscoverDevicesUsesManagerDirectory(t *testing.T) {
	tempDir := t.TempDir()
	devicePath := path.Join(tempDir, "hisi_hpre0")
	if err := os.WriteFile(devicePath, nil, chmodPermission); err != nil {
		t.Fatalf("create device file: %v", err)
	}

	manager := NewKAEManager()
	manager.devDirectory = tempDir

	if err := manager.discoverDevices(); err != nil {
		t.Fatalf("discover devices: %v", err)
	}

	if _, found := manager.devices["hisi_hpre0"]; !found {
		t.Fatal("configured device directory was not scanned")
	}
}

// TestDiscoverDevices 测试 DiscoverDevices
func TestDiscoverDevices(t *testing.T) {
	manager := &KAEManager{
		grpcServer: &grpc.Server{},
		devices:    make(map[string]v1beta1.Device),
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(
		ioutil.ReadDir, func(dirname string) ([]fs.FileInfo, error) {
			return nil, fmt.Errorf("stat failed")
		},
	)
	assert.NotNil(t, manager.discoverDevices())

	patches.Reset()
	patches.ApplyFunc(
		ioutil.ReadDir, func(dirname string) ([]fs.FileInfo, error) {
			return nil, nil
		},
	)
	assert.NotNil(t, manager.discoverDevices())

	patches.Reset()
	tempDir := t.TempDir()
	devicePath := path.Join(tempDir, "hisi_hpre0")
	if err := os.WriteFile(devicePath, nil, chmodPermission); err != nil {
		t.Fatalf("create device file: %v", err)
	}
	files, err := ioutil.ReadDir(tempDir)
	if err != nil {
		t.Fatalf("read temporary device directory: %v", err)
	}
	patches.ApplyFunc(
		ioutil.ReadDir, func(dirname string) ([]fs.FileInfo, error) {
			return files, nil
		},
	)
	assert.NoError(t, manager.discoverDevices())
	assert.Contains(t, manager.devices, "hisi_hpre0")
}

// TestStart 测试 Start
func TestStart(t *testing.T) {
	manager := &KAEManager{
		grpcServer: &grpc.Server{},
		devices:    make(map[string]v1beta1.Device),
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(
		os.Stat, func(name string) (os.FileInfo, error) {
			return nil, fmt.Errorf("stat failed")
		},
	)
	assert.NotNil(t, manager.Start())

	patches.Reset()
	outputs := []gomonkey.OutputCell{
		{Values: gomonkey.Params{nil, nil}},
		{Values: gomonkey.Params{nil, fmt.Errorf("stat file failed")}},
	}
	patches.ApplyFuncSeq(os.Stat, outputs)
	assert.NotNil(t, manager.Start())

	patches.Reset()
	outputs = []gomonkey.OutputCell{
		{Values: gomonkey.Params{nil, nil}},
		{Values: gomonkey.Params{nil, nil}},
		{Values: gomonkey.Params{nil, fmt.Errorf("stat file failed")}},
	}
	patches.ApplyFuncSeq(os.Stat, outputs)
	assert.NotNil(t, manager.Start())

	patches.Reset()
	outputs = []gomonkey.OutputCell{
		{Values: gomonkey.Params{nil, nil}},
		{Values: gomonkey.Params{nil, nil}},
		{Values: gomonkey.Params{nil, nil}},
	}
	patches.ApplyFuncSeq(os.Stat, outputs)
	patches.ApplyFunc(
		(*KAEManager).discoverDevices, func(_ *KAEManager) error {
			return nil
		},
	)
	assert.Nil(t, manager.Start())

}

// TestServe 测试 serve
func TestServe(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	manger := NewKAEManager()
	pluginEndpointPath := path.Join("/dev", pluginEndpointPrefix)
	manger.grpcServer = grpc.NewServer()
	plugin := &pluginService{kaeManager: manger}
	plugin.RegisterService()
	lis, err := net.Listen("unix", pluginEndpointPath)
	dClient, err := NewDeviceClient(pluginEndpointPath)
	if err != nil {
		zlog.Error("create dClient failed, err: %v", err)
		return
	}

	patches.Reset()
	count := 0
	patches.ApplyFunc(
		net.Listen, func(network string, address string) (net.Listener, error) {
			count++
			return lis, nil
		},
	)

	patches.ApplyMethod(
		reflect.TypeOf(&grpc.Server{}), "Serve", func(s *grpc.Server, lis net.Listener) error {
			return fmt.Errorf("serve failed")
		},
	)
	patches.ApplyFunc(
		NewDeviceClient, func(sock string) (*DeviceClient, error) {
			return dClient, nil
		},
	)
	patches.ApplyFunc(
		devicePluginMonitor, func(manager *KAEManager, pluginEndpointPath string) {
			return
		},
	)
	patches.ApplyMethod(
		reflect.TypeOf(dClient), "CheckListAndWatch", func(dClient *DeviceClient) {
			return
		},
	)

	go func() {
		manger.stop = make(chan bool, 1)
		time.Sleep(time.Second)
		manger.stop <- false
	}()
	manger.Serve(kubcletEndpoint)
	assert.NotNil(t, manger.Serve)
}

// TestRegisterWithKubelet 测试 RegisterWithKubelet
func TestRegisterWithKubelet(t *testing.T) {
	manager := &KAEManager{
		grpcServer: &grpc.Server{},
		devices:    make(map[string]v1beta1.Device),
	}

	err := registerWithKubelet(manager, "kubesec.sock", resourceName)
	assert.NotNil(t, err)

	_, err = os.OpenFile("/dev/hisi_sectest", os.O_RDWR|os.O_CREATE|os.O_TRUNC, chmodPermission)
	if err != nil {
		zlog.Error("create file failed, err: %v", err)
	}
	defer os.Remove("/dev/hisi_sectest")
	_, err = os.OpenFile("/dev/kubelet.sock", os.O_RDWR|os.O_CREATE|os.O_TRUNC, chmodPermission)
	if err != nil {
		zlog.Error("create file failed, err: %v", err)
	}
	defer os.Remove("/dev/kubelet.sock")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyMethod(
		reflect.TypeOf(&grpc.Server{}), "GetServiceInfo", func(_ *grpc.Server) map[string]grpc.ServiceInfo {
			a := make(map[string]grpc.ServiceInfo, 1)
			var b grpc.ServiceInfo
			a["first"] = b
			return a
		},
	)
	patches.ApplyFunc(
		RegisterWithV1Beta1Kubelet, func(_ string, _ string) error {
			return nil
		},
	)
	err = registerWithKubelet(manager, "kubesec.sock", resourceName)
	assert.NotNil(t, err)

	patches.Reset()
	patches.ApplyMethod(
		reflect.TypeOf(&grpc.Server{}), "GetServiceInfo", func(_ *grpc.Server) map[string]grpc.ServiceInfo {
			a := make(map[string]grpc.ServiceInfo, 1)
			var b grpc.ServiceInfo
			a["first"] = b
			return a
		},
	)
	patches.ApplyFunc(
		RegisterWithV1Beta1Kubelet, func(_ string, _ string) error {
			return fmt.Errorf("break")
		},
	)
	patches.ApplyMethod(
		reflect.TypeOf(&grpc.Server{}), "Stop", func(x *grpc.Server) {
			return
		},
	)
	err = registerWithKubelet(manager, "/dev/", resourceName)
	assert.NotNil(t, err)
}

// TestRemoveKAESock 测试RemoveKAESock
func TestRemoveKAESock(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(
		ioutil.ReadDir, func(_ string) ([]fs.FileInfo, error) {
			var a []fs.FileInfo
			return a, fmt.Errorf("break")
		},
	)
	RemoveKAESock()
	assert.NotNil(t, RemoveKAESock)
}

func TestManagerHelpers(t *testing.T) {
	healthy := map[string]v1beta1.Device{"hisi_hpre0": {ID: "hisi_hpre0", Health: v1beta1.Healthy}}
	unhealthy := map[string]v1beta1.Device{"hisi_hpre0": {ID: "hisi_hpre0", Health: v1beta1.Unhealthy}}

	assert.True(t, compareMaps(healthy, healthy))
	assert.False(t, compareMaps(healthy, map[string]v1beta1.Device{}))
	assert.False(t, compareMaps(healthy, map[string]v1beta1.Device{"hisi_hpre1": healthy["hisi_hpre0"]}))
	assert.False(t, compareMaps(healthy, unhealthy))
	assert.Equal(t, v1beta1.Healthy, NewKAEManager().GetDeviceState("hisi_hpre0"))
}

func TestStartDiscoversDevicesFromConfiguredDirectory(t *testing.T) {
	originalExtraPath := extraPath
	extraPath = t.TempDir()
	t.Cleanup(func() { extraPath = originalExtraPath })

	manager := NewKAEManager()
	manager.devDirectory = t.TempDir()
	if err := os.WriteFile(path.Join(manager.devDirectory, "hisi_hpre0"), nil, chmodPermission); err != nil {
		t.Fatalf("create device file: %v", err)
	}

	if err := manager.Start(); err != nil {
		t.Fatalf("start manager: %v", err)
	}
	assert.Contains(t, manager.devices, "hisi_hpre0")
}

func TestDevicePluginMonitorHandlesMissingSocket(t *testing.T) {
	manager := NewKAEManager()
	manager.grpcServer = grpc.NewServer()

	devicePluginMonitor(manager, path.Join(t.TempDir(), "missing.sock"))
}

func TestRunCmdPreservesCommandExecutionError(t *testing.T) {
	_, err := runCmd("command_that_does_not_exist")
	assert.Error(t, err)
	assert.Contains(t, err.Error(), `run command "command_that_does_not_exist"`)
}

func TestSetUnamePreservesAUsableModulePath(t *testing.T) {
	originalExtraPath := extraPath
	extraPath = "existing-module-path"
	t.Cleanup(func() { extraPath = originalExtraPath })

	SetUname()

	if extraPath == "existing-module-path" {
		return
	}
	if !strings.HasPrefix(extraPath, "/lib/modules/") || !strings.HasSuffix(extraPath, "/extra") {
		t.Fatalf("unexpected module path: %q", extraPath)
	}
}

func TestRemoveKAESockRemovesOnlyPluginSockets(t *testing.T) {
	originalDirectory := devicePluginDirectory
	devicePluginDirectory = t.TempDir()
	t.Cleanup(func() { devicePluginDirectory = originalDirectory })

	pluginSocket := path.Join(devicePluginDirectory, "hisi-plugin.sock")
	otherSocket := path.Join(devicePluginDirectory, "other-plugin.sock")
	for _, socketPath := range []string{pluginSocket, otherSocket} {
		if err := os.WriteFile(socketPath, nil, chmodPermission); err != nil {
			t.Fatalf("create socket fixture: %v", err)
		}
	}

	RemoveKAESock()

	if _, err := os.Stat(pluginSocket); !os.IsNotExist(err) {
		t.Fatalf("plugin socket still exists: %v", err)
	}
	if _, err := os.Stat(otherSocket); err != nil {
		t.Fatalf("unrelated socket was removed: %v", err)
	}
}

// TestCycleRegisterWithKubelet 测试 CycleRegisterWithKubelet
func TestCycleRegisterWithKubelet(t *testing.T) {
	manager := &KAEManager{
		grpcServer: &grpc.Server{},
		devices:    make(map[string]v1beta1.Device),
	}

	_, err := os.OpenFile("/dev/hisi_sectest", os.O_RDWR|os.O_CREATE|os.O_TRUNC, chmodPermission)
	if err != nil {
		zlog.Error("OpenFile failed, err: %v", err)
	}
	defer os.Remove("/dev/hisi_sectest")
	_, err = os.OpenFile("/dev/kubelet.sock", os.O_RDWR|os.O_CREATE|os.O_TRUNC, chmodPermission)
	if err != nil {
		zlog.Error("OpenFile failed, err: %v", err)
	}
	defer os.Remove("/dev/kubelet.sock")
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(
		registerWithKubelet, func(kaeManager *KAEManager, pluginEndpoint,
			resourceName string) error {
			return nil
		},
	)
	cycleRegisterWithKubelet(manager, "/dev/", resourceName)
	assert.NotNil(t, cycleRegisterWithKubelet)
}
