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
	"bufio"
	"context"
	"crypto/md5"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"google.golang.org/grpc"
	v1 "k8s.io/api/core/v1"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

var countRestart = -1
var event = make(chan int)

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

func TestDial(t *testing.T) {
	pluginInst := NewDevicePlugin("name", nil, "socket")
	patch := gomonkey.ApplyFunc(grpc.Dial, func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
		err := errors.New("test")
		return nil, err
	})
	defer patch.Reset()

	_, err := pluginInst.dial("test", time.Second)
	if err == nil {
		t.Error("mock failed")
	}
}

func TestInitialize(t *testing.T) {
	pl := NewDevicePlugin("name", nil, "socket")
	pl.initialize()
	if pl.server == nil || pl.health == nil || pl.stop == nil {
		t.Error("device plugin server haven't been initialized")
	}
}

func TestCleanUp(t *testing.T) {
	pl := NewDevicePlugin("name", nil, "socket")
	pl.initialize()
	pl.cleanup()
	if pl.server != nil || pl.health != nil || pl.stop != nil {
		t.Error("device plugin server haven't been cleaned up")
	}
}

func TestGetDevicePluginOptions(t *testing.T) {
	pl := NewDevicePlugin("name", nil, "socket")
	ops, _ := pl.GetDevicePluginOptions(context.TODO(), &v1beta1.Empty{})
	if ops == nil {
		t.Error("get device plugin options failed")
	}
}

func TestGetPreferredAllocation(t *testing.T) {
	pl := NewDevicePlugin("name", nil, "socket")
	ops, _ := pl.GetPreferredAllocation(context.TODO(), &v1beta1.PreferredAllocationRequest{})
	if ops == nil {
		t.Error("device plugin server haven't been cleaned up")
	}
}

var testCasesForServe = []struct {
	desc          string
	patches       []func() *gomonkey.Patches
	expected      int
	exitWithError int
}{
	{
		desc: "checking os operations error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Remove, func(string) error {
					fmt.Println("os remove")
					err := errors.New("os remove error")
					return err
				})
			},
		},
		expected:      1,
		exitWithError: -1,
	},
	{
		desc: "checking net operations error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Remove, func(string) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(net.Listen, func(string, string) (net.Listener, error) {
					err := errors.New("net/listen error")
					return nil, err
				})
			},
		},
		expected:      1,
		exitWithError: -1,
	},
	{
		desc: "checking normal test case",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Remove, func(string) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(net.Listen, func(string, string) (net.Listener, error) {
					return nil, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Chmod, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return nil, nil
				})
			},
		},
		expected:      0,
		exitWithError: 0,
	},
	{
		desc: "checking serve with dial err",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Remove, func(string) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(net.Listen, func(string, string) (net.Listener, error) {
					return nil, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Chmod, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return nil, fmt.Errorf("test error from dial method")
				})
			},
		},
		expected:      1,
		exitWithError: 0,
	},
	{
		desc: "checking test case with grpc server serve error",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Remove, func(string) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(net.Listen, func(string, string) (net.Listener, error) {
					return nil, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Chmod, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return nil, nil
				})
			},
		},
		expected:      0,
		exitWithError: 1,
	},
}

func TestServe(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForServe {
		tc := testCasesForServe[i]
		var g *grpc.Server
		p := gomonkey.ApplyMethod(g, "Serve", func(_ *grpc.Server, lis net.Listener) error {
			countRestart++
			var ev int
			ev = <-event
			if ev == 0 {
				return nil
			} else if ev == 1 {
				return fmt.Errorf("exit Serve with error")
			}
			return nil
		})
		t.Run(tc.desc, func(t *testing.T) {
			countRestart = -1
			pl := NewDevicePlugin("name", nil, "socket")
			pl.initialize()
			appliedPatches := ApplyPatches(tc.patches)
			err := pl.serve()
			if tc.exitWithError >= 0 {
				event <- tc.exitWithError
			}
			if tc.exitWithError == 1 {
				time.Sleep(1 * time.Millisecond)
			}
			if tc.exitWithError == 1 && countRestart <= 0 {
				t.Error("error grpc restart failed", tc.desc)
			}
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			p.Reset()
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForStart = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
}{
	{
		desc: "checking DevicePlugin serve() error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "serve", func(_ *DevicePlugin) error {
					return fmt.Errorf("test error from serve() function")
				})
			},
		},
		expected: 1,
	},
	{
		desc: "checking DevicePlugin Register() error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "serve", func(_ *DevicePlugin) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "register", func(_ *DevicePlugin) error {
					return fmt.Errorf("test error from Register() function")
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyMethod(t, "Stop", func(_ *DevicePlugin) {
				})
			},
		},
		expected: 1,
	},
	{
		desc: "checking happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "serve", func(_ *DevicePlugin) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "register", func(_ *DevicePlugin) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyMethod(t, "Stop", func(_ *DevicePlugin) {
				})
			},
		},
		expected: 0,
	},
}

func TestStart(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForStart {
		tc := testCasesForStart[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			pl := NewDevicePlugin("name", nil, "socket")
			pl.deviceCache = NewDeviceCache()
			err := pl.Start()
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

func TestPluginStop(t *testing.T) {
	if t == nil {
		return
	}
	{
		pl := NewDevicePlugin("name", nil, "socket")
		pl.deviceCache = NewDeviceCache()
		pl.deviceCache.AddNotifyChannel("plugin", make(chan *xpu.Device))
		pl.Stop()
		if pl.deviceCache.notifyCh["plugin"] == nil {
			t.Error("error: if error channel will not be deleted")
		}
	}

	{
		p := gomonkey.ApplyFunc(grpc.NewServer, func(...grpc.ServerOption) *grpc.Server {
			return &grpc.Server{}
		})
		var s *grpc.Server
		p1 := gomonkey.ApplyMethod(s, "Stop", func(_ *grpc.Server) {
		})
		pl := NewDevicePlugin("name", nil, "socket")
		pl.initialize()
		pl.deviceCache = NewDeviceCache()
		pl.deviceCache.AddNotifyChannel("plugin", make(chan *xpu.Device))
		pl.Stop()
		if pl.deviceCache.notifyCh["plugin"] != nil {
			t.Error("error: after stop channel should be removed")
		}
		p.Reset()
		p1.Reset()
	}
}

var mrc mockRegClient
var mrc1 mockRegClient1

type mockRegClient struct{}

func (mrc *mockRegClient) Register(ctx context.Context, in *v1beta1.RegisterRequest,
	opts ...grpc.CallOption) (*v1beta1.Empty, error) {
	fmt.Println("patched register!!!")
	return &v1beta1.Empty{}, fmt.Errorf("error for client connection")
}

type mockRegClient1 struct{}

func (mrc *mockRegClient1) Register(ctx context.Context, in *v1beta1.RegisterRequest,
	opts ...grpc.CallOption) (*v1beta1.Empty, error) {
	fmt.Println("patched register1!!!")
	return &v1beta1.Empty{}, nil
}

var testCasesForRegister = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
}{
	{
		desc: "checking DevicePlugin Register() error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return nil, fmt.Errorf("test error from dial method")
				})
			},
		},
		expected: 1,
	},
	{
		desc: "checking DevicePlugin Register() nil pointer defer error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return nil, nil
				})
			},
		},
		expected: 1,
	},
	{
		desc: "checking DevicePlugin Register() client registration error handling",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return &grpc.ClientConn{}, nil
				})
			},
			func() *gomonkey.Patches {
				var t *grpc.ClientConn
				return gomonkey.ApplyMethod(t, "Close", func(_ *grpc.ClientConn) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *grpc.ClientConn
				return gomonkey.ApplyMethod(t, "Invoke", func(_ *grpc.ClientConn, ctx context.Context,
					method string, args, reply interface{}, opts ...grpc.CallOption) error {
					return fmt.Errorf("mock invoke error")
				})
			},
			func() *gomonkey.Patches {
				mrc := &mockRegClient{}
				return gomonkey.ApplyFunc(v1beta1.NewRegistrationClient, func(cc *grpc.ClientConn) v1beta1.RegistrationClient {
					return mrc
				})
			},
		},
		expected: 1,
	},
	{
		desc: "checking DevicePlugin Register() happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "dial", func(_ *DevicePlugin, _ string,
					_ time.Duration) (*grpc.ClientConn, error) {
					return &grpc.ClientConn{}, nil
				})
			},
			func() *gomonkey.Patches {
				var t *grpc.ClientConn
				return gomonkey.ApplyMethod(t, "Close", func(_ *grpc.ClientConn) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var t *grpc.ClientConn
				return gomonkey.ApplyMethod(t, "Invoke", func(_ *grpc.ClientConn, ctx context.Context,
					method string, args, reply interface{}, opts ...grpc.CallOption) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				mrc1 := &mockRegClient1{}
				return gomonkey.ApplyFunc(v1beta1.NewRegistrationClient, func(cc *grpc.ClientConn) v1beta1.RegistrationClient {
					return mrc1
				})
			},
		},
		expected: 0,
	},
}

func TestRegister(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForRegister {
		tc := testCasesForRegister[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			pl := NewDevicePlugin("name", nil, "socket")
			err := pl.register()
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

type MockLWServer struct {
	grpc.ServerStream
}

var callCounter int

func NewMockServer() v1beta1.DevicePlugin_ListAndWatchServer {
	return &MockLWServer{}
}

func (mock *MockLWServer) Send(*v1beta1.ListAndWatchResponse) error {
	callCounter++
	return nil
}

var testCasesForListAndWatch = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
	sig      string
}{
	{
		desc: "test stop signal in ListAndWatch()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "apiDevices", func(_ *DevicePlugin) []*v1beta1.Device {
					return []*v1beta1.Device{}
				})
			},
		},
		expected: 0,
		sig:      "stop",
	},
}

func TestListAndWatchStop(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForListAndWatch {
		tc := testCasesForListAndWatch[i]
		t.Run(tc.desc, func(t *testing.T) {
			callCounter = 0
			appliedPatches := ApplyPatches(tc.patches)
			exit := make(chan int)
			var err error
			ms := NewMockServer()
			lw := func(pl *DevicePlugin) {
				err = pl.ListAndWatch(&v1beta1.Empty{}, ms)
				exit <- 1
			}
			pl := NewDevicePlugin("name", nil, "socket")
			pl.initialize()
			go lw(pl)
			pl.stop <- 1
			select {
			case <-exit:
			case <-time.After(time.Second * 2):
				t.Error("error in test case: ", tc.desc, " timeout")
			}
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForListAndWatchHealth = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
	sig      string
}{
	{
		desc: "test health signal in ListAndWatch()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var t *DevicePlugin
				return gomonkey.ApplyPrivateMethod(t, "apiDevices", func(_ *DevicePlugin) []*v1beta1.Device {
					return []*v1beta1.Device{}
				})
			},
		},
		expected: 0,
		sig:      "health",
	},
}

func TestListAndWatchHealth(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForListAndWatchHealth {
		tc := testCasesForListAndWatchHealth[i]
		t.Run(tc.desc, func(t *testing.T) {
			callCounter = 0
			appliedPatches := ApplyPatches(tc.patches)
			exit := make(chan int)
			var err error
			ms := NewMockServer()
			lw := func(pl *DevicePlugin) {
				err = pl.ListAndWatch(&v1beta1.Empty{}, ms)
				exit <- 1
			}
			pl := NewDevicePlugin("name", nil, "socket")
			pl.initialize()
			go lw(pl)
			pl.health <- &xpu.Device{}
			time.Sleep(2 * time.Millisecond)
			if callCounter > 1 {
				err = error(nil)
			} else {
				err = fmt.Errorf("mock server should call 2 times in ListAndWatch func")
			}

			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForWriteVxpu = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
	sig      string
}{
	{
		desc: "test error at mkdir",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
					err := errors.New("os mkdir error")
					return err
				})
			},
		},
		expected: 1,
	},
	{
		desc: "test error at open file",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.OpenFile, func(string, int, os.FileMode) (*os.File, error) {
					return nil, fmt.Errorf("test open file error")
				})
			},
		},
		expected: 1,
	},
	{
		desc: "test error at write string",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.OpenFile, func(string, int, os.FileMode) (*os.File, error) {
					return &os.File{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(bufio.NewWriter, func(io.Writer) *bufio.Writer {
					return &bufio.Writer{}
				})
			},
			func() *gomonkey.Patches {
				var w *bufio.Writer
				return gomonkey.ApplyMethod(w, "WriteString", func(_ *bufio.Writer, _ string) (int, error) {
					return 0, fmt.Errorf("write error")
				})
			},
		},
		expected: 1,
	},
	{
		desc: "happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.OpenFile, func(string, int, os.FileMode) (*os.File, error) {
					return &os.File{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(bufio.NewWriter, func(io.Writer) *bufio.Writer {
					return &bufio.Writer{}
				})
			},
			func() *gomonkey.Patches {
				var w *bufio.Writer
				return gomonkey.ApplyMethod(w, "WriteString", func(_ *bufio.Writer, _ string) (int, error) {
					return 1, nil
				})
			},
			func() *gomonkey.Patches {
				var w *bufio.Writer
				return gomonkey.ApplyMethod(w, "Flush", func(_ *bufio.Writer) error {
					return nil
				})
			},
		},
		expected: 0,
	},
}

func TestWriteVxpuConfig(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForWriteVxpu {
		tc := testCasesForWriteVxpu[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			contDevs := types.ContainerDevices{
				{Index: 0, Usedmem: 500, Usedcores: 2},
			}
			err := writeVxpuInfo("test", contDevs)
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

func fileCheckSum(filename string) string {
	f, err := os.Open(filename)
	if err != nil {
		return ""
	}
	defer f.Close()

	h := md5.New()
	if _, err = io.Copy(h, f); err != nil {
		return ""
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

func TestWriteVxpuIdsConfig(t *testing.T) {
	if t == nil {
		fmt.Println("param *testing.T object is nil!")
		return
	}
	cd0 := types.ContainerDevice{
		Index:     0,
		UUID:      "NPU-ca1387d2-33e9-1f4a-d66c-512e5273d689",
		Type:      "ASCEND",
		Usedmem:   1024,
		Usedcores: 10,
		Vid:       10,
	}
	cd1 := types.ContainerDevice{
		Index:     1,
		UUID:      "NPU-ca1387d2-33e9-1f4a-d66c-512e5273d690",
		Type:      "ASCEND",
		Usedmem:   2048,
		Usedcores: 20,
		Vid:       11,
	}
	contDevs := types.ContainerDevices{cd0, cd1}
	err := writeVxpuInfo(".", contDevs)
	if err == nil && fileCheckSum("./npu_info.config") == fileCheckSum("./expected.npu_info.config") {
		t.Log("test succeed")
	} else {
		t.Error("test failed")
	}
	err = os.Remove("./npu_info.config")
	if err != nil {
		t.Errorf("test failed, remove npu_info.config error: %v", err)
	}
}

var testCasesForCreateDirAndWriteFile = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
	sig      string
}{
	{
		desc: "test error handling at writeVxpuConfig()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(writeVxpuInfo, func(string, types.ContainerDevices) error {
					return fmt.Errorf("test error in function writeVxpuConfig()")
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
		},
		expected: 1,
	},
	{
		desc: "happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(writeVxpuInfo, func(string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.MkdirAll, func(string, os.FileMode) error {
					return nil
				})
			},
		},
		expected: 0,
	},
}

func TestCreateDirAndWriteFile(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForCreateDirAndWriteFile {
		tc := testCasesForCreateDirAndWriteFile[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			cd0 := types.ContainerDevice{
				Index:     0,
				UUID:      "12345678-abcd-bcde-cdef-123456789000",
				Type:      "ASCEND",
				Usedmem:   1024,
				Usedcores: 10,
				Vid:       10,
			}
			err := createDirAndWriteFile("test_pod_id", "test_container_name", types.ContainerDevices{cd0})

			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForCreateCas = []struct {
	desc             string
	patches          []func() *gomonkey.Patches
	expectedErr      bool
	expectedMountLen int
}{
	{
		desc: "preload non-full card",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return false
				})
			},
		},
		expectedErr:      false,
		expectedMountLen: 7,
	},
	{
		desc: "override non-full card",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return false
				})
			},
		},
		expectedErr:      false,
		expectedMountLen: 7,
	},
	{
		desc: "preload mode full card",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return true
				})
			},
		},
		expectedErr:      false,
		expectedMountLen: 1,
	},
	{
		desc: "override mode full single card",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return true
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(isSingleCard, func(devReq types.ContainerDevices) bool {
					return true
				})
			},
		},
		expectedErr:      false,
		expectedMountLen: 1,
	},
	{
		desc: "override mode full multiple card",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(filepath.Clean, func(string) string {
					return "test"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return true
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(isSingleCard, func(devReq types.ContainerDevices) bool {
					return false
				})
			},
		},
		expectedErr:      false,
		expectedMountLen: 1,
	},
}

func TestCreateContainerAllocateResponse(t *testing.T) {
	if t == nil {
		return
	}
	pl := NewDevicePlugin("name", nil, "socket")
	for i := range testCasesForCreateCas {
		tc := testCasesForCreateCas[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			resp, respErr := pl.createContainerAllocateResponse("test_pod_id", "test_container_name", types.ContainerDevices{})

			if tc.expectedErr {
				if respErr == nil {
					t.Error("expect have error")
				}
			} else {
				if len(resp.Mounts) != tc.expectedMountLen {
					t.Errorf("expect mount len %d, but got %d", tc.expectedMountLen, len(resp.Mounts))
				}
			}

			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForIsFullCard = []struct {
	desc     string
	devReq   types.ContainerDevices
	expected bool
}{
	{
		desc:     "is full Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0, Usedcores: xpu.FullCardCores, Usedmem: 2048}},
		expected: true,
	},
	{
		desc:     "is not full Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0, Usedcores: xpu.FullCardCores, Usedmem: 1024}},
		expected: false,
	},
	{
		desc:     "is not full Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0, Usedcores: 1, Usedmem: 2048}},
		expected: false,
	},
	{
		desc:     "is not full Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0, Usedcores: 1, Usedmem: 1024}},
		expected: false,
	},
}

func TestIsFullCard(t *testing.T) {
	if t == nil {
		return
	}
	appliedPatches := ApplyPatches([]func() *gomonkey.Patches{
		func() *gomonkey.Patches {
			var p *DeviceCache
			return gomonkey.ApplyMethod(p, "GetCache",
				func(*DeviceCache) []*xpu.Device {
					return nil
				})
		},
		func() *gomonkey.Patches {
			return gomonkey.ApplyFunc(xpu.GetDeviceInfo, func(devs []*xpu.Device) []*types.DeviceInfo {
				dev0 := types.DeviceInfo{Index: 0, Devmem: 2048}
				dev1 := types.DeviceInfo{Index: 1, Devmem: 2048}
				return []*types.DeviceInfo{&dev0, &dev1}
			})
		},
	})

	pl := NewDevicePlugin("name", nil, "socket")
	pl.deviceCache = NewDeviceCache()
	for i := range testCasesForIsFullCard {
		tc := testCasesForIsFullCard[i]
		t.Run(tc.desc, func(t *testing.T) {
			res := pl.isFullCard(tc.devReq)
			if res != tc.expected {
				t.Errorf("expect %v, but got %v", tc.expected, res)
			}
		})
	}

	ResetPatches(appliedPatches)
}

var testCasesForIsSingleCard = []struct {
	desc     string
	devReq   types.ContainerDevices
	expected bool
}{
	{
		desc:     "is single Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0}},
		expected: true,
	},
	{
		desc:     "is multiple Card",
		devReq:   types.ContainerDevices{types.ContainerDevice{Index: 0}, types.ContainerDevice{Index: 1}},
		expected: false,
	},
}

func TestIsSingleCard(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForIsSingleCard {
		tc := testCasesForIsSingleCard[i]
		t.Run(tc.desc, func(t *testing.T) {
			res := isSingleCard(tc.devReq)
			if res != tc.expected {
				t.Errorf("expect %v, but got %v", tc.expected, res)
			}
		})
	}
}

var testCasesForAllocate = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	expected int
	allocReq *v1beta1.AllocateRequest
}{
	{
		desc: "error handling at wrong number of container requests",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{}, {}}},
	},
	{
		desc: "error handling at util.GetPendingPod()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return nil, fmt.Errorf("test error getting pending pod")
				})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{}}},
	},
	{
		desc: "error handling when user pod is nil at GetPendingNode()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return nil, nil
				})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{}}},
	},
	{
		desc: "error handling when GetNextDeviceRequest() error not nil",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return &v1.Pod{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetNextDeviceRequest, func(string,
					v1.Pod) (v1.Container, types.ContainerDevices, error) {
					return v1.Container{}, types.ContainerDevices{}, fmt.Errorf("test error")
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PodAllocationFailed, func(string, *v1.Pod) {})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{}}},
	},
	{
		desc: "error handling at GetNextDeviceRequest() len(devReq) != len(reqs.ContainerRequests[idx].DevicesIDs)",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return &v1.Pod{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PodAllocationFailed, func(string, *v1.Pod) {})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetNextDeviceRequest, func(string,
					v1.Pod) (v1.Container, types.ContainerDevices, error) {
					return v1.Container{}, types.ContainerDevices{types.ContainerDevice{}, types.ContainerDevice{}}, nil
				})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{}}},
	},
	{
		desc: "error handling at EraseNextDeviceTypeFromAnnotation()",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return &v1.Pod{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PodAllocationFailed, func(string, *v1.Pod) {})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetNextDeviceRequest, func(string,
					v1.Pod) (v1.Container, types.ContainerDevices, error) {
					return v1.Container{}, types.ContainerDevices{types.ContainerDevice{}}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.EraseNextDeviceTypeFromAnnotation, func(string, v1.Pod) error {
					return fmt.Errorf("test error from util.EraseNextDeviceTypeFromAnnotation()")
				})
			},
		},
		expected: 1,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{DevicesIDs: []string{"1"}}}},
	},
	{
		desc: "happy path",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(os.Getenv, func(string) string {
					return "testnode"
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetPendingPod, func(string) (*v1.Pod, error) {
					return &v1.Pod{}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PodAllocationFailed, func(string, *v1.Pod) {})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.GetNextDeviceRequest, func(string,
					v1.Pod) (v1.Container, types.ContainerDevices, error) {
					return v1.Container{}, types.ContainerDevices{types.ContainerDevice{}}, nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.EraseNextDeviceTypeFromAnnotation, func(string, v1.Pod) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(createDirAndWriteFile, func(string, string, types.ContainerDevices) error {
					return nil
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "isFullCard", func(_ *DevicePlugin) bool {
					return false
				})
			},
			func() *gomonkey.Patches {
				var p *DevicePlugin
				return gomonkey.ApplyPrivateMethod(p, "createContainerAllocateResponse",
					func(_ *DevicePlugin) (*v1beta1.ContainerAllocateResponse, error) {
						return &v1beta1.ContainerAllocateResponse{}, nil
					})
			},
			func() *gomonkey.Patches {
				return gomonkey.ApplyFunc(util.PodAllocationTrySuccess, func(string, *v1.Pod) {
				})
			},
		},
		expected: 0,
		allocReq: &v1beta1.AllocateRequest{ContainerRequests: []*v1beta1.ContainerAllocateRequest{
			{DevicesIDs: []string{"1"}}}},
	},
}

func TestAllocate(t *testing.T) {
	if t == nil {
		return
	}
	config.NodeMode = "soft"
	for i := range testCasesForAllocate {
		tc := testCasesForAllocate[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			pl := NewDevicePlugin("name", nil, "socket")
			pl.initialize()
			resp, err := pl.Allocate(context.TODO(), tc.allocReq)
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
			if tc.expected == 0 && resp == nil {
				t.Error("error in test case: ", tc.desc, "nil error, but response is nil")
			}
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForApiDevices = []struct {
	desc     string
	patches  []func() *gomonkey.Patches
	retCount int
}{
	{
		desc: "test if no devices are returned",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var dm *xpu.DeviceManager
				return gomonkey.ApplyMethod(dm, "Devices", func(*xpu.DeviceManager) []*xpu.Device {
					fmt.Println("patched Devices")
					return nil
				})
			},
		},
		retCount: 0,
	},
	{
		desc: "test if empty devices are returned",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var dm *xpu.DeviceManager
				return gomonkey.ApplyMethod(dm, "Devices", func(*xpu.DeviceManager) []*xpu.Device {
					return []*xpu.Device{}
				})
			},
		},
		retCount: 0,
	},
	{
		desc: "test if 1 device is returned",
		patches: []func() *gomonkey.Patches{
			func() *gomonkey.Patches {
				var dm *xpu.DeviceManager
				return gomonkey.ApplyMethod(dm, "Devices", func(*xpu.DeviceManager) []*xpu.Device {
					dev := xpu.Device{}
					dev.ID = "test-device-0"
					return []*xpu.Device{&dev}
				})
			},
		},
		retCount: 1,
	},
}

func TestApiDevices(t *testing.T) {
	if t == nil {
		return
	}
	for i := range testCasesForApiDevices {
		tc := testCasesForApiDevices[i]
		t.Run(tc.desc, func(t *testing.T) {
			appliedPatches := ApplyPatches(tc.patches)
			config.DeviceSplitCount = 1
			dc := NewDeviceCache()
			dc.Start()
			pl := NewDevicePlugin("name", dc, "socket")
			pl.initialize()
			devs := pl.apiDevices()
			if len(devs) != tc.retCount {
				t.Error("error in test case: ", tc.desc, ": number of devices should be=", tc.retCount, "returned=", len(devs))
			}
			dc.Stop()
			ResetPatches(appliedPatches)
		})
	}
}

var testCasesForCreateCasHardMode = []struct {
	desc           string
	devReq         types.ContainerDevices
	expectedEnvLen int
}{
	{
		desc: "single card with template",
		devReq: types.ContainerDevices{
			{Index: 7, Template: "vir01"},
		},
		expectedEnvLen: 2,
	},
	{
		desc: "full card",
		devReq: types.ContainerDevices{
			{Index: 7, Template: "full"},
			{Index: 6, Template: "full"},
		},
		expectedEnvLen: 1,
	},
}

func TestCreateContainerAllocateResponseHardMode(t *testing.T) {
	if t == nil {
		return
	}
	pl := NewDevicePlugin("name", nil, "socket")
	for i := range testCasesForCreateCasHardMode {
		tc := testCasesForCreateCasHardMode[i]
		t.Run(tc.desc, func(t *testing.T) {
			resp, _ := pl.createContainerAllocateResponseHardMode(tc.devReq)
			if len(resp.Envs) != tc.expectedEnvLen {
				t.Errorf("expect env len %d, but got %d", tc.expectedEnvLen, len(resp.Envs))
			}
		})
	}
}

func TestPreStartContainer(t *testing.T) {
	pl := &DevicePlugin{}
	resp, err := pl.PreStartContainer(context.Background(), &v1beta1.PreStartContainerRequest{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if resp == nil {
		t.Errorf("expected non-nil response")
	}
}

func TestDialSuccess(t *testing.T) {
	ln, err := net.Listen("unix", "/tmp/test-plugin-dial.sock")
	if err != nil {
		t.Skipf("cannot create unix socket: %v", err)
	}
	defer ln.Close()
	defer os.Remove("/tmp/test-plugin-dial.sock")

	go func() {
		c, err := ln.Accept()
		if err != nil {
			return
		}
		if err := c.Close(); err != nil {
			return
		}
	}()

	pluginInst := &DevicePlugin{}
	_, err = pluginInst.dial("/tmp/test-plugin-dial.sock", 2*time.Second)
	if err != nil {
		t.Logf("dial result: %v", err)
	}
}
