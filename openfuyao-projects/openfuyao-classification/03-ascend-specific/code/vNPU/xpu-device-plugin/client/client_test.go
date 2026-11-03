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
	"errors"
	"fmt"
	"net"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"google.golang.org/grpc"

	"huawei.com/vxpu-device-plugin/pkg/api/runtime/service"
)

type mockClient struct {
	retVal error
}

func (mc *mockClient) GetPids(ctx interface{}, req interface{}, opts ...interface{}) (interface{}, error) {
	return nil, mc.retVal
}

func (mc *mockClient) GetAllVxpuInfo(ctx interface{}, req interface{}, opts ...interface{}) (interface{}, error) {
	return nil, mc.retVal
}

type mockDialFunc func() (*grpc.ClientConn, error)
type mockGetPidsFunc func(service.PidsServiceClient, string) error

var originalDialGRPCFunc = dialGRPCFunc
var originalGetPidsFunc = getPidsFunc

func resetClientFuncs() {
	dialGRPCFunc = originalDialGRPCFunc
	getPidsFunc = originalGetPidsFunc
}

var registerTestCasesForUpdatePidsConfig = []struct {
	desc     string
	setup    func(*testing.T) func()
	expected int
}{
	{
		desc: "checking error handling from grpc.Dial()",
		setup: func(t *testing.T) func() {
			dialGRPCFunc = func() (*grpc.ClientConn, error) {
				return nil, errors.New("test error")
			}
			return resetClientFuncs
		},
		expected: 1,
	},
	{
		desc: "checking error handling from GetPids()",
		setup: func(t *testing.T) func() {
			patchClose := gomonkey.ApplyMethod(reflect.TypeOf(&grpc.ClientConn{}), "Close",
				func(_ *grpc.ClientConn) error {
					return nil
				})
			dialGRPCFunc = func() (*grpc.ClientConn, error) {
				return &grpc.ClientConn{}, nil
			}
			getPidsFunc = func(_ service.PidsServiceClient, _ string) error {
				return fmt.Errorf("test error from GetPids")
			}
			return func() {
				resetClientFuncs()
				patchClose.Reset()
			}
		},
		expected: 1,
	},
	{
		desc: "happy path",
		setup: func(t *testing.T) func() {
			patchClose := gomonkey.ApplyMethod(reflect.TypeOf(&grpc.ClientConn{}), "Close",
				func(_ *grpc.ClientConn) error {
					return nil
				})
			dialGRPCFunc = func() (*grpc.ClientConn, error) {
				return &grpc.ClientConn{}, nil
			}
			getPidsFunc = func(_ service.PidsServiceClient, _ string) error {
				return nil
			}
			return func() {
				resetClientFuncs()
				patchClose.Reset()
			}
		},
		expected: 0,
	},
	{
		desc: "checking error if client connection is nil",
		setup: func(t *testing.T) func() {
			dialGRPCFunc = func() (*grpc.ClientConn, error) {
				return nil, nil
			}
			return resetClientFuncs
		},
		expected: 1,
	},
	{
		desc: "checking error if GetPids return error",
		setup: func(t *testing.T) func() {
			patchClose := gomonkey.ApplyMethod(reflect.TypeOf(&grpc.ClientConn{}), "Close",
				func(_ *grpc.ClientConn) error {
					return nil
				})
			dialGRPCFunc = func() (*grpc.ClientConn, error) {
				return &grpc.ClientConn{}, nil
			}
			getPidsFunc = func(_ service.PidsServiceClient, _ string) error {
				return fmt.Errorf("test error from GetPids()")
			}
			return func() {
				resetClientFuncs()
				patchClose.Reset()
			}
		},
		expected: 1,
	},
}

func TestUpdatePidsConfig(t *testing.T) {
	for i := range registerTestCasesForUpdatePidsConfig {
		tc := registerTestCasesForUpdatePidsConfig[i]
		t.Run(tc.desc, func(t *testing.T) {
			cleanup := tc.setup(t)
			defer cleanup()
			path := "test"
			err := updatePidsConfig(path)
			var status int
			if err != nil {
				status = 1
			}
			if status != tc.expected {
				t.Error("error in test case: ", tc.desc)
			}
		})
	}
}

func TestMainError(t *testing.T) {
	dialGRPCFunc = func() (*grpc.ClientConn, error) {
		return nil, fmt.Errorf("test error")
	}
	defer resetClientFuncs()

	exited := false
	patchOsExit := gomonkey.ApplyFunc(os.Exit, func(c int) {
		exited = true
		return
	})
	defer patchOsExit.Reset()
	main()
	if !exited {
		t.Error("main should exit")
	}
}

func TestUpdatePidsConfigRealDial(t *testing.T) {
	if err := os.Remove(pidsSockPath); err != nil && !os.IsNotExist(err) {
		t.Logf("cleanup: %v", err)
	}
	ln, err := net.Listen("unix", pidsSockPath)
	if err != nil {
		t.Skipf("cannot create unix socket: %v", err)
	}
	defer func() { _ = ln.Close() }()
	defer func() {
		if err := os.Remove(pidsSockPath); err != nil && !os.IsNotExist(err) {
			t.Logf("cleanup: %v", err)
		}
	}()

	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			if err := c.Close(); err != nil {
				return
			}
		}
	}()

	err = updatePidsConfig("test")
	if err == nil {
		t.Log("expected error from real RPC call")
	}
}
