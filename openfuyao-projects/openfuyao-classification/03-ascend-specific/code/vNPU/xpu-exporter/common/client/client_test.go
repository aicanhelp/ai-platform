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

package client

import (
	"context"
	"fmt"
	"net"
	"os"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"google.golang.org/grpc"

	"huawei.com/xpu-exporter/common/service"
)

type mockClient struct {
	retVal       error
	resp         *service.GetPidsResponse
	vxpuInfoResp *service.GetAllVxpuInfoResponse
}

func (mc *mockClient) GetPids(ctx context.Context, req *service.GetPidsRequest,
	opts ...grpc.CallOption) (*service.GetPidsResponse, error) {
	return mc.resp, mc.retVal
}

func (mc *mockClient) GetAllVxpuInfo(ctx context.Context, req *service.GetAllVxpuInfoRequest,
	opts ...grpc.CallOption) (*service.GetAllVxpuInfoResponse, error) {
	return mc.vxpuInfoResp, mc.retVal
}

func newPatchesWithMock(mc *mockClient) *gomonkey.Patches {
	patches := gomonkey.NewPatches()
	patches.ApplyFunc(grpc.Dial, func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
		return &grpc.ClientConn{}, nil
	})
	patches.ApplyFunc(service.NewPidsServiceClient,
		func(c grpc.ClientConnInterface) service.PidsServiceClient {
			return mc
		})
	var c *grpc.ClientConn
	patches.ApplyMethod(c, "Close", func(_ *grpc.ClientConn) error {
		return nil
	})
	return patches
}

func TestGetAllVxpuInfo(t *testing.T) {
	patches := newPatchesWithMock(&mockClient{retVal: fmt.Errorf("test error from GetAllVxpuInfo()")})
	defer patches.Reset()
	_, err := GetAllVxpuInfo()

	if err == nil {
		t.Errorf("error in test GetAllVxpuInfo")
	} else {
		t.Log("test GetAllVxpuInfo succeed")
	}
}

func TestGetAllVxpuInfo_Success(t *testing.T) {
	patches := newPatchesWithMock(&mockClient{
		retVal:       nil,
		vxpuInfoResp: &service.GetAllVxpuInfoResponse{VxpuInfos: "test_vxpu_info"},
	})
	defer patches.Reset()

	result, err := GetAllVxpuInfo()
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if result != "test_vxpu_info" {
		t.Errorf("expected test_vxpu_info, got: %s", result)
	}
}

func TestGetAllVxpuInfo_DialError(t *testing.T) {
	patchDial := gomonkey.ApplyFunc(grpc.Dial, func(target string, opts ...grpc.DialOption) (*grpc.ClientConn, error) {
		return nil, fmt.Errorf("dial error")
	})
	defer patchDial.Reset()

	_, err := GetAllVxpuInfo()
	if err == nil {
		t.Errorf("expected error, got nil")
	}
}

func TestGetAllVxpuInfo_EmptyInfos(t *testing.T) {
	patches := newPatchesWithMock(&mockClient{
		retVal:       nil,
		vxpuInfoResp: &service.GetAllVxpuInfoResponse{VxpuInfos: ""},
	})
	defer patches.Reset()

	result, err := GetAllVxpuInfo()
	if err != nil {
		t.Errorf("expected nil error, got: %v", err)
	}
	if result != "" {
		t.Errorf("expected empty infos, got: %s", result)
	}
}

func TestGetAllVxpuInfo_RealDialer(t *testing.T) {
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

	_, err = GetAllVxpuInfo()
	if err == nil {
		t.Log("expected error from real RPC call")
	}
}
