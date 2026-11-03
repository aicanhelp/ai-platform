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
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"google.golang.org/grpc"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	clientIDZero    int    = 0
	clientIDOne     int    = 1
	PluginSocketDir string = "/dev"
)

type kaeClient struct {
	ID int
	cc *grpc.ClientConn
}

type kaeListAndWatchClient struct {
	grpc.ClientStream
}

type failingDevicePluginClient struct {
	v1beta1.DevicePluginClient
}

func (failingDevicePluginClient) GetDevicePluginOptions(context.Context, *v1beta1.Empty,
	...grpc.CallOption) (*v1beta1.DevicePluginOptions, error) {
	return nil, errors.New("device plugin is unavailable")
}

func (kaeListAndWatchClient) Recv() (*v1beta1.ListAndWatchResponse, error) {
	return nil, nil
}

func (d kaeClient) GetDevicePluginOptions(ctx context.Context, in *v1beta1.Empty,
	opts ...grpc.CallOption) (*v1beta1.DevicePluginOptions, error) {
	return nil, nil
}

func (d kaeClient) ListAndWatch(ctx context.Context, in *v1beta1.Empty,
	opts ...grpc.CallOption) (v1beta1.DevicePlugin_ListAndWatchClient, error) {
	if d.ID == 0 {
		return nil, fmt.Errorf("called listAndWatch failed")
	}
	return &kaeListAndWatchClient{}, nil
}

func (d kaeClient) GetPreferredAllocation(ctx context.Context, in *v1beta1.PreferredAllocationRequest,
	opts ...grpc.CallOption) (*v1beta1.PreferredAllocationResponse, error) {
	return nil, nil
}

func (d kaeClient) Allocate(ctx context.Context, in *v1beta1.AllocateRequest,
	opts ...grpc.CallOption) (*v1beta1.AllocateResponse, error) {
	return nil, nil
}

func (d kaeClient) PreStartContainer(ctx context.Context, in *v1beta1.PreStartContainerRequest,
	opts ...grpc.CallOption) (*v1beta1.PreStartContainerResponse, error) {
	return nil, nil
}

func newKaePluginClient(cc *grpc.ClientConn) v1beta1.DevicePluginClient {
	return &kaeClient{
		ID: clientIDZero,
		cc: cc,
	}
}

func newKaePluginClientElse(cc *grpc.ClientConn) v1beta1.DevicePluginClient {
	return &kaeClient{
		ID: clientIDOne,
		cc: cc,
	}
}

// TestNewDeviceClient 测试 NewDeviceClient
func TestNewDeviceClient(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	sock := PluginSocketDir + pluginEndpointPrefix

	patches.ApplyFunc(
		dial, func(string) (v1beta1.DevicePluginClient, *grpc.ClientConn, error) {
			return nil, nil, errors.New("dial failed")
		},
	)

	_, err := NewDeviceClient(sock)
	assert.NotNil(t, err)

	patches.Reset()
	patches.ApplyFunc(
		dial, func(string) (v1beta1.DevicePluginClient, *grpc.ClientConn, error) {
			return nil, nil, nil
		},
	)

	_, err = NewDeviceClient(sock)
	assert.Nil(t, err)
}

// TestCheckListAndWatch 测试 CheckListAndWatch
func TestCheckListAndWatch(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	_, err := net.Listen("unix", "127.0.0.1")
	if err != nil {
		zlog.Infof("%v", err)
		return
	}
	defer os.Remove("127.0.0.1")
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	conn, err := grpc.DialContext(
		ctx,
		"127.0.0.1",
		grpc.WithInsecure(),
		grpc.WithBlock(),
		grpc.WithContextDialer(
			func(ctx context.Context, address string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", address)
			},
		),
	)
	client := newKaePluginClient(conn)

	dClient := &DeviceClient{
		client: client,
	}

	dClient.CheckListAndWatch()
	assert.NotNil(t, err)

	patches.Reset()
	dClient.client = newKaePluginClientElse(conn)
	dClient.CheckListAndWatch()
	assert.NotNil(t, err)
}

func TestCheckListAndWatchRemovesStaleSocketAfterHealthCheckFailure(t *testing.T) {
	originalDirectory := devicePluginDirectory
	devicePluginDirectory = t.TempDir()
	t.Cleanup(func() { devicePluginDirectory = originalDirectory })

	staleSocket := devicePluginDirectory + "/hisi-stale.sock"
	if err := os.WriteFile(staleSocket, nil, chmodPermission); err != nil {
		t.Fatalf("create stale socket: %v", err)
	}

	client := &DeviceClient{client: failingDevicePluginClient{}}
	client.CheckListAndWatch()

	if _, err := os.Stat(staleSocket); !os.IsNotExist(err) {
		t.Fatalf("stale socket still exists: %v", err)
	}
}
