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

// Package deviceplugin KAE 设备插件实现包，主要实现注册、发现、分配设备
package deviceplugin

import (
	"context"
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	timeOut = 10
)

// DeviceClient defines a csi client conn for unix socket
type DeviceClient struct {
	client     v1beta1.DevicePluginClient
	clientConn *grpc.ClientConn
	socketPath string
}

// NewDeviceClient new a CSI client for unix socket
func NewDeviceClient(sock string) (*DeviceClient, error) {
	client, c, err := dial(sock)
	if err != nil {
		zlog.Errorf("can't create new endpoint with socket path: %s, err: %v", sock, err)
		return nil, err
	}

	return &DeviceClient{
		client:     client,
		clientConn: c,
		socketPath: sock,
	}, nil
}

// CheckListAndWatch 调用 device client 的 ListAndWatch 检查 sock 文件是否正常
func (c *DeviceClient) CheckListAndWatch() {
	_, err := c.client.GetDevicePluginOptions(context.Background(), &v1beta1.Empty{})
	if err != nil {
		zlog.Errorf("getDevicePluginOptions failed, err: %v", err)
		RemoveKAESock()
	}
	return
}

// dial 使用 sock 建立链接
func dial(unixSocketPath string) (v1beta1.DevicePluginClient, *grpc.ClientConn, error) {
	ctx, cancelFunc := context.WithTimeout(context.Background(), timeOut*time.Second)
	defer cancelFunc()

	conn, err := grpc.DialContext(
		ctx,
		unixSocketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(
			func(ctx context.Context, address string) (net.Conn, error) {
				return (&net.Dialer{}).DialContext(ctx, "unix", address)
			},
		),
	)

	if err != nil {
		return nil, nil, fmt.Errorf("failed to dial device plugin: %v", err)
	}
	return v1beta1.NewDevicePluginClient(conn), conn, nil
}
