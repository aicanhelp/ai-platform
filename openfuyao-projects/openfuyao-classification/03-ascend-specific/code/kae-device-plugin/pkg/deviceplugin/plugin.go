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
	"fmt"
	"net"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	checkDeviceInterval = 5 * time.Second
)

var mountPaths = []string{
	"/usr/local/lib/engines-1.1",
	"/usr/local/lib/libwd_comp.la",
	"/usr/local/lib/libwd_comp.so",
	"/usr/local/lib/libwd_comp.so.2",
	"/usr/local/lib/libwd_comp.so.2.6.0",
	"/usr/local/lib/libwd_crypto.la",
	"/usr/local/lib/libwd_crypto.so",
	"/usr/local/lib/libwd_crypto.so.2",
	"/usr/local/lib/libwd_crypto.so.2.6.0",
	"/usr/local/lib/libwd.la",
	"/usr/local/lib/libwd.so",
	"/usr/local/lib/libwd.so.2",
	"/usr/local/lib/libwd.so.2.6.0",
	"/usr/local/lib/pkgconfig",
	"/usr/local/lib/uadk",
	"/usr/lib64/libnuma.so.1",
}

type pluginService struct {
	kaeManager *KAEManager
}

// GetDevicePluginOptions returns options to device manager
func (s *pluginService) GetDevicePluginOptions(ctx context.Context, e *v1beta1.Empty) (*v1beta1.DevicePluginOptions,
	error) {
	return &v1beta1.DevicePluginOptions{}, nil
}

// ListAndWatch watch and retur list of devices as stream
func (s *pluginService) ListAndWatch(_ *v1beta1.Empty, stream v1beta1.DevicePlugin_ListAndWatchServer) error {
	for {
		select {
		case <-s.kaeManager.stopMonitor:
			zlog.Infof("grpc service has stop")
			return nil
		default:
			// 此处报错则说明没有设备
			_ = s.kaeManager.discoverDevices()
			err := s.sendDeviceUpdates(stream)
			if err != nil {
				return err
			}
			s.kaeManager.hasDeviceChange = false
			time.Sleep(checkDeviceInterval)
		}
	}
}

// sendDeviceUpdates is the sub function of ListAndWatch, check and discover devices then send updates
func (s *pluginService) sendDeviceUpdates(stream v1beta1.DevicePlugin_ListAndWatchServer) error {
	if s.kaeManager.hasDeviceChange {
		resp := new(v1beta1.ListAndWatchResponse)
		for _, dev := range s.kaeManager.devices {
			resp.Devices = append(resp.Devices, &v1beta1.Device{ID: dev.ID, Health: dev.Health})
		}
		zlog.Infof("listAndWatch: send devices %v", resp)
		if err := stream.Send(resp); err != nil {
			zlog.Errorf("device-plugin: cannot update device states: %v", err)
			s.kaeManager.grpcServer.Stop()
			return err
		}
	}
	return nil
}

func (s *pluginService) Allocate(ctx context.Context, requests *v1beta1.AllocateRequest) (*v1beta1.
	AllocateResponse, error) {
	resps := new(v1beta1.AllocateResponse)
	for _, rqt := range requests.ContainerRequests {
		resp := new(v1beta1.ContainerAllocateResponse)
		// add all requested devices to Allocate response
		for _, id := range rqt.DevicesIDs {
			dev, ok := s.kaeManager.devices[id]
			if !ok {
				return nil, fmt.Errorf("invalid allocation request with non-existing device %s", id)
			}
			if dev.Health != v1beta1.Healthy {
				return nil, fmt.Errorf("invalid allocation request with unhealthy device %s", id)
			}
			respDevice := &v1beta1.DeviceSpec{
				HostPath:      "/dev/" + id,
				ContainerPath: "/dev/" + id,
				Permissions:   "mrw",
			}
			resp.Devices = append(resp.Devices, respDevice)
		}
		for _, device := range s.kaeManager.defaultDevices {
			respDevice := &v1beta1.DeviceSpec{
				HostPath:      device,
				ContainerPath: device,
				Permissions:   "mrw",
			}
			resp.Devices = append(resp.Devices, respDevice)
		}
		for _, path := range mountPaths {
			respMount := &v1beta1.Mount{
				ContainerPath: path,
				HostPath:      path,
				ReadOnly:      false, // 根据需要修改
			}
			resp.Mounts = append(resp.Mounts, respMount)
		}
		resps.ContainerResponses = append(resps.ContainerResponses, resp)
	}

	zlog.Infof("allocate:kubelet request response %v", resps)
	return resps, nil
}

func (s *pluginService) PreStartContainer(context.Context, *v1beta1.PreStartContainerRequest) (*v1beta1.
	PreStartContainerResponse, error) {
	zlog.Infof("device-plugin: PreStart should not be called for kae device plugin")
	return &v1beta1.PreStartContainerResponse{}, nil
}

func (s *pluginService) GetPreferredAllocation(ctx context.Context, r *v1beta1.PreferredAllocationRequest) (*v1beta1.
	PreferredAllocationResponse, error) {
	zlog.Infof("device-plugin: PreStart should not be called for kae device plugin")
	return &v1beta1.PreferredAllocationResponse{}, nil
}

func (s *pluginService) RegisterService() {
	v1beta1.RegisterDevicePluginServer(s.kaeManager.grpcServer, s)
}

// RegisterWithV1Beta1Kubelet register the plugin server to kubelet.
func RegisterWithV1Beta1Kubelet(pluginEndpoint, resourceName string) error {
	dialOptions := []grpc.DialOption{
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithContextDialer(createUnixDialer()),
	}
	conn, err := grpc.Dial(v1beta1.KubeletSocket, dialOptions...)
	if err != nil {
		return fmt.Errorf("device-plugin: cannot connect to kubelet service: %v", err)
	}

	defer closeConnection(conn)

	registerReq := &v1beta1.RegisterRequest{
		ResourceName: resourceName,
		Version:      v1beta1.Version,
		Endpoint:     pluginEndpoint,
	}

	registrationClient := v1beta1.NewRegistrationClient(conn)
	if _, err = registrationClient.Register(context.Background(), registerReq); err != nil {
		return fmt.Errorf("device-plugin: cannot register to kubelet service: %v", err)
	}

	return nil
}

func createUnixDialer() func(ctx context.Context, addr string) (net.Conn, error) {
	return func(ctx context.Context, addr string) (net.Conn, error) {
		if deadline, ok := ctx.Deadline(); ok {
			return net.DialTimeout("unix", addr, time.Until(deadline))
		}
		return net.DialTimeout("unix", addr, 0)
	}
}

func closeConnection(conn *grpc.ClientConn) {
	if err := conn.Close(); err != nil {
		zlog.Warnf("device-plugin: failed to close connection: %v", err)
	}
}
