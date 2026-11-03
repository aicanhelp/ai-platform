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
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	urmasdk "atomgit.com/openeuler/ubs-engine.git/src/sdk/go/urma"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"k8s.io/apimachinery/pkg/util/wait"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	remote "k8s.io/cri-client/pkg"
	"k8s.io/klog/v2"
	pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"gitcode.com/openFuyao/ub-network-device-plugin/resource"
	"gitcode.com/openFuyao/ub-network-device-plugin/urma"
	"gitcode.com/openFuyao/ub-network-device-plugin/utils"
)

type urmaDevicePlugin struct {
	ctx context.Context
	rm  resource.Manager

	socket string
	server *grpc.Server

	stop   chan interface{}
	update chan struct{}

	devMap sync.Map // key: deviceID, value: BondingEid
}

func (o *options) devicePluginForResource(ctx context.Context, resourceManager resource.Manager) (Interface, error) {
	plugin := urmaDevicePlugin{
		ctx:    ctx,
		rm:     resourceManager,
		socket: getPluginSocketPath(resourceManager.Resource()),

		server: nil,
		stop:   nil,
		update: nil,
	}
	return &plugin, nil
}

// getPluginSocketPath returns the socket to use for the specified resource.
func getPluginSocketPath(resource resource.Name) string {
	_, name := resource.Split()
	return filepath.Join(pluginapi.DevicePluginPath, name) + ".sock"
}

func (p *urmaDevicePlugin) initialize() {
	p.server = grpc.NewServer([]grpc.ServerOption{}...)
	p.stop = make(chan interface{})
	p.update = make(chan struct{})
	p.devMap = sync.Map{}
}

func (p *urmaDevicePlugin) cleanup() {
	close(p.stop)
	p.server = nil
	p.stop = nil
	p.update = nil
	p.devMap = sync.Map{}
}

func (p *urmaDevicePlugin) Devices() resource.Devices {
	return p.rm.Devices()
}

func (p *urmaDevicePlugin) Start(o *StartOpts) error {
	p.initialize()

	err := p.Serve()
	if err != nil {
		klog.Errorf("Could not start device p for '%s': %s", p.rm.Resource(), err)
		p.cleanup()
		return err
	}
	klog.Infof("Starting to serve '%s' on %s", p.rm.Resource(), p.socket)

	err = p.Register(o.KubeletSocket)
	if err != nil {
		klog.Errorf("Could not register device p: %s", err)
		return errors.Join(err, p.Stop())
	}
	klog.Infof("Registered device p for '%s' with Kubelet", p.rm.Resource())

	if _, err := os.Stat(utils.KubeletCheckpoint); os.IsNotExist(err) {
		klog.Warning("Kubelet checkpoint file does not exist")
	} else if err != nil {
		klog.Warningf("Stat kubelet checkpoint file err %v", err)
	} else {
		go func() {
			klog.Infof("Recover from kubelet checkpoint")
			err := p.rm.Recover()
			if err != nil {
				klog.Errorf("Could not recover device: %s", err)
				return
			}
		}()
	}

	go func() {
		klog.Infof("Starting to sync device states for '%s', interval: %v", p.rm.Resource(), o.SyncInterval)
		err = p.rm.SyncDevices(o.SyncInterval, p.stop, p.update)
		if err != nil {
			klog.Errorf("Failed to start sync devices: %v", err)
		}
	}()

	return nil
}

func (p *urmaDevicePlugin) Stop() error {
	if p == nil || p.server == nil {
		return nil
	}

	klog.Infof("Stopping to serve '%s' on %s", p.rm.Resource(), p.socket)
	p.server.Stop()
	if err := os.Remove(p.socket); err != nil && !os.IsNotExist(err) {
		return err
	}
	p.cleanup()

	return nil
}

func (p *urmaDevicePlugin) Serve() error {
	if err := os.Remove(p.socket); err != nil {
		return err
	}
	listener, err := net.Listen("unix", p.socket)
	if err != nil {
		return err
	}
	pluginapi.RegisterDevicePluginServer(p.server, p)

	go func() {
		lastCrashTime := time.Now()
		restartCount := 0

		for {
			if restartCount > 5 {
				klog.Fatalf("GRPC server for '%s' has repeatedly crashed recently. Quitting", p.rm.Resource())
			}

			klog.Infof("Starting GRPC server for '%s'", p.rm.Resource())
			err := p.server.Serve(listener)
			if err == nil {
				klog.Infof("GRPC server for '%s' stopped", p.rm.Resource())
				break
			}

			klog.Infof("GRPC server for '%s' crashed with error: %v", p.rm.Resource(), err)
			timeSinceLastCrash := time.Since(lastCrashTime).Seconds()
			lastCrashTime = time.Now()
			if timeSinceLastCrash > 3600 {
				restartCount = 0
			} else {
				restartCount++
			}
		}
	}()

	conn, err := p.dial(p.socket, 5*time.Second)
	if err != nil {
		return err
	}
	conn.Close()

	return nil
}

func (p *urmaDevicePlugin) dial(unixSocketPath string, timeout time.Duration) (*grpc.ClientConn, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	c, err := grpc.DialContext(ctx, unixSocketPath,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
		grpc.WithBlock(),
		grpc.WithContextDialer(func(ctx context.Context, addr string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, "unix", addr)
		}),
	)
	if err != nil {
		return nil, err
	}

	return c, nil
}

func (p *urmaDevicePlugin) Register(kubeletSocket string) error {
	if kubeletSocket == "" {
		klog.Info("Skipping registration with Kubelet")
		return nil
	}

	conn, err := p.dial(kubeletSocket, 5*time.Second)
	if err != nil {
		return err
	}
	defer conn.Close()

	client := pluginapi.NewRegistrationClient(conn)
	reqt := &pluginapi.RegisterRequest{
		Version:      pluginapi.Version,
		Endpoint:     path.Base(p.socket),
		ResourceName: string(p.rm.Resource()),
		Options: &pluginapi.DevicePluginOptions{
			PreStartRequired:                true,
			GetPreferredAllocationAvailable: true,
		},
	}

	_, err = client.Register(p.ctx, reqt)
	return err
}

func (p *urmaDevicePlugin) ListAndWatch(_ *pluginapi.Empty, s pluginapi.DevicePlugin_ListAndWatchServer) error {
	klog.Infof("kubelet start ListAndWatch.")
	klog.Infof("ListAndWatch for '%s'", p.rm.Resource())
	klog.Infof("plugin socket: %v", p.socket)
	klog.Infof("plugin devices: %v", p.apiDevices())
	if err := s.Send(&pluginapi.ListAndWatchResponse{Devices: p.apiDevices()}); err != nil {
		klog.Errorf("failed to send ListAndWatchResponse first time: %v", err)
		return err
	}

	for {
		select {
		case <-p.stop:
			return nil

		case <-p.update:
			klog.Infof("'%s' device list changed, send update to kubelet.", p.rm.Resource())
			if err := s.Send(&pluginapi.ListAndWatchResponse{Devices: p.apiDevices()}); err != nil {
				klog.Errorf("failed to send ListAndWatchResponse: %v", err)
				continue
			}
		}
	}
}

func (p *urmaDevicePlugin) GetPreferredAllocation(_ context.Context, req *pluginapi.PreferredAllocationRequest) (*pluginapi.PreferredAllocationResponse, error) {

	deviceIDs := p.getDeviceIDs(req.ContainerRequests[0].AvailableDeviceIDs, int(req.ContainerRequests[0].AllocationSize))

	var response []*pluginapi.ContainerPreferredAllocationResponse

	response = append(response, &pluginapi.ContainerPreferredAllocationResponse{DeviceIDs: deviceIDs})
	return &pluginapi.PreferredAllocationResponse{ContainerResponses: response}, nil
}

func (p *urmaDevicePlugin) getDeviceIDs(available []string, size int) []string {
	if size == 0 || len(available) < size {
		return []string{}
	}
	deviceMap := make(map[uint64][]string)
	devices := p.rm.Devices()

	for _, device := range available {
		val := devices[device]
		deviceMap[val.HwResId] = append(deviceMap[val.HwResId], val.Name)
	}
	keys := make([]uint64, 0, len(deviceMap))
	for k := range deviceMap {
		keys = append(keys, k)
	}
	if len(keys) == 0 {
		return []string{}
	}
	sort.Slice(keys, func(i, j int) bool {
		if len(deviceMap[keys[i]]) > len(deviceMap[keys[j]]) {
			return true
		} else if len(deviceMap[keys[i]]) == len(deviceMap[keys[j]]) && keys[i] < keys[j] {
			return true
		}
		return false
	})
	deviceIDs := make([]string, 0, size)
	var index = 0
	klog.Infof("devices Info %v %v", deviceMap, keys)
	for {
		for _, key := range keys {
			deviceIDs = append(deviceIDs, deviceMap[key][index])
			if len(deviceIDs) == size {
				return deviceIDs
			}
		}
		index++
	}
}

func (p *urmaDevicePlugin) Allocate(_ context.Context, in *pluginapi.AllocateRequest) (*pluginapi.AllocateResponse, error) {
	resp := &pluginapi.AllocateResponse{}
	for _, req := range in.GetContainerRequests() {
		klog.Infof("Allocate req devices ids: %v", req.DevicesIDs)
		response, err := p.getAllocateResponse(req.DevicesIDs)
		if err != nil {
			return nil, fmt.Errorf("failed to get allocate response: %v", err)
		}
		resp.ContainerResponses = append(resp.ContainerResponses, response)
	}

	return resp, nil
}

func (p *urmaDevicePlugin) PreStartContainer(ctx context.Context, in *pluginapi.PreStartContainerRequest) (*pluginapi.PreStartContainerResponse, error) {
	klog.Info("PreStartContainer has been called.")

	if err := p.SetUrmaDevNetworkNS(ctx, in); err != nil {
		return nil, err
	}

	return &pluginapi.PreStartContainerResponse{}, nil
}

func (p *urmaDevicePlugin) GetDevicePluginOptions(context.Context, *pluginapi.Empty) (*pluginapi.DevicePluginOptions, error) {
	return &pluginapi.DevicePluginOptions{
		PreStartRequired:                true,
		GetPreferredAllocationAvailable: true,
	}, nil
}

func (p *urmaDevicePlugin) getAllocateResponse(requestIDs []string) (*pluginapi.ContainerAllocateResponse, error) {
	devices := make([]*pluginapi.DeviceSpec, 0, len(requestIDs))

	for _, id := range requestIDs {
		devInfo, err := urma.AllocateDevice(id)
		if err != nil {
			return nil, err
		}
		klog.Infof("AllocateDevice: %v", devInfo)
		p.devMap.Store(id, devInfo.BondingEid)

		for _, devPath := range devInfo.VfePaths {
			devices = append(devices, &pluginapi.DeviceSpec{
				ContainerPath: devPath,
				HostPath:      devPath,
				Permissions:   "rw",
			})
		}

		devices = append(devices, &pluginapi.DeviceSpec{
			ContainerPath: devInfo.BondingPath,
			HostPath:      devInfo.BondingPath,
			Permissions:   "rw",
		})

		devices = append(devices, &pluginapi.DeviceSpec{
			ContainerPath: utils.TidDevicePath,
			HostPath:      utils.TidDevicePath,
			Permissions:   "rw",
		})

		devices = append(devices, &pluginapi.DeviceSpec{
			ContainerPath: utils.UBCorePath,
			HostPath:      utils.UBCorePath,
			Permissions:   "rw",
		})
	}

	return &pluginapi.ContainerAllocateResponse{
		Devices: devices,
	}, nil
}

func (p *urmaDevicePlugin) GetBondingEid(deviceID string) (string, error) {
	if value, ok := p.devMap.Load(deviceID); ok {
		return value.(string), nil
	}

	var eid string
	err := wait.PollUntilContextTimeout(context.Background(), 3*time.Second, 30*time.Second, true, func(ctx context.Context) (bool, error) {
		devInfo, err := urmasdk.UbsAllocateDevice(deviceID)
		if err != nil {
			klog.Errorf("failed to get bonding eid for device from ubs_urma_dev_alloc: %s, err: %v", deviceID, err)
			return false, nil
		}
		eid = devInfo.BondingEid
		return true, nil
	})
	if err != nil {
		return "", fmt.Errorf("timeout waiting for bonding eid for device %s: %w", deviceID, err)
	}

	p.devMap.Store(deviceID, eid)
	return eid, nil
}

func (p *urmaDevicePlugin) apiDevices() []*pluginapi.Device {
	return p.rm.Devices().GetPluginDevices()
}

func (p *urmaDevicePlugin) SetUrmaDevNetworkNS(ctx context.Context, req *pluginapi.PreStartContainerRequest) error {
	klog.Infof("Set network namespace for device: %v", req.GetDevicesIDs())

	client, err := remote.NewRemoteRuntimeService(utils.RuntimeSock, 30*time.Second, nil, nil)
	if err != nil {
		return err
	}

	for _, id := range req.GetDevicesIDs() {
		podUid, err := utils.GetPodUIDByDeviceID(id, utils.KubeletCheckpoint)
		if err != nil {
			return fmt.Errorf("failed to get pod uid: %v", err)
		}

		sandboxes, err := utils.GetSandboxes(ctx, client, podUid)
		if len(sandboxes) == 0 || err != nil {
			return fmt.Errorf("failed to get sandboxes: %v", err)
		}

		var sandboxPid string
		for _, sandbox := range sandboxes {
			if sandbox.State == runtimeapi.PodSandboxState_SANDBOX_NOTREADY {
				continue
			}
			sandboxPid, err = utils.GetSandboxPid(ctx, client, sandbox.Id)
			if sandboxPid == "" || err != nil {
				klog.Warningf("failed to get sandbox pid: %v", err)
				continue
			}
			klog.Infof("PreStartContainer set network namespace for device: %v, podUid: %v, sandboxPid: %v", id, podUid, sandboxPid)
			break
		}

		if len(sandboxPid) == 0 {
			return fmt.Errorf("failed to get sandbox pid: %v", err)
		}

		// Set network namespace for device
		err = p.SetUrmaNetworkNamespace(id, sandboxPid)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *urmaDevicePlugin) SetUrmaNetworkNamespace(deviceID string, sandboxPid string) error {
	bondingEid, err := p.GetBondingEid(deviceID)
	if err != nil {
		return fmt.Errorf("failed to get bonding eid for device %s: %w", deviceID, err)
	}
	klog.Infof("set network namespace for device: %s, pid %s, bondingEid %s", deviceID, sandboxPid, bondingEid)

	ctx, cancel := context.WithTimeout(context.Background(), utils.GrpcTimeOut)
	defer cancel()

	nsPath := fmt.Sprintf("/proc/%s/ns/net", sandboxPid)
	args := []string{"agg", "expose", bondingEid, nsPath}
	cmd := exec.CommandContext(ctx, "urma_admin", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		klog.Errorf("failed to set dev ns: %v, cmd: %s, output: %s", err, cmd.String(), string(output))
		return fmt.Errorf("failed to set dev ns: %w, output: %s", err, string(output))
	}
	klog.Infof("Successfully set dev ns: %s", strings.TrimSpace(string(output)))
	return nil
}
