/*
Copyright 2023 The Volcano Authors.
Copyright(C)2020-2023. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package plugin

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"net"
	"os"
	"path"
	"path/filepath"
	"time"

	"google.golang.org/grpc"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	grpcServeTryCount = 5
	secondsPerHour    = 3600
	dialTimeout       = 5
	pluginNotify      = "plugin"
	pluginSockPerm    = 0600
)

type DevicePlugin struct {
	deviceCache  *DeviceCache
	resourceName string
	socket       string

	server *grpc.Server
	health chan *xpu.Device
	stop   chan interface{}
}

func NewDevicePlugin(resourceName string, deviceCache *DeviceCache, socket string) *DevicePlugin {
	return &DevicePlugin{
		deviceCache:  deviceCache,
		resourceName: resourceName,
		socket:       socket,

		server: nil,
		health: nil,
		stop:   nil,
	}
}

func (m *DevicePlugin) initialize() {
	m.server = grpc.NewServer([]grpc.ServerOption{}...)
	m.health = make(chan *xpu.Device)
	m.stop = make(chan interface{})
}

func (m *DevicePlugin) cleanup() {
	close(m.stop)
	m.server = nil
	m.health = nil
	m.stop = nil
}

func (m *DevicePlugin) Start() error {
	m.initialize()

	err := m.serve()
	if err != nil {
		log.Errorf("Could not start device plugin for '%s': %s", m.resourceName, err)
		m.cleanup()
		return err
	}

	log.Infof("Starting to serve '%s' on %s", m.resourceName, m.socket)

	err = m.register()
	if err != nil {
		log.Errorf("Could not register device plugin: %s", err)
		m.Stop()
		return err
	}
	log.Infof("Registered device plugin for '%s' with Kubelet", m.resourceName)

	m.deviceCache.AddNotifyChannel(pluginNotify, m.health)
	return nil
}

func (m *DevicePlugin) Stop() {
	if m == nil || m.server == nil {
		return
	}
	log.Infof("Stopping to serve '%s' on %s", m.resourceName, m.socket)
	m.deviceCache.RemoveNotifyChannel(pluginNotify)
	m.server.Stop()
	if err := os.Remove(m.socket); err != nil && !os.IsNotExist(err) {
		log.Errorf("remove sock error: %v, path: %s", err, m.socket)
	}
	m.cleanup()
}

func (m *DevicePlugin) serve() error {
	err := os.Remove(m.socket)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	sock, err := net.Listen("unix", m.socket)
	if err != nil {
		return err
	}
	err = os.Chmod(m.socket, pluginSockPerm)
	if err != nil {
		log.Errorf("modify plugin socket file permissions error: %v", err)
		return err
	}

	v1beta1.RegisterDevicePluginServer(m.server, m)

	go func() {
		lastCrashTime := time.Now()
		restartCount := 0
		for {
			select {
			case <-m.stop:
				return
			default:
			}

			log.Infof("Starting GRPC server for '%s'", m.resourceName)
			err := m.server.Serve(sock)
			if err == nil {
				break
			}

			select {
			case <-m.stop:
				log.Infof("GRPC server for '%s' stopped", m.resourceName)
				return
			default:
			}

			log.Errorf("GRPC server for '%s' crashed with error: %v", m.resourceName, err)

			if restartCount > grpcServeTryCount {
				log.Fatalf("GRPC server for '%s' has repeatedly crashed recently. Quitting", m.resourceName)
			}
			timeSinceLastCrash := time.Since(lastCrashTime).Seconds()
			lastCrashTime = time.Now()
			if timeSinceLastCrash > secondsPerHour {
				restartCount = 1
			} else {
				restartCount++
			}
		}
	}()

	conn, err := m.dial(m.socket, dialTimeout*time.Second)
	if err != nil {
		return err
	}
	if conn != nil {
		conn.Close()
	}

	return nil
}

func (m *DevicePlugin) register() error {
	conn, err := m.dial(v1beta1.KubeletSocket, dialTimeout*time.Second)
	if err != nil {
		return err
	}
	if conn == nil {
		return fmt.Errorf("client connection is nil")
	}
	defer conn.Close()

	client := v1beta1.NewRegistrationClient(conn)
	req := &v1beta1.RegisterRequest{
		Version:      v1beta1.Version,
		Endpoint:     path.Base(m.socket),
		ResourceName: m.resourceName,
		Options:      &v1beta1.DevicePluginOptions{},
	}

	_, err = client.Register(context.Background(), req)
	return err
}

func (m *DevicePlugin) GetDevicePluginOptions(context.Context, *v1beta1.Empty) (
	*v1beta1.DevicePluginOptions, error) {
	options := &v1beta1.DevicePluginOptions{}
	return options, nil
}

func (m *DevicePlugin) GetPreferredAllocation(context.Context, *v1beta1.PreferredAllocationRequest) (
	*v1beta1.PreferredAllocationResponse, error) {
	resp := &v1beta1.PreferredAllocationResponse{}
	return resp, nil
}

func (m *DevicePlugin) ListAndWatch(e *v1beta1.Empty, s v1beta1.DevicePlugin_ListAndWatchServer) error {
	_ = s.Send(&v1beta1.ListAndWatchResponse{Devices: m.apiDevices()})
	for {
		select {
		case <-m.stop:
			return nil
		case d := <-m.health:
			log.Warningf("'%s' device marked unhealthy: %s", m.resourceName, d.ID)
			_ = s.Send(&v1beta1.ListAndWatchResponse{Devices: m.apiDevices()})
		}
	}
}

const (
	configBaseDir           = "/etc/xpu"
	containerDirPerm        = 0755
	configFilePerm          = 0644
	pidsSockDir             = "/var/lib/xpu"
	xpuPath                 = "/opt/xpu"
	hostOriginalPath        = "/opt/xpu/lib/libruntime_original.so"
	containerOriginalPath   = "/usr/local/Ascend/ascend-toolkit/latest/lib64/libruntime_original.so"
	preloadLibPath          = "/opt/xpu/lib/libvruntime.so"
	hostPreloadPath         = "/opt/xpu/lib/ld.so.preload"
	containerPreloadPath    = "/etc/ld.so.preload"
	containerVxpuInfoDir    = "/etc/enpu/vcann-rt"
	containerDetectVirtPath = "/usr/bin/systemd-detect-virt"
	hostDetectVirtPath      = "/opt/xpu/bin/systemd-detect-virt"
)

func writeVxpuInfo(dir string, contDevs types.ContainerDevices) error {
	err := os.MkdirAll(dir, containerDirPerm)
	if err != nil {
		log.Errorf("mkdir vxpu info dir error: %v", err)
		return err
	}

	vxpuInfoFilePath := filepath.Clean(filepath.Join(dir, xpu.VxpuInfoFileName))
	vxpuInfoFile, err := os.OpenFile(vxpuInfoFilePath, os.O_WRONLY|os.O_CREATE, configFilePerm)
	if err != nil {
		log.Errorf("create vxpu info file error: %v", err)
		return err
	}
	defer vxpuInfoFile.Close()

	w := bufio.NewWriter(vxpuInfoFile)

	for _, contDev := range contDevs {
		info := fmt.Sprintf("physical-npu-id=%d\n", contDev.Index)
		info += fmt.Sprintf("virtual-npu-id=%d\n", contDev.Vid)
		info += fmt.Sprintf("aicore-quota=%d\n", contDev.Usedcores)
		info += fmt.Sprintf("memory-quota=%d\n", contDev.Usedmem)
		info += fmt.Sprintf("shm-id=%s\n", contDev.UUID)
		info += fmt.Sprintf("scheduling-policy=%d\n", contDev.Policy)
		_, err = w.WriteString(info)
		if err != nil {
			log.Errorf("bufio Writer WriteString error: %v", err)
			return err
		}
	}
	return w.Flush()
}

func createDirAndWriteFile(podId, containerName string, contDevs types.ContainerDevices) error {
	vxpuConfigDirInHost := filepath.Clean(filepath.Join(configBaseDir, podId, containerName))

	err := writeVxpuInfo(vxpuConfigDirInHost, contDevs)
	if err != nil {
		log.Errorf("write vxpu info error: %v, podId: %s, containerName: %s", err, podId, containerName)
		return err
	}

	return nil
}

func (m *DevicePlugin) createContainerAllocateResponse(podId, containerName string,
	devReq types.ContainerDevices) (*v1beta1.ContainerAllocateResponse, error) {
	response := v1beta1.ContainerAllocateResponse{}
	response.Envs = make(map[string]string)
	response.Envs[xpu.VisibleDevices] = xpu.GetVisibleDevices(devReq)
	response.Mounts = []*v1beta1.Mount{}
	response.Mounts = append(response.Mounts, xpu.GetDevShmMount())

	// preload mode, support full card
	if m.isFullCard(devReq) {
		return &response, nil
	}

	err := createDirAndWriteFile(podId, containerName, devReq)
	if err != nil {
		log.Errorf("create dir and write file error: %v, podId: %s, containerName: %s",
			err, podId, containerName)
		return nil, err
	}

	response.Mounts = append(response.Mounts, getXpuMount(podId, containerName)...)

	preloadLibMount := v1beta1.Mount{
		ContainerPath: filepath.Clean(preloadLibPath),
		HostPath:      filepath.Clean(preloadLibPath),
		ReadOnly:      true,
	}

	preloadFileMount := v1beta1.Mount{
		ContainerPath: filepath.Clean(containerPreloadPath),
		HostPath:      filepath.Clean(hostPreloadPath),
		ReadOnly:      true,
	}

	systemdDetectVirtMount := v1beta1.Mount{
		ContainerPath: filepath.Clean(containerDetectVirtPath),
		HostPath:      filepath.Clean(hostDetectVirtPath),
		ReadOnly:      true,
	}
	response.Mounts = append(response.Mounts, &preloadLibMount, &preloadFileMount, &systemdDetectVirtMount)

	return &response, nil
}

func getXpuMount(podId, containerName string) []*v1beta1.Mount {
	pidsSockMount := &v1beta1.Mount{
		ContainerPath: filepath.Clean(pidsSockDir),
		HostPath:      filepath.Clean(pidsSockDir),
		ReadOnly:      true,
	}
	configFileMount := &v1beta1.Mount{
		ContainerPath: filepath.Clean(containerVxpuInfoDir),
		HostPath:      filepath.Clean(filepath.Join(configBaseDir, podId, containerName)),
		ReadOnly:      true,
	}
	xpuPathMount := &v1beta1.Mount{
		ContainerPath: filepath.Clean(xpuPath),
		HostPath:      filepath.Clean(xpuPath),
		ReadOnly:      true,
	}
	return []*v1beta1.Mount{pidsSockMount, configFileMount, xpuPathMount}
}

func (m *DevicePlugin) isFullCard(devReq types.ContainerDevices) bool {
	devs := m.deviceCache.GetCache()
	devInfo := xpu.GetDeviceInfo(devs)

	for _, dev := range devReq {
		if dev.Usedcores != xpu.FullCardCores {
			log.Infof("dev.core=%d", dev.Usedcores)
			return false
		}
		if int32(len(devInfo)) > dev.Index {
			maxMem := devInfo[dev.Index].Devmem
			if dev.Usedmem != maxMem {
				log.Infof("dev.mem=%d, maxMem=%d", dev.Usedmem, maxMem)
				return false
			}
		}
	}
	return true
}

func isSingleCard(devReq types.ContainerDevices) bool {
	return len(devReq) == 1
}

func (m *DevicePlugin) createContainerAllocateResponseHardMode(devReq types.ContainerDevices,
) (*v1beta1.ContainerAllocateResponse, error) {
	response := v1beta1.ContainerAllocateResponse{}
	response.Envs = make(map[string]string)
	response.Envs[xpu.VisibleDevices] = xpu.GetVisibleDevices(devReq)
	if spec := getTemplate(devReq); spec != "" {
		response.Envs[xpu.VxpuSpecs] = spec
	}

	return &response, nil
}

func getTemplate(devReq types.ContainerDevices) string {
	for _, dev := range devReq {
		if dev.Template == xpu.FullCardTemplate {
			return ""
		}
	}
	if len(devReq) == 1 {
		return devReq[0].Template
	}
	return ""
}

func (m *DevicePlugin) Allocate(ctx context.Context, reqs *v1beta1.AllocateRequest) (
	*v1beta1.AllocateResponse, error) {
	log.Infoln("Allocate", reqs.ContainerRequests)
	if len(reqs.ContainerRequests) > 1 {
		return &v1beta1.AllocateResponse{}, errors.New("multiple Container Requests not supported")
	}
	responses := v1beta1.AllocateResponse{}
	nodename := config.NodeName

	current, err := util.GetPendingPod(nodename)
	if err != nil {
		return &v1beta1.AllocateResponse{}, err
	}
	if current == nil {
		log.Errorln("no pending pod found via node lock")
		return &v1beta1.AllocateResponse{}, errors.New("no pending pod found via node lock")
	}
	log.Infoln("Allocate pod ", current.Name)

	for idx := range reqs.ContainerRequests {
		curContainer, devReq, err := util.GetNextDeviceRequest(xpu.DeviceType, *current)
		if err != nil {
			log.Errorln("get device from annotation failed", err.Error())
			util.PodAllocationFailed(nodename, current)
			return &v1beta1.AllocateResponse{}, err
		}
		log.Infoln("deviceAllocateFromAnnotation=", devReq)
		if len(devReq) != len(reqs.ContainerRequests[idx].DevicesIDs) {
			log.Errorln("device number not matched", devReq, reqs.ContainerRequests[idx].DevicesIDs)
			util.PodAllocationFailed(nodename, current)
			return &v1beta1.AllocateResponse{}, errors.New("device number not matched")
		}

		err = util.EraseNextDeviceTypeFromAnnotation(xpu.DeviceType, *current)
		if err != nil {
			log.Errorln("Erase annotation failed", err.Error())
			util.PodAllocationFailed(nodename, current)
			return &v1beta1.AllocateResponse{}, err
		}

		var response *v1beta1.ContainerAllocateResponse
		if util.IsSoftMode() {
			response, err = m.createContainerAllocateResponse(string(current.UID), curContainer.Name, devReq)
		} else {
			response, err = m.createContainerAllocateResponseHardMode(devReq)
		}
		if err != nil {
			util.PodAllocationFailed(nodename, current)
			return &v1beta1.AllocateResponse{}, err
		}
		responses.ContainerResponses = append(responses.ContainerResponses, response)
	}
	log.Infoln("Allocate Response", responses.ContainerResponses)
	util.PodAllocationTrySuccess(nodename, current)
	return &responses, nil
}

func (m *DevicePlugin) PreStartContainer(context.Context, *v1beta1.PreStartContainerRequest) (
	*v1beta1.PreStartContainerResponse, error) {
	return &v1beta1.PreStartContainerResponse{}, nil
}

func (m *DevicePlugin) dial(unixSocketPath string, timeout time.Duration) (*grpc.ClientConn, error) {
	c, err := grpc.Dial(unixSocketPath, grpc.WithInsecure(), grpc.WithBlock(),
		grpc.WithTimeout(timeout),
		grpc.WithDialer(func(addr string, timeout time.Duration) (net.Conn, error) {
			return net.DialTimeout("unix", addr, timeout)
		}),
	)

	if err != nil {
		return nil, err
	}

	return c, nil
}

func (m *DevicePlugin) Devices() []*xpu.Device {
	return m.deviceCache.GetCache()
}

func (m *DevicePlugin) apiDevices() []*v1beta1.Device {
	devices := m.Devices()
	var res []*v1beta1.Device
	for _, dev := range devices {
		for i := uint(0); i < config.DeviceSplitCount; i++ {
			id := fmt.Sprintf("%v-%v", dev.ID, i)
			res = append(res, &v1beta1.Device{
				ID:       id,
				Health:   dev.Health,
				Topology: nil,
			})
		}
	}
	return res
}
