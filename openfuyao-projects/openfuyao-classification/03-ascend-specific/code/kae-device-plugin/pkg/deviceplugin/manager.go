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
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"path"
	"reflect"
	"regexp"
	"strings"
	"sync"
	"time"

	"google.golang.org/grpc"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/kubernetes"
	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"openfuyao.com/kae-device-plugin/pkg/zlog"
)

const (
	devDirectory         = "/dev"
	resourceName         = "openfuyao.com/kae.hpre"
	devPluginDir         = v1beta1.DevicePluginPath
	hpreDevice           = `^hisi_hpre*`
	pluginEndpointPrefix = "hisi"
	noPCIRetryTime       = 30
	checkMinute          = 10
)

var (
	extraPath             = ""
	devicePluginDirectory = devPluginDir
)

// Device is the structure of device managed by Device Plugins.
type Device struct {
	ID          string
	health      string
	DevType     string
	DevFileName []string
}

// KAEManager manages KAE devices.
type KAEManager struct {
	kubeClient      kubernetes.Interface
	devDirectory    string
	mountPath       []v1beta1.Mount
	defaultDevices  []string
	devices         map[string]v1beta1.Device
	devAllocateResp *v1beta1.ContainerAllocateResponse
	grpcServer      *grpc.Server
	devicesInfo     map[string]*Device
	socket          string
	stop            chan bool
	stopMonitor     chan struct{}
	stopCheck       chan struct{}
	devicesMutex    sync.Mutex
	hasDeviceChange bool
}

// NewKAEManager create a new manager
func NewKAEManager() *KAEManager {
	return &KAEManager{
		devDirectory: devDirectory,
		devices:      make(map[string]v1beta1.Device),
		stop:         make(chan bool, 1),
	}
}

// 查找设备ID，更新设备状态
func (m *KAEManager) discoverDevices() error {
	hpreReg := regexp.MustCompile(hpreDevice)
	m.defaultDevices = []string{}
	devices := make(map[string]v1beta1.Device)

	deviceDirectory := m.devDirectory
	if deviceDirectory == "" {
		deviceDirectory = devDirectory
	}

	files, err := ioutil.ReadDir(deviceDirectory)
	if err != nil {
		zlog.Errorf("read directory failed! error: %v", err)
		return err
	}

	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if hpreReg.MatchString(f.Name()) {
			devices[f.Name()] = v1beta1.Device{ID: f.Name(), Health: v1beta1.Healthy}
		}
	}

	if !compareMaps(devices, m.devices) {
		m.hasDeviceChange = true
		zlog.Infof("find kae hpre device: %v", devices)
	}

	m.devices = devices
	if len(m.devices) == 0 {
		zlog.Error("not found kae hpre device")
		return fmt.Errorf("not found kae hpre device")
	}
	return nil
}

func compareMaps(map1, map2 map[string]v1beta1.Device) bool {
	// 首先比较 map 的大小，如果大小不同，则 map 内容不同
	if len(map1) != len(map2) {
		return false
	}

	// 然后遍历 map1 的键，判断 map2 是否也包含相同的键
	for key, value := range map1 {
		if _, exists := map2[key]; !exists {
			return false
		}

		// 如果键存在于 map2 中，则比较对应的值是否相同
		if !reflect.DeepEqual(value, map2[key]) {
			return false
		}
	}
	return true
}

// GetDeviceState return device's state
func (m *KAEManager) GetDeviceState(DeviceName string) string {
	// 当前kae没有接口获取其健康状态，发现即认为健康
	return v1beta1.Healthy
}

// Start the KAEManager, check input enc params, look for the matched KAE
func (m *KAEManager) Start() error {
	if _, err := os.Stat(extraPath); err != nil && os.IsNotExist(err) {
		zlog.Errorf("%s is not exist! error: %v", extraPath, err)
		time.Sleep(noPCIRetryTime * time.Second)
		return err
	} else if err != nil {
		zlog.Errorf("stat  extraPath failed! error: %v", err)
		return err
	}

	// 检查 VM 上的 device
	if err := m.discoverDevices(); err != nil {
		return fmt.Errorf("get kae device failed: %v", err)
	}
	return nil
}

// Serve start grpc server, register KAE resource.
func (m *KAEManager) Serve(pluginEndpoint string) {
	for {
		select {
		case _, ok := <-m.stop:
			if ok {
				close(m.stop)
			}
			return
		default:
			pluginEndpointPath := path.Join(v1beta1.DevicePluginPath, pluginEndpoint)
			zlog.Infof("starting device-plugin server at: %s", pluginEndpointPath)

			lis, err := net.Listen("unix", pluginEndpointPath)
			if err != nil {
				zlog.Errorf("starting device-plugin server failed: %v", err)
				continue
			}

			m.socket = pluginEndpointPath
			m.grpcServer = grpc.NewServer()
			m.stopCheck, m.stopMonitor = make(chan struct{}, 1), make(chan struct{}, 1)
			plugin := &pluginService{kaeManager: m}
			plugin.RegisterService()

			var wg sync.WaitGroup
			wg.Add(1)
			// 启动device plugin服务
			go func() {
				defer wg.Done()
				err := m.grpcServer.Serve(lis)
				zlog.Errorf("device-plugin server stopped serving: %v", err)
				// 关闭devicePluginMonitor的协程
				m.stopMonitor <- struct{}{}
				// 关闭CheckListAndWatch的go协程
				m.stopCheck <- struct{}{}
				return
			}()

			if err = registerWithKubelet(m, pluginEndpoint, resourceName); err != nil {
				zlog.Errorf("cloud not register device plugin: %s", err.Error())
			}

			// kubelet 重启时，会删除/var/lib/kubelet/device-plugins/下所有sockets，
			// 检查device plugin socket 是否存在，不存在时，停止grpc server， 重新启动
			go wait.Until(func() { devicePluginMonitor(m, pluginEndpointPath) }, time.Second, m.stopMonitor)

			dClient, err := NewDeviceClient(pluginEndpointPath)
			if err != nil {
				zlog.Errorf("create dClient err: %v", err)
				return
			}
			go wait.Until(func() { dClient.CheckListAndWatch() }, checkMinute*time.Minute, m.stopCheck)

			wg.Wait()
			// kubelet 重启后manager设备还存在，但是节点上的数据已经为0了，需要重启发送设备列表
			m.hasDeviceChange = true
			time.Sleep(time.Second)
		}
	}
}

// registerWithKubelet 向 kubelet 注册
func registerWithKubelet(kaeManager *KAEManager, pluginEndpoint, resourceName string) error {
	// 需要监控 KubeletSocket 是否存在， KubeletSocket 存在时， device plugin 向 Kubelet 注册
	if _, err := os.Stat(v1beta1.KubeletSocket); err == nil {
		// 等待grpcServer Ready
		for len(kaeManager.grpcServer.GetServiceInfo()) <= 0 {
			time.Sleep(time.Second)
		}

		// 向kubelet注册
		zlog.Info("device-plugin server started serving")

		err := RegisterWithV1Beta1Kubelet(pluginEndpoint, resourceName)
		if err != nil {
			kaeManager.grpcServer.Stop()
			return err
		}
		zlog.Info("device-plugin registered with the kubelet")
		return nil
	} else {
		return fmt.Errorf("registerWithKubelet failed! err: %v", err)
	}
}

// devicePluginMonitor 检查Device Plugin Socket是否存在，不存在时，停止grpc server
func devicePluginMonitor(kaeManager *KAEManager, pluginEndpointPath string) {
	if _, err := os.Lstat(pluginEndpointPath); err != nil {
		zlog.Errorf("stopping device-plugin server at: %s, err is %s", pluginEndpointPath, err.Error())
		kaeManager.grpcServer.Stop()
	}
}

func cycleRegisterWithKubelet(kaeManager *KAEManager, pluginEndpoint, resourceName string) {
	zlog.Infof("cycleRegisterWithKubelet")
	err := registerWithKubelet(kaeManager, pluginEndpoint, resourceName)
	if err != nil {
		zlog.Errorf("could not register device plugin: %s", err.Error())
	}
}

// RemoveKAESock 删除sock文件
func RemoveKAESock() {
	files, err := ioutil.ReadDir(devicePluginDirectory)
	if err != nil {
		zlog.Errorf("read directory failed! error: %v", err)
	}
	for _, f := range files {
		if f.IsDir() {
			continue
		}
		if strings.HasPrefix(f.Name(), pluginEndpointPrefix) {
			zlog.Infof("found hisi sock %q", f.Name())
			if err := os.Remove(fmt.Sprintf("%s/%s", devicePluginDirectory, f.Name())); err != nil {
				zlog.Errorf("delete hisi sock: %q failed! error: %v", f.Name(), err)
			}
		}
	}
}

// SetUname 获取uname -r信息
func SetUname() {
	libPath, err := runCmd("uname -r")
	if err != nil {
		zlog.Errorf("get uname failde! error is : %v", err)
		return
	}
	zlog.Infof("uname: %s", libPath)
	extraPath = "/lib/modules/" + strings.TrimSpace(libPath) + "/extra"
}

// SetMod 设置KAE驱动和日志权限，此操作是保证非root用户可以使用KAE设备
func SetMod() {
	_, err := runCmd("chmod 777 /dev/hisi_*")
	if err != nil {
		zlog.Errorf("chmod dev failed! error is : %v", err)
	}
	_, err = runCmd("chmod 777 /var/log/kae.log")
	if err != nil {
		zlog.Errorf("chmod log failed! error is : %v", err)
	}
}

func runCmd(cmd string) (string, error) {
	out, err := exec.Command("sh", "-c", cmd).CombinedOutput()
	if err != nil {
		return "", fmt.Errorf("run command %q: output %q: %w", cmd, string(out), err)
	}
	return string(out), nil
}
