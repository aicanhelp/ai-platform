/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package plugin

import (
	"time"

	"huawei.com/vxpu-device-plugin/pkg/graph"
	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/vxpu-device-plugin/pkg/plugin/config"
	"huawei.com/vxpu-device-plugin/pkg/plugin/types"
	"huawei.com/vxpu-device-plugin/pkg/plugin/util"
	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

const (
	failRetryInterval = 5
	registerInterval  = 30
)

// DeviceRegister handles device discovery and registration with Kubernetes
type DeviceRegister struct {
	deviceCache      *DeviceCache
	topologyProvider graph.TopologyProvider
}

// NewDeviceRegister creates a new device register instance
func NewDeviceRegister(deviceCache *DeviceCache) *DeviceRegister {
	return &DeviceRegister{
		deviceCache:      deviceCache,
		topologyProvider: xpu.NewTopologyProvider(),
	}
}

// Start begins the device registration monitoring process
func (r *DeviceRegister) Start() {
	go r.watchAndRegister()
}

// apiDevices converts cached devices to API device information format
func (r *DeviceRegister) apiDevices() []*types.DeviceInfo {
	devs := r.deviceCache.GetCache()
	return xpu.GetDeviceInfo(devs)
}

// registerInAnnotation updates Kubernetes node annotations with device information
func (r *DeviceRegister) registerInAnnotation() error {
	devices := r.apiDevices()
	encodedDevices := util.EncodeNodeDevices(devices)
	annotations := map[string]string{
		xpu.NodeVXPURegister:  encodedDevices,
		xpu.NodeVXPUHandshake: "Reported_" + time.Now().Format("2006.01.02 15:04:05"),
		xpu.NodeXpuTopology:   r.topologyProvider.Topology(),
	}

	log.Infoln("Reporting devices", encodedDevices, "in", time.Now().Format("2006.01.02 15:04:05"))
	err := util.PatchNodeAnnotations(config.NodeName, annotations)
	if err != nil {
		log.Errorln("k8s patch node error:", err.Error(), "node name:", config.NodeName)
		return err
	}
	return nil
}

// watchAndRegister continuously monitors and registers devices with Kubernetes
func (r *DeviceRegister) watchAndRegister() {
	log.Infof("into watchAndRegister")
	lastSucceed := true
	for {
		err := r.registerInAnnotation()
		if err != nil {
			if lastSucceed == false {
				break
			}
			lastSucceed = false
			log.Errorln("register vxpu failed once, try again.")
			time.Sleep(time.Second * failRetryInterval)
		} else {
			lastSucceed = true
			time.Sleep(time.Second * registerInterval)
		}
	}
	log.Fatalln("register vxpu failed twice, exit program!")
}
