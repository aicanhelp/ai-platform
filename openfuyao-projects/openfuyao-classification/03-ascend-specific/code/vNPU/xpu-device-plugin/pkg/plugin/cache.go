/*
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
	"sync"

	"k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

	"huawei.com/vxpu-device-plugin/pkg/plugin/xpu"
)

type DeviceCache struct {
	xpu.DeviceManager
	cache     []*xpu.Device
	stopCh    chan interface{}
	unhealthy chan *xpu.Device
	notifyCh  map[string]chan *xpu.Device
	mutex     sync.Mutex
}

func NewDeviceCache() *DeviceCache {
	return &DeviceCache{
		stopCh:    make(chan interface{}),
		unhealthy: make(chan *xpu.Device),
		notifyCh:  make(map[string]chan *xpu.Device),
	}
}

func (d *DeviceCache) AddNotifyChannel(name string, ch chan *xpu.Device) {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	d.notifyCh[name] = ch
}

func (d *DeviceCache) RemoveNotifyChannel(name string) {
	d.mutex.Lock()
	defer d.mutex.Unlock()
	delete(d.notifyCh, name)
}

func (d *DeviceCache) Start() {
	d.cache = d.Devices()
	go d.CheckHealth(d.stopCh, d.cache, d.unhealthy)
	go d.notifyLoop()
}

func (d *DeviceCache) Stop() {
	close(d.stopCh)
}

func (d *DeviceCache) GetCache() []*xpu.Device {
	return d.cache
}

func (d *DeviceCache) notifyLoop() {
	for {
		select {
		case <-d.stopCh:
			return
		case dev := <-d.unhealthy:
			dev.Health = v1beta1.Unhealthy
			d.notify(dev)
		}
	}
}

func (d *DeviceCache) notify(dev *xpu.Device) {
	d.mutex.Lock()
	chs := make([]chan *xpu.Device, 0, len(d.notifyCh))
	for _, ch := range d.notifyCh {
		if ch != nil {
			chs = append(chs, ch)
		}
	}
	d.mutex.Unlock()

	for _, ch := range chs {
		if ch != nil {
			ch <- dev
		}
	}
}
