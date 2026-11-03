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
package npuservice

import (
	"encoding/json"
	"reflect"
	"strconv"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/common/cache"
	"huawei.com/xpu-exporter/common/client"
	"huawei.com/xpu-exporter/common/utils"
	"huawei.com/xpu-exporter/versions"
)

const (
	podName       = "pod_name"
	cntrName      = "container_name"
	npuUUid       = "npu_uuid"
	podUid        = "pod_uuid"
	vnpuId        = "vnpu_id"
	vnpuCoreLimit = "vnpu_core_limit"
	vnpuMemLimit  = "vnpu_mem_limit"
	nodeName      = "node_name"
	nodeIp        = "node_ip"
)

var (
	vnpuLabel = []string{npuUUid, nodeName, nodeIp, podUid, cntrName, vnpuId, vnpuCoreLimit, vnpuMemLimit}
)

var (
	versionInfoDesc = prometheus.NewDesc("npu_exporter_version_info",
		"exporter version with value '1'", []string{"exporterVersion"}, nil)
	xpuVnpuMemoryUtilizationDesc = prometheus.NewDesc("xpu_vnpu_mem_util",
		"the utilization rate of memory for vnpu", vnpuLabel, nil)
	xpuVnpuNumberDesc = prometheus.NewDesc("xpu_vnpu_num",
		"real time quantity of vnpu", []string{nodeName, nodeIp, npuUUid}, nil)
	xpuVnpuPodNumberDesc = prometheus.NewDesc("xpu_vnpu_pod_num",
		"real time quantity of vnpu pods", []string{nodeName, nodeIp, npuUUid}, nil)
	xpuVnpuModeDesc = prometheus.NewDesc("xpu_vnpu_mode",
		"the split mode of vNPU (0: soft, 1: hard)", []string{nodeName, nodeIp, npuUUid}, nil)
	xpuVnpuSoftSchedulerPolicyDesc = prometheus.NewDesc("xpu_vnpu_soft_scheduler_policy",
		"vNPU soft split scheduling policy (1: fixed-share, 2: elastic, 3: best-effort)", []string{nodeName, nodeIp, npuUUid}, nil)

	descriptions = []*prometheus.Desc{versionInfoDesc, xpuVnpuMemoryUtilizationDesc, xpuVnpuNumberDesc,
		xpuVnpuPodNumberDesc, xpuVnpuModeDesc, xpuVnpuSoftSchedulerPolicyDesc}
)

const (
	cacheSize = 128
)

type npuCollector struct {
	cache      *cache.ConcurrencyLRUCache
	updateTime time.Duration
	cacheTime  time.Duration
}

func (n *npuCollector) Describe(ch chan<- *prometheus.Desc) {
	if ch == nil {
		log.Warningln("Invalid param in function Describe")
		return
	}
	ch <- versionInfoDesc
	ch <- xpuVnpuMemoryUtilizationDesc
	ch <- xpuVnpuNumberDesc
	ch <- xpuVnpuPodNumberDesc
	ch <- xpuVnpuModeDesc
	ch <- xpuVnpuSoftSchedulerPolicyDesc
}

func (n *npuCollector) Collect(ch chan<- prometheus.Metric) {
	ch <- prometheus.MustNewConstMetric(versionInfoDesc, prometheus.GaugeValue, 1,
		[]string{versions.BuildVersion}...)

	npuDeviceMap := getVnpuInfoInCache(ch, n)
	var vnpuDeviceTotalCount = 0
	for _, npuDevice := range npuDeviceMap {
		vnpuDeviceCount := len(npuDevice.VxpuDeviceList)
		if vnpuDeviceCount <= 0 {
			continue
		}
		vnpuDeviceTotalCount += vnpuDeviceCount
		updateVnpuDeviceInfo(ch, npuDevice)
	}
}

func getVnpuInfoInCache(ch chan<- prometheus.Metric, n *npuCollector) map[string]*utils.XPUDevice {
	if ch == nil {
		log.Errorln("metric channel is nil")
		return nil
	}
	obj, _ := n.cache.Get(vnpuInfoCacheKey)
	if obj == nil {
		log.Warningln("no cache, start to get vnpuInfo and rebuild cache.")
		vnpuInfo, err := client.GetAllVxpuInfo()
		if err != nil {
			log.Errorf("get vnpuInfo error: %v", err)
			return nil
		}
		if err = n.cache.Set(vnpuInfoCacheKey, vnpuInfo, n.cacheTime); err != nil {
			log.Errorf("no cache for prometheus, try to build cache failed, error is: %v", err)
		} else {
			log.Infoln("rebuild cache successfully")
		}
		obj = vnpuInfo
	}
	var npuDeviceMap map[string]*utils.XPUDevice
	err := json.Unmarshal([]byte(obj.(string)), &npuDeviceMap)
	if err != nil {
		log.Errorf("Error vnpu info cache and convert failed: %v", err)
	}
	return npuDeviceMap
}

func updateVnpuDeviceInfo(ch chan<- prometheus.Metric, npu *utils.XPUDevice) {
	if !validate(ch) {
		log.Warningln("Invalid param in function updateVnpuDeviceInfo")
		return
	}
	var vnpuPodNumber = 0
	vnpuPodMap := make(map[string]int)
	for _, vnpu := range npu.VxpuDeviceList {
		ch <- prometheus.MustNewConstMetric(xpuVnpuMemoryUtilizationDesc, prometheus.GaugeValue,
			vnpu.VxpuMemoryUtilization, []string{npu.Id, npu.NodeName, npu.NodeIp, vnpu.PodUID,
				vnpu.ContainerName, vnpu.Id, strconv.Itoa(int(vnpu.VxpuCoreLimit)),
				strconv.Itoa(int(vnpu.VxpuMemoryLimit))}...)
		if _, ok := vnpuPodMap[vnpu.PodUID]; !ok {
			vnpuPodNumber += 1
			vnpuPodMap[vnpu.PodUID] = vnpuPodNumber
		}
	}
	ch <- prometheus.MustNewConstMetric(xpuVnpuNumberDesc, prometheus.GaugeValue, float64(len(npu.VxpuDeviceList)),
		[]string{npu.NodeName, npu.NodeIp, npu.Id}...)
	ch <- prometheus.MustNewConstMetric(xpuVnpuPodNumberDesc, prometheus.GaugeValue, float64(vnpuPodNumber),
		[]string{npu.NodeName, npu.NodeIp, npu.Id}...)
	modeValue := 0.0
	if npu.Mode == "hard" {
		modeValue = 1.0
	}
	ch <- prometheus.MustNewConstMetric(xpuVnpuModeDesc, prometheus.GaugeValue,
		modeValue, []string{npu.NodeName, npu.NodeIp, npu.Id}...)
	if npu.Mode == "soft" {
		ch <- prometheus.MustNewConstMetric(xpuVnpuSoftSchedulerPolicyDesc, prometheus.GaugeValue,
			float64(npu.Policy), []string{npu.NodeName, npu.NodeIp, npu.Id}...)
	}
}

func validate(ch chan<- prometheus.Metric, objs ...interface{}) bool {
	if ch == nil {
		return false
	}
	for _, v := range objs {
		val := reflect.ValueOf(v)
		if val.Kind() != reflect.Ptr {
			return false
		}
		if val.IsNil() {
			return false
		}
	}
	return true
}
