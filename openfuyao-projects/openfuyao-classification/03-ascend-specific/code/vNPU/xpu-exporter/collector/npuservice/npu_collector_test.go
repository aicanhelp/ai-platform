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

package npuservice

import (
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"

	"huawei.com/xpu-exporter/common/cache"
	"huawei.com/xpu-exporter/common/client"
	"huawei.com/xpu-exporter/common/utils"
)

var (
	testNodeName  = "test-node"
	testNodeIP    = "127.0.0.1"
	testNPUID     = "test-npu-0"
	testPodUID1   = "pod-uid-1"
	testPodUID2   = "pod-uid-2"
	testCntrName  = "test-container"
	testVnpuID1   = "vnpu-1"
	testVnpuID2   = "vnpu-2"
	testCacheTime = 60 * time.Second
	testCoreLimit = int64(4)
	testMemLimit  = int64(8192)
	testMemUtil   = 50.0
)

func TestDescribe(t *testing.T) {
	n := &npuCollector{}
	ch := make(chan *prometheus.Desc, len(descriptions))
	n.Describe(ch)

	for i := 0; i < len(descriptions); i++ {
		assert.Equal(t, descriptions[i], <-ch)
	}
}

func TestDescribeNilChannel(t *testing.T) {
	n := &npuCollector{}
	n.Describe(nil)
}

func TestCollect(t *testing.T) {
	n := &npuCollector{}
	ch := make(chan prometheus.Metric, 1)
	n.Collect(ch)

	metric := <-ch
	assert.Equal(t, versionInfoDesc, metric.Desc())
}

func TestCollectWithCachedData(t *testing.T) {
	_, n := setupNpuCollectorWithCache(t, buildTestNPUDeviceMap())
	ch := make(chan prometheus.Metric, 10)
	n.Collect(ch)
	metric := <-ch
	assert.Equal(t, versionInfoDesc, metric.Desc())
	for i := 0; i < 5; i++ {
		m := <-ch
		assert.NotNil(t, m)
	}
}

func TestCollectEmptyVnpuDeviceList(t *testing.T) {
	npuDeviceMap := map[string]*utils.XPUDevice{
		testNPUID: &utils.XPUDevice{
			Id:             testNPUID,
			NodeName:       testNodeName,
			NodeIp:         testNodeIP,
			VxpuDeviceList: utils.VxpuDevices{},
		},
	}
	_, n := setupNpuCollectorWithCache(t, npuDeviceMap)
	ch := make(chan prometheus.Metric, 10)
	n.Collect(ch)
	metric := <-ch
	assert.Equal(t, versionInfoDesc, metric.Desc())
}

func TestGetVnpuInfoInCacheNilChannel(t *testing.T) {
	testCache := cache.New(cacheSize)
	n := &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
	result := getVnpuInfoInCache(nil, n)
	assert.Nil(t, result)
}

func TestGetVnpuInfoInCacheCacheHit(t *testing.T) {
	_, n := setupNpuCollectorWithCache(t, buildTestNPUDeviceMap())
	ch := make(chan prometheus.Metric, 10)
	result := getVnpuInfoInCache(ch, n)
	assert.NotNil(t, result)
	assert.Contains(t, result, testNPUID)
}

func TestGetVnpuInfoInCacheInvalidJSON(t *testing.T) {
	testCache := cache.New(cacheSize)
	err := testCache.Set(vnpuInfoCacheKey, "invalid-json-data", testCacheTime)
	assert.NoError(t, err)
	n := &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
	ch := make(chan prometheus.Metric, 10)
	result := getVnpuInfoInCache(ch, n)
	assert.Nil(t, result)
}

func TestGetVnpuInfoInCacheMissRebuildError(t *testing.T) {
	testCache := cache.New(cacheSize)
	n := &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "", fmt.Errorf("mock get error")
	})
	defer patch.Reset()
	ch := make(chan prometheus.Metric, 10)
	result := getVnpuInfoInCache(ch, n)
	assert.Nil(t, result)
}

func TestGetVnpuInfoInCacheMissSetError(t *testing.T) {
	testCache := cache.New(cacheSize)
	n := &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
	patchGet := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		data, err := json.Marshal(buildTestNPUDeviceMap())
		if err != nil {
			return "", err
		}
		return string(data), nil
	})
	defer patchGet.Reset()

	var c *cache.ConcurrencyLRUCache
	patchSet := gomonkey.ApplyMethod(c, "Set", func(_ *cache.ConcurrencyLRUCache, key string, value interface{}, expireTime time.Duration) error {
		return fmt.Errorf("mock set error")
	})
	defer patchSet.Reset()

	ch := make(chan prometheus.Metric, 10)
	result := getVnpuInfoInCache(ch, n)
	assert.NotNil(t, result)
	assert.Contains(t, result, testNPUID)
}

func TestGetVnpuInfoInCacheMissRebuildSuccess(t *testing.T) {
	testCache := cache.New(cacheSize)
	n := &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		data, err := json.Marshal(buildTestNPUDeviceMap())
		if err != nil {
			return "", err
		}
		return string(data), nil
	})
	defer patch.Reset()

	ch := make(chan prometheus.Metric, 10)
	result := getVnpuInfoInCache(ch, n)
	assert.NotNil(t, result)
	assert.Contains(t, result, testNPUID)
}

func TestUpdateVnpuDeviceInfoNilChannel(t *testing.T) {
	npuDevice := buildTestNPDDevice("hard", 0)
	updateVnpuDeviceInfo(nil, npuDevice)
}

func TestUpdateVnpuDeviceInfoHardMode(t *testing.T) {
	npuDevice := buildTestNPDDevice("hard", 0)
	ch := make(chan prometheus.Metric, 10)
	updateVnpuDeviceInfo(ch, npuDevice)
	for i := 0; i < 4; i++ {
		m := <-ch
		assert.NotNil(t, m)
	}
}

func TestUpdateVnpuDeviceInfoSoftMode(t *testing.T) {
	npuDevice := buildTestNPDDevice("soft", 2)
	ch := make(chan prometheus.Metric, 10)
	updateVnpuDeviceInfo(ch, npuDevice)
	for i := 0; i < 5; i++ {
		m := <-ch
		assert.NotNil(t, m)
	}
}

func TestUpdateVnpuDeviceInfoDuplicatePodUID(t *testing.T) {
	npuDevice := buildTestNPDDevice("hard", 0)
	npuDevice.VxpuDeviceList[1].PodUID = testPodUID1
	ch := make(chan prometheus.Metric, 10)
	updateVnpuDeviceInfo(ch, npuDevice)
	for i := 0; i < 4; i++ {
		m := <-ch
		assert.NotNil(t, m)
	}
}

func TestValidateNilChannel(t *testing.T) {
	result := validate(nil)
	assert.False(t, result)
}

func TestValidateNonPointer(t *testing.T) {
	result := validate(make(chan prometheus.Metric, 1), "string-value")
	assert.False(t, result)
}

func TestValidateNilPointer(t *testing.T) {
	var ptr *utils.XPUDevice
	result := validate(make(chan prometheus.Metric, 1), ptr)
	assert.False(t, result)
}

func TestValidateValidPointer(t *testing.T) {
	npuDevice := &utils.XPUDevice{Id: testNPUID}
	ch := make(chan prometheus.Metric, 1)
	result := validate(ch, npuDevice)
	assert.True(t, result)
}

func TestValidateOnlyChannel(t *testing.T) {
	ch := make(chan prometheus.Metric, 1)
	result := validate(ch)
	assert.True(t, result)
}

func buildTestNPUDeviceMap() map[string]*utils.XPUDevice {
	npuDevice := buildTestNPDDevice("hard", 0)
	return map[string]*utils.XPUDevice{
		testNPUID: npuDevice,
	}
}

func buildTestNPDDevice(mode string, policy int32) *utils.XPUDevice {
	return &utils.XPUDevice{
		Id:       testNPUID,
		NodeName: testNodeName,
		NodeIp:   testNodeIP,
		Mode:     mode,
		Policy:   policy,
		VxpuDeviceList: utils.VxpuDevices{
			utils.VxpuDevice{
				Id:                    testVnpuID1,
				PodUID:                testPodUID1,
				ContainerName:         testCntrName,
				VxpuMemoryUtilization: testMemUtil,
				VxpuCoreLimit:         testCoreLimit,
				VxpuMemoryLimit:       testMemLimit,
			},
			utils.VxpuDevice{
				Id:                    testVnpuID2,
				PodUID:                testPodUID2,
				ContainerName:         testCntrName,
				VxpuMemoryUtilization: testMemUtil,
				VxpuCoreLimit:         testCoreLimit,
				VxpuMemoryLimit:       testMemLimit,
			},
		},
	}
}

func setupNpuCollectorWithCache(t *testing.T, npuDeviceMap map[string]*utils.XPUDevice) (*cache.ConcurrencyLRUCache, *npuCollector) {
	testCache := cache.New(cacheSize)
	data, err := json.Marshal(npuDeviceMap)
	if err != nil {
		t.Fatalf("failed to marshal test data: %v", err)
	}
	err = testCache.Set(vnpuInfoCacheKey, string(data), testCacheTime)
	if err != nil {
		t.Fatalf("failed to set cache: %v", err)
	}
	return testCache, &npuCollector{
		cache:     testCache,
		cacheTime: testCacheTime,
	}
}
