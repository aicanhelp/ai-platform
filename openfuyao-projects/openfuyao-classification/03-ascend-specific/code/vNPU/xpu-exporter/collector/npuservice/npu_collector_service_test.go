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
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"

	"huawei.com/xpu-exporter/common/cache"
	"huawei.com/xpu-exporter/common/client"
)

const (
	cacheTime  = 60 * time.Second
	updateTime = 15 * time.Second
)

func TestNew(t *testing.T) {
	collectorService := New(CollectorName)
	assert.Equal(t, CollectorName, collectorService.GetName())
}

func TestGetName(t *testing.T) {
	collectorService := New(CollectorName)
	serviceName := collectorService.GetName()
	assert.Equal(t, CollectorName, serviceName)
}

func TestCreateCollector(t *testing.T) {
	expectedColletor := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  cacheTime,
		updateTime: updateTime,
	}
	collectorService := New(CollectorName)
	collector := collectorService.CreateCollector(cacheTime, updateTime)
	assert.Equal(t, expectedColletor, collector)
}

func TestStart_GetVnpuInfoError(t *testing.T) {
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "", fmt.Errorf("mock error")
	})
	defer patch.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	s := &npuCollectorService{
		serviceName: CollectorName,
		collector: npuCollector{
			cache:      cache.New(cacheSize),
			cacheTime:  100 * time.Millisecond,
			updateTime: 100 * time.Millisecond,
		},
	}
	s.Start(ctx, cancel)
}

func TestStart_ContextCancelled(t *testing.T) {
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "mock vnpu info", nil
	})
	defer patch.Reset()

	ctx, cancel := context.WithCancel(context.Background())

	s := &npuCollectorService{
		serviceName: CollectorName,
		collector: npuCollector{
			cache:      cache.New(cacheSize),
			cacheTime:  100 * time.Millisecond,
			updateTime: 50 * time.Millisecond,
		},
	}

	done := make(chan struct{})
	go func() {
		s.Start(ctx, cancel)
		done <- struct{}{}
	}()

	time.Sleep(200 * time.Millisecond)
	cancel()

	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("start did not finish within timeout")
	}
}

func TestVnpuInfoCollect(t *testing.T) {
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "", fmt.Errorf("mock error")
	})
	defer patch.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	group := &sync.WaitGroup{}
	n := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  100 * time.Millisecond,
		updateTime: 100 * time.Millisecond,
	}
	vnpuInfoCollect(ctx, group, n, func() { /* no-op: test verifies goroutine exit without panic */ })
	group.Wait()
}

func TestSetVnpuInfoToCache_ContextCancelled(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	group := &sync.WaitGroup{}
	group.Add(1)
	n := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  100 * time.Millisecond,
		updateTime: 100 * time.Millisecond,
	}
	setVnpuInfoToCache(ctx, group, n, nil)
}

func TestSetVnpuInfoToCache_GetVnpuInfoError(t *testing.T) {
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "", fmt.Errorf("mock error")
	})
	defer patch.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	group := &sync.WaitGroup{}
	group.Add(1)
	n := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  100 * time.Millisecond,
		updateTime: 100 * time.Millisecond,
	}
	setVnpuInfoToCache(ctx, group, n, func() { /* no-op: test verifies goroutine exit without panic */ })
}

type setVnpuInfoTestEnv struct {
	ctx    context.Context
	cancel context.CancelFunc
	group  *sync.WaitGroup
	n      *npuCollector
}

func setupSetVnpuInfoTest() *setVnpuInfoTestEnv {
	ctx, cancel := context.WithCancel(context.Background())
	group := &sync.WaitGroup{}
	group.Add(1)
	n := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  100 * time.Millisecond,
		updateTime: 50 * time.Millisecond,
	}
	go func() {
		time.Sleep(150 * time.Millisecond)
		cancel()
	}()
	return &setVnpuInfoTestEnv{ctx: ctx, cancel: cancel, group: group, n: n}
}

func TestSetVnpuInfoToCache_CacheSetSuccess(t *testing.T) {
	patch := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "mock vnpu info", nil
	})
	defer patch.Reset()

	env := setupSetVnpuInfoTest()
	setVnpuInfoToCache(env.ctx, env.group, env.n, nil)
}

func TestSetVnpuInfoToCache_CacheSetError(t *testing.T) {
	patchGetVnpuInfo := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "mock vnpu info", nil
	})
	defer patchGetVnpuInfo.Reset()

	var c *cache.ConcurrencyLRUCache
	patchCacheSet := gomonkey.ApplyMethod(c, "Set", func(_ *cache.ConcurrencyLRUCache, key string, value interface{}, expireTime time.Duration) error {
		return fmt.Errorf("mock cache set error")
	})
	defer patchCacheSet.Reset()

	env := setupSetVnpuInfoTest()
	setVnpuInfoToCache(env.ctx, env.group, env.n, nil)
}

func TestSetVnpuInfoToCache_TickerClosed(t *testing.T) {
	patchGetVnpuInfo := gomonkey.ApplyFunc(client.GetAllVxpuInfo, func() (string, error) {
		return "mock vnpu info", nil
	})
	defer patchGetVnpuInfo.Reset()

	ch := make(chan time.Time)
	close(ch)
	patchNewTicker := gomonkey.ApplyFunc(time.NewTicker, func(_ time.Duration) *time.Ticker {
		return &time.Ticker{C: ch}
	})
	defer patchNewTicker.Reset()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	group := &sync.WaitGroup{}
	group.Add(1)
	n := &npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  100 * time.Millisecond,
		updateTime: 50 * time.Millisecond,
	}
	setVnpuInfoToCache(ctx, group, n, nil)
}
