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
	"context"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"huawei.com/vxpu-device-plugin/pkg/log"
	"huawei.com/xpu-exporter/collector"
	"huawei.com/xpu-exporter/common/cache"
	"huawei.com/xpu-exporter/common/client"
)

const (
	CollectorName       = "npu"
	vnpuInfoCacheKey    = "xpu-exporter-vnpu-info"
	updateCachePattern  = "update cache,key is %s"
	tickerFailedPattern = "%s ticker failed, task shutdown"
)

type npuCollectorService struct {
	serviceName string
	collector   npuCollector
}

func New(name string) collector.ICollectorService {
	return &npuCollectorService{serviceName: name}
}

func (s *npuCollectorService) GetName() string {
	return s.serviceName
}

func (s *npuCollectorService) CreateCollector(cacheTime time.Duration, updateTime time.Duration) prometheus.Collector {
	s.collector = npuCollector{
		cache:      cache.New(cacheSize),
		cacheTime:  cacheTime,
		updateTime: updateTime,
	}
	return &s.collector
}

func (s *npuCollectorService) Start(ctx context.Context, fn context.CancelFunc) {
	group := &sync.WaitGroup{}

	vnpuInfoCollect(ctx, group, &s.collector, fn)

	group.Wait()
}

func vnpuInfoCollect(ctx context.Context, group *sync.WaitGroup, n *npuCollector, fn context.CancelFunc) {
	group.Add(1)
	go func() {
		setVnpuInfoToCache(ctx, group, n, fn)
	}()
}

func setVnpuInfoToCache(ctx context.Context, group *sync.WaitGroup, n *npuCollector, fn context.CancelFunc) {
	defer group.Done()
	ticker := time.NewTicker(n.updateTime)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			log.Infoln("received the stop signal,STOP vnpu info collect")
			return
		default:
			vnpuInfo, err := client.GetAllVxpuInfo()
			if err != nil {
				log.Errorf("get vnpuInfo error: %v", err)
				fn()
				return
			}
			if err = n.cache.Set(vnpuInfoCacheKey, vnpuInfo, n.cacheTime); err != nil {
				log.Errorln(err)
			} else {
				log.Infof(updateCachePattern, vnpuInfoCacheKey)
			}
			if _, ok := <-ticker.C; !ok {
				log.Errorf(tickerFailedPattern, vnpuInfoCacheKey)
				return
			}
		}
	}
}
