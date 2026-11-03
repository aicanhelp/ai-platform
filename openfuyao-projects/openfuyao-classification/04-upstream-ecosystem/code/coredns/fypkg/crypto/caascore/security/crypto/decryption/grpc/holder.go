// Copyright(c) Huawei Technologies Co., Ltd. 2025-2025. All right reserved.

package grpc

import (
	"fmt"
	"sync"
	"time"

	"github.com/coredns/coredns/fypkg/crypto/caascore/security/crypto/decryption/api"
)

// 清理轮询周期必须大于grpc请求超时时间
var defaultCleanPeriod = 5 * time.Minute

type cachedService struct {
	mu       sync.Mutex
	services map[string]*serviceWrap
}

type serviceWrap struct {
	cacheKey  string
	service   api.Service
	timestamp time.Time
	period    time.Duration
}

var Holder = &cachedService{services: make(map[string]*serviceWrap, 1)}

func (c *cachedService) GetGRPCService(endpoint string, callTimeout time.Duration) (api.Service, error) {
	// Ensure we only create signal grpc for the given endpoint
	c.mu.Lock()
	defer c.mu.Unlock()

	// See if we already have custom grpc for this endpoint
	cacheKey := key(endpoint, callTimeout)
	if s, ok := c.services[cacheKey]; ok {
		s.update()
		return s.service, nil
	}

	service, err := createGRPCService(endpoint, callTimeout)
	if err != nil {
		return nil, err
	}
	c.register(cacheKey, service, callTimeout)
	return service, nil
}

func (c *cachedService) tryClean(wrap *serviceWrap) bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if wrap.isTimeout() {
		delete(c.services, wrap.cacheKey)
		wrap.service.Release()
		return true
	}
	return false
}

func (c *cachedService) register(cacheKey string, service api.Service, apiTimeout time.Duration) {
	period := apiTimeout + time.Second
	cached := &serviceWrap{
		cacheKey:  cacheKey,
		service:   service,
		period:    period,
		timestamp: time.Now(),
	}
	c.services[cacheKey] = cached
	go cached.timeToClean(c)
}

func (c *serviceWrap) timeToClean(cache *cachedService) {
	for {
		timer := time.NewTimer(c.period)
		select {
		case <-timer.C:
			timer.Stop()
			if cache.tryClean(c) {
				return
			}
		}
	}
}

func (c *serviceWrap) isTimeout() bool {
	return time.Since(c.timestamp) > c.period
}

func (c *serviceWrap) update() {
	c.timestamp = time.Now()
}

func key(endpoint string, callTimeout time.Duration) string {
	return fmt.Sprintf("%s:%v", endpoint, callTimeout)
}
