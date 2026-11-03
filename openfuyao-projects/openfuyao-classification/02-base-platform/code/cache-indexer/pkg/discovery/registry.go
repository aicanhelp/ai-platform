/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package discovery

import (
	"sync"

	"github.com/go-logr/logr"

	"gitcode.com/openFuyao/cache-indexer/pkg/apis"
)

// registry holds the latest snapshot plus O(1) lookup maps.
// byServerIP maps vLLM pods; serverToTransport maps resolved Mooncake endpoints.
type registry struct {
	mu                sync.RWMutex
	log               logr.Logger
	snapshot          *Snapshot
	byServerIP        map[string]*VLLMPodEndpoint
	serverToTransport map[string]apis.TransportEndpoint
}

// NewRegistry builds the read-side discovery registry used by the query plane.
func NewRegistry(log logr.Logger) Registry {
	return &registry{
		log:               log,
		byServerIP:        map[string]*VLLMPodEndpoint{},
		serverToTransport: map[string]apis.TransportEndpoint{},
	}
}

func (r *registry) Update(snapshot *Snapshot) error {
	if snapshot == nil {
		return nil
	}
	byIP := make(map[string]*VLLMPodEndpoint, len(snapshot.VLLMPods))
	byEP := make(map[string]apis.TransportEndpoint)
	for ip, ep := range snapshot.VLLMPods {
		byIP[ip] = ep
		if ep.MooncakeClient.Resolved {
			byEP[ip] = ep.MooncakeClient.TransportEndpoint
		}
	}

	r.mu.Lock()
	r.snapshot = snapshot
	r.byServerIP = byIP
	r.serverToTransport = byEP
	r.mu.Unlock()
	return nil
}

func (r *registry) Snapshot() *Snapshot {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.snapshot
}

func (r *registry) ResolveTransportEndpoint(serverIP string) (apis.TransportEndpoint, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ep, ok := r.serverToTransport[serverIP]
	return ep, ok
}

func (r *registry) ResolveVLLMPod(serverIP string) (*VLLMPodEndpoint, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ep, ok := r.byServerIP[serverIP]
	return ep, ok
}
