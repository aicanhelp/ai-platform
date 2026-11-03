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

package inflight

import (
	"maps"
	"time"
)

// SelectedRoute describes SelectedRoute data used by Hermes Router.
type SelectedRoute struct {
	// Leader stores the Leader value.
	Leader *RouteEndpointSnapshot `json:"leader,omitempty"`
	// Prefill stores the Prefill value.
	Prefill *RouteEndpointSnapshot `json:"prefill,omitempty"`
	// Decode stores the Decode value.
	Decode *RouteEndpointSnapshot `json:"decode,omitempty"`
}

// Clone returns an independent copy of the receiver.
func (r *SelectedRoute) Clone() *SelectedRoute {
	if r == nil {
		return nil
	}
	return &SelectedRoute{
		Leader:  r.Leader.Clone(),
		Prefill: r.Prefill.Clone(),
		Decode:  r.Decode.Clone(),
	}
}

// RouteEndpointSnapshot describes RouteEndpointSnapshot data used by Hermes Router.
type RouteEndpointSnapshot struct {
	// ID stores the ID value.
	ID string `json:"id,omitempty"`
	// Metrics stores the Metrics value.
	Metrics *EndpointMetricsSnapshot `json:"metrics,omitempty"`
	// Inflight stores the Inflight value.
	Inflight *Load `json:"inflight,omitempty"`
	// NPU stores the NPU value.
	NPU *NPUStateSnapshot `json:"npu,omitempty"`
}

// Clone returns an independent copy of the receiver.
func (s *RouteEndpointSnapshot) Clone() *RouteEndpointSnapshot {
	if s == nil {
		return nil
	}
	cloned := &RouteEndpointSnapshot{
		ID:      s.ID,
		Metrics: s.Metrics.Clone(),
		NPU:     s.NPU.Clone(),
	}
	if s.Inflight != nil {
		load := *s.Inflight
		cloned.Inflight = &load
	}
	return cloned
}

// EndpointMetricsSnapshot describes EndpointMetricsSnapshot data used by Hermes Router.
type EndpointMetricsSnapshot struct {
	// ActiveModels stores the ActiveModels value.
	ActiveModels map[string]int `json:"activeModels,omitempty"`
	// WaitingModels stores the WaitingModels value.
	WaitingModels map[string]int `json:"waitingModels,omitempty"`
	// MaxActiveModels stores the MaxActiveModels value.
	MaxActiveModels int `json:"maxActiveModels,omitempty"`
	// RunningRequestsSize stores the RunningRequestsSize value.
	RunningRequestsSize int `json:"runningRequestsSize,omitempty"`
	// WaitingQueueSize stores the WaitingQueueSize value.
	WaitingQueueSize int `json:"waitingQueueSize,omitempty"`
	// KVCacheUsagePercent stores the KVCacheUsagePercent value.
	KVCacheUsagePercent float64 `json:"kvCacheUsagePercent,omitempty"`
	// KVCacheMaxTokenCapacity stores the KVCacheMaxTokenCapacity value.
	KVCacheMaxTokenCapacity int `json:"kvCacheMaxTokenCapacity,omitempty"`
	// CacheBlockSize stores the CacheBlockSize value.
	CacheBlockSize int `json:"cacheBlockSize,omitempty"`
	// CacheNumBlocks stores the CacheNumBlocks value.
	CacheNumBlocks int `json:"cacheNumBlocks,omitempty"`
	// UpdateTime stores the UpdateTime value.
	UpdateTime time.Time `json:"updateTime,omitempty"`
}

// Clone returns an independent copy of the receiver.
func (s *EndpointMetricsSnapshot) Clone() *EndpointMetricsSnapshot {
	if s == nil {
		return nil
	}
	cloned := *s
	cloned.ActiveModels = maps.Clone(s.ActiveModels)
	cloned.WaitingModels = maps.Clone(s.WaitingModels)
	return &cloned
}

// NPUStateSnapshot describes NPUStateSnapshot data used by Hermes Router.
type NPUStateSnapshot struct {
	// ObservationTime stores the ObservationTime value.
	ObservationTime time.Time `json:"observationTime,omitempty"`
	// DataStatus stores the DataStatus value.
	DataStatus string `json:"dataStatus,omitempty"`
	// DeviceCount stores the DeviceCount value.
	DeviceCount int `json:"deviceCount,omitempty"`
	// AICoreUtilizationAvg stores the AICoreUtilizationAvg value.
	AICoreUtilizationAvg float64 `json:"aiCoreUtilizationAvg,omitempty"`
	// AICoreUtilizationMax stores the AICoreUtilizationMax value.
	AICoreUtilizationMax float64 `json:"aiCoreUtilizationMax,omitempty"`
	// HBMUtilizationAvg stores the HBMUtilizationAvg value.
	HBMUtilizationAvg float64 `json:"hbmUtilizationAvg,omitempty"`
	// HBMUtilizationMax stores the HBMUtilizationMax value.
	HBMUtilizationMax float64 `json:"hbmUtilizationMax,omitempty"`
	// HBMBandwidthUtilizationAvg stores the HBMBandwidthUtilizationAvg value.
	HBMBandwidthUtilizationAvg float64 `json:"hbmBandwidthUtilizationAvg,omitempty"`
	// HBMBandwidthUtilizationMax stores the HBMBandwidthUtilizationMax value.
	HBMBandwidthUtilizationMax float64 `json:"hbmBandwidthUtilizationMax,omitempty"`
	// Devices stores the Devices value.
	Devices []NPUDeviceSnapshot `json:"devices,omitempty"`
}

// Clone returns an independent copy of the receiver.
func (s *NPUStateSnapshot) Clone() *NPUStateSnapshot {
	if s == nil {
		return nil
	}
	cloned := *s
	if s.Devices != nil {
		cloned.Devices = make([]NPUDeviceSnapshot, len(s.Devices))
		copy(cloned.Devices, s.Devices)
	}
	return &cloned
}

// NPUDeviceSnapshot describes NPUDeviceSnapshot data used by Hermes Router.
type NPUDeviceSnapshot struct {
	// Target stores the Target value.
	Target string `json:"target,omitempty"`
	// DeviceID stores the DeviceID value.
	DeviceID string `json:"deviceID,omitempty"`
	// AICoreUtilization stores the AICoreUtilization value.
	AICoreUtilization float64 `json:"aiCoreUtilization,omitempty"`
	// HBMUtilization stores the HBMUtilization value.
	HBMUtilization float64 `json:"hbmUtilization,omitempty"`
	// HBMBandwidthUtilization stores the HBMBandwidthUtilization value.
	HBMBandwidthUtilization float64 `json:"hbmBandwidthUtilization,omitempty"`
}

func ensureSelectedRoute(route *SelectedRoute, leaderID string) *SelectedRoute {
	cloned := route.Clone()
	if cloned == nil {
		cloned = &SelectedRoute{}
	}
	if cloned.Leader == nil {
		cloned.Leader = &RouteEndpointSnapshot{}
	}
	if cloned.Leader.ID == "" {
		cloned.Leader.ID = leaderID
	}
	return cloned
}
