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

// Package npu provides NPU data-layer source, extractor, and attribute support.
package npu

import (
	"time"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

// DefaultStateAttributeKey is the endpoint attribute key under which the
// extractor publishes the current NPU state for a pod.
const DefaultStateAttributeKey = "NPUStateKey"

// DataStatus describes whether the published state is fresh enough to act on.
type DataStatus string

const (
	// DataPresent is an exported constant used by Hermes Router.
	DataPresent DataStatus = "Present"
	// DataStale is an exported constant used by Hermes Router.
	DataStale DataStatus = "Stale"
)

// State is the per-endpoint, pod-scoped NPU snapshot stored on
// Endpoint.GetAttributes(). All utilization values are percentages (0-100).
type State struct {
	// ObservationTime stores the ObservationTime value.
	ObservationTime time.Time
	// DataStatus stores the DataStatus value.
	DataStatus DataStatus
	// DeviceCount stores the DeviceCount value.
	DeviceCount int
	// AICoreUtilizationAvg stores the AICoreUtilizationAvg value.
	AICoreUtilizationAvg float64
	// AICoreUtilizationMax stores the AICoreUtilizationMax value.
	AICoreUtilizationMax float64
	// HBMUtilizationAvg stores the HBMUtilizationAvg value.
	HBMUtilizationAvg float64
	// HBMUtilizationMax stores the HBMUtilizationMax value.
	HBMUtilizationMax float64
	// HBMBandwidthUtilizationAvg stores the HBMBandwidthUtilizationAvg value.
	HBMBandwidthUtilizationAvg float64
	// HBMBandwidthUtilizationMax stores the HBMBandwidthUtilizationMax value.
	HBMBandwidthUtilizationMax float64
	// Devices stores the Devices value.
	Devices []DeviceState
}

// DeviceState mirrors a single device's per-poll utilization snapshot.
// (Target, DeviceID) is the device identity: DeviceID is exporter-local and
// the same value can legitimately appear under different Targets when a pod
// spans multiple nodes.
type DeviceState struct {
	// Target stores the Target value.
	Target string
	// DeviceID stores the DeviceID value.
	DeviceID string
	// AICoreUtilization stores the AICoreUtilization value.
	AICoreUtilization float64
	// HBMUtilization stores the HBMUtilization value.
	HBMUtilization float64
	// HBMBandwidthUtilization stores the HBMBandwidthUtilization value.
	HBMBandwidthUtilization float64
}

// Clone deep-copies the Devices slice so callers cannot mutate stored state
// through the cloned pointer.
func (s *State) Clone() fwkdl.Cloneable {
	if s == nil {
		return (*State)(nil)
	}
	cloned := *s
	if s.Devices != nil {
		cloned.Devices = make([]DeviceState, len(s.Devices))
		copy(cloned.Devices, s.Devices)
	}
	return &cloned
}
