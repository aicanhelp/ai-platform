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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"

	attrnpu "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	npusrc "hermes-router/pkg/epp/framework/plugins/datalayer/source/npu"
	"hermes-router/pkg/epp/internal/utils"
)

// NPUExporterExtractorType is the registered plugin type identifier for the
// NPU extractor.
const NPUExporterExtractorType = "npu-exporter-extractor"

var _ fwkdl.Extractor = (*Extractor)(nil)

// Extractor converts a npusrc.Pod snapshot into an *State attribute on
// the endpoint. Matched=false snapshots are no-ops so the previously stored
// value remains visible to downstream consumers.
type Extractor struct {
	typedName    fwkplugin.TypedName
	attributeKey string
	staleAfter   time.Duration
	nowFn        func() time.Time
}

// Parameters is the user-facing extractor configuration.
type Parameters struct {
	// AttributeKey stores the AttributeKey value.
	AttributeKey string `json:"attributeKey"`
	// StaleAfter stores the StaleAfter value.
	StaleAfter utils.Duration `json:"staleAfter"`
}

// New constructs an Extractor. nowFn is exposed for test injection.
func New(name, attributeKey string, staleAfter time.Duration) *Extractor {
	return &Extractor{
		typedName:    fwkplugin.TypedName{Type: NPUExporterExtractorType, Name: name},
		attributeKey: attributeKey,
		staleAfter:   staleAfter,
		nowFn:        time.Now,
	}
}

// Factory is the registered plugin factory function for NPUExporterExtractorType.
func Factory(name string, raw json.RawMessage, _ fwkplugin.Handle) (fwkplugin.Plugin, error) {
	params := Parameters{AttributeKey: attrnpu.DefaultStateAttributeKey}
	if len(raw) != 0 {
		decoder := json.NewDecoder(bytes.NewReader(raw))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&params); err != nil {
			return nil, fmt.Errorf("npu extractor %q: invalid parameters: %w", name, err)
		}
	}
	if params.AttributeKey == "" {
		return nil, fmt.Errorf("npu extractor %q: attributeKey must not be empty", name)
	}
	if params.StaleAfter.Std() <= 0 {
		return nil, fmt.Errorf("npu extractor %q: staleAfter must be positive", name)
	}
	return New(name, params.AttributeKey, params.StaleAfter.Std()), nil
}

// TypedName returns the framework plugin type and instance name.
func (e *Extractor) TypedName() fwkplugin.TypedName {
	return e.typedName
}

// ExpectedInputType returns the data type accepted by the extractor.
func (e *Extractor) ExpectedInputType() reflect.Type {
	return npusrc.PodType()
}

// Extract publishes data-layer attributes for an endpoint.
func (e *Extractor) Extract(ctx context.Context, data any, ep fwkdl.Endpoint) error {
	logger := log.FromContext(ctx).WithName("npu-extractor")
	pod, ok := data.(npusrc.Pod)
	if !ok {
		return fmt.Errorf("npu extractor %q expected %s input, got %T", e.typedName.Name, npusrc.PodType(), data)
	}
	if !pod.Matched {
		logger.V(logutil.DEBUG).Info(
			"NPU extractor received unmatched pod snapshot",
			"namespace", pod.Namespace,
			"pod", pod.PodName,
			"deviceCount", pod.DeviceCount,
			"observedAt", pod.ObservedAt,
		)
		// No fresh observation this cycle. Re-evaluate the previously published
		// state so its DataStatus ages to Stale once it falls outside the
		// staleAfter window; consumers reading the attribute therefore never
		// see an indefinitely-Present value backed by an old ObservationTime.
		e.ageStoredState(ep)
		return nil
	}
	state := e.stateFromPod(pod)
	ep.GetAttributes().Put(e.attributeKey, state)
	logger.V(logutil.DEBUG).Info(
		"NPU extractor published aggregated state",
		"namespace", pod.Namespace,
		"pod", pod.PodName,
		"deviceCount", state.DeviceCount,
		"dataStatus", state.DataStatus,
		"aiCoreUtilizationMax", state.AICoreUtilizationMax,
		"hbmUtilizationMax", state.HBMUtilizationMax,
		"hbmBandwidthUtilizationMax", state.HBMBandwidthUtilizationMax,
		"observedAt", state.ObservationTime,
	)
	return nil
}

// ageStoredState transitions a previously stored *State to DataStale
// when its ObservationTime is now older than staleAfter.
func (e *Extractor) ageStoredState(ep fwkdl.Endpoint) {
	value, ok := ep.GetAttributes().Get(e.attributeKey)
	if !ok {
		return
	}
	state, ok := value.(*attrnpu.State)
	if !ok || state == nil {
		return
	}
	if state.DataStatus == attrnpu.DataStale {
		return
	}
	_, status := e.normalizeObservation(state.ObservationTime)
	if status == attrnpu.DataPresent {
		return
	}
	aged := *state
	aged.DataStatus = status
	ep.GetAttributes().Put(e.attributeKey, &aged)
}

func (e *Extractor) stateFromPod(pod npusrc.Pod) *attrnpu.State {
	observed, status := e.normalizeObservation(pod.ObservedAt)
	devices := convertDevices(pod.Devices)
	return &attrnpu.State{
		ObservationTime:            observed,
		DataStatus:                 status,
		DeviceCount:                len(devices),
		AICoreUtilizationAvg:       pod.AICoreUtilizationAvg,
		AICoreUtilizationMax:       pod.AICoreUtilizationMax,
		HBMUtilizationAvg:          pod.HBMUtilizationAvg,
		HBMUtilizationMax:          pod.HBMUtilizationMax,
		HBMBandwidthUtilizationAvg: pod.HBMBandwidthUtilizationAvg,
		HBMBandwidthUtilizationMax: pod.HBMBandwidthUtilizationMax,
		Devices:                    devices,
	}
}

// normalizeObservation returns the clamped observation time callers should
// persist, plus the status to publish for that observation.
func (e *Extractor) normalizeObservation(observationTime time.Time) (time.Time, attrnpu.DataStatus) {
	now := e.nowFn()
	if observationTime.After(now) {
		observationTime = now
	}
	if now.Sub(observationTime) > e.staleAfter {
		return observationTime, attrnpu.DataStale
	}
	return observationTime, attrnpu.DataPresent
}

func convertDevices(devices []npusrc.Device) []attrnpu.DeviceState {
	out := make([]attrnpu.DeviceState, 0, len(devices))
	for _, device := range devices {
		if device.ID == "" {
			continue
		}
		out = append(out, attrnpu.DeviceState{
			Target:                  device.Target,
			DeviceID:                device.ID,
			AICoreUtilization:       device.AICoreUtilization,
			HBMUtilization:          device.HBMUtilization,
			HBMBandwidthUtilization: device.HBMBandwidthUtilization,
		})
	}
	return out
}
