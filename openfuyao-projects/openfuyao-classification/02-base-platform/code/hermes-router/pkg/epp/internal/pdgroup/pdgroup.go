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

// Package pdgroup stores and prepares prefill/decode group metadata.
package pdgroup

import (
	"fmt"

	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
)

// AttributeKey is an exported constant used by Hermes Router.
const AttributeKey = "pd-group/info"

// Info carries the leader's prefill/decode member candidates and selections.
type Info struct {
	// GroupID stores the GroupID value.
	GroupID string
	// PrefillPods stores the PrefillPods value.
	PrefillPods []fwkscheduling.ScoredEndpoint
	// DecodePods stores the DecodePods value.
	DecodePods []fwkscheduling.ScoredEndpoint
	// SelectedPrefillPod stores the SelectedPrefillPod value.
	SelectedPrefillPod *fwkscheduling.ScoredEndpoint
	// SelectedDecodePod stores the SelectedDecodePod value.
	SelectedDecodePod *fwkscheduling.ScoredEndpoint
}

// MemberInspection summarizes missing prediction and metric inputs for a group.
type MemberInspection struct {
	// MissingPrefillPredictions stores the MissingPrefillPredictions value.
	MissingPrefillPredictions int
	// MissingDecodePredictions stores the MissingDecodePredictions value.
	MissingDecodePredictions int
	// MissingPrefillMetrics stores the MissingPrefillMetrics value.
	MissingPrefillMetrics int
	// MissingDecodeMetrics stores the MissingDecodeMetrics value.
	MissingDecodeMetrics int
}

// Clone returns an independent copy of the receiver.
func (info *Info) Clone() fwkdl.Cloneable {
	if info == nil {
		return (*Info)(nil)
	}
	return Clone(info)
}

// Validate checks that the value is internally consistent.
func Validate(info *Info) error {
	if info == nil {
		return fmt.Errorf("PD group info cannot be nil")
	}
	if info.GroupID == "" {
		return fmt.Errorf("PD group info groupID cannot be empty")
	}
	if len(info.PrefillPods) == 0 {
		return fmt.Errorf("PD group info %s: prefill pods cannot be empty", info.GroupID)
	}
	if len(info.DecodePods) == 0 {
		return fmt.Errorf("PD group info %s: decode pods cannot be empty", info.GroupID)
	}
	return nil
}

// InspectMembers provides the exported InspectMembers operation.
func InspectMembers(
	info *Info,
	hasPrefillPrediction func(fwkscheduling.Endpoint) bool,
	hasDecodePrediction func(fwkscheduling.Endpoint) bool,
	hasMetrics func(fwkscheduling.Endpoint) bool,
) MemberInspection {
	inspection := MemberInspection{}
	if hasPrefillPrediction == nil {
		hasPrefillPrediction = func(fwkscheduling.Endpoint) bool { return false }
	}
	if hasDecodePrediction == nil {
		hasDecodePrediction = func(fwkscheduling.Endpoint) bool { return false }
	}
	if hasMetrics == nil {
		hasMetrics = func(fwkscheduling.Endpoint) bool { return false }
	}
	if info == nil {
		return inspection
	}
	for _, prefill := range info.PrefillPods {
		if !hasPrefillPrediction(prefill.Endpoint) {
			inspection.MissingPrefillPredictions++
		}
		if !hasMetrics(prefill.Endpoint) {
			inspection.MissingPrefillMetrics++
		}
	}
	for _, decode := range info.DecodePods {
		if !hasDecodePrediction(decode.Endpoint) {
			inspection.MissingDecodePredictions++
		}
		if !hasMetrics(decode.Endpoint) {
			inspection.MissingDecodeMetrics++
		}
	}
	return inspection
}

// PredictionsAvailable provides the exported PredictionsAvailable operation.
func (inspection MemberInspection) PredictionsAvailable() bool {
	return inspection.MissingPrefillPredictions == 0 && inspection.MissingDecodePredictions == 0
}

// HasMissingPredictions provides the exported HasMissingPredictions operation.
func (inspection MemberInspection) HasMissingPredictions() bool {
	return inspection.MissingPrefillPredictions > 0 || inspection.MissingDecodePredictions > 0
}

// HasMissingMetrics provides the exported HasMissingMetrics operation.
func (inspection MemberInspection) HasMissingMetrics() bool {
	return inspection.MissingPrefillMetrics > 0 || inspection.MissingDecodeMetrics > 0
}

// Clone returns an independent copy of the receiver.
func Clone(info *Info) *Info {
	if info == nil {
		return nil
	}
	cloned := &Info{
		GroupID:     info.GroupID,
		PrefillPods: cloneScoredEndpoints(info.PrefillPods),
		DecodePods:  cloneScoredEndpoints(info.DecodePods),
	}
	if info.SelectedPrefillPod != nil {
		selected := endpointmeta.CloneScoredEndpoint(*info.SelectedPrefillPod)
		cloned.SelectedPrefillPod = &selected
	}
	if info.SelectedDecodePod != nil {
		selected := endpointmeta.CloneScoredEndpoint(*info.SelectedDecodePod)
		cloned.SelectedDecodePod = &selected
	}
	return cloned
}

// Set stores the value on an endpoint.
func Set(endpoint fwkscheduling.Endpoint, info *Info) {
	if endpoint == nil {
		return
	}
	endpoint.Put(AttributeKey, Clone(info))
}

// SetOwned stores info without cloning it first.
// Callers must pass an owned instance that will not be mutated afterwards.
func SetOwned(endpoint fwkscheduling.Endpoint, info *Info) {
	if endpoint == nil {
		return
	}
	endpoint.Put(AttributeKey, info)
}

// Get retrieves the value from an endpoint.
func Get(endpoint fwkscheduling.Endpoint) (*Info, bool) {
	info, ok := GetDirect(endpoint)
	if !ok {
		return nil, false
	}
	return Clone(info), true
}

// GetDirect returns the Info value from the endpoint without applying
// the extra repository-level Clone used by Get. The framework's Endpoint.Get
// may already have cloned the stored attribute.
// Callers must treat the returned object as read-only unless they write it
// back with SetOwned.
func GetDirect(endpoint fwkscheduling.Endpoint) (*Info, bool) {
	if endpoint == nil {
		return nil, false
	}
	value, ok := endpoint.Get(AttributeKey)
	if !ok {
		return nil, false
	}
	info, ok := value.(*Info)
	if !ok || info == nil {
		return nil, false
	}
	return info, true
}

// ContainsMember reports whether endpoint belongs to the group's original
// prefill/decode member lists. It does not check selected pods separately.
func ContainsMember(info *Info, endpoint fwkscheduling.Endpoint) bool {
	if info == nil || endpoint == nil {
		return false
	}
	for _, member := range info.PrefillPods {
		if fwkscheduling.EndpointComparer(member.Endpoint, endpoint) {
			return true
		}
	}
	for _, member := range info.DecodePods {
		if fwkscheduling.EndpointComparer(member.Endpoint, endpoint) {
			return true
		}
	}
	return false
}

func cloneScoredEndpoints(pods []fwkscheduling.ScoredEndpoint) []fwkscheduling.ScoredEndpoint {
	if len(pods) == 0 {
		return nil
	}
	cloned := make([]fwkscheduling.ScoredEndpoint, len(pods))
	for i, pod := range pods {
		cloned[i] = endpointmeta.CloneScoredEndpoint(pod)
	}
	return cloned
}
