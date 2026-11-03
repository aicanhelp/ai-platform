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
	"math/rand"
	"time"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

// HasPDLeader reports whether any endpoint carries PD group metadata.
func HasPDLeader(pods []fwkscheduling.Endpoint) bool {
	for _, pod := range pods {
		if _, ok := GetDirect(pod); ok {
			return true
		}
	}
	return false
}

// IsScoredPDMode reports whether the scored candidates are PD-group leaders.
func IsScoredPDMode(scoredPods []*fwkscheduling.ScoredEndpoint) bool {
	for _, scoredPod := range scoredPods {
		if scoredPod == nil || scoredPod.Endpoint == nil {
			continue
		}
		if _, ok := GetDirect(scoredPod.Endpoint); ok {
			return true
		}
	}
	return false
}

// PreparePickerCandidates returns scored candidates after PD leader preparation
// when the candidate set represents disaggregated routing mode.
func PreparePickerCandidates(
	scoredPods []*fwkscheduling.ScoredEndpoint,
	rng *rand.Rand,
) []*fwkscheduling.ScoredEndpoint {
	if !IsScoredPDMode(scoredPods) {
		return scoredPods
	}
	return PreparePDCandidates(scoredPods, rng)
}

// PreparePDCandidates filters valid PD leaders and ensures selected members exist.
func PreparePDCandidates(scoredPods []*fwkscheduling.ScoredEndpoint, rng *rand.Rand) []*fwkscheduling.ScoredEndpoint {
	prepared := make([]*fwkscheduling.ScoredEndpoint, 0, len(scoredPods))
	resolvedRNG := ensurePDCandidateRNG(rng)

	for _, scoredPod := range scoredPods {
		if scoredPod == nil || scoredPod.Endpoint == nil {
			continue
		}

		info, ok := GetDirect(scoredPod.Endpoint)
		if !ok || Validate(info) != nil {
			continue
		}

		if pdSelectedMembersReady(info) {
			prepared = append(prepared, scoredPod)
			continue
		}

		updated, ok := fillRandomPDSelections(info, resolvedRNG)
		if !ok {
			continue
		}
		SetOwned(scoredPod.Endpoint, updated)
		prepared = append(prepared, scoredPod)
	}

	return prepared
}

func pdSelectedMembersReady(info *Info) bool {
	if info == nil {
		return false
	}
	return selectedScoredEndpointReady(info.SelectedPrefillPod) &&
		selectedScoredEndpointReady(info.SelectedDecodePod)
}

func fillRandomPDSelections(info *Info, rng *rand.Rand) (*Info, bool) {
	updated := Clone(info)
	if updated == nil {
		return nil, false
	}

	if !selectedScoredEndpointReady(updated.SelectedPrefillPod) {
		selected, ok := randomScoredEndpoint(updated.PrefillPods, rng)
		if !ok {
			return nil, false
		}
		updated.SelectedPrefillPod = selected
	}
	if !selectedScoredEndpointReady(updated.SelectedDecodePod) {
		selected, ok := randomScoredEndpoint(updated.DecodePods, rng)
		if !ok {
			return nil, false
		}
		updated.SelectedDecodePod = selected
	}

	return updated, true
}

func selectedScoredEndpointReady(endpoint *fwkscheduling.ScoredEndpoint) bool {
	return endpoint != nil && endpoint.Endpoint != nil
}

func randomScoredEndpoint(pods []fwkscheduling.ScoredEndpoint, rng *rand.Rand) (*fwkscheduling.ScoredEndpoint, bool) {
	indices := make([]int, 0, len(pods))
	for i := range pods {
		if pods[i].Endpoint != nil {
			indices = append(indices, i)
		}
	}
	if len(indices) == 0 {
		return nil, false
	}

	resolvedRNG := ensurePDCandidateRNG(rng)
	selected := pods[indices[resolvedRNG.Intn(len(indices))]]
	return &selected, true
}

func ensurePDCandidateRNG(rng *rand.Rand) *rand.Rand {
	if rng != nil {
		return rng
	}
	return rand.New(rand.NewSource(time.Now().UnixNano()))
}
