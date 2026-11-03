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

// Package common provides common utilities and structures for plugins.
package common

import (
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

const (
	// PDGroupsCycleStateKey is the key name for the PDGroups struct in CycleState.
	PDGroupsCycleStateKey plugins.StateKey = "filter-by-pd-label/pd-groups"
)

// PDGroup represents a Pod group in PD (Prefill-Decode) architecture.
type PDGroup struct {
	ID          string
	PrefillPods []types.ScoredPod
	DecodePods  []types.ScoredPod
	// LeaderPod.Score is the score of the entire PDGroup.
	LeaderPod          types.ScoredPod
	SelectedPrefillPod types.ScoredPod
	SelectedDecodePod  types.ScoredPod
}

// PDGroupList represents a list of multiple PD groups.
type PDGroupList struct {
	Groups []PDGroup
}

// Clone implements the plugins.StateData interface, creating a deep copy of PDGroupList.
func (p *PDGroupList) Clone() plugins.StateData {
	if p == nil {
		return &PDGroupList{}
	}

	cloned := make([]PDGroup, len(p.Groups))
	for i, group := range p.Groups {
		clonedPrefillPods := make([]types.ScoredPod, len(group.PrefillPods))
		for j, pod := range group.PrefillPods {
			clonedPrefillPods[j] = types.ScoredPod{
				Pod:   pod.Pod,
				Score: pod.Score,
			}
		}
		clonedDecodePods := make([]types.ScoredPod, len(group.DecodePods))
		for j, pod := range group.DecodePods {
			clonedDecodePods[j] = types.ScoredPod{
				Pod:   pod.Pod,
				Score: pod.Score,
			}
		}
		cloned[i] = PDGroup{
			ID:          group.ID,
			PrefillPods: clonedPrefillPods,
			DecodePods:  clonedDecodePods,
			LeaderPod: types.ScoredPod{
				Pod:   group.LeaderPod.Pod,
				Score: group.LeaderPod.Score,
			},
			SelectedPrefillPod: types.ScoredPod{
				Pod:   group.SelectedPrefillPod.Pod,
				Score: group.SelectedPrefillPod.Score,
			},
			SelectedDecodePod: types.ScoredPod{
				Pod:   group.SelectedDecodePod.Pod,
				Score: group.SelectedDecodePod.Score,
			},
		}
	}
	return &PDGroupList{Groups: cloned}
}
