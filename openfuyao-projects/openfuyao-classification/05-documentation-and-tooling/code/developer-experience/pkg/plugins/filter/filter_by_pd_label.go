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

// Package filter provides various filter plugins.
package filter

import (
	"context"
	"encoding/json"
	"sort"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
)

const (
	FilterByPDLabelType = "filter-by-pd-label"
)

// FilterByPDLabelParameters is the plugin parameters that need to be specified by users in EPP configuration files.
type FilterByPDLabelParameters struct {
	PDLabelName string `json:"pdLabelName"`
	// PDGroupLabelName is the name of the PD group that the pod belongs to.
	PDGroupLabelName string `json:"pdGroupLabelName"`
	PrefillValue     string `json:"prefillValue"`
	DecodeValue      string `json:"decodeValue"`
	LeaderValue      string `json:"leaderValue"`
}

var _ framework.Filter = &FilterByPDLabel{}

// FilterByPDLabel is the plugin struct that implements the framework.Filter interface.
type FilterByPDLabel struct {
	typedName      plugins.TypedName
	roleLabelName  string
	groupLabelName string
	prefillValue   string
	decodeValue    string
	leaderValue    string
}

// FilterByPDLabelFactory is a factory function that creates a plugin instance based on parameters specified by users in EPP configuration files.
func FilterByPDLabelFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	return NewFilterByPDLabel(handle.Context(), name, rawParameters), nil
}

// NewFilterByPDLabel creates a plugin instance based on parameters specified by users in EPP configuration files.
func NewFilterByPDLabel(ctx context.Context, name string, rawParameters json.RawMessage) *FilterByPDLabel {
	params := FilterByPDLabelParameters{}
	if err := json.Unmarshal(rawParameters, &params); err != nil {
		logger := log.FromContext(ctx).WithName("FilterByPDLabel.NewFilterByPDLabel")
		logger.V(logging.DEFAULT).Error(err, "failed to unmarshal parameters of the '%s' filter", FilterByPDLabelType)
		return nil
	}

	return &FilterByPDLabel{
		typedName:      plugins.TypedName{Type: FilterByPDLabelType, Name: name},
		roleLabelName:  params.PDLabelName,
		groupLabelName: params.PDGroupLabelName,
		prefillValue:   params.PrefillValue,
		decodeValue:    params.DecodeValue,
		leaderValue:    params.LeaderValue,
	}
}

// TypedName returns the plugin's type name.
func (f *FilterByPDLabel) TypedName() plugins.TypedName {
	return f.typedName
}

// WithName sets the plugin's name.
func (f *FilterByPDLabel) WithName(name string) *FilterByPDLabel {
	return &FilterByPDLabel{
		typedName:      plugins.TypedName{Type: FilterByPDLabelType, Name: name},
		roleLabelName:  f.roleLabelName,
		groupLabelName: f.groupLabelName,
		prefillValue:   f.prefillValue,
		decodeValue:    f.decodeValue,
		leaderValue:    f.leaderValue,
	}
}

// Filter groups pods by PD group labels and saves them in state.
func (f *FilterByPDLabel) Filter(ctx context.Context, state *types.CycleState, request *types.LLMRequest, pods []types.Pod) []types.Pod {
	groupMap := f.groupPodsByPDLabels(ctx, pods)

	// Delete PD group state if no groups found, to clean up residual data from multiple scheduling attempts of the same request.
	if len(groupMap) == 0 {
		state.Delete(common.PDGroupsCycleStateKey)
		return pods
	}

	// Sort PD group names alphabetically to avoid inconsistent order when reading from state due to map's unordered nature.
	groupIDs := make([]string, 0, len(groupMap))
	for id := range groupMap {
		groupIDs = append(groupIDs, id)
	}
	sort.Strings(groupIDs)

	groups := make([]common.PDGroup, 0, len(groupMap))
	for _, id := range groupIDs {
		g := groupMap[id]
		groups = append(groups, common.PDGroup{
			ID:          id,
			PrefillPods: append([]types.ScoredPod(nil), g.PrefillPods...),
			DecodePods:  append([]types.ScoredPod(nil), g.DecodePods...),
			LeaderPod:   g.LeaderPod,
		})
	}

	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: groups})

	return pods
}

// groupPodsByPDLabels groups pods by PD labels and returns a map of group ID to PDGroup.
func (f *FilterByPDLabel) groupPodsByPDLabels(ctx context.Context, pods []types.Pod) map[string]*common.PDGroup {
	logger := log.FromContext(ctx)
	groupMap := make(map[string]*common.PDGroup)

	for _, pod := range pods {
		backendPod := pod.GetPod()

		roleValue, roleExists := backendPod.Labels[f.roleLabelName]
		groupID, groupExists := backendPod.Labels[f.groupLabelName]

		if !roleExists || !groupExists || groupID == "" {
			logger.V(logging.DEFAULT).Info("pod missing pd grouping labels, skip",
				"pod", backendPod.PodName, "roleExists", roleExists, "groupExists", groupExists)
			continue
		}

		group := groupMap[groupID]
		if group == nil {
			group = &common.PDGroup{}
			groupMap[groupID] = group
		}

		switch roleValue {
		case f.prefillValue:
			group.PrefillPods = append(group.PrefillPods, types.ScoredPod{
				Pod:   pod,
				Score: 0.0,
			})
		case f.decodeValue:
			group.DecodePods = append(group.DecodePods, types.ScoredPod{
				Pod:   pod,
				Score: 0.0,
			})
		case f.leaderValue:
			group.LeaderPod = types.ScoredPod{
				Pod:   pod,
				Score: 0.0,
			}
		default:
			logger.V(logging.DEFAULT).Info("pod role does not match any pd role value, skip",
				"pod", backendPod.PodName, "roleValue", roleValue)
		}
	}

	return groupMap
}
