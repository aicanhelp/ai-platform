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

// Package picker provides various picker plugins.
package picker

import (
	"context"
	"encoding/json"
	"errors"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
)

const (
	PickerPDKVCacheAwareType = "picker-pd-kv-cache-aware"
)

// PickerPDKVCacheAware is the plugin struct that implements the framework.Picker interface.
type PickerPDKVCacheAware struct {
	typedName plugins.TypedName
}

var _ framework.Picker = &PickerPDKVCacheAware{}

// PickerPDKVCacheAwareFactory is a factory function that creates a plugin instance based on parameters specified by users in EPP configuration files.
func PickerPDKVCacheAwareFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	return NewPickerPDKVCacheAware(name, handle.Context()), nil
}

// NewPickerPDKVCacheAware creates a plugin instance based on parameters specified by users in EPP configuration files.
func NewPickerPDKVCacheAware(name string, ctx context.Context) *PickerPDKVCacheAware {
	return &PickerPDKVCacheAware{
		typedName: plugins.TypedName{Type: PickerPDKVCacheAwareType, Name: name},
	}
}

// TypedName returns the plugin's type name.
func (p *PickerPDKVCacheAware) TypedName() plugins.TypedName {
	return p.typedName
}

// WithName sets the plugin's name.
func (p *PickerPDKVCacheAware) WithName(name string) *PickerPDKVCacheAware {
	p.typedName.Name = name
	return p
}

// Pick selects the PodGroup with the lowest score from podGroupList
// as the routing target and writes its leader, prefill, and decode pods to ProfileRunResult.
func (p *PickerPDKVCacheAware) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	logger := log.FromContext(ctx)

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](cycleState, common.PDGroupsCycleStateKey)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to read pd group list from state")
		return nil
	}

	if len(pdGroupList.Groups) == 0 {
		logger.V(logging.DEFAULT).Error(errors.New("pd group list is empty"), "pd group list", pdGroupList)
		return nil
	}

	bestPDGroup := pdGroupList.Groups[0]
	for _, pdGroup := range pdGroupList.Groups {
		if pdGroup.LeaderPod.Score < bestPDGroup.LeaderPod.Score {
			bestPDGroup = pdGroup
		}
	}

	return &types.ProfileRunResult{
		TargetPods: []types.Pod{
			bestPDGroup.LeaderPod.Pod,
			bestPDGroup.SelectedPrefillPod.Pod,
			bestPDGroup.SelectedDecodePod.Pod},
	}
}
