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
	"math/rand"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/framework"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/util/logging"

	"hermes-router/pkg/plugins/common"
)

const (
	RandomPDBucketType = "picker-random-pd-bucket"
)

// RandomPDBucket is the plugin struct that implements the framework.Picker interface.
type RandomPDBucket struct {
	typedName plugins.TypedName
	rng       *rand.Rand
}

var _ framework.Picker = &RandomPDBucket{}

// RandomPDBucketFactory is a factory function that creates a plugin instance based on parameters specified by users in EPP configuration files.
func RandomPDBucketFactory(name string, rawParameters json.RawMessage, handle plugins.Handle) (plugins.Plugin, error) {
	return NewRandomPDBucket(name, handle.Context()), nil
}

// NewRandomPDBucket creates a plugin instance based on parameters specified by users in EPP configuration files.
func NewRandomPDBucket(name string, ctx context.Context) *RandomPDBucket {
	return &RandomPDBucket{
		typedName: plugins.TypedName{Type: RandomPDBucketType, Name: name},
		rng:       rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// TypedName returns the plugin's type name.
func (p *RandomPDBucket) TypedName() plugins.TypedName {
	return p.typedName
}

// WithName sets the plugin's name.
func (p *RandomPDBucket) WithName(name string) *RandomPDBucket {
	p.typedName.Name = name
	return p
}

// Pick selects a random PDGroup, and then randomly selects a prefill pod and a decode pod from that group.
func (p *RandomPDBucket) Pick(ctx context.Context, cycleState *types.CycleState, scoredPods []*types.ScoredPod) *types.ProfileRunResult {
	logger := log.FromContext(ctx).WithName("RandomPDBucket.Pick")

	pdGroupList, err := types.ReadCycleStateKey[*common.PDGroupList](cycleState, common.PDGroupsCycleStateKey)
	if err != nil {
		logger.V(logging.DEFAULT).Error(err, "failed to read PDGroupList from cycleState")
		return &types.ProfileRunResult{}
	}

	if pdGroupList == nil || len(pdGroupList.Groups) == 0 {
		logger.V(logging.DEFAULT).Info("PDGroupList is empty")
		return &types.ProfileRunResult{}
	}

	p.ensureRNG()

	selectedGroup := p.selectRandomGroup(pdGroupList)
	if !validPDGroup(selectedGroup) {
		logger.V(logging.DEFAULT).Info("selected PDGroup is invalid", "groupID", selectedGroup.ID)
		return &types.ProfileRunResult{}
	}

	selectedPrefill, ok := p.selectRandomPrefill(selectedGroup)
	if !ok {
		logger.V(logging.DEFAULT).Info("selected PDGroup has no valid prefill pods", "groupID", selectedGroup.ID)
		return &types.ProfileRunResult{}
	}

	selectedDecode, ok := p.selectRandomDecode(selectedGroup)
	if !ok {
		logger.V(logging.DEFAULT).Info("selected PDGroup has no valid decode pods", "groupID", selectedGroup.ID)
		return &types.ProfileRunResult{}
	}

	logger.V(logging.DEFAULT).Info("RandomPDBucket selected",
		"groupID", selectedGroup.ID,
		"leader", selectedGroup.LeaderPod.GetPod().PodName,
		"prefill", selectedPrefill.GetPod().PodName,
		"decode", selectedDecode.GetPod().PodName,
	)

	return &types.ProfileRunResult{
		TargetPods: []types.Pod{
			selectedGroup.LeaderPod.Pod,
			selectedPrefill.Pod,
			selectedDecode.Pod,
		},
	}
}

func validPDGroup(group *common.PDGroup) bool {
	if group == nil {
		return false
	}
	if group.LeaderPod.Pod == nil {
		return false
	}
	if len(group.PrefillPods) == 0 || len(group.DecodePods) == 0 {
		return false
	}
	prefillValid := false
	for _, pod := range group.PrefillPods {
		if pod.Pod != nil {
			prefillValid = true
			break
		}
	}
	if !prefillValid {
		return false
	}
	decodeValid := false
	for _, pod := range group.DecodePods {
		if pod.Pod != nil {
			decodeValid = true
			break
		}
	}
	return decodeValid
}

func (p *RandomPDBucket) ensureRNG() {
	if p.rng == nil {
		p.rng = rand.New(rand.NewSource(time.Now().UnixNano()))
	}
}

func (p *RandomPDBucket) selectRandomGroup(list *common.PDGroupList) *common.PDGroup {
	return &list.Groups[p.rng.Intn(len(list.Groups))]
}

func (p *RandomPDBucket) selectRandomPrefill(group *common.PDGroup) (types.ScoredPod, bool) {
	if len(group.PrefillPods) == 0 {
		return types.ScoredPod{}, false
	}
	pod := group.PrefillPods[p.rng.Intn(len(group.PrefillPods))]
	if pod.Pod == nil {
		return types.ScoredPod{}, false
	}
	return pod, true
}

func (p *RandomPDBucket) selectRandomDecode(group *common.PDGroup) (types.ScoredPod, bool) {
	if len(group.DecodePods) == 0 {
		return types.ScoredPod{}, false
	}
	pod := group.DecodePods[p.rng.Intn(len(group.DecodePods))]
	if pod.Pod == nil {
		return types.ScoredPod{}, false
	}
	return pod, true
}
