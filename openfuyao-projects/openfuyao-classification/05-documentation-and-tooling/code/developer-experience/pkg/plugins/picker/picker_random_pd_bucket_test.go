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

package picker

import (
	"context"
	"math/rand"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"hermes-router/pkg/plugins/common"
)

const (
	testRandomPDBucketPickerName = "test-picker-random-pd-bucket"
	testRandomSeed               = 12345 // Fixed seed for deterministic testing
)

func TestNewRandomPDBucket(t *testing.T) {
	picker := NewRandomPDBucket(testRandomPDBucketPickerName, context.Background())

	if picker.typedName.Name != testRandomPDBucketPickerName {
		t.Errorf("expected name %s, got %s", testRandomPDBucketPickerName, picker.typedName.Name)
	}
	if picker.typedName.Type != RandomPDBucketType {
		t.Errorf("expected type %s, got %s", RandomPDBucketType, picker.typedName.Type)
	}
	if picker.rng == nil {
		t.Error("expected non-nil rng")
	}
}

// TestRandomPDBucketTypedName tests TypedName method.
const (
	testRandomPDBucketTypedNameName = "test-typed-name"
)

func TestRandomPDBucketTypedName(t *testing.T) {
	picker := &RandomPDBucket{
		typedName: plugins.TypedName{Type: RandomPDBucketType, Name: testRandomPDBucketTypedNameName},
	}

	typedName := picker.TypedName()

	if typedName.Type != RandomPDBucketType {
		t.Errorf("expected type %s, got %s", RandomPDBucketType, typedName.Type)
	}
	if typedName.Name != testRandomPDBucketTypedNameName {
		t.Errorf("expected name %s, got %s", testRandomPDBucketTypedNameName, typedName.Name)
	}
}

// TestRandomPDBucketWithName tests WithName method.
const (
	testRandomPDBucketOriginalName = "original-name"
	testRandomPDBucketNewName      = "new-name"
)

func TestRandomPDBucketWithName(t *testing.T) {
	original := &RandomPDBucket{
		typedName: plugins.TypedName{Type: RandomPDBucketType, Name: testRandomPDBucketOriginalName},
	}

	newPicker := original.WithName(testRandomPDBucketNewName)

	if newPicker.typedName.Name != testRandomPDBucketNewName {
		t.Errorf("expected name %s, got %s", testRandomPDBucketNewName, newPicker.typedName.Name)
	}
	if original.typedName.Name != testRandomPDBucketNewName {
		t.Error("expected original picker to be modified")
	}
}

// TestPickRandomSelection tests Pick with multiple groups and pods to verify selection.
// Note: Since it's random, we can't strictly predict the output without fixing the seed,
// but we can verify the structure of the result.
const (
	testGroupCount         = 3
	testPodsPerGroup       = 2
	testExpectedResultPods = 3 // leader + prefill + decode
)

func TestPickRandomSelection(t *testing.T) {
	groups := createRandomTestGroups(testGroupCount, testPodsPerGroup)
	state := &types.CycleState{}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: groups})

	// Use fixed seed for deterministic behavior if needed, or just test valid structure
	picker := &RandomPDBucket{
		typedName: plugins.TypedName{Type: RandomPDBucketType, Name: testRandomPDBucketPickerName},
		rng:       rand.New(rand.NewSource(testRandomSeed)),
	}

	result := picker.Pick(context.Background(), state, nil)

	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.TargetPods) != testExpectedResultPods {
		t.Errorf("expected %d target pods, got %d", testExpectedResultPods, len(result.TargetPods))
	}

	// Verify selected pods belong to one of the valid groups
	leader := result.TargetPods[0]
	prefill := result.TargetPods[1]
	decode := result.TargetPods[2]

	found := false
	for _, g := range groups {
		if g.LeaderPod.Pod == leader && containsPod(g.PrefillPods, prefill) && containsPod(g.DecodePods, decode) {
			found = true
			break
		}
	}

	if !found {
		t.Error("selected pods do not form a valid group combination from the input")
	}
}

// TestPickEmptyState tests Pick with empty or missing state.
func TestPickEmptyState(t *testing.T) {
	picker := NewRandomPDBucket(testRandomPDBucketPickerName, context.Background())

	// Case 1: Missing state key
	state := &types.CycleState{}
	result := picker.Pick(context.Background(), state, nil)
	if result == nil || len(result.TargetPods) != 0 {
		t.Error("expected empty result for missing state key")
	}

	// Case 2: Empty group list
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: []common.PDGroup{}})
	result = picker.Pick(context.Background(), state, nil)
	if result == nil || len(result.TargetPods) != 0 {
		t.Error("expected empty result for empty group list")
	}
}

// TestPickInvalidGroups tests Pick with groups missing required pods.
func TestPickInvalidGroups(t *testing.T) {
	picker := NewRandomPDBucket(testRandomPDBucketPickerName, context.Background())
	state := &types.CycleState{}

	// Group with no prefill pods
	invalidGroup := common.PDGroup{
		ID:          "invalid-group",
		LeaderPod:   types.ScoredPod{Pod: newTestPod("leader")},
		PrefillPods: []types.ScoredPod{}, // Empty
		DecodePods:  []types.ScoredPod{{Pod: newTestPod("decode")}},
	}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: []common.PDGroup{invalidGroup}})

	result := picker.Pick(context.Background(), state, nil)
	if result == nil || len(result.TargetPods) != 0 {
		t.Error("expected empty result for group with no prefill pods")
	}
}

// Helper functions

func createRandomTestGroups(numGroups, podsPerGroup int) []common.PDGroup {
	groups := make([]common.PDGroup, numGroups)
	for i := 0; i < numGroups; i++ {
		leader := types.ScoredPod{Pod: newTestPod("leader")}
		prefills := make([]types.ScoredPod, podsPerGroup)
		decodes := make([]types.ScoredPod, podsPerGroup)

		for j := 0; j < podsPerGroup; j++ {
			prefills[j] = types.ScoredPod{Pod: newTestPod("prefill")}
			decodes[j] = types.ScoredPod{Pod: newTestPod("decode")}
		}

		groups[i] = common.PDGroup{
			ID:          "group",
			LeaderPod:   leader,
			PrefillPods: prefills,
			DecodePods:  decodes,
		}
	}
	return groups
}

func newTestPod(name string) types.Pod {
	return &types.PodMetrics{
		Pod:          &backend.Pod{PodName: name},
		MetricsState: &metrics.MetricsState{},
	}
}

func containsPod(list []types.ScoredPod, target types.Pod) bool {
	for _, sp := range list {
		if sp.Pod == target {
			return true
		}
	}
	return false
}
