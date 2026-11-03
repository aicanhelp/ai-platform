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
	"encoding/json"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"

	"hermes-router/pkg/plugins/common"
)

// TestPickerPDKVCacheAwareFactory tests PickerPDKVCacheAwareFactory function.
const (
	testPDKVCacheAwareFactoryName = "test-factory-picker-pd-kv-cache-aware"
)

func TestPickerPDKVCacheAwareFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := PickerPDKVCacheAwareFactory(testPDKVCacheAwareFactoryName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
	picker, ok := plugin.(*PickerPDKVCacheAware)
	if !ok {
		t.Fatal("expected *PickerPDKVCacheAware type")
	}
	if picker.typedName.Name != testPDKVCacheAwareFactoryName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareFactoryName, picker.typedName.Name)
	}
}

// TestNewPickerPDKVCacheAware tests NewPickerPDKVCacheAware function.
const (
	testPDKVCacheAwareNewPickerName = "test-new-picker-pd-kv-cache-aware"
)

func TestNewPickerPDKVCacheAware(t *testing.T) {
	picker := NewPickerPDKVCacheAware(testPDKVCacheAwareNewPickerName, context.Background())

	if picker.typedName.Name != testPDKVCacheAwareNewPickerName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareNewPickerName, picker.typedName.Name)
	}
	if picker.typedName.Type != PickerPDKVCacheAwareType {
		t.Errorf("expected type %s, got %s", PickerPDKVCacheAwareType, picker.typedName.Type)
	}
}

// TestPickerPDKVCacheAwareTypedName tests TypedName method.
const (
	testPDKVCacheAwareTypedNameName = "test-typed-name"
)

func TestPickerPDKVCacheAwareTypedName(t *testing.T) {
	picker := &PickerPDKVCacheAware{
		typedName: plugins.TypedName{Type: PickerPDKVCacheAwareType, Name: testPDKVCacheAwareTypedNameName},
	}

	typedName := picker.TypedName()

	if typedName.Type != PickerPDKVCacheAwareType {
		t.Errorf("expected type %s, got %s", PickerPDKVCacheAwareType, typedName.Type)
	}
	if typedName.Name != testPDKVCacheAwareTypedNameName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareTypedNameName, typedName.Name)
	}
}

// TestPickerPDKVCacheAwareWithName tests WithName method.
const (
	testPDKVCacheAwareOriginalName = "original-name"
	testPDKVCacheAwareNewName      = "new-name"
)

func TestPickerPDKVCacheAwareWithName(t *testing.T) {
	original := &PickerPDKVCacheAware{
		typedName: plugins.TypedName{Type: PickerPDKVCacheAwareType, Name: testPDKVCacheAwareOriginalName},
	}

	newPicker := original.WithName(testPDKVCacheAwareNewName)

	if newPicker.typedName.Name != testPDKVCacheAwareNewName {
		t.Errorf("expected name %s, got %s", testPDKVCacheAwareNewName, newPicker.typedName.Name)
	}
	if original.typedName.Name != testPDKVCacheAwareNewName {
		t.Error("expected original picker to be modified")
	}
}

// TestPickMultipleGroups tests Pick with multiple groups.
const (
	testPDKVCacheAwarePickerName = "test-picker-pd-kv-cache-aware"
	testGroup1ID                 = "group-1"
	testGroup2ID                 = "group-2"
	testGroup3ID                 = "group-3"
	testGroup1LeaderScore        = 1.5
	testGroup2LeaderScore        = 0.8
	testGroup3LeaderScore        = 1.2
	testGroup1PrefillScore       = 0.5
	testGroup1DecodeScore        = 0.3
	testGroup2PrefillScore       = 0.2
	testGroup2DecodeScore        = 0.3
	testGroup3PrefillScore       = 0.35
	testGroup3DecodeScore        = 0.35
	testExpectedTargetPods       = 3
	testGroupIndex1              = 1
	testPodIndex0                = 0
	testPodIndex1                = 1
	testPodIndex2                = 2
)

func TestPickMultipleGroups(t *testing.T) {
	groups := createTestGroups()
	state := createStateWithGroups(groups)
	picker := NewPickerPDKVCacheAware(testPDKVCacheAwarePickerName, context.Background())

	result := picker.Pick(context.Background(), state, nil)

	if result == nil {
		t.Fatal("expected non-nil result")
	}
	if len(result.TargetPods) != testExpectedTargetPods {
		t.Errorf("expected %d target pods, got %d", testExpectedTargetPods, len(result.TargetPods))
	}
	verifySelectedGroup(t, result, groups[testGroupIndex1])
}

// TestPickEmptyGroupList tests Pick with empty group list.
const (
	testEmptyGroupsLength = 0
)

func TestPickEmptyGroupList(t *testing.T) {
	state := &types.CycleState{}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{
		Groups: []common.PDGroup{},
	})
	picker := NewPickerPDKVCacheAware(testPDKVCacheAwarePickerName, context.Background())

	result := picker.Pick(context.Background(), state, nil)

	if result != nil {
		t.Error("expected nil result for empty group list")
	}
}

// TestPickMissingState tests Pick with missing state.
func TestPickMissingState(t *testing.T) {
	state := &types.CycleState{}
	picker := NewPickerPDKVCacheAware(testPDKVCacheAwarePickerName, context.Background())

	result := picker.Pick(context.Background(), state, nil)

	if result != nil {
		t.Error("expected nil result for missing state")
	}
}

// TestPickSingleGroup tests Pick with single group.
const (
	testSingleGroupLeaderScore  = 1.0
	testSingleGroupPrefillScore = 0.5
	testSingleGroupDecodeScore  = 0.4
)

func TestPickSingleGroup(t *testing.T) {
	groups := []common.PDGroup{
		createPDGroup(testGroup1ID, testSingleGroupLeaderScore, testSingleGroupPrefillScore, testSingleGroupDecodeScore),
	}
	state := createStateWithGroups(groups)
	picker := NewPickerPDKVCacheAware(testPDKVCacheAwarePickerName, context.Background())

	result := picker.Pick(context.Background(), state, nil)

	if result == nil {
		t.Fatal("expected non-nil result for single group")
	}
	if len(result.TargetPods) != testExpectedTargetPods {
		t.Errorf("expected %d target pods, got %d", testExpectedTargetPods, len(result.TargetPods))
	}
}

// Helper functions

const (
	testGroup1Prefill0Name  = "group-1-prefill-0"
	testGroup1Prefill0IP    = "10.0.0.1"
	testGroup1Prefill1Name  = "group-1-prefill-1"
	testGroup1Prefill1IP    = "10.0.0.2"
	testGroup1Decode0Name   = "group-1-decode-0"
	testGroup1Decode0IP     = "10.0.1.1"
	testGroup1Decode1Name   = "group-1-decode-1"
	testGroup1Decode1IP     = "10.0.1.2"
	testGroup1Decode2Name   = "group-1-decode-2"
	testGroup1Decode2IP     = "10.0.1.3"
	testGroup1LeaderName    = "group-1-leader"
	testGroup1LeaderIP      = "10.0.2.1"
	testGroup2Prefill0Name  = "group-2-prefill-0"
	testGroup2Prefill0IP    = "10.0.0.3"
	testGroup2Prefill1Name  = "group-2-prefill-1"
	testGroup2Prefill1IP    = "10.0.0.4"
	testGroup2Prefill2Name  = "group-2-prefill-2"
	testGroup2Prefill2IP    = "10.0.0.5"
	testGroup2Decode0Name   = "group-2-decode-0"
	testGroup2Decode0IP     = "10.0.1.4"
	testGroup2LeaderName    = "group-2-leader"
	testGroup2LeaderIP      = "10.0.2.2"
	testGroup3Prefill0Name  = "group-3-prefill-0"
	testGroup3Prefill0IP    = "10.0.0.6"
	testGroup3Prefill1Name  = "group-3-prefill-1"
	testGroup3Prefill1IP    = "10.0.0.7"
	testGroup3Prefill2Name  = "group-3-prefill-2"
	testGroup3Prefill2IP    = "10.0.0.8"
	testGroup3Decode0Name   = "group-3-decode-0"
	testGroup3Decode0IP     = "10.0.1.5"
	testGroup3Decode1Name   = "group-3-decode-1"
	testGroup3Decode1IP     = "10.0.1.6"
	testGroup3Decode2Name   = "group-3-decode-2"
	testGroup3Decode2IP     = "10.0.1.7"
	testGroup3LeaderName    = "group-3-leader"
	testGroup3LeaderIP      = "10.0.2.3"
	testGroup1Prefill0Score = 0.5
	testGroup1Prefill1Score = 0.6
	testGroup1Decode0Score  = 0.4
	testGroup1Decode1Score  = 0.5
	testGroup1Decode2Score  = 0.3
	testGroup2Prefill0Score = 0.2
	testGroup2Prefill1Score = 0.3
	testGroup2Prefill2Score = 0.25
	testGroup2Decode0Score  = 0.3
	testGroup3Prefill0Score = 0.4
	testGroup3Prefill1Score = 0.45
	testGroup3Prefill2Score = 0.35
	testGroup3Decode0Score  = 0.4
	testGroup3Decode1Score  = 0.35
	testGroup3Decode2Score  = 0.45
)

func createTestGroups() []common.PDGroup {
	return []common.PDGroup{
		{
			ID: testGroup1ID,
			PrefillPods: []types.ScoredPod{
				newScoredPodValue(testGroup1Prefill0Name, testGroup1Prefill0IP, testGroup1Prefill0Score),
				newScoredPodValue(testGroup1Prefill1Name, testGroup1Prefill1IP, testGroup1Prefill1Score),
			},
			DecodePods: []types.ScoredPod{
				newScoredPodValue(testGroup1Decode0Name, testGroup1Decode0IP, testGroup1Decode0Score),
				newScoredPodValue(testGroup1Decode1Name, testGroup1Decode1IP, testGroup1Decode1Score),
				newScoredPodValue(testGroup1Decode2Name, testGroup1Decode2IP, testGroup1Decode2Score),
			},
			LeaderPod:          newScoredPodValue(testGroup1LeaderName, testGroup1LeaderIP, testGroup1LeaderScore),
			SelectedPrefillPod: newScoredPodValue(testGroup1Prefill0Name, testGroup1Prefill0IP, testGroup1Prefill0Score),
			SelectedDecodePod:  newScoredPodValue(testGroup1Decode2Name, testGroup1Decode2IP, testGroup1Decode2Score),
		},
		{
			ID: testGroup2ID,
			PrefillPods: []types.ScoredPod{
				newScoredPodValue(testGroup2Prefill0Name, testGroup2Prefill0IP, testGroup2Prefill0Score),
				newScoredPodValue(testGroup2Prefill1Name, testGroup2Prefill1IP, testGroup2Prefill1Score),
				newScoredPodValue(testGroup2Prefill2Name, testGroup2Prefill2IP, testGroup2Prefill2Score),
			},
			DecodePods: []types.ScoredPod{
				newScoredPodValue(testGroup2Decode0Name, testGroup2Decode0IP, testGroup2Decode0Score),
			},
			LeaderPod:          newScoredPodValue(testGroup2LeaderName, testGroup2LeaderIP, testGroup2LeaderScore),
			SelectedPrefillPod: newScoredPodValue(testGroup2Prefill0Name, testGroup2Prefill0IP, testGroup2Prefill0Score),
			SelectedDecodePod:  newScoredPodValue(testGroup2Decode0Name, testGroup2Decode0IP, testGroup2Decode0Score),
		},
		{
			ID: testGroup3ID,
			PrefillPods: []types.ScoredPod{
				newScoredPodValue(testGroup3Prefill0Name, testGroup3Prefill0IP, testGroup3Prefill0Score),
				newScoredPodValue(testGroup3Prefill1Name, testGroup3Prefill1IP, testGroup3Prefill1Score),
				newScoredPodValue(testGroup3Prefill2Name, testGroup3Prefill2IP, testGroup3Prefill2Score),
			},
			DecodePods: []types.ScoredPod{
				newScoredPodValue(testGroup3Decode0Name, testGroup3Decode0IP, testGroup3Decode0Score),
				newScoredPodValue(testGroup3Decode1Name, testGroup3Decode1IP, testGroup3Decode1Score),
				newScoredPodValue(testGroup3Decode2Name, testGroup3Decode2IP, testGroup3Decode2Score),
			},
			LeaderPod:          newScoredPodValue(testGroup3LeaderName, testGroup3LeaderIP, testGroup3LeaderScore),
			SelectedPrefillPod: newScoredPodValue(testGroup3Prefill2Name, testGroup3Prefill2IP, testGroup3Prefill2Score),
			SelectedDecodePod:  newScoredPodValue(testGroup3Decode1Name, testGroup3Decode1IP, testGroup3Decode1Score),
		},
	}
}

const (
	testDefaultPrefillIP = "10.0.0.1"
	testDefaultDecodeIP  = "10.0.1.1"
	testDefaultLeaderIP  = "10.0.2.1"
	testPrefillSuffix    = "-prefill-0"
	testDecodeSuffix     = "-decode-0"
	testLeaderSuffix     = "-leader"
)

func createPDGroup(groupID string, leaderScore, prefillScore, decodeScore float64) common.PDGroup {
	return common.PDGroup{
		ID: groupID,
		PrefillPods: []types.ScoredPod{
			newScoredPodValue(groupID+testPrefillSuffix, testDefaultPrefillIP, prefillScore),
		},
		DecodePods: []types.ScoredPod{
			newScoredPodValue(groupID+testDecodeSuffix, testDefaultDecodeIP, decodeScore),
		},
		LeaderPod:          newScoredPodValue(groupID+testLeaderSuffix, testDefaultLeaderIP, leaderScore),
		SelectedPrefillPod: newScoredPodValue(groupID+testPrefillSuffix, testDefaultPrefillIP, prefillScore),
		SelectedDecodePod:  newScoredPodValue(groupID+testDecodeSuffix, testDefaultDecodeIP, decodeScore),
	}
}

func createStateWithGroups(groups []common.PDGroup) *types.CycleState {
	state := &types.CycleState{}
	state.Write(common.PDGroupsCycleStateKey, &common.PDGroupList{Groups: groups})
	return state
}

func verifySelectedGroup(t *testing.T, result *types.ProfileRunResult, expectedGroup common.PDGroup) {
	expectedLeaderPod := expectedGroup.LeaderPod.Pod
	expectedPrefillPod := expectedGroup.SelectedPrefillPod.Pod
	expectedDecodePod := expectedGroup.SelectedDecodePod.Pod

	if result.TargetPods[testPodIndex0] != expectedLeaderPod {
		t.Errorf("expected leader pod %v, got %v", expectedLeaderPod, result.TargetPods[testPodIndex0])
	}
	if result.TargetPods[testPodIndex1] != expectedPrefillPod {
		t.Errorf("expected prefill pod %v, got %v", expectedPrefillPod, result.TargetPods[testPodIndex1])
	}
	if result.TargetPods[testPodIndex2] != expectedDecodePod {
		t.Errorf("expected decode pod %v, got %v", expectedDecodePod, result.TargetPods[testPodIndex2])
	}

	leaderPod := result.TargetPods[testPodIndex0].GetPod()
	prefillPod := result.TargetPods[testPodIndex1].GetPod()
	decodePod := result.TargetPods[testPodIndex2].GetPod()

	if leaderPod == nil || prefillPod == nil || decodePod == nil {
		t.Fatal("expected all pods to have valid backend pod")
	}
	if leaderPod.Address != testGroup2LeaderIP {
		t.Errorf("expected leader pod address %s, got %s", testGroup2LeaderIP, leaderPod.Address)
	}
	if prefillPod.Address != testGroup2Prefill0IP {
		t.Errorf("expected prefill pod address %s, got %s", testGroup2Prefill0IP, prefillPod.Address)
	}
	if decodePod.Address != testGroup2Decode0IP {
		t.Errorf("expected decode pod address %s, got %s", testGroup2Decode0IP, decodePod.Address)
	}
}

func newScoredPodValue(name, address string, score float64) types.ScoredPod {
	return types.ScoredPod{
		Pod: &types.PodMetrics{
			Pod: &backend.Pod{
				PodName: name,
				Address: address,
			},
			MetricsState: &metrics.MetricsState{},
		},
		Score: score,
	}
}
