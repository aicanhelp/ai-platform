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
	"math/rand"
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// TestPickerMinRandomFactory tests PickerMinRandomFactory function.
const (
	testFactoryName = "test-factory-picker"
)

func TestPickerMinRandomFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := plugins.NewEppHandle(context.Background(), nil)

	plugin, err := PickerMinRandomFactory(testFactoryName, rawParams, handle)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin, got nil")
	}
	picker, ok := plugin.(*PickerMinRandom)
	if !ok {
		t.Fatal("expected *PickerMinRandom type")
	}
	if picker.typedName.Name != testFactoryName {
		t.Errorf("expected name %s, got %s", testFactoryName, picker.typedName.Name)
	}
}

// TestNewPickerMinRandom tests NewPickerMinRandom function.
const (
	testNewPickerName = "test-new-picker"
)

func TestNewPickerMinRandom(t *testing.T) {
	picker := NewPickerMinRandom(testNewPickerName, context.Background())

	if picker.typedName.Name != testNewPickerName {
		t.Errorf("expected name %s, got %s", testNewPickerName, picker.typedName.Name)
	}
	if picker.typedName.Type != PickerMinRandomType {
		t.Errorf("expected type %s, got %s", PickerMinRandomType, picker.typedName.Type)
	}
	if picker.rng == nil {
		t.Error("expected non-nil rng, got nil")
	}
}

// TestTypedName tests TypedName method.
const (
	testTypedNameName = "test-typed-name"
)

func TestTypedName(t *testing.T) {
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testTypedNameName},
	}

	typedName := picker.TypedName()

	if typedName.Type != PickerMinRandomType {
		t.Errorf("expected type %s, got %s", PickerMinRandomType, typedName.Type)
	}
	if typedName.Name != testTypedNameName {
		t.Errorf("expected name %s, got %s", testTypedNameName, typedName.Name)
	}
}

// TestWithName tests WithName method.
const (
	testOriginalName = "original-name"
	testNewName      = "new-name"
)

func TestWithName(t *testing.T) {
	original := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testOriginalName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	newPicker := original.WithName(testNewName)

	if newPicker.typedName.Name != testNewName {
		t.Errorf("expected name %s, got %s", testNewName, newPicker.typedName.Name)
	}
	if original.typedName.Name != testNewName {
		t.Error("expected original picker to be modified")
	}
}

// TestPickEmptyList tests Pick with empty list.
const (
	testEmptyListLength = 0
	testSeedEmpty       = 123
)

func TestPickEmptyList(t *testing.T) {
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeedEmpty)),
	}

	result := picker.Pick(context.Background(), &types.CycleState{}, nil)

	if len(result.TargetPods) != testEmptyListLength {
		t.Errorf("expected %d target pods, got %d", testEmptyListLength, len(result.TargetPods))
	}
}

// TestPickSinglePod tests Pick with single pod.
const (
	testPickerName       = "test-picker"
	testSeed             = 123
	testScoreLow         = 0.5
	testPodNameSingle    = "pod-single"
	testPodAddress       = "127.0.0.1"
	testWaitingQueueZero = 0
	testSinglePodCount   = 1
)

func TestPickSinglePod(t *testing.T) {
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	scoredPod := newScoredPod(testPodNameSingle, testScoreLow)
	result := picker.Pick(context.Background(), &types.CycleState{}, []*types.ScoredPod{scoredPod})

	if len(result.TargetPods) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetPods))
	}
	if result.TargetPods[0] != scoredPod.Pod {
		t.Errorf("expected target pod %v, got %v", scoredPod.Pod, result.TargetPods[0])
	}
}

// TestPickRandomAmongLowestScore tests Pick with multiple pods having same lowest score.
const (
	testScoreHigh             = 1.5
	testPodNameHigh           = "pod-high"
	testPodNameLowPrimary     = "pod-low-primary"
	testPodNameLowAlternative = "pod-low-alternative"
	testScoreLowAlternative   = 0.5
)

func TestPickRandomAmongLowestScore(t *testing.T) {
	rng := rand.New(rand.NewSource(testSeed))
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testPickerName},
		rng:       rng,
	}

	podHigh := newScoredPod(testPodNameHigh, testScoreHigh)
	podLowPrimary := newScoredPod(testPodNameLowPrimary, testScoreLow)
	podLowAlternative := newScoredPod(testPodNameLowAlternative, testScoreLowAlternative)

	scoredPods := []*types.ScoredPod{podHigh, podLowPrimary, podLowAlternative}

	result := picker.Pick(context.Background(), &types.CycleState{}, scoredPods)

	if len(result.TargetPods) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetPods))
	}

	expectedCandidates := []types.Pod{podLowPrimary.Pod, podLowAlternative.Pod}
	expectedRng := rand.New(rand.NewSource(testSeed))
	expectedIndex := expectedRng.Intn(len(expectedCandidates))
	expectedTarget := expectedCandidates[expectedIndex]

	if result.TargetPods[0] != expectedTarget {
		t.Errorf("expected target pod %v, got %v", expectedTarget, result.TargetPods[0])
	}
}

// TestPickWithNilRng tests Pick when rng is nil.
const (
	testPodName1 = "pod-1"
	testPodName2 = "pod-2"
	testScore1   = 0.3
	testScore2   = 0.7
)

func TestPickWithNilRng(t *testing.T) {
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testPickerName},
		rng:       nil,
	}

	pod1 := newScoredPod(testPodName1, testScore1)
	pod2 := newScoredPod(testPodName2, testScore2)

	result := picker.Pick(context.Background(), &types.CycleState{}, []*types.ScoredPod{pod1, pod2})

	if len(result.TargetPods) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetPods))
	}
	if result.TargetPods[0] != pod1.Pod {
		t.Errorf("expected target pod %v, got %v", pod1.Pod, result.TargetPods[0])
	}
	if picker.rng == nil {
		t.Error("expected rng to be initialized, got nil")
	}
}

// TestPickMultipleCandidates tests Pick with multiple candidates having same score.
const (
	testPodName3  = "pod-3"
	testPodName4  = "pod-4"
	testScoreSame = 0.4
)

func TestPickMultipleCandidates(t *testing.T) {
	picker := &PickerMinRandom{
		typedName: plugins.TypedName{Type: PickerMinRandomType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	pod1 := newScoredPod(testPodName1, testScoreSame)
	pod2 := newScoredPod(testPodName2, testScoreSame)
	pod3 := newScoredPod(testPodName3, testScoreSame)
	pod4 := newScoredPod(testPodName4, testScoreHigh)

	result := picker.Pick(context.Background(), &types.CycleState{}, []*types.ScoredPod{pod1, pod2, pod3, pod4})

	if len(result.TargetPods) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetPods))
	}

	selectedPod := result.TargetPods[0]
	if selectedPod != pod1.Pod && selectedPod != pod2.Pod && selectedPod != pod3.Pod {
		t.Errorf("expected target pod to be one of pod1, pod2, or pod3, got %v", selectedPod)
	}
	if selectedPod == pod4.Pod {
		t.Error("expected target pod not to be pod4 (highest score)")
	}
}

// Helper functions

func newScoredPod(name string, score float64) *types.ScoredPod {
	return &types.ScoredPod{
		Pod: &types.PodMetrics{
			Pod: &backend.Pod{
				PodName: name,
				Address: testPodAddress,
			},
			MetricsState: &metrics.MetricsState{
				WaitingQueueSize: testWaitingQueueZero,
			},
		},
		Score: score,
	}
}
