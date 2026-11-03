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

package random

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
)

// TestPickerMinRandomFactory tests Factory.
const (
	testFactoryName = "test-factory-picker"
)

func TestFactory(t *testing.T) {
	rawParams := json.RawMessage(`{}`)
	handle := fwkplugin.NewEppHandle(context.Background(), nil)

	plugin, err := Factory(testFactoryName, rawParams, handle)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	picker, ok := plugin.(*Picker)
	if !ok {
		t.Fatalf("expected *Picker, got %T", plugin)
	}
	if picker.typedName.Name != testFactoryName {
		t.Errorf("expected name %s, got %s", testFactoryName, picker.typedName.Name)
	}
}

func TestFactoryRejectsNullParameters(t *testing.T) {
	_, err := Factory(testFactoryName, json.RawMessage(`null`), fwkplugin.NewEppHandle(context.Background(), nil))
	if err == nil {
		t.Fatal("expected explicit null parameters error")
	}
}

func TestFactoryRejectsUnknownFields(t *testing.T) {
	_, err := Factory(
		testFactoryName,
		json.RawMessage(`{"unexpected":1}`),
		fwkplugin.NewEppHandle(context.Background(), nil),
	)
	if err == nil {
		t.Fatal("expected unknown field error")
	}
}

// TestNewPickerMinRandom tests New.
const (
	testNewPickerName = "test-new-picker"
)

func TestNew(t *testing.T) {
	picker := New(testNewPickerName)

	if picker.typedName.Name != testNewPickerName {
		t.Errorf("expected name %s, got %s", testNewPickerName, picker.typedName.Name)
	}
	if picker.typedName.Type != PluginType {
		t.Errorf("expected type %s, got %s", PluginType, picker.typedName.Type)
	}
	if picker.rng == nil {
		t.Error("expected non-nil rng, got nil")
	}
}

// TestTypedName tests TypedName method.
const (
	testTypedNameName = "test-typed-name"
	beforeName        = "old-name"
	newName           = "new-name"
)

func TestTypedName(t *testing.T) {
	picker := &Picker{typedName: fwkplugin.TypedName{Type: PluginType, Name: testTypedNameName}}

	typedName := picker.TypedName()

	if typedName.Type != PluginType {
		t.Errorf("expected type %s, got %s", PluginType, typedName.Type)
	}
	if typedName.Name != testTypedNameName {
		t.Errorf("expected name %s, got %s", testTypedNameName, typedName.Name)
	}
}

func TestWithName(t *testing.T) {
	original := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: beforeName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	newPicker := original.WithName(newName)

	if newPicker.typedName.Name != newName {
		t.Errorf("expected name %s, got %s", newName, newPicker.typedName.Name)
	}
	if original.typedName.Name != newName {
		t.Error("expected original picker to be modified")
	}
}

// TestPickEmptyList tests Pick with empty list.
const (
	testEmptyListLength = 0
	testSeedEmpty       = 123
)

func TestPickEmptyList(t *testing.T) {
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeedEmpty)),
	}

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), nil)

	if len(result.TargetEndpoints) != testEmptyListLength {
		t.Errorf("expected %d target pods, got %d", testEmptyListLength, len(result.TargetEndpoints))
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
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	scoredPod := newScoredPod(testPodNameSingle, testScoreLow)
	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{scoredPod})

	if len(result.TargetEndpoints) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetEndpoints))
	}
	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], scoredPod.Endpoint) {
		t.Errorf("expected target endpoint %v, got %v", scoredPod.Endpoint, result.TargetEndpoints[0])
	}
}

// TestPickRandomAmongCandidates tests Pick with multiple pods.
const (
	testScoreHigh             = 1.5
	testPodNameHigh           = "pod-high"
	testPodNameLowPrimary     = "pod-low-primary"
	testPodNameLowAlternative = "pod-low-alternative"
	testScoreLowAlternative   = 0.5
)

func TestPickRandomAmongCandidates(t *testing.T) {
	rng := rand.New(rand.NewSource(testSeed))
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       rng,
	}

	podHigh := newScoredPod(testPodNameHigh, testScoreHigh)
	podLowPrimary := newScoredPod(testPodNameLowPrimary, testScoreLow)
	podLowAlternative := newScoredPod(testPodNameLowAlternative, testScoreLowAlternative)

	scoredPods := []*fwkscheduling.ScoredEndpoint{podHigh, podLowPrimary, podLowAlternative}

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), scoredPods)

	if len(result.TargetEndpoints) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetEndpoints))
	}

	expectedCandidates := []fwkscheduling.Endpoint{podHigh.Endpoint, podLowPrimary.Endpoint, podLowAlternative.Endpoint}
	expectedRng := rand.New(rand.NewSource(testSeed))
	expectedIndex := expectedRng.Intn(len(expectedCandidates))
	expectedTarget := expectedCandidates[expectedIndex]

	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], expectedTarget) {
		t.Errorf("expected target endpoint %v, got %v", expectedTarget, result.TargetEndpoints[0])
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
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       nil,
	}

	pod1 := newScoredPod(testPodName1, testScore1)
	pod2 := newScoredPod(testPodName2, testScore2)

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{pod1, pod2})

	if len(result.TargetEndpoints) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetEndpoints))
	}
	if picker.rng == nil {
		t.Error("expected rng to be initialized, got nil")
	}
}

// TestPickMultipleCandidates tests Pick with multiple candidates having different scores.
const (
	testPodName3  = "pod-3"
	testPodName4  = "pod-4"
	testScoreSame = 0.4
)

func TestPickMultipleCandidates(t *testing.T) {
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	pod1 := newScoredPod(testPodName1, testScoreSame)
	pod2 := newScoredPod(testPodName2, testScoreSame)
	pod3 := newScoredPod(testPodName3, testScoreSame)
	pod4 := newScoredPod(testPodName4, testScoreHigh)

	result := picker.Pick(
		context.Background(),
		fwkscheduling.NewCycleState(),
		[]*fwkscheduling.ScoredEndpoint{pod1, pod2, pod3, pod4},
	)

	if len(result.TargetEndpoints) != testSinglePodCount {
		t.Errorf("expected %d target pod, got %d", testSinglePodCount, len(result.TargetEndpoints))
	}

	selectedPod := result.TargetEndpoints[0]
	if !fwkscheduling.EndpointComparer(selectedPod, pod1.Endpoint) &&
		!fwkscheduling.EndpointComparer(selectedPod, pod2.Endpoint) &&
		!fwkscheduling.EndpointComparer(selectedPod, pod3.Endpoint) &&
		!fwkscheduling.EndpointComparer(selectedPod, pod4.Endpoint) {
		t.Errorf("expected target endpoint to be one of pod1, pod2, or pod3, got %v", selectedPod)
	}
}

func TestPickPreparesPDCandidates(t *testing.T) {
	picker := &Picker{
		typedName: fwkplugin.TypedName{Type: PluginType, Name: testPickerName},
		rng:       rand.New(rand.NewSource(testSeed)),
	}

	invalidLeader := newRandomTestEndpoint("leader-invalid")
	pdgroup.Set(invalidLeader, &pdgroup.Info{GroupID: "invalid"})

	validLeader := newRandomTestEndpoint("leader-valid")
	pdgroup.Set(validLeader, &pdgroup.Info{
		GroupID:     "group-valid",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newRandomTestEndpoint("prefill-valid"), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newRandomTestEndpoint("decode-valid"), Score: 0.2}},
	})

	result := picker.Pick(context.Background(), fwkscheduling.NewCycleState(), []*fwkscheduling.ScoredEndpoint{
		{Endpoint: invalidLeader, Score: 0.1},
		{Endpoint: validLeader, Score: 0.2},
	})

	if len(result.TargetEndpoints) != 1 {
		t.Fatalf("expected one selected PD leader, got %d", len(result.TargetEndpoints))
	}
	if !fwkscheduling.EndpointComparer(result.TargetEndpoints[0], validLeader) {
		t.Fatal("expected valid PD leader to be selected")
	}
	preparedInfo, ok := pdgroup.Get(validLeader)
	if !ok {
		t.Fatal("expected selected PD leader to retain PD metadata")
	}
	if preparedInfo.SelectedPrefillPod == nil || preparedInfo.SelectedPrefillPod.Endpoint == nil {
		t.Fatal("expected selected PD leader to have chosen prefill pod")
	}
	if preparedInfo.SelectedDecodePod == nil || preparedInfo.SelectedDecodePod.Endpoint == nil {
		t.Fatal("expected selected PD leader to have chosen decode pod")
	}
}

func TestLogCandidatePodsSummaryToleratesNilScoredPod(t *testing.T) {
	t.Helper()

	defer func() {
		if recovered := recover(); recovered != nil {
			t.Fatalf("expected nil scored pod logging to avoid panic, got %v", recovered)
		}
	}()

	logCandidatePodsSummary(context.Background(), "Picker.Pick", "candidate pods", []*fwkscheduling.ScoredEndpoint{nil})
}

// Helper functions
func newRandomTestEndpoint(name string) fwkscheduling.Endpoint {
	return newTestScoredPod(name, testPodAddress, 0).Endpoint
}

func newScoredPod(name string, score float64) *fwkscheduling.ScoredEndpoint {
	scored := newTestScoredPod(name, testPodAddress, score)
	scored.Endpoint.GetMetrics().WaitingQueueSize = testWaitingQueueZero
	return &scored
}

func newTestScoredPod(name, address string, score float64) fwkscheduling.ScoredEndpoint {
	return fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(
			&fwkdl.EndpointMetadata{
				NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
				PodName:        name,
				Address:        address,
				Port:           "8080",
				Labels:         map[string]string{},
			},
			fwkdl.NewMetrics(),
			fwkdl.NewAttributes(),
		),
		Score: score,
	}
}
