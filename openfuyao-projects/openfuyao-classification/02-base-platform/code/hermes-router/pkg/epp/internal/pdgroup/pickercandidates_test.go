/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package pdgroup

import (
	"math/rand"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestPreparePickerCandidatesReturnsInputWithoutLeaders(t *testing.T) {
	scoredPods := []*fwkscheduling.ScoredEndpoint{
		{Endpoint: testutil.NewEndpoint("pod-a", "10.0.80.1", testutil.TestInferencePort, nil)},
		{Endpoint: testutil.NewEndpoint("pod-b", "10.0.80.2", testutil.TestInferencePort, nil)},
	}

	got := PreparePickerCandidates(scoredPods, rand.New(rand.NewSource(1)))
	if len(got) != len(scoredPods) {
		t.Fatalf("expected candidates to be returned unchanged, got %d", len(got))
	}
	if got[0] != scoredPods[0] || got[1] != scoredPods[1] {
		t.Fatal("expected the same scored candidates in non-PD mode")
	}
}

func TestPreparePickerCandidatesPreparesLeadersInPDMode(t *testing.T) {
	prefill := testutil.NewScoredEndpoint("prefill-a", "10.0.81.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{}, nil)
	decode := testutil.NewScoredEndpoint("decode-a", "10.0.82.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{}, nil)
	leader := &fwkscheduling.ScoredEndpoint{
		Endpoint: testutil.NewEndpoint("leader-a", "10.0.83.1", testutil.TestInferencePort, nil),
		Score:    0.9,
	}
	Set(leader.Endpoint, &Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefill},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decode},
	})

	// A candidate without PD metadata is dropped once PD mode is active.
	plain := &fwkscheduling.ScoredEndpoint{
		Endpoint: testutil.NewEndpoint("plain", "10.0.83.2", testutil.TestInferencePort, nil),
		Score:    0.1,
	}

	got := PreparePickerCandidates([]*fwkscheduling.ScoredEndpoint{leader, plain}, rand.New(rand.NewSource(1)))
	if len(got) != 1 {
		t.Fatalf("expected only the prepared leader to remain, got %d", len(got))
	}
	if got[0] != leader {
		t.Fatal("expected the leader candidate to be preserved")
	}
	info, ok := GetDirect(got[0].Endpoint)
	if !ok {
		t.Fatal("expected leader metadata to remain attached")
	}
	if info.SelectedPrefillPod == nil || info.SelectedDecodePod == nil {
		t.Fatalf("expected selected members to be filled, got %+v", info)
	}
	if info.SelectedPrefillPod.GetMetadata().PodName != "prefill-a" {
		t.Fatalf("expected prefill-a to be selected, got %+v", info.SelectedPrefillPod)
	}
	if info.SelectedDecodePod.GetMetadata().PodName != "decode-a" {
		t.Fatalf("expected decode-a to be selected, got %+v", info.SelectedDecodePod)
	}
}
