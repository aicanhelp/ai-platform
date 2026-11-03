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

package common

import (
	"testing"

	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/backend/metrics"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/scheduling/types"
)

// TestCloneNil tests Clone method with nil PDGroupList.
func TestCloneNil(t *testing.T) {
	var pdGroupList *PDGroupList

	cloned := pdGroupList.Clone()

	if cloned == nil {
		t.Fatal("expected non-nil cloned result, got nil")
	}
	clonedList, ok := cloned.(*PDGroupList)
	if !ok {
		t.Fatal("expected cloned result to be *PDGroupList")
	}
	if len(clonedList.Groups) != 0 {
		t.Fatalf("expected empty groups, got %d", len(clonedList.Groups))
	}
}

// TestCloneEmpty tests Clone method with empty PDGroupList.
const (
	testEmptyGroupsLength = 0
)

func TestCloneEmpty(t *testing.T) {
	pdGroupList := &PDGroupList{
		Groups: []PDGroup{},
	}

	cloned := pdGroupList.Clone()

	clonedList, ok := cloned.(*PDGroupList)
	if !ok {
		t.Fatal("expected cloned result to be *PDGroupList")
	}
	if len(clonedList.Groups) != testEmptyGroupsLength {
		t.Fatalf("expected %d groups, got %d", testEmptyGroupsLength, len(clonedList.Groups))
	}
}

// TestCloneSingleGroup tests Clone method with single PDGroup.
const (
	testG1ID        = "group-1"
	testG1P1Name    = "prefill-1"
	testG1P1IP      = "10.0.0.1"
	testG1P1Score   = 0.5
	testG1D1Name    = "decode-1"
	testG1D1IP      = "10.0.1.1"
	testG1D1Score   = 0.6
	testG1LName     = "leader-1"
	testG1LIP       = "10.0.2.1"
	testG1LScore    = 1.0
	testG1SelPName  = "selected-prefill-1"
	testG1SelPIP    = "10.0.3.1"
	testG1SelPScore = 0.7
	testG1SelDName  = "selected-decode-1"
	testG1SelDIP    = "10.0.4.1"
	testG1SelDScore = 0.8
)

func TestCloneSingleGroup(t *testing.T) {
	pdGroupList := &PDGroupList{
		Groups: []PDGroup{
			{
				ID: testG1ID,
				PrefillPods: []types.ScoredPod{
					newScoredPod(testG1P1Name, testG1P1IP, testG1P1Score),
				},
				DecodePods: []types.ScoredPod{
					newScoredPod(testG1D1Name, testG1D1IP, testG1D1Score),
				},
				LeaderPod:          newScoredPod(testG1LName, testG1LIP, testG1LScore),
				SelectedPrefillPod: newScoredPod(testG1SelPName, testG1SelPIP, testG1SelPScore),
				SelectedDecodePod:  newScoredPod(testG1SelDName, testG1SelDIP, testG1SelDScore),
			},
		},
	}

	cloned := pdGroupList.Clone()

	clonedList, ok := cloned.(*PDGroupList)
	if !ok {
		t.Fatal("expected cloned result to be *PDGroupList")
	}
	if len(clonedList.Groups) != 1 {
		t.Fatalf("expected 1 group, got %d", len(clonedList.Groups))
	}
	verifyGroupClone(t, pdGroupList.Groups[0], clonedList.Groups[0])
}

// TestCloneMultipleGroups tests Clone method with multiple PDGroups.
const (
	testG2ID        = "group-2"
	testG2P1Name    = "prefill-2-1"
	testG2P1IP      = "10.0.0.2"
	testG2P1Score   = 0.3
	testG2P2Name    = "prefill-2-2"
	testG2P2IP      = "10.0.0.3"
	testG2P2Score   = 0.4
	testG2D1Name    = "decode-2-1"
	testG2D1IP      = "10.0.1.2"
	testG2D1Score   = 0.5
	testG2LName     = "leader-2"
	testG2LIP       = "10.0.2.2"
	testG2LScore    = 1.5
	testG2SelPName  = "selected-prefill-2"
	testG2SelPIP    = "10.0.3.2"
	testG2SelPScore = 0.6
	testG2SelDName  = "selected-decode-2"
	testG2SelDIP    = "10.0.4.2"
	testG2SelDScore = 0.7
	testExpGrpCnt   = 2
)

func TestCloneMultipleGroups(t *testing.T) {
	pdGroupList := &PDGroupList{
		Groups: []PDGroup{
			{
				ID: testG1ID,
				PrefillPods: []types.ScoredPod{
					newScoredPod(testG1P1Name, testG1P1IP, testG1P1Score),
				},
				DecodePods: []types.ScoredPod{
					newScoredPod(testG1D1Name, testG1D1IP, testG1D1Score),
				},
				LeaderPod:          newScoredPod(testG1LName, testG1LIP, testG1LScore),
				SelectedPrefillPod: newScoredPod(testG1SelPName, testG1SelPIP, testG1SelPScore),
				SelectedDecodePod:  newScoredPod(testG1SelDName, testG1SelDIP, testG1SelDScore),
			},
			{
				ID: testG2ID,
				PrefillPods: []types.ScoredPod{
					newScoredPod(testG2P1Name, testG2P1IP, testG2P1Score),
					newScoredPod(testG2P2Name, testG2P2IP, testG2P2Score),
				},
				DecodePods: []types.ScoredPod{
					newScoredPod(testG2D1Name, testG2D1IP, testG2D1Score),
				},
				LeaderPod:          newScoredPod(testG2LName, testG2LIP, testG2LScore),
				SelectedPrefillPod: newScoredPod(testG2SelPName, testG2SelPIP, testG2SelPScore),
				SelectedDecodePod:  newScoredPod(testG2SelDName, testG2SelDIP, testG2SelDScore),
			},
		},
	}

	cloned := pdGroupList.Clone()

	clonedList, ok := cloned.(*PDGroupList)
	if !ok {
		t.Fatal("expected cloned result to be *PDGroupList")
	}
	if len(clonedList.Groups) != testExpGrpCnt {
		t.Fatalf("expected %d groups, got %d", testExpGrpCnt, len(clonedList.Groups))
	}
	for i := range pdGroupList.Groups {
		verifyGroupClone(t, pdGroupList.Groups[i], clonedList.Groups[i])
	}
}

// TestCloneDeepCopy tests that Clone creates a deep copy.
const (
	testModifiedScore = 999.0
)

func TestCloneDeepCopy(t *testing.T) {
	pdGroupList := &PDGroupList{
		Groups: []PDGroup{
			{
				ID: testG1ID,
				PrefillPods: []types.ScoredPod{
					newScoredPod(testG1P1Name, testG1P1IP, testG1P1Score),
				},
				DecodePods: []types.ScoredPod{
					newScoredPod(testG1D1Name, testG1D1IP, testG1D1Score),
				},
				LeaderPod:          newScoredPod(testG1LName, testG1LIP, testG1LScore),
				SelectedPrefillPod: newScoredPod(testG1SelPName, testG1SelPIP, testG1SelPScore),
				SelectedDecodePod:  newScoredPod(testG1SelDName, testG1SelDIP, testG1SelDScore),
			},
		},
	}

	cloned := pdGroupList.Clone()
	clonedList, ok := cloned.(*PDGroupList)
	if !ok {
		t.Fatal("expected cloned result to be *PDGroupList")
	}

	pdGroupList.Groups[0].ID = testG2ID
	pdGroupList.Groups[0].PrefillPods[0].Score = testModifiedScore
	pdGroupList.Groups[0].DecodePods[0].Score = testModifiedScore
	pdGroupList.Groups[0].LeaderPod.Score = testModifiedScore
	pdGroupList.Groups[0].SelectedPrefillPod.Score = testModifiedScore
	pdGroupList.Groups[0].SelectedDecodePod.Score = testModifiedScore

	if clonedList.Groups[0].ID == testG2ID {
		t.Error("cloned group ID should not be affected by original modification")
	}
	if clonedList.Groups[0].PrefillPods[0].Score == testModifiedScore {
		t.Error("cloned prefill pod score should not be affected by original modification")
	}
	if clonedList.Groups[0].DecodePods[0].Score == testModifiedScore {
		t.Error("cloned decode pod score should not be affected by original modification")
	}
	if clonedList.Groups[0].LeaderPod.Score == testModifiedScore {
		t.Error("cloned leader pod score should not be affected by original modification")
	}
	if clonedList.Groups[0].SelectedPrefillPod.Score == testModifiedScore {
		t.Error("cloned selected prefill pod score should not be affected by original modification")
	}
	if clonedList.Groups[0].SelectedDecodePod.Score == testModifiedScore {
		t.Error("cloned selected decode pod score should not be affected by original modification")
	}
}

// Helper functions

func newScoredPod(name, address string, score float64) types.ScoredPod {
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

func verifyGroupClone(t *testing.T, original, cloned PDGroup) {
	if original.ID != cloned.ID {
		t.Errorf("expected cloned group ID %s, got %s", original.ID, cloned.ID)
	}
	if len(original.PrefillPods) != len(cloned.PrefillPods) {
		t.Errorf("expected %d prefill pods, got %d", len(original.PrefillPods), len(cloned.PrefillPods))
	}
	for i := range original.PrefillPods {
		if original.PrefillPods[i].Score != cloned.PrefillPods[i].Score {
			t.Errorf("prefill pod %d: expected score %f, got %f",
				i, original.PrefillPods[i].Score, cloned.PrefillPods[i].Score)
		}
	}
	if len(original.DecodePods) != len(cloned.DecodePods) {
		t.Errorf("expected %d decode pods, got %d", len(original.DecodePods), len(cloned.DecodePods))
	}
	for i := range original.DecodePods {
		if original.DecodePods[i].Score != cloned.DecodePods[i].Score {
			t.Errorf("decode pod %d: expected score %f, got %f",
				i, original.DecodePods[i].Score, cloned.DecodePods[i].Score)
		}
	}
	if original.LeaderPod.Score != cloned.LeaderPod.Score {
		t.Errorf("expected leader pod score %f, got %f", original.LeaderPod.Score, cloned.LeaderPod.Score)
	}
	if original.SelectedPrefillPod.Score != cloned.SelectedPrefillPod.Score {
		t.Errorf("expected selected prefill pod score %f, got %f",
			original.SelectedPrefillPod.Score, cloned.SelectedPrefillPod.Score)
	}
	if original.SelectedDecodePod.Score != cloned.SelectedDecodePod.Score {
		t.Errorf("expected selected decode pod score %f, got %f",
			original.SelectedDecodePod.Score, cloned.SelectedDecodePod.Score)
	}
}
