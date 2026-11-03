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

package pdgroup

import (
	"math/rand"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestHasPDLeader(t *testing.T) {
	plain := newPDCandidateTestEndpoint("plain")
	if HasPDLeader([]fwkscheduling.Endpoint{plain}) {
		t.Fatal("expected endpoint without PD metadata not to be treated as PD leader")
	}

	leader := newPDCandidateTestEndpoint("leader")
	Set(leader, newPDCandidateInfo("group-1", "prefill-1", "decode-1"))
	if !HasPDLeader([]fwkscheduling.Endpoint{plain, leader}) {
		t.Fatal("expected endpoint with PD metadata to be treated as PD leader")
	}
}

func TestIsScoredPDMode(t *testing.T) {
	plain := &fwkscheduling.ScoredEndpoint{Endpoint: newPDCandidateTestEndpoint("plain"), Score: 0.1}
	if IsScoredPDMode([]*fwkscheduling.ScoredEndpoint{plain}) {
		t.Fatal("expected plain scored endpoints not to be treated as PD mode")
	}

	leader := newPDCandidateTestEndpoint("leader")
	Set(leader, newPDCandidateInfo("group-1", "prefill-1", "decode-1"))
	if !IsScoredPDMode([]*fwkscheduling.ScoredEndpoint{{Endpoint: leader, Score: 0.2}}) {
		t.Fatal("expected PD leader scored endpoints to be treated as PD mode")
	}
}

func TestPreparePDCandidatesFillsSelectionsAndFiltersInvalidLeaders(t *testing.T) {
	nonPD := &fwkscheduling.ScoredEndpoint{Endpoint: newPDCandidateTestEndpoint("plain"), Score: 0.1}
	invalidLeader := newPDCandidateTestEndpoint("invalid")
	Set(invalidLeader, &Info{GroupID: "broken"})

	readyLeader := newPDCandidateTestEndpoint("ready")
	readyInfo := newPDCandidateInfo("group-ready", "prefill-ready", "decode-ready")
	Set(readyLeader, readyInfo)

	unselectedLeader := newPDCandidateTestEndpoint("pending")
	Set(unselectedLeader, &Info{
		GroupID:     "group-pending",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newPDCandidateTestEndpoint("prefill-pending"), Score: 0.3}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newPDCandidateTestEndpoint("decode-pending"), Score: 0.4}},
	})

	prepared := PreparePDCandidates([]*fwkscheduling.ScoredEndpoint{
		nonPD,
		{Endpoint: invalidLeader, Score: 0.2},
		{Endpoint: readyLeader, Score: 0.3},
		{Endpoint: unselectedLeader, Score: 0.4},
	}, rand.New(rand.NewSource(1)))

	if len(prepared) != 2 {
		t.Fatalf("expected 2 valid PD leaders, got %d", len(prepared))
	}
	if !fwkscheduling.EndpointComparer(prepared[0].Endpoint, readyLeader) ||
		!fwkscheduling.EndpointComparer(prepared[1].Endpoint, unselectedLeader) {
		t.Fatal("expected prepared leaders to preserve valid ready and pending leaders")
	}

	readyPrepared, ok := Get(readyLeader)
	if !ok {
		t.Fatal("expected ready leader to keep PD metadata")
	}
	if readyPrepared.SelectedPrefillPod == nil ||
		!fwkscheduling.EndpointComparer(readyPrepared.SelectedPrefillPod.Endpoint, readyInfo.SelectedPrefillPod.Endpoint) {
		t.Fatal("expected ready leader to preserve selected prefill pod")
	}
	if readyPrepared.SelectedDecodePod == nil ||
		!fwkscheduling.EndpointComparer(readyPrepared.SelectedDecodePod.Endpoint, readyInfo.SelectedDecodePod.Endpoint) {
		t.Fatal("expected ready leader to preserve selected decode pod")
	}

	pendingPrepared, ok := Get(unselectedLeader)
	if !ok {
		t.Fatal("expected pending leader to keep PD metadata")
	}
	if pendingPrepared.SelectedPrefillPod == nil || pendingPrepared.SelectedPrefillPod.Endpoint == nil {
		t.Fatal("expected pending leader to receive selected prefill pod")
	}
	if pendingPrepared.SelectedDecodePod == nil || pendingPrepared.SelectedDecodePod.Endpoint == nil {
		t.Fatal("expected pending leader to receive selected decode pod")
	}
	if !ContainsMember(pendingPrepared, pendingPrepared.SelectedPrefillPod.Endpoint) {
		t.Fatal("expected selected prefill pod to come from PD group members")
	}
	if !ContainsMember(pendingPrepared, pendingPrepared.SelectedDecodePod.Endpoint) {
		t.Fatal("expected selected decode pod to come from PD group members")
	}
}

func newPDCandidateTestEndpoint(name string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

func newPDCandidateInfo(groupID, prefillName, decodeName string) *Info {
	prefill := fwkscheduling.ScoredEndpoint{Endpoint: newPDCandidateTestEndpoint(prefillName), Score: 0.1}
	decode := fwkscheduling.ScoredEndpoint{Endpoint: newPDCandidateTestEndpoint(decodeName), Score: 0.2}
	return &Info{
		GroupID:            groupID,
		PrefillPods:        []fwkscheduling.ScoredEndpoint{prefill},
		DecodePods:         []fwkscheduling.ScoredEndpoint{decode},
		SelectedPrefillPod: &prefill,
		SelectedDecodePod:  &decode,
	}
}
