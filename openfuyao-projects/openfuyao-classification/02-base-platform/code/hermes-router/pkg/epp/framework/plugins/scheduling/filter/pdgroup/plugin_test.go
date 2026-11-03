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
	"context"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	pdg "hermes-router/pkg/epp/internal/pdgroup"
)

func TestFilterKeepsOnlySelectedGroupLeadersWhenValidGroupsExist(t *testing.T) {
	filter := New("pd-group-filter")
	leader := newTestEndpoint("leader-a")
	otherLeader := newTestEndpoint("leader-b")
	prefill := newTestEndpoint("prefill-a")
	decode := newTestEndpoint("decode-a")
	aggregate := newTestEndpoint("aggregate-a")
	pdg.Set(leader, &pdg.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
	})

	got := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader, otherLeader, prefill, decode, aggregate})
	if len(got) != 1 {
		t.Fatalf("expected only the selected leader to remain, got %d endpoints", len(got))
	}
	if !fwkscheduling.EndpointComparer(got[0], leader) {
		t.Fatal("expected selected leader to remain")
	}
}

func TestFilterDropsUnhealthyLeaderAndPDMembers(t *testing.T) {
	filter := New("pd-group-filter")
	leader := newTestEndpoint("leader-a")
	prefill := newTestEndpoint("prefill-a")
	pdg.Set(leader, &pdg.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
	})

	got := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader, prefill})
	if len(got) != 0 {
		t.Fatalf("expected unhealthy group endpoints to be removed, got %d endpoints", len(got))
	}
}

func TestFilterPassesThroughAggregateOnlyPools(t *testing.T) {
	filter := New("pd-group-filter")
	a := newTestEndpoint("aggregate-a")
	b := newTestEndpoint("aggregate-b")

	got := filter.Filter(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{a, b})
	if len(got) != 2 {
		t.Fatalf("expected aggregate-only pool to pass through, got %d endpoints", len(got))
	}
	if !fwkscheduling.EndpointComparer(got[0], a) || !fwkscheduling.EndpointComparer(got[1], b) {
		t.Fatal("expected aggregate endpoints to preserve order")
	}
}

func newTestEndpoint(name string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
		Labels:         map[string]string{},
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}
