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

package autokvaware

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestFactoryAcceptsKVUsageWeightNames(t *testing.T) {
	plugin, err := Factory("auto-kv", json.RawMessage(`{
		"prefillKVUsageWeight":1,
		"prefillPrefixWeight":1,
		"prefillQueueWeight":1,
		"prefillInflightWeight":1,
		"decodeKVUsageWeight":1,
		"decodeQueueWeight":1,
		"decodeInflightWeight":1,
		"prefillScoreWeight":1,
		"decodeScoreWeight":1
	}`), nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	scorer, ok := plugin.(*AutoKVCacheAwareScorer)
	if !ok {
		t.Fatalf("expected *AutoKVCacheAwareScorer, got %T", plugin)
	}
	if scorer.config.Prefill.KVWeight != 1 || scorer.config.Decode.KVWeight != 1 {
		t.Fatalf("expected KV usage weights to be applied, got %+v", scorer.config)
	}
}

func TestScoreHandlesAggregateAndPDSeparatedEndpoints(t *testing.T) {
	aggregate := newTestEndpoint("aggregate", 1, 20, 0)
	aggregate.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(8, 10, 16))

	leader := newTestEndpoint("leader", 0, 0, 0)
	prefillFast := newTestEndpoint("prefill-fast", 1, 10, 0)
	prefillSlow := newTestEndpoint("prefill-slow", 4, 60, 0)
	decodeFast := newTestEndpoint("decode-fast", 1, 10, 1)
	decodeSlow := newTestEndpoint("decode-slow", 4, 60, 5)
	prefillFast.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))
	prefillSlow.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(1, 10, 16))
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID: "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillFast, Score: 0},
			{Endpoint: prefillSlow, Score: 0},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFast, Score: 0},
			{Endpoint: decodeSlow, Score: 0},
		},
	})

	scorer := New("auto-kv", Config{})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{aggregate, leader})
	if len(scores) != 2 {
		t.Fatalf("expected aggregate and PD leader to be scored, got %d entries", len(scores))
	}
	if scores[aggregate] <= 0 {
		t.Fatalf("expected positive aggregate score, got %v", scores[aggregate])
	}
	if scores[leader] <= 0 {
		t.Fatalf("expected positive PD score, got %v", scores[leader])
	}
	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	if info.SelectedPrefillPod == nil || !fwkscheduling.EndpointComparer(info.SelectedPrefillPod.Endpoint, prefillFast) {
		t.Fatal("expected fastest prefill pod to be selected")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeFast) {
		t.Fatal("expected fastest decode pod to be selected")
	}
}

func TestScoreBypassesInvalidPDInfoInsteadOfDowngrading(t *testing.T) {
	aggregate := newTestEndpoint("aggregate", 0, 0, 0)
	aggregate.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(10, 10, 16))
	invalidLeader := newTestEndpoint("invalid", 0, 0, 0)
	pdgroup.Set(invalidLeader, &pdgroup.Info{GroupID: "broken"})

	scorer := New("auto-kv", Config{})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{aggregate, invalidLeader})
	if len(scores) != 1 {
		t.Fatalf("expected only aggregate endpoint to be scored, got %d entries", len(scores))
	}
	if _, ok := scores[invalidLeader]; ok {
		t.Fatal("expected invalid PD endpoint to be bypassed")
	}
	testutil.AssertFloatClose(t, scores[aggregate], 1.0)
}

func TestScoreRandomizesEqualPDMembers(t *testing.T) {
	leader := newTestEndpoint("leader-equal", 0, 0, 0)
	prefillFirst := newTestEndpoint("prefill-first", 2, 20, 1)
	prefillSecond := newTestEndpoint("prefill-second", 2, 20, 1)
	decodeFirst := newTestEndpoint("decode-first", 3, 30, 4)
	decodeSecond := newTestEndpoint("decode-second", 3, 30, 4)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID: "group-equal",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillFirst, Score: 0},
			{Endpoint: prefillSecond, Score: 0},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFirst, Score: 0},
			{Endpoint: decodeSecond, Score: 0},
		},
	})

	scorer := New("auto-kv", Config{})
	scorer.rng = rand.New(rand.NewSource(1))
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})
	if len(scores) != 1 {
		t.Fatalf("expected one scored endpoint, got %d", len(scores))
	}

	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	if info.SelectedPrefillPod == nil || !fwkscheduling.EndpointComparer(info.SelectedPrefillPod.Endpoint, prefillSecond) {
		t.Fatal("expected second tied prefill pod to be selected")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeSecond) {
		t.Fatal("expected second tied decode pod to be selected")
	}
}

func newTestEndpoint(name string, waiting int, kvUsage float64, running int) fwkscheduling.Endpoint {
	return testutil.NewScoredEndpoint(name, name+".svc.cluster.local", "8080", testutil.ScoredEndpointMetrics{Score: 0, Waiting: waiting, KVUsage: kvUsage, Running: running}, nil).Endpoint
}
