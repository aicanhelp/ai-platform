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

package decodekvaware

import (
	"context"
	"encoding/json"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestFactoryRejectsInvalidWeights(t *testing.T) {
	_, err := Factory(
		"decode-kv",
		json.RawMessage(`{"prefixWeight":0,"runningWeight":0,"kvUsageWeight":0,"queueWeight":0,"inflightWeight":0}`),
		nil,
	)
	if err == nil {
		t.Fatal("expected validation error")
	}
}

func TestFactoryDefaultsAndTypedName(t *testing.T) {
	plugin, err := Factory("decode-kv", json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	scorer, ok := plugin.(*DecodeKVCacheAwareScorer)
	if !ok {
		t.Fatalf("unexpected type %T", plugin)
	}
	if got := scorer.TypedName(); got.Type != PluginType || got.Name != "decode-kv" {
		t.Fatalf("TypedName=%+v, want type=%q name=decode-kv", got, PluginType)
	}
	if scorer.Category() != fwkscheduling.Balance {
		t.Fatalf("Category=%v, want Balance", scorer.Category())
	}
}

func TestScoreSelectsBestDecodeOnly(t *testing.T) {
	leader := newTestEndpoint("leader", 0, 0, 0)
	prefill := newTestEndpoint("prefill", 5, 70, 2)
	decodeFast := newTestEndpoint("decode-fast", 1, 10, 1)
	decodeSlow := newTestEndpoint("decode-slow", 6, 80, 8)
	decodeFast.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))
	decodeSlow.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(1, 10, 16))
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeFast, Score: 0},
			{Endpoint: decodeSlow, Score: 0},
		},
	})

	scorer := New("decode-kv", Config{})
	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})
	if len(scores) != 1 {
		t.Fatalf("expected one scored endpoint, got %d", len(scores))
	}
	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info after scoring")
	}
	if info.SelectedPrefillPod != nil {
		t.Fatal("decode scorer must not select prefill pod")
	}
	if info.SelectedDecodePod == nil || !fwkscheduling.EndpointComparer(info.SelectedDecodePod.Endpoint, decodeFast) {
		t.Fatal("expected fastest decode pod to be selected")
	}
}

func newTestEndpoint(name string, waiting int, kvUsage float64, running int) fwkscheduling.Endpoint {
	return testutil.NewScoredEndpoint(
		name,
		name+".svc.cluster.local",
		"8080",
		testutil.ScoredEndpointMetrics{Score: 0, Waiting: waiting, KVUsage: kvUsage, Running: running},
		nil,
	).Endpoint
}
