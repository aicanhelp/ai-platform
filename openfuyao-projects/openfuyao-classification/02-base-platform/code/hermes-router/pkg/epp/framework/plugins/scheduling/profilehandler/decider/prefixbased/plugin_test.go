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

package prefixbased

import (
	"context"
	"encoding/json"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/testutil"
)

func TestDecideDPathWhenThresholdsPass(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(165), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if shouldRun {
		t.Fatal("expected d path (shouldRunPrefill=false)")
	}
}

func TestDecidePDPathWhenHitRatioTooLow(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(7, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(160), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when hit ratio is below threshold")
	}
}

func TestDecidePDPathWhenHitRatioEqualsMin(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(8, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(160), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when hit ratio equals MinHitRatio")
	}
}

func TestDecidePDPathWhenUncachedTooHigh(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 16,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(165), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when uncached tokens meet/exceed threshold")
	}
}

func TestDecideCountsPartialTrailingBlock(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 20,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(165), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when trailing partial-block tokens push uncached over threshold")
	}
}

func TestDecidePDPathWhenCacheMissing(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)

	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(160), decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when PrefixCacheMatchInfo is missing")
	}
}

func TestDecidePDPathWhenRequestTokensMissing(t *testing.T) {
	decider := New("prefix", Config{
		MinHitRatio:       0.8,
		MaxUncachedTokens: 256,
	})
	decode := testutil.NewEndpoint("decode-a", "10.0.0.1", "8000", nil)
	decode.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(9, 10, 16))

	shouldRun, err := decider.Decide(context.Background(), nil, decode)
	if err != nil {
		t.Fatalf("Decide error: %v", err)
	}
	if !shouldRun {
		t.Fatal("expected pd path when request token count is missing")
	}
}

func TestFactoryDefaults(t *testing.T) {
	plugin, err := Factory("prefix", json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatalf("Factory: %v", err)
	}
	decider, ok := plugin.(*Decider)
	if !ok {
		t.Fatalf("unexpected plugin type %T", plugin)
	}
	if got := decider.TypedName(); got.Type != PluginType || got.Name != "prefix" {
		t.Fatalf("TypedName=%+v, want type=%q name=prefix", got, PluginType)
	}
	if decider.config.MinHitRatio != DefaultMinHitRatio {
		t.Fatalf("MinHitRatio=%v, want %v", decider.config.MinHitRatio, DefaultMinHitRatio)
	}
	if decider.config.MaxUncachedTokens != DefaultMaxUncachedTokens {
		t.Fatalf("MaxUncachedTokens=%v, want %v", decider.config.MaxUncachedTokens, DefaultMaxUncachedTokens)
	}
}

func TestFactoryRejectsInvalidParameters(t *testing.T) {
	if _, err := Factory("prefix", json.RawMessage(`{"minHitRatio":1.5}`), nil); err == nil {
		t.Fatal("expected minHitRatio validation error")
	}
	if _, err := Factory("prefix", json.RawMessage(`{"maxUncachedTokens":-1}`), nil); err == nil {
		t.Fatal("expected maxUncachedTokens validation error")
	}
	if _, err := Factory("prefix", json.RawMessage(`{"unexpected":1}`), nil); err == nil {
		t.Fatal("expected unknown-field error")
	}
}

func TestDecideNilSelectedDecode(t *testing.T) {
	decider := New("prefix", Config{MinHitRatio: 0.8, MaxUncachedTokens: 256})
	shouldRun, err := decider.Decide(context.Background(), requestWithTokens(16), nil)
	if err == nil {
		t.Fatal("expected error for nil selected decode")
	}
	if !shouldRun {
		t.Fatal("nil decode should still request Prefill")
	}
}

func requestWithTokens(count int) *fwkscheduling.LLMRequest {
	tokenIDs := make([]uint32, count)
	for i := range tokenIDs {
		tokenIDs[i] = uint32(i + 1)
	}
	return &fwkscheduling.LLMRequest{
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: tokenIDs},
	}
}
