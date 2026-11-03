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

package bucket

import (
	"context"
	"encoding/json"
	"math/rand"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/endpointmeta"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestPDBucketFactoryAndTypedName(t *testing.T) {
	plugin, err := PDBucketFactory(
		"pd-bucket",
		json.RawMessage(`{"alpha":0.5,"beta":0.5,"decayFactor":1.0,"bucketSeparateLength":10,"tpSizeLabelKey":"tp-size"}`),
		testutil.NewHandle(),
	)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	bucket, ok := plugin.(*PDBucket)
	if !ok {
		t.Fatalf("expected *PDBucket, got %T", plugin)
	}
	bucket.WithName("renamed")
	if got := bucket.TypedName(); got.Type != PDBucketType || got.Name != "renamed" {
		t.Fatalf("unexpected typed name: %+v", got)
	}
	if bucket.tpSizeLabelKey != "tp-size" {
		t.Fatalf("expected tp-size label key, got %q", bucket.tpSizeLabelKey)
	}
}

func TestPDBucketFactoryRejectsMissingParams(t *testing.T) {
	plugin, err := PDBucketFactory("pd-bucket", nil, testutil.NewHandle())
	if err == nil {
		t.Fatal("expected missing parameters error")
	}
	if plugin != nil {
		t.Fatalf("expected nil plugin on invalid params, got %T", plugin)
	}
}

func TestNewPDBucketRejectsInvalidParams(t *testing.T) {
	if bucket := NewPDBucket(context.Background(), nil); bucket != nil {
		t.Fatal("expected nil bucket for nil params")
	}
}

func TestPDBucketReadsInflightLoadAndTPLabels(t *testing.T) {
	bucket := NewPDBucket(context.Background(), &scorerPDBucketParameters{
		Alpha:                0.5,
		Beta:                 0.5,
		DecayFactor:          1.0,
		BucketSeparateLength: 512,
		TPSizeLabelKey:       "tp-size",
	})
	prefillLabeled := testutil.NewScoredEndpoint("prefill-labeled", "10.0.21.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "4"})
	prefillFallback := testutil.NewScoredEndpoint("prefill-fallback", "10.0.21.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "bad"})
	decodeFast := testutil.NewScoredEndpoint("decode-fast", "10.0.21.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	decodeSlow := testutil.NewScoredEndpoint("decode-slow", "10.0.21.4", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	setInflightLoad(prefillLabeled.Endpoint, 30)
	setInflightLoad(prefillFallback.Endpoint, 20)
	setInflightLoad(decodeFast.Endpoint, 10)
	setInflightLoad(decodeSlow.Endpoint, 20)

	load, tpSum := bucket.calculateSingleGroupMetrics(&pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillLabeled, prefillFallback},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeFast, decodeSlow},
	})

	if load != 40 {
		t.Fatalf("expected averaged group load 40, got %v", load)
	}
	if tpSum != 5 {
		t.Fatalf("expected tp sum 5, got %v", tpSum)
	}
}

func TestPDBucketScoreSelectsLowestInflightMembers(t *testing.T) {
	bucket := NewPDBucket(context.Background(), &scorerPDBucketParameters{
		Alpha:                1.0,
		Beta:                 0.1,
		DecayFactor:          1.0,
		BucketSeparateLength: 512,
		TPSizeLabelKey:       "tp-size",
	})
	leaderA := testutil.NewScoredEndpoint("leader-a", "10.0.22.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	leaderB := testutil.NewScoredEndpoint("leader-b", "10.0.23.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	prefillA1 := testutil.NewScoredEndpoint("prefill-a-1", "10.0.22.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "2"})
	prefillA2 := testutil.NewScoredEndpoint("prefill-a-2", "10.0.22.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "2"})
	decodeA1 := testutil.NewScoredEndpoint("decode-a-1", "10.0.22.4", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	decodeA2 := testutil.NewScoredEndpoint("decode-a-2", "10.0.22.5", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	prefillB := testutil.NewScoredEndpoint("prefill-b", "10.0.23.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "2"})
	decodeB := testutil.NewScoredEndpoint("decode-b", "10.0.23.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	setInflightLoad(prefillA1.Endpoint, 50)
	setInflightLoad(prefillA2.Endpoint, 60)
	setInflightLoad(decodeA1.Endpoint, 40)
	setInflightLoad(decodeA2.Endpoint, 30)
	setInflightLoad(prefillB.Endpoint, 5)
	setInflightLoad(decodeB.Endpoint, 10)
	pdgroup.Set(leaderA, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillA1, prefillA2},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeA1, decodeA2},
	})
	pdgroup.Set(leaderB, &pdgroup.Info{
		GroupID:     "group-b",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillB},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeB},
	})

	scores := bucket.Score(
		context.Background(),
		fwkscheduling.NewCycleState(),
		testutil.NewCompletionsRequest("req-1", "model", "hello hello"),
		[]fwkscheduling.Endpoint{leaderA, leaderB},
	)

	if len(scores) != 2 {
		t.Fatalf("expected 2 scored leaders, got %d", len(scores))
	}
	if scores[leaderB] >= scores[leaderA] {
		t.Fatalf("expected lower-load leader-b score to be lower than leader-a: got leader-b=%v leader-a=%v", scores[leaderB], scores[leaderA])
	}

	infoA, ok := pdgroup.Get(leaderA)
	if !ok {
		t.Fatal("expected scored Info on leader-a")
	}
	if infoA.SelectedPrefillPod == nil || endpointmeta.EndpointPodName(infoA.SelectedPrefillPod.Endpoint) != "prefill-a-1" {
		t.Fatalf("expected prefill-a-1 to be selected, got %+v", infoA.SelectedPrefillPod)
	}
	if infoA.SelectedDecodePod == nil || endpointmeta.EndpointPodName(infoA.SelectedDecodePod.Endpoint) != "decode-a-2" {
		t.Fatalf("expected decode-a-2 to be selected, got %+v", infoA.SelectedDecodePod)
	}

	infoB, ok := pdgroup.Get(leaderB)
	if !ok {
		t.Fatal("expected scored Info on leader-b")
	}
	if infoB.SelectedPrefillPod == nil || endpointmeta.EndpointPodName(infoB.SelectedPrefillPod.Endpoint) != "prefill-b" {
		t.Fatalf("expected prefill-b to be selected, got %+v", infoB.SelectedPrefillPod)
	}
	if infoB.SelectedDecodePod == nil || endpointmeta.EndpointPodName(infoB.SelectedDecodePod.Endpoint) != "decode-b" {
		t.Fatalf("expected decode-b to be selected, got %+v", infoB.SelectedDecodePod)
	}
}

func TestPDBucketUsesTPLabelWhenLoadsAreBalanced(t *testing.T) {
	bucket := NewPDBucket(context.Background(), &scorerPDBucketParameters{
		Alpha:                1.0,
		Beta:                 1.0,
		DecayFactor:          1.0,
		BucketSeparateLength: 512,
		TPSizeLabelKey:       "tp-size",
	})
	leaderSmallTP := testutil.NewScoredEndpoint("leader-small", "10.0.24.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	leaderLargeTP := testutil.NewScoredEndpoint("leader-large", "10.0.25.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	prefillSmallTP := testutil.NewScoredEndpoint("prefill-small", "10.0.24.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "1"})
	decodeSmallTP := testutil.NewScoredEndpoint("decode-small", "10.0.24.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	prefillLargeTP := testutil.NewScoredEndpoint("prefill-large", "10.0.25.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, map[string]string{"tp-size": "4"})
	decodeLargeTP := testutil.NewScoredEndpoint("decode-large", "10.0.25.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	setInflightLoad(prefillSmallTP.Endpoint, 10)
	setInflightLoad(decodeSmallTP.Endpoint, 10)
	setInflightLoad(prefillLargeTP.Endpoint, 10)
	setInflightLoad(decodeLargeTP.Endpoint, 10)
	pdgroup.Set(leaderSmallTP, &pdgroup.Info{
		GroupID:     "group-small",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillSmallTP},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeSmallTP},
	})
	pdgroup.Set(leaderLargeTP, &pdgroup.Info{
		GroupID:     "group-large",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefillLargeTP},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decodeLargeTP},
	})

	scores := bucket.Score(
		context.Background(),
		fwkscheduling.NewCycleState(),
		testutil.NewCompletionsRequest("req-2", "model", "hello"),
		[]fwkscheduling.Endpoint{leaderSmallTP, leaderLargeTP},
	)

	if scores[leaderSmallTP] >= scores[leaderLargeTP] {
		t.Fatalf("expected smaller TP group to receive a lower score for short requests: got small=%v large=%v", scores[leaderSmallTP], scores[leaderLargeTP])
	}
}

func TestPDBucketScoreNoOpsWhenInputsAreNotAllPDLeaders(t *testing.T) {
	bucket := NewPDBucket(context.Background(), &scorerPDBucketParameters{
		Alpha:                1.0,
		Beta:                 0.5,
		DecayFactor:          1.0,
		BucketSeparateLength: 256,
	})
	leader := testutil.NewScoredEndpoint("leader", "10.0.26.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	aggregate := testutil.NewScoredEndpoint("aggregate", "10.0.26.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil).Endpoint
	prefill := testutil.NewScoredEndpoint("prefill", "10.0.26.3", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	decode := testutil.NewScoredEndpoint("decode", "10.0.26.4", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	setInflightLoad(prefill.Endpoint, 10)
	setInflightLoad(decode.Endpoint, 5)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{prefill},
		DecodePods:  []fwkscheduling.ScoredEndpoint{decode},
	})

	scores := bucket.Score(
		context.Background(),
		fwkscheduling.NewCycleState(),
		testutil.NewCompletionsRequest("req-mixed", "model", "hello world"),
		[]fwkscheduling.Endpoint{leader, aggregate},
	)

	if len(scores) != 0 {
		t.Fatalf("expected PD bucket scorer to no-op on mixed PD and aggregate inputs, got %d scores", len(scores))
	}
	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected leader Info to remain stored")
	}
	if info.SelectedPrefillPod != nil || info.SelectedDecodePod != nil {
		t.Fatalf("expected no PD selections on fail-open path, got prefill=%+v decode=%+v", info.SelectedPrefillPod, info.SelectedDecodePod)
	}
}

func TestPickMinInflightPodRandomizesEqualInflight(t *testing.T) {
	bucket := NewPDBucket(context.Background(), &scorerPDBucketParameters{
		Alpha:                1.0,
		Beta:                 0.5,
		DecayFactor:          1.0,
		BucketSeparateLength: 256,
	})
	bucket.rng = rand.New(rand.NewSource(1))
	first := testutil.NewScoredEndpoint("prefill-first", "10.0.27.1", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	second := testutil.NewScoredEndpoint("prefill-second", "10.0.27.2", testutil.TestInferencePort, testutil.ScoredEndpointMetrics{Score: 0, Waiting: 0, KVUsage: 0, Running: 0}, nil)
	setInflightLoad(first.Endpoint, 11)
	setInflightLoad(second.Endpoint, 11)

	selected, tokens := bucket.pickMinInflightPod([]fwkscheduling.ScoredEndpoint{first, second})

	if selected == nil || selected.Endpoint == nil {
		t.Fatal("expected selected endpoint for equal inflight tie")
	}
	if got := endpointmeta.EndpointPodName(selected.Endpoint); got != "prefill-second" {
		t.Fatalf("expected second tied inflight endpoint, got %q", got)
	}
	if tokens != 11 {
		t.Fatalf("expected inflight token count 11, got %v", tokens)
	}
}

func setInflightLoad(endpoint fwkscheduling.Endpoint, tokens int) {
	endpoint.Put(internalinflight.DefaultAttributeKey, &internalinflight.Load{Tokens: tokens, Requests: 1})
}
