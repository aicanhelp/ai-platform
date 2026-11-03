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

package prediction

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"math/rand"
	"strings"
	"testing"
	"time"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkplugin "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/predictiondata"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestScoreUsesPredictedLatencyWhenAvailable(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 3, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	fast := newTestEndpoint("fast", "10.0.70.2")
	slow := newTestEndpoint("slow", "10.0.70.3")
	setPrediction(fast, 1, 1)
	setPrediction(slow, 3, 1)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{fast, slow})

	// Saturating latency badness: lat/(lat+1000). Fast: ttft=tpot=1ms → both
	// 1/1001. Slow: ttft=3ms → 3/1003, tpot=1ms → 1/1001.
	testutil.AssertFloatClose(t, scores[fast], 1.0-1.0/1001.0)
	testutil.AssertFloatClose(t, scores[slow], 1.0-(3.0*(3.0/1003.0)+1.0/1001.0)/4.0)
	if scores[fast] <= scores[slow] {
		t.Fatalf("expected fast endpoint score %v to be greater than slow score %v", scores[fast], scores[slow])
	}
}

func TestScoreDispatchesPredictionsBeforeScoring(t *testing.T) {
	client := &mockPredictorClient{
		outputs: []PredictOutput{
			{InputID: 0, PredictedTTFTMillis: 1, PredictedTPOTMillis: 1},
			{InputID: 1, PredictedTTFTMillis: 3, PredictedTPOTMillis: 1},
		},
	}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      3,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	fast := newTestEndpointWithInput("fast-dispatch", "10.0.70.12", predictiondata.Input{ServerIP: "10.0.70.12"})
	slow := newTestEndpointWithInput("slow-dispatch", "10.0.70.13", predictiondata.Input{ServerIP: "10.0.70.13"})

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{fast, slow})

	if client.calls != 1 {
		t.Fatalf("expected one prediction call, got %d", client.calls)
	}
	assertPrediction(t, fast, 1, 1)
	assertPrediction(t, slow, 3, 1)
	if scores[fast] <= scores[slow] {
		t.Fatalf("expected dispatched prediction to make fast endpoint win, got fast=%v slow=%v", scores[fast], scores[slow])
	}
}

func TestScoreDispatchFailsOpenOnPredictError(t *testing.T) {
	client := &mockPredictorClient{err: errors.New("sidecar unavailable")}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	endpoint := newTestEndpointWithInput("error-dispatch", "10.0.70.14", predictiondata.Input{ServerIP: "10.0.70.14"})
	setMetrics(endpoint, 0, 0)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{endpoint})

	if _, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey); ok {
		t.Fatal("expected no prediction attribute when predictor fails")
	}
	if got := scores[endpoint]; got == 0 {
		t.Fatalf("expected fallback score after predictor failure, got %v", got)
	}
}

func TestScoreDispatchFailsOpenOnTimeout(t *testing.T) {
	client := &mockPredictorClient{delay: time.Second}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         5 * time.Millisecond,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	endpoint := newTestEndpointWithInput("timeout-dispatch", "10.0.70.15", predictiondata.Input{ServerIP: "10.0.70.15"})

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{endpoint})

	if _, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey); ok {
		t.Fatal("expected no prediction attribute when predictor times out")
	}
}

func TestScoreDispatchSkipsPredictWhenTargetModelMissing(t *testing.T) {
	client := &mockPredictorClient{}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	endpoint := newTestEndpointWithInput("missing-model", "10.0.70.16", predictiondata.Input{ServerIP: "10.0.70.16"})

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{}, []fwkscheduling.Endpoint{endpoint})

	if client.calls != 0 {
		t.Fatalf("expected no prediction call without target model, got %d", client.calls)
	}
}

func TestScoreDispatchSkipsPredictWhenFeaturesMissing(t *testing.T) {
	client := &mockPredictorClient{}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	endpoint := newTestEndpoint("missing-feature", "10.0.70.17")

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{endpoint})

	if client.calls != 0 {
		t.Fatalf("expected no prediction call without feature input, got %d", client.calls)
	}
}

func TestScoreLogsFailOpenReasonWhenFeatureSnapshotMissing(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	client := &mockPredictorClient{
		outputs: []PredictOutput{{InputID: 0, PredictedTTFTMillis: 1, PredictedTPOTMillis: 1}},
	}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	withInput := newTestEndpointWithInput("with-input", "10.0.70.18", predictiondata.Input{ServerIP: "10.0.70.18"})
	withoutInput := newTestEndpoint("without-input", "10.0.70.19")
	setMetrics(withInput, 0, 0)
	setMetrics(withoutInput, 0, 0)

	scorer.Score(ctx, fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{withInput, withoutInput})

	entry := sink.LastInfo("Prediction scorer skipped endpoints with missing or invalid feature snapshots; affected endpoints will fail open to fallback scoring")
	if entry.Message == "" {
		t.Fatal("expected fail-open log for missing feature snapshots")
	}
	if got := entry.Keys["skippedEndpoints"]; got != 1 {
		t.Fatalf("expected one skipped endpoint in log, got %#v", got)
	}
	if got := entry.Keys["collectedInputs"]; got != 1 {
		t.Fatalf("expected one collected input in log, got %#v", got)
	}
}

func TestScoreLogsFailOpenReasonWhenPredictorOutputInvalid(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	client := &mockPredictorClient{
		outputs: []PredictOutput{{InputID: 0, PredictedTTFTMillis: math.NaN(), PredictedTPOTMillis: 1}},
	}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	endpoint := newTestEndpointWithInput("invalid-output", "10.0.70.20", predictiondata.Input{ServerIP: "10.0.70.20"})
	setMetrics(endpoint, 0, 0)

	scores := scorer.Score(ctx, fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{endpoint})

	if got := scores[endpoint]; got == 0 {
		t.Fatalf("expected fallback score after invalid predictor output, got %v", got)
	}
	entry := sink.LastInfo("Prediction scorer ignored predictor outputs that did not match the expected schema; affected endpoints will fail open to fallback scoring")
	if entry.Message == "" {
		t.Fatal("expected fail-open log for invalid predictor outputs")
	}
	if got := entry.Keys["invalidOutputs"]; got != 1 {
		t.Fatalf("expected one invalid output in log, got %#v", got)
	}
}

func TestScoreFallsBackToSnapshotComposite(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 0.6, TPOTWeight: 0.4, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	good := newTestEndpoint("good", "10.0.71.2")
	bad := newTestEndpoint("bad", "10.0.71.3")
	setMetrics(good, 1, 0.2)
	setMetrics(bad, 4, 0.8)
	setPrefixScore(good, 3, 4)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{good, bad})

	// Fallback path averages KV, Queue, Prefix (Inflight weight 0 → skipped).
	// good: (0.2 + 0.25 + 0.25)/3 ≈ 0.2333 → score ≈ 0.7667
	// bad:  (0.8 + 1.0  + 1.0)/3  ≈ 0.9333 → score ≈ 0.0667
	testutil.AssertFloatClose(t, scores[good], 1.0-(0.2+0.25+0.25)/3.0)
	testutil.AssertFloatClose(t, scores[bad], 1.0-(0.8+1.0+1.0)/3.0)
	if scores[good] <= scores[bad] {
		t.Fatalf("expected good endpoint score %v to be greater than bad score %v", scores[good], scores[bad])
	}
}

func TestScoreFallsBackForAllAggregateEndpointsWhenAnyPredictionIsMissing(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 1, TPOTWeight: 1, KVWeight: 2, QueueWeight: 1, PrefixWeight: 1})
	predicted := newTestEndpoint("predicted", "10.0.72.2")
	fallback := newTestEndpoint("fallback", "10.0.72.3")
	setPrediction(predicted, 2, 2)
	setMetrics(predicted, 1, 0.1)
	setPrefixScore(predicted, 1, 1)
	setMetrics(fallback, 5, 0.5)
	setPrefixScore(fallback, 1, 2)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{predicted, fallback})

	// Both pods drop to fallback because the fallback pod has no prediction.
	// predicted: (2·0.1 + 1·0.2 + 1·0)/4 = 0.1 → score 0.9
	// fallback:  (2·0.5 + 1·1.0 + 1·0.5)/4 = 0.625 → score 0.375
	testutil.AssertFloatClose(t, scores[predicted], 1.0-0.1)
	testutil.AssertFloatClose(t, scores[fallback], 1.0-0.625)
	if len(scores) != 2 {
		t.Fatalf("expected scores for both endpoints, got %d", len(scores))
	}
	if scores[predicted] <= scores[fallback] {
		t.Fatalf("expected predicted endpoint fallback score %v to be greater than fallback score %v", scores[predicted], scores[fallback])
	}
}

func TestScoreUsesPredictionForAllAggregateEndpointsOnlyWhenAllPredictionsExist(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 0.6, TPOTWeight: 0.4, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	fast := newTestEndpoint("fast", "10.0.72.4")
	slow := newTestEndpoint("slow", "10.0.72.5")
	setPrediction(fast, 200, 50)
	setPrediction(slow, 400, 100)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{fast, slow})

	// Saturating latency badness:
	// fast: ttft=200/1200=1/6, tpot=50/1050=1/21 → (0.6·1/6 + 0.4·1/21)/1
	// slow: ttft=400/1400=2/7, tpot=100/1100=1/11 → (0.6·2/7 + 0.4·1/11)/1
	testutil.AssertFloatClose(t, scores[fast], 1.0-(0.6*(1.0/6.0)+0.4*(1.0/21.0)))
	testutil.AssertFloatClose(t, scores[slow], 1.0-(0.6*(2.0/7.0)+0.4*(1.0/11.0)))
	if scores[fast] <= scores[slow] {
		t.Fatalf("expected fast predicted endpoint score %v to be greater than slow score %v", scores[fast], scores[slow])
	}
}

func TestScoreFallbackTreatsAllZeroQueueAsZeroBadness(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 0.6, TPOTWeight: 0.4, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	endpoint := newTestEndpoint("candidate", "10.0.73.2")
	setMetrics(endpoint, 0, 0)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{endpoint})

	// Single cold pod with no queue or KV activity and no prefix info: kv=0,
	// queue=0 (pool max is 0 → 0/0 = 0 in RelativeValue), prefix=1.
	// Average over 3 active signals = 1/3 → score = 2/3.
	testutil.AssertFloatClose(t, scores[endpoint], 1.0-1.0/3.0)
}

func TestScoreLogsFailOpenReasonWhenMetricsMissing(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	scorer := New("prediction", Config{TTFTWeight: 1, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	endpoint := newTestEndpointWithoutMetrics("missing-metrics", "10.0.73.4")

	scores := scorer.Score(ctx, fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{endpoint})

	if got := scores[endpoint]; got == 0 {
		t.Fatalf("expected zero-metrics fail-open score, got %v", got)
	}
	entry := sink.LastInfo("Prediction scorer used zero-value metrics because endpoint metrics are missing")
	if entry.Message == "" {
		t.Fatal("expected fail-open log for missing metrics")
	}
	if got := entry.Keys["endpoint"]; got != "missing-metrics" {
		t.Fatalf("expected endpoint name in log, got %#v", got)
	}
}

func TestScoreReturnsOneForZeroLatency(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 1, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	endpoint := newTestEndpoint("zero", "10.0.73.3")
	setPrediction(endpoint, 0, 0)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{endpoint})

	testutil.AssertFloatClose(t, scores[endpoint], 1.0)
}

func TestScoreUsesPDPredictionsPerGroup(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 3, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	prefillA1 := scoredEndpoint(newTestEndpoint("prefill-a-1", "10.0.80.2"))
	prefillA2 := scoredEndpoint(newTestEndpoint("prefill-a-2", "10.0.80.3"))
	decodeA1 := scoredEndpoint(newTestEndpoint("decode-a-1", "10.0.80.4"))
	decodeA2 := scoredEndpoint(newTestEndpoint("decode-a-2", "10.0.80.5"))
	setPrediction(prefillA1.Endpoint, 10, 100)
	setPrediction(prefillA2.Endpoint, 5, 100)
	setPrediction(decodeA1.Endpoint, 100, 4)
	setPrediction(decodeA2.Endpoint, 100, 2)
	leaderA := newPDLeader("leader-a", "10.0.80.1", "group-a", []fwkscheduling.ScoredEndpoint{prefillA1, prefillA2}, []fwkscheduling.ScoredEndpoint{decodeA1, decodeA2})

	prefillB := scoredEndpoint(newTestEndpoint("prefill-b", "10.0.81.2"))
	decodeB := scoredEndpoint(newTestEndpoint("decode-b", "10.0.81.3"))
	setPrediction(prefillB.Endpoint, 20, 100)
	setPrediction(decodeB.Endpoint, 100, 4)
	leaderB := newPDLeader("leader-b", "10.0.81.1", "group-b", []fwkscheduling.ScoredEndpoint{prefillB}, []fwkscheduling.ScoredEndpoint{decodeB})

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leaderA, leaderB})

	// Best prefill / decode per group go through saturating latency badness.
	// groupA: prefillA2 ttft=5 → 5/1005, decodeA2 tpot=2 → 2/1002.
	//   group = (3·5/1005 + 1·2/1002)/4
	// groupB: prefillB ttft=20 → 20/1020, decodeB tpot=4 → 4/1004.
	//   group = (3·20/1020 + 1·4/1004)/4
	testutil.AssertFloatClose(t, scores[leaderA], 1.0-(3.0*(5.0/1005.0)+1.0*(2.0/1002.0))/4.0)
	testutil.AssertFloatClose(t, scores[leaderB], 1.0-(3.0*(20.0/1020.0)+1.0*(4.0/1004.0))/4.0)
	if scores[leaderA] <= scores[leaderB] {
		t.Fatalf("expected predicted group-a score %v to be greater than group-b score %v", scores[leaderA], scores[leaderB])
	}

	infoA, ok := pdgroup.Get(leaderA)
	if !ok {
		t.Fatal("expected PD group info on leader-a")
	}
	if infoA.SelectedPrefillPod == nil || infoA.SelectedPrefillPod.Endpoint.GetMetadata().PodName != "prefill-a-2" {
		t.Fatalf("expected prefill-a-2 to be selected, got %+v", infoA.SelectedPrefillPod)
	}
	if infoA.SelectedDecodePod == nil || infoA.SelectedDecodePod.Endpoint.GetMetadata().PodName != "decode-a-2" {
		t.Fatalf("expected decode-a-2 to be selected, got %+v", infoA.SelectedDecodePod)
	}
}

func TestScoreDispatchesPredictionsForPDGroupMembers(t *testing.T) {
	client := &mockPredictorClient{
		outputs: []PredictOutput{
			{InputID: 0, PredictedTTFTMillis: 10, PredictedTPOTMillis: 100},
			{InputID: 1, PredictedTTFTMillis: 5, PredictedTPOTMillis: 100},
			{InputID: 2, PredictedTTFTMillis: 100, PredictedTPOTMillis: 4},
			{InputID: 3, PredictedTTFTMillis: 100, PredictedTPOTMillis: 2},
		},
	}
	scorer := New("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      3,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	prefillA1 := scoredEndpoint(newTestEndpointWithInput("dispatch-prefill-a-1", "10.0.80.12", predictiondata.Input{PodType: predictiondata.PodTypePrefill}))
	prefillA2 := scoredEndpoint(newTestEndpointWithInput("dispatch-prefill-a-2", "10.0.80.13", predictiondata.Input{PodType: predictiondata.PodTypePrefill}))
	decodeA1 := scoredEndpoint(newTestEndpointWithInput("dispatch-decode-a-1", "10.0.80.14", predictiondata.Input{PodType: predictiondata.PodTypeDecode}))
	decodeA2 := scoredEndpoint(newTestEndpointWithInput("dispatch-decode-a-2", "10.0.80.15", predictiondata.Input{PodType: predictiondata.PodTypeDecode}))
	leader := newPDLeader(
		"dispatch-leader-a",
		"10.0.80.11",
		"group-dispatch-a",
		[]fwkscheduling.ScoredEndpoint{prefillA1, prefillA2},
		[]fwkscheduling.ScoredEndpoint{decodeA1, decodeA2},
	)

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), &fwkscheduling.LLMRequest{TargetModel: "test-model"}, []fwkscheduling.Endpoint{leader})

	if client.calls != 1 {
		t.Fatalf("expected one prediction call for PD members, got %d", client.calls)
	}
	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected PD group info on leader")
	}
	if info.SelectedPrefillPod == nil || info.SelectedPrefillPod.Endpoint.GetMetadata().PodName != "dispatch-prefill-a-2" {
		t.Fatalf("expected dispatched prediction to select second prefill, got %+v", info.SelectedPrefillPod)
	}
	if info.SelectedDecodePod == nil || info.SelectedDecodePod.Endpoint.GetMetadata().PodName != "dispatch-decode-a-2" {
		t.Fatalf("expected dispatched prediction to select second decode, got %+v", info.SelectedDecodePod)
	}
}

func TestScoreUsesPerGroupFallbackWhenPredictionsMissing(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 0.6, TPOTWeight: 0.4, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	predictedPrefill := scoredEndpoint(newTestEndpoint("prefill-pred", "10.0.82.2"))
	predictedDecode := scoredEndpoint(newTestEndpoint("decode-pred", "10.0.82.3"))
	setPrediction(predictedPrefill.Endpoint, 10, 100)
	setPrediction(predictedDecode.Endpoint, 100, 4)
	leaderPredicted := newPDLeader("leader-pred", "10.0.82.1", "group-pred", []fwkscheduling.ScoredEndpoint{predictedPrefill}, []fwkscheduling.ScoredEndpoint{predictedDecode})

	prefillPreferred := scoredEndpoint(newTestEndpoint("prefill-preferred", "10.0.83.2"))
	prefillRejected := scoredEndpoint(newTestEndpoint("prefill-rejected", "10.0.83.3"))
	decodeNoPrefix := scoredEndpoint(newTestEndpoint("decode-no-prefix", "10.0.83.4"))
	decodeRichPrefix := scoredEndpoint(newTestEndpoint("decode-rich-prefix", "10.0.83.5"))
	setMetrics(prefillPreferred.Endpoint, 1, 0.2)
	setMetrics(prefillRejected.Endpoint, 4, 0.8)
	setMetrics(decodeNoPrefix.Endpoint, 1, 0.1)
	setMetrics(decodeRichPrefix.Endpoint, 1, 0.2)
	setPrefixScore(prefillPreferred.Endpoint, 8, 10)
	setPrefixScore(prefillRejected.Endpoint, 1, 10)
	setPrefixScore(decodeRichPrefix.Endpoint, 10, 10)
	leaderFallback := newPDLeader(
		"leader-fallback",
		"10.0.83.1",
		"group-fallback",
		[]fwkscheduling.ScoredEndpoint{prefillPreferred, prefillRejected},
		[]fwkscheduling.ScoredEndpoint{decodeNoPrefix, decodeRichPrefix},
	)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leaderPredicted, leaderFallback})

	// leaderPredicted (predicted path): ttft=10 → 10/1010, tpot=4 → 4/1004.
	//   group = (0.6·10/1010 + 0.4·4/1004) / 1
	// leaderFallback (fallback path):
	//   best prefill = prefillPreferred:
	//     (1·0.2 + 1·0.25 + 1·0.2) / 3 = 0.65/3
	//   best decode = decodeNoPrefix (prefix is not a decode signal):
	//     (1·0.1 + 1·1.0) / 2 = 1.1/2 = 0.55
	//   group = (0.6·0.65/3 + 0.4·0.55) / 1
	testutil.AssertFloatClose(t, scores[leaderPredicted], 1.0-(0.6*(10.0/1010.0)+0.4*(4.0/1004.0)))
	testutil.AssertFloatClose(t, scores[leaderFallback], 1.0-(0.6*(0.65/3.0)+0.4*0.55))
	if scores[leaderPredicted] <= scores[leaderFallback] {
		t.Fatalf("expected predicted group score %v to be greater than fallback group score %v", scores[leaderPredicted], scores[leaderFallback])
	}

	infoFallback, ok := pdgroup.Get(leaderFallback)
	if !ok {
		t.Fatal("expected PD group info on fallback leader")
	}
	if infoFallback.SelectedPrefillPod == nil || infoFallback.SelectedPrefillPod.Endpoint.GetMetadata().PodName != "prefill-preferred" {
		t.Fatalf("expected prefill-preferred to be selected, got %+v", infoFallback.SelectedPrefillPod)
	}
	if infoFallback.SelectedDecodePod == nil || infoFallback.SelectedDecodePod.Endpoint.GetMetadata().PodName != "decode-no-prefix" {
		t.Fatalf("expected decode-no-prefix to be selected when decode prefix is ignored, got %+v", infoFallback.SelectedDecodePod)
	}
}

func TestScoreRandomizesEqualPredictedGroupMembers(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 1, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	scorer.rng = rand.New(rand.NewSource(1))
	prefillFirst := scoredEndpoint(newTestEndpoint("prefill-first", "10.0.84.2"))
	prefillSecond := scoredEndpoint(newTestEndpoint("prefill-second", "10.0.84.3"))
	decodeFirst := scoredEndpoint(newTestEndpoint("decode-first", "10.0.84.4"))
	decodeSecond := scoredEndpoint(newTestEndpoint("decode-second", "10.0.84.5"))
	setPrediction(prefillFirst.Endpoint, 15, 100)
	setPrediction(prefillSecond.Endpoint, 15, 100)
	setPrediction(decodeFirst.Endpoint, 100, 6)
	setPrediction(decodeSecond.Endpoint, 100, 6)
	leader := newPDLeader(
		"leader-equal-predicted",
		"10.0.84.1",
		"group-equal-predicted",
		[]fwkscheduling.ScoredEndpoint{prefillFirst, prefillSecond},
		[]fwkscheduling.ScoredEndpoint{decodeFirst, decodeSecond},
	)

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})

	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected PD group info on predicted leader")
	}
	if info.SelectedPrefillPod == nil || info.SelectedPrefillPod.Endpoint.GetMetadata().PodName != "prefill-second" {
		t.Fatalf("expected second tied predicted prefill endpoint, got %+v", info.SelectedPrefillPod)
	}
	if info.SelectedDecodePod == nil || info.SelectedDecodePod.Endpoint.GetMetadata().PodName != "decode-second" {
		t.Fatalf("expected second tied predicted decode endpoint, got %+v", info.SelectedDecodePod)
	}
}

func TestScoreRandomizesEqualFallbackGroupMembers(t *testing.T) {
	scorer := New("prediction", Config{TTFTWeight: 1, TPOTWeight: 1, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	scorer.rng = rand.New(rand.NewSource(1))
	prefillFirst := scoredEndpoint(newTestEndpoint("fallback-prefill-first", "10.0.85.2"))
	prefillSecond := scoredEndpoint(newTestEndpoint("fallback-prefill-second", "10.0.85.3"))
	decodeFirst := scoredEndpoint(newTestEndpoint("fallback-decode-first", "10.0.85.4"))
	decodeSecond := scoredEndpoint(newTestEndpoint("fallback-decode-second", "10.0.85.5"))
	setMetrics(prefillFirst.Endpoint, 2, 0.2)
	setMetrics(prefillSecond.Endpoint, 2, 0.2)
	setMetrics(decodeFirst.Endpoint, 3, 0.3)
	setMetrics(decodeSecond.Endpoint, 3, 0.3)
	leader := newPDLeader(
		"leader-equal-fallback",
		"10.0.85.1",
		"group-equal-fallback",
		[]fwkscheduling.ScoredEndpoint{prefillFirst, prefillSecond},
		[]fwkscheduling.ScoredEndpoint{decodeFirst, decodeSecond},
	)

	scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{leader})

	info, ok := pdgroup.Get(leader)
	if !ok {
		t.Fatal("expected PD group info on fallback leader")
	}
	if info.SelectedPrefillPod == nil || info.SelectedPrefillPod.Endpoint.GetMetadata().PodName != "fallback-prefill-second" {
		t.Fatalf("expected second tied fallback prefill endpoint, got %+v", info.SelectedPrefillPod)
	}
	if info.SelectedDecodePod == nil || info.SelectedDecodePod.Endpoint.GetMetadata().PodName != "fallback-decode-second" {
		t.Fatalf("expected second tied fallback decode endpoint, got %+v", info.SelectedDecodePod)
	}
}

func TestFactoryPreservesRawWeights(t *testing.T) {
	// Weights are no longer normalized at construction time; WeightedBadness
	// divides by their active sum at scoring time, so the raw configured
	// values flow through untouched.
	plugin, err := Factory("prediction", json.RawMessage(`{"predictorAddress":"unix:///tmp/predict.sock","timeout":"750ms","maxBatchSize":16,"ttftWeight":3,"tpotWeight":1,"kvWeight":2,"queueWeight":1,"prefixWeight":1,"inflightWeight":1}`), nil)
	if err != nil {
		t.Fatalf("Factory returned error: %v", err)
	}
	scorer, ok := plugin.(*Scorer)
	if !ok {
		t.Fatalf("expected *Scorer, got %T", plugin)
	}
	testutil.AssertFloatClose(t, scorer.ttftWeight, 3)
	testutil.AssertFloatClose(t, scorer.tpotWeight, 1)
	testutil.AssertFloatClose(t, scorer.kvWeight, 2)
	testutil.AssertFloatClose(t, scorer.queueWeight, 1)
	testutil.AssertFloatClose(t, scorer.prefixWeight, 1)
	testutil.AssertFloatClose(t, scorer.inflightWeight, 1)
	if scorer.timeout != 750*time.Millisecond {
		t.Fatalf("expected configured timeout, got %v", scorer.timeout)
	}
	if scorer.maxBatchSize != 16 {
		t.Fatalf("expected configured max batch size, got %d", scorer.maxBatchSize)
	}
	if scorer.Category() != fwkscheduling.Balance {
		t.Fatalf("expected Balance category, got %q", scorer.Category())
	}
}

func TestFactoryRejectsInvalidWeights(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{name: "prediction weights sum to zero", parameters: `{"ttftWeight":0,"tpotWeight":0}`, wantErr: "ttftWeight and tpotWeight must sum to greater than 0"},
		{name: "fallback weights sum to zero", parameters: `{"kvWeight":0,"queueWeight":0,"prefixWeight":0,"inflightWeight":0}`, wantErr: "kvWeight, queueWeight, prefixWeight, and inflightWeight must sum to greater than 0"},
		{name: "negative weight", parameters: `{"ttftWeight":-1}`, wantErr: "weights must be greater than or equal to 0"},
		{name: "non unix address", parameters: `{"predictorAddress":"tcp://localhost:9000"}`, wantErr: "predictorAddress must use unix://"},
		{name: "zero timeout", parameters: `{"predictorAddress":"unix:///tmp/p.sock","timeout":"0s"}`, wantErr: "timeout must be greater than 0"},
		{name: "negative max batch size", parameters: `{"predictorAddress":"unix:///tmp/p.sock","maxBatchSize":-1}`, wantErr: "maxBatchSize must be greater than or equal to 0"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("prediction", json.RawMessage(tt.parameters), nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestFactoryAllowsFallbackOnlyConfiguration(t *testing.T) {
	plugin, err := Factory("prediction", json.RawMessage(`{}`), nil)
	if err != nil {
		t.Fatalf("expected fallback-only scorer configuration to be valid, got %v", err)
	}
	scorer, ok := plugin.(*Scorer)
	if !ok {
		t.Fatalf("expected *Scorer, got %T", plugin)
	}
	if scorer.client != nil {
		t.Fatalf("expected no predictor client without predictorAddress, got %T", scorer.client)
	}
}

func TestFactoryRegistersPredictorCloseOnHandleContextDone(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	client := &mockPredictorClient{closed: make(chan struct{})}
	scorer, err := newScorer("prediction", Config{
		PredictorClient: client,
		Timeout:         time.Second,
		TTFTWeight:      1,
		TPOTWeight:      1,
		KVWeight:        1,
		QueueWeight:     1,
		PrefixWeight:    1,
	})
	if err != nil {
		t.Fatalf("unexpected scorer construction error: %v", err)
	}
	registerShutdownClose(fwkplugin.NewEppHandle(ctx, nil), scorer.client)

	cancel()
	select {
	case <-client.closed:
	case <-time.After(time.Second):
		t.Fatal("expected predictor client to close when handle context is done")
	}
}

func TestShadowModeIgnoresPredictionsAndRoutesByFallback(t *testing.T) {
	scorer := New("prediction", Config{Mode: ModeShadow, TTFTWeight: 0.6, TPOTWeight: 0.4, KVWeight: 1, QueueWeight: 1, PrefixWeight: 1})
	good := newTestEndpoint("good", "10.0.91.2")
	bad := newTestEndpoint("bad", "10.0.91.3")
	setMetrics(good, 1, 0.2)
	setMetrics(bad, 4, 0.8)
	setPrefixScore(good, 3, 4)
	// A great prediction on the worse-loaded endpoint must be ignored in shadow.
	setPrediction(bad, 1, 1)

	scores := scorer.Score(context.Background(), fwkscheduling.NewCycleState(), nil, []fwkscheduling.Endpoint{good, bad})

	// Identical to the active fallback math — predictions are never consulted.
	testutil.AssertFloatClose(t, scores[good], 1.0-(0.2+0.25+0.25)/3.0)
	testutil.AssertFloatClose(t, scores[bad], 1.0-(0.8+1.0+1.0)/3.0)
	if scores[good] <= scores[bad] {
		t.Fatalf("shadow mode must ignore the prediction on bad endpoint; got good=%v bad=%v", scores[good], scores[bad])
	}
}

func TestParsePredictionMode(t *testing.T) {
	if _, err := parseAndValidateParameters([]byte(`{"predictionMode":"shadow"}`)); err != nil {
		t.Fatalf("shadow mode should parse, got %v", err)
	}
	if _, err := parseAndValidateParameters([]byte(`{"predictionMode":"bogus"}`)); err == nil {
		t.Fatalf("invalid mode should be rejected")
	}
	cfg, err := parseAndValidateParameters([]byte(`{}`))
	if err != nil {
		t.Fatalf("empty params should parse, got %v", err)
	}
	if cfg.Mode != ModeActive {
		t.Fatalf("default mode should be %q, got %q", ModeActive, cfg.Mode)
	}
}

func newTestEndpoint(name, address string) fwkscheduling.Endpoint {
	return fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
			PodName:        name,
			Address:        address,
			Port:           "8000",
		},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}

func newTestEndpointWithoutMetrics(name, address string) fwkscheduling.Endpoint {
	return fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
			PodName:        name,
			Address:        address,
			Port:           "8000",
		},
		nil,
		fwkdl.NewAttributes(),
	)
}

func scoredEndpoint(endpoint fwkscheduling.Endpoint) fwkscheduling.ScoredEndpoint {
	return fwkscheduling.ScoredEndpoint{Endpoint: endpoint}
}

func newPDLeader(
	name, address, groupID string,
	prefills []fwkscheduling.ScoredEndpoint,
	decodes []fwkscheduling.ScoredEndpoint,
) fwkscheduling.Endpoint {
	leader := newTestEndpoint(name, address)
	pdgroup.Set(leader, &pdgroup.Info{GroupID: groupID, PrefillPods: prefills, DecodePods: decodes})
	return leader
}

func setPrediction(endpoint fwkscheduling.Endpoint, ttft, tpot float64) {
	endpoint.Put(latencyattr.LatencyPredictionInfoKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, ttft, tpot))
}

func setMetrics(endpoint fwkscheduling.Endpoint, waiting int, kvUsage float64) {
	metrics := endpoint.GetMetrics()
	metrics.WaitingQueueSize = waiting
	metrics.KVCacheUsagePercent = kvUsage
}

func setPrefixScore(endpoint fwkscheduling.Endpoint, matchBlocks, totalBlocks int) {
	endpoint.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(matchBlocks, totalBlocks, 16))
}

type mockPredictorClient struct {
	outputs []PredictOutput
	err     error
	delay   time.Duration
	inputs  []PredictInput
	batches [][]PredictInput
	models  []string
	calls   int
	closed  chan struct{}
}

func (m *mockPredictorClient) Predict(ctx context.Context, targetModel string, inputs []PredictInput) ([]PredictOutput, error) {
	m.calls++
	m.models = append(m.models, targetModel)
	batch := append([]PredictInput(nil), inputs...)
	m.batches = append(m.batches, batch)
	m.inputs = append(m.inputs, inputs...)
	if m.delay > 0 {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(m.delay):
		}
	}
	if m.err != nil {
		return nil, m.err
	}
	if m.outputs != nil {
		return append([]PredictOutput(nil), m.outputs...), nil
	}
	outputs := make([]PredictOutput, 0, len(inputs))
	for _, input := range inputs {
		outputs = append(outputs, PredictOutput{
			InputID:             input.InputID,
			ServerIP:            input.ServerIP,
			PredictedTTFTMillis: inputIDLatency(input.InputID),
			PredictedTPOTMillis: 1,
		})
	}
	return outputs, nil
}

func (m *mockPredictorClient) Close() error {
	if m == nil || m.closed == nil {
		return nil
	}
	close(m.closed)
	return nil
}

func inputIDLatency(inputID int) float64 {
	return float64(inputID + 1)
}

func assertPrediction(t *testing.T, endpoint fwkscheduling.Endpoint, wantTTFT, wantTPOT float64) {
	t.Helper()
	value, ok := endpoint.Get(latencyattr.LatencyPredictionInfoKey)
	if !ok {
		t.Fatal("expected active prediction attribute")
	}
	info, ok := value.(*latencyattr.LatencyPredictionInfo)
	if !ok || info == nil {
		t.Fatalf("expected *LatencyPredictionInfo, got %T", value)
	}
	if info.TTFT() != wantTTFT || info.TPOT() != wantTPOT {
		t.Fatalf("unexpected prediction: ttft=%v tpot=%v want %v/%v", info.TTFT(), info.TPOT(), wantTTFT, wantTPOT)
	}
}
