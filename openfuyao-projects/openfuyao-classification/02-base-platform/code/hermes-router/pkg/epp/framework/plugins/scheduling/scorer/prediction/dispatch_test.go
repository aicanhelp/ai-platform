/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package prediction

import (
	"context"
	"errors"
	"math"
	"testing"

	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	latencyattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/latency"

	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	"hermes-router/pkg/epp/internal/predictiondata"
)

func TestCollectFeatureInputsRenumbersAndSkipsMissingSnapshots(t *testing.T) {
	withoutInput := newTestEndpoint("missing-input", "10.0.90.1")
	wrongType := newTestEndpoint("wrong-input-type", "10.0.90.4")
	wrongType.Put(predictioninputattr.PredictionInputKey, latencyattr.NewLatencyPredictionInfo(true, true, 0, 0, 1, 1))
	first := newTestEndpointWithInput("first-input", "10.0.90.2", predictiondata.Input{InputID: 42, ServerIP: "10.0.90.2"})
	second := newTestEndpointWithInput("second-input", "10.0.90.3", predictiondata.Input{InputID: 42, ServerIP: "10.0.90.3"})

	features := collectFeatureInputs([]fwkscheduling.Endpoint{withoutInput, wrongType, first, second})

	if len(features.inputs) != 2 {
		t.Fatalf("expected two collected inputs, got %+v", features.inputs)
	}
	if len(features.groups) != 0 {
		t.Fatalf("expected no PD groups, got %+v", features.groups)
	}
	if features.inputs[0].InputID != 0 || features.inputs[1].InputID != 1 {
		t.Fatalf("expected inputs to be renumbered by batch position, got %+v", features.inputs)
	}
	if features.endpointByInputID[0].GetMetadata().PodName != "first-input" ||
		features.endpointByInputID[1].GetMetadata().PodName != "second-input" {
		t.Fatalf("unexpected endpoint mapping: %+v", features.endpointByInputID)
	}
}

func TestCollectFeatureInputsExpandsPDGroupMembers(t *testing.T) {
	prefill := scoredEndpoint(newTestEndpointWithInput("prefill-input", "10.0.91.2", predictiondata.Input{PodType: predictiondata.PodTypePrefill}))
	decode := scoredEndpoint(newTestEndpointWithInput("decode-input", "10.0.91.3", predictiondata.Input{PodType: predictiondata.PodTypeDecode}))
	leader := newPDLeader("leader-input", "10.0.91.1", "group-input", []fwkscheduling.ScoredEndpoint{prefill}, []fwkscheduling.ScoredEndpoint{decode})

	features := collectFeatureInputs([]fwkscheduling.Endpoint{leader})

	if len(features.inputs) != 2 {
		t.Fatalf("expected prefill and decode inputs, got %+v", features.inputs)
	}
	if len(features.groups) != 1 || features.groups[0].leader != leader {
		t.Fatalf("expected one collected PD group, got %+v", features.groups)
	}
	if features.inputs[0].PodType != predictiondata.PodTypePrefill ||
		features.inputs[1].PodType != predictiondata.PodTypeDecode {
		t.Fatalf("expected PD member order prefill then decode, got %+v", features.inputs)
	}
	if features.endpointByInputID[0].GetMetadata().PodName != "prefill-input" ||
		features.endpointByInputID[1].GetMetadata().PodName != "decode-input" {
		t.Fatalf("unexpected endpoint mapping: %+v", features.endpointByInputID)
	}
}

func TestCollectFeatureInputsDeduplicatesEndpoints(t *testing.T) {
	prefill := scoredEndpoint(newTestEndpointWithInput("prefill-duplicate", "10.0.92.2", predictiondata.Input{PodType: predictiondata.PodTypePrefill}))
	decode := scoredEndpoint(newTestEndpointWithInput("decode-duplicate", "10.0.92.3", predictiondata.Input{PodType: predictiondata.PodTypeDecode}))
	leader := newPDLeader("leader-duplicate", "10.0.92.1", "group-duplicate", []fwkscheduling.ScoredEndpoint{prefill}, []fwkscheduling.ScoredEndpoint{decode})

	features := collectFeatureInputs([]fwkscheduling.Endpoint{leader, prefill.Endpoint, decode.Endpoint})

	if len(features.inputs) != 2 {
		t.Fatalf("expected duplicated group members to be collected once, got %+v", features.inputs)
	}
}

func TestPublishOutputsSkipsUnknownAndInvalidOutputs(t *testing.T) {
	good := newTestEndpoint("publish-good", "10.0.93.1")
	endpoints := map[int]fwkscheduling.Endpoint{0: good}

	publishOutputs([]predictiondata.Output{
		{InputID: 99, PredictedTTFTMillis: 10, PredictedTPOTMillis: 1},
		{InputID: 0, PredictedTTFTMillis: math.NaN(), PredictedTPOTMillis: 1},
	}, endpoints)
	if _, ok := good.Get(latencyattr.LatencyPredictionInfoKey); ok {
		t.Fatal("expected invalid and unknown outputs to be skipped")
	}

	publishOutputs([]predictiondata.Output{{InputID: 0, PredictedTTFTMillis: 10, PredictedTPOTMillis: 1}}, endpoints)
	if _, ok := good.Get(latencyattr.LatencyPredictionInfoKey); !ok {
		t.Fatal("expected valid output to publish active latency prediction")
	}
}

func TestPredictBatchesSplitsInputs(t *testing.T) {
	client := &mockPredictorClient{}
	inputs := []predictiondata.Input{{InputID: 0}, {InputID: 1}, {InputID: 2}, {InputID: 3}, {InputID: 4}}

	outputs, err := predictBatches(context.Background(), client, "test-model", inputs, 2)
	if err != nil {
		t.Fatalf("unexpected predict error: %v", err)
	}
	if len(outputs) != 5 {
		t.Fatalf("expected generated output per input, got %+v", outputs)
	}
	if got := len(client.batches); got != 3 {
		t.Fatalf("expected three batches, got %d", got)
	}
	if len(client.batches[0]) != 2 || len(client.batches[1]) != 2 || len(client.batches[2]) != 1 {
		t.Fatalf("unexpected batch sizes: %+v", client.batches)
	}
}

func TestPredictBatchesReturnsClientError(t *testing.T) {
	wantErr := errors.New("predict failed")
	client := &mockPredictorClient{err: wantErr}

	_, err := predictBatches(context.Background(), client, "test-model", []predictiondata.Input{{InputID: 0}}, 1)
	if !errors.Is(err, wantErr) {
		t.Fatalf("expected client error %v, got %v", wantErr, err)
	}
}

func newTestEndpointWithInput(name, address string, input predictiondata.Input) fwkscheduling.Endpoint {
	endpoint := newTestEndpoint(name, address)
	endpoint.Put(predictioninputattr.PredictionInputKey, &predictioninputattr.InputSnapshot{Input: input})
	return endpoint
}
