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

package predictionfeature

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	npuattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/npu"
	predictioninputattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/predictioninput"
	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	internalinflight "hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/predictiondata"
)

const (
	testPDLabel      = "test/pd-role"
	testPrefillLabel = "prefill"
	testDecodeLabel  = "decode"
)

func TestExtractorPublishesInputForAggregateEndpoint(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpoint("agg-a", "10.0.60.1", nil)
	seedFullFeatures(endpoint, featureSeeds{waiting: 3, running: 1, kvUsagePercent: 0.4, prefixMatch: 2, prefixTotal: 4, inflightTokens: 64, aiCoreMax: 50, hbmMax: 60, bandwidthMax: 30})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(5), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	assertInputSnapshot(t, endpoint, predictiondata.Input{
		PodType:              predictiondata.PodTypeAggregate,
		InputTokenLength:     5,
		PrefixCacheScore:     0.5,
		NumRequestWaiting:    3,
		NumRequestRunning:    1,
		KVCacheUsage:         0.4,
		InflightTokens:       64,
		HBMUtilization:       0.6,
		AICoreUtilization:    0.5,
		BandwidthUtilization: 0.3,
		ServerIP:             "10.0.60.1",
	})
	assertNoMissingReason(t, endpoint)
}

func TestExtractorResolvesPDLabelsIntoCanonicalPodTypes(t *testing.T) {
	extractor := newExtractor(t, Config{
		PDLabelName:  testPDLabel,
		PrefillValue: testPrefillLabel,
		DecodeValue:  testDecodeLabel,
	})
	prefill := newTestEndpoint("prefill-a", "10.0.60.2", map[string]string{testPDLabel: testPrefillLabel})
	decode := newTestEndpoint("decode-a", "10.0.60.3", map[string]string{testPDLabel: testDecodeLabel})
	seedFullFeatures(prefill, featureSeeds{inflightTokens: 1, aiCoreMax: 1, hbmMax: 1, bandwidthMax: 1})
	seedFullFeatures(decode, featureSeeds{inflightTokens: 1, aiCoreMax: 1, hbmMax: 1, bandwidthMax: 1})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(3), []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := inputSnapshot(t, prefill).PodType; got != predictiondata.PodTypePrefill {
		t.Fatalf("expected prefill PodType, got %q", got)
	}
	if got := inputSnapshot(t, decode).PodType; got != predictiondata.PodTypeDecode {
		t.Fatalf("expected decode PodType, got %q", got)
	}
}

func TestExtractorDefaultsToAggregateWhenLabelValueUnknown(t *testing.T) {
	extractor := newExtractor(t, Config{
		PDLabelName:  testPDLabel,
		PrefillValue: testPrefillLabel,
		DecodeValue:  testDecodeLabel,
	})
	endpoint := newTestEndpoint("mystery", "10.0.60.4", map[string]string{testPDLabel: "unknown"})
	seedFullFeatures(endpoint, featureSeeds{inflightTokens: 1, aiCoreMax: 1, hbmMax: 1, bandwidthMax: 1})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(3), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if got := inputSnapshot(t, endpoint).PodType; got != predictiondata.PodTypeAggregate {
		t.Fatalf("expected aggregate fallback for unrecognized label, got %q", got)
	}
}

func TestExtractorPublishesMissingReasonWhenMetricsAbsent(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpointNoMetrics("no-metrics", "10.0.61.1")

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMissingReason(t, endpoint, reasonMissingMetrics)
	assertNoInputSnapshot(t, endpoint)
}

func TestExtractorPublishesMissingReasonWhenInflightAbsent(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpoint("no-inflight", "10.0.61.2", nil)
	// NPU state and metrics present, inflight load deliberately missing.
	endpoint.Put(npuattr.DefaultStateAttributeKey, &npuattr.State{HBMUtilizationMax: 50, AICoreUtilizationMax: 60, HBMBandwidthUtilizationMax: 70})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMissingReason(t, endpoint, reasonMissingInflight)
}

func TestExtractorPublishesMissingReasonWhenNPUStateAbsent(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpoint("no-npu", "10.0.61.3", nil)
	endpoint.Put(internalinflight.DefaultAttributeKey, &internalinflight.Load{Tokens: 8, Requests: 1})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMissingReason(t, endpoint, reasonMissingNPUState)
}

func TestExtractorPublishesMissingReasonWhenNPUStateStale(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpoint("stale-npu", "10.0.61.4", nil)
	endpoint.Put(internalinflight.DefaultAttributeKey, &internalinflight.Load{Tokens: 8, Requests: 1})
	// Stale snapshot carries plausible-looking values but its DataStatus marks
	// the observation as too old to trust; the extractor must refuse it
	// instead of letting expired numbers reach the predictor.
	endpoint.Put(npuattr.DefaultStateAttributeKey, &npuattr.State{
		DataStatus:                 npuattr.DataStale,
		HBMUtilizationMax:          50,
		AICoreUtilizationMax:       60,
		HBMBandwidthUtilizationMax: 70,
	})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMissingReason(t, endpoint, reasonStaleNPUState)
	assertNoInputSnapshot(t, endpoint)
}

func TestExtractorPublishesMissingReasonWhenAddressAbsent(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: "no-addr"}, PodName: "no-addr"},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertMissingReason(t, endpoint, reasonMissingAddress)
}

func TestExtractorSkipsAllWhenRequestNotTokenized(t *testing.T) {
	extractor := newExtractor(t, Config{})
	endpoint := newTestEndpoint("not-tokenized", "10.0.62.1", nil)
	seedFullFeatures(endpoint, featureSeeds{inflightTokens: 4, aiCoreMax: 10, hbmMax: 10, bandwidthMax: 10})

	if err := extractor.PrepareRequestData(context.Background(), &fwkscheduling.LLMRequest{TargetModel: "test"}, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	assertNoInputSnapshot(t, endpoint)
	assertNoMissingReason(t, endpoint)
}

func TestExtractorRespectsTargetSelector(t *testing.T) {
	extractor := newExtractor(t, Config{
		TargetPodRoleLabel: testPDLabel,
		TargetPodRoles:     []string{testPrefillLabel},
	})
	included := newTestEndpoint("included", "10.0.63.1", map[string]string{testPDLabel: testPrefillLabel})
	excluded := newTestEndpoint("excluded", "10.0.63.2", map[string]string{testPDLabel: testDecodeLabel})
	seedFullFeatures(included, featureSeeds{inflightTokens: 1, aiCoreMax: 1, hbmMax: 1, bandwidthMax: 1})
	seedFullFeatures(excluded, featureSeeds{inflightTokens: 1, aiCoreMax: 1, hbmMax: 1, bandwidthMax: 1})

	if err := extractor.PrepareRequestData(context.Background(), tokenizedRequest(2), []fwkscheduling.Endpoint{included, excluded}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := included.Get(predictioninputattr.PredictionInputKey); !ok {
		t.Fatalf("expected included endpoint to receive a snapshot")
	}
	if _, ok := excluded.Get(predictioninputattr.PredictionInputKey); ok {
		t.Fatalf("expected excluded endpoint to be skipped without a snapshot")
	}
	if _, ok := excluded.Get(predictioninputattr.MissingFeatureReasonKey); ok {
		t.Fatalf("expected selector skip to be silent — no MissingFeatureReason attribute")
	}
}

func TestProducesAndConsumesAdvertiseExpectedKeys(t *testing.T) {
	extractor := newExtractor(t, Config{})
	produces := extractor.Produces()
	if _, ok := produces[predictioninputattr.PredictionInputKey]; !ok {
		t.Fatalf("expected Produces to include %q", predictioninputattr.PredictionInputKey)
	}
	if _, ok := produces[predictioninputattr.MissingFeatureReasonKey]; !ok {
		t.Fatalf("expected Produces to include %q", predictioninputattr.MissingFeatureReasonKey)
	}
	consumes := extractor.Consumes()
	requiredConsumes := []string{
		tokenizedpromptattr.TokenizedPromptKey,
		prefixattr.PrefixCacheMatchInfoKey,
		internalinflight.DefaultAttributeKey,
		npuattr.DefaultStateAttributeKey,
	}
	for _, key := range requiredConsumes {
		if _, ok := consumes[key]; !ok {
			t.Fatalf("expected Consumes to include %q so DAG ordering blocks the extractor until the upstream attribute is published", key)
		}
	}
}

func TestFactoryRejectsBlankCanonicalRolesWhenPDLabelSet(t *testing.T) {
	empty := ""
	raw, err := json.Marshal(pluginParameters{PDLabelName: testPDLabel, PrefillValue: &empty, DecodeValue: &empty})
	if err != nil {
		t.Fatalf("marshal helper failed: %v", err)
	}
	_, err = Factory("extractor", raw, nil)
	if err == nil {
		t.Fatal("expected factory to reject blank prefill and decode values when pdLabelName is set")
	}
	if !strings.Contains(err.Error(), "prefillValue and decodeValue must be non-empty") {
		t.Fatalf("unexpected error message: %v", err)
	}
}

func TestFactoryParsesSelectorConfig(t *testing.T) {
	raw, err := json.Marshal(pluginParameters{
		TargetPodRoleLabel: testPDLabel,
		TargetPodRoles:     []string{testPrefillLabel, testDecodeLabel},
	})
	if err != nil {
		t.Fatalf("marshal helper failed: %v", err)
	}
	plugin, err := Factory("extractor", raw, nil)
	if err != nil {
		t.Fatalf("unexpected factory error: %v", err)
	}
	if plugin == nil {
		t.Fatal("expected non-nil plugin")
	}
	if plugin.TypedName().Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, plugin.TypedName().Type)
	}
}

func TestFactoryRejectsUnknownJSONFields(t *testing.T) {
	raw := json.RawMessage(`{"unknownField":"oops"}`)
	if _, err := Factory("extractor", raw, nil); err == nil {
		t.Fatal("expected factory to reject unknown JSON fields")
	}
}

type featureSeeds struct {
	waiting        int
	running        int
	kvUsagePercent float64
	prefixMatch    int
	prefixTotal    int
	inflightTokens int
	aiCoreMax      float64
	hbmMax         float64
	bandwidthMax   float64
}

func newExtractor(t *testing.T, config Config) *FeatureExtractor {
	t.Helper()
	extractor, err := New("extractor", config)
	if err != nil {
		t.Fatalf("constructing extractor failed: %v", err)
	}
	return extractor
}

func tokenizedRequest(tokenCount int) *fwkscheduling.LLMRequest {
	tokens := make([]uint32, tokenCount)
	for index := range tokens {
		tokens[index] = uint32(index + 1)
	}
	return &fwkscheduling.LLMRequest{
		TargetModel:     "test-model",
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: tokens},
	}
}

func newTestEndpoint(name, address string, labels map[string]string) fwkscheduling.Endpoint {
	return fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
			PodName:        name,
			Address:        address,
			Port:           "8000",
			Labels:         cloneLabels(labels),
		},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}

// newTestEndpointNoMetrics builds an endpoint whose metrics pointer is nil so
// the extractor's metrics-presence branch can be exercised. NewEndpoint with a
// nil metrics argument achieves that without forcing tests to reach into
// unexported state.
func newTestEndpointNoMetrics(name, address string) fwkscheduling.Endpoint {
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

func cloneLabels(labels map[string]string) map[string]string {
	if len(labels) == 0 {
		return nil
	}
	cloned := make(map[string]string, len(labels))
	for key, value := range labels {
		cloned[key] = value
	}
	return cloned
}

func seedFullFeatures(endpoint fwkscheduling.Endpoint, seeds featureSeeds) {
	metrics := endpoint.GetMetrics()
	metrics.WaitingQueueSize = seeds.waiting
	metrics.RunningRequestsSize = seeds.running
	metrics.KVCacheUsagePercent = seeds.kvUsagePercent
	if seeds.prefixTotal > 0 {
		endpoint.Put(prefixattr.PrefixCacheMatchInfoKey, prefixattr.NewPrefixCacheMatchInfo(seeds.prefixMatch, seeds.prefixTotal, 16))
	}
	endpoint.Put(internalinflight.DefaultAttributeKey, &internalinflight.Load{Tokens: seeds.inflightTokens, Requests: 1})
	endpoint.Put(npuattr.DefaultStateAttributeKey, &npuattr.State{
		AICoreUtilizationMax:       seeds.aiCoreMax,
		HBMUtilizationMax:          seeds.hbmMax,
		HBMBandwidthUtilizationMax: seeds.bandwidthMax,
	})
}

func inputSnapshot(t *testing.T, endpoint fwkscheduling.Endpoint) predictiondata.Input {
	t.Helper()
	value, ok := endpoint.Get(predictioninputattr.PredictionInputKey)
	if !ok {
		t.Fatalf("expected prediction input snapshot")
	}
	snapshot, ok := value.(*predictioninputattr.InputSnapshot)
	if !ok || snapshot == nil {
		t.Fatalf("expected *InputSnapshot, got %T", value)
	}
	return snapshot.Input
}

func assertInputSnapshot(t *testing.T, endpoint fwkscheduling.Endpoint, want predictiondata.Input) {
	t.Helper()
	got := inputSnapshot(t, endpoint)
	if got != want {
		t.Fatalf("unexpected snapshot:\n got %+v\nwant %+v", got, want)
	}
}

func assertNoInputSnapshot(t *testing.T, endpoint fwkscheduling.Endpoint) {
	t.Helper()
	if _, ok := endpoint.Get(predictioninputattr.PredictionInputKey); ok {
		t.Fatalf("expected no prediction input snapshot to be written")
	}
}

func assertMissingReason(t *testing.T, endpoint fwkscheduling.Endpoint, want string) {
	t.Helper()
	value, ok := endpoint.Get(predictioninputattr.MissingFeatureReasonKey)
	if !ok {
		t.Fatalf("expected MissingFeatureReason attribute on endpoint")
	}
	reason, ok := value.(*predictioninputattr.MissingFeatureReason)
	if !ok || reason == nil {
		t.Fatalf("expected *MissingFeatureReason, got %T", value)
	}
	if reason.Reason != want {
		t.Fatalf("unexpected missing reason: got %q want %q", reason.Reason, want)
	}
}

func assertNoMissingReason(t *testing.T, endpoint fwkscheduling.Endpoint) {
	t.Helper()
	if _, ok := endpoint.Get(predictioninputattr.MissingFeatureReasonKey); ok {
		t.Fatalf("expected no MissingFeatureReason attribute on a successfully featurized endpoint")
	}
}
