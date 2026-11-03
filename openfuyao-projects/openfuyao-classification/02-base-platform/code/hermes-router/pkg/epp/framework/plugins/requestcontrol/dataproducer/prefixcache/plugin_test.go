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

package prefixcache

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"strings"
	"testing"
	"time"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
	prefixattr "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/plugins/datalayer/attribute/prefix"

	tokenizedpromptattr "hermes-router/pkg/epp/framework/plugins/datalayer/attribute/tokenizedprompt"
	"hermes-router/pkg/epp/internal/endpointmeta"
	"hermes-router/pkg/epp/internal/testutil"
)

func TestPrepareRequestDataPublishesPrefixCacheMatchInfo(t *testing.T) {
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[{"server_ip":"10.0.40.2","l1_hit_ratio":0.75,"l3_hit_ratio":0.50},{"server_ip":"10.0.40.3","l1_hit_ratio":0.25,"l3_hit_ratio":0.50}],"message":"","status":0}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	prefill := newTestEndpoint("prefill-a", "10.0.40.2")
	decode := newTestEndpoint("decode-a", "10.0.40.3")
	request := newTokenizedRequest(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	assertPrefixCacheInfo(t, prefill, 2, 3, 4)
	assertPrefixCacheInfo(t, decode, 1, 3, 4)
	assertJSONRequest(t, client.lastRequest, hitRateRequest{
		ServerIP: []string{"10.0.40.2", "10.0.40.3"},
		Body: hitRateRequestBody{
			TokenIDs:  []uint32{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
			CacheSalt: "",
			BlockSize: 4,
		},
	})
	if client.lastRequest.URL.Path != "/kv-cache/hit-rate" {
		t.Fatalf("expected request path /kv-cache/hit-rate, got %q", client.lastRequest.URL.Path)
	}
}

func TestPrepareRequestDataUsesCompleteBlocksOnly(t *testing.T) {
	tests := []struct {
		name             string
		tokenIDs         []uint32
		wantIndexerCalls int
		wantMatchBlocks  int
		wantTotalBlocks  int
	}{
		{
			name:             "fewer tokens than one block",
			tokenIDs:         []uint32{1, 2, 3},
			wantIndexerCalls: 0,
		},
		{
			name:             "exactly one block",
			tokenIDs:         []uint32{1, 2, 3, 4},
			wantIndexerCalls: 1,
			wantMatchBlocks:  1,
			wantTotalBlocks:  1,
		},
		{
			name:             "incomplete trailing tokens",
			tokenIDs:         []uint32{1, 2, 3, 4, 5},
			wantIndexerCalls: 1,
			wantMatchBlocks:  1,
			wantTotalBlocks:  1,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := &mockHTTPClient{
				t:            t,
				responseBody: `{"server_score_list":[{"server_ip":"10.0.40.2","l1_hit_ratio":1.0,"l3_hit_ratio":1.0}],"message":"","status":0}`,
			}
			producer := New("producer", Config{
				CacheIndexerAddress: "http://cache-indexer:8080",
				BlockSize:           4,
				Timeout:             3 * time.Second,
				L1Weight:            0.7,
				L3Weight:            0.3,
				HTTPClient:          client,
			})
			endpoint := newTestEndpoint("prefill-complete-blocks", "10.0.40.2")

			if err := producer.PrepareRequestData(
				context.Background(),
				newTokenizedRequest(tt.tokenIDs...),
				[]fwkscheduling.Endpoint{endpoint},
			); err != nil {
				t.Fatalf("expected nil error, got %v", err)
			}
			if client.calls != tt.wantIndexerCalls {
				t.Fatalf("unexpected cache-indexer calls: got %d want %d", client.calls, tt.wantIndexerCalls)
			}
			if tt.wantTotalBlocks == 0 {
				if _, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
					t.Fatal("expected no prefix cache attribute for a prompt without complete blocks")
				}
				return
			}
			assertPrefixCacheInfo(t, endpoint, tt.wantMatchBlocks, tt.wantTotalBlocks, 4)
		})
	}
}

func TestPrepareRequestDataSkipsIndexerWhenTokenIDsMissing(t *testing.T) {
	client := &mockHTTPClient{t: t}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	endpoint := newTestEndpoint("prefill-b", "10.0.41.2")

	if err := producer.PrepareRequestData(context.Background(), &fwkscheduling.LLMRequest{}, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if client.calls != 0 {
		t.Fatalf("expected no cache-indexer calls, got %d", client.calls)
	}
	if _, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatalf("expected no prefix cache attribute to be written")
	}
}

func TestPrepareRequestDataClampsRatiosAndSkipsUnknownIPs(t *testing.T) {
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[{"server_ip":"10.0.42.2","l1_hit_ratio":1.5,"l3_hit_ratio":-0.1},{"server_ip":"10.0.42.99","l1_hit_ratio":0.8,"l3_hit_ratio":0.8}],"message":"","status":0}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.5,
		L3Weight:            0.5,
		HTTPClient:          client,
	})
	matched := newTestEndpoint("prefill-c", "10.0.42.2")
	unknown := newTestEndpoint("decode-c", "10.0.42.3")
	request := newTokenizedRequest(1, 2, 3, 4, 5)

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{matched, unknown}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	assertPrefixCacheInfo(t, matched, 1, 1, 4)
	if _, ok := unknown.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatalf("expected unknown endpoint without matching score to remain unset")
	}
}

func TestPrepareRequestDataFiltersByTargetPodSelector(t *testing.T) {
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[{"server_ip":"10.0.42.2","l1_hit_ratio":1.0,"l3_hit_ratio":1.0}],"message":"","status":0}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		TargetPodSelector: &endpointmeta.LabelSelectorConfig{
			Exists: []string{"test/pd-group"},
			In: map[string][]string{
				"test/pd-role": {"prefill"},
			},
		},
		HTTPClient: client,
	})
	prefill := newTestEndpoint("prefill-selector", "10.0.42.2")
	prefill.GetMetadata().Labels = map[string]string{"test/pd-group": "group-a", "test/pd-role": "prefill"}
	decode := newTestEndpoint("decode-selector", "10.0.42.3")
	decode.GetMetadata().Labels = map[string]string{"test/pd-group": "group-a", "test/pd-role": "decode"}
	outside := newTestEndpoint("outside-selector", "10.0.42.4")
	outside.GetMetadata().Labels = map[string]string{"test/pd-role": "prefill"}

	if err := producer.PrepareRequestData(context.Background(), newTokenizedRequest(1, 2, 3, 4), []fwkscheduling.Endpoint{prefill, decode, outside}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	assertPrefixCacheInfo(t, prefill, 1, 1, 4)
	if _, ok := decode.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatal("expected decode endpoint to be skipped by targetPodSelector")
	}
	if _, ok := outside.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatal("expected endpoint missing required labels to be skipped by targetPodSelector")
	}
	assertJSONRequest(t, client.lastRequest, hitRateRequest{
		ServerIP: []string{"10.0.42.2"},
		Body:     hitRateRequestBody{TokenIDs: []uint32{1, 2, 3, 4}, CacheSalt: "", BlockSize: 4},
	})
}

func TestPrepareRequestDataLogsCacheIndexerScoresAndSummary(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[{"server_ip":"10.0.40.2","l1_hit_ratio":0.75,"l3_hit_ratio":0.50},{"server_ip":"10.0.40.3","l1_hit_ratio":0.25,"l3_hit_ratio":0.50},{"server_ip":"10.0.40.4","l1_hit_ratio":0.10,"l3_hit_ratio":0.90},{"server_ip":"10.0.40.5","l1_hit_ratio":0.00,"l3_hit_ratio":0.00}],"message":"","status":0}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	request := newTokenizedRequest(1, 2, 3, 4, 5, 6, 7, 8)
	endpoints := []fwkscheduling.Endpoint{
		newTestEndpoint("prefill-a", "10.0.40.2"),
		newTestEndpoint("decode-a", "10.0.40.3"),
		newTestEndpoint("decode-b", "10.0.40.4"),
		newTestEndpoint("prefill-b", "10.0.40.5"),
	}

	if err := producer.PrepareRequestData(ctx, request, endpoints); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	entry := sink.LastInfo("Prefix cache match info published to matched endpoints")
	if entry.Message == "" {
		t.Fatal("expected prefix-cache success log")
	}
	if _, ok := entry.Keys["matchedEndpointCount"]; ok {
		t.Fatalf("expected matchedEndpointCount to be omitted, got %v", entry.Keys["matchedEndpointCount"])
	}
	if entry.Keys["cacheIndexerScoreCount"] != 4 {
		t.Fatalf("unexpected cacheIndexerScoreCount: %v", entry.Keys["cacheIndexerScoreCount"])
	}
	firstScore := sink.LastInfo("Prefix cache producer received cache-indexer score")
	if firstScore.Message == "" {
		t.Fatal("expected per-score cache-indexer log")
	}
	if firstScore.Keys["serverIP"] != "10.0.40.5" {
		t.Fatalf("unexpected serverIP on last score log: %v", firstScore.Keys["serverIP"])
	}
	if firstScore.Keys["l1HitRatio"] != 0.0 {
		t.Fatalf("unexpected l1HitRatio on last score log: %v", firstScore.Keys["l1HitRatio"])
	}
	if firstScore.Keys["l3HitRatio"] != 0.0 {
		t.Fatalf("unexpected l3HitRatio on last score log: %v", firstScore.Keys["l3HitRatio"])
	}
}

func TestPrepareRequestDataLogsSkipWhenTokenIDsMissing(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          &mockHTTPClient{t: t},
	})

	if err := producer.PrepareRequestData(ctx, &fwkscheduling.LLMRequest{}, []fwkscheduling.Endpoint{newTestEndpoint("prefill-a", "10.0.41.2")}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	entry := sink.LastInfo("Prefix cache producer skipped cache lookup")
	if entry.Message == "" {
		t.Fatal("expected prefix-cache skip log")
	}
	if entry.Keys["reason"] != "missing_tokenized_prompt" {
		t.Fatalf("unexpected skip reason: %v", entry.Keys["reason"])
	}
}

func TestPrepareRequestDataLogsErrorWhenIndexerFails(t *testing.T) {
	ctx, sink := testutil.ContextWithRecordingLogger()
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          &mockHTTPClient{t: t, err: errors.New("boom")},
	})

	if err := producer.PrepareRequestData(ctx, newTokenizedRequest(1, 2, 3, 4), []fwkscheduling.Endpoint{newTestEndpoint("prefill-a", "10.0.43.2")}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	entry := sink.LastError("Prefix cache producer failed to publish cache match info; continuing without prefix cache data")
	if entry.Message == "" {
		t.Fatal("expected cache-indexer error log")
	}
}

func TestPrepareRequestDataSkipsEndpointsOutsideTargetRoles(t *testing.T) {
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[{"server_ip":"10.0.42.2","l1_hit_ratio":0.8,"l3_hit_ratio":0.2}],"message":"","status":0}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		TargetPodRoleLabel:  "test/pd-role",
		TargetPodRoles:      []string{"prefill"},
		HTTPClient:          client,
	})
	prefill := newTestEndpoint("prefill-target", "10.0.42.2")
	prefill.GetMetadata().Labels = map[string]string{"test/pd-role": "prefill"}
	decode := newTestEndpoint("decode-target", "10.0.42.3")
	decode.GetMetadata().Labels = map[string]string{"test/pd-role": "decode"}

	if err := producer.PrepareRequestData(context.Background(), newTokenizedRequest(1, 2, 3, 4), []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	assertPrefixCacheInfo(t, prefill, 1, 1, 4)
	if _, ok := decode.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatal("expected decode endpoint to be skipped by targetPodRoles")
	}
	assertJSONRequest(t, client.lastRequest, hitRateRequest{
		ServerIP: []string{"10.0.42.2"},
		Body:     hitRateRequestBody{TokenIDs: []uint32{1, 2, 3, 4}, CacheSalt: "", BlockSize: 4},
	})
}

func TestPrepareRequestDataFailsOpenOnIndexerError(t *testing.T) {
	client := &mockHTTPClient{t: t, err: errors.New("boom")}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	endpoint := newTestEndpoint("prefill-d", "10.0.43.2")
	request := newTokenizedRequest(1, 2, 3, 4)

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected fail-open nil error, got %v", err)
	}
	if _, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatalf("expected no prefix cache attribute when indexer request fails")
	}
}

func TestPrepareRequestDataFailsOpenOnIndexerErrorStatus(t *testing.T) {
	client := &mockHTTPClient{
		t:            t,
		responseBody: `{"server_score_list":[],"message":"internal error","status":1}`,
	}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	endpoint := newTestEndpoint("prefill-e", "10.0.44.2")
	request := newTokenizedRequest(1, 2, 3, 4)

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected fail-open nil error, got %v", err)
	}
	if _, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey); ok {
		t.Fatalf("expected no prefix cache attribute when indexer returns non-zero status")
	}
}

func TestFetchHitRatesReturnsErrorForNonZeroIndexerStatus(t *testing.T) {
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient: &mockHTTPClient{
			t:            t,
			responseBody: `{"server_score_list":[],"message":"internal error","status":1}`,
		},
	})

	_, err := producer.fetchHitRates(context.Background(), hitRateRequest{
		ServerIP: []string{"10.0.44.2"},
		Body: hitRateRequestBody{
			TokenIDs:  []uint32{1, 2, 3, 4},
			CacheSalt: "",
			BlockSize: 4,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "cache-indexer returned status 1: internal error") {
		t.Fatalf("expected non-zero status error, got %v", err)
	}
}

func TestFetchHitRatesIncludesHTTPErrorBody(t *testing.T) {
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient: &mockHTTPClient{
			t:            t,
			statusCode:   http.StatusBadGateway,
			responseBody: "upstream timeout",
		},
	})

	_, err := producer.fetchHitRates(context.Background(), hitRateRequest{
		ServerIP: []string{"10.0.44.2"},
		Body: hitRateRequestBody{
			TokenIDs:  []uint32{1, 2, 3, 4},
			CacheSalt: "",
			BlockSize: 4,
		},
	})
	if err == nil || !strings.Contains(err.Error(), "cache-indexer returned HTTP 502: upstream timeout") {
		t.Fatalf("expected HTTP error with body, got %v", err)
	}
}

func TestPrepareRequestDataSkipsIndexerWhenEndpointsEmpty(t *testing.T) {
	client := &mockHTTPClient{t: t}
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          client,
	})
	request := newTokenizedRequest(1, 2, 3, 4)

	if err := producer.PrepareRequestData(context.Background(), request, []fwkscheduling.Endpoint{}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	if client.calls != 0 {
		t.Fatalf("expected no cache-indexer calls, got %d", client.calls)
	}
}

func TestFactoryRejectsMixedLegacyAndSelectorFields(t *testing.T) {
	_, err := Factory("producer", json.RawMessage(`{
		"cacheIndexerAddress":"http://cache-indexer:8080",
		"targetPodRoleLabel":"test/pd-role",
		"targetPodRoles":["prefill"],
		"targetPodSelector":{
			"in":{
				"test/pd-role":["prefill"]
			}
		}
	}`), nil)
	if err == nil || !strings.Contains(err.Error(), "targetPodSelector cannot be combined with targetPodRoleLabel/targetPodRoles") {
		t.Fatalf("expected mixed selector error, got %v", err)
	}
}

func TestFactoryRejectsLegacyRoleLabelWithoutRoles(t *testing.T) {
	_, err := Factory("producer", json.RawMessage(`{
		"cacheIndexerAddress":"http://cache-indexer:8080",
		"targetPodRoleLabel":"test/pd-role",
		"targetPodRoles":[" ",""]
	}`), nil)
	if err == nil || !strings.Contains(err.Error(), "targetPodRoleLabel requires at least one non-empty targetPodRoles value") {
		t.Fatalf("expected incomplete legacy selector error, got %v", err)
	}
}

func TestFactoryRejectsInvalidWeightsAndAddress(t *testing.T) {
	tests := []struct {
		name       string
		parameters string
		wantErr    string
	}{
		{
			name:       "missing address",
			parameters: `{}`,
			wantErr:    "cacheIndexerAddress is required",
		},
		{
			name:       "invalid url",
			parameters: `{"cacheIndexerAddress":"cache-indexer:8080"}`,
			wantErr:    "cacheIndexerAddress must be a valid URL with scheme",
		},
		{
			name:       "non-positive block size",
			parameters: `{"cacheIndexerAddress":"http://cache-indexer:8080","blockSize":0}`,
			wantErr:    "blockSize must be greater than 0",
		},
		{
			name:       "weights sum to zero",
			parameters: `{"cacheIndexerAddress":"http://cache-indexer:8080","l1Weight":0,"l3Weight":0}`,
			wantErr:    "at least one of l1Weight or l3Weight must be greater than 0",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := Factory("producer", json.RawMessage(tt.parameters), nil)
			if err == nil || !strings.Contains(err.Error(), tt.wantErr) {
				t.Fatalf("expected error containing %q, got %v", tt.wantErr, err)
			}
		})
	}
}

func TestProducesAndConsumesDeclarations(t *testing.T) {
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
		HTTPClient:          &mockHTTPClient{t: t},
	})

	produces := producer.Produces()
	if len(produces) != 1 {
		t.Fatalf("expected one produced key, got %d", len(produces))
	}
	value, ok := produces[prefixattr.PrefixCacheMatchInfoKey]
	if !ok {
		t.Fatalf("expected produced key %q", prefixattr.PrefixCacheMatchInfoKey)
	}
	if _, ok := value.(prefixattr.PrefixCacheMatchInfo); !ok {
		t.Fatalf("expected produced value to be PrefixCacheMatchInfo, got %T", value)
	}

	consumes := producer.Consumes()
	if len(consumes) != 1 {
		t.Fatalf("expected one consumed key, got %d", len(consumes))
	}
	consumed, ok := consumes[tokenizedpromptattr.TokenizedPromptKey]
	if !ok {
		t.Fatalf("expected consumed key %q", tokenizedpromptattr.TokenizedPromptKey)
	}
	if _, ok := consumed.(tokenizedpromptattr.Info); !ok {
		t.Fatalf("expected consumed value to be TokenizedPromptInfo, got %T", consumed)
	}
}

func TestNewPrefixCacheProducerDefaultHTTPClientUsesContextDeadlineOnly(t *testing.T) {
	producer := New("producer", Config{
		CacheIndexerAddress: "http://cache-indexer:8080",
		BlockSize:           4,
		Timeout:             3 * time.Second,
		L1Weight:            0.7,
		L3Weight:            0.3,
	})

	client, ok := producer.client.(*http.Client)
	if !ok {
		t.Fatalf("expected default HTTP client to be *http.Client, got %T", producer.client)
	}
	if client.Timeout != 0 {
		t.Fatalf("expected default HTTP client timeout to be 0, got %v", client.Timeout)
	}
}

type mockHTTPClient struct {
	t            *testing.T
	err          error
	statusCode   int
	responseBody string
	lastRequest  *http.Request
	calls        int
}

func (m *mockHTTPClient) Do(request *http.Request) (*http.Response, error) {
	m.calls++
	m.lastRequest = request
	if m.err != nil {
		return nil, m.err
	}
	statusCode := m.statusCode
	if statusCode == 0 {
		statusCode = http.StatusOK
	}
	return &http.Response{
		StatusCode: statusCode,
		Body:       io.NopCloser(strings.NewReader(m.responseBody)),
		Header:     make(http.Header),
	}, nil
}

func newTokenizedRequest(tokenIDs ...uint32) *fwkscheduling.LLMRequest {
	return &fwkscheduling.LLMRequest{
		TokenizedPrompt: &fwkscheduling.TokenizedPrompt{TokenIDs: tokenIDs},
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

func assertPrefixCacheInfo(t *testing.T, endpoint fwkscheduling.Endpoint, wantMatch, wantTotal, wantBlockSize int) {
	t.Helper()
	value, ok := endpoint.Get(prefixattr.PrefixCacheMatchInfoKey)
	if !ok {
		t.Fatalf("expected attribute %q to be written", prefixattr.PrefixCacheMatchInfoKey)
	}
	info, ok := value.(*prefixattr.PrefixCacheMatchInfo)
	if !ok || info == nil {
		t.Fatalf("expected *PrefixCacheMatchInfo attribute, got %T", value)
	}
	if info.MatchBlocks() != wantMatch || info.TotalBlocks() != wantTotal || info.BlockSizeTokens() != wantBlockSize {
		t.Fatalf(
			"unexpected prefix cache info: match=%d total=%d blockSize=%d",
			info.MatchBlocks(),
			info.TotalBlocks(),
			info.BlockSizeTokens(),
		)
	}
}

func assertJSONRequest(t *testing.T, request *http.Request, want hitRateRequest) {
	t.Helper()
	if request == nil {
		t.Fatal("expected HTTP request to be captured")
	}
	body, err := io.ReadAll(request.Body)
	if err != nil {
		t.Fatalf("read request body: %v", err)
	}
	if err := request.Body.Close(); err != nil {
		t.Fatalf("close request body: %v", err)
	}
	got := hitRateRequest{}
	if err := json.Unmarshal(body, &got); err != nil {
		t.Fatalf("unmarshal request body: %v", err)
	}
	if !equalHitRateRequest(got, want) {
		t.Fatalf("unexpected request payload: got %+v want %+v", got, want)
	}
}

func equalHitRateRequest(got, want hitRateRequest) bool {
	if len(got.ServerIP) != len(want.ServerIP) || len(got.Body.TokenIDs) != len(want.Body.TokenIDs) {
		return false
	}
	for i := range got.ServerIP {
		if got.ServerIP[i] != want.ServerIP[i] {
			return false
		}
	}
	for i := range got.Body.TokenIDs {
		if got.Body.TokenIDs[i] != want.Body.TokenIDs[i] {
			return false
		}
	}
	return got.Body.CacheSalt == want.Body.CacheSalt && got.Body.BlockSize == want.Body.BlockSize
}
