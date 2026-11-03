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

package inflight

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	internalinflight "hermes-router/pkg/epp/internal/inflight"
)

func TestPrepareRequestDataPublishesInflightSnapshotsToEndpointAttributes(t *testing.T) {
	store := internalinflight.StoreForName("producer-snapshots")
	t.Cleanup(func() { _ = store.Close() })
	prefill := newTestEndpoint("prefill-a", "10.0.30.2")
	decode := newTestEndpoint("decode-a", "10.0.30.3")
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-1", SelectedEndpoint: prefill.GetMetadata(), InputTokenLength: 5})
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-2", SelectedEndpoint: decode.GetMetadata(), InputTokenLength: 5})
	producer := NewLoadProducer("producer", store, internalinflight.DefaultAttributeKey)

	if err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	assertEndpointLoad(t, prefill, 5, 1)
	assertEndpointLoad(t, decode, 5, 1)
}

func TestPrepareRequestDataPublishesZeroLoadsForEndpointsWithoutTrackedRequests(t *testing.T) {
	store := internalinflight.StoreForName("producer-zero")
	t.Cleanup(func() { _ = store.Close() })
	endpoint := newTestEndpoint("prefill-b", "10.0.31.2")
	producer := NewLoadProducer("producer", store, internalinflight.DefaultAttributeKey)

	if err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{endpoint}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}

	assertEndpointLoad(t, endpoint, 0, 0)
}

func TestPrepareRequestDataPublishesSnapshotsOnlyToMatchedPods(t *testing.T) {
	store := internalinflight.StoreForName("producer-selector")
	t.Cleanup(func() { _ = store.Close() })
	plugin, err := Factory("producer", json.RawMessage(`{
		"storeName":"producer-selector",
		"targetPodSelector":{
			"exists":["test/pd-group"],
			"in":{
				"test/pd-role":["prefill"]
			}
		}
	}`), nil)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	producer, ok := plugin.(*LoadProducer)
	if !ok {
		t.Fatalf("expected *LoadProducer, got %T", plugin)
	}
	prefill := newTestEndpoint("prefill-selector", "10.0.32.2")
	prefill.GetMetadata().Labels = map[string]string{"test/pd-group": "group-a", "test/pd-role": "prefill"}
	decode := newTestEndpoint("decode-selector", "10.0.32.3")
	decode.GetMetadata().Labels = map[string]string{"test/pd-group": "group-a", "test/pd-role": "decode"}
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-1", SelectedEndpoint: prefill.GetMetadata(), InputTokenLength: 5})
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-2", SelectedEndpoint: decode.GetMetadata(), InputTokenLength: 7})

	if err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	assertEndpointLoad(t, prefill, 5, 1)
	assertNoEndpointLoad(t, decode)
}

func TestFactoryMapsLegacyRoleFieldsToSelector(t *testing.T) {
	store := internalinflight.StoreForName("producer-legacy-selector")
	t.Cleanup(func() { _ = store.Close() })
	plugin, err := Factory("producer", json.RawMessage(`{
		"storeName":"producer-legacy-selector",
		"targetPodRoleLabel":"test/pd-role",
		"targetPodRoles":["prefill"]
	}`), nil)
	if err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	producer, ok := plugin.(*LoadProducer)
	if !ok {
		t.Fatalf("expected *LoadProducer, got %T", plugin)
	}
	prefill := newTestEndpoint("prefill-legacy", "10.0.33.2")
	prefill.GetMetadata().Labels = map[string]string{"test/pd-role": "prefill"}
	decode := newTestEndpoint("decode-legacy", "10.0.33.3")
	decode.GetMetadata().Labels = map[string]string{"test/pd-role": "decode"}
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-1", SelectedEndpoint: prefill.GetMetadata(), InputTokenLength: 5})
	store.StartRequest(internalinflight.RequestStart{RequestID: "req-2", SelectedEndpoint: decode.GetMetadata(), InputTokenLength: 7})

	if err := producer.PrepareRequestData(context.Background(), nil, []fwkscheduling.Endpoint{prefill, decode}); err != nil {
		t.Fatalf("expected nil error, got %v", err)
	}
	assertEndpointLoad(t, prefill, 5, 1)
	assertNoEndpointLoad(t, decode)
}

func TestFactoryRejectsMixedLegacyAndSelectorFields(t *testing.T) {
	_, err := Factory("producer", json.RawMessage(`{
		"storeName":"producer-selector-conflict",
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
		"storeName":"producer-selector-conflict",
		"targetPodRoleLabel":"test/pd-role",
		"targetPodRoles":[" ",""]
	}`), nil)
	if err == nil || !strings.Contains(err.Error(), "targetPodRoleLabel requires at least one non-empty targetPodRoles value") {
		t.Fatalf("expected incomplete legacy selector error, got %v", err)
	}
}

func TestProducesAndConsumesDeclarations(t *testing.T) {
	producer := NewLoadProducer("producer", internalinflight.StoreForName("declaration-test"), internalinflight.DefaultAttributeKey)

	produces := producer.Produces()
	if len(produces) != 1 {
		t.Fatalf("expected one produced key, got %d", len(produces))
	}
	value, ok := produces[internalinflight.DefaultAttributeKey]
	if !ok {
		t.Fatalf("expected produced key %q", internalinflight.DefaultAttributeKey)
	}
	if _, ok := value.(internalinflight.Load); !ok {
		t.Fatalf("expected produced value to be inflight.Load, got %T", value)
	}
	if len(producer.Consumes()) != 0 {
		t.Fatalf("expected no consumed keys")
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

func assertEndpointLoad(t *testing.T, endpoint fwkscheduling.Endpoint, wantTokens, wantRequests int) {
	t.Helper()
	value, ok := endpoint.Get(internalinflight.DefaultAttributeKey)
	if !ok {
		t.Fatalf("expected attribute %q to be written", internalinflight.DefaultAttributeKey)
	}
	load, ok := value.(*internalinflight.Load)
	if !ok || load == nil {
		t.Fatalf("expected *inflight.Load attribute, got %T", value)
	}
	if load.Tokens != wantTokens || load.Requests != wantRequests {
		t.Fatalf("unexpected inflight load: got %+v, want tokens=%d requests=%d", load, wantTokens, wantRequests)
	}
}

func assertNoEndpointLoad(t *testing.T, endpoint fwkscheduling.Endpoint) {
	t.Helper()
	if _, ok := endpoint.Get(internalinflight.DefaultAttributeKey); ok {
		t.Fatalf("expected attribute %q to be absent", internalinflight.DefaultAttributeKey)
	}
}
