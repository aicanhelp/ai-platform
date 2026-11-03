/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *         http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package lifecycle

import (
	"context"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"

	"hermes-router/pkg/epp/internal/inflight"
	"hermes-router/pkg/epp/internal/testutil"
)

func newDatalayerEndpoint(name string) fwkdl.Endpoint {
	return fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: testutil.TestNamespace, Name: name},
		PodName:        name,
		Address:        "10.0.70.1",
		Port:           testutil.TestInferencePort,
	}, fwkdl.NewMetrics())
}

func TestTrackerIdentityAccessors(t *testing.T) {
	tracker := NewRequestLifecycleTracker("tracker", inflight.StoreForName("accessor-test"))

	typedName := tracker.TypedName()
	if typedName.Name != "tracker" {
		t.Fatalf("expected plugin name tracker, got %q", typedName.Name)
	}
	if typedName.Type != PluginType {
		t.Fatalf("expected plugin type %q, got %q", PluginType, typedName.Type)
	}
	if got := tracker.ExpectedInputType(); got != fwkdl.EndpointEventReflectType {
		t.Fatalf("expected endpoint event input type, got %v", got)
	}
}

func TestTrackerExtractIsNoOp(t *testing.T) {
	tracker := NewRequestLifecycleTracker("tracker", inflight.StoreForName("accessor-test"))
	if err := tracker.Extract(context.Background(), nil, newDatalayerEndpoint("pod")); err != nil {
		t.Fatalf("expected Extract to be a no-op, got %v", err)
	}
}

func TestTrackerExtractEndpointAppliesEvents(t *testing.T) {
	store := inflight.StoreForName("accessor-test")
	tracker := NewRequestLifecycleTracker("tracker", store)
	endpoint := newDatalayerEndpoint("pod")

	if err := tracker.ExtractEndpoint(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventAddOrUpdate,
		Endpoint: endpoint,
	}); err != nil {
		t.Fatalf("expected add event to be applied, got %v", err)
	}
	if err := tracker.ExtractEndpoint(context.Background(), fwkdl.EndpointEvent{
		Type:     fwkdl.EventDelete,
		Endpoint: endpoint,
	}); err != nil {
		t.Fatalf("expected delete event to be applied, got %v", err)
	}

	// Events without an endpoint and trackers without a store are ignored.
	if err := tracker.ExtractEndpoint(context.Background(), fwkdl.EndpointEvent{}); err != nil {
		t.Fatalf("expected empty event to be ignored, got %v", err)
	}
	bare := NewRequestLifecycleTracker("bare", nil)
	if err := bare.ExtractEndpoint(context.Background(), fwkdl.EndpointEvent{}); err != nil {
		t.Fatalf("expected nil store to be ignored, got %v", err)
	}
}
