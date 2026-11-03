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

package prerequest

import (
	"context"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"hermes-router/pkg/epp/internal/pdgroup"
	"hermes-router/pkg/epp/internal/utils"
)

const (
	testPrimaryProfileName = "primary"
	testLeaderName         = "leader"
	testLeaderAddress      = "10.0.0.1"
	testPrefillName        = "prefill"
	testPrefillAddress     = "10.0.0.2"
	testDecodeName         = "decode"
	testDecodeAddress      = "10.0.0.3"
	testEndpointPort       = "8080"
	testHeaderHandlerName  = "header-handler"
)

func TestExtractPrimaryRouteReadsSelectedLeaderPods(t *testing.T) {
	leader := newTestEndpoint(testLeaderName, testLeaderAddress)
	prefill := newTestEndpoint(testPrefillName, testPrefillAddress)
	decode := newTestEndpoint(testDecodeName, testDecodeAddress)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:            "group-a",
		PrefillPods:        []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:         []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{Endpoint: prefill, Score: 0},
		SelectedDecodePod:  &fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0},
	})
	schedulingResult := &fwkscheduling.SchedulingResult{
		PrimaryProfileName: testPrimaryProfileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			testPrimaryProfileName: {
				TargetEndpoints: []fwkscheduling.Endpoint{leader},
			},
		},
	}

	route := ExtractPrimaryRoute(context.Background(), schedulingResult)

	if route == nil {
		t.Fatal("expected non-nil route")
	}
	if !isPDRoute(route) {
		t.Fatalf("expected Prefill/Decode route, got %+v", route)
	}
	if route.Leader == nil || route.Leader.Address != testLeaderAddress {
		t.Fatalf("expected leader address %s, got %+v", testLeaderAddress, route.Leader)
	}
	if route.Prefill == nil || route.Prefill.Address != testPrefillAddress {
		t.Fatalf("expected prefill address %s, got %+v", testPrefillAddress, route.Prefill)
	}
	if route.Decode == nil || route.Decode.Address != testDecodeAddress {
		t.Fatalf("expected decode address %s, got %+v", testDecodeAddress, route.Decode)
	}
}

func TestExtractPrimaryRouteReadsDecodeOnlySelection(t *testing.T) {
	leader := newTestEndpoint(testLeaderName, testLeaderAddress)
	decode := newTestEndpoint(testDecodeName, testDecodeAddress)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:           "group-a",
		DecodePods:        []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedDecodePod: &fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0},
	})
	schedulingResult := newTestSchedulingResult(decodeProfileName, leader)

	route := ExtractPrimaryRoute(context.Background(), schedulingResult)

	if route == nil {
		t.Fatal("expected non-nil Decode-only route")
	}
	if !isDecodeOnlyRoute(route) {
		t.Fatalf("expected Decode-only route, got %+v", route)
	}
	if route.Leader == nil || route.Leader.Address != testLeaderAddress {
		t.Fatalf("expected leader address %s, got %+v", testLeaderAddress, route.Leader)
	}
	if route.Prefill != nil {
		t.Fatalf("expected Decode-only route without Prefill, got %+v", route.Prefill)
	}
	if route.Decode == nil || route.Decode.Address != testDecodeAddress {
		t.Fatalf("expected decode address %s, got %+v", testDecodeAddress, route.Decode)
	}
}

func TestExtractPrimaryRouteRejectsDecodeProfileWithPrefillSelection(t *testing.T) {
	leader := newTestEndpoint(testLeaderName, testLeaderAddress)
	prefill := newTestEndpoint(testPrefillName, testPrefillAddress)
	decode := newTestEndpoint(testDecodeName, testDecodeAddress)
	pdgroup.Set(leader, &pdgroup.Info{
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{Endpoint: prefill, Score: 0},
		SelectedDecodePod:  &fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0},
	})

	route := ExtractPrimaryRoute(context.Background(), newTestSchedulingResult(decodeProfileName, leader))
	if route != nil {
		t.Fatalf("expected nil route when Decode profile also selects Prefill, got %+v", route)
	}
}

func TestExtractPrimaryRouteRejectsIncompleteSelectedLeaderPods(t *testing.T) {
	leader := newTestEndpoint(testLeaderName, testLeaderAddress)
	prefill := newTestEndpoint(testPrefillName, testPrefillAddress)
	decode := newTestEndpoint(testDecodeName, testDecodeAddress)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:            "group-a",
		PrefillPods:        []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:         []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{Endpoint: prefill, Score: 0},
	})
	schedulingResult := &fwkscheduling.SchedulingResult{
		PrimaryProfileName: testPrimaryProfileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			testPrimaryProfileName: {
				TargetEndpoints: []fwkscheduling.Endpoint{leader},
			},
		},
	}

	route := ExtractPrimaryRoute(context.Background(), schedulingResult)
	if route != nil {
		t.Fatalf("expected nil route for incomplete selected leader route, got %+v", route)
	}
}

func TestInjectHeadersFromRouteSetsPrefillAndDecode(t *testing.T) {
	request := &fwkscheduling.LLMRequest{}
	route := &Route{
		Leader:  newTestEndpoint(testLeaderName, testLeaderAddress).GetMetadata(),
		Prefill: newTestEndpoint(testPrefillName, testPrefillAddress).GetMetadata(),
		Decode:  newTestEndpoint(testDecodeName, testDecodeAddress).GetMetadata(),
	}

	InjectHeadersFromRoute(context.Background(), request, route)

	if len(request.Headers) != 2 {
		t.Fatalf("expected only Prefill and Decode headers, got %+v", request.Headers)
	}
	if request.Headers[utils.PrefillPodHeader] != testPrefillAddress+":"+testEndpointPort {
		t.Fatalf("expected prefill header to be set, got %q", request.Headers[utils.PrefillPodHeader])
	}
	if request.Headers[utils.DecodePodHeader] != testDecodeAddress+":"+testEndpointPort {
		t.Fatalf("expected decode header to be set, got %q", request.Headers[utils.DecodePodHeader])
	}
}

func TestInjectHeadersFromRouteSetsDecodeOnlyAndRemovesPrefill(t *testing.T) {
	request := &fwkscheduling.LLMRequest{Headers: map[string]string{
		utils.PrefillPodHeader: "stale-prefill:8000",
	}}
	route := &Route{
		Leader: newTestEndpoint(testLeaderName, testLeaderAddress).GetMetadata(),
		Decode: newTestEndpoint(testDecodeName, testDecodeAddress).GetMetadata(),
	}

	InjectHeadersFromRoute(context.Background(), request, route)

	if len(request.Headers) != 1 {
		t.Fatalf("expected only Decode header, got %+v", request.Headers)
	}
	if request.Headers[utils.DecodePodHeader] != testDecodeAddress+":"+testEndpointPort {
		t.Fatalf("expected decode header to be set, got %q", request.Headers[utils.DecodePodHeader])
	}
	if _, ok := request.Headers[utils.PrefillPodHeader]; ok {
		t.Fatal("expected Decode-only route to remove stale Prefill header")
	}
}

func TestInjectHeadersFromRouteIgnoresInvalidRoute(t *testing.T) {
	request := &fwkscheduling.LLMRequest{Headers: map[string]string{"keep": "value"}}
	route := &Route{
		Leader:  newTestEndpoint(testLeaderName, testLeaderAddress).GetMetadata(),
		Prefill: newTestEndpoint(testPrefillName, testPrefillAddress).GetMetadata(),
	}

	InjectHeadersFromRoute(context.Background(), request, route)

	if len(request.Headers) != 1 || request.Headers["keep"] != "value" {
		t.Fatalf("expected invalid route to leave headers unchanged, got %+v", request.Headers)
	}
}

func TestHeaderHandlerPreRequestInjectsHeaders(t *testing.T) {
	handler := NewHeaderHandler().WithName(testHeaderHandlerName)
	request := &fwkscheduling.LLMRequest{}
	leader := newTestEndpoint(testLeaderName, testLeaderAddress)
	prefill := newTestEndpoint(testPrefillName, testPrefillAddress)
	decode := newTestEndpoint(testDecodeName, testDecodeAddress)
	pdgroup.Set(leader, &pdgroup.Info{
		GroupID:            "group-a",
		PrefillPods:        []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0}},
		DecodePods:         []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0}},
		SelectedPrefillPod: &fwkscheduling.ScoredEndpoint{Endpoint: prefill, Score: 0},
		SelectedDecodePod:  &fwkscheduling.ScoredEndpoint{Endpoint: decode, Score: 0},
	})
	result := &fwkscheduling.SchedulingResult{
		PrimaryProfileName: testPrimaryProfileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			testPrimaryProfileName: {
				TargetEndpoints: []fwkscheduling.Endpoint{leader},
			},
		},
	}

	handler.PreRequest(context.Background(), request, result)

	if request.Headers[utils.PrefillPodHeader] == "" {
		t.Fatal("expected prefill header to be injected")
	}
	if request.Headers[utils.DecodePodHeader] == "" {
		t.Fatal("expected decode header to be injected")
	}
	if len(request.Headers) != 2 {
		t.Fatalf("expected only Prefill and Decode headers, got %+v", request.Headers)
	}
}

func newTestSchedulingResult(profileName string, targets ...fwkscheduling.Endpoint) *fwkscheduling.SchedulingResult {
	return &fwkscheduling.SchedulingResult{
		PrimaryProfileName: profileName,
		ProfileResults: map[string]*fwkscheduling.ProfileRunResult{
			profileName: {TargetEndpoints: targets},
		},
	}
}

func newTestEndpoint(name, address string) fwkscheduling.Endpoint {
	return fwkscheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{
			NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
			PodName:        name,
			Address:        address,
			Port:           testEndpointPort,
			Labels:         map[string]string{},
		},
		fwkdl.NewMetrics(),
		fwkdl.NewAttributes(),
	)
}
