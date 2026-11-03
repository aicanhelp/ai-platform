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

package pdgroup

import (
	"maps"
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

func TestSetAndGetRoundTripsInfo(t *testing.T) {
	leader := newTestEndpoint("leader", nil)
	info := &Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newTestEndpoint("prefill", nil), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newTestEndpoint("decode", nil), Score: 0.2}},
	}

	Set(leader, info)

	got, ok := Get(leader)
	if !ok {
		t.Fatal("expected Info attribute to be present")
	}
	if got == info {
		t.Fatal("expected stored Info to be cloned")
	}
	if got.GroupID != info.GroupID {
		t.Fatalf("expected group ID %q, got %q", info.GroupID, got.GroupID)
	}
	if len(got.PrefillPods) != 1 || len(got.DecodePods) != 1 {
		t.Fatalf("unexpected pod counts: %+v", got)
	}
	if !fwkscheduling.EndpointComparer(got.PrefillPods[0].Endpoint, info.PrefillPods[0].Endpoint) {
		t.Fatal("expected cloned prefill endpoint to compare equal to source")
	}
	if got.PrefillPods[0].Endpoint == info.PrefillPods[0].Endpoint {
		t.Fatal("expected cloned prefill endpoint to be a distinct object")
	}
}

func TestGetDirectReturnsStoredReference(t *testing.T) {
	leader := newTestEndpoint("leader", nil)
	info := &Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: newTestEndpoint("prefill", nil), Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: newTestEndpoint("decode", nil), Score: 0.2}},
	}

	Set(leader, info)

	direct, ok := GetDirect(leader)
	if !ok {
		t.Fatal("expected Info attribute to be present")
	}
	cloned, ok := Get(leader)
	if !ok {
		t.Fatal("expected cloned Info attribute to be present")
	}
	if direct == cloned {
		t.Fatal("expected Get to return a clone distinct from GetDirect")
	}
	if direct.GroupID != info.GroupID {
		t.Fatalf("expected group ID %q, got %q", info.GroupID, direct.GroupID)
	}
}

func TestGroupMemberContainsEndpoint(t *testing.T) {
	prefill := newTestEndpoint("prefill", nil)
	decode := newTestEndpoint("decode", nil)
	info := &Info{
		GroupID:     "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0.2}},
	}

	if !ContainsMember(info, prefill) {
		t.Fatal("expected prefill endpoint to be recognized as member")
	}
	if !ContainsMember(info, decode) {
		t.Fatal("expected decode endpoint to be recognized as member")
	}
	if ContainsMember(info, newTestEndpoint("aggregate", nil)) {
		t.Fatal("did not expect aggregate endpoint to be recognized as member")
	}
}

func TestValidateRejectsMissingMembers(t *testing.T) {
	info := &Info{GroupID: "group-a"}
	if err := Validate(info); err == nil {
		t.Fatal("expected validation error when members are missing")
	}
}

func TestInspectMembersCountsMissingPredictionsAndMetrics(t *testing.T) {
	prefillReady := newTestEndpoint("prefill-ready", nil)
	prefillMissingPrediction := newTestEndpointWithoutMetrics("prefill-missing-prediction", nil)
	decodeReady := newTestEndpoint("decode-ready", nil)
	decodeMissingMetrics := newTestEndpointWithoutMetrics("decode-missing-metrics", nil)
	info := &Info{
		GroupID: "group-a",
		PrefillPods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: prefillReady, Score: 0.1},
			{Endpoint: prefillMissingPrediction, Score: 0.2},
		},
		DecodePods: []fwkscheduling.ScoredEndpoint{
			{Endpoint: decodeReady, Score: 0.3},
			{Endpoint: decodeMissingMetrics, Score: 0.4},
		},
	}

	inspection := InspectMembers(
		info,
		func(endpoint fwkscheduling.Endpoint) bool {
			return endpoint != nil && endpoint.GetMetadata() != nil && endpoint.GetMetadata().PodName == "prefill-ready"
		},
		func(endpoint fwkscheduling.Endpoint) bool {
			return endpoint != nil && endpoint.GetMetadata() != nil && endpoint.GetMetadata().PodName == "decode-ready"
		},
		func(endpoint fwkscheduling.Endpoint) bool {
			return endpoint != nil && endpoint.GetMetrics() != nil
		},
	)

	if inspection.MissingPrefillPredictions != 1 {
		t.Fatalf("expected one missing prefill prediction, got %d", inspection.MissingPrefillPredictions)
	}
	if inspection.MissingDecodePredictions != 1 {
		t.Fatalf("expected one missing decode prediction, got %d", inspection.MissingDecodePredictions)
	}
	if inspection.MissingPrefillMetrics != 1 {
		t.Fatalf("expected one missing prefill metrics, got %d", inspection.MissingPrefillMetrics)
	}
	if inspection.MissingDecodeMetrics != 1 {
		t.Fatalf("expected one missing decode metrics, got %d", inspection.MissingDecodeMetrics)
	}
	if inspection.PredictionsAvailable() {
		t.Fatal("expected predictions to be incomplete")
	}
	if !inspection.HasMissingPredictions() {
		t.Fatal("expected missing predictions to be reported")
	}
	if !inspection.HasMissingMetrics() {
		t.Fatal("expected missing metrics to be reported")
	}
}

func TestInspectMembersReportsCompleteAvailability(t *testing.T) {
	prefill := newTestEndpoint("prefill", nil)
	decode := newTestEndpoint("decode", nil)
	info := &Info{
		GroupID:     "group-b",
		PrefillPods: []fwkscheduling.ScoredEndpoint{{Endpoint: prefill, Score: 0.1}},
		DecodePods:  []fwkscheduling.ScoredEndpoint{{Endpoint: decode, Score: 0.2}},
	}

	inspection := InspectMembers(
		info,
		func(endpoint fwkscheduling.Endpoint) bool { return endpoint == prefill },
		func(endpoint fwkscheduling.Endpoint) bool { return endpoint == decode },
		func(endpoint fwkscheduling.Endpoint) bool { return endpoint != nil && endpoint.GetMetrics() != nil },
	)

	if !inspection.PredictionsAvailable() {
		t.Fatal("expected predictions to be complete")
	}
	if inspection.HasMissingPredictions() {
		t.Fatal("did not expect missing predictions")
	}
	if inspection.HasMissingMetrics() {
		t.Fatal("did not expect missing metrics")
	}
}

func newTestEndpoint(name string, labels map[string]string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
		Labels:         maps.Clone(labels),
	}
	return fwkscheduling.NewEndpoint(metadata, fwkdl.NewMetrics(), fwkdl.NewAttributes())
}

func newTestEndpointWithoutMetrics(name string, labels map[string]string) fwkscheduling.Endpoint {
	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: "default", Name: name},
		PodName:        name,
		Address:        name + ".svc.cluster.local",
		Port:           "8080",
		Labels:         maps.Clone(labels),
	}
	return fwkscheduling.NewEndpoint(metadata, nil, fwkdl.NewAttributes())
}
