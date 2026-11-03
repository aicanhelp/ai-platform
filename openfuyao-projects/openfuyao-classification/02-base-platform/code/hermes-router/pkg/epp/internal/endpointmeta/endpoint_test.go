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

package endpointmeta

import (
	"testing"

	metav1types "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	fwkscheduling "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	testNamespace        = "default"
	testEndpointName     = "prefill-a"
	testEndpointAddress  = "10.0.0.10"
	testEndpointPort     = "8080"
	testEndpointScore    = 0.25
	testRoleLabelKey     = "role"
	testRoleLabelValue   = "prefill"
	testMutatedPodName   = "prefill-mutated"
	testMutatedLabel     = "mutated"
	testMutatedQueueSize = 42
)

func TestEndpointAddressReturnsMetadataAddress(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint

	if got := EndpointAddress(endpoint); got != testEndpointAddress {
		t.Fatalf("expected endpoint address %q, got %q", testEndpointAddress, got)
	}
}

func TestEndpointHelpersHandleNilInput(t *testing.T) {
	tests := []struct {
		name  string
		check func(t *testing.T)
	}{
		{
			name: "metadata",
			check: func(t *testing.T) {
				if got := MetadataOf(nil); got != nil {
					t.Fatalf("expected nil metadata, got %#v", got)
				}
			},
		},
		{
			name: "pod name",
			check: func(t *testing.T) {
				if got := EndpointPodName(nil); got != "" {
					t.Fatalf("expected empty pod name, got %q", got)
				}
			},
		},
		{
			name: "address",
			check: func(t *testing.T) {
				if got := EndpointAddress(nil); got != "" {
					t.Fatalf("expected empty address, got %q", got)
				}
			},
		},
		{
			name: "port",
			check: func(t *testing.T) {
				if got := EndpointPort(nil); got != "" {
					t.Fatalf("expected empty port, got %q", got)
				}
			},
		},
		{
			name: "label value",
			check: func(t *testing.T) {
				if got, ok := EndpointLabelValue(nil, testRoleLabelKey); ok || got != "" {
					t.Fatalf("expected missing label lookup, got value=%q ok=%t", got, ok)
				}
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, test.check)
	}
}

func TestEndpointPodNameReturnsMetadataPodName(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint

	if got := EndpointPodName(endpoint); got != testEndpointName {
		t.Fatalf("expected endpoint pod name %q, got %q", testEndpointName, got)
	}
}

func TestEndpointPortReturnsMetadataPort(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint

	if got := EndpointPort(endpoint); got != testEndpointPort {
		t.Fatalf("expected endpoint port %q, got %q", testEndpointPort, got)
	}
}

func TestEndpointLabelValueReturnsMetadataLabel(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint

	if got, ok := EndpointLabelValue(endpoint, testRoleLabelKey); !ok || got != testRoleLabelValue {
		t.Fatalf("expected label %q=%q, got value=%q ok=%t", testRoleLabelKey, testRoleLabelValue, got, ok)
	}
}

func TestEndpointLabelValueReturnsMissingLabel(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint

	if got, ok := EndpointLabelValue(endpoint, "missing"); ok || got != "" {
		t.Fatalf("expected missing label lookup, got value=%q ok=%t", got, ok)
	}
}

func TestNormalizeLabelValuesTrimsDeduplicatesAndDropsEmpty(t *testing.T) {
	got := NormalizeLabelValues([]string{" prefill ", "", "decode", "prefill", "   ", "decode"})
	if len(got) != 2 {
		t.Fatalf("expected 2 normalized values, got %d", len(got))
	}
	if got[0] != "prefill" || got[1] != "decode" {
		t.Fatalf("unexpected normalized values: %#v", got)
	}
}

func TestEndpointLabelValueAllowedMatchesConfiguredSet(t *testing.T) {
	endpoint := newTestScoredEndpoint().Endpoint
	allowed := LabelValueSet([]string{testRoleLabelValue})

	if got, ok := EndpointLabelValueAllowed(endpoint, testRoleLabelKey, allowed); !ok || got != testRoleLabelValue {
		t.Fatalf("expected allowed label %q, got value=%q ok=%t", testRoleLabelValue, got, ok)
	}
	if got, ok := EndpointLabelValueAllowed(endpoint, testRoleLabelKey, LabelValueSet([]string{"decode"})); ok || got != "" {
		t.Fatalf("expected disallowed label lookup to fail, got value=%q ok=%t", got, ok)
	}
	if got, ok := EndpointLabelValueAllowed(endpoint, testRoleLabelKey, nil); !ok || got != testRoleLabelValue {
		t.Fatalf("expected nil allowed set to accept existing label, got value=%q ok=%t", got, ok)
	}
}

func TestCloneScoredEndpointDeepCopiesEndpointMetadata(t *testing.T) {
	original := newTestScoredEndpoint()

	cloned := CloneScoredEndpoint(original)
	clonedMetadata := cloned.Endpoint.GetMetadata()
	clonedMetadata.PodName = testMutatedPodName
	clonedMetadata.Labels[testRoleLabelKey] = testMutatedLabel
	cloned.Endpoint.GetMetrics().WaitingQueueSize = testMutatedQueueSize

	originalMetadata := original.Endpoint.GetMetadata()
	if originalMetadata.PodName != testEndpointName {
		t.Fatalf("expected original pod name %q, got %q", testEndpointName, originalMetadata.PodName)
	}
	if got := originalMetadata.Labels[testRoleLabelKey]; got != testRoleLabelValue {
		t.Fatalf("expected original label %q, got %q", testRoleLabelValue, got)
	}
	if got := original.Endpoint.GetMetrics().WaitingQueueSize; got == testMutatedQueueSize {
		t.Fatalf("expected original metrics to remain unchanged, got waiting queue size %d", got)
	}
}

func TestCloneScoredEndpointHandlesNilEndpoint(t *testing.T) {
	cloned := CloneScoredEndpoint(fwkscheduling.ScoredEndpoint{Score: testEndpointScore})

	if cloned.Endpoint != nil {
		t.Fatalf("expected nil endpoint, got %v", cloned.Endpoint)
	}
	if cloned.Score != testEndpointScore {
		t.Fatalf("expected score %.2f, got %.2f", testEndpointScore, cloned.Score)
	}
}

func TestCloneScoredEndpointMaterializesNilMetadata(t *testing.T) {
	originalMetrics := fwkdl.NewMetrics()
	originalMetrics.WaitingQueueSize = 3
	original := fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(nil, originalMetrics, fwkdl.NewAttributes()),
		Score:    testEndpointScore,
	}

	cloned := CloneScoredEndpoint(original)

	if original.Endpoint.GetMetadata() != nil {
		t.Fatalf("expected original metadata to remain nil, got %#v", original.Endpoint.GetMetadata())
	}
	clonedMetadata := cloned.Endpoint.GetMetadata()
	if clonedMetadata == nil {
		t.Fatal("expected cloned metadata to be initialized")
	}
	clonedMetadata.PodName = testMutatedPodName
	if original.Endpoint.GetMetadata() != nil {
		t.Fatalf("expected original metadata to remain nil after clone mutation, got %#v", original.Endpoint.GetMetadata())
	}
	if got := cloned.Endpoint.GetMetrics().WaitingQueueSize; got != originalMetrics.WaitingQueueSize {
		t.Fatalf("expected cloned metrics waiting queue size %d, got %d", originalMetrics.WaitingQueueSize, got)
	}
}

func TestCloneScoredEndpointMaterializesNilMetrics(t *testing.T) {
	originalMetadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: testNamespace, Name: testEndpointName},
		PodName:        testEndpointName,
		Address:        testEndpointAddress,
		Port:           testEndpointPort,
		Labels: map[string]string{
			testRoleLabelKey: testRoleLabelValue,
		},
	}
	original := fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(originalMetadata, nil, fwkdl.NewAttributes()),
		Score:    testEndpointScore,
	}

	cloned := CloneScoredEndpoint(original)

	if original.Endpoint.GetMetrics() != nil {
		t.Fatalf("expected original metrics to remain nil, got %#v", original.Endpoint.GetMetrics())
	}
	clonedMetrics := cloned.Endpoint.GetMetrics()
	if clonedMetrics == nil {
		t.Fatal("expected cloned metrics to be initialized")
	}
	clonedMetrics.WaitingQueueSize = testMutatedQueueSize
	if original.Endpoint.GetMetrics() != nil {
		t.Fatalf("expected original metrics to remain nil after clone mutation, got %#v", original.Endpoint.GetMetrics())
	}
	if got := cloned.Endpoint.GetMetadata().PodName; got != testEndpointName {
		t.Fatalf("expected cloned pod name %q, got %q", testEndpointName, got)
	}
}

func newTestScoredEndpoint() fwkscheduling.ScoredEndpoint {
	return newTestScoredEndpointWithLabels(map[string]string{
		testRoleLabelKey: testRoleLabelValue,
	})
}

func newTestScoredEndpointWithLabels(labels map[string]string) fwkscheduling.ScoredEndpoint {
	metrics := fwkdl.NewMetrics()
	metrics.WaitingQueueSize = 2

	copiedLabels := make(map[string]string, len(labels))
	for key, value := range labels {
		copiedLabels[key] = value
	}

	metadata := &fwkdl.EndpointMetadata{
		NamespacedName: metav1types.NamespacedName{Namespace: testNamespace, Name: testEndpointName},
		PodName:        testEndpointName,
		Address:        testEndpointAddress,
		Port:           testEndpointPort,
		Labels:         copiedLabels,
	}

	return fwkscheduling.ScoredEndpoint{
		Endpoint: fwkscheduling.NewEndpoint(metadata, metrics, fwkdl.NewAttributes()),
		Score:    testEndpointScore,
	}
}
