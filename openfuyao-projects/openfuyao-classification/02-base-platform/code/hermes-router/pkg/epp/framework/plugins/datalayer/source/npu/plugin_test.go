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

package npu

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apitypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
)

type resolverStub struct {
	targetURL string
	err       error
}

func (r *resolverStub) resolveTargetURL(_ context.Context, _, _ string) (string, error) {
	if r.err != nil {
		return "", r.err
	}
	return r.targetURL, nil
}

func TestPollSkipsEndpointWithEmptyIdentity(t *testing.T) {
	var calls atomic.Int64
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		calls.Add(1)
		fmt.Fprint(w, "# TYPE container_npu_utilization gauge\ncontainer_npu_utilization{id=\"0\",namespace=\"\",pod_name=\"pod-a\"} 7\n")
	}))
	defer server.Close()

	parsedURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	params := Parameters{
		Path:                     "/metrics",
		ExporterNamespace:        "npu-exporter",
		ExporterPodLabelSelector: "app=npu-exporter",
		ExporterPort:             8082,
		CacheTTL:                 Duration(time.Second),
		ScrapeTimeout:            Duration(time.Second),
	}
	params.applyDefaults()
	source := newDataSourceWithResolver("npu-source", params, &resolverStub{targetURL: "http://" + parsedURL.Host + "/metrics"})

	cases := map[string]*fwkdl.EndpointMetadata{
		"empty namespace": {
			NamespacedName: apitypes.NamespacedName{Name: "pod-a"},
			PodName:        "pod-a",
		},
		"empty pod name": {
			NamespacedName: apitypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
		},
	}
	for name, metadata := range cases {
		t.Run(name, func(t *testing.T) {
			out, err := source.Poll(context.Background(), fwkdl.NewEndpoint(metadata, nil))
			if err != nil {
				t.Fatalf("Poll: %v", err)
			}
			pod, ok := out.(Pod)
			if !ok {
				t.Fatalf("Poll returned %T, want Pod", out)
			}
			if pod.Matched || pod.DeviceCount != 0 {
				t.Fatalf("expected unmatched pod for empty endpoint identity, got %+v", pod)
			}
		})
	}
	if got := calls.Load(); got != 0 {
		t.Fatalf("expected empty endpoint identity to skip scrapes, got %d scrape calls", got)
	}
}

func TestPollUsesDiscoveredExporterTarget(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		fmt.Fprint(w, "# TYPE container_npu_utilization gauge\ncontainer_npu_utilization{id=\"0\",namespace=\"ns\",pod_name=\"pod-a\"} 7\n")
	}))
	defer server.Close()

	parsedURL, err := url.Parse(server.URL)
	if err != nil {
		t.Fatalf("parse server URL: %v", err)
	}
	params := Parameters{
		Path:                     "/metrics",
		ExporterNamespace:        "npu-exporter",
		ExporterPodLabelSelector: "app=npu-exporter",
		ExporterPort:             8082,
		CacheTTL:                 Duration(time.Second),
		ScrapeTimeout:            Duration(time.Second),
	}
	params.applyDefaults()
	source := newDataSourceWithResolver("npu-source", params, &resolverStub{targetURL: "http://" + parsedURL.Host + "/metrics"})
	source.client = server.Client()

	out, err := source.Poll(context.Background(), fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: apitypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
		PodName:        "pod-a",
	}, nil))
	if err != nil {
		t.Fatalf("Poll: %v", err)
	}
	pod, ok := out.(Pod)
	if !ok {
		t.Fatalf("Poll returned %T, want Pod", out)
	}
	if !pod.Matched || pod.DeviceCount != 1 {
		t.Fatalf("unexpected pod: %+v", pod)
	}
}

func TestPollReturnsUnmatchedWhenDiscoveryFails(t *testing.T) {
	params := Parameters{
		Path:                     "/metrics",
		ExporterNamespace:        "npu-exporter",
		ExporterPodLabelSelector: "app=npu-exporter",
		ExporterPort:             8082,
		CacheTTL:                 Duration(time.Second),
		ScrapeTimeout:            Duration(time.Second),
	}
	params.applyDefaults()
	source := newDataSourceWithResolver("npu-source", params, &resolverStub{err: errors.New("boom")})

	out, err := source.Poll(context.Background(), fwkdl.NewEndpoint(&fwkdl.EndpointMetadata{
		NamespacedName: apitypes.NamespacedName{Namespace: "ns", Name: "pod-a"},
		PodName:        "pod-a",
	}, nil))
	if err != nil {
		t.Fatalf("Poll: %v", err)
	}
	pod, ok := out.(Pod)
	if !ok {
		t.Fatalf("Poll returned %T, want Pod", out)
	}
	if pod.Matched || pod.DeviceCount != 0 {
		t.Fatalf("expected unmatched pod, got %+v", pod)
	}
}

func TestNewDataSourceBuildsResolverFromKubeClient(t *testing.T) {
	params := validParameters()
	source, err := NewDataSource("npu-source", params, newFakePodClient(t,
		newWorkloadPod("ns", "pod-a", "n1"),
		newExporterPod("npu-exporter", "exporter-a", "n1", "10.0.0.2", corev1.PodRunning),
	))
	if err != nil {
		t.Fatalf("NewDataSource: %v", err)
	}
	if source.resolver == nil {
		t.Fatal("expected resolver to be initialized")
	}
}
