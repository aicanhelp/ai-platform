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

package discovery

import (
	"context"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"testing"
	"time"

	"github.com/go-logr/logr"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes/fake"

	"gitcode.com/openFuyao/cache-indexer/pkg/config"
)

const (
	testManagerNamespace = "ns1"
	testLoopDelay        = 10 * time.Millisecond
)

func newReadyPod(name, ip string, labels, annos map[string]string, ports []corev1.ContainerPort) corev1.Pod {
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        name,
			Namespace:   testManagerNamespace,
			UID:         types.UID("uid-" + name),
			Labels:      labels,
			Annotations: annos,
		},
		Spec: corev1.PodSpec{
			Containers: []corev1.Container{{
				Name: "c", Ports: ports,
			}},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
			PodIP: ip,
			Conditions: []corev1.PodCondition{{
				Type: corev1.PodReady, Status: corev1.ConditionTrue,
			}},
		},
	}
}

func TestBuildSnapshot_FiltersAndPorts(t *testing.T) {
	vllmPrefill := newReadyPod("v1", "10.0.0.1",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "prefill"},
		map[string]string{"openfuyao.com/mooncake-local-hostname": "host-a:14001"},
		[]corev1.ContainerPort{{Name: "zmq-pub", ContainerPort: 5557}},
	)
	vllmDecode := newReadyPod("v2", "10.0.0.2",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "decode"},
		nil,
		[]corev1.ContainerPort{{Name: "zmq-pub", ContainerPort: 5559}},
	)
	vllmAggregate := newReadyPod("v3", "10.0.0.3",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "aggregate"},
		nil,
		[]corev1.ContainerPort{{Name: "zmq-pub", ContainerPort: 5558}},
	)
	legacyAggregate := newReadyPod("v4", "10.0.0.4",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdtype": "aggregate"},
		nil,
		nil,
	)
	master := newReadyPod("m1", "10.0.0.10",
		map[string]string{"openfuyao.com/kvmanager": "mooncake"},
		nil,
		[]corev1.ContainerPort{{Name: "http", ContainerPort: 9003}},
	)
	other := newReadyPod("o1", "10.0.0.20", nil, nil, nil)

	cs := fake.NewSimpleClientset(&vllmPrefill, &vllmDecode, &vllmAggregate, &legacyAggregate, &master, &other)
	cfg := normalizeDiscoveryConfig(config.DiscoveryConfig{
		Namespace:       "ns1",
		RefreshInterval: time.Second,
		VLLM: config.VLLMDiscoveryConfig{
			ZMQPortName: "zmq-pub",
		},
		MooncakeMaster: config.MooncakeMasterDiscoveryConfig{
			HTTPPortName: "http",
			HTTPPort:     9003,
		},
	})
	m := &manager{cfg: cfg, log: logr.Discard(), client: cs}

	pods, err := cs.CoreV1().Pods("ns1").List(context.Background(), metav1.ListOptions{})
	if err != nil {
		t.Fatalf("list: %v", err)
	}
	snap := m.buildSnapshot(pods.Items)
	if got, want := len(snap.VLLMPods), 3; got != want {
		t.Fatalf("vllm pod count = %d, want %d (prefill, decode, and aggregate should be included)", got, want)
	}
	if v2 := snap.VLLMPods["10.0.0.2"]; v2 == nil || v2.PodIP != "10.0.0.2" || v2.ZMQPort != 5559 {
		t.Errorf("decode pod endpoint wrong: %+v", v2)
	}
	if _, ok := snap.VLLMPods["10.0.0.3"]; ok {
		v3 := snap.VLLMPods["10.0.0.3"]
		if v3 == nil || v3.ZMQPort != 5558 {
			t.Errorf("aggregate pod ports wrong: %+v", v3)
		}
	} else {
		t.Errorf("pdRole aggregate pod should be included")
	}
	if _, ok := snap.VLLMPods["10.0.0.4"]; ok {
		t.Errorf("legacy pdtype aggregate pod should be excluded")
	}
	v1 := snap.VLLMPods["10.0.0.1"]
	if v1 == nil || v1.ZMQPort != 5557 {
		t.Errorf("v1 ports wrong: %+v", v1)
	}
	if v1.MooncakeClient.Resolved {
		t.Errorf("v1 mooncake client must not resolve from annotation: %+v", v1.MooncakeClient)
	}
	if snap.MooncakeMasterPod == nil || snap.MooncakeMasterPod.PodIP != "10.0.0.10" || snap.MooncakeMasterPod.HTTPPort != 9003 {
		t.Errorf("master endpoint wrong: %+v", snap.MooncakeMasterPod)
	}
}

func testServerAddress(t *testing.T, srv *httptest.Server) (string, int32) {
	t.Helper()
	host, portText, err := net.SplitHostPort(srv.Listener.Addr().String())
	if err != nil {
		t.Fatalf("split test server addr: %v", err)
	}
	port, err := strconv.Atoi(portText)
	if err != nil {
		t.Fatalf("parse test server port: %v", err)
	}
	return host, int32(port)
}

func discoveryTestConfig(httpPort int32) config.DiscoveryConfig {
	return normalizeDiscoveryConfig(config.DiscoveryConfig{
		Namespace:       "ns1",
		RefreshInterval: time.Second,
		VLLM: config.VLLMDiscoveryConfig{
			ZMQPortName: "zmq-pub",
		},
		MooncakeMaster: config.MooncakeMasterDiscoveryConfig{
			HTTPPortName: "http",
			HTTPPort:     httpPort,
		},
	})
}

func TestRefreshOnce_UsesMasterSegmentsWithoutAnnotation(t *testing.T) {
	const vllmIP = "10.0.0.1"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/get_all_segments" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(vllmIP + ":15111\n"))
	}))
	defer srv.Close()

	masterIP, masterPort := testServerAddress(t, srv)
	vllm := newReadyPod("v1", vllmIP,
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "prefill"},
		nil,
		[]corev1.ContainerPort{{Name: "zmq-pub", ContainerPort: 5557}},
	)
	master := newReadyPod("m1", masterIP,
		map[string]string{"openfuyao.com/kvmanager": "mooncake"},
		nil,
		[]corev1.ContainerPort{{Name: "http", ContainerPort: masterPort}},
	)
	cs := fake.NewSimpleClientset(&vllm, &master)
	reg := NewRegistry(logr.Discard())
	m := &manager{cfg: discoveryTestConfig(masterPort), log: logr.Discard(), client: cs, registry: reg, segments: newSegmentsClient(0)}

	m.refreshOnce(context.Background())

	if m.Snapshot() == nil {
		t.Fatalf("snapshot must update when all vLLM pods have master segments")
	}
	ep, ok := reg.ResolveTransportEndpoint(vllmIP)
	if !ok || string(ep) != vllmIP+":15111" {
		t.Fatalf("resolved endpoint = %q ok=%v, want %s:15111", ep, ok, vllmIP)
	}
	pod, ok := reg.ResolveVLLMPod(vllmIP)
	if !ok || pod.MooncakeClient.Source != "master-segment" {
		t.Fatalf("pod endpoint source = %+v ok=%v", pod, ok)
	}
}

// TestRefreshOnce_FetchFailsL1Unaffected verifies that when /get_all_segments
// returns an HTTP error, the refresh still commits the snapshot so L1 (ZMQ)
// subscriptions can proceed. L3 is naturally disabled.
func TestRefreshOnce_FetchFailsL1Unaffected(t *testing.T) {
	const vllmIP = "10.0.0.1"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		http.Error(w, "boom", http.StatusInternalServerError)
	}))
	defer srv.Close()

	masterIP, masterPort := testServerAddress(t, srv)
	vllm := newReadyPod("v1", vllmIP,
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "prefill"}, nil, nil)
	master := newReadyPod("m1", masterIP,
		map[string]string{"openfuyao.com/kvmanager": "mooncake"}, nil,
		[]corev1.ContainerPort{{Name: "http", ContainerPort: masterPort}})
	cs := fake.NewSimpleClientset(&vllm, &master)
	reg := NewRegistry(logr.Discard())
	m := &manager{cfg: discoveryTestConfig(masterPort), log: logr.Discard(), client: cs, registry: reg, segments: newSegmentsClient(0)}

	m.refreshOnce(context.Background())

	// Snapshot must be committed so L1 can work.
	snap := m.Snapshot()
	if snap == nil {
		t.Fatalf("snapshot must not be nil when /get_all_segments fails; L1 must still work")
	}
	ep := snap.VLLMPods[vllmIP]
	if ep == nil {
		t.Fatalf("vLLM pod missing from snapshot")
	}
	if ep.MooncakeClient.Resolved {
		t.Errorf("MooncakeClient.Resolved must be false when /get_all_segments fails")
	}
	// Registry must not resolve L3 transport endpoint.
	if transportEP, ok := reg.ResolveTransportEndpoint(vllmIP); ok {
		t.Fatalf("registry must not resolve endpoint after failed discovery: %q", transportEP)
	}
}

// TestRefreshOnce_PartialSegmentsMissing verifies that when some vLLM pods
// lack a Mooncake transport_endpoint, the refresh still proceeds. The pod with
// a segment gets L3 enabled; the missing pod only has L1 (ZMQ) active.
func TestRefreshOnce_PartialSegmentsMissing(t *testing.T) {
	const firstIP = "10.0.0.1"
	const missingIP = "10.0.0.2"
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/get_all_segments" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(firstIP + ":15111\n"))
	}))
	defer srv.Close()

	masterIP, masterPort := testServerAddress(t, srv)
	vllmOne := newReadyPod("v1", firstIP,
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "prefill"}, nil, nil)
	vllmTwo := newReadyPod("v2", missingIP,
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "aggregate"}, nil, nil)
	master := newReadyPod("m1", masterIP,
		map[string]string{"openfuyao.com/kvmanager": "mooncake"}, nil,
		[]corev1.ContainerPort{{Name: "http", ContainerPort: masterPort}})
	cs := fake.NewSimpleClientset(&vllmOne, &vllmTwo, &master)
	reg := NewRegistry(logr.Discard())
	m := &manager{cfg: discoveryTestConfig(masterPort), log: logr.Discard(), client: cs, registry: reg, segments: newSegmentsClient(0)}

	m.refreshOnce(context.Background())

	// Snapshot must be updated so L1 (ZMQ) subscriptions can proceed.
	snap := m.Snapshot()
	if snap == nil {
		t.Fatalf("snapshot must not be nil when partial segments are missing")
	}
	if got, want := len(snap.VLLMPods), 2; got != want {
		t.Fatalf("vLLM pod count = %d, want %d (both pods must be in snapshot)", got, want)
	}

	// firstIP has a segment → L3 enabled.
	ep1 := snap.VLLMPods[firstIP]
	if ep1 == nil {
		t.Fatalf("firstIP pod missing from snapshot")
	}
	if !ep1.MooncakeClient.Resolved {
		t.Errorf("firstIP must have Resolved=true, got false")
	}
	if ep, ok := reg.ResolveTransportEndpoint(firstIP); !ok {
		t.Errorf("registry must resolve firstIP transport endpoint")
	} else if ep == "" {
		t.Errorf("registry transport endpoint for firstIP must not be empty")
	}

	// missingIP has no segment → L3 disabled, L1 unaffected.
	ep2 := snap.VLLMPods[missingIP]
	if ep2 == nil {
		t.Fatalf("missingIP pod missing from snapshot")
	}
	if ep2.MooncakeClient.Resolved {
		t.Errorf("missingIP must have Resolved=false")
	}
	if _, ok := reg.ResolveTransportEndpoint(missingIP); ok {
		t.Errorf("registry must not resolve missingIP transport endpoint")
	}
}

type recordingSubscriber struct {
	added       []string
	removed     []string
	masterCalls int
}

func (r *recordingSubscriber) OnVLLMAdded(ep *VLLMPodEndpoint) error {
	r.added = append(r.added, ep.PodIP)
	return nil
}
func (r *recordingSubscriber) OnVLLMRemoved(podIP string) error {
	r.removed = append(r.removed, podIP)
	return nil
}
func (r *recordingSubscriber) OnMooncakeMasterChanged(*MooncakeMasterEndpoint) error {
	r.masterCalls++
	return nil
}

func TestReconciler_DiffDrivesSubscriber(t *testing.T) {
	sub := &recordingSubscriber{}
	r := NewReconciler(logr.Discard(), sub)

	prev := &Snapshot{VLLMPods: map[string]*VLLMPodEndpoint{
		"1.1.1.1": {PodIP: "1.1.1.1", Identity: PodIdentity{UID: "u1"}},
	}}
	next := &Snapshot{VLLMPods: map[string]*VLLMPodEndpoint{
		"2.2.2.2": {PodIP: "2.2.2.2", Identity: PodIdentity{UID: "u2"}},
	}, MooncakeMasterPod: &MooncakeMasterEndpoint{PodIP: "9.9.9.9"}}

	if err := r.Reconcile(prev, next); err != nil {
		t.Fatalf("reconcile: %v", err)
	}
	if len(sub.added) != 1 || sub.added[0] != "2.2.2.2" {
		t.Errorf("added = %v", sub.added)
	}
	if len(sub.removed) != 1 || sub.removed[0] != "1.1.1.1" {
		t.Errorf("removed = %v", sub.removed)
	}
	if sub.masterCalls != 1 {
		t.Errorf("master calls = %d", sub.masterCalls)
	}
}

func TestIsVLLMPod_UsesInferNexPDRoleOnly(t *testing.T) {
	prefill := newReadyPod("prefill", "10.0.0.11",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "prefill"},
		nil, nil,
	)
	aggregate := newReadyPod("aggregate", "10.0.0.13",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "aggregate"},
		nil, nil,
	)
	decode := newReadyPod("decode", "10.0.0.12",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdRole": "decode"},
		nil, nil,
	)
	legacyAggregate := newReadyPod("legacy-aggregate", "10.0.0.14",
		map[string]string{"openfuyao.com/engine": "vllm", "openfuyao.com/pdtype": "aggregate"},
		nil, nil,
	)

	if !isVLLMPod(&prefill) {
		t.Fatalf("expected InferNex prefill pod to be discovered")
	}
	if !isVLLMPod(&aggregate) {
		t.Fatalf("expected InferNex aggregate pod to be discovered")
	}
	if !isVLLMPod(&decode) {
		t.Fatalf("expected InferNex decode pod to be discovered")
	}
	if isVLLMPod(&legacyAggregate) {
		t.Fatalf("expected legacy pdtype aggregate pod to be excluded")
	}
}

func TestRegistry_ResolveTransport(t *testing.T) {
	reg := NewRegistry(logr.Discard())
	snap := &Snapshot{VLLMPods: map[string]*VLLMPodEndpoint{
		"1.1.1.1": {PodIP: "1.1.1.1", MooncakeClient: MooncakeClientEndpoint{Resolved: true, TransportEndpoint: "host-a:14001"}},
		"2.2.2.2": {PodIP: "2.2.2.2"},
	}}
	if err := reg.Update(snap); err != nil {
		t.Fatalf("update: %v", err)
	}
	if ep, ok := reg.ResolveTransportEndpoint("1.1.1.1"); !ok || string(ep) != "host-a:14001" {
		t.Errorf("resolve 1.1.1.1 = %q ok=%v", ep, ok)
	}
	if _, ok := reg.ResolveTransportEndpoint("2.2.2.2"); ok {
		t.Errorf("2.2.2.2 has no master segment, must be unresolved")
	}
}

func TestNewManagerAndLifecycle(t *testing.T) {
	cs := fake.NewSimpleClientset()
	mgr := NewManager(config.DiscoveryConfig{
		Namespace:       testManagerNamespace,
		RefreshInterval: testLoopDelay,
	}, logr.Discard(), cs, nil, nil)
	if mgr == nil {
		t.Fatal("NewManager returned nil")
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	if err := mgr.Start(ctx); err != nil {
		t.Fatalf("Start: %v", err)
	}
	time.Sleep(testLoopDelay)
	if err := mgr.Stop(); err != nil {
		t.Fatalf("Stop: %v", err)
	}
	if err := mgr.Stop(); err != nil {
		t.Fatalf("second Stop: %v", err)
	}
}

func TestManager_StartValidatesConfig(t *testing.T) {
	cs := fake.NewSimpleClientset()
	noNamespace := &manager{cfg: config.DiscoveryConfig{}, log: logr.Discard(), client: cs, done: make(chan struct{})}
	if err := noNamespace.Start(context.Background()); err == nil {
		t.Fatal("expected namespace validation error")
	}
	badInterval := &manager{
		cfg:    config.DiscoveryConfig{Namespace: testManagerNamespace},
		log:    logr.Discard(),
		client: cs,
		done:   make(chan struct{}),
	}
	if err := badInterval.Start(context.Background()); err == nil {
		t.Fatal("expected refresh interval validation error")
	}
}

func TestManager_StopBeforeStartAndRegistrySnapshot(t *testing.T) {
	mgr := &manager{done: make(chan struct{})}
	if err := mgr.Stop(); err != nil {
		t.Fatalf("Stop before Start: %v", err)
	}

	reg := NewRegistry(logr.Discard())
	snap := &Snapshot{VLLMPods: map[string]*VLLMPodEndpoint{
		"1.1.1.1": {PodIP: "1.1.1.1"},
	}}
	if err := reg.Update(nil); err != nil {
		t.Fatalf("Update nil: %v", err)
	}
	if err := reg.Update(snap); err != nil {
		t.Fatalf("Update snapshot: %v", err)
	}
	if got := reg.Snapshot(); got != snap {
		t.Fatalf("Snapshot=%p want=%p", got, snap)
	}
	if _, ok := reg.ResolveVLLMPod("1.1.1.1"); !ok {
		t.Fatal("ResolveVLLMPod should succeed")
	}
}

func TestDiscoveryHelpers(t *testing.T) {
	cfg := normalizeDiscoveryConfig(config.DiscoveryConfig{})
	if cfg.RefreshInterval <= 0 || cfg.MooncakeMaster.HTTPPort <= 0 {
		t.Fatalf("cfg=%+v", cfg)
	}
	if host, port := splitHostPort("host:1234:npu_1"); host != "host" || port != 1234 {
		t.Fatalf("splitHostPort got host=%q port=%d", host, port)
	}
	if host, port := splitHostPort("hostonly"); host != "hostonly" || port != 0 {
		t.Fatalf("splitHostPort without port got host=%q port=%d", host, port)
	}
	if got, ok := atoi32("1234"); !ok || got != 1234 {
		t.Fatalf("atoi32=%d ok=%v", got, ok)
	}
	if _, ok := atoi32("12x"); ok {
		t.Fatal("atoi32 should reject non-digits")
	}
	if copyLabels(nil) != nil {
		t.Fatal("copyLabels(nil) should return nil")
	}
}

func TestBuildSnapshot_PicksLexicographicallySmallestMaster(t *testing.T) {
	cfg := normalizeDiscoveryConfig(config.DiscoveryConfig{
		Namespace:       testManagerNamespace,
		RefreshInterval: time.Second,
	})
	mgr := &manager{cfg: cfg, log: logr.Discard()}
	first := newReadyPod("master-a", "10.0.0.10", map[string]string{"openfuyao.com/kvmanager": "mooncake"}, nil, nil)
	second := newReadyPod("master-b", "10.0.0.11", map[string]string{"openfuyao.com/kvmanager": "mooncake"}, nil, nil)
	snap := mgr.buildSnapshot([]corev1.Pod{second, first})
	if snap.MooncakeMasterPod == nil || snap.MooncakeMasterPod.Identity.Name != "master-a" {
		t.Fatalf("master=%+v want master-a", snap.MooncakeMasterPod)
	}
}
