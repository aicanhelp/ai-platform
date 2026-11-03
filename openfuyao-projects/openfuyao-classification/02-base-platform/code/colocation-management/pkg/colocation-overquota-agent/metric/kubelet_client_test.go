/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package metric

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/rest"
	msutils "sigs.k8s.io/metrics-server/pkg/utils"

	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/exporter"
	"openfuyao.com/colocation-management/pkg/common"
)

func testResolver() msutils.NodeAddressResolver {
	return msutils.NewPriorityNodeAddressResolver([]corev1.NodeAddressType{corev1.NodeHostName})
}

func hostNode() *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "n1"},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{{Type: corev1.NodeHostName, Address: "127.0.0.1"}},
			DaemonEndpoints: corev1.NodeDaemonEndpoints{
				KubeletEndpoint: corev1.DaemonEndpoint{Port: 10250},
			},
		},
	}
}

func qosPod(qos string, withBE bool) corev1.Pod {
	c := corev1.Container{Name: "c"}
	if withBE {
		c.Resources = corev1.ResourceRequirements{Requests: corev1.ResourceList{
			common.ExtenderResourceCPU:    resource.MustParse("2000m"),
			common.ExtenderResourceMemory: resource.MustParse("1Gi"),
		}}
	}
	return corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "p-" + qos,
			Namespace:   "ns",
			Annotations: map[string]string{common.QosClassAnnotationKey: qos},
		},
		Spec: corev1.PodSpec{Containers: []corev1.Container{c}},
	}
}

// ---- pure: getBeRequst ----

func TestGetBeRequst(t *testing.T) {
	cpu, mem, err := getBeRequst(qosPod("BE", true), "ns", "p-BE")
	assert.NoError(t, err)
	assert.Equal(t, float64(2), cpu)     // 2000m -> 2 cores
	assert.Equal(t, float64(1<<30), mem) // 1Gi

	cpu2, mem2, err2 := getBeRequst(qosPod("BE", false), "ns", "p-BE") // no extender requests
	assert.NoError(t, err2)
	assert.Equal(t, float64(0), cpu2)
	assert.Equal(t, float64(0), mem2)
}

// ---- pure: buildPodExporter ----

func TestBuildPodExporter(t *testing.T) {
	list := corev1.PodList{Items: []corev1.Pod{
		qosPod("LS", false),
		qosPod("HLS", false),
		qosPod("BE", true),
		qosPod("None", false),
	}}
	pods, err := buildPodExporter(list, "node1")
	assert.NoError(t, err)
	assert.Equal(t, 4, len(pods))
	// BE pod is the 3rd (index 2)
	assert.Equal(t, float64(2), pods[2].BeCpuRequest)
	assert.Equal(t, float64(1<<30), pods[2].BeMemoryRequest)
}

// ---- pure: urlBuild ----

func TestUrlBuild(t *testing.T) {
	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)
	u, err := kc.urlBuild("/metrics/resource", hostNode(), 10250)
	assert.NoError(t, err)
	assert.Equal(t, "/metrics/resource", u.Path)
	assert.Equal(t, "http", u.Scheme)

	t.Run("use node status port", func(t *testing.T) {
		kc2 := newClient(http.DefaultClient, testResolver(), 9999, "http", true)
		u2, err := kc2.urlBuild("/pods", hostNode(), 9999)
		assert.NoError(t, err)
		assert.Contains(t, u2.Host, "10250") // overridden from node status
	})

	t.Run("no resolvable address -> error", func(t *testing.T) {
		_, err := kc.urlBuild("/pods", &corev1.Node{}, 10250)
		assert.Error(t, err)
	})
}

// ---- httptest: doRequest / getPods ----

func TestDoRequest(t *testing.T) {
	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)

	t.Run("200 body", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte("payload"))
		}))
		defer srv.Close()
		b, err := kc.doRequest(context.Background(), srv.URL)
		assert.NoError(t, err)
		assert.Equal(t, "payload", string(b))
	})

	t.Run("non-200 -> error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer srv.Close()
		_, err := kc.doRequest(context.Background(), srv.URL)
		assert.Error(t, err)
	})

	t.Run("unreachable -> error", func(t *testing.T) {
		_, err := kc.doRequest(context.Background(), "http://127.0.0.1:0/notrunning")
		assert.Error(t, err)
	})
}

func TestGetPodsHTTP(t *testing.T) {
	list := corev1.PodList{Items: []corev1.Pod{{
		ObjectMeta: metav1.ObjectMeta{Name: "p1", Namespace: "ns"},
		Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "c1"}}},
	}}}
	body, _ := json.Marshal(list)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)
	pods, err := kc.getPods(context.Background(), srv.URL)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(pods))
	assert.Equal(t, "p1", pods[0].Name)
}

// ---- Get* error paths (urlBuild fails on address-less node) ----

func TestGetMetricsAndPodsErrors(t *testing.T) {
	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)
	emptyNode := &corev1.Node{}
	_, err := kc.GetMetrics(context.Background(), emptyNode)
	assert.Error(t, err)
	_, err = kc.GetPods(context.Background(), emptyNode)
	assert.Error(t, err)
}

// ---- NewForConfig ----

func TestNewForConfig(t *testing.T) {
	t.Run("nil client -> default transport", func(t *testing.T) {
		kc, err := NewForConfig(&KubeletClientConfig{
			AddressTypePriority: []corev1.NodeAddressType{corev1.NodeHostName},
			Scheme:              "http",
			DefaultPort:         10250,
		})
		assert.NoError(t, err)
		assert.NotNil(t, kc)
	})

	t.Run("with rest config", func(t *testing.T) {
		kc, err := NewForConfig(&KubeletClientConfig{
			Client:              &rest.Config{Host: "https://localhost", TLSClientConfig: rest.TLSClientConfig{}},
			AddressTypePriority: []corev1.NodeAddressType{corev1.NodeHostName},
			Scheme:              "https",
			DefaultPort:         10250,
		})
		assert.NoError(t, err)
		assert.NotNil(t, kc)
	})
}

// ---------------------------------------------------------------------------
// Helpers + tests for the previously uncovered HTTP-driven methods:
//   getMetrics, GetMetrics (success), getPodMetrics, getPodsSummary,
//   GetPodMetrics, GetOnlinePods, plus error/decode-failure branches of
//   getPods / getPodMetrics / getPodsSummary / getMetrics.
// ---------------------------------------------------------------------------

// resolverForServer builds a NodeAddressResolver that resolves the supplied
// node's Hostname to 127.0.0.1, and a *corev1.Node that carries that address.
// Combined with a KubeletClient whose defaultPort matches the test server,
// the high-level Get* methods can be driven end-to-end against httptest.
func resolverForServer(t *testing.T) (msutils.NodeAddressResolver, *corev1.Node) {
	t.Helper()
	resolver := msutils.NewPriorityNodeAddressResolver(
		[]corev1.NodeAddressType{corev1.NodeInternalIP, corev1.NodeHostName})
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: "n1"},
		Status: corev1.NodeStatus{
			Addresses: []corev1.NodeAddress{
				{Type: corev1.NodeInternalIP, Address: "127.0.0.1"},
			},
		},
	}
	return resolver, node
}

// serverPort extracts the TCP port from an httptest server URL.
func serverPort(t *testing.T, srvURL string) int {
	t.Helper()
	u, err := url.Parse(srvURL)
	assert.NoError(t, err)
	p, err := strconv.Atoi(u.Port())
	assert.NoError(t, err)
	return p
}

// clientAtServer returns a KubeletClient whose resolver+defaultPort route
// requests to the given httptest server.
func clientAtServer(t *testing.T, srvURL string) *KubeletClient {
	t.Helper()
	resolver, _ := resolverForServer(t)
	return newClient(http.DefaultClient, resolver, serverPort(t, srvURL), "http", false)
}

// validPrometheusResourceMetrics is a minimal but complete /metrics/resource
// payload that decodeBatch accepts (node cpu+mem, plus one pod+container).
func validPrometheusResourceMetrics() []byte {
	return []byte(`
node_cpu_usage_seconds_total{node="n1"} 123.456 1609459200000
node_memory_working_set_bytes{node="n1"} 1048576 1609459200000
pod_cpu_usage_seconds_total{namespace="default",pod="p1"} 1.0 1609459200000
pod_memory_working_set_bytes{namespace="default",pod="p1"} 2048 1609459200000
container_cpu_usage_seconds_total{namespace="default",pod="p1",container="c1"} 1.0 1609459200000
container_memory_working_set_bytes{namespace="default",pod="p1",container="c1"} 2048 1609459200000
`)
}

// ---- getMetrics (private): success and decode-error branches ----

func TestGetMetricsPrivate(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write(validPrometheusResourceMetrics())
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		mb, err := kc.getMetrics(context.Background(), srv.URL+"/metrics/resource", "n1")
		assert.NoError(t, err)
		assert.NotNil(t, mb)
		// node cpu/mem both present -> Node populated
		assert.NotEqual(t, uint64(0), mb.Node.CumulativeCpuUsed)
		assert.NotEqual(t, uint64(0), mb.Node.MemoryUsage)
	})

	t.Run("decode error (malformed prometheus data)", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{invalid`))
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		_, err := kc.getMetrics(context.Background(), srv.URL+"/metrics/resource", "n1")
		assert.Error(t, err)
	})

	t.Run("http error (non-200)", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusBadGateway)
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		_, err := kc.getMetrics(context.Background(), srv.URL+"/metrics/resource", "n1")
		assert.Error(t, err)
	})
}

// ---- GetMetrics: success end-to-end against httptest ----

func TestGetMetricsSuccess(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(validPrometheusResourceMetrics())
	}))
	defer srv.Close()

	kc := clientAtServer(t, srv.URL)
	_, node := resolverForServer(t)
	mb, err := kc.GetMetrics(context.Background(), node)
	assert.NoError(t, err)
	assert.NotNil(t, mb)
	assert.NotEqual(t, uint64(0), mb.Node.CumulativeCpuUsed)
}

// ---- getPodMetrics (private) ----

func TestGetPodMetricsPrivate(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		list := corev1.PodList{Items: []corev1.Pod{
			qosPod("BE", true),
			qosPod("LS", false),
		}}
		body, _ := json.Marshal(list)
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		pods, err := kc.getPodMetrics(context.Background(), srv.URL+"/pods", "n1")
		assert.NoError(t, err)
		assert.Equal(t, 2, len(pods))
		// BE pod has populated BE requests (2000m -> 2 cores, 1Gi)
		var beFound bool
		for _, p := range pods {
			if p.QosLevel == string(common.QosBe) {
				beFound = true
				assert.Equal(t, float64(2), p.BeCpuRequest)
				assert.Equal(t, float64(1<<30), p.BeMemoryRequest)
			}
		}
		assert.True(t, beFound)
	})

	t.Run("bad JSON", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{not-json`))
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		_, err := kc.getPodMetrics(context.Background(), srv.URL+"/pods", "n1")
		assert.Error(t, err)
	})
}

// ---- getPodsSummary (private) ----

func TestGetPodsSummaryPrivate(t *testing.T) {
	// Build a stats-summary payload whose first pod matches the exporter
	// named "p1"; a second exporter ("p-missing") has no matching pod.
	summary := exporter.KubeletStatsSummary{}
	summary.Pods = []struct {
		PodRef struct {
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"podRef"`
		CPU        exporter.CPU    `json:"cpu"`
		Memory     exporter.Memory `json:"memory"`
		Containers []struct {
			Name   string          `json:"name"`
			CPU    exporter.CPU    `json:"cpu"`
			Memory exporter.Memory `json:"memory"`
		} `json:"containers"`
	}{
		{
			PodRef: struct {
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			}{Name: "p1", Namespace: "default"},
			CPU:    exporter.CPU{UsageNanoCores: 2_000_000_000}, // 2 cores
			Memory: exporter.Memory{UsageBytes: 1024},
		},
	}
	body, _ := json.Marshal(summary)

	t.Run("success - matching pod gets usage filled", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write(body)
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		// copy exporters so subtests don't share state
		in := []*exporter.PodExport{
			{Namespace: "default", Name: "p1", NodeName: "n1"},
			{Namespace: "default", Name: "p-missing", NodeName: "n1"},
		}
		out, err := kc.getPodsSummary(context.Background(), srv.URL+"/stats/summary", in)
		assert.NoError(t, err)
		assert.Equal(t, 2, len(out))
		assert.Equal(t, float64(2), out[0].CpuUsage)       // 2e9 nano -> 2 cores
		assert.Equal(t, float64(1024), out[0].MemoryUsage) // bytes
		// non-matching exporter untouched
		assert.Equal(t, float64(0), out[1].CpuUsage)
	})

	t.Run("bad JSON", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			_, _ = w.Write([]byte(`{not-json`))
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		_, err := kc.getPodsSummary(context.Background(), srv.URL+"/stats/summary", nil)
		assert.Error(t, err)
	})

	t.Run("http error", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(http.StatusInternalServerError)
		}))
		defer srv.Close()

		kc := clientAtServer(t, srv.URL)
		_, err := kc.getPodsSummary(context.Background(), srv.URL+"/stats/summary", nil)
		assert.Error(t, err)
	})
}

// ---- GetPodMetrics: full happy path through one httptest server ----

func TestGetPodMetricsMethod(t *testing.T) {
	podsBody, _ := json.Marshal(corev1.PodList{Items: []corev1.Pod{
		qosPod("BE", true),
		qosPod("LS", false),
	}})

	summary := exporter.KubeletStatsSummary{}
	summary.Pods = []struct {
		PodRef struct {
			Name      string `json:"name"`
			Namespace string `json:"namespace"`
		} `json:"podRef"`
		CPU        exporter.CPU    `json:"cpu"`
		Memory     exporter.Memory `json:"memory"`
		Containers []struct {
			Name   string          `json:"name"`
			CPU    exporter.CPU    `json:"cpu"`
			Memory exporter.Memory `json:"memory"`
		} `json:"containers"`
	}{
		{
			PodRef: struct {
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			}{Name: "p-BE", Namespace: "ns"},
			CPU:    exporter.CPU{UsageNanoCores: 1_000_000_000},
			Memory: exporter.Memory{UsageBytes: 512},
		},
	}
	summaryBody, _ := json.Marshal(summary)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/pods":
			_, _ = w.Write(podsBody)
		case "/stats/summary":
			_, _ = w.Write(summaryBody)
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer srv.Close()

	kc := clientAtServer(t, srv.URL)
	_, node := resolverForServer(t)
	out, err := kc.GetPodMetrics(context.Background(), node)
	assert.NoError(t, err)
	assert.Equal(t, 2, len(out))
	// find the BE exporter and confirm summary usage was merged in
	var beCpu, beMem float64
	var found bool
	for _, e := range out {
		if e.Name == "p-BE" {
			found = true
			beCpu = e.CpuUsage
			beMem = e.MemoryUsage
		}
	}
	assert.True(t, found)
	assert.Equal(t, float64(1), beCpu) // 1e9 nano -> 1 core
	assert.Equal(t, float64(512), beMem)
}

// GetPodMetrics also returns an error when urlBuild fails (address-less node).

func TestGetPodMetricsErrorBadNode(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{}`))
	}))
	defer srv.Close()

	kc := clientAtServer(t, srv.URL)
	// node with no resolvable address -> urlBuild error
	_, err := kc.GetPodMetrics(context.Background(), &corev1.Node{})
	assert.Error(t, err)
}

// ---- GetOnlinePods ----

func TestGetOnlinePods(t *testing.T) {
	// Mix: online, offline (preemptable=true), and HLS pods.
	list := corev1.PodList{Items: []corev1.Pod{
		{ // online
			ObjectMeta: metav1.ObjectMeta{Name: "online-pod", Namespace: "ns"},
			Spec:       corev1.PodSpec{Containers: []corev1.Container{{Name: "c"}}},
		},
		{ // offline via annotation
			ObjectMeta: metav1.ObjectMeta{
				Name: "offline-pod", Namespace: "ns",
				Annotations: map[string]string{common.PriorityAnnotationKey: "true"},
			},
			Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "c"}}},
		},
		{ // HLS
			ObjectMeta: metav1.ObjectMeta{
				Name: "hls-pod", Namespace: "ns",
				Annotations: map[string]string{common.QosClassAnnotationKey: string(common.QosHLS)},
			},
			Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "c"}}},
		},
	}}
	body, _ := json.Marshal(list)
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write(body)
	}))
	defer srv.Close()

	kc := clientAtServer(t, srv.URL)
	_, node := resolverForServer(t)
	pods, err := kc.GetOnlinePods(context.Background(), node)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(pods))
	assert.Equal(t, "online-pod", pods[0].Name)
}

func TestGetOnlinePodsError(t *testing.T) {
	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)
	// address-less node -> GetPods -> urlBuild error
	_, err := kc.GetOnlinePods(context.Background(), &corev1.Node{})
	assert.Error(t, err)
}

// ---- getPods: bad-JSON error branch ----

func TestGetPodsBadJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{not-json`))
	}))
	defer srv.Close()

	kc := newClient(http.DefaultClient, testResolver(), 10250, "http", false)
	_, err := kc.getPods(context.Background(), srv.URL)
	assert.Error(t, err)
}
