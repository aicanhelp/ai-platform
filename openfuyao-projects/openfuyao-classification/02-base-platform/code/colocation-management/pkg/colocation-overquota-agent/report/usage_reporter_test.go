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
package report

import (
	"context"
	"encoding/json"
	"errors"
	"net"
	"net/http"
	"net/http/httptest"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"

	agentapps "openfuyao.com/colocation-management/cmd/colocation-overquota-agent/apps"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
	"openfuyao.com/colocation-management/pkg/common"
)

// ---- helpers ----

func newAgentCfg() *agentapps.Configuration {
	return &agentapps.Configuration{
		Node:                "test-node",
		KubeletIP:           "127.0.0.1",
		UsageReportAddr:     "127.0.0.1:9999",
		ReportsBufCapacity:  5,
		MetricsScrapePeriod: 1 * time.Second,
		ReportsUploadPeriod: 1 * time.Second,
		ServerAddress:       "127.0.0.1:0",
	}
}

// newFakeClientset builds a real *kubernetes.Clientset pointed at an httptest
// server that serves the provided nodes. The server is returned so the caller
// can close it.
func newFakeClientset(t *testing.T, nodes ...*corev1.Node) (*kubernetes.Clientset, func()) {
	t.Helper()
	mux := http.NewServeMux()
	mux.HandleFunc("/api/v1/nodes/", func(w http.ResponseWriter, r *http.Request) {
		// path looks like /api/v1/nodes/<name>
		name := r.URL.Path[len("/api/v1/nodes/"):]
		for _, n := range nodes {
			if n.Name == name {
				w.Header().Set("Content-Type", "application/json")
				_ = json.NewEncoder(w).Encode(n)
				return
			}
		}
		http.NotFound(w, r)
	})
	srv := httptest.NewServer(mux)

	cfg := &rest.Config{
		Host: srv.URL,
		ContentConfig: rest.ContentConfig{
			GroupVersion:         &corev1.SchemeGroupVersion,
			NegotiatedSerializer: scheme.Codecs.WithoutConversion(),
		},
	}
	cs, err := kubernetes.NewForConfig(cfg)
	require.NoError(t, err)
	return cs, srv.Close
}

func colocationNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{common.ColocationNodeLabel: "true"},
		},
	}
}

func nonColocationNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{},
		},
	}
}

// ---- peekReportsInBytes ----

func TestPeekReportsInBytes(t *testing.T) {
	t.Run("empty buffer returns error", func(t *testing.T) {
		mr := &MetricsReporter{
			reportBuffer: NewReportBuffer(2),
		}
		_, err := mr.peekReportsInBytes()
		assert.Error(t, err)
	})

	t.Run("populated buffer returns bytes", func(t *testing.T) {
		mr := &MetricsReporter{
			reportBuffer: NewReportBuffer(2),
		}
		mr.reportBuffer.Put(metric.NodeUsageReport{
			Node:      "n1",
			Timestamp: metav1.Time{Time: time.Now()},
		})
		b, err := mr.peekReportsInBytes()
		require.NoError(t, err)
		assert.NotEmpty(t, b)
	})
}

// ---- doMetricsUpload ----

func TestDoMetricsUpload(t *testing.T) {
	t.Run("against httptest server returns non-nil response", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("{}"))
		}))
		defer srv.Close()

		mr := &MetricsReporter{
			ctx:    context.Background(),
			config: newAgentCfg(),
		}
		// Override UsageReportAddr to the test server's host:port.
		mr.config.UsageReportAddr = srv.Listener.Addr().String()

		resp := mr.doMetricsUpload(nil, []byte(`[{"node":"n1"}]`))
		require.NotNil(t, resp)
		defer resp.Body.Close()
		assert.Equal(t, http.StatusOK, resp.StatusCode)
	})

	t.Run("bad URL host returns nil", func(t *testing.T) {
		mr := &MetricsReporter{
			ctx:    context.Background(),
			config: newAgentCfg(),
		}
		// A host with spaces is an invalid URL -> http.NewRequest fails.
		mr.config.UsageReportAddr = "bad host with spaces:9999"

		resp := mr.doMetricsUpload(nil, []byte(`[]`))
		assert.Nil(t, resp)
	})
}

// ---- uploadMetrics ----

func TestUploadMetrics(t *testing.T) {
	t.Run("empty buffer returns nil,false", func(t *testing.T) {
		mr := &MetricsReporter{
			ctx:          context.Background(),
			config:       newAgentCfg(),
			reportBuffer: NewReportBuffer(2),
		}
		resp, ok := mr.uploadMetrics(time.Now())
		assert.Nil(t, resp)
		assert.False(t, ok)
	})

	t.Run("server 200 returns resp,true", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()

		mr := &MetricsReporter{
			ctx:          context.Background(),
			config:       newAgentCfg(),
			reportBuffer: NewReportBuffer(2),
		}
		mr.config.UsageReportAddr = srv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "n1", Timestamp: metav1.Time{Time: time.Now()}})

		resp, ok := mr.uploadMetrics(time.Now())
		require.NotNil(t, resp)
		defer resp.Body.Close()
		assert.True(t, ok)
	})

	t.Run("server non-200 returns resp,false", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer srv.Close()

		mr := &MetricsReporter{
			ctx:          context.Background(),
			config:       newAgentCfg(),
			reportBuffer: NewReportBuffer(2),
		}
		mr.config.UsageReportAddr = srv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "n1", Timestamp: metav1.Time{Time: time.Now()}})

		resp, ok := mr.uploadMetrics(time.Now())
		require.NotNil(t, resp)
		defer resp.Body.Close()
		assert.False(t, ok)
		assert.Equal(t, http.StatusInternalServerError, resp.StatusCode)
	})

	t.Run("unreachable server returns nil,false", func(t *testing.T) {
		mr := &MetricsReporter{
			ctx:          context.Background(),
			config:       newAgentCfg(),
			reportBuffer: NewReportBuffer(2),
		}
		// Use a port that is not listening.
		mr.config.UsageReportAddr = "127.0.0.1:1"
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "n1", Timestamp: metav1.Time{Time: time.Now()}})

		resp, ok := mr.uploadMetrics(time.Now())
		assert.Nil(t, resp)
		assert.False(t, ok)
	})
}

// ---- getLastReporter ----

func TestGetLastReporter(t *testing.T) {
	t.Run("empty buffer returns usage metrics empty", func(t *testing.T) {
		mr := &MetricsReporter{
			reportBuffer: NewReportBuffer(2),
		}
		req := httptest.NewRequest(http.MethodGet, "/debug/usage", nil)
		w := httptest.NewRecorder()
		mr.getLastReporter(w, req)
		// implementation writes http.StatusOK + error text
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Body.String(), "usage metrics are empty")
	})

	t.Run("populated buffer returns JSON summary", func(t *testing.T) {
		mr := &MetricsReporter{
			reportBuffer: NewReportBuffer(2),
		}
		mr.reportBuffer.Put(metric.NodeUsageReport{
			Node:      "n1",
			Timestamp: metav1.Time{Time: time.Unix(1700_000_000, 0)},
			Window:    metav1.Duration{Duration: time.Second},
			Containers: map[metric.PodRef]map[string]corev1.ResourceList{
				metric.NewPodRef("ns", "pod1"): {
					"c1": corev1.ResourceList{corev1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI)},
				},
			},
		})

		req := httptest.NewRequest(http.MethodGet, "/debug/usage", nil)
		w := httptest.NewRecorder()
		mr.getLastReporter(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Contains(t, w.Header().Get("Content-Type"), "application/json")
		assert.NotEmpty(t, w.Body.Bytes())
	})

	t.Run("POST method not allowed", func(t *testing.T) {
		mr := &MetricsReporter{
			reportBuffer: NewReportBuffer(2),
		}
		req := httptest.NewRequest(http.MethodPost, "/debug/usage", nil)
		w := httptest.NewRecorder()
		mr.getLastReporter(w, req)
		assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	})
}

// ---- getKubeletPods ----

func TestGetKubeletPods(t *testing.T) {
	t.Run("non-GET method not allowed", func(t *testing.T) {
		mr := &MetricsReporter{}
		req := httptest.NewRequest(http.MethodPost, "/debug/pods", nil)
		w := httptest.NewRecorder()
		mr.getKubeletPods(w, req)
		assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	})
}

// ---- DeleteMetricsReporter ----

func TestDeleteMetricsReporter(t *testing.T) {
	mr := &MetricsReporter{
		config: newAgentCfg(),
	}

	t.Run("non-pod object is ignored", func(t *testing.T) {
		// should not panic and should simply return
		assert.NotPanics(t, func() {
			mr.DeleteMetricsReporter("not-a-pod")
		})
	})

	t.Run("real pod object is processed", func(t *testing.T) {
		pod := &corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: "pod1"},
		}
		assert.NotPanics(t, func() {
			mr.DeleteMetricsReporter(pod)
		})
	})
}

// ---- Shutdown ----

func TestShutdown(t *testing.T) {
	t.Run("nil exporterServer is a no-op", func(t *testing.T) {
		mr := &MetricsReporter{}
		assert.NotPanics(t, func() { mr.Shutdown() })
	})

	t.Run("with an exporterServer shuts it down", func(t *testing.T) {
		srv := &http.Server{Addr: "127.0.0.1:0", Handler: http.NewServeMux()}
		// Start it so Shutdown has something to do.
		go func() { _ = srv.ListenAndServe() }()
		// Give it a moment to bind.
		time.Sleep(50 * time.Millisecond)

		mr := &MetricsReporter{exporterServer: srv}
		assert.NotPanics(t, func() { mr.Shutdown() })
	})
}

// ---- isColocationNode ----

func TestIsColocationNode(t *testing.T) {
	t.Run("node with colocation label returns true", func(t *testing.T) {
		node := colocationNode("test-node")
		cs, cleanup := newFakeClientset(t, node)
		defer cleanup()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
		}
		assert.True(t, mr.isColocationNode())
	})

	t.Run("node without colocation label returns false", func(t *testing.T) {
		node := nonColocationNode("test-node")
		cs, cleanup := newFakeClientset(t, node)
		defer cleanup()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
		}
		assert.False(t, mr.isColocationNode())
	})

	t.Run("node not found returns false", func(t *testing.T) {
		cs, cleanup := newFakeClientset(t) // no nodes
		defer cleanup()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
		}
		assert.False(t, mr.isColocationNode())
	})
}

// ---- uploadTick ----

func TestUploadTick(t *testing.T) {
	now := time.Now()

	t.Run("not a colocation node - early return", func(t *testing.T) {
		node := nonColocationNode("test-node")
		cs, cleanup := newFakeClientset(t, node)
		defer cleanup()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
			reportBuffer:    NewReportBuffer(2),
			buffers:         newBufPool(),
		}
		// even with a populated buffer, uploadTick should early-return
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "n1", Timestamp: metav1.Time{Time: now}})
		assert.NotPanics(t, func() { mr.uploadTick(now) })
	})

	t.Run("colocation node, server returns codeSuccess", func(t *testing.T) {
		node := colocationNode("test-node")
		cs, nodeCleanup := newFakeClientset(t, node)
		defer nodeCleanup()

		lastTS := metav1.Time{Time: now.Add(1 * time.Hour)}
		uploadSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			// echo back a success response
			resp := metric.NodeUsageResponse[metav1.Time]{
				Code: codeSuccess,
				Msg:  "",
				Data: lastTS,
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		}))
		defer uploadSrv.Close()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
			reportBuffer:    NewReportBuffer(2),
			buffers:         newBufPool(),
		}
		mr.config.UsageReportAddr = uploadSrv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "test-node", Timestamp: metav1.Time{Time: now}})

		assert.NotPanics(t, func() { mr.uploadTick(now) })
		// on success, lastUsageTimestampCollected is updated
		assert.Equal(t, lastTS.Time.Unix(), mr.lastUsageTimestampCollected.Time.Unix())
	})

	t.Run("colocation node, server returns codeUsageTimestampExpired branch", func(t *testing.T) {
		node := colocationNode("test-node")
		cs, nodeCleanup := newFakeClientset(t, node)
		defer nodeCleanup()

		lastTS := metav1.Time{Time: now.Add(2 * time.Hour)}
		uploadSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := metric.NodeUsageResponse[metav1.Time]{
				Code: codeUsageTimestampExpired, // >= codeUsageTimestampExpired
				Msg:  "expired",
				Data: lastTS,
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		}))
		defer uploadSrv.Close()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
			reportBuffer:    NewReportBuffer(2),
			buffers:         newBufPool(),
		}
		mr.config.UsageReportAddr = uploadSrv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "test-node", Timestamp: metav1.Time{Time: now}})

		assert.NotPanics(t, func() { mr.uploadTick(now) })
		assert.Equal(t, lastTS.Time.Unix(), mr.lastUsageTimestampCollected.Time.Unix())
	})

	t.Run("colocation node, server returns low error code branch", func(t *testing.T) {
		node := colocationNode("test-node")
		cs, nodeCleanup := newFakeClientset(t, node)
		defer nodeCleanup()

		uploadSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			resp := metric.NodeUsageResponse[metav1.Time]{
				Code: codeReportBufferFull, // < codeUsageTimestampExpired, != codeSuccess
				Msg:  "buffer full",
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(resp)
		}))
		defer uploadSrv.Close()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
			reportBuffer:    NewReportBuffer(2),
			buffers:         newBufPool(),
		}
		mr.config.UsageReportAddr = uploadSrv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "test-node", Timestamp: metav1.Time{Time: now}})

		assert.NotPanics(t, func() { mr.uploadTick(now) })
		// on low-error branch, lastUsageTimestampCollected is NOT updated
		assert.True(t, mr.lastUsageTimestampCollected.IsZero())
	})

	t.Run("colocation node, server returns non-JSON body", func(t *testing.T) {
		node := colocationNode("test-node")
		cs, nodeCleanup := newFakeClientset(t, node)
		defer nodeCleanup()

		uploadSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("not-json"))
		}))
		defer uploadSrv.Close()

		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			apiserverClient: cs,
			reportBuffer:    NewReportBuffer(2),
			buffers:         newBufPool(),
		}
		mr.config.UsageReportAddr = uploadSrv.Listener.Addr().String()
		mr.reportBuffer.Put(metric.NodeUsageReport{Node: "test-node", Timestamp: metav1.Time{Time: now}})

		assert.NotPanics(t, func() { mr.uploadTick(now) })
	})
}

// ---- newKubeletClient (pure constructor, both TLS branches) ----

func TestNewKubeletClient(t *testing.T) {
	t.Run("secure TLS branch (default)", func(t *testing.T) {
		cfg := &agentapps.Configuration{
			KubeletSecureMetricsPort: 10250,
		}
		restCfg := &rest.Config{Host: "https://127.0.0.1"}
		kc, err := newKubeletClient(cfg, restCfg)
		require.NoError(t, err)
		assert.NotNil(t, kc)
	})

	t.Run("insecure TLS branch", func(t *testing.T) {
		cfg := &agentapps.Configuration{
			InsecureKubeletMetricsTLS:  true,
			KubeletInsecureMetricsPort: 10255,
		}
		restCfg := &rest.Config{Host: "http://127.0.0.1"}
		kc, err := newKubeletClient(cfg, restCfg)
		require.NoError(t, err)
		assert.NotNil(t, kc)
	})
}

// ---- scrapeTick / exportTick / getKubeletPods GET / processUsageReports ----

// fakePodStorage is a minimal metric.PodStorage stub for scrapeTick.
type fakePodStorage struct {
	storeCall  chan *metric.MetricsBatch
	lastReport *metric.NodeUsageReport
	lastErr    error
}

func (f *fakePodStorage) Store(podsMetrics *metric.MetricsBatch) {
	if f.storeCall != nil {
		f.storeCall <- podsMetrics
	}
}

func (f *fakePodStorage) GetLastMetrics(pods ...*metav1.PartialObjectMetadata) (*metric.NodeUsageReport, error) {
	if f.lastErr != nil {
		return nil, f.lastErr
	}
	return f.lastReport, nil
}

// newTestKubeletClient builds a real *metric.KubeletClient that talks to the
// given httptest server (HTTP, defaultPort = server's port).
func newTestKubeletClient(t *testing.T, srv *httptest.Server) *metric.KubeletClient {
	t.Helper()
	host, port, err := net.SplitHostPort(srv.Listener.Addr().String())
	require.NoError(t, err)
	portInt, err := strconv.Atoi(port)
	require.NoError(t, err)

	kc, err := metric.NewForConfig(&metric.KubeletClientConfig{
		Client:              nil, // uses http.DefaultTransport
		AddressTypePriority: []corev1.NodeAddressType{corev1.NodeHostName},
		Scheme:              "http",
		DefaultPort:         portInt,
		UseNodeStatusPort:   false,
	})
	require.NoError(t, err)
	_ = host // host is taken from the node's NodeHostName address in tests
	return kc
}

// scrapeNode builds the node passed to kubeletClient.Get* in scrape/export ticks.
func scrapeNode(ip string) *corev1.Node {
	return &corev1.Node{
		Status: corev1.NodeStatus{Addresses: []corev1.NodeAddress{
			{Type: corev1.NodeHostName, Address: ip},
		}},
	}
}

func TestScrapeTick(t *testing.T) {
	t.Run("GetMetrics succeeds, report buffered", func(t *testing.T) {
		// Kubelet server returns an empty (but valid) Prometheus body for
		// /metrics/resource and an empty pod list for /pods.
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics/resource", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("# EOF\n"))
		})
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"items":[]}`))
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		// Point the resolver at 127.0.0.1 via the node address; the port comes
		// from the client's defaultPort.
		_ = kc

		buf := NewReportBuffer(2)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
			storage: &fakePodStorage{
				lastReport: &metric.NodeUsageReport{
					Node:      "test-node",
					Timestamp: metav1.Time{Time: time.Now()},
					Containers: map[metric.PodRef]map[string]corev1.ResourceList{
						metric.NewPodRef("ns", "p1"): {
							"c1": corev1.ResourceList{corev1.ResourceCPU: *resource.NewMilliQuantity(1, resource.DecimalSI)},
						},
					},
				},
			},
			reportBuffer: buf,
		}

		// scrapeTick uses mr.config.KubeletIP as the NodeHostName.
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.scrapeTick(time.Now()) })
		// report has containers -> should be buffered
		assert.GreaterOrEqual(t, buf.CurSize(), 1)
	})

	t.Run("GetMetrics error returns early, nothing buffered", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics/resource", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "fail", http.StatusInternalServerError)
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		buf := NewReportBuffer(2)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
			storage:       &fakePodStorage{},
			reportBuffer:  buf,
		}
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.scrapeTick(time.Now()) })
		assert.Equal(t, 0, buf.CurSize())
	})

	t.Run("GetLastMetrics error returns early", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics/resource", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("# EOF\n"))
		})
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"items":[]}`))
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		buf := NewReportBuffer(2)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
			storage:       &fakePodStorage{lastErr: errors.New("storage down")},
			reportBuffer:  buf,
		}
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.scrapeTick(time.Now()) })
		assert.Equal(t, 0, buf.CurSize())
	})

	t.Run("empty report not buffered", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/metrics/resource", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte("# EOF\n"))
		})
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"items":[]}`))
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		buf := NewReportBuffer(2)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
			// empty report: no containers and no pods
			storage: &fakePodStorage{
				lastReport: &metric.NodeUsageReport{Node: "test-node"},
			},
			reportBuffer: buf,
		}
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.scrapeTick(time.Now()) })
		assert.Equal(t, 0, buf.CurSize())
	})
}

func TestExportTick(t *testing.T) {
	node := colocationNode("test-node")
	cs, nodeCleanup := newFakeClientset(t, node)
	defer nodeCleanup()

	t.Run("GetPodMetrics succeeds, node collected", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"items":[]}`))
		})
		mux.HandleFunc("/stats/summary", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"node":{},"pods":[]}`))
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			kubeletClient:   kc,
			apiserverClient: cs,
		}
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.exportTick() })
	})

	t.Run("GetPodMetrics error returns early", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "fail", http.StatusInternalServerError)
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		mr := &MetricsReporter{
			ctx:             context.Background(),
			config:          newAgentCfg(),
			kubeletClient:   kc,
			apiserverClient: cs,
		}
		mr.config.KubeletIP = "127.0.0.1"

		assert.NotPanics(t, func() { mr.exportTick() })
	})
}

func TestGetKubeletPodsGET(t *testing.T) {
	t.Run("GET with working kubelet returns pods JSON", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			_, _ = w.Write([]byte(`{"items":[]}`))
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
		}
		mr.config.KubeletIP = "127.0.0.1"

		req := httptest.NewRequest(http.MethodGet, "/debug/pods", nil)
		w := httptest.NewRecorder()
		mr.getKubeletPods(w, req)
		assert.Equal(t, http.StatusOK, w.Code)
	})

	t.Run("GET with failing kubelet returns 500", func(t *testing.T) {
		mux := http.NewServeMux()
		mux.HandleFunc("/pods", func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "fail", http.StatusInternalServerError)
		})
		srv := httptest.NewServer(mux)
		defer srv.Close()

		kc := newTestKubeletClient(t, srv)
		mr := &MetricsReporter{
			ctx:           context.Background(),
			config:        newAgentCfg(),
			kubeletClient: kc,
		}
		mr.config.KubeletIP = "127.0.0.1"

		req := httptest.NewRequest(http.MethodGet, "/debug/pods", nil)
		w := httptest.NewRecorder()
		mr.getKubeletPods(w, req)
		assert.Equal(t, http.StatusInternalServerError, w.Code)
	})
}

// ---- processUsageReports (the collector's channel-consumer loop) ----
//
// processUsageReports exits cleanly on ctx.Done, so we can run it briefly.
func TestProcessUsageReportsLoop(t *testing.T) {
	cs := newTestClusterState(t)
	c := newTestUsageCollector(t, 4, cs)
	registerNode(t, c, cs, "node-loop")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		c.processUsageReports(ctx)
		close(done)
	}()

	// Push a report through the channel; processSingleUsageReport will run.
	// (container sample will fail to add since pod is not in cluster state,
	//  but that path is covered; the loop itself just must not deadlock.)
	r := makeReport("node-loop", time.Now())
	c.usageReportChan <- &r

	// cancel -> loop should exit
	cancel()
	select {
	case <-done:
		// success
	case <-time.After(2 * time.Second):
		t.Fatal("processUsageReports did not exit after context cancellation")
	}
}

// ---- helpers for sync.Pool ----

// newBufPool returns a sync.Pool mimicking the one in NewMetricsReporter so
// that uploadTick's mr.buffers.Get() yields a usable *[]byte.
func newBufPool() sync.Pool {
	return sync.Pool{
		New: func() interface{} {
			buf := make([]byte, 10e3)
			return &buf
		},
	}
}
