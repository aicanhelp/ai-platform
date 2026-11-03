/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
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
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	mgrapps "openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/colocation-overquota-agent/metric"
)

// ---- helpers ----

func newTestClusterState(t *testing.T) *aggregate.ClusterState {
	t.Helper()
	return aggregate.NewClusterState(context.Background(), &mgrapps.Configuration{
		AggregateStateGCInterval: 1 * time.Hour,
	})
}

func newTestUsageCollector(t *testing.T, bufCap int, cs *aggregate.ClusterState) *UsageCollector {
	t.Helper()
	if cs == nil {
		cs = newTestClusterState(t)
	}
	cfg := &mgrapps.Configuration{
		UsageBufCapacity: bufCap,
		UsageReportAddr:  ":0",
	}
	return NewUsageCollector(cfg, cs)
}

func mustMarshalJSON(t *testing.T, v interface{}) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	require.NoError(t, err)
	return b
}

// registerNode registers the node both in the collector (via callback) and in
// the cluster state so processSingleUsageReport can find it.
func registerNode(t *testing.T, c *UsageCollector, cs *aggregate.ClusterState, nodeName string) {
	t.Helper()
	cs.AddOrUpdateNode(&corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName}})
	c.addNodeCallback(nodeName)
}

// makeReport builds a NodeUsageReport for the given node at the given time.
func makeReport(node string, ts time.Time) metric.NodeUsageReport {
	return metric.NodeUsageReport{
		Node:      node,
		Timestamp: metav1.Time{Time: ts},
		Window:    metav1.Duration{Duration: 10 * time.Second},
		Containers: map[metric.PodRef]map[string]corev1.ResourceList{
			metric.NewPodRef("default", "pod-a"): {
				"c1": corev1.ResourceList{
					corev1.ResourceCPU: *resource.NewMilliQuantity(100, resource.DecimalSI),
				},
			},
		},
	}
}

// ---- parseRequest ----

func TestParseRequest(t *testing.T) {
	c := &UsageCollector{}

	t.Run("valid JSON", func(t *testing.T) {
		reports := []metric.NodeUsageReport{makeReport("n1", time.Now())}
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage", bytes.NewReader(mustMarshalJSON(t, reports)))
		got, err := c.parseRequest(req)
		require.NoError(t, err)
		require.Len(t, got, 1)
		assert.Equal(t, "n1", got[0].Node)
	})

	t.Run("malformed JSON", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage", strings.NewReader("{not-json"))
		_, err := c.parseRequest(req)
		assert.Error(t, err)
	})

	t.Run("read error", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage", errReader{})
		_, err := c.parseRequest(req)
		assert.Error(t, err)
	})
}

// errReader is an io.Reader that always fails on Read.
type errReader struct{}

func (errReader) Read(p []byte) (int, error) { return 0, errors.New("synthetic read failure") }

// ---- validateReport ----

func TestValidateReport(t *testing.T) {
	c := &UsageCollector{}

	tests := []struct {
		name          string
		report        metric.NodeUsageReport
		expectedNode  string
		wantCode      int
		wantNonEmpty  bool // expect a non-empty error message
	}{
		{
			name:         "empty expected node - first report accepted",
			report:       makeReport("n1", time.Now()),
			expectedNode: "",
			wantCode:     codeSuccess,
		},
		{
			name:         "same node as expected",
			report:       makeReport("n1", time.Now()),
			expectedNode: "n1",
			wantCode:     codeSuccess,
		},
		{
			name:         "mismatched node",
			report:       makeReport("n2", time.Now()),
			expectedNode: "n1",
			wantCode:     codeInvalidFormat,
			wantNonEmpty: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			code, msg := c.validateReport(tt.report, tt.expectedNode)
			assert.Equal(t, tt.wantCode, code)
			if tt.wantNonEmpty {
				assert.NotEmpty(t, msg)
			} else {
				assert.Empty(t, msg)
			}
		})
	}
}

// ---- processReports ----

func TestProcessReports(t *testing.T) {
	cs := newTestClusterState(t)
	c := newTestUsageCollector(t, 10, cs)
	// drain the channel so sendToChannel always succeeds without blocking
	go func() {
		for range c.usageReportChan {
		}
	}()

	now := time.Now()

	t.Run("empty slice returns success with nil timestamp", func(t *testing.T) {
		code, msg, ts := c.processReports(nil)
		assert.Equal(t, codeSuccess, code)
		assert.Empty(t, msg)
		assert.Nil(t, ts)
	})

	t.Run("valid report success", func(t *testing.T) {
		registerNode(t, c, cs, "node-ok")
		r := makeReport("node-ok", now)
		code, msg, ts := c.processReports([]metric.NodeUsageReport{r})
		assert.Equal(t, codeSuccess, code)
		assert.Empty(t, msg)
		// lastUsageTimestampCollected is zero-time initially, returned non-nil
		require.NotNil(t, ts)
	})

	t.Run("node not in range", func(t *testing.T) {
		r := makeReport("node-missing", now)
		code, msg, ts := c.processReports([]metric.NodeUsageReport{r})
		assert.Equal(t, codeNodeNotInRange, code)
		assert.NotEmpty(t, msg)
		assert.Nil(t, ts)
	})

	t.Run("timestamp expired", func(t *testing.T) {
		registerNode(t, c, cs, "node-expired")
		// First send a recent report to update the last collected timestamp.
		recent := makeReport("node-expired", now)
		_, _, _ = c.processReports([]metric.NodeUsageReport{recent})
		// Now send an older report; it should be rejected.
		old := makeReport("node-expired", now.Add(-1*time.Hour))
		code, msg, ts := c.processReports([]metric.NodeUsageReport{old})
		assert.Equal(t, codeUsageTimestampExpired, code)
		assert.NotEmpty(t, msg)
		assert.Nil(t, ts)
	})

	t.Run("buffer full", func(t *testing.T) {
		// dedicated collector with a tiny channel that nobody drains
		cs2 := newTestClusterState(t)
		c2 := newTestUsageCollector(t, 1, cs2)
		registerNode(t, c2, cs2, "node-full")
		// fill the channel (capacity 1) before sending.
		c2.usageReportChan <- &metric.NodeUsageReport{Node: "node-full"}
		r := makeReport("node-full", now.Add(99*time.Hour))
		code, msg, ts := c2.processReports([]metric.NodeUsageReport{r})
		assert.Equal(t, codeReportBufferFull, code)
		assert.NotEmpty(t, msg)
		assert.Nil(t, ts)
	})

	t.Run("mismatched nodes in one batch", func(t *testing.T) {
		registerNode(t, c, cs, "n-a")
		registerNode(t, c, cs, "n-b")
		reports := []metric.NodeUsageReport{
			makeReport("n-a", now.Add(10*time.Hour)),
			makeReport("n-b", now.Add(10*time.Hour)),
		}
		code, msg, ts := c.processReports(reports)
		assert.Equal(t, codeInvalidFormat, code)
		assert.NotEmpty(t, msg)
		assert.Nil(t, ts)
	})
}

// ---- sendToChannel ----

func TestSendToChannel(t *testing.T) {
	t.Run("open channel returns true", func(t *testing.T) {
		c := &UsageCollector{
			usageReportChan: make(chan *metric.NodeUsageReport, 1),
			nodes:           make(map[string]metav1.Time),
		}
		r := makeReport("n1", time.Now())
		ok := c.sendToChannel(&r, "n1")
		assert.True(t, ok)
		// should have updated the timestamp map
		ts := c.findNodeLastUsageTimestampCollected("n1")
		require.NotNil(t, ts)
	})

	t.Run("full channel returns false", func(t *testing.T) {
		c := &UsageCollector{
			usageReportChan: make(chan *metric.NodeUsageReport, 1),
			nodes:           make(map[string]metav1.Time),
		}
		// pre-fill the channel
		c.usageReportChan <- &metric.NodeUsageReport{Node: "n1"}
		r := makeReport("n1", time.Now())
		ok := c.sendToChannel(&r, "n1")
		assert.False(t, ok)
	})
}

// ---- sendResponse ----

func TestSendResponse(t *testing.T) {
	c := &UsageCollector{}

	t.Run("without timestamp", func(t *testing.T) {
		w := httptest.NewRecorder()
		c.sendResponse(w, codeSuccess, "ok", nil)
		assert.Equal(t, http.StatusOK, w.Code)
		assert.Equal(t, "application/json", w.Header().Get("Content-Type"))

		var resp metric.NodeUsageResponse[metav1.Time]
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, codeSuccess, resp.Code)
		assert.Equal(t, "ok", resp.Msg)
		assert.True(t, resp.Data.IsZero())
	})

	t.Run("with timestamp", func(t *testing.T) {
		w := httptest.NewRecorder()
		ts := metav1.Time{Time: time.Unix(1700_000_000, 0)}
		c.sendResponse(w, codeNodeNotInRange, "err", &ts)

		var resp metric.NodeUsageResponse[metav1.Time]
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, codeNodeNotInRange, resp.Code)
		assert.False(t, resp.Data.IsZero())
	})
}

// ---- receiveUsageReport via httptest ----

func TestReceiveUsageReport(t *testing.T) {
	cs := newTestClusterState(t)
	c := newTestUsageCollector(t, 5, cs)
	registerNode(t, c, cs, "node-srv")
	// drain channel so the valid path does not block
	var stop = make(chan struct{})
	go func() {
		for {
			select {
			case <-c.usageReportChan:
			case <-stop:
				return
			}
		}
	}()
	defer close(stop)

	t.Run("GET method not allowed", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodGet, "/v1/metrics/usage", nil)
		w := httptest.NewRecorder()
		c.receiveUsageReport(w, req)
		assert.Equal(t, http.StatusMethodNotAllowed, w.Code)
	})

	t.Run("POST with bad body", func(t *testing.T) {
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage", strings.NewReader("garbage"))
		w := httptest.NewRecorder()
		c.receiveUsageReport(w, req)
		assert.Equal(t, http.StatusBadRequest, w.Code)
	})

	t.Run("POST happy path", func(t *testing.T) {
		reports := []metric.NodeUsageReport{makeReport("node-srv", time.Now().Add(1*time.Hour))}
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage",
			bytes.NewReader(mustMarshalJSON(t, reports)))
		w := httptest.NewRecorder()
		c.receiveUsageReport(w, req)
		assert.Equal(t, http.StatusOK, w.Code)

		var resp metric.NodeUsageResponse[metav1.Time]
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, codeSuccess, resp.Code)
	})

	t.Run("POST node not in range", func(t *testing.T) {
		reports := []metric.NodeUsageReport{makeReport("unknown-node", time.Now().Add(1*time.Hour))}
		req := httptest.NewRequest(http.MethodPost, "/v1/metrics/usage",
			bytes.NewReader(mustMarshalJSON(t, reports)))
		w := httptest.NewRecorder()
		c.receiveUsageReport(w, req)
		assert.Equal(t, http.StatusOK, w.Code)

		var resp metric.NodeUsageResponse[metav1.Time]
		require.NoError(t, json.Unmarshal(w.Body.Bytes(), &resp))
		assert.Equal(t, codeNodeNotInRange, resp.Code)
	})
}

// ---- node callbacks + timestamp map ----

func TestNodeCallbacksAndTimestampMap(t *testing.T) {
	c := &UsageCollector{nodes: make(map[string]metav1.Time)}

	// addNodeCallback
	c.addNodeCallback("n1")
	c.addNodeCallback("n1") // idempotent
	ts := c.findNodeLastUsageTimestampCollected("n1")
	require.NotNil(t, ts)
	assert.True(t, ts.IsZero(), "initial timestamp should be zero")

	// update
	now := metav1.Time{Time: time.Unix(123, 0)}
	c.updateNodeLastUsageTimestampCollected("n1", now)
	ts = c.findNodeLastUsageTimestampCollected("n1")
	require.NotNil(t, ts)
	assert.Equal(t, now.Time.Unix(), ts.Time.Unix())

	// unknown node
	assert.Nil(t, c.findNodeLastUsageTimestampCollected("nope"))

	// delete
	c.deleteNodeCallback("n1")
	assert.Nil(t, c.findNodeLastUsageTimestampCollected("n1"))
}

func TestNodeCallbacksConcurrency(t *testing.T) {
	c := &UsageCollector{nodes: make(map[string]metav1.Time)}

	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(2)
		node := "node"
		go func() { defer wg.Done(); c.addNodeCallback(node) }()
		go func() { defer wg.Done(); c.findNodeLastUsageTimestampCollected(node) }()
	}
	wg.Wait()
}

// ---- getPodIDFromRef ----

func TestGetPodIDFromRef(t *testing.T) {
	t.Run("valid ref", func(t *testing.T) {
		id, err := getPodIDFromRef(metric.NewPodRef("ns", "pod"))
		require.NoError(t, err)
		assert.Equal(t, "ns", id.Namespace)
		assert.Equal(t, "pod", id.PodName)
	})

	t.Run("invalid ref - missing slash", func(t *testing.T) {
		_, err := getPodIDFromRef(metric.PodRef("noporeshere"))
		assert.Error(t, err)
	})

	t.Run("invalid ref - empty name", func(t *testing.T) {
		_, err := getPodIDFromRef(metric.NewPodRef("ns", ""))
		assert.Error(t, err)
	})
}

// ---- createContainerSamples + createSingleSample ----

func TestCreateContainerSamples(t *testing.T) {
	podID := aggregate.PodID{Namespace: "ns", PodName: "pod1"}
	ts := time.Now()

	t.Run("mixed resources - cpu, memory, unknown", func(t *testing.T) {
		resources := corev1.ResourceList{
			corev1.ResourceCPU:     *resource.NewMilliQuantity(500, resource.DecimalSI),
			corev1.ResourceMemory:  *resource.NewQuantity(2048, resource.BinarySI),
			corev1.ResourceStorage: *resource.NewQuantity(10, resource.DecimalSI), // unknown -> dropped
		}
		samples := createContainerSamples(podID, "c1", resources, ts)
		assert.Len(t, samples, 2)

		seen := map[aggregate.ResourceName]bool{}
		for _, s := range samples {
			assert.Equal(t, "c1", s.Container.ContainerName)
			assert.Equal(t, "ns", s.Container.PodID.Namespace)
			assert.Equal(t, "pod1", s.Container.PodID.PodName)
			assert.Equal(t, ts, s.ContainerUsageSample.MeasureStart)
			seen[s.Resource] = true
		}
		assert.True(t, seen[aggregate.ResourceCPU])
		assert.True(t, seen[aggregate.ResourceMemory])
	})

	t.Run("empty resource list", func(t *testing.T) {
		samples := createContainerSamples(podID, "c1", corev1.ResourceList{}, ts)
		assert.Empty(t, samples)
	})
}

func TestCreateSingleSample(t *testing.T) {
	podID := aggregate.PodID{Namespace: "ns", PodName: "pod1"}
	ts := time.Now()

	t.Run("cpu sample", func(t *testing.T) {
		q := resource.NewMilliQuantity(750, resource.DecimalSI)
		s := createSingleSample(podID, "c1", corev1.ResourceCPU, *q, ts)
		require.NotNil(t, s)
		assert.Equal(t, aggregate.ResourceCPU, s.Resource)
		assert.Equal(t, aggregate.ResourceAmount(750), s.Usage)
	})

	t.Run("memory sample", func(t *testing.T) {
		q := resource.NewQuantity(4096, resource.BinarySI)
		s := createSingleSample(podID, "c1", corev1.ResourceMemory, *q, ts)
		require.NotNil(t, s)
		assert.Equal(t, aggregate.ResourceMemory, s.Resource)
		assert.Equal(t, aggregate.ResourceAmount(4096), s.Usage)
	})

	t.Run("unknown resource returns nil", func(t *testing.T) {
		q := resource.NewQuantity(1, resource.DecimalSI)
		s := createSingleSample(podID, "c1", corev1.ResourceStorage, *q, ts)
		assert.Nil(t, s)
	})
}

// ---- convertResourceAmount ----

func TestConvertResourceAmount(t *testing.T) {
	tests := []struct {
		name     string
		res      aggregate.ResourceName
		quantity resource.Quantity
		want     aggregate.ResourceAmount
		wantOK   bool
	}{
		{
			name:     "cpu milli-scale",
			res:      aggregate.ResourceCPU,
			quantity: *resource.NewMilliQuantity(1500, resource.DecimalSI),
			want:     1500,
			wantOK:   true,
		},
		{
			name:     "cpu whole core scales to milli",
			res:      aggregate.ResourceCPU,
			quantity: *resource.NewQuantity(2, resource.DecimalSI),
			want:     2000,
			wantOK:   true,
		},
		{
			name:     "memory value",
			res:      aggregate.ResourceMemory,
			quantity: *resource.NewQuantity(1<<30, resource.BinarySI),
			want:     aggregate.ResourceAmount(1 << 30),
			wantOK:   true,
		},
		{
			name:     "unknown resource",
			res:      aggregate.ResourceName("storage"),
			quantity: *resource.NewQuantity(100, resource.DecimalSI),
			want:     0,
			wantOK:   false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := convertResourceAmount(tt.res, tt.quantity)
			assert.Equal(t, tt.wantOK, ok)
			if ok {
				assert.Equal(t, tt.want, got)
			}
		})
	}
}

// ---- processSingleUsageReport ----

func TestProcessSingleUsageReport(t *testing.T) {
	cs := newTestClusterState(t)
	c := newTestUsageCollector(t, 5, cs)
	registerNode(t, c, cs, "node-x")

	t.Run("node not present in cluster state increments neither", func(t *testing.T) {
		r := makeReport("ghost-node", time.Now())
		suc, fail := c.processSingleUsageReport(&r, 3, 7)
		assert.Equal(t, 3, suc)
		assert.Equal(t, 7, fail)
	})

	t.Run("container sample added but pod missing -> fail count", func(t *testing.T) {
		// Node exists in cluster state, but the pod is not registered there,
		// so AddSample returns an error and failCount increments.
		r := makeReport("node-x", time.Now())
		suc, fail := c.processSingleUsageReport(&r, 0, 0)
		assert.Equal(t, 0, suc)
		assert.Equal(t, 1, fail) // one container sample, pod missing -> 1 failure
	})

	t.Run("registered pod+container -> success count increments", func(t *testing.T) {
		// Register the pod and container so AddSample succeeds.
		require.NoError(t, cs.AddPodAndContainer(&corev1.Pod{
			ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "pod-a"},
			Spec: corev1.PodSpec{
				NodeName: "node-x",
				Containers: []corev1.Container{
					{Name: "c1"},
				},
			},
		}))
		r := makeReport("node-x", time.Now())
		suc, fail := c.processSingleUsageReport(&r, 0, 0)
		assert.Equal(t, 1, suc) // one container sample added successfully
		assert.Equal(t, 0, fail)
	})
}

// ---- newContainerUsageSamplesWithKey (additional edge) ----

func TestNewContainerUsageSamplesWithKeyInvalidRef(t *testing.T) {
	usage := &metric.NodeUsageReport{
		Node:      "n1",
		Timestamp: metav1.Time{Time: time.Now()},
		Containers: map[metric.PodRef]map[string]corev1.ResourceList{
			metric.PodRef("bad-format"): {
				"c": corev1.ResourceList{corev1.ResourceCPU: *resource.NewMilliQuantity(1, resource.DecimalSI)},
			},
		},
	}
	// invalid pod ref -> skipped -> no samples
	assert.Empty(t, newContainerUsageSamplesWithKey(usage))
}

// ---- io.Reader read-error path for parseRequest uses errReader above ----

// ensure io is used (errReader implements io.Reader)
var _ io.Reader = errReader{}
