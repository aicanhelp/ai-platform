/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

func TestNewBKEClusterCollector(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	if collector == nil {
		t.Fatal("NewBKEClusterCollector returned nil")
	}
}

func TestBkeClusterCollector_Describe(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	ch := make(chan *prometheus.Desc, 10)
	go func() {
		collector.Describe(ch)
		close(ch)
	}()
	count := 0
	for range ch {
		count++
	}
	if count == 0 {
		t.Error("no descriptors collected")
	}
}

func TestBkeClusterCollector_Collect(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	ch := make(chan prometheus.Metric, 10)
	go func() {
		collector.Collect(ch)
		close(ch)
	}()
	for range ch {
	}
}

func TestBkeClusterCollector_RecordGaugeVecWithLabelValues(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	gauge := collector.RecordGaugeVecWithLabelValues(PhaseDurationSeconds, "phase1", "2024-01-01 00:00:00", "2024-01-01 01:00:00", "test")
	if gauge == nil {
		t.Error("gauge is nil")
	} else {
		gauge.Set(100)
	}

	gauge = collector.RecordGaugeVecWithLabelValues("non-existent", "test")
	if gauge != nil {
		t.Error("expected nil for non-existent vec")
	}
}

func TestBkeClusterCollector_RecordCounterVecWithLabelValues(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	counter := collector.RecordCounterVecWithLabelValues(ClusterUnhealthyCount)
	if counter == nil {
		t.Error("counter is nil")
	} else {
		counter.Inc()
	}

	counter = collector.RecordCounterVecWithLabelValues("non-existent")
	if counter != nil {
		t.Error("expected nil for non-existent vec")
	}
}

func TestBkeClusterCollector_DeleteMetricWithLabels(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	labels := prometheus.Labels{"phase": "test"}
	collector.DeleteMetricWithLabels(labels)
}

func TestBkeClusterCollector_RegisterVec(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")
	bkeCollector := collector.(*bkeClusterCollector)
	initialCount := len(bkeCollector.vecs)

	testVecFunc := func(clusterKey string) (string, prometheus.Collector) {
		return "test_vec", prometheus.NewGaugeVec(
			prometheus.GaugeOpts{Name: "test_vec"},
			[]string{"label"},
		)
	}

	collector.RegisterVec(testVecFunc)
	if len(bkeCollector.vecs) != initialCount+1 {
		t.Error("vec not registered")
	}
}

func TestBkeClusterCollector_Export(t *testing.T) {
	collector := NewBKEClusterCollector("test-cluster")

	gauge := collector.RecordGaugeVecWithLabelValues(PhaseDurationSeconds, "phase1", "2024-01-01 00:00:00", "2024-01-01 01:00:00", "test")
	if gauge != nil {
		gauge.Set(100)
	}

	result, err := collector.Export(PhaseDurationSeconds)
	if err != nil {
		t.Fatalf("Export failed: %v", err)
	}
	if result == nil {
		t.Error("result is nil")
	}
}

func TestNewGaugeOpts(t *testing.T) {
	opts := newGaugeOpts("test_metric", "test-cluster")
	if opts.Name != "test_metric" {
		t.Errorf("expected test_metric, got %s", opts.Name)
	}
	if opts.Namespace != namespace {
		t.Errorf("expected %s, got %s", namespace, opts.Namespace)
	}
	if opts.Subsystem != clusterSubSystem {
		t.Errorf("expected %s, got %s", clusterSubSystem, opts.Subsystem)
	}
	if opts.ConstLabels["bke_cluster"] != "test-cluster" {
		t.Error("bke_cluster label not set correctly")
	}
}

func TestNewCounterOpts(t *testing.T) {
	opts := newCounterOpts("test_metric", "test-cluster")
	if opts.Name != "test_metric" {
		t.Errorf("expected test_metric, got %s", opts.Name)
	}
	if opts.Namespace != namespace {
		t.Errorf("expected %s, got %s", namespace, opts.Namespace)
	}
	if opts.Subsystem != clusterSubSystem {
		t.Errorf("expected %s, got %s", clusterSubSystem, opts.Subsystem)
	}
	if opts.ConstLabels["bke_cluster"] != "test-cluster" {
		t.Error("bke_cluster label not set correctly")
	}
}
