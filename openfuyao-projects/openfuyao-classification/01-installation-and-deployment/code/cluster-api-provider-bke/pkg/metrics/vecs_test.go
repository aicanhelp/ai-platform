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

func TestNewMetricsVectors(t *testing.T) {
	mv := NewMetricsVectors()
	if mv == nil {
		t.Fatal("NewMetricsVectors returned nil")
	}
}

func TestMetricsVectors_PhaseDurationVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.PhaseDurationVec("test-cluster")
	if name != PhaseDurationSeconds {
		t.Errorf("expected %s, got %s", PhaseDurationSeconds, name)
	}
	if collector == nil {
		t.Fatal("collector is nil")
	}
	if _, ok := collector.(*prometheus.GaugeVec); !ok {
		t.Error("collector is not GaugeVec")
	}
}

func TestMetricsVectors_ClusterUnhealthyCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.ClusterUnhealthyCountVec("test-cluster")
	if name != ClusterUnhealthyCount {
		t.Errorf("expected %s, got %s", ClusterUnhealthyCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestMetricsVectors_ClusterReadyStatusVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.ClusterReadyStatusVec("test-cluster")
	if name != ClusterReadyStatus {
		t.Errorf("expected %s, got %s", ClusterReadyStatus, name)
	}
	if _, ok := collector.(*prometheus.GaugeVec); !ok {
		t.Error("collector is not GaugeVec")
	}
}

func TestMetricsVectors_ClusterBootstrapDurationVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.ClusterBootstrapDurationVec("test-cluster")
	if name != ClusterBootstrapDurationSeconds {
		t.Errorf("expected %s, got %s", ClusterBootstrapDurationSeconds, name)
	}
	if _, ok := collector.(*prometheus.GaugeVec); !ok {
		t.Error("collector is not GaugeVec")
	}
}

func TestMetricsVectors_NodeBootstrapDurationVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.NodeBootstrapDurationVec("test-cluster")
	if name != NodeBootstrapDurationSeconds {
		t.Errorf("expected %s, got %s", NodeBootstrapDurationSeconds, name)
	}
	if _, ok := collector.(*prometheus.GaugeVec); !ok {
		t.Error("collector is not GaugeVec")
	}
}

func TestMetricsVectors_NodeBootstrapFailedCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.NodeBootstrapFailedCountVec("test-cluster")
	if name != NodeBootstrapFailedCount {
		t.Errorf("expected %s, got %s", NodeBootstrapFailedCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestMetricsVectors_NodeBootstrapSuccessCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.NodeBootstrapSuccessCountVec("test-cluster")
	if name != NodeBootstrapSuccessCount {
		t.Errorf("expected %s, got %s", NodeBootstrapSuccessCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestMetricsVectors_AddonInstallDurationVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.AddonInstallDurationVec("test-cluster")
	if name != AddonInstallDurationSeconds {
		t.Errorf("expected %s, got %s", AddonInstallDurationSeconds, name)
	}
	if _, ok := collector.(*prometheus.GaugeVec); !ok {
		t.Error("collector is not GaugeVec")
	}
}

func TestMetricsVectors_AddonInstallFailedCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.AddonInstallFailedCountVec("test-cluster")
	if name != AddonInstallFailedCount {
		t.Errorf("expected %s, got %s", AddonInstallFailedCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestMetricsVectors_AddonInstallSuccessCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.AddonInstallSuccessCountVec("test-cluster")
	if name != AddonInstallSuccessCount {
		t.Errorf("expected %s, got %s", AddonInstallSuccessCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestMetricsVectors_AddonInstallCountVec(t *testing.T) {
	mv := NewMetricsVectors()
	name, collector := mv.AddonInstallCountVec("test-cluster")
	if name != AddonInstallCount {
		t.Errorf("expected %s, got %s", AddonInstallCount, name)
	}
	if _, ok := collector.(*prometheus.CounterVec); !ok {
		t.Error("collector is not CounterVec")
	}
}

func TestGlobalFunctions(t *testing.T) {
	tests := []struct {
		name     string
		fn       func(string) (string, prometheus.Collector)
		expected string
	}{
		{"PhaseDurationVec", PhaseDurationVec, PhaseDurationSeconds},
		{"ClusterUnhealthyCountVec", ClusterUnhealthyCountVec, ClusterUnhealthyCount},
		{"ClusterReadyStatusVec", ClusterReadyStatusVec, ClusterReadyStatus},
		{"ClusterBootstrapDurationVec", ClusterBootstrapDurationVec, ClusterBootstrapDurationSeconds},
		{"NodeBootstrapDurationVec", NodeBootstrapDurationVec, NodeBootstrapDurationSeconds},
		{"NodeBootstrapFailedCountVec", NodeBootstrapFailedCountVec, NodeBootstrapFailedCount},
		{"NodeBootstrapSuccessCountVec", NodeBootstrapSuccessCountVec, NodeBootstrapSuccessCount},
		{"AddonInstallDurationVec", AddonInstallDurationVec, AddonInstallDurationSeconds},
		{"AddonInstallFailedCountVec", AddonInstallFailedCountVec, AddonInstallFailedCount},
		{"AddonInstallSuccessCountVec", AddonInstallSuccessCountVec, AddonInstallSuccessCount},
		{"AddonInstallCountVec", AddonInstallCountVec, AddonInstallCount},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			name, collector := tt.fn("test-cluster")
			if name != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, name)
			}
			if collector == nil {
				t.Error("collector is nil")
			}
		})
	}
}
