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
)

func TestNewBKEClusterMetricRegister(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	if register.collectors == nil {
		t.Error("collectors map is nil")
	}
}

func TestBKEClusterMetricRegister_Register(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-1")

	register.mux.RLock()
	if register.collectors["test-cluster-1"] == nil {
		t.Error("collector not registered")
	}
	register.mux.RUnlock()

	register.Register("test-cluster-1")
	register.mux.RLock()
	count := len(register.collectors)
	register.mux.RUnlock()
	if count != 1 {
		t.Error("duplicate registration should be ignored")
	}
}

func TestBKEClusterMetricRegister_Unregister(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-2")
	register.Unregister("test-cluster-2")

	register.mux.RLock()
	if register.collectors["test-cluster-2"] != nil {
		t.Error("collector not unregistered")
	}
	register.mux.RUnlock()

	register.Unregister("non-existent")
}

func TestBKEClusterMetricRegister_Gather(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-3")

	result, err := register.Gather("test-cluster-3", PhaseDurationSeconds)
	if err != nil {
		t.Errorf("Gather failed: %v", err)
	}
	if result == nil {
		t.Error("result is nil")
	}

	result, err = register.Gather("non-existent")
	if err != nil {
		t.Errorf("Gather failed: %v", err)
	}
	if result != nil {
		t.Error("expected nil for non-existent cluster")
	}
}

func TestBKEClusterMetricRegister_RecordGaugeVec(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-4")

	gauge := register.RecordGaugeVec(PhaseDurationSeconds, "test-cluster-4", "phase1", "2024-01-01 00:00:00", "2024-01-01 01:00:00", "test")
	if gauge == nil {
		t.Error("gauge is nil")
	}

	gauge = register.RecordGaugeVec(PhaseDurationSeconds, "non-existent")
	if gauge != nil {
		t.Error("expected nil for non-existent cluster")
	}
}

func TestBKEClusterMetricRegister_RecordCounterVec(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-5")

	counter := register.RecordCounterVec(ClusterUnhealthyCount, "test-cluster-5")
	if counter == nil {
		t.Error("counter is nil")
	}

	counter = register.RecordCounterVec(ClusterUnhealthyCount, "non-existent")
	if counter != nil {
		t.Error("expected nil for non-existent cluster")
	}
}

func TestProcessMetrics(t *testing.T) {
	metricsGather := map[string][]map[string]string{
		PhaseDurationSeconds: {
			{
				"phase":      "test-phase",
				"start_time": "2024-01-01 00:00:00",
				"end_time":   "2024-01-01 01:00:00",
				"describe":   "test description",
			},
		},
		NodeBootstrapDurationSeconds: {
			{
				"node":         "node1",
				"start_time":   "2024-01-01 00:00:00",
				"end_time":     "2024-01-01 00:30:00",
				"boot_success": "true",
			},
		},
	}

	result := ProcessMetrics(metricsGather)
	if len(result) != 2 {
		t.Errorf("expected 2 results, got %d", len(result))
	}

	for _, item := range result {
		if item.Name == "" {
			t.Error("Name is empty")
		}
		if item.StartTime == "" {
			t.Error("StartTime is empty")
		}
		if item.EndTime == "" {
			t.Error("EndTime is empty")
		}
	}
}
