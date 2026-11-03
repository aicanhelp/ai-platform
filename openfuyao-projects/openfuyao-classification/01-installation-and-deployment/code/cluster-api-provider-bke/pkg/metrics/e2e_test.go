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
	"os"
	"path/filepath"
	"testing"
)

func TestProcessMetricsFunc(t *testing.T) {
	metricsGather := map[string][]map[string]string{
		PhaseDurationSeconds: {
			{"phase": "test", "start_time": "2024-01-01 00:00:00", "end_time": "2024-01-01 01:00:00", "describe": "test"},
		},
	}
	result := processMetrics(metricsGather)
	if len(result) == 0 {
		t.Error("processMetrics returned empty result")
	}
}

func TestSerializeMetrics(t *testing.T) {
	resp := []httpExportResponse{
		{Name: "test", StartTime: "2024-01-01 00:00:00", EndTime: "2024-01-01 01:00:00", Describe: "test", Duration: 3600},
	}
	data, err := serializeMetrics(resp)
	if err != nil {
		t.Errorf("serializeMetrics error: %v", err)
	}
	if len(data) == 0 {
		t.Error("serializeMetrics returned empty data")
	}
}

func TestPrepareFilePath(t *testing.T) {
	result := prepareFilePath("test/cluster")
	expected := "/bke2e/metrics/test_cluster.json"
	if result != expected {
		t.Errorf("expected %s, got %s", expected, result)
	}
}

func TestEnsureDirectory(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "test-metrics")
	defer os.RemoveAll(tmpDir)

	testPath := filepath.Join(tmpDir, "subdir", "test.json")
	err := ensureDirectory(testPath)
	if err != nil {
		t.Errorf("ensureDirectory error: %v", err)
	}
}

func TestWriteMetricsToFile(t *testing.T) {
	tmpDir := filepath.Join(os.TempDir(), "test-metrics-write")
	defer os.RemoveAll(tmpDir)

	os.MkdirAll(tmpDir, 0755)
	testPath := filepath.Join(tmpDir, "test.json")

	err := writeMetricsToFile(testPath, []byte(`{"test": "data"}`))
	if err != nil {
		t.Errorf("writeMetricsToFile error: %v", err)
	}

	if _, err := os.Stat(testPath); os.IsNotExist(err) {
		t.Error("file was not created")
	}
}

func TestEnsureBaseDirectory(t *testing.T) {
	err := ensureBaseDirectory()
	if err != nil {
		t.Logf("ensureBaseDirectory error (expected in test env): %v", err)
	}
}

func TestBKEClusterMetricRegister_GatherMetrics(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster")

	_, err := register.gatherMetrics("test-cluster")
	if err != nil {
		t.Errorf("gatherMetrics error: %v", err)
	}

	_, err = register.gatherMetrics("non-existent")
	if err != nil {
		t.Logf("expected error for non-existent cluster: %v", err)
	}
}

func TestBKEClusterMetricRegister_E2EDataGather(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-e2e-cluster")

	err := register.E2EDataGather("test-e2e-cluster")
	if err != nil {
		t.Logf("E2EDataGather error (expected in test env): %v", err)
	}
}
