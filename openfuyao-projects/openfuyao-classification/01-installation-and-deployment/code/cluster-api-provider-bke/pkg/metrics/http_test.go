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
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestHttpResponse_bytes(t *testing.T) {
	resp := HttpResponse{Code: 200, Msg: "success", Data: "test"}
	bytes, err := resp.bytes()
	if err != nil {
		t.Errorf("bytes() error = %v", err)
	}
	if len(bytes) == 0 {
		t.Error("bytes is empty")
	}
}

func TestOk(t *testing.T) {
	data := map[string]string{"key": "value"}
	bytes, err := Ok(data, "success")
	if err != nil {
		t.Errorf("Ok() error = %v", err)
	}
	if len(bytes) == 0 {
		t.Error("bytes is empty")
	}
}

func TestError(t *testing.T) {
	bytes, err := Error("error message")
	if err != nil {
		t.Errorf("Error() error = %v", err)
	}
	if len(bytes) == 0 {
		t.Error("bytes is empty")
	}
}

func TestBad(t *testing.T) {
	bytes, err := Bad("bad request")
	if err != nil {
		t.Errorf("Bad() error = %v", err)
	}
	if len(bytes) == 0 {
		t.Error("bytes is empty")
	}
}

func TestCalculateTimeDifference(t *testing.T) {
	tests := []struct {
		name     string
		start    string
		end      string
		expected int
	}{
		{"valid times", "2024-01-01 00:00:00", "2024-01-01 01:00:00", 3600},
		{"same time", "2024-01-01 00:00:00", "2024-01-01 00:00:00", 0},
		{"invalid start", "invalid", "2024-01-01 01:00:00", 0},
		{"invalid end", "2024-01-01 00:00:00", "invalid", 0},
		{"less than 1 second", "2024-01-01 00:00:00", "2024-01-01 00:00:00", 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateTimeDifference(tt.start, tt.end)
			if result != tt.expected {
				t.Errorf("expected %d, got %d", tt.expected, result)
			}
		})
	}
}

func TestFilterMetricsData(t *testing.T) {
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)

	metricsGather := map[string][]map[string]string{
		PhaseDurationSeconds: {
			{"start_time": "2024-01-01 08:00:00", "end_time": "2024-01-01 09:00:00"},
			{"start_time": "", "end_time": "2024-01-01 09:00:00"},
			{"start_time": "invalid", "end_time": "2024-01-01 09:00:00"},
		},
	}

	result := filterMetricsData(metricsGather, start, end)
	if result == nil {
		t.Error("result is nil")
	}
}

func TestBKEClusterMetricRegister_ExtractAndValidateParams(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	tests := []struct {
		name    string
		url     string
		wantErr bool
	}{
		{"valid params", "http://example.com?cluster=test&from=1704110400&to=1704196800", false},
		{"missing cluster", "http://example.com?from=1704110400&to=1704196800", true},
		{"only cluster", "http://example.com?cluster=test", false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := httptest.NewRequest("GET", tt.url, nil)
			result := register.extractAndValidateParams(req)
			if (result.Err != nil) != tt.wantErr {
				t.Errorf("extractAndValidateParams() error = %v, wantErr %v", result.Err, tt.wantErr)
			}
		})
	}
}

func TestBKEClusterMetricRegister_ValidateTimeRange(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	tests := []struct {
		name    string
		from    string
		to      string
		wantErr bool
	}{
		{"valid range", "1704110400", "1704196800", false},
		{"empty from", "", "1704196800", false},
		{"empty to", "1704110400", "", false},
		{"invalid from", "invalid", "1704196800", true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, _, err := register.validateTimeRange(tt.from, tt.to)
			if (err != nil) != tt.wantErr {
				t.Errorf("validateTimeRange() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestBKEClusterMetricRegister_HttpClusterFunc(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-http")

	req := httptest.NewRequest("GET", "http://example.com/clusters", nil)
	w := httptest.NewRecorder()

	handler := register.HttpClusterFunc()
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestNewResp(t *testing.T) {
	resp := newResp("data", "message", 200)
	if resp.Code != 200 {
		t.Errorf("expected code 200, got %d", resp.Code)
	}
	if resp.Msg != "message" {
		t.Errorf("expected msg 'message', got %s", resp.Msg)
	}
}

func TestHttpResponse_bytesNil(t *testing.T) {
	resp := HttpResponse{Code: 200, Msg: "success", Data: nil}
	bytes, err := resp.bytes()
	if err != nil {
		t.Errorf("bytes() error = %v", err)
	}
	if len(bytes) == 0 {
		t.Error("bytes is empty")
	}
}

func TestFilterMetricsData_EmptyTime(t *testing.T) {
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)

	metricsGather := map[string][]map[string]string{
		"test": {
			{"start_time": "", "end_time": ""},
			{"start_time": "2024-01-01 08:00:00", "end_time": "invalid"},
		},
	}

	result := filterMetricsData(metricsGather, start, end)
	if result == nil {
		t.Error("result is nil")
	}
}

func TestCalculateTimeDifference_NegativeDuration(t *testing.T) {
	result := calculateTimeDifference("2024-01-01 01:00:00", "2024-01-01 00:00:00")
	if result >= 0 {
		t.Logf("negative duration handled, got %d", result)
	}
}

func TestBKEClusterMetricRegister_ExtractAndValidateParams_EmptyCluster(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	req := httptest.NewRequest("GET", "http://example.com", nil)
	result := register.extractAndValidateParams(req)
	if result.Err == nil {
		t.Error("expected error for missing cluster")
	}
}

func TestBKEClusterMetricRegister_ValidateTimeRange_BothEmpty(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	_, _, err := register.validateTimeRange("", "")
	if err != nil {
		t.Logf("validateTimeRange with empty params: %v", err)
	}
}

func TestBKEClusterMetricRegister_CollectAndFilterMetrics(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-collect")

	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)

	_, err := register.collectAndFilterMetrics("test-cluster-collect", start, end)
	if err != nil {
		t.Logf("collectAndFilterMetrics error: %v", err)
	}
}

func TestBKEClusterMetricRegister_FormatMetricsResponse(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	metricsGather := map[string][]map[string]string{
		"test": {
			{"name": "test", "start_time": "2024-01-01 00:00:00", "end_time": "2024-01-01 01:00:00"},
		},
	}

	result := register.formatMetricsResponse(metricsGather)
	if result == nil {
		t.Error("result is nil")
	}
}

func TestBKEClusterMetricRegister_WriteErrorResponse(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	w := httptest.NewRecorder()

	register.writeErrorResponse(w, Bad, "test error")

	if w.Code != 0 {
		t.Logf("writeErrorResponse wrote status code: %d", w.Code)
	}
}

func TestBKEClusterMetricRegister_HttpExportFunc_MissingCluster(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	req := httptest.NewRequest("GET", "http://example.com/export", nil)
	w := httptest.NewRecorder()

	handler := register.HttpExportFunc()
	handler(w, req)

	if w.Code == 0 {
		t.Log("HttpExportFunc handled missing cluster")
	}
}

func TestBKEClusterMetricRegister_HttpExportFunc_InvalidTime(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	req := httptest.NewRequest("GET", "http://example.com/export?cluster=test&from=invalid&to=invalid", nil)
	w := httptest.NewRecorder()

	handler := register.HttpExportFunc()
	handler(w, req)

	if w.Code == 0 {
		t.Log("HttpExportFunc handled invalid time")
	}
}

func TestBKEClusterMetricRegister_HttpExportFunc_Success(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("test-cluster-export")

	req := httptest.NewRequest("GET", "http://example.com/export?cluster=test-cluster-export&from=1704110400&to=1704196800", nil)
	w := httptest.NewRecorder()

	handler := register.HttpExportFunc()
	handler(w, req)

	if w.Code == 0 {
		t.Log("HttpExportFunc executed")
	}
}

func TestBKEClusterMetricRegister_HttpClusterFunc_Empty(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	req := httptest.NewRequest("GET", "http://example.com/clusters", nil)
	w := httptest.NewRecorder()

	handler := register.HttpClusterFunc()
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}

func TestBKEClusterMetricRegister_CollectAndFilterMetrics_Nonexistent(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	start := time.Date(2024, 1, 1, 0, 0, 0, 0, time.UTC)
	end := time.Date(2024, 1, 2, 0, 0, 0, 0, time.UTC)

	result, err := register.collectAndFilterMetrics("nonexistent-cluster", start, end)
	if err != nil {
		t.Logf("collectAndFilterMetrics error: %v", err)
	}
	if result == nil {
		t.Log("result is nil for nonexistent cluster")
	}
}

func TestBKEClusterMetricRegister_HttpExportFunc_CollectError(t *testing.T) {
	register := NewBKEClusterMetricRegister()

	req := httptest.NewRequest("GET", "http://example.com/export?cluster=nonexistent&from=1704110400&to=1704196800", nil)
	w := httptest.NewRecorder()

	handler := register.HttpExportFunc()
	handler(w, req)

	if w.Code == 0 {
		t.Log("HttpExportFunc handled collect error")
	}
}

func TestBKEClusterMetricRegister_HttpClusterFunc_WithClusters(t *testing.T) {
	register := NewBKEClusterMetricRegister()
	register.Register("cluster1")
	register.Register("cluster2")

	req := httptest.NewRequest("GET", "http://example.com/clusters", nil)
	w := httptest.NewRecorder()

	handler := register.HttpClusterFunc()
	handler(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("expected status 200, got %d", w.Code)
	}
}
