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

package services

import (
	"fmt"
	"testing"
	"time"

	"openfuyao.com/monitoring-service/pkg/client"
)

// mockClient is a mock implementation of client.Interface for testing
type mockClient struct {
	queryInstantMetricsFunc       func(expr string, ts time.Time) client.Metric
	queryRangeMetricsFunc         func(expr string, start, end time.Time, step time.Duration) client.Metric
	queryInstantCustomMetricsFunc func(metrics []string, ts time.Time, o client.QueryOption) []client.Metric
	queryRangeCustomMetricsFunc   func(metrics []string, start, end time.Time, step time.Duration, o client.QueryOption) []client.Metric
	getMetricNamesFunc            func() ([]client.NativeMetricName, error)
	getTargetsFunc                func() ([]client.Target, error)
	getAlertingRulesFunc          func() ([]client.AlertingRule, error)
}

func (m *mockClient) QueryInstantMetrics(expr string, ts time.Time) client.Metric {
	if m.queryInstantMetricsFunc != nil {
		return m.queryInstantMetricsFunc(expr, ts)
	}
	return client.Metric{}
}

func (m *mockClient) QueryRangeMetrics(expr string, start, end time.Time, step time.Duration) client.Metric {
	if m.queryRangeMetricsFunc != nil {
		return m.queryRangeMetricsFunc(expr, start, end, step)
	}
	return client.Metric{}
}

func (m *mockClient) QueryInstantCustomMetrics(metrics []string, ts time.Time, o client.QueryOption) []client.Metric {
	if m.queryInstantCustomMetricsFunc != nil {
		return m.queryInstantCustomMetricsFunc(metrics, ts, o)
	}
	return []client.Metric{}
}

func (m *mockClient) QueryRangeCustomMetrics(metrics []string, start, end time.Time, step time.Duration, o client.QueryOption) []client.Metric {
	if m.queryRangeCustomMetricsFunc != nil {
		return m.queryRangeCustomMetricsFunc(metrics, start, end, step, o)
	}
	return []client.Metric{}
}

func (m *mockClient) GetMetricNames() ([]client.NativeMetricName, error) {
	if m.getMetricNamesFunc != nil {
		return m.getMetricNamesFunc()
	}
	return []client.NativeMetricName{}, nil
}

func (m *mockClient) GetTargets() ([]client.Target, error) {
	if m.getTargetsFunc != nil {
		return m.getTargetsFunc()
	}
	return []client.Target{}, nil
}

func (m *mockClient) GetAlertingRules() ([]client.AlertingRule, error) {
	if m.getAlertingRulesFunc != nil {
		return m.getAlertingRulesFunc()
	}
	return []client.AlertingRule{}, nil
}

func TestNewMonitoringOperator(t *testing.T) {
	mock := &mockClient{}
	operator := NewMonitoringOperator(mock)

	if operator == nil {
		t.Fatal("NewMonitoringOperator() returned nil")
	}

	// Verify it implements the interface
	var _ MonitoringOperator = operator
}

func TestMonitoringOperatorQueryInstantMetrics(t *testing.T) {
	expectedMetric := client.Metric{
		Code: func() *int { var c int = 0; return &c }(),
		MetricOutput: client.MetricOutput{
			Type: client.MetricTypeVector,
		},
	}

	mock := &mockClient{
		queryInstantMetricsFunc: func(expr string, ts time.Time) client.Metric {
			return expectedMetric
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.QueryInstantMetrics("up", time.Now())

	if result.Code == nil || *result.Code != 0 {
		t.Errorf("QueryInstantMetrics() = %v, want code 0", result)
	}
}

func TestMonitoringOperatorQueryRangeMetrics(t *testing.T) {
	expectedMetric := client.Metric{
		Code: func() *int { var c int = 0; return &c }(),
		MetricOutput: client.MetricOutput{
			Type: client.MetricTypeMatrix,
		},
	}

	mock := &mockClient{
		queryRangeMetricsFunc: func(expr string, start, end time.Time, step time.Duration) client.Metric {
			return expectedMetric
		},
	}

	operator := NewMonitoringOperator(mock)
	start := time.Now()
	end := start.Add(5 * time.Minute)
	result := operator.QueryRangeMetrics("up", start, end, 15*time.Second)

	if result.Code == nil || *result.Code != 0 {
		t.Errorf("QueryRangeMetrics() = %v, want code 0", result)
	}
}

func TestMonitoringOperatorQueryInstantCustomMetrics(t *testing.T) {
	expectedMetrics := []client.Metric{
		{
			MetricName: "cluster_cpu_utilisation",
			MetricOutput: client.MetricOutput{
				Type: client.MetricTypeVector,
			},
		},
	}

	mock := &mockClient{
		queryInstantCustomMetricsFunc: func(metrics []string, ts time.Time, o client.QueryOption) []client.Metric {
			return expectedMetrics
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.QueryInstantCustomMetrics([]string{"cluster_cpu_utilisation"}, time.Now(), client.WithClusterLevel())

	if len(result.Results) != 1 {
		t.Errorf("QueryInstantCustomMetrics() returned %d results, want 1", len(result.Results))
	}
	if result.Results[0].MetricName != "cluster_cpu_utilisation" {
		t.Errorf("QueryInstantCustomMetrics() MetricName = %v, want cluster_cpu_utilisation", result.Results[0].MetricName)
	}
}

func TestMonitoringOperatorQueryRangeCustomMetrics(t *testing.T) {
	expectedMetrics := []client.Metric{
		{
			MetricName: "cluster_cpu_utilisation",
			MetricOutput: client.MetricOutput{
				Type: client.MetricTypeMatrix,
			},
		},
	}

	mock := &mockClient{
		queryRangeCustomMetricsFunc: func(metrics []string, start, end time.Time, step time.Duration, o client.QueryOption) []client.Metric {
			return expectedMetrics
		},
	}

	operator := NewMonitoringOperator(mock)
	start := time.Now()
	end := start.Add(5 * time.Minute)
	result := operator.QueryRangeCustomMetrics([]string{"cluster_cpu_utilisation"}, start, end, 15*time.Second, client.WithClusterLevel())

	if len(result.Results) != 1 {
		t.Errorf("QueryRangeCustomMetrics() returned %d results, want 1", len(result.Results))
	}
	if result.Results[0].MetricName != "cluster_cpu_utilisation" {
		t.Errorf("QueryRangeCustomMetrics() MetricName = %v, want cluster_cpu_utilisation", result.Results[0].MetricName)
	}
}

func TestMonitoringOperatorGetMetricNames(t *testing.T) {
	expectedNames := []client.NativeMetricName{
		{NativeMetricName: "up"},
		{NativeMetricName: "cpu_usage"},
	}

	mock := &mockClient{
		getMetricNamesFunc: func() ([]client.NativeMetricName, error) {
			return expectedNames, nil
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetMetricNames()

	if result.Msg != "" {
		t.Errorf("GetMetricNames() Msg = %v, want empty", result.Msg)
	}
	if len(result.Names) != 2 {
		t.Errorf("GetMetricNames() returned %d names, want 2", len(result.Names))
	}
}

func TestMonitoringOperatorGetMetricNames_Error(t *testing.T) {
	mock := &mockClient{
		getMetricNamesFunc: func() ([]client.NativeMetricName, error) {
			return []client.NativeMetricName{}, fmt.Errorf("connection error")
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetMetricNames()

	if result.Msg == "" {
		t.Error("GetMetricNames() should have error message")
	}
}

func TestMonitoringOperatorGetTargets(t *testing.T) {
	expectedTargets := []client.Target{
		{
			TargetName: "test-target",
			Health:     "up",
		},
	}

	mock := &mockClient{
		getTargetsFunc: func() ([]client.Target, error) {
			return expectedTargets, nil
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetTargets()

	if result.Msg != "" {
		t.Errorf("GetTargets() Msg = %v, want empty", result.Msg)
	}
	if len(result.Targets) != 1 {
		t.Errorf("GetTargets() returned %d targets, want 1", len(result.Targets))
	}
}

func TestMonitoringOperatorGetTargetsError(t *testing.T) {
	mock := &mockClient{
		getTargetsFunc: func() ([]client.Target, error) {
			return []client.Target{}, fmt.Errorf("connection error")
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetTargets()

	if result.Msg == "" {
		t.Error("GetTargets() should have error message")
	}
}

func TestMonitoringOperatorGetAlertingRules(t *testing.T) {
	expectedRules := []client.AlertingRule{
		{
			Name:   "TestRule",
			Status: "inactive",
		},
	}

	mock := &mockClient{
		getAlertingRulesFunc: func() ([]client.AlertingRule, error) {
			return expectedRules, nil
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetAlertingRules()

	if result.Msg != "" {
		t.Errorf("GetAlertingRules() Msg = %v, want empty", result.Msg)
	}
	if len(result.AlertingRules) != 1 {
		t.Errorf("GetAlertingRules() returned %d rules, want 1", len(result.AlertingRules))
	}
}

func TestMonitoringOperatorGetAlertingRules_Error(t *testing.T) {
	mock := &mockClient{
		getAlertingRulesFunc: func() ([]client.AlertingRule, error) {
			return []client.AlertingRule{}, fmt.Errorf("connection error")
		},
	}

	operator := NewMonitoringOperator(mock)
	result := operator.GetAlertingRules()

	if result.Msg == "" {
		t.Error("GetAlertingRules() should have error message")
	}
}
