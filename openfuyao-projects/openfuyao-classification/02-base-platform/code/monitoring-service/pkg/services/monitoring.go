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

// Package services provides interfaces and implementations for interacting with Prometheus servers.
// It defines the MonitoringOperator interface, which encapsulates
// methods for querying metrics, retrieving metric names, targets information, and alerting rules
// from a Prometheus server.
package services

import (
	"time"

	"openfuyao.com/monitoring-service/pkg/client"
)

// MonitoringOperator defines an interface for querying metrics from Prometheus.
// It provides methods for both instant and range queries, retrieving available metric names,
// targets information, and configured alerting rules.
type MonitoringOperator interface {
	QueryInstantMetrics(expr string, time time.Time) client.Metric
	QueryRangeMetrics(expr string, start, end time.Time, step time.Duration) client.Metric
	QueryInstantCustomMetrics(metrics []string, time time.Time, opt client.QueryOption) Metrics
	QueryRangeCustomMetrics(metrics []string, start, end time.Time, step time.Duration,
		opt client.QueryOption) Metrics
	GetMetricNames() NativeMetricNames
	GetTargets() TargetsData
	GetAlertingRules() AlertingRulesData
}

type monitoringOperator struct {
	prometheus client.Interface
}

// NewMonitoringOperator creates a new monitoring operator using the provided Prometheus client interface.
func NewMonitoringOperator(monitoringClient client.Interface) MonitoringOperator {
	return &monitoringOperator{
		prometheus: monitoringClient,
	}
}

// QueryInstantMetrics executes an instant query on the Prometheus server using the given expression and time.
func (mo monitoringOperator) QueryInstantMetrics(expr string, time time.Time) client.Metric {
	return mo.prometheus.QueryInstantMetrics(expr, time)
}

// QueryRangeMetrics executes a range query on the Prometheus server using the given expression,
// start and end times, and the time step.
func (mo monitoringOperator) QueryRangeMetrics(expr string, start, end time.Time,
	step time.Duration) client.Metric {
	return mo.prometheus.QueryRangeMetrics(expr, start, end, step)
}

// QueryInstantCustomMetrics performs instant queries for custom metrics with specific options, at a given time.
func (mo monitoringOperator) QueryInstantCustomMetrics(metrics []string, time time.Time,
	opt client.QueryOption) Metrics {
	res := mo.prometheus.QueryInstantCustomMetrics(metrics, time, opt)
	return Metrics{Results: res}
}

// QueryRangeCustomMetrics performs range queries for custom metrics over a specified time range with specific options.
func (mo monitoringOperator) QueryRangeCustomMetrics(metrics []string, start, end time.Time,
	step time.Duration, opt client.QueryOption) Metrics {
	res := mo.prometheus.QueryRangeCustomMetrics(metrics, start, end, step, opt)
	return Metrics{Results: res}
}

// GetMetricNames retrieves the names of all metrics available in the Prometheus server.
func (mo monitoringOperator) GetMetricNames() NativeMetricNames {
	res, err := mo.prometheus.GetMetricNames()
	if err != nil {
		return NativeMetricNames{
			Msg:   err.Error(),
			Names: res,
		}
	}
	return NativeMetricNames{Names: res}
}

// GetTargets retrieves the current monitoring targets with their metadata.
func (mo monitoringOperator) GetTargets() TargetsData {
	res, err := mo.prometheus.GetTargets()
	if err != nil {
		return TargetsData{
			Msg:     err.Error(),
			Targets: res,
		}
	}
	return TargetsData{Targets: res}
}

// GetAlertingRules retrieves the alerting rules currently configured in the Prometheus server.
func (mo monitoringOperator) GetAlertingRules() AlertingRulesData {
	res, err := mo.prometheus.GetAlertingRules()
	if err != nil {
		return AlertingRulesData{
			Msg:           err.Error(),
			AlertingRules: res,
		}
	}
	return AlertingRulesData{AlertingRules: res}
}
