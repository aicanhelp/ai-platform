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

package client

import "time"

// MetricTypeMatrix and MetricTypeVector represent the supported types of metric results.
const (
	MetricTypeMatrix = "matrix" // Matrix type is used for range vectors.
	MetricTypeVector = "vector" // Vector type is used for instant vectors.
)

// Metric encapsulates the details of a single metric including its name, associated data, and any errors.
type Metric struct {
	Code         *int   `json:"code,omitempty"`
	Msg          string `json:"msg,omitempty" csv:"-"`
	MetricName   string `json:"metricName,omitempty" description:"metric name, eg. cluster_nodes_count" csv:"metric_name"`
	MetricOutput `json:"data,omitempty" description:"actual metric result"`
}

// MetricOutput contains the result type and the actual data for a metric query.
type MetricOutput struct {
	Type          string         `json:"resultType,omitempty" description:"result type (matrix/vector)" csv:"metric_type"`
	MetricRecords []MetricRecord `json:"result,omitempty" description:"metric data" csv:"metric_values"`
}

// Point represents a single data point where each point is a pair (timestamp, value).
type Point struct {
	Timestamp float64 `json:"timestamp"`
	Value     float64 `json:"value"`
}

// MetricRecord holds the metadata and actual values for each time series of a metric.
type MetricRecord struct {
	Labels       map[string]string `json:"labels,omitempty" description:"time series labels"`
	InstantValue *Point            `json:"sample,omitempty" description:"time series, values of vector type"`
	RangeValues  []Point           `json:"series,omitempty" description:"time series, values of matrix type"`
}

// NativeMetricName represents the name of a metric.
type NativeMetricName struct {
	NativeMetricName string `json:"nativeMetricName" description:"native metric name"`
}

// Target details a specific monitoring target with metadata, scrape status, and health.
type Target struct {
	TargetName         string    `json:"targetName" description:"target name"`
	Endpoint           string    `json:"endpoint" description:"endpoint"`
	Service            string    `json:"service" description:"service"`
	Health             string    `json:"health" description:"discovered labels health. eg. up, down, unknown"`
	Namespace          string    `json:"namespace" description:"namespace"`
	LastError          string    `json:"lastError" description:"error encountered during the last scrape attempt"`
	LastScrape         time.Time `json:"lastScrape" description:"last scrape"`
	LastScrapeDuration float64   `json:"lastScrapeDuration" description:"last scrape duration"`
}

// AlertingRule defines the properties of an alerting rule including its group, name, severity, and current status.
type AlertingRule struct {
	Group       string  `json:"group" description:"alerting rule group"`
	Name        string  `json:"name" description:"alerting rule name"`
	Severity    string  `json:"severity" description:"alerting rule severity"`
	Status      string  `json:"status" description:"alerting rule status"`
	Duration    float64 `json:"duration" description:"alerting rule duration"`
	Description string  `json:"description" description:"alerting rule description"`
	Summary     string  `json:"summary" description:"alerting rule summary"`
	Query       string  `json:"query" description:"alerting rule query"`
}
