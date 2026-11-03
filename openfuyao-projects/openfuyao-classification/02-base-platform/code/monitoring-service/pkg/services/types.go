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
	"openfuyao.com/monitoring-service/pkg/client"
)

// Metrics holds a collection of metrics results.
type Metrics struct {
	Results []client.Metric `json:"results" description:"actual array of results"`
}

// NativeMetricNames holds a list of metric names.
type NativeMetricNames struct {
	Msg   string                    `json:"msg,omitempty" csv:"-"`
	Names []client.NativeMetricName `json:"names"`
}

// TargetsData holds a list of monitoring targets.
type TargetsData struct {
	Msg     string          `json:"msg,omitempty" csv:"-"`
	Targets []client.Target `json:"targets"`
}

// AlertingRulesData holds a collection of alerting rules.
type AlertingRulesData struct {
	Msg           string                `json:"msg,omitempty" csv:"-"`
	AlertingRules []client.AlertingRule `json:"alertingRules"`
}
