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

import (
	"time"
)

// Interface defines a set of methods for querying metrics.
// It provides methods to query both instant and range metrics, custom metrics,
// and to retrieve information like metric names, targets, and alerting rules.
type Interface interface {
	QueryInstantMetrics(expr string, time time.Time) Metric
	QueryRangeMetrics(expr string, start, end time.Time, step time.Duration) Metric
	QueryInstantCustomMetrics(metrics []string, time time.Time, opt QueryOption) []Metric
	QueryRangeCustomMetrics(metrics []string, start, end time.Time, step time.Duration, opt QueryOption) []Metric
	GetMetricNames() ([]NativeMetricName, error)
	GetTargets() ([]Target, error)
	GetAlertingRules() ([]AlertingRule, error)
}
