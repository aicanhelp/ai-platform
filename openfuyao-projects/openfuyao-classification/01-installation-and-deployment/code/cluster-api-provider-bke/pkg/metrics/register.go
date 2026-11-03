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
	"fmt"
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"sigs.k8s.io/controller-runtime/pkg/metrics"
)

var metricRegister = NewBKEClusterMetricRegister()

func DefaultRegister() *BKEClusterMetricRegister {
	return &metricRegister
}

type BKEClusterMetricRegister struct {
	collectors map[string]Collector
	mux        sync.RWMutex
}

func NewBKEClusterMetricRegister() BKEClusterMetricRegister {
	return BKEClusterMetricRegister{
		collectors: make(map[string]Collector),
	}
}

func (r *BKEClusterMetricRegister) Register(key string) {
	r.mux.Lock()
	defer r.mux.Unlock()
	if r.collectors[key] != nil {
		return
	}
	r.collectors[key] = NewBKEClusterCollector(key)
	metrics.Registry.MustRegister(r.collectors[key])
}

func (r *BKEClusterMetricRegister) Unregister(key string) {
	r.mux.Lock()
	defer r.mux.Unlock()
	if r.collectors[key] == nil {
		return
	}
	metrics.Registry.Unregister(r.collectors[key])
	delete(r.collectors, key)
}

func (r *BKEClusterMetricRegister) Gather(key string, vecName ...string) (map[string][]map[string]string, error) {
	r.mux.RLock()
	collector, ok := r.collectors[key]
	r.mux.RUnlock()
	if !ok {
		return nil, nil
	}

	return collector.Export(vecName...)
}

func (r *BKEClusterMetricRegister) RecordGaugeVec(vecName, key string, lvs ...string) prometheus.Gauge {
	r.mux.RLock()
	defer r.mux.RUnlock()
	if collector, ok := r.collectors[key]; ok {
		return collector.RecordGaugeVecWithLabelValues(vecName, lvs...)
	}
	return nil
}

func (r *BKEClusterMetricRegister) RecordCounterVec(vecName, key string, lvs ...string) prometheus.Counter {
	r.mux.RLock()
	defer r.mux.RUnlock()
	if collector, ok := r.collectors[key]; ok {
		return collector.RecordCounterVecWithLabelValues(vecName, lvs...)
	}
	return nil
}

// ProcessMetrics converts metrics to httpExportResponse format
func ProcessMetrics(metricsGather map[string][]map[string]string) []httpExportResponse {
	resp := make([]httpExportResponse, 0)
	for name, metrics := range metricsGather {
		switch name {
		case PhaseDurationSeconds:
			for _, metric := range metrics {
				respItem := httpExportResponse{
					Name:      metric["phase"],
					StartTime: metric["start_time"],
					EndTime:   metric["end_time"],
					Describe:  metric["describe"],
					Duration:  calculateTimeDifference(metric["start_time"], metric["end_time"]),
				}
				resp = append(resp, respItem)
			}
		case NodeBootstrapDurationSeconds:
			for _, metric := range metrics {
				respItem := httpExportResponse{
					Name:      metric["node"],
					StartTime: metric["start_time"],
					EndTime:   metric["end_time"],
					Describe:  fmt.Sprintf("node: %s, bootstrap success: %s", metric["node"], metric["boot_success"]),
					Duration:  calculateTimeDifference(metric["start_time"], metric["end_time"]),
				}
				resp = append(resp, respItem)
			}
		default:
		}
	}
	return resp
}
