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

// Package collector provides functionality to collect NUMA-related metrics for Prometheus.
package collector

import (
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/zlog"
)

var metricDescriptors = struct {
	duration *prometheus.Desc
	success  *prometheus.Desc
}{
	duration: prometheus.NewDesc(
		prometheus.BuildFQName("numa", "scrape", "collector_duration_seconds"),
		"numa_exporter: Duration of a collector scrape.",
		[]string{"collector"},
		nil,
	),
	success: prometheus.NewDesc(
		prometheus.BuildFQName("numa", "scrape", "collector_success"),
		"numa_exporter: Whether a collector succeeded or failed.",
		[]string{"collector"},
		nil,
	),
}

var collectorFactories = make(map[string]func() (MetricCollector, error))

// NumaCollector aggregates multiple collectors for scraping NUMA-related metrics.
type NumaCollector struct {
	collectors map[string]MetricCollector
}

// MetricCollector is an interface for objects that can collect Prometheus metrics.
type MetricCollector interface {
	Update(ch chan<- prometheus.Metric) error
}

func registerCollector(name string, factory func() (MetricCollector, error)) {
	collectorFactories[name] = factory
}

// NewNumaCollector creates a new NumaCollector instance with all registered collectors
func NewNumaCollector() (*NumaCollector, error) {
	collectors := make(map[string]MetricCollector)

	for name, factory := range collectorFactories {
		collector, err := factory()
		if err != nil {
			return nil, err
		}
		collectors[name] = collector
	}
	return &NumaCollector{collectors: collectors}, nil
}

// Describe sends the descriptions of all metrics to the Prometheus metrics channel.
func (n NumaCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- metricDescriptors.duration
	ch <- metricDescriptors.success
}

// Collect scrapes metrics from all registered collectors and sends them to the Prometheus metrics channel.
func (n NumaCollector) Collect(ch chan<- prometheus.Metric) {
	wg := sync.WaitGroup{}
	wg.Add(len(n.collectors))

	for name, collector := range n.collectors {
		go scrapeCollector(name, collector, ch, &wg)
	}

	wg.Wait()
}

func scrapeCollector(name string, collector MetricCollector,
	ch chan<- prometheus.Metric, wg *sync.WaitGroup) {
	defer wg.Done()
	collectMetrics(name, collector, ch)
}

func collectMetrics(name string, collector MetricCollector, ch chan<- prometheus.Metric) {
	start := time.Now()
	err := collector.Update(ch)
	duration := time.Since(start)

	success := float64(0)
	if err != nil {
		zlog.FormatError("Collector failed. Collector Name: %s, Duration: %.2f seconds, Error: %s",
			name, duration.Seconds(), err)
	} else {
		zlog.FormatInfo("Collector succeeded. Collector Name: %s, Duration: %.2f seconds",
			name, duration.Seconds())
		success = 1
	}

	ch <- prometheus.MustNewConstMetric(metricDescriptors.duration, prometheus.GaugeValue, duration.Seconds(), name)
	ch <- prometheus.MustNewConstMetric(metricDescriptors.success, prometheus.GaugeValue, success, name)
}
