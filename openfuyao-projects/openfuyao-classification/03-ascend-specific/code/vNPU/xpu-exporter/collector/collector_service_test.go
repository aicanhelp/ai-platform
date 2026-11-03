/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package collector

import (
	"context"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
)

type mockCollector struct {
	desc *prometheus.Desc
}

func (m *mockCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- m.desc
}

func (m *mockCollector) Collect(ch chan<- prometheus.Metric) {
	metric, err := prometheus.NewConstMetric(m.desc, prometheus.GaugeValue, 1.0)
	if err != nil {
		return
	}
	ch <- metric
}

type mockCollectorService struct {
	name      string
	collector prometheus.Collector
	started   bool
}

func (m *mockCollectorService) CreateCollector(cacheTime time.Duration, updateTime time.Duration) prometheus.Collector {
	return m.collector
}

func (m *mockCollectorService) Start(ctx context.Context, fn context.CancelFunc) {
	m.started = true
	fn()
}

func (m *mockCollectorService) GetName() string {
	return m.name
}

func TestICollectorService_CreateCollector(t *testing.T) {
	mc := &mockCollector{
		desc: prometheus.NewDesc("test_metric", "test help", nil, nil),
	}
	service := &mockCollectorService{
		collector: mc,
	}
	result := service.CreateCollector(60*time.Second, 15*time.Second)
	assert.NotNil(t, result)
}

func TestICollectorService_Start(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	service := &mockCollectorService{}
	service.Start(ctx, cancel)
	assert.True(t, service.started)
}

func TestICollectorService_GetName(t *testing.T) {
	expectedName := "test_service"
	service := &mockCollectorService{
		name: expectedName,
	}
	name := service.GetName()
	assert.Equal(t, expectedName, name)
}

func TestICollectorService_InterfaceCompliance(t *testing.T) {
	var _ ICollectorService = &mockCollectorService{}
	mc := &mockCollector{
		desc: prometheus.NewDesc("test_metric", "test help", nil, nil),
	}
	service := &mockCollectorService{
		name:      "npu",
		collector: mc,
	}
	assert.NotNil(t, service)
	assert.Equal(t, "npu", service.GetName())
	assert.NotNil(t, service.CreateCollector(60*time.Second, 15*time.Second))
	ctx, cancel := context.WithCancel(context.Background())
	service.Start(ctx, cancel)
	assert.True(t, service.started)
}
