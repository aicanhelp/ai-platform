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
	"errors"
	"reflect"
	"testing"

	"github.com/prometheus/client_golang/prometheus"
)

type MockMetricCollector struct {
	errToReturn error
}

func (m *MockMetricCollector) Update(ch chan<- prometheus.Metric) error {
	if m.errToReturn != nil {
		return m.errToReturn
	}
	return nil
}

func NewMockMetricCollectorSuccess() (MetricCollector, error) {
	return &MockMetricCollector{}, nil
}

func NewMockMetricCollectorFail() (MetricCollector, error) {
	return nil, errors.New("failed to create collector")
}

func TestNewNumaCollector(t *testing.T) {
	tests := []struct {
		name      string
		factories map[string]func() (MetricCollector, error)
		want      *NumaCollector
		wantErr   bool
	}{
		{
			name: "successfully creates NumaCollector with collectors",
			factories: map[string]func() (MetricCollector, error){
				"cpu":    NewMockMetricCollectorSuccess,
				"memory": NewMockMetricCollectorSuccess,
			},
			want: &NumaCollector{
				collectors: map[string]MetricCollector{
					"cpu":    &MockMetricCollector{},
					"memory": &MockMetricCollector{},
				},
			},
			wantErr: false,
		},
		{
			name: "factory function fails",
			factories: map[string]func() (MetricCollector, error){
				"cpu":    NewMockMetricCollectorSuccess,
				"memory": NewMockMetricCollectorFail,
			},
			want:    nil,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			originalFactories := collectorFactories
			collectorFactories = tt.factories
			got, err := NewNumaCollector()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewNumaCollector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewNumaCollector() got = %v, want %v", got, tt.want)
			}
			collectorFactories = originalFactories
		})
	}
}

func TestNumaCollectorDescribe(t *testing.T) {
	type fields struct {
		collectors map[string]MetricCollector
	}
	type args struct {
		ch chan *prometheus.Desc
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "success to describe Numa Collector",
			fields: fields{
				collectors: map[string]MetricCollector{},
			},
			args: args{
				ch: make(chan *prometheus.Desc, 2),
			},
		},
	}
	var channelNum = 2
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := NumaCollector{
				collectors: tt.fields.collectors,
			}
			n.Describe(tt.args.ch)
			close(tt.args.ch)
			var Desc []*prometheus.Desc
			for desc := range tt.args.ch {
				Desc = append(Desc, desc)
			}
			if len(Desc) != channelNum {
				t.Fatalf("Excepted length of description is 2 but got %d", len(Desc))
				return
			}
			if Desc[0] != metricDescriptors.duration {
				t.Errorf("Expected first description to be %v, but got %v", metricDescriptors.duration, Desc[0])
			}

			if Desc[1] != metricDescriptors.success {
				t.Errorf("Expected second description to be %v, but got %v", metricDescriptors.success, Desc[1])
			}
		})
	}
}

func TestNumaCollectorCollect(t *testing.T) {
	type fields struct {
		collectors map[string]MetricCollector
	}
	type args struct {
		ch chan prometheus.Metric
	}
	tests := []struct {
		name   string
		fields fields
		args   args
	}{
		{
			name: "success",
			fields: fields{
				map[string]MetricCollector{
					"collector1": &MockMetricCollector{},
					"collector2": &MockMetricCollector{},
				},
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			n := NumaCollector{
				collectors: tt.fields.collectors,
			}
			n.Collect(tt.args.ch)
		})
	}
}

func TestCollectMetrics(t *testing.T) {
	type args struct {
		name      string
		collector MetricCollector
		ch        chan<- prometheus.Metric
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "fail",
			args: args{
				name:      "",
				collector: &MockMetricCollector{errToReturn: errors.New("")},
				ch:        make(chan<- prometheus.Metric, 2),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			collectMetrics(tt.args.name, tt.args.collector, tt.args.ch)
		})
	}
}
