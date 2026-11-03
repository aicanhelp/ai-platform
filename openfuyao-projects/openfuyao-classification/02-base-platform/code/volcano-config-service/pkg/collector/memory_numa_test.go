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

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
)

func TestNewMemNumaCollector(t *testing.T) {
	tests := []struct {
		name    string
		want    MetricCollector
		wantErr bool
	}{
		{
			name: "success",
			want: &memNumaCollector{
				metricDescriptions: map[string]*prometheus.Desc{},
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewMemNumaCollector()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewMemNumaCollector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewMemNumaCollector() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMemNumaCollectorUpdate(t *testing.T) {
	type fields struct {
		metricDescriptions map[string]*prometheus.Desc
	}
	type args struct {
		ch chan<- prometheus.Metric
	}
	tests := []struct {
		name          string
		fields        fields
		args          args
		wantErr       bool
		mockNodeName  string
		mockMetric    []memNumaMetric
		mockNodeErr   error
		mockMetricErr error
	}{
		{
			name: "success",
			fields: fields{
				metricDescriptions: map[string]*prometheus.Desc{},
			},
			args: args{
				ch: make(chan<- prometheus.Metric, 10),
			},
			wantErr:      false,
			mockNodeName: "node0",
			mockMetric: []memNumaMetric{
				{
					metricName:      "MemAvailable",
					metricValueType: prometheus.GaugeValue,
					nodeName:        "node0",
					numaNodeName:    "0",
					metricValue:     23456,
				},
			},
			mockNodeErr:   nil,
			mockMetricErr: nil,
		},
		{
			name: "fail",
			fields: fields{
				metricDescriptions: map[string]*prometheus.Desc{},
			},
			args: args{
				ch: make(chan<- prometheus.Metric, 10),
			},
			wantErr:      true,
			mockNodeName: "",
			mockMetric: []memNumaMetric{
				{
					metricName:      "MemAvailable",
					metricValueType: prometheus.GaugeValue,
					nodeName:        "node0",
					numaNodeName:    "0",
					metricValue:     23456,
				},
			},
			mockNodeErr:   errors.New("node error"),
			mockMetricErr: nil,
		},
		{
			name: "fail Metric",
			fields: fields{
				metricDescriptions: map[string]*prometheus.Desc{},
			},
			args: args{
				ch: make(chan<- prometheus.Metric, 10),
			},
			wantErr:       true,
			mockNodeName:  "node0",
			mockMetric:    []memNumaMetric{},
			mockNodeErr:   nil,
			mockMetricErr: errors.New("metric error"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(utils.GetNodeName, func() (string, error) {
				if tt.mockNodeErr != nil {
					return "", tt.mockNodeErr
				}
				return tt.mockNodeName, nil
			})
			metric := gomonkey.ApplyFunc(getMemInfoNuma, func(string) ([]memNumaMetric, error) {
				if tt.mockMetricErr != nil {
					return nil, tt.mockMetricErr
				}
				return tt.mockMetric, nil
			})
			defer patch.Reset()
			defer metric.Reset()
			m := memNumaCollector{
				metricDescriptions: tt.fields.metricDescriptions,
			}
			if err := m.Update(tt.args.ch); (err != nil) != tt.wantErr {
				t.Errorf("Update() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}
