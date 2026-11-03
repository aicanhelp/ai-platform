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
	"fmt"
	"io/ioutil"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
)

func TestNewDistanceNumaCollector(t *testing.T) {
	tests := []struct {
		name    string
		want    *distanceNumaCollector
		wantErr bool
	}{
		{
			name: "success to creates a new distanceNumaCollector instance",
			want: &distanceNumaCollector{
				numaNodeDistance: prometheus.NewDesc(
					prometheus.BuildFQName("numa", "node", "distance"),
					"NUMA nodes access latency.",
					[]string{"node", "source_numa_node", "target_numa_node"}, nil,
				),
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewDistanceNumaCollector()
			if (err != nil) != tt.wantErr {
				t.Errorf("NewDistanceNumaCollector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			gotCollector, ok := got.(*distanceNumaCollector)
			if !ok {
				t.Errorf("NewDistanceNumaCollector() got is not of type *distanceNumaCollector")
				return
			}
			if gotCollector.numaNodeDistance.String() != tt.want.numaNodeDistance.String() {
				t.Errorf("A %v, %v", gotCollector.numaNodeDistance.String(), tt.want.numaNodeDistance.String())
			}

		})
	}
}

func TestDistanceNumaCollectorUpdate(t *testing.T) {
	type fields struct {
		numaNodeDistance *prometheus.Desc
	}
	tests := []struct {
		name          string
		fields        fields
		mockNodeName  string
		mockDistances map[int]map[int]int
		wantMetrics   int
		wantErr       bool
	}{
		{
			name: "Valid NUMA data",
			fields: fields{
				numaNodeDistance: prometheus.NewDesc("numa_node_distance",
					"Distance between NUMA nodes", []string{"node", "source_numa_node", "target_numa_node"},
					nil),
			},
			mockNodeName: "test-node",
			mockDistances: map[int]map[int]int{
				0: {1: 10, 2: 20},
				1: {0: 10, 2: 15},
			},
			wantMetrics: 4,
			wantErr:     false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			metricsCh := make(chan prometheus.Metric, 10)

			sendOnlyCh := (chan<- prometheus.Metric)(metricsCh)

			patchGetNodeName := gomonkey.ApplyFunc(utils.GetNodeName, func() (string, error) {
				return tt.mockNodeName, nil
			})
			defer patchGetNodeName.Reset()

			patchGetNumaNodeDistances := gomonkey.ApplyFunc(getNumaNodeDistances, func() (map[int]map[int]int, error) {
				if tt.mockDistances == nil {
					return nil, fmt.Errorf("error fetching NUMA distances")
				}
				return tt.mockDistances, nil
			})
			defer patchGetNumaNodeDistances.Reset()

			d := distanceNumaCollector{
				numaNodeDistance: tt.fields.numaNodeDistance,
			}

			err := d.Update(sendOnlyCh)
			if (err != nil) != tt.wantErr {
				t.Errorf("Update() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			close(metricsCh)
			metricsCount := 0
			for range metricsCh {
				metricsCount++
			}

			if metricsCount != tt.wantMetrics {
				t.Errorf("Update() sent %d metrics, want %d", metricsCount, tt.wantMetrics)
			}
		})
	}
}

func TestParseNumaNodeIDs(t *testing.T) {
	type args struct {
		numaNodePaths []string
	}
	tests := []struct {
		name    string
		args    args
		want    []int
		wantErr bool
	}{
		{
			name: "success to parse NumaNodeID",
			args: args{
				numaNodePaths: []string{"/sys/devices/system/node/node1", "/sys/devices/system/node/node2"},
			},
			want:    []int{1, 2},
			wantErr: false,
		},
		{
			name: "Invalid NumaNodeID",
			args: args{
				numaNodePaths: []string{"/sys/devices/system/node/nodeB"},
			},
			want:    nil,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseNumaNodeIDs(tt.args.numaNodePaths)
			if (err != nil) != tt.wantErr {
				t.Errorf("parseNumaNodeIDs() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("parseNumaNodeIDs() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetNumaNodeDistances(t *testing.T) {
	tests := []struct {
		name        string
		mockpaths   []string
		mockdata    map[string]string
		want        map[int]map[int]int
		wantErr     bool
		mockdataErr error
		mockpathErr error
	}{
		{
			name:      "success",
			mockpaths: []string{"/sys/devices/system/node/node0"},
			mockdata: map[string]string{
				"/sys/devices/system/node/node0/distance": "10",
			},
			want: map[int]map[int]int{
				0: {0: 10},
			},
			wantErr:     false,
			mockdataErr: nil,
			mockpathErr: nil,
		},
		{
			name:        "fail to glob",
			mockpaths:   []string{},
			mockdata:    map[string]string{},
			want:        nil,
			wantErr:     true,
			mockdataErr: nil,
			mockpathErr: errors.New("glob error"),
		},
		{
			name:        "fail to parse",
			mockpaths:   []string{"invalid/path"},
			mockdata:    map[string]string{},
			want:        nil,
			wantErr:     true,
			mockdataErr: nil,
			mockpathErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pathmorck := gomonkey.ApplyFunc(filepath.Glob, func(path string) ([]string, error) {
				if tt.mockpathErr != nil {
					return nil, tt.mockpathErr
				}
				return tt.mockpaths, nil
			})
			datamock := gomonkey.ApplyFunc(ioutil.ReadFile, func(filename string) ([]byte, error) {
				if tt.mockdataErr != nil {
					return nil, tt.mockdataErr
				}
				if data, exists := tt.mockdata[filename]; exists {
					return []byte(data), nil
				}
				return nil, fmt.Errorf("file not found: %s", filename)
			})
			defer pathmorck.Reset()
			defer datamock.Reset()
			got, err := getNumaNodeDistances()
			if (err != nil) != tt.wantErr {
				t.Errorf("getNumaNodeDistances() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getNumaNodeDistances() got = %v, want %v", got, tt.want)
			}
		})
	}
}
