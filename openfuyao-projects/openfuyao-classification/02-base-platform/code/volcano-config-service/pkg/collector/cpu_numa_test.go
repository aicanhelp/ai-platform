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
	"sort"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
)

func TestNewCPUNumaCollector(t *testing.T) {
	tests := []struct {
		name    string
		wantErr bool
	}{
		{
			name:    "Successful creation of cpuNumaCollector",
			wantErr: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := NewCPUNumaCollector()

			if (err != nil) != tt.wantErr {
				t.Errorf("NewCPUNumaCollector() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if !tt.wantErr {

				cpuNumaCollectorInstance, ok := got.(*cpuNumaCollector)
				if !ok {
					t.Errorf("NewCPUNumaCollector() got = %v, want type *cpuNumaCollector", reflect.TypeOf(got))
					return
				}

				expectedNumaNodesCountDesc := prometheus.NewDesc(
					prometheus.BuildFQName("numa", "nodes", "count"),
					"Online numa nodes count.",
					[]string{"node"}, nil,
				)
				if !reflect.DeepEqual(cpuNumaCollectorInstance.numaNodesCount, expectedNumaNodesCountDesc) {
					t.Errorf("numaNodesCount descriptor mismatch: got = %v, want = %v",
						cpuNumaCollectorInstance.numaNodesCount, expectedNumaNodesCountDesc)
				}

				expectedNumaNodeCPUsCountDesc := prometheus.NewDesc(
					prometheus.BuildFQName("numa", "node", "cpus_count"),
					"Total number of CPUs in each NUMA node.",
					[]string{"node", "numa_node"}, nil,
				)
				if !reflect.DeepEqual(cpuNumaCollectorInstance.numaNodeCPUsCount, expectedNumaNodeCPUsCountDesc) {
					t.Errorf("numaNodeCPUsCount descriptor mismatch: got = %v, want = %v",
						cpuNumaCollectorInstance.numaNodeCPUsCount, expectedNumaNodeCPUsCountDesc)
				}

				expectedNodeFreeCPUsCountDesc := prometheus.NewDesc(
					prometheus.BuildFQName("node", "free_cpus", "count"),
					"Number of free CPUs in the node.",
					[]string{"node"}, nil,
				)
				if !reflect.DeepEqual(cpuNumaCollectorInstance.nodeFreeCPUsCount, expectedNodeFreeCPUsCountDesc) {
					t.Errorf("nodeFreeCPUsCount descriptor mismatch: got = %v, want = %v",
						cpuNumaCollectorInstance.nodeFreeCPUsCount, expectedNodeFreeCPUsCountDesc)
				}

				expectedNumaNodeFreeCPUsCountDesc := prometheus.NewDesc(
					prometheus.BuildFQName("numa", "node", "free_cpus_count"),
					"Total number of free CPUs in each NUMA node.",
					[]string{"node", "numa_node"}, nil,
				)
				if !reflect.DeepEqual(cpuNumaCollectorInstance.numaNodeFreeCPUsCount,
					expectedNumaNodeFreeCPUsCountDesc) {
					t.Errorf("numaNodeFreeCPUsCount descriptor mismatch: got = %v, want = %v",
						cpuNumaCollectorInstance.numaNodeFreeCPUsCount, expectedNumaNodeFreeCPUsCountDesc)
				}

				expectedContainerNumaNodeAssignmentDesc := prometheus.NewDesc(
					prometheus.BuildFQName("container_numa", "node", "assignment"),
					"Total number of free CPUs in each NUMA node.",
					[]string{"node", "pod_uuid", "container", "cpu_numa_node", "cpu_list"}, nil,
				)
				if !reflect.DeepEqual(cpuNumaCollectorInstance.containerNumaNodeAssignment,
					expectedContainerNumaNodeAssignmentDesc) {
					t.Errorf("containerNumaNodeAssignment descriptor mismatch: got = %v, want = %v",
						cpuNumaCollectorInstance.containerNumaNodeAssignment, expectedContainerNumaNodeAssignmentDesc)
				}
			}
		})
	}
}

func TestGetNumaNodeFreeCPUs(t *testing.T) {
	type args struct {
		freeCPUs         []int
		numaNodeCPUInfos map[int]numaNodeCPUInfo
	}
	tests := []struct {
		name string
		args args
		want map[int]int
	}{
		{
			name: "successful with multiple free CPUs across multiple NUMA nodes",
			args: args{
				freeCPUs: []int{1, 2, 3, 7, 8},
				numaNodeCPUInfos: map[int]numaNodeCPUInfo{
					0: {numaNodeCPUsCount: 4, numaNodeCPUs: []int{0, 1, 2, 3}},
					1: {numaNodeCPUsCount: 5, numaNodeCPUs: []int{4, 5, 6, 7, 8}},
				},
			},
			want: map[int]int{
				0: 3,
				1: 2,
			},
		},
		{
			name: "no free CPUs",
			args: args{
				freeCPUs: []int{},
				numaNodeCPUInfos: map[int]numaNodeCPUInfo{
					0: {numaNodeCPUsCount: 2, numaNodeCPUs: []int{2, 3}},
					1: {numaNodeCPUsCount: 3, numaNodeCPUs: []int{4, 5, 6}},
				},
			},
			want: map[int]int{
				0: 0,
				1: 0,
			},
		},
		{
			name: "NUMA node with no free CPUs",
			args: args{
				freeCPUs: []int{9, 10},
				numaNodeCPUInfos: map[int]numaNodeCPUInfo{
					0: {numaNodeCPUsCount: 4, numaNodeCPUs: []int{0, 1, 2, 3}},
					1: {numaNodeCPUsCount: 5, numaNodeCPUs: []int{4, 5, 6, 7, 8}},
				},
			},
			want: map[int]int{
				0: 0,
				1: 0,
			},
		},
		{
			name: "NUMA node with all CPUs free",
			args: args{
				freeCPUs: []int{1, 2, 3, 4, 5},
				numaNodeCPUInfos: map[int]numaNodeCPUInfo{
					0: {numaNodeCPUsCount: 2, numaNodeCPUs: []int{1, 2}},
					1: {numaNodeCPUsCount: 3, numaNodeCPUs: []int{3, 4, 5}},
				},
			},
			want: map[int]int{
				0: 2,
				1: 3,
			},
		},
		{
			name: "empty NUMA node list",
			args: args{
				freeCPUs:         []int{1, 2, 3},
				numaNodeCPUInfos: map[int]numaNodeCPUInfo{},
			},
			want: map[int]int{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getNumaNodeFreeCPUs(tt.args.freeCPUs, tt.args.numaNodeCPUInfos); !reflect.DeepEqual(got,
				tt.want) {
				t.Errorf("getNumaNodeFreeCPUs() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDetermineNumaNodes(t *testing.T) {
	type args struct {
		cpuList     []int
		cpuNumaNode map[int]int
	}
	tests := []struct {
		name string
		args args
		want []int
	}{
		{
			name: "success to get NumaNodeList",
			args: args{
				cpuList:     []int{0, 1, 2, 3},
				cpuNumaNode: map[int]int{0: 1, 1: 2, 2: 4, 3: 3},
			},
			want: []int{1, 2, 4, 3},
		},
		{
			name: "fail to get NumaNodeList",
			args: args{
				cpuList:     []int{0, 1, 2},
				cpuNumaNode: map[int]int{},
			},
			want: []int{},
		},
		{
			name: "empty cpuList",
			args: args{
				cpuList:     []int{},
				cpuNumaNode: map[int]int{0: 1, 1: 2, 2: 4},
			},
			want: []int{},
		},
		{
			name: "cpuList with th same cpuNumaNode",
			args: args{
				cpuList:     []int{0, 1, 2},
				cpuNumaNode: map[int]int{0: 1, 1: 1, 2: 1},
			},
			want: []int{1},
		},
		{
			name: "some cpus not found in cpuNumaNode",
			args: args{
				cpuList:     []int{0, 1, 2, 5, 50},
				cpuNumaNode: map[int]int{0: 1, 1: 3, 2: 2, 3: 4},
			},
			want: []int{1, 3, 2},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := determineNumaNodes(tt.args.cpuList, tt.args.cpuNumaNode)
			sort.Ints(got)
			sort.Ints(tt.want)
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("determineNumaNodes() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetCPUNumaNode(t *testing.T) {
	type args struct {
		numaNodes []int
	}
	tests := []struct {
		name     string
		args     args
		mockData map[string]string
		want     map[int]int
		mockErr  error
	}{
		{
			name: "success",
			args: args{
				numaNodes: []int{0, 1},
			},
			mockData: map[string]string{
				filepath.Join("/sys/devices/system/node/node0/cpulist"): "3",
				filepath.Join("/sys/devices/system/node/node1/cpulist"): "4",
			},
			want:    map[int]int{3: 0, 4: 1},
			mockErr: nil,
		},
		{
			name: "fail",
			args: args{
				numaNodes: []int{0, 1},
			},
			mockData: map[string]string{
				filepath.Join("/sys/devices/system/node/node1/cpulist"): "3",
				filepath.Join("/sys/devices/system/node/node2/cpulist"): "4",
			},
			want:    map[int]int{3: 1},
			mockErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				if data, get := tt.mockData[path]; get {
					return []byte(data), nil
				}
				return nil, fmt.Errorf("no path found")
			})
			defer patches.Reset()
			if got := getCPUNumaNode(tt.args.numaNodes); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getCPUNumaNode() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetContainerNumaNodeAssignment(t *testing.T) {
	type args struct {
		numaNodes []int
	}
	tests := []struct {
		name          string
		args          args
		mockcpuData   string
		mockcpu       map[string]string
		want          map[string]map[string]containerNumaInfo
		mockErr       error
		wantErr       bool
		cpumanagerErr error
	}{
		{
			name: "fail cpu manager state",
			args: args{
				numaNodes: []int{0, 1},
			},
			mockcpuData:   `{}`,
			mockcpu:       map[string]string{},
			want:          nil,
			mockErr:       nil,
			wantErr:       true,
			cpumanagerErr: errors.New("cpumanager err"),
		},
		{
			name: "fail cpu checkpoint",
			args: args{
				numaNodes: []int{0, 1},
			},
			mockcpuData:   `invalid-json`,
			mockcpu:       map[string]string{},
			want:          nil,
			mockErr:       nil,
			wantErr:       true,
			cpumanagerErr: nil,
		},
		{
			name: "policy is none",
			args: args{
				numaNodes: []int{0, 1},
			},
			mockcpuData: `{"policyName":"none","defaultCpuSet":"0,2-3",
									"entries":{"a09db296-2ca6-44dd-8f5e-dff48865fa52":{"container-1":"1"}},
										"checksum":347313536}`,
			mockcpu:       map[string]string{},
			want:          nil,
			mockErr:       nil,
			wantErr:       true,
			cpumanagerErr: nil,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
				if tt.mockErr != nil {
					return nil, tt.mockErr
				}
				if path == "/var/lib/kubelet/cpu_manager_state" {
					if tt.cpumanagerErr != nil {
						return nil, tt.cpumanagerErr
					}
					return []byte(tt.mockcpuData), nil
				} else {
					if data, ok := tt.mockcpu[path]; ok {
						return []byte(data), nil
					}
				}
				return nil, fmt.Errorf("no path")
			})

			defer patches.Reset()
			got, err := getContainerNumaNodeAssignment(tt.args.numaNodes)
			if (err != nil) != tt.wantErr {
				t.Errorf("getContainerNumaNodeAssignment() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getContainerNumaNodeAssignment() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestCpuNumaCollectorUpdate(t *testing.T) {
	type fields struct {
		numaNodesCount              *prometheus.Desc
		numaNodeCPUsCount           *prometheus.Desc
		nodeFreeCPUsCount           *prometheus.Desc
		numaNodeFreeCPUsCount       *prometheus.Desc
		containerNumaNodeAssignment *prometheus.Desc
	}
	type args struct {
		ch chan<- prometheus.Metric
	}
	tests := []struct {
		name             string
		fields           fields
		args             args
		wantErr          bool
		mockNode         string
		mockOnlineNuma   []int
		mockNodeCpu      map[int]numaNodeCPUInfo
		mockFreeCpu      []int
		mockNumaCpu      map[int]int
		mockContainer    map[string]map[string]containerNumaInfo
		mockNodeErr      error
		mockOnlineErr    error
		mockNodeCpuErr   error
		mockFreeErr      error
		mockNumaErr      error
		mockContainerErr error
	}{
		{name: "success case",
			fields: fields{
				nodeFreeCPUsCount: prometheus.NewDesc("node_free_cpus_count",
					"Number of free CPUs", []string{"node"}, nil),
				numaNodeFreeCPUsCount: prometheus.NewDesc("numa_node_free_cpus_count",
					"Number of free CPUs in NUMA node", []string{"node", "numa_node"}, nil),
				numaNodesCount: prometheus.NewDesc("numa_nodes_count",
					"Number of NUMA nodes", []string{"node"}, nil),
				containerNumaNodeAssignment: prometheus.NewDesc("container_numa_node_assignment",
					"NUMA node assignment of container",
					[]string{"node", "pod_uuid", "container_name", "cpu_numa_nodes", "cpu_list"},
					nil),
				numaNodeCPUsCount: prometheus.NewDesc("numa_node_cpus_count",
					"Number of CPUs in NUMA node", []string{"node", "numa_node"}, nil),
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
			wantErr:        false,
			mockNode:       "node0",
			mockOnlineNuma: []int{0},
			mockNodeCpu: map[int]numaNodeCPUInfo{
				0: {numaNodeCPUsCount: 1, numaNodeCPUs: []int{0}},
			},
			mockFreeCpu: []int{0},
			mockNumaCpu: map[int]int{
				0: 0,
			},
			mockContainer: map[string]map[string]containerNumaInfo{
				"pod-uuid": {
					"container-1": {
						numaNodes:    []int{0},
						numaNodeCPUs: []int{0},
					},
				},
			},
			mockNodeErr:      nil,
			mockOnlineErr:    nil,
			mockNodeCpuErr:   nil,
			mockFreeErr:      nil,
			mockNumaErr:      nil,
			mockContainerErr: nil,
		},
		{
			name: "fail to get node name",
			fields: fields{
				numaNodesCount: prometheus.NewDesc("numa_nodes_count",
					"Number of NUMA nodes", []string{"node"}, nil),
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
			wantErr:     true,
			mockNodeErr: errors.New("failed to get node name"),
		},
		{
			name: "fail to get online NUMA nodes",
			fields: fields{
				numaNodesCount: prometheus.NewDesc("numa_nodes_count",
					"Number of NUMA nodes", []string{"node"}, nil),
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
			wantErr:       true,
			mockOnlineErr: errors.New("failed to get online NUMA nodes"),
		},
		{
			name: "fail to get free CPUs",
			fields: fields{
				numaNodesCount: prometheus.NewDesc("numa_nodes_count",
					"Number of NUMA nodes", []string{"node"}, nil),
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
			wantErr:     true,
			mockFreeErr: errors.New("failed to get free CPUs"),
		},
		{
			name: "fail to get container",
			fields: fields{
				numaNodesCount: prometheus.NewDesc("numa_nodes_count",
					"Number of NUMA nodes", []string{"node"}, nil),
				numaNodeCPUsCount: prometheus.NewDesc("numa_node_cpus_count",
					"Number of CPUs in NUMA node", []string{"node", "numa_node"}, nil),
				nodeFreeCPUsCount: prometheus.NewDesc("node_free_cpus_count",
					"Number of free CPUs", []string{"node"}, nil),
				numaNodeFreeCPUsCount: prometheus.NewDesc("numa_node_free_cpus_count",
					"Number of free CPUs in NUMA node", []string{"node", "numa_node"}, nil),
				containerNumaNodeAssignment: prometheus.NewDesc("container_numa_node_assignment",
					"NUMA node assignment of container",
					[]string{"node", "pod_uuid", "container_name", "cpu_numa_nodes", "cpu_list"},
					nil),
			},
			args: args{
				ch: make(chan prometheus.Metric, 10),
			},
			wantErr:          true,
			mockNode:         "node0",
			mockOnlineNuma:   []int{0},
			mockNodeCpu:      map[int]numaNodeCPUInfo{0: {numaNodeCPUsCount: 1, numaNodeCPUs: []int{0}}},
			mockFreeCpu:      []int{0},
			mockNumaCpu:      map[int]int{0: 0},
			mockContainer:    nil,
			mockNodeErr:      nil,
			mockOnlineErr:    nil,
			mockNodeCpuErr:   nil,
			mockFreeErr:      nil,
			mockNumaErr:      nil,
			mockContainerErr: errors.New("failed to get container numa node assignment"), // 返回错误
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gomonkey.ApplyFunc(utils.GetNodeName, func() (string, error) {
				if tt.mockNodeErr != nil {
					return "", tt.mockNodeErr
				}
				return tt.mockNode, nil
			})

			// Patch getOnlineNumaNodes
			gomonkey.ApplyFunc(getOnlineNumaNodes, func() ([]int, int, error) {
				if tt.mockOnlineErr != nil {
					return nil, 0, tt.mockOnlineErr
				}
				return tt.mockOnlineNuma, len(tt.mockOnlineNuma), nil
			})

			// Patch getNumaNodeCPUs
			gomonkey.ApplyFunc(getNumaNodeCPUs, func(numaNodes []int) map[int]numaNodeCPUInfo {
				if tt.mockNodeCpuErr != nil {
					return nil
				}
				return tt.mockNodeCpu
			})

			// Patch getNodeFreeCPUs
			gomonkey.ApplyFunc(getNodeFreeCPUs, func() ([]int, int, error) {
				if tt.mockFreeErr != nil {
					return nil, 0, tt.mockFreeErr
				}
				return tt.mockFreeCpu, len(tt.mockFreeCpu), nil
			})

			// Patch getNumaNodeFreeCPUs
			gomonkey.ApplyFunc(getNumaNodeFreeCPUs, func(freeCPUs []int,
				numaNodeCPUInfos map[int]numaNodeCPUInfo) map[int]int {
				if tt.mockNumaErr != nil {
					return nil
				}
				return tt.mockNumaCpu
			})

			// Patch getContainerNumaNodeAssignment
			gomonkey.ApplyFunc(getContainerNumaNodeAssignment,
				func(numaNodes []int) (map[string]map[string]containerNumaInfo,
					error) {
					if tt.mockContainerErr != nil {
						return nil, tt.mockContainerErr
					}
					return tt.mockContainer, nil
				})
			c := cpuNumaCollector{
				numaNodesCount:              tt.fields.numaNodesCount,
				numaNodeCPUsCount:           tt.fields.numaNodeCPUsCount,
				nodeFreeCPUsCount:           tt.fields.nodeFreeCPUsCount,
				numaNodeFreeCPUsCount:       tt.fields.numaNodeFreeCPUsCount,
				containerNumaNodeAssignment: tt.fields.containerNumaNodeAssignment,
			}
			if err := c.Update(tt.args.ch); (err != nil) != tt.wantErr {
				t.Errorf("Update() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func runTestWithMockData(t *testing.T, tt struct {
	name     string
	mockData string
	want     []int
	want1    int
	wantErr  bool
	mockErr  error
	errMsg   error
}, testFunc func() ([]int, int, error)) {
	t.Run(tt.name, func(t *testing.T) {
		if tt.wantErr && tt.mockErr == nil && tt.errMsg == nil {
			t.Errorf("Invalid test case: wantErr is true, but both mockError and errMsg are nil")
			return
		}

		patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
			if tt.mockErr != nil {
				return nil, tt.mockErr
			}
			return []byte(tt.mockData), nil
		})
		defer patches.Reset()

		got, got1, err := testFunc()

		if (err != nil) != tt.wantErr {
			t.Errorf("%s error = %v, wantErr = %v", tt.name, err, tt.wantErr)
			return
		}

		if err != nil && tt.wantErr {
			if tt.mockErr != nil && !strings.Contains(err.Error(), tt.mockErr.Error()) {
				t.Errorf("%s error = %v, expected error = %v", tt.name, err, tt.mockErr)
			}
			if tt.errMsg != nil && !strings.Contains(err.Error(), tt.errMsg.Error()) {
				t.Errorf("%s error = %v, expected error = %v", tt.name, err, tt.errMsg)
			}
			return
		}

		if !reflect.DeepEqual(got, tt.want) {
			t.Errorf("%s got = %v, want = %v", tt.name, got, tt.want)
		}

		if got1 != tt.want1 {
			t.Errorf("%s got1 = %v, want1 = %v", tt.name, got1, tt.want1)
		}
	})
}
