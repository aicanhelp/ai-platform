//go:build linux

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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/stretchr/testify/assert"
)

func TestGetOnlineNumaNodes_Success(t *testing.T) {
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if path == "/sys/devices/system/node/online" {
			return []byte("0-1"), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches.Reset()

	nodes, count, err := getOnlineNumaNodes()
	assert.NoError(t, err)
	assert.Equal(t, 2, count)
	assert.Equal(t, []int{0, 1}, nodes)
}

func TestGetOnlineNumaNodes_ReadError(t *testing.T) {
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		return nil, fmt.Errorf("file not found")
	})
	defer patches.Reset()

	_, _, err := getOnlineNumaNodes()
	assert.Error(t, err)
}

func TestGetOnlineNumaNodes_ParseError(t *testing.T) {
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		return []byte("invalid-format-!!!"), nil
	})
	defer patches.Reset()

	_, _, err := getOnlineNumaNodes()
	assert.Error(t, err)
}

func TestGetNodeFreeCPUs_Success(t *testing.T) {
	checkpointData := `{"policyName":"static","defaultCpuSet":"0-7","entries":{},"checksum":0}`
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if path == "/var/lib/kubelet/cpu_manager_state" {
			return []byte(checkpointData), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches.Reset()

	cpus, count, err := getNodeFreeCPUs()
	assert.NoError(t, err)
	assert.Equal(t, 8, count)
	assert.Equal(t, []int{0, 1, 2, 3, 4, 5, 6, 7}, cpus)
}

func TestGetNodeFreeCPUs_PolicyNone(t *testing.T) {
	checkpointData := `{"policyName":"none","defaultCpuSet":"0-7","entries":{},"checksum":0}`
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if path == "/var/lib/kubelet/cpu_manager_state" {
			return []byte(checkpointData), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches.Reset()

	_, _, err := getNodeFreeCPUs()
	assert.Error(t, err)
}

func TestGetNodeFreeCPUs_ReadError(t *testing.T) {
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		return nil, fmt.Errorf("file not found")
	})
	defer patches.Reset()

	_, _, err := getNodeFreeCPUs()
	assert.Error(t, err)
}

func TestGetNodeFreeCPUs_InvalidJSON(t *testing.T) {
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		return []byte("invalid json"), nil
	})
	defer patches.Reset()

	_, _, err := getNodeFreeCPUs()
	assert.Error(t, err)
}

func TestReadAndParseCPUs(t *testing.T) {
	mockData := map[string]string{
		filepath.Join("/sys/devices/system/node/node0/cpulist"): "0-3",
		filepath.Join("/sys/devices/system/node/node1/cpulist"): "4-7",
	}
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if data, ok := mockData[path]; ok {
			return []byte(data), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches.Reset()

	result := readAndParseCPUs([]int{0, 1})
	assert.Equal(t, 2, len(result))
	assert.Equal(t, []int{0, 1, 2, 3}, result[0])
	assert.Equal(t, []int{4, 5, 6, 7}, result[1])
}

func TestGetNumaNodeCPUs(t *testing.T) {
	mockData := map[string]string{
		filepath.Join("/sys/devices/system/node/node0/cpulist"): "0-3",
		filepath.Join("/sys/devices/system/node/node1/cpulist"): "4-7",
	}
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if data, ok := mockData[path]; ok {
			return []byte(data), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches.Reset()

	result := getNumaNodeCPUs([]int{0, 1})
	assert.Equal(t, 2, len(result))
	assert.Equal(t, 4, result[0].numaNodeCPUsCount)
	assert.Equal(t, 4, result[1].numaNodeCPUsCount)
}

func TestGetNumaNodeFreeCPUs_Calculation(t *testing.T) {
	freeCPUs := []int{0, 1, 4}
	numaNodeCPUInfos := map[int]numaNodeCPUInfo{
		0: {numaNodeCPUsCount: 4, numaNodeCPUs: []int{0, 1, 2, 3}},
		1: {numaNodeCPUsCount: 4, numaNodeCPUs: []int{4, 5, 6, 7}},
	}
	result := getNumaNodeFreeCPUs(freeCPUs, numaNodeCPUInfos)
	assert.Equal(t, 2, result[0])
	assert.Equal(t, 1, result[1])
}

func TestCPUNumaCollector_Update(t *testing.T) {
	t.Setenv("NODE_NAME", "test-node")

	checkpointData := `{"policyName":"static","defaultCpuSet":"0-7","entries":{"pod-uuid":{"container1":"0-3"}},"checksum":0}`
	mockData := map[string]string{
		"/sys/devices/system/node/online":                "0-1",
		filepath.Join("/sys/devices/system/node/node0/cpulist"): "0-3",
		filepath.Join("/sys/devices/system/node/node1/cpulist"): "4-7",
		"/var/lib/kubelet/cpu_manager_state":              checkpointData,
	}
	patches := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if data, ok := mockData[path]; ok {
			return []byte(data), nil
		}
		return nil, fmt.Errorf("no path found: %s", path)
	})
	defer patches.Reset()

	collector := &cpuNumaCollector{
		numaNodesCount:              prometheus.NewDesc("test_nodes_count", "test", []string{"node"}, nil),
		numaNodeCPUsCount:           prometheus.NewDesc("test_node_cpus_count", "test", []string{"node", "numa_node"}, nil),
		nodeFreeCPUsCount:           prometheus.NewDesc("test_free_cpus_count", "test", []string{"node"}, nil),
		numaNodeFreeCPUsCount:       prometheus.NewDesc("test_node_free_cpus_count", "test", []string{"node", "numa_node"}, nil),
		containerNumaNodeAssignment: prometheus.NewDesc("test_container_assignment", "test", []string{"node", "pod_uuid", "container", "cpu_numa_node", "cpu_list"}, nil),
	}

	ch := make(chan prometheus.Metric, 100)
	err := collector.Update(ch)
	assert.NoError(t, err)
	close(ch)

	count := 0
	for range ch {
		count++
	}
	assert.Greater(t, count, 0)
}

func TestCPUNumaCollector_Update_NoNodeName(t *testing.T) {
	os.Unsetenv("NODE_NAME")
	collector := &cpuNumaCollector{}
	ch := make(chan prometheus.Metric, 10)
	err := collector.Update(ch)
	assert.Error(t, err)
}

func TestParseMemInfoNuma(t *testing.T) {
	tempDir := t.TempDir()
	memInfoPath := filepath.Join(tempDir, "meminfo")
	memInfoContent := `Node 0 MemTotal:       16384000 kB
Node 0 MemFree:         8192000 kB
`
	err := os.WriteFile(memInfoPath, []byte(memInfoContent), 0644)
	assert.NoError(t, err)

	metrics, err := parseMemInfoNuma("test-node", tempDir)
	assert.NoError(t, err)
	assert.Greater(t, len(metrics), 0)
}

func TestParseMemInfoNuma_FileNotFound(t *testing.T) {
	_, err := parseMemInfoNuma("test-node", "/nonexistent/path")
	assert.Error(t, err)
}

func TestParseMemInfoNuma_InvalidFormat(t *testing.T) {
	tempDir := t.TempDir()
	memInfoPath := filepath.Join(tempDir, "meminfo")
	err := os.WriteFile(memInfoPath, []byte("short"), 0644)
	assert.NoError(t, err)

	_, err = parseMemInfoNuma("test-node", tempDir)
	assert.Error(t, err)
}

func TestGetMemInfoNuma(t *testing.T) {
	tempDir := t.TempDir()
	nodeDir := filepath.Join(tempDir, "node0")
	os.MkdirAll(nodeDir, 0755)
	memInfoContent := "Node 0 MemTotal:       16384000 kB\n"
	os.WriteFile(filepath.Join(nodeDir, "meminfo"), []byte(memInfoContent), 0644)

	patches := gomonkey.ApplyFunc(filepath.Glob, func(pattern string) ([]string, error) {
		if pattern == "/sys/devices/system/node/node[0-9]*" {
			return []string{nodeDir}, nil
		}
		return nil, nil
	})
	defer patches.Reset()

	metrics, err := getMemInfoNuma("test-node")
	assert.NoError(t, err)
	_ = metrics
}

func TestGetMemInfoNuma_GlobError(t *testing.T) {
	patches := gomonkey.ApplyFunc(filepath.Glob, func(pattern string) ([]string, error) {
		return nil, fmt.Errorf("glob error")
	})
	defer patches.Reset()

	_, err := getMemInfoNuma("test-node")
	assert.Error(t, err)
}

func TestGetNumaNodeDistances_Mocked(t *testing.T) {
	patches := gomonkey.ApplyFunc(filepath.Glob, func(pattern string) ([]string, error) {
		return []string{
			"/sys/devices/system/node/node0",
			"/sys/devices/system/node/node1",
		}, nil
	})
	defer patches.Reset()

	patches2 := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if path == "/sys/devices/system/node/node0/distance" {
			return []byte("10 20"), nil
		}
		if path == "/sys/devices/system/node/node1/distance" {
			return []byte("20 10"), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches2.Reset()

	distances, err := getNumaNodeDistances()
	assert.NoError(t, err)
	assert.Equal(t, 2, len(distances))
	assert.Equal(t, 10, distances[0][0])
	assert.Equal(t, 20, distances[0][1])
}

func TestGetNumaNodeDistances_GlobError_Mocked(t *testing.T) {
	patches := gomonkey.ApplyFunc(filepath.Glob, func(pattern string) ([]string, error) {
		return nil, fmt.Errorf("glob error")
	})
	defer patches.Reset()

	_, err := getNumaNodeDistances()
	assert.Error(t, err)
}

func TestDistanceNumaCollector_Update(t *testing.T) {
	t.Setenv("NODE_NAME", "test-node")

	patches := gomonkey.ApplyFunc(filepath.Glob, func(pattern string) ([]string, error) {
		return []string{"/sys/devices/system/node/node0"}, nil
	})
	defer patches.Reset()

	patches2 := gomonkey.ApplyFunc(ioutil.ReadFile, func(path string) ([]byte, error) {
		if path == "/sys/devices/system/node/node0/distance" {
			return []byte("10"), nil
		}
		return nil, fmt.Errorf("no path found")
	})
	defer patches2.Reset()

	collector := &distanceNumaCollector{
		numaNodeDistance: prometheus.NewDesc("test_distance", "test", []string{"node", "source_numa_node", "target_numa_node"}, nil),
	}
	ch := make(chan prometheus.Metric, 10)
	err := collector.Update(ch)
	assert.NoError(t, err)
}

func TestDistanceNumaCollector_Update_NoNodeName(t *testing.T) {
	os.Unsetenv("NODE_NAME")
	collector := &distanceNumaCollector{}
	ch := make(chan prometheus.Metric, 10)
	err := collector.Update(ch)
	assert.Error(t, err)
}
