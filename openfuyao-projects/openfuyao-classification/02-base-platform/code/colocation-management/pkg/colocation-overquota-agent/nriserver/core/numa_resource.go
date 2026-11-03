/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 * Author: Junlin Liu
 * Date: 2025-05-20
 */

package core

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync"

	"k8s.io/api/core/v1"
)

// NumaResourceSpec 代表一个NUMA节点的资源规格，包含CPU核数和该节点上的CPU列表
type NumaResourceSpec struct {
	// CPUset CPUset
	CPUset string
	// NUMANodeResourcesMap  NUMA节点上的CPU资源规格，键为节点ID，值为该节点上的CPU资源规格
	NUMANodeResourcesMap map[int]v1.ResourceList `json:"NUMANodeResourcesMap,omitempty"`
}

const (
	// CPURangePartsCount CPU范围格式 "start-end" 分割后的期望部分数量
	CPURangePartsCount = 2
)

// NumaNode 代表一个NUMA节点，包含节点ID和该节点上的CPU列表
type NumaNode struct {
	ID      int
	CPUList []int
}

// NumaTopology 代表系统的NUMA拓扑信息，包含多个NUMA节点
type NumaTopology struct {
	// NUMA节点列表，键为节点ID
	Nodes map[int]*NumaNode
	mutex sync.RWMutex
}

// GetNumaTopology 获取系统的NUMA拓扑信息
func GetNumaTopology() (*NumaTopology, error) {
	topology := &NumaTopology{
		Nodes: make(map[int]*NumaNode),
	}

	// 读取系统NUMA信息目录
	numaDir := "/sys/devices/system/node"
	nodePattern := filepath.Join(numaDir, "node*")

	nodeDirs, err := filepath.Glob(nodePattern)
	if err != nil {
		return nil, fmt.Errorf("unable to list numa nodes: %v", err)
	}

	for _, nodeDir := range nodeDirs {
		nodeID, err := strconv.Atoi(strings.TrimPrefix(filepath.Base(nodeDir), "node"))
		if err != nil {
			return nil, fmt.Errorf("unable to parse numa node ID: %v", err)
		}

		cpulistFile := filepath.Join(nodeDir, "cpulist")
		cpulistData, err := os.ReadFile(cpulistFile)
		if err != nil {
			return nil, fmt.Errorf("unable to read cpulist for node %d: %v", nodeID, err)
		}

		cpuList, err := parseCPUList(string(cpulistData))
		if err != nil {
			return nil, fmt.Errorf("unable to parse cpulist for node %d: %v", nodeID, err)
		}

		topology.Nodes[nodeID] = &NumaNode{
			ID:      nodeID,
			CPUList: cpuList,
		}
	}

	return topology, nil
}

func parseCPUList(cpulist string) ([]int, error) {
	var result []int

	cpulist = strings.TrimSpace(cpulist)
	if cpulist == "" {
		return []int{}, nil
	}

	parts := strings.Split(cpulist, ",")
	for _, part := range parts {
		if strings.Contains(part, "-") {
			rangeParts := strings.Split(part, "-")
			if len(rangeParts) != CPURangePartsCount {
				return nil, fmt.Errorf("invalid CPU range: %s", part)
			}

			start, err := strconv.Atoi(rangeParts[0])
			if err != nil {
				return nil, fmt.Errorf("invalid CPU range start: %s", rangeParts[0])
			}

			end, err := strconv.Atoi(rangeParts[1])
			if err != nil {
				return nil, fmt.Errorf("invalid CPU range end: %s", rangeParts[1])
			}

			for i := start; i <= end; i++ {
				result = append(result, i)
			}
		} else {
			cpu, err := strconv.Atoi(part)
			if err != nil {
				return nil, fmt.Errorf("invalid CPU: %s", part)
			}
			result = append(result, cpu)
		}
	}

	return result, nil
}

// GetNumaNodeForCPU 返回包含指定CPU的NUMA节点ID
func (t *NumaTopology) GetNumaNodeForCPU(cpu int) (int, bool) {
	for nodeID, node := range t.Nodes {
		for _, nodeCPU := range node.CPUList {
			if nodeCPU == cpu {
				return nodeID, true
			}
		}
	}
	return -1, false
}

// GetNumaNodeCPUs 返回指定NUMA节点ID的CPU列表
func (t *NumaTopology) GetNumaNodeCPUs(nodeID int) ([]int, error) {
	t.mutex.RLock()
	defer t.mutex.RUnlock()

	node, exists := t.Nodes[nodeID]
	if !exists {
		return nil, fmt.Errorf("NUMA node %d does not exist", nodeID)
	}

	result := make([]int, len(node.CPUList))
	copy(result, node.CPUList)
	return result, nil
}

// GetBestNumaNodeForAffinity 返回具有足够CPU数量的最佳NUMA节点ID
func (t *NumaTopology) GetBestNumaNodeForAffinity(minCPUCount int) (int, error) {
	t.mutex.RLock()
	defer t.mutex.RUnlock()

	var bestNodeID = -1
	var maxAvailable = -1

	for nodeID, node := range t.Nodes {
		// 检查该节点是否有足够的 CPU
		totalCPUs := len(node.CPUList)

		if totalCPUs >= minCPUCount && totalCPUs > maxAvailable {
			maxAvailable = totalCPUs
			bestNodeID = nodeID
		}
	}

	if bestNodeID == -1 {
		return -1, fmt.Errorf("no NUMA node has enough CPUs for affinity, requested minimum %d", minCPUCount)
	}

	return bestNodeID, nil
}

// CPUListToString 将CPU列表转换为字符串格式
func CPUListToString(cpus []int) string {
	if len(cpus) == 0 {
		return ""
	}

	sort.Ints(cpus)

	var result []string
	rangeStart := cpus[0]
	prev := cpus[0]

	for i := 1; i < len(cpus); i++ {
		if cpus[i] != prev+1 {
			if rangeStart == prev {
				// 单个CPU
				result = append(result, strconv.Itoa(rangeStart))
			} else {
				// CPU范围
				result = append(result, fmt.Sprintf("%d-%d", rangeStart, prev))
			}
			rangeStart = cpus[i]
		}
		prev = cpus[i]
	}

	if rangeStart == prev {
		result = append(result, strconv.Itoa(rangeStart))
	} else {
		result = append(result, fmt.Sprintf("%d-%d", rangeStart, prev))
	}

	return strings.Join(result, ",")
}
