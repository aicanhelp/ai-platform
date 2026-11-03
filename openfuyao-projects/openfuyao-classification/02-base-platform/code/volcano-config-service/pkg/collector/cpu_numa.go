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

package collector

import (
	"fmt"
	"io/ioutil"
	"path/filepath"

	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
	"volcano-config-service/pkg/zlog"
)

type cpuNumaCollector struct {
	numaNodesCount              *prometheus.Desc
	numaNodeCPUsCount           *prometheus.Desc
	nodeFreeCPUsCount           *prometheus.Desc
	numaNodeFreeCPUsCount       *prometheus.Desc
	containerNumaNodeAssignment *prometheus.Desc
}

type numaNodeCPUInfo struct {
	numaNodeCPUsCount int
	numaNodeCPUs      []int
}

type containerNumaInfo struct {
	numaNodes      []int
	numaNodeCPUs   []int
	numaNodeMemory []int
}

// NewCPUNumaCollector creates a new cpuNumaCollector instance.
func NewCPUNumaCollector() (MetricCollector, error) {
	return &cpuNumaCollector{
		numaNodesCount: prometheus.NewDesc(
			prometheus.BuildFQName("numa", "nodes", "count"),
			"Online numa nodes count.",
			[]string{"node"}, nil,
		),
		numaNodeCPUsCount: prometheus.NewDesc(
			prometheus.BuildFQName("numa", "node", "cpus_count"),
			"Total number of CPUs in each NUMA node.",
			[]string{"node", "numa_node"}, nil,
		),
		nodeFreeCPUsCount: prometheus.NewDesc(
			prometheus.BuildFQName("node", "free_cpus", "count"),
			"Number of free CPUs in the node.",
			[]string{"node"}, nil,
		),
		numaNodeFreeCPUsCount: prometheus.NewDesc(
			prometheus.BuildFQName("numa", "node", "free_cpus_count"),
			"Total number of free CPUs in each NUMA node.",
			[]string{"node", "numa_node"}, nil,
		),
		containerNumaNodeAssignment: prometheus.NewDesc(
			prometheus.BuildFQName("container_numa", "node", "assignment"),
			"Total number of free CPUs in each NUMA node.",
			[]string{"node", "pod_uuid", "container", "cpu_numa_node",
				"cpu_list"}, nil,
		),
	}, nil
}

// Update collects NUMA-related CPU metrics and sends them to the provided channel.
func (c cpuNumaCollector) Update(ch chan<- prometheus.Metric) error {
	nodeName, err := utils.GetNodeName()
	if err != nil {
		return err
	}

	numaNodes, numaNodesCount, err := getOnlineNumaNodes()
	if err != nil {
		return err
	}
	ch <- prometheus.MustNewConstMetric(c.numaNodesCount, prometheus.GaugeValue, float64(numaNodesCount), nodeName)

	numaNodeCPUInfos := getNumaNodeCPUs(numaNodes)
	for numaNode, numaNodeCPUInfo := range numaNodeCPUInfos {
		ch <- prometheus.MustNewConstMetric(c.numaNodeCPUsCount, prometheus.GaugeValue,
			float64(numaNodeCPUInfo.numaNodeCPUsCount), nodeName,
			fmt.Sprintf("%d", numaNode))
	}

	nodeFreeCPUs, nodeFreeCPUsCount, err := getNodeFreeCPUs()
	if err != nil {
		return err
	}
	ch <- prometheus.MustNewConstMetric(c.nodeFreeCPUsCount, prometheus.GaugeValue, float64(nodeFreeCPUsCount), nodeName)

	numaNodeFreeCPUsMap := getNumaNodeFreeCPUs(nodeFreeCPUs, numaNodeCPUInfos)
	for numaNode, freeCPUsCount := range numaNodeFreeCPUsMap {
		ch <- prometheus.MustNewConstMetric(c.numaNodeFreeCPUsCount, prometheus.GaugeValue,
			float64(freeCPUsCount), nodeName,
			fmt.Sprintf("%d", numaNode))
	}

	containerNumaNodeAssignments, err := getContainerNumaNodeAssignment(numaNodes)
	if err != nil {
		return err
	}
	for podUUID, containers := range containerNumaNodeAssignments {
		for containerName, containerInfo := range containers {
			cpuNumaNodes := utils.IntSliceToFormattedString(containerInfo.numaNodes)
			cpuList := utils.IntSliceToFormattedString(containerInfo.numaNodeCPUs)

			ch <- prometheus.MustNewConstMetric(c.containerNumaNodeAssignment, prometheus.GaugeValue,
				1, nodeName, podUUID, containerName, cpuNumaNodes, cpuList)
		}
	}

	return nil

}

func getOnlineNumaNodes() ([]int, int, error) {
	data, err := ioutil.ReadFile("/sys/devices/system/node/online")
	if err != nil {
		return []int{}, 0, fmt.Errorf("read numa online file failed, err: %v", err)
	}

	nodeList, err := utils.ParseRanges(string(data))
	if err != nil {
		return []int{}, 0, fmt.Errorf("parse numa online file failed, err: %v", err)
	}

	return nodeList, len(nodeList), nil
}

func getNodeFreeCPUs() ([]int, int, error) {
	data, err := ioutil.ReadFile("/var/lib/kubelet/cpu_manager_state")
	if err != nil {
		return []int{}, 0, fmt.Errorf("read cpu_manager_state file failed, err: %v", err)
	}

	checkpoint := utils.NewCPUManagerCheckpoint()
	if err := utils.LoadCheckpoint(data, checkpoint); err != nil {
		return []int{}, 0, fmt.Errorf("unmarshal cpu_manager_state failed, err: %v", err)
	}

	if checkpoint.PolicyName == "none" {
		return []int{}, 0, fmt.Errorf("cpu_manager_state policy is 'none'")
	}

	cpuList, err := utils.ParseRanges(checkpoint.DefaultCPUSet)
	if err != nil {
		return []int{}, 0, fmt.Errorf("parse cpu_manager_state failed, err: %v", err)
	}

	return cpuList, len(cpuList), nil
}

func readAndParseCPUs(numaNodes []int) map[int][]int {
	numaNodeCPUs := make(map[int][]int)
	for _, numaNode := range numaNodes {
		cpuPath := filepath.Join("/sys/devices/system/node/", fmt.Sprintf("node%d", numaNode),
			"cpulist")
		data, err := ioutil.ReadFile(cpuPath)
		if err != nil {
			zlog.FormatWarn("Read cpulist file failed. CPU Path: %s, Error: %v", cpuPath, err)
			continue
		}

		cpuList, err := utils.ParseRanges(string(data))
		if err != nil {
			zlog.FormatWarn("Parse cpulist file failed. CPU Path: %s, Error: %v", cpuPath, err)
			continue
		}
		numaNodeCPUs[numaNode] = cpuList
	}
	return numaNodeCPUs
}

func getNumaNodeCPUs(numaNodes []int) map[int]numaNodeCPUInfo {
	numaNodeCPUs := readAndParseCPUs(numaNodes)

	numaNodeCPUInfos := make(map[int]numaNodeCPUInfo)
	for numaNode, cpuList := range numaNodeCPUs {
		numaNodeCPUInfos[numaNode] = numaNodeCPUInfo{
			numaNodeCPUsCount: len(cpuList),
			numaNodeCPUs:      cpuList,
		}
	}
	return numaNodeCPUInfos
}

func getCPUNumaNode(numaNodes []int) map[int]int {
	numaNodeCPUs := readAndParseCPUs(numaNodes)

	cpuNumaNode := make(map[int]int)
	for numaNode, cpuList := range numaNodeCPUs {
		for _, cpu := range cpuList {
			cpuNumaNode[cpu] = numaNode
		}
	}
	return cpuNumaNode
}

func getNumaNodeFreeCPUs(freeCPUs []int, numaNodeCPUInfos map[int]numaNodeCPUInfo) map[int]int {
	freeCPUsSet := make(map[int]struct{})
	for _, cpu := range freeCPUs {
		freeCPUsSet[cpu] = struct{}{}
	}

	numaNodeFreeCPUsMap := make(map[int]int)
	for numaNode, numaNodeCPUInfo := range numaNodeCPUInfos {
		freeCPUsCount := 0
		for _, numaNodeCPU := range numaNodeCPUInfo.numaNodeCPUs {
			if _, exists := freeCPUsSet[numaNodeCPU]; exists {
				freeCPUsCount++
			}
		}
		numaNodeFreeCPUsMap[numaNode] = freeCPUsCount
	}
	return numaNodeFreeCPUsMap
}

func getContainerNumaNodeAssignment(numaNodes []int) (map[string]map[string]containerNumaInfo, error) {
	cpuManagerState, err := ioutil.ReadFile("/var/lib/kubelet/cpu_manager_state")
	if err != nil {
		return nil, fmt.Errorf("read cpu_manager_state file failed, err: %v", err)
	}

	cpuCheckpoint := utils.NewCPUManagerCheckpoint()
	if err := utils.LoadCheckpoint(cpuManagerState, cpuCheckpoint); err != nil {
		return nil, fmt.Errorf("unmarshal cpu_manager_state failed, err: %v", err)
	}

	if cpuCheckpoint.PolicyName == "none" {
		return nil, fmt.Errorf("cpu_manager_state policy is 'none'")
	}

	cpuNumaNode := getCPUNumaNode(numaNodes)

	containerNumaInfoMap := make(map[string]map[string]containerNumaInfo)
	for podUUID, containerEntries := range cpuCheckpoint.Entries {
		containerNumaInfoMap[podUUID] = make(map[string]containerNumaInfo)
		for containerName, cpuListStr := range containerEntries {
			cpuList, err := utils.ParseRanges(cpuListStr)
			if err != nil {
				zlog.FormatWarn("Parse CPU list failed. Pod UUID: %s, Container Name: %s, Error: %v",
					podUUID, containerName, err)
				continue
			}

			numaNodesList := determineNumaNodes(cpuList, cpuNumaNode)

			containerNumaInfoMap[podUUID][containerName] = containerNumaInfo{
				numaNodes:    numaNodesList,
				numaNodeCPUs: cpuList,
			}
		}
	}
	return containerNumaInfoMap, nil
}

func determineNumaNodes(cpuList []int, cpuNumaNode map[int]int) []int {
	numaNodeSet := make(map[int]struct{})
	for _, cpu := range cpuList {
		if numaNode, ok := cpuNumaNode[cpu]; ok {
			numaNodeSet[numaNode] = struct{}{}
		}
	}

	numaNodesList := make([]int, 0, len(numaNodeSet))
	for numaNode := range numaNodeSet {
		numaNodesList = append(numaNodesList, numaNode)
	}

	return numaNodesList
}

func init() {
	registerCollector("cpu_numa", NewCPUNumaCollector)
}
