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
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
	"volcano-config-service/pkg/zlog"
)

type distanceNumaCollector struct {
	numaNodeDistance *prometheus.Desc
}

// NewDistanceNumaCollector creates a new distanceNumaCollector instance.
func NewDistanceNumaCollector() (MetricCollector, error) {
	return &distanceNumaCollector{
		numaNodeDistance: prometheus.NewDesc(
			prometheus.BuildFQName("numa", "node", "distance"),
			"NUMA nodes access latency.",
			[]string{"node", "source_numa_node", "target_numa_node"}, nil,
		),
	}, nil
}

// Update collects NUMA node distance metrics and sends them to the provided channel.
func (d distanceNumaCollector) Update(ch chan<- prometheus.Metric) error {
	nodeName, err := utils.GetNodeName()
	if err != nil {
		return err
	}

	numaNodeDistances, err := getNumaNodeDistances()
	if err != nil {
		return err
	}
	for sourceNumaNode, distances := range numaNodeDistances {
		for targetNumaNode, distance := range distances {
			ch <- prometheus.MustNewConstMetric(d.numaNodeDistance, prometheus.GaugeValue, float64(distance),
				nodeName, fmt.Sprintf("%d", sourceNumaNode), fmt.Sprintf("%d", targetNumaNode))
		}
	}
	return nil
}

func getNumaNodeDistances() (map[int]map[int]int, error) {
	numaNodePaths, err := filepath.Glob("/sys/devices/system/node/node[0-9]*")
	if err != nil {
		return nil, err
	}

	numaNodeIDs, err := parseNumaNodeIDs(numaNodePaths)
	if err != nil {
		return nil, err
	}

	numaNodeDistances := make(map[int]map[int]int)
	for _, numaNodeID := range numaNodeIDs {
		distancePath := filepath.Join("/sys/devices/system/node/",
			fmt.Sprintf("node%d", numaNodeID), "distance")
		data, err := ioutil.ReadFile(distancePath)
		if err != nil {
			zlog.FormatWarn("Read distance file failed. Path: %s, Error: %v", distancePath, err)
			continue
		}

		distances := strings.Fields(string(data))
		if len(distances) != len(numaNodeIDs) {
			zlog.FormatWarn("Unexpected number of distances. Path: %s, Expected: %d, Got: %d",
				distancePath, len(numaNodeIDs), len(distances))
			continue
		}

		numaNodeDistances[numaNodeID] = make(map[int]int)
		for i, distanceStr := range distances {
			distance, err := strconv.Atoi(distanceStr)
			if err != nil {
				zlog.FormatWarn("Parse distance failed. NUMA Node ID: %d, Distance String: %s, Error: %v",
					numaNodeID, distanceStr, err)
				continue
			}
			numaNodeDistances[numaNodeID][numaNodeIDs[i]] = distance
		}
	}
	return numaNodeDistances, nil
}

func parseNumaNodeIDs(numaNodePaths []string) ([]int, error) {
	numaNodeIDs := make([]int, 0, len(numaNodePaths))
	for _, numaNodePath := range numaNodePaths {
		numaNodeID, err := strconv.Atoi(strings.TrimPrefix(filepath.Base(numaNodePath), "node"))
		if err != nil {
			return nil, fmt.Errorf("parse NUMA node ID failed for path %s: %v", numaNodePath, err)
		}
		numaNodeIDs = append(numaNodeIDs, numaNodeID)
	}
	return numaNodeIDs, nil
}

func init() {
	registerCollector("distances_numa", NewDistanceNumaCollector)
}
