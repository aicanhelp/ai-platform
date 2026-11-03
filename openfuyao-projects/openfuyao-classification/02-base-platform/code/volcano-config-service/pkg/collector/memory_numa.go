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
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"

	"github.com/prometheus/client_golang/prometheus"

	"volcano-config-service/pkg/utils"
	"volcano-config-service/pkg/zlog"
)

const (
	fieldIndexValue = 3
	fieldSize       = 3
	fieldIndexUnit  = 4
	float64BitSize  = 64
)

type memNumaMetric struct {
	metricName      string
	metricValueType prometheus.ValueType
	nodeName        string
	numaNodeName    string
	metricValue     float64
}

type memNumaCollector struct {
	metricDescriptions map[string]*prometheus.Desc
}

// NewMemNumaCollector creates a new memNumaCollector instance.
func NewMemNumaCollector() (MetricCollector, error) {
	return &memNumaCollector{
		metricDescriptions: map[string]*prometheus.Desc{},
	}, nil
}

// Update collects NUMA memory metrics and sends them to the provided channel.
func (mem memNumaCollector) Update(ch chan<- prometheus.Metric) error {
	nodeName, err := utils.GetNodeName()
	if err != nil {
		return err
	}

	metrics, err := getMemInfoNuma(nodeName)
	if err != nil {
		return fmt.Errorf("couldn't get NUMA meminfo: %w", err)
	}

	for _, c := range metrics {
		description, ok := mem.metricDescriptions[c.metricName]
		if !ok {
			description = prometheus.NewDesc(
				prometheus.BuildFQName("numa_node", "memory", c.metricName),
				fmt.Sprintf("metric name %s.", c.metricName),
				[]string{"node", "numa_node"}, nil)
			mem.metricDescriptions[c.metricName] = description
		}
		ch <- prometheus.MustNewConstMetric(description, c.metricValueType, c.metricValue, c.nodeName, c.numaNodeName)
	}

	return nil
}

func getMemInfoNuma(nodeName string) ([]memNumaMetric, error) {
	var metrics []memNumaMetric

	numaNodePaths, err := filepath.Glob("/sys/devices/system/node/node[0-9]*")
	if err != nil {
		return nil, err
	}
	for _, numaNodePath := range numaNodePaths {
		numaInfo, err := parseMemInfoNuma(nodeName, numaNodePath)
		if err != nil {
			zlog.FormatWarn("Parse meminfo failed. NUMA Node Path: %s, Error: %v", numaNodePath, err)
			continue
		}
		metrics = append(metrics, numaInfo...)
	}

	return metrics, nil
}

func parseMemInfoNuma(nodeName string, numaNodePath string) ([]memNumaMetric, error) {
	var memMetrics []memNumaMetric

	memInfoPath := filepath.Join(numaNodePath, "meminfo")
	memInfoFile, err := os.Open(memInfoPath)
	if err != nil {
		return nil, fmt.Errorf("read meminfo file %s failed, err: %v", memInfoPath, err)
	}
	defer memInfoFile.Close()

	scanner := bufio.NewScanner(memInfoFile)
	re := regexp.MustCompile(`\((.*)\)`)

	for scanner.Scan() {
		context := strings.TrimSpace(scanner.Text())
		if context == "" {
			continue
		}
		fields := strings.Fields(context)
		if len(fields) <= fieldSize {
			return nil, fmt.Errorf("meminfo is invalid: %s", context)
		}

		fv, err := strconv.ParseFloat(fields[fieldIndexValue], float64BitSize)
		if err != nil {
			return nil, fmt.Errorf("meminfo have invalid value: %w", err)
		}
		if len(fields) == fieldIndexUnit+1 && fields[fieldIndexUnit] == "kB" {
			fv *= 1024
		}

		memMetricName := strings.TrimRight(fields[2], ":")
		memMetricName = re.ReplaceAllString(memMetricName, "_${1}")
		memMetric := memNumaMetric{
			metricName:      memMetricName,
			metricValueType: prometheus.GaugeValue,
			nodeName:        nodeName,
			numaNodeName:    fields[1],
			metricValue:     fv}
		memMetrics = append(memMetrics, memMetric)
	}

	return memMetrics, scanner.Err()
}

func init() {
	registerCollector("memory_numa", NewMemNumaCollector)
}
