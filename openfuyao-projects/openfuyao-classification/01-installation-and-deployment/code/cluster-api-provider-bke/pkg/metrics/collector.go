/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	dto "github.com/prometheus/client_model/go"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/log"
)

const (
	clusterSubSystem string = "cluster"
	namespace               = "bke"
)

// Collector is the interface a collector has to implement.
type Collector interface {
	Collect(chan<- prometheus.Metric)
	Describe(chan<- *prometheus.Desc)
	DeleteMetricWithLabels(labels prometheus.Labels)
	RegisterVec(vec ...func(clusterKey string) (string, prometheus.Collector))
	Export(vecNames ...string) (map[string][]map[string]string, error)
	RecordGaugeVecWithLabelValues(vecName string, lvs ...string) prometheus.Gauge
	RecordCounterVecWithLabelValues(vecName string, lvs ...string) prometheus.Counter
}

type bkeClusterCollector struct {
	vecs       map[string]prometheus.Collector
	clusterKey string
}

func NewBKEClusterCollector(clusterKey string) Collector {
	collector := &bkeClusterCollector{
		vecs:       make(map[string]prometheus.Collector),
		clusterKey: clusterKey,
	}
	RegisterBkeVec(collector)
	return collector
}

func (c *bkeClusterCollector) Describe(ch chan<- *prometheus.Desc) {
	for _, vec := range c.vecs {
		vec.Describe(ch)
	}
}

func (c *bkeClusterCollector) Collect(ch chan<- prometheus.Metric) {
	for _, vec := range c.vecs {
		vec.Collect(ch)
	}
}

func (c *bkeClusterCollector) DeleteMetricWithLabels(labels prometheus.Labels) {
	for _, vec := range c.vecs {
		if gv, ok := vec.(*prometheus.GaugeVec); ok {
			gv.DeletePartialMatch(labels)
		}
		if cv, ok := vec.(*prometheus.CounterVec); ok {
			cv.DeletePartialMatch(labels)
		}
	}
}

func (c *bkeClusterCollector) RecordGaugeVecWithLabelValues(vecName string, lvs ...string) prometheus.Gauge {
	if vec, ok := c.vecs[vecName]; ok {
		if gv, ok := vec.(*prometheus.GaugeVec); ok {
			return gv.WithLabelValues(lvs...)
		}
		return nil
	}
	return nil
}

func (c *bkeClusterCollector) RecordCounterVecWithLabelValues(vecName string, lvs ...string) prometheus.Counter {
	if vec, ok := c.vecs[vecName]; ok {
		if cv, ok := vec.(*prometheus.CounterVec); ok {
			return cv.WithLabelValues(lvs...)
		}
		return nil
	}
	return nil
}

func (c *bkeClusterCollector) RegisterVec(vecFuncs ...func(clusterKey string) (string, prometheus.Collector)) {
	for _, f := range vecFuncs {
		vecName, vec := f(c.clusterKey)
		c.vecs[vecName] = vec
	}
}

func (c *bkeClusterCollector) Export(vecNames ...string) (map[string][]map[string]string, error) {

	res := make(map[string][]map[string]string, 0)
	for vecName, vec := range c.vecs {

		if !utils.ContainsString(vecNames, vecName) {
			continue
		}

		log.Debugf("export vec: %s", vecName)
		metricChan := make(chan prometheus.Metric)
		res[vecName] = make([]map[string]string, 0)

		// Capture vec in a local variable to avoid closure issue
		currentVec := vec
		go func() {
			currentVec.Collect(metricChan)
			close(metricChan)
		}()

		for metric := range metricChan {
			dtoMetric := &dto.Metric{}
			if err := metric.Write(dtoMetric); err != nil {
				log.Errorf("export vec: %s, error: %v", vecName, err)
				continue
			}
			item := make(map[string]string, 0)
			for _, label := range dtoMetric.Label {
				item[label.GetName()] = label.GetValue()
			}

			res[vecName] = append(res[vecName], item)
		}

		log.Debugf("export vec: %s done, res: %d", vecName, len(res[vecName]))
	}
	return res, nil
}

func newGaugeOpts(name, clusterKey string) prometheus.GaugeOpts {
	return prometheus.GaugeOpts{
		Namespace: namespace,
		Subsystem: clusterSubSystem,
		Name:      name,
		ConstLabels: prometheus.Labels{
			"bke_cluster": clusterKey,
		},
	}
}

func newCounterOpts(name, clusterKey string) prometheus.CounterOpts {
	return prometheus.CounterOpts{
		Namespace: namespace,
		Subsystem: clusterSubSystem,
		Name:      name,
		ConstLabels: prometheus.Labels{
			"bke_cluster": clusterKey,
		},
	}
}
