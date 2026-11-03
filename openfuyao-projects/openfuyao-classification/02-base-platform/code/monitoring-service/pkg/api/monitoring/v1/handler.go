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

// Package v1 provides the HTTP handlers for the v1 API of the monitoring-service. This package
// defines routes and their corresponding handlers that interface with the MonitoringOperator
// to perform various metric-related operations. These operations include querying Prometheus for
// metrics across different levels such as clusters, nodes, deployments, stateful sets, daemon sets,
// pods, and containers. Additionally, it handles queries for control plane, node components, and
// addon metrics, as well as user-defined custom queries.
package v1

import (
	"regexp"

	"github.com/emicklei/go-restful/v3"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/server/handlestatus"
	"openfuyao.com/monitoring-service/pkg/services"
)

type handler struct {
	mo services.MonitoringOperator
}

// NewHandler creates and returns a new handler instance.
func NewHandler(monitoringClient client.Interface) *handler {
	return &handler{
		mo: services.NewMonitoringOperator(monitoringClient),
	}
}

func (h handler) handleClusterMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelCluster)
}

func (h handler) handleNodeMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelNode)
}

func (h handler) handleDeploymentMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelDeployment)
}

func (h handler) handleStatefulsetMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelStatefulset)
}

func (h handler) handleDaemonsetMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelDaemonset)
}

func (h handler) handlePodMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelPod)
}

func (h handler) handleContainerMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelContainer)
}

func (h handler) handleControlPlaneComponentsMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelControlPlaneComponents)
}

func (h handler) handleNodeComponentsMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelNodeComponents)
}

func (h handler) handleAddonsComponentsMetricsQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelAddonsComponents)
}

func (h handler) handleQuery(req *restful.Request, resp *restful.Response) {
	h.handleMetricsQuery(req, resp, client.LevelQuery)
}

func (h handler) handleMetricsQuery(req *restful.Request, resp *restful.Response, level client.Level) {
	var params reqParams
	var opt queryOptions
	var err error

	if level >= client.LevelControlPlaneComponents {
		params = parseRequestParamsComponents(req)
		opt, err = transformQueryOptionsComponents(params, level)
	} else {
		params = parseRequestParamsResources(req)
		opt, err = transformQueryOptionsResources(params, level)
	}

	if err != nil {
		handlestatus.BadRequest(resp, nil, err)
		return
	}

	if level == client.LevelQuery {
		h.handleUserQuery(resp, params, opt)
	} else {
		h.handleCustomMetricsQuery(resp, opt)
	}

}

func (h handler) handleUserQuery(resp *restful.Response, reqParams reqParams, queryOpts queryOptions) {
	var res client.Metric

	if queryOpts.isRangeQuery() {
		res = h.mo.QueryRangeMetrics(reqParams.expression, queryOpts.start, queryOpts.end, queryOpts.step)
	} else {
		res = h.mo.QueryInstantMetrics(reqParams.expression, queryOpts.time)
	}

	resp.WriteAsJson(res)

}

func (h handler) handleCustomMetricsQuery(resp *restful.Response, queryOpts queryOptions) {
	var res services.Metrics

	var metrics []string
	for _, metric := range queryOpts.namedMetrics {
		ok, err := regexp.MatchString(queryOpts.metricFilter, metric)
		if err != nil {
			handlestatus.BadRequest(resp, nil, err)
			return
		}
		if ok {
			metrics = append(metrics, metric)
		}
	}
	if len(metrics) == 0 {
		resp.WriteAsJson(res)
		return
	}

	if queryOpts.isRangeQuery() {
		res = h.mo.QueryRangeCustomMetrics(metrics, queryOpts.start, queryOpts.end, queryOpts.step, queryOpts.option)
	} else {
		res = h.mo.QueryInstantCustomMetrics(metrics, queryOpts.time, queryOpts.option)
	}
	resp.WriteAsJson(res)
}

func (h handler) handleMetricNames(req *restful.Request, resp *restful.Response) {
	var res services.NativeMetricNames

	res = h.mo.GetMetricNames()

	resp.WriteAsJson(res)
}

func (h handler) handleTargetsDiscovery(req *restful.Request, resp *restful.Response) {
	var res services.TargetsData

	res = h.mo.GetTargets()

	resp.WriteAsJson(res)

}

func (h handler) handleAlertingRules(req *restful.Request, resp *restful.Response) {
	var res services.AlertingRulesData

	res = h.mo.GetAlertingRules()

	resp.WriteAsJson(res)

}
