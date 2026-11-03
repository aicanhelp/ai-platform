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

package v1

import (
	"net/http"

	"github.com/emicklei/go-restful-openapi/v2"
	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/server/runtime"
	"openfuyao.com/monitoring-service/pkg/services"
)

const (
	groupName = "monitoring"
	respOK    = "ok"
)

const (
	clusterMetricsTag                = "Cluster Metrics"
	nodeMetricsTag                   = "Node Metrics"
	workloadMetricsTag               = "Workload Metrics"
	podMetricsTag                    = "Pod Metrics"
	containerMetricsTag              = "Container Metrics"
	controlPlaneComponentsMetricsTag = "Control Plane Components Metrics"
	nodeComponentsMetricsTag         = "Node Components Metrics"
	addonsComponentsMetricsTag       = "Addons Components Metrics"
	customMetricsTag                 = "Custom Metrics"
	allMetricNamesTag                = "All Metric Names"
	targetsTag                       = "Targets"
	alertingRulesTarget              = "Alerting Rules"
)

var apiGroupVersion = schema.GroupVersion{Group: groupName, Version: "v1"}

func addClusterMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/cluster/metrics").
		To(h.handleClusterMetricsQuery).
		Doc("Obtain cluster-wide monitoring metrics.").
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter cluster-level metric data. "+
			"For example, 'cluster_cpu_usage|cluster_disk_size_usage' "+
			"matches metrics related to cluster CPU and disk usage.").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{clusterMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addNodeMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/nodes/{node}/metrics").
		To(h.handleNodeMetricsQuery).
		Doc("Obtain node-wide monitoring metrics.").
		Param(ws.PathParameter("node", "The name of the node.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter node-level metric data. "+
			"For example, 'node_cpu_usage|node_memory_usage' "+
			"matches metrics related to node CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{nodeMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addDeploymentMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/namespaces/{namespace}/deployments/{deployment}/metrics").
		To(h.handleDeploymentMetricsQuery).
		Doc("Get pod-level metric data for pods within a specific deployment.").
		Param(ws.PathParameter("namespace", "The name of the namespace.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("deployment", "Deployment name.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter",
			"Regexp pattern to filter metric data based on the workload type - deployment. "+
				"For example, 'workload_cpu_usage|workload_memory_usage' "+
				"matches metrics related to workload CPU usage and memory usage.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("resources_filter", "Regexp pattern to filter pod. For example, 'alert.*'"+
			" matches any pod whose name begins with alert.").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{workloadMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addStatefulsetMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/namespaces/{namespace}/statefulsets/{statefulset}/metrics").
		To(h.handleStatefulsetMetricsQuery).
		Doc("Get pod-level metric data for pods within a specific statefulset.").
		Param(ws.PathParameter("namespace", "The name of the namespace.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("statefulset", "Statefulset name.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter",
			"Regexp pattern to filter pod-level metric data based on the workload type - statefulset. "+
				"For example, 'workload_cpu_usage|workload_network_packets_received' "+
				"can be used to retrieve metrics for workload CPU usage and network packets received.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("resources_filter", "Regexp pattern to filter pod. For example, 'redi.*'"+
			" matches any pod whose name begins with redis.").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{workloadMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addDaemonsetMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/namespaces/{namespace}/daemonsets/{daemonset}/metrics").
		To(h.handleDaemonsetMetricsQuery).
		Doc("Get pod-level metric data for pods within a specific daemonset.").
		Param(ws.PathParameter("namespace", "The name of the namespace.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("daemonset", "Daemonset name.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter",
			"Regexp pattern to filter pod-level metric data based on the workload type - daemonset. "+
				"For example, 'workload_cpu_usage|workload_network_packets_transmitted' "+
				"matches metrics related to workload CPU usage and network packets transmitted.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("resources_filter", "Regexp pattern to filter pod. For example, 'node.* '"+
			" matches any pod whose name begins with node.").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{workloadMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addPodMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/namespaces/{namespace}/pods/{pod}/metrics").
		To(h.handlePodMetricsQuery).
		Doc("Get pod-level metric data for a specific pod within a namespace.").
		Param(ws.PathParameter("namespace", "The name of the namespace.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("pod", "Pod name.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter pod-level metric data. "+
			"For example, 'pod_cpu_usage|pod_memory_usage' "+
			"matches metrics related to pod CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{podMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addContainerMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/namespaces/{namespace}/pods/{pod}/containers/{container}/metrics").
		To(h.handleContainerMetricsQuery).
		Doc("Get container-level metric data for a specific container within a pod.").
		Param(ws.PathParameter("namespace", "The name of the namespace.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("pod", "Pod name.").
			DataType("string").Required(true)).
		Param(ws.PathParameter("container", "Container name.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter container-level metric data. "+
			"For example, 'container_cpu_usage|container_memory_usage' "+
			"matches metrics related to container CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{containerMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addControlPlaneComponentsMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("control-plane-components/{component}").
		To(h.handleControlPlaneComponentsMetricsQuery).
		Doc("Get metric data of a specific control plane component.").
		Param(ws.PathParameter("component", "The name of the control plane component.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("verb", "The type of HTTP request (e.g., GET, POST, DELETE) "+
			"being made to the API server.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("resource", "The specific Kubernetes resource type "+
			"(e.g., Configmaps, Pods, deployments) "+
			"that the request is targeting.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("instance", "The combination of the service's "+
			"IP address and port number.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter metric data. "+
			"For example, 'etcd_cpu_usage|etcd_memory_usage' "+
			"matches metrics related to etcd CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{controlPlaneComponentsMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addNodeComponentsMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("node-components/{component}").
		To(h.handleNodeComponentsMetricsQuery).
		Doc("Get metric data of a specific node component.").
		Param(ws.PathParameter("component", "The name of the node component.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("instance", "The combination of the service's "+
			"IP address and port number.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter metric data. "+
			"For example, 'kubelet_cpu_usage|kubelet_memory_usage' "+
			"matches metrics related to kubelet CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{nodeComponentsMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addAddonsComponentsMetricsQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("add-ons-components/{component}").
		To(h.handleAddonsComponentsMetricsQuery).
		Doc("Get metric data of a specific addons component.").
		Param(ws.PathParameter("component", "The name of the addons component.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("instance", "The combination of the service's "+
			"IP address and port number.").
			DataType("string").Required(true)).
		Param(ws.QueryParameter("metrics_filter", "Regexp pattern to filter metric data. "+
			"For example, 'coredns_cpu_usage|coredns_memory_usage' "+
			"matches metrics related to coredns CPU usage and memory usage: ").
			DataType("string").Required(false)).
		Metadata(restfulspec.KeyOpenAPITags, []string{addonsComponentsMetricsTag}).
		Writes(services.TargetsData{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addQueryRoute(ws *restful.WebService, h *handler) {
	ws.Route(addCommonQueryParams(ws, ws.GET("/query").
		To(h.handleQuery).
		Doc("Execute a PromQL query for existing Prometheus metrics.").
		Param(ws.QueryParameter("expr", "The expression to be evaluated.").
			DataType("string").Required(true)).
		Metadata(restfulspec.KeyOpenAPITags, []string{customMetricsTag}).
		Writes(services.Metrics{}).
		Returns(http.StatusOK, respOK, services.Metrics{})).
		Produces(restful.MIME_JSON))
}

func addMetricNamesRoute(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/metric-names").
		To(h.handleMetricNames).
		Doc("Get all the metric names.").
		Metadata(restfulspec.KeyOpenAPITags, []string{allMetricNamesTag}).
		Writes(services.NativeMetricNames{}).
		Returns(http.StatusOK, respOK, services.NativeMetricNames{})).
		Produces(restful.MIME_JSON)
}

func addTargetsDiscoveryRoute(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/targets").
		To(h.handleTargetsDiscovery).
		Doc("Make an overview of all the monitoring targets.").
		Metadata(restfulspec.KeyOpenAPITags, []string{targetsTag}).
		Writes(services.TargetsData{}).
		Returns(http.StatusOK, respOK, services.TargetsData{})).
		Produces(restful.MIME_JSON)
}

func addAlertingRulesRoute(ws *restful.WebService, h *handler) {
	ws.Route(ws.GET("/alerting-rules").
		To(h.handleAlertingRules).
		Doc("Make an overview of all the alerting rules.").
		Metadata(restfulspec.KeyOpenAPITags, []string{alertingRulesTarget}).
		Writes(services.AlertingRulesData{}).
		Returns(http.StatusOK, respOK, services.AlertingRulesData{})).
		Produces(restful.MIME_JSON)
}

func addCommonQueryParams(ws *restful.WebService, route *restful.RouteBuilder) *restful.RouteBuilder {
	return route.
		Param(ws.QueryParameter("start", "Start time of query in the format YYYY-MM-DD HH:MM:SS. "+
			"Use with 'end' to retrieve metric data over a span.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("end", "End time of query in the format YYYY-MM-DD HH:MM:SS. "+
			"Use with 'start' to retrieve metric data over a span.").
			DataType("string").Required(false)).
		Param(ws.QueryParameter("step", "Time interval in [0-9]+[smhdwy] format, "+
			"e.g., 10m for 10 minutes. Requires both 'start' and 'end'.").
			DataType("string").DefaultValue("10m").Required(false)).
		Param(ws.QueryParameter("time", "Single point in time in the format YYYY-MM-DD HH:MM:SS. "+
			"Defaults to now. Mutually exclusive with 'start', 'end', 'step'.").
			DataType("string").Required(false))
}

// AddToContainer initializes and adds routes to a RESTful container for monitoring functionalities.
func AddToContainer(c *restful.Container, monitoringClient client.Interface) error {
	ws := runtime.NewRestWebService(apiGroupVersion)

	h := NewHandler(monitoringClient)

	addClusterMetricsQueryRoute(ws, h)
	addNodeMetricsQueryRoute(ws, h)
	addDeploymentMetricsQueryRoute(ws, h)
	addStatefulsetMetricsQueryRoute(ws, h)
	addDaemonsetMetricsQueryRoute(ws, h)
	addPodMetricsQueryRoute(ws, h)
	addContainerMetricsQueryRoute(ws, h)
	addControlPlaneComponentsMetricsQueryRoute(ws, h)
	addNodeComponentsMetricsQueryRoute(ws, h)
	addAddonsComponentsMetricsQueryRoute(ws, h)
	addQueryRoute(ws, h)
	addMetricNamesRoute(ws, h)
	addTargetsDiscoveryRoute(ws, h)
	addAlertingRulesRoute(ws, h)

	c.Add(ws)
	return nil
}
