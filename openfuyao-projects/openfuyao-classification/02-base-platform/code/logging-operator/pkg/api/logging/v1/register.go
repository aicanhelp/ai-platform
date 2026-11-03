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

// Package v1 provides the HTTP request handling and interaction logic for
// a logging server designed to work within Kubernetes environments. This package
// encapsulates functionalities for making and processing HTTP requests
// to Loki, a horizontally-scalable, highly-available, multi-tenant log aggregation
// system. It aims to facilitate logging data retrieval and analysis by
// providing structured access to logs stored in Loki.
//
// The package defines the APIServer and Handler types that manage the HTTP server
// and request routing respectively. The APIServer serves as the entry point for
// incoming HTTP requests, whereas Handler implements the logic for processing
// these requests, including querying Loki for log data based on various
// parameters such as filename, time range, and log level.
//
// Key Components:
//
//   - HTTPRequest: A struct that encapsulates details of an HTTP request, including
//     URL, method, headers, and body. It serves as a foundational element for
//     constructing requests to Loki.
//
//   - Handler: The core struct that handles incoming requests, routes them to
//     appropriate processing functions, and interacts with Loki to fetch or
//     manipulate log data. It maintains a map of predefined Loki query configurations
//     and a dictionary mapping human-readable time units to their respective
//     durations for flexible time range queries.
//
//   - LogData: A struct designed to unmarshal JSON responses from Loki, specifically
//     targeting the data structure returned by Loki's series and query_range APIs.
//
// Functionalities:
//
//   - Dynamic Loki Query Construction: Allows building and executing complex Loki
//     queries based on request parameters, enabling clients to perform detailed log
//     searches and analysis.
//
//   - Time Range Calculation: Offers utilities to calculate start and end times for
//     queries based on either specific timestamps or relative time units (e.g., "last 5 minutes").
//
//   - API Endpoints: Defines RESTful endpoints for interacting with the logging server,
//     including endpoints for listing available log sources and querying logs based
//     on various filters.
//
// Usage:
// To use this package, create an instance of the Handler struct via the New function,
// then call its methods to handle incoming requests or interact with Loki. The
// package's functionality is encapsulated in the Handler type, making it the primary
// interface for clients.
//
// Note:
// This package assumes that Loki is accessible via HTTP and that clients have
// necessary permissions to query Loki. It also relies on external packages such
// as "github.com/emicklei/go-restful/v3" for RESTful routing and
// "k8s.io/apimachinery/pkg/util/runtime" for Kubernetes-related utilities.
//
// The package is designed with extensibility in mind, allowing for future
// enhancements such as additional Loki query types, improved error handling,
// and richer response parsing.
package v1

import (
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"

	"openfuyao/logging-operator/pkg/server"
)

const (
	groupName = "logging"
)

// GroupVersion defines the API group and version for the logging service.
var groupVersion = schema.GroupVersion{Group: groupName, Version: "v1"}

func addFileListRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/filelist").
		To(handler.getLokiFileName).
		Deprecate().
		Doc("Return total file name list").
		Param(webservice.QueryParameter(server.PodName, "The name of filter pod name")).
		Returns(http.StatusOK, "List of files returned successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error while fetching files", nil))

}

func addFilenamesLogsRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/filenames/logs").
		To(handler.lokiFilterSearch).
		Deprecate().
		Doc("Filtering logs based on specific file name").
		Param(webservice.QueryParameter(server.TimePeriod, server.TimePeriodDiscription)).
		Param(webservice.QueryParameter(server.FileName, server.FileNameDiscription)).
		Param(webservice.QueryParameter(server.StartTime, server.StartTimeDiscription)).
		Param(webservice.QueryParameter(server.EndTime, server.EndTimeDiscription)).
		Param(webservice.QueryParameter(server.Level, server.LevelDiscription)).
		Param(webservice.QueryParameter(server.Keyword, server.KeywordDiscription)).
		Param(webservice.QueryParameter(server.Sizes, server.SizesDiscription)).
		Returns(http.StatusOK, "Logs filtered successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Failed to filter logs due to server error", nil))
}

func addNamespaceLogsRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/namespaces/{namespace}/logs").
		To(handler.lokiNamespaceSearch).
		Deprecate().
		Doc("Filtering logs based on specific namespace").
		Param(webservice.PathParameter("namespace", "The namespace of a log resource")).
		Param(webservice.QueryParameter(server.StartTime, server.StartTimeDiscription)).
		Param(webservice.QueryParameter(server.TimePeriod, server.TimePeriodDiscription)).
		Param(webservice.QueryParameter(server.EndTime, server.EndTimeDiscription)).
		Param(webservice.QueryParameter(server.Level, server.LevelDiscription)).
		Param(webservice.QueryParameter(server.Keyword, server.KeywordDiscription)).
		Param(webservice.QueryParameter(server.Sizes, server.SizesDiscription)).
		Returns(http.StatusOK, "Namespace logs returned successfully", nil).
		Returns(http.StatusBadRequest, "Invalid namespace specified", nil).
		Returns(http.StatusInternalServerError, "Internal server error during namespace log retrieval", nil))
}

func addResourceListRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/resource-list").
		To(handler.getResourceList).
		Deprecate().
		Doc("Demonstrate the resources list").
		Param(webservice.QueryParameter(server.Namespace, server.NamespaceDiscription)).
		Param(webservice.QueryParameter(server.PodName, server.PodNameDiscription)).
		Param(webservice.QueryParameter(server.Container, server.ContainerDiscription)).
		Returns(http.StatusOK, "Resource list provided successfully", nil).
		Returns(http.StatusBadRequest, "Invalid parameters for resource listing", nil).
		Returns(http.StatusInternalServerError, "Failed to retrieve resources due to server error", nil))
}

func addPodLogsRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/namespaces/{namespace}/pods/{pod}/logs").
		To(handler.lokiPodSearch).
		Deprecate().
		Doc("Filtering logs based on specific pod").
		Param(webservice.PathParameter("namespace", "The namespace of a log resource")).
		Param(webservice.PathParameter("pod", "The pod of a log resource")).
		Param(webservice.QueryParameter(server.StartTime, server.StartTimeDiscription)).
		Param(webservice.QueryParameter(server.TimePeriod, server.TimePeriodDiscription)).
		Param(webservice.QueryParameter(server.EndTime, server.EndTimeDiscription)).
		Param(webservice.QueryParameter(server.Level, server.LevelDiscription)).
		Param(webservice.QueryParameter(server.Keyword, server.KeywordDiscription)).
		Param(webservice.QueryParameter(server.Sizes, server.SizesDiscription)).
		Returns(http.StatusOK, "Pod logs retrieved successfully", nil).
		Returns(http.StatusBadRequest, "Invalid pod name or other parameters", nil).
		Returns(http.StatusInternalServerError, "Server error while retrieving pod logs", nil))
}

func addContainerLogsRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/namespaces/{namespace}/pods/{pod}/containers/{container}/logs").
		To(handler.lokiContainerSearch).
		Deprecate().
		Doc("Filtering logs based on specific container").
		Param(webservice.PathParameter("namespace", "The namespace of a log resource")).
		Param(webservice.PathParameter("pod", "The pod of a log resource")).
		Param(webservice.PathParameter("container", "The container of a log resource")).
		Param(webservice.QueryParameter(server.StartTime, server.StartTimeDiscription)).
		Param(webservice.QueryParameter(server.TimePeriod, server.TimePeriodDiscription)).
		Param(webservice.QueryParameter(server.EndTime, server.EndTimeDiscription)).
		Param(webservice.QueryParameter(server.Level, server.LevelDiscription)).
		Param(webservice.QueryParameter(server.Keyword, server.KeywordDiscription)).
		Param(webservice.QueryParameter(server.Sizes, server.SizesDiscription)).
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error occurred", nil))
}

func addAlertingRulesRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/rules").
		To(handler.getAlertingRules).
		Deprecate().
		Doc("List all alerting rules").
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error occurred", nil))
}

func addDisplaySourcesRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/sources").
		To(handler.displayconfigmap).
		Deprecate().
		Doc("Filtering logs based on specific container").
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error occurred", nil))
}

func addtestresources(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/configmap").
		To(handler.getConfigmapDetails).
		Deprecate().
		Doc("Filtering logs based on specific container").
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error occurred", nil))
}

func addtestpatch(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.PATCH("/update-config").
		To(handler.updatePromtailConfig).
		Doc("Update promtail config with new job and path").
		Reads(PromtailUpdateRequest{}). // 定义接收的结构体
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request", nil).
		Returns(http.StatusInternalServerError, "Server error", nil))
}

func addDeleteJobRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.DELETE("/delete-job/{job-name}").
		To(handler.deleteJob).
		Doc("Deletes a job configuration from the promtail config").
		Param(webservice.PathParameter("job-name", "Name of the job to delete")).
		Returns(http.StatusOK, "Job deleted successfully", nil).
		Returns(http.StatusBadRequest, "Invalid job name", nil).
		Returns(http.StatusInternalServerError, "Failed to delete job", nil))
}

func addCheckPodStatusRoute(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/check").
		To(handler.getPodRunningStatus).
		Doc("Check the Promtail whether is available").
		Returns(http.StatusOK, "The promtail now is available", nil).
		Returns(http.StatusBadRequest, "The promtail is configuring", nil).
		Returns(http.StatusInternalServerError, "The promtail is not found", nil))
}
func addContainerContext(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/detail").
		To(handler.handleContextLogDetails).
		Deprecate().
		Doc("Filtering logs based on specific container").
		Param(webservice.QueryParameter(server.FileName, server.FileNameDiscription)).
		Param(webservice.QueryParameter(server.TimeStamp, server.StartTimeDiscription)).
		Param(webservice.QueryParameter(server.TimePeriod, server.TimePeriodDiscription)).
		Param(webservice.QueryParameter(server.Level, server.LevelDiscription)).
		Param(webservice.QueryParameter(server.Keyword, server.KeywordDiscription)).
		Param(webservice.QueryParameter(server.Sizes, server.SizesDiscription)).
		Returns(http.StatusOK, "Config updated successfully", nil).
		Returns(http.StatusBadRequest, "Invalid request due to incorrect parameters", nil).
		Returns(http.StatusInternalServerError, "Internal server error occurred", nil))
}

func addCheckLogStorageStragy(webservice *restful.WebService, handler *Handler) {
	webservice.Route(webservice.GET("/secret").
		To(handler.getlogSecrets).
		Doc("Check the Promtail whether is available").
		Returns(http.StatusOK, "The promtail now is available", nil).
		Returns(http.StatusBadRequest, "The promtail is configuring", nil).
		Returns(http.StatusInternalServerError, "The promtail is not found", nil))
}

// AddLogsContainer initializes and adds routes to a RESTful container for logging functionalities.
func AddLogsContainer(c *restful.Container) error {
	webservice := server.NewWebService(groupVersion)
	handler := New()

	addFileListRoute(webservice, handler)
	addFilenamesLogsRoute(webservice, handler)
	addNamespaceLogsRoute(webservice, handler)
	addPodLogsRoute(webservice, handler)
	addContainerLogsRoute(webservice, handler)
	addAlertingRulesRoute(webservice, handler)
	addtestresources(webservice, handler)
	addtestpatch(webservice, handler)
	addDeleteJobRoute(webservice, handler)
	addCheckPodStatusRoute(webservice, handler)
	addContainerContext(webservice, handler)
	addCheckLogStorageStragy(webservice, handler)
	addResourceListRoute(webservice, handler)
	addDisplaySourcesRoute(webservice, handler)
	c.Add(webservice)
	return nil
}
