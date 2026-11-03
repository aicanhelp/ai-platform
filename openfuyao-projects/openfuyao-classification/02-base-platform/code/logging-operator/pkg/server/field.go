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

// Package server provides the foundational infrastructure for setting up
// a server within a Kubernetes-oriented microservice architecture. It includes
// structures and functions for configuring and running both secure and insecure
// HTTP servers, facilitating the creation and management of RESTful web services,
// and handling common server-related tasks such as parsing query parameters
// from incoming requests.
//
// The Server struct encapsulates basic server configurations, including
// bind addresses and port numbers for both insecure and secure communication.
// This struct serves as the core configuration entity around which the server
// functionalities revolve.
//
// The RunConfig struct wraps around the Server configuration, providing a
// higher-level abstraction for managing server run configurations. This allows
// for easy instantiation of server settings and can be extended to include
// more complex configuration scenarios in the future.
//
// Key Functionalities:
//   - Server initialization with default settings through the NewServer function,
//     offering a quick setup for development and testing environments.
//   - Flexible server configuration management using the RunConfig struct,
//     allowing for dynamic adjustments to server settings.
//   - Creation of RESTful web services tailored for Kubernetes applications,
//     facilitated by the NewWebService function. This function prepares a
//     web service with a specific API versioning, adhering to best practices
//     in API design and versioning.
//   - Utility function GetQueryParamOrDefault to efficiently extract query
//     parameters from incoming HTTP requests, with support for default values.
//     This simplifies the handling of optional query parameters in web service
//     endpoints.
//
// Usage:
// To utilize the server package, start by creating a server instance with
// default configurations or customize it as needed. Then, define web services
// using the NewWebService function, specifying the desired API group and version.
// Finally, leverage the GetQueryParamOrDefault function within your handler
// functions to read and manage query parameters gracefully.
// This package is designed to simplify server setup and configuration in
// Go-based Kubernetes applications, promoting best practices in RESTful API
// development and providing common utilities needed for server management.
package server

const (
	// StartTime represents the starting point of the time range for log queries.
	StartTime = "startTime"
	// EndTime represents the ending point of the time range for log queries.
	EndTime = "endTime"
	// FileName is used to specify the log file name in log filtering operations.
	FileName = "fileName"
	// TimePeriod is used to specify the duration for the log query.
	TimePeriod = "timePeriod"
	// Sizes defines the maximum number of log entries to return in the query result.
	Sizes = "size"
	// Keyword is used to filter log entries containing specific terms.
	Keyword = "keyword"
	// Level is used to filter logs based on severity level (e.g., ERROR, WARN).
	Level = "level"
	// PodName is used to specify the target pod for querying logs or resources.
	PodName = "podName"
	// Namespace is used to specify the Kubernetes namespace for the query.
	Namespace = "namespace"
	// Container specifies the container within a pod to target for log queries.
	Container = "container"
	// TimeStamp specific the time input for a context searching
	TimeStamp = "timestamp"

	// StartTimeDiscription Description for StartTime parameter.
	StartTimeDiscription = "Start time for log filtering"
	// EndTimeDiscription Description for EndTime parameter.
	EndTimeDiscription = "End time for log filtering"
	// FileNameDiscription Description for FileName parameter.
	FileNameDiscription = "Filename for log filtering"
	// TimePeriodDiscription Description for TimePeriod parameter.
	TimePeriodDiscription = "Time period for log filtering"
	// SizesDiscription Description for Sizes parameter.
	SizesDiscription = "Max return instances for log results"
	// KeywordDiscription Description for Keyword parameter.
	KeywordDiscription = "Keyword for log searching"
	// LevelDiscription Description for Level parameter.
	LevelDiscription = "Log level for filtering"
	// PodNameDiscription Description for PodName parameter.
	PodNameDiscription = "Pod name for resources listing"
	// NamespaceDiscription Description for Namespace parameter.
	NamespaceDiscription = "Namespace for resources listing"
	// ContainerDiscription Description for Container parameter.
	ContainerDiscription = "Container name for resources listing"
)
