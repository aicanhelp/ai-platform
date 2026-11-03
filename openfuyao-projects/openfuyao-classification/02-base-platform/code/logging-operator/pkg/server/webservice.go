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

import (
	"strings"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	// ApiRootPath defines the api path
	ApiRootPath = "/rest"
)

// NewWebService creates a new RESTful web service for the specified API group and version.
// The web service is configured to serve JSON content and is rooted at a path derived
// from the provided GroupVersion information and the ApiRootPath.
func NewWebService(gv schema.GroupVersion) *restful.WebService {
	webservice := restful.WebService{}
	webservice.Path(strings.TrimRight(ApiRootPath+"/"+gv.String(), "/")).
		Produces(restful.MIME_JSON)

	return &webservice
}

// GetQueryParamOrDefault extracts the value of a specified query parameter from a request.
// If the parameter is not present or its value is empty, a provided default value is returned.
func GetQueryParamOrDefault(r *restful.Request, param, defaultValue string) string {
	value := r.Request.URL.Query().Get(param)
	if value == "" {
		return defaultValue
	}
	return value
}
