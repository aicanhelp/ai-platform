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

// Config holds the configuration for a server instance, including
// network bindings and port settings.
type Config struct {
	// server bind address
	BindAddress string

	// insecure port number
	InsecurePort int

	// secure port number
	SecurePort int
}

// RunConfig encapsulates the runtime configuration of the server,
// potentially including other settings beyond the basic server configuration.
type RunConfig struct {
	Config *Config
}

// NewServer creates and initializes a Server struct with default settings:
// listens on all network interfaces, binds to port 9197 for HTTP, and disables HTTPS.
func NewServer() *Config {
	// create default server run options
	s := Config{
		BindAddress:  "0.0.0.0",
		InsecurePort: 9197,
		SecurePort:   0,
	}

	return &s
}

// NewRunConfig creates a RunConfig struct with a default Server configuration.
func NewRunConfig() *RunConfig {
	return &RunConfig{
		Config: NewServer(),
	}
}

// validate的模块之后需要添加在里面
