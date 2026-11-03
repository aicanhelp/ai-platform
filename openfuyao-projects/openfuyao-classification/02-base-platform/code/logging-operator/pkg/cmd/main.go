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

// Package main serves as the entry point for a logging server application designed
// to provide a robust and scalable solution for managing and querying logs within
// a distributed system, particularly tailored for Kubernetes environments. This package
// leverages the internal logging API and server configurations to instantiate and run
// a specialized logging server capable of interfacing with Loki for log aggregation and
// query functionalities.
//
// The application initializes with a predefined set of run configurations, creating an
// API server that facilitates interaction between the client-side logging requests and
// the backend log storage system (Loki). It aims to abstract the complexities involved
// in direct log management and queries, offering a simplified and unified API endpoint
// for log operations.
//
// Core Functionalities:
//   - Initializes the logging server with configurable run options, allowing for
//     customization of server behavior such as port bindings, authentication mechanisms,
//     and more.
//   - Sets up an API server that acts as the middleware between the client applications
//     and the underlying log storage, ensuring efficient log data retrieval and management.
//   - Provides a context-aware execution environment, supporting graceful shutdown and
//     runtime configuration adjustments in response to operational or environmental changes.
//
// Usage:
// To deploy the logging server, compile and run the main package. The server will start
// listening for incoming HTTP requests based on the configurations specified in the
// server.RunConfig struct. Clients can interact with the server through its RESTful API
// to perform log-related operations such as querying logs, fetching log sources, and
// filtering logs based on various criteria.
//
// This command compiles the application and starts the logging server, making it ready
// to accept logging requests. The actual API endpoints and functionalities are defined
// within the internal `api` and `server` packages, which this main package orchestrates
// to provide a cohesive logging service.
//
// Note:
// This package assumes the presence of a configuration mechanism (via `server.NewRunConfig`)
// to initialize the server with necessary settings. Additionally, it relies on the successful
// creation and execution of an API server instance (`api.NewAPIServer`) that handles all
// logging-specific logic and interactions with Loki.
//
// It is designed with extensibility and scalability in mind, facilitating future enhancements
// such as support for multiple backend log storage solutions, advanced query capabilities,
// and integration with Kubernetes for log source discovery and management.
package main

import (
	"context"
	"fmt"

	"openfuyao/logging-operator/pkg/api/logging"
	"openfuyao/logging-operator/pkg/server"
	"openfuyao/logging-operator/pkg/tools"
)

func main() {
	runOptions := server.NewRunConfig()
	ctx := context.TODO()
	apiServer, err := logging.NewAPIServer(runOptions)

	if err != nil {
		errorMessage := fmt.Sprintf("LogError build apiServer: %s", err)
		tools.LogError(errorMessage)
	}
	err = apiServer.Run(ctx)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError running apiServer: %s", err)
		tools.LogError(errorMessage)
	}
}
