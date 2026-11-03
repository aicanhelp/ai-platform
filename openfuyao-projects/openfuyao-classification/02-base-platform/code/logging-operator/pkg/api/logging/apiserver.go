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

// Package logging provides a flexible and extensible logging server
// designed to integrate with Kubernetes environments. It offers
// functionalities for capturing, storing, and querying log data
// from various sources within a Kubernetes cluster.
//
// The APIServer struct is central to the logging package, serving
// as the HTTP server for handling log data requests. It integrates
// with the restful framework to offer a RESTful API for log operations,
// including adding, querying, and managing log entries.
//
// Key Features:
//   - RESTful API for log management: Utilizes the go-restful framework
//     to expose a set of HTTP endpoints for log data operations, making
//     it easy to integrate with existing systems and Kubernetes components.
//   - Kubernetes integration: Designed with Kubernetes environments in
//     mind, allowing for seamless log data capture from pods, services,
//     and other Kubernetes objects.
//   - Extensible design: Supports customization and extension to cater
//     to specific logging requirements and integration with external
//     logging solutions or databases.
//   - Context-aware logging: Facilitates capturing and querying logs
//     in a context-rich environment, providing more insightful and
//     actionable log data.
//
// Usage:
// To utilize the logging server, instantiate an APIServer using
// NewAPIServer with a provided server.RunConfig configuration. Once
// instantiated, call the Run method with a context to start serving
// log data requests over HTTP. The APIServer will register the necessary
// API endpoints and start listening on the specified port.
//
// The logging package is built to facilitate advanced logging
// capabilities with a focus on Kubernetes workloads, making it
// a suitable choice for developers looking to implement comprehensive
// logging solutions in their Kubernetes applications.
package logging

import (
	"context"
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"github.com/pkg/errors"
	"k8s.io/apimachinery/pkg/util/runtime"

	"openfuyao/logging-operator/pkg/api/logging/v1"
	"openfuyao/logging-operator/pkg/server"
	"openfuyao/logging-operator/pkg/tools"
)

// APIServer encapsulates an HTTP server and a container for RESTful services,
// facilitating the setup and management of API endpoints.
type APIServer struct {
	Server    *http.Server
	container *restful.Container
}

// NewAPIServer creates and initializes a new instance of APIServer with the
// provided server configuration. It sets up an HTTP server and a RESTful container.
func NewAPIServer(cfg *server.RunConfig) (*APIServer, error) {
	apiServer := &APIServer{}
	httpServer, err := initServer(cfg)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError creating a new ApiServer: %s", err)
		tools.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	apiServer.container = restful.NewContainer()
	apiServer.container.Router(restful.CurlyRouter{})
	apiServer.Server = httpServer
	return apiServer, nil

}

func initServer(cfg *server.RunConfig) (*http.Server, error) {
	httpServer := &http.Server{
		Addr: fmt.Sprintf(":%d", cfg.Config.InsecurePort),
	}
	return httpServer, nil
}

// Run starts the APIServer, registering API endpoints and handling requests.
// It supports graceful shutdown on context cancellation.
func (s *APIServer) Run(ctx context.Context) error {
	s.registerAPI()
	s.Server.Handler = s.container
	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go func() {
		<-ctx.Done()
		err := s.Server.Shutdown(shutdownCtx)
		if err != nil {
			errorMessage := fmt.Sprintf("ApiServer shutdown error: %s", err)
			tools.LogError(errorMessage)
		}
	}()
	out := s.Server.ListenAndServe()
	return out
}

func (s *APIServer) registerAPI() {
	runtime.Must(v1.AddLogsContainer(s.container))
}
