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

// Package mdserver provides the API server for the monitoring monitor service.
package mdserver

import (
	"context"
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"github.com/pkg/errors"

	"monitoring-dashboard-service/pkg/config/k8s"
	"monitoring-dashboard-service/pkg/config/runtime"
	"monitoring-dashboard-service/pkg/monitor/controller"
	"monitoring-dashboard-service/pkg/utils/zlog"
)

// APIServer encapsulates an HTTP server and a container for RESTful services,
// facilitating the setup and management of API endpoints.
type APIServer struct {
	// server
	Server *http.Server

	// Container 表示一个 Web Server（服务器），由多个 WebServices 组成，此外还包含了若干个 Filters（过滤器）
	container *restful.Container

	// helm用到的k8s client
	KubernetesClient k8s.Client
}

// NewAPIServer creates a new APIServer instance with the given configuration.
func NewAPIServer(cfg *runtime.RunConfig) (*APIServer, error) {
	apiServer := &APIServer{}
	httpServer, err := initServer(cfg)
	if err != nil {
		errorMessage := fmt.Sprintf("LogError creating a new ApiServer: %s", err)
		zlog.LogError(errorMessage)
		return nil, errors.New(errorMessage)
	}
	apiServer.Server = httpServer

	apiServer.container = restful.NewContainer()
	apiServer.container.Router(restful.CurlyRouter{})

	// 初始化k8s
	kubernetesClient, err := k8s.NewKubernetesClient(cfg.KubernetesCfg)
	if err != nil {
		return nil, err
	}
	apiServer.KubernetesClient = kubernetesClient

	return apiServer, nil

}

// initServer initializes and returns an HTTP server. used in NewAPIServer.
func initServer(cfg *runtime.RunConfig) (*http.Server, error) {
	httpServer := &http.Server{
		Addr: fmt.Sprintf(":%d", cfg.Server.InsecurePort),
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
			zlog.LogError(errorMessage)
		}
	}()
	out := s.Server.ListenAndServe()
	return out
}

// registerAPI registers API endpoints. used in Run.
func (s *APIServer) registerAPI() {
	indicatorWebService := runtime.GetIndicatorWebService()

	controller.BindMonitorRoute(indicatorWebService, s.KubernetesClient.Config())

	s.container.Add(indicatorWebService)

}
