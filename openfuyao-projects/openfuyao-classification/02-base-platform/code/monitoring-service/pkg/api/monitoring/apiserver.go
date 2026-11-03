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

// Package monitoring provides the core functionality for initializing and managing an API server
// that interfaces with a Prometheus monitoring backend. This package is designed to configure,
// launch, and manage the lifecycle of a web server that serves the monitoring API, facilitating
// interactions between client requests and Prometheus data.
package monitoring

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/util/runtime"

	"openfuyao.com/monitoring-service/pkg/api/monitoring/v1"
	"openfuyao.com/monitoring-service/pkg/client"
	"openfuyao.com/monitoring-service/pkg/client/prometheus"
	"openfuyao.com/monitoring-service/pkg/cmd/config"
	"openfuyao.com/monitoring-service/pkg/server/filters"
	"openfuyao.com/monitoring-service/pkg/zlog"
)

// APIServer defines a server structure for handling API services.
type APIServer struct {
	Server           *http.Server
	container        *restful.Container
	MonitoringClient client.Interface
}

// NewAPIServer creates an APIServer instance using the provided configuration and context.
// It initializes a new HTTP server and sets up a container for managing web services and filters.
// Returns an error if there is a failure in creating the HTTP server or connecting to Prometheus.
func NewAPIServer(cfg *config.RunConfig, ctx context.Context) (*APIServer, error) {
	apiServer := &APIServer{}

	err := readFromConfig(cfg, "/config.json")
	if err != nil {
		return nil, err
	}

	httpServer, err := initServer(cfg)
	if err != nil {
		return nil, err
	}
	apiServer.Server = httpServer

	apiServer.container = restful.NewContainer()
	apiServer.container.Router(restful.CurlyRouter{})
	apiServer.container.Filter(filters.RecordAccessLogs)

	if apiServer.MonitoringClient, err = prometheus.NewPrometheus(cfg.MonitoringOptions); err != nil {
		return nil, fmt.Errorf("failed to connect to prometheus, please check prometheus status, error: %v", err)
	}

	return apiServer, nil
}

func readFromConfig(cfg *config.RunConfig, filePath string) error {
	data, err := ioutil.ReadFile(filePath)
	if err != nil {
		return fmt.Errorf("failed to read config file: %v", err)
	}

	if err = json.Unmarshal(data, cfg); err != nil {
		return fmt.Errorf("failed to parse config JSON: %v", err)
	}

	return nil
}

func initServer(cfg *config.RunConfig) (*http.Server, error) {
	httpServer := &http.Server{
		Addr: fmt.Sprintf(":%d", cfg.Server.InsecurePort),
	}

	if cfg.Server.SecurePort != 0 {
		certificate, err := tls.LoadX509KeyPair(cfg.Server.CertFile,
			cfg.Server.PrivateKeyFile)
		if err != nil {
			return nil, fmt.Errorf("failed to load TLS certificate and private key, error: %v", err)
		}

		caCert, err := os.ReadFile(cfg.Server.RootCAFile)
		if err != nil {
			return nil, fmt.Errorf("failed to read Root CA file, error: %v", err)
		}

		caCertPool := x509.NewCertPool()
		if !caCertPool.AppendCertsFromPEM(caCert) {
			return nil, fmt.Errorf("failed to add Root CA certificate to the certificate pool")
		}

		httpServer.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
			ClientAuth:   tls.VerifyClientCertIfGiven,
			MinVersion:   tls.VersionTLS12,
			MaxVersion:   tls.VersionTLS13,
			ClientCAs:    caCertPool,
		}
		httpServer.Addr = fmt.Sprintf(":%d", cfg.Server.SecurePort)
	}

	return httpServer, nil
}

// Run starts the API server using the configuration and context provided during server creation.
// It registers the server's API endpoints, binds the server handler to the container, and handles graceful shutdown.
// Returns an error if the server fails to start or encounters an error during runtime.
func (s *APIServer) Run(ctx context.Context) error {
	s.registerAPI()
	// apiServer.Server.handler 绑定了一个 container, 添加了各种拦截器
	s.Server.Handler = s.container

	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
		if err := s.Server.Shutdown(shutdownCtx); err != nil {
			zlog.Warn("Error shutting down server: ", err)
		} else {
			zlog.Info("Server shutdown successfully")
		}
	}()

	zlog.FormatInfo("Start listening on %s", s.Server.Addr)
	if s.Server.TLSConfig != nil {
		return s.Server.ListenAndServeTLS("", "")
	}
	return s.Server.ListenAndServe()
}

func (s *APIServer) registerAPI() {
	runtime.Must(v1.AddToContainer(s.container, s.MonitoringClient))
}
