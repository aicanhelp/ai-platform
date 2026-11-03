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

// Package v1 provides the necessary tools and utilities to interact with mcs api service.
// It is responsible for creating API clients, API servers, registering endpoints and handling
// requests and reponses efficiently.
package v1

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/util/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/client/k8s"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/config"
	"openfuyao.com/multi-cluster-service/pkg/apis/multicluster/v1/filters"
	"openfuyao.com/multi-cluster-service/pkg/zlog"
)

const (
	// watchUserManagerTimeout is the timeout for connecting UserManager service.
	watchUserManagerTimeout = 2 * time.Minute

	// watchUserManagerInterval is the interval at which are polled for changes.
	watchUserManagerInterval = 5 * time.Second

	// waitUserManagerService is the waiting time for connecting UserManager service.
	waitUserManagerService = 20 * time.Second
)

// APIServer defines the structure for an API server that includes an HTTP server,
// a client for interacting with Kubernetes, a client for interacting with controller manager
// and a restful API web server.
type APIServer struct {
	// server
	Server *http.Server

	// Container 表示一个 Web Server（服务器），由多个 WebServices 组成，此外还包含了若干个 Filters（过滤器）、
	container *restful.Container

	// helm用到的k8s client
	KubernetesClient k8s.Client

	// 控制器用到的client
	MgrClient client.Client

	// http.client
	ApiClient *APIClient
}

// NewServer creates an cServer instance using given options
func NewServer(cfg *config.RunConfig, ctx context.Context, client client.Client) (*APIServer, error) {
	server := &APIServer{
		MgrClient: client,
		ApiClient: NewAPIClient(),
	}

	httpServer, err := initServer(cfg)
	if err != nil {
		return nil, err
	}
	server.Server = httpServer

	// 初始化 Container
	server.container = restful.NewContainer()
	server.container.Router(restful.CurlyRouter{})
	server.container.Filter(filters.RecordAccessLogs)
	server.container.Filter(filters.ExactSubjectAccess)

	// 初始化client和informers
	kubernetesClient, err := k8s.NewKubernetesClient(cfg.KubernetesCfg)
	if err != nil {
		return nil, err
	}
	server.KubernetesClient = kubernetesClient

	return server, nil
}

func initServer(cfg *config.RunConfig) (*http.Server, error) {
	// 初始化 cServer
	httpServer := &http.Server{
		Addr: fmt.Sprintf(":%d", cfg.Server.InsecurePort),
	}
	// https 证书配置
	if cfg.Server.SecurePort != 0 {
		certificate, err := tls.LoadX509KeyPair(cfg.Server.CertFile, cfg.Server.PrivateKeyFile)
		if err != nil {
			zlog.LogErrorf("error loading %s and %s , %v", cfg.Server.CertFile, cfg.Server.PrivateKeyFile, err)
			return nil, err
		}
		// load RootCA
		caCert, err := os.ReadFile(cfg.Server.CAFile)
		if err != nil {
			zlog.LogErrorf("error read %s, err: %v", cfg.Server.CAFile, err)
			return nil, err
		}

		// create the cert pool
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)

		httpServer.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
			ClientAuth:   tls.RequireAndVerifyClientCert,
			MinVersion:   tls.VersionTLS12,
			ClientCAs:    caCertPool,
		}
		httpServer.Addr = fmt.Sprintf(":%d", cfg.Server.SecurePort)
	}
	return httpServer, nil
}

// Run is the implementation of APIServer.
func (s *APIServer) Run(ctx context.Context) error {
	// 向 container 注册 api
	s.registerAPI()
	// apiServer.cServer.handler 绑定了一个 container
	s.Server.Handler = s.container
	// 安全相关响应头
	s.Server.Handler = addSecurityHeader(s.Server.Handler)

	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
		if err := s.Server.Shutdown(shutdownCtx); err != nil {
			zlog.LogWarn("Error shutting down server: ", err)
		} else {
			zlog.LogInfo("Server shutdown successfully")
		}
	}()

	if s.Server.TLSConfig != nil {
		return s.Server.ListenAndServeTLS("", "")
	}
	return s.Server.ListenAndServe()
}

func (s *APIServer) registerAPI() {
	runtime.Must(AddToContainer(s.container, s.MgrClient))
}

func addSecurityHeader(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		csp := "connect-src 'self' https:;frame-ancestors 'none';object-src 'none'"
		w.Header().Set("Content-Security-Policy", csp)
		w.Header().Set("Cache-Control", "no-cache, no-store, must-revalidate")
		w.Header().Set("X-Frame-Options", "DENY")
		w.Header().Set("X-Content-Type-Options", "nosniff")
		w.Header().Set("X-XSS-Protection", "1")
		w.Header().Set("Strict-Transport-Security", "max-age=31536000")
		next.ServeHTTP(w, r)
	})
}

func (s *APIServer) sendHellowithRetry() {
	helloURL := fmt.Sprintf("%s%s", UserManagerEndpoint, UserManagerSignalUrl)
	helloMsg := "HELLO HERE IS Multi Cluster Service"
	for {
		_, err := s.ApiClient.Post(helloURL, helloMsg, "")
		if err != nil {
			zlog.Warnf("Waiting Sending Hello to User-Manager: %v: wait for service to become ready", err)
			time.Sleep(watchUserManagerInterval)
			continue
		} else {
			zlog.LogInfof("Successfully Sending HELLO to USER-MANAGEMENT!")
			break
		}
	}
}

// StartAPIServer initializes and starts an API server using the provided configurations.
func StartAPIServer(client client.Client) {
	runOptions := config.NewRunConfig()
	// 校验server和k8s配置
	if errs := runOptions.Validate(); len(errs) != 0 {
		zlog.LogFatalf("Failed to Validate RunConfig: %v", errs)
	}

	ctx := context.TODO()
	apiServer, err := NewServer(runOptions, ctx, client)
	if err != nil {
		zlog.LogFatalf("Failed to Init Multi-Cluster API Service : %v", err)
	}

	go func() {
		err = apiServer.Run(ctx)
		if err != nil {
			zlog.LogFatalf("Failed to Run APIServer: %v", err)
		}
	}()

	time.Sleep(waitUserManagerService)
	apiServer.sendHellowithRetry()
}
