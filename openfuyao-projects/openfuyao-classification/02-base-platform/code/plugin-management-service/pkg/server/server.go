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

package server

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"net/http"
	"os"

	"github.com/emicklei/go-restful/v3"

	pluginv1beta1 "plugin-management-service/pkg/api/consoleplugin/v1beta1"
	"plugin-management-service/pkg/client/k8s"
	"plugin-management-service/pkg/server/config"
	"plugin-management-service/pkg/server/filters"
	"plugin-management-service/pkg/server/runtime"
	"plugin-management-service/pkg/zlog"
)

var newKubernetesClient = k8s.NewKubernetesClient

// CServer including http server config, go-restful container and kubernetes client for connection
type CServer struct {
	// server
	Server *http.Server

	// Container a Web Server（服务器），con WebServices 组成，此外还包含了若干个 Filters（过滤器）、
	container *restful.Container

	// helm用到的k8s client
	KubernetesClient k8s.BaseClient
}

// NewServer creates an cServer instance using given options
func NewServer(cfg *config.RunConfig, ctx context.Context) (*CServer, error) {
	server := &CServer{}

	httpServer, err := initServer(cfg)
	if err != nil {
		return nil, err
	}
	server.Server = httpServer

	server.container = restful.NewContainer()
	server.container.Router(restful.CurlyRouter{})
	server.container.Filter(RecordAccessLogs)
	server.container.Filter(filters.AuthenticateOpenFuyaoUser)

	kubernetesClient, err := newKubernetesClient(cfg.KubernetesCfg)
	if err != nil {
		return nil, err
	}
	server.KubernetesClient = kubernetesClient

	return server, nil
}

func initServer(cfg *config.RunConfig) (*http.Server, error) {
	httpServer := &http.Server{
		Addr: fmt.Sprintf(":%d", cfg.Server.InsecurePort),
	}

	if cfg.Server.SecurePort != 0 {
		certificate, err := tls.LoadX509KeyPair(cfg.Server.CertFile, cfg.Server.PrivateKeyFile)
		if err != nil {
			zlog.Errorf("error loading %s and %s , %v", cfg.Server.CertFile, cfg.Server.PrivateKeyFile, err)
			return nil, err
		}
		// load RootCA
		caCert, err := os.ReadFile(cfg.Server.CAFile)
		if err != nil {
			zlog.Errorf("error read %s, err: %v", cfg.Server.CAFile, err)
			return nil, err
		}

		// create the cert pool
		caCertPool := x509.NewCertPool()
		caCertPool.AppendCertsFromPEM(caCert)

		httpServer.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
			ClientAuth:   tls.VerifyClientCertIfGiven,
			MinVersion:   tls.VersionTLS12,
			ClientCAs:    caCertPool,
		}
		httpServer.Addr = fmt.Sprintf(":%d", cfg.Server.SecurePort)
	}
	return httpServer, nil
}

// Run init consoleplugin-management-service server, bind route, set tls config, etc.
func (s *CServer) Run(ctx context.Context) error {
	var err error = nil
	s.registerAPI()
	s.Server.Handler = s.container

	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
		err = s.Server.Shutdown(shutdownCtx)
	}()

	if s.Server.TLSConfig != nil {
		err = s.Server.ListenAndServeTLS("", "")
	} else {
		err = s.Server.ListenAndServe()
	}
	return err
}

func (s *CServer) registerAPI() {
	pluginWebService := runtime.GetPluginWebService()
	pluginv1beta1.BindPluginRoute(pluginWebService, s.KubernetesClient.ConfigClient())
	s.container.Add(pluginWebService)
}
