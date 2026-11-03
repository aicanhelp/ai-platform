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
	"net"
	"net/http"
	"os"

	"github.com/emicklei/go-restful/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"

	"marketplace-service/cmd/config"
	helmv1 "marketplace-service/pkg/api/marketplace/v1beta1"
	"marketplace-service/pkg/client/k8s"
	"marketplace-service/pkg/constant"
	"marketplace-service/pkg/rpc"
	pb "marketplace-service/pkg/rpc/helmchart"
	"marketplace-service/pkg/server/runtime"
	"marketplace-service/pkg/utils/httputil"
	"marketplace-service/pkg/zlog"
)

// CServer including http server config, go-restful container and kubernetes client for connection
type CServer struct {
	// server
	Server *http.Server

	// Container a Web Server（服务器），con WebServices 组成，此外还包含了若干个 Filters（过滤器）、
	container *restful.Container

	// helm用到的k8s client
	KubernetesClient k8s.Client
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

	kubernetesClient, err := k8s.NewKubernetesClient(cfg.KubernetesCfg)
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
		certificate, err := tls.LoadX509KeyPair(cfg.Server.CertFile, cfg.Server.PrivateKey)
		if err != nil {
			zlog.Errorf("error loading %s and %s , %v", cfg.Server.CertFile, cfg.Server.PrivateKey, err)
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

// Run init marketplace-service server, bind route, set tls config, etc.
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

func initRPCServer(handler *helmv1.Handler) {
	lis, err := net.Listen("tcp", ":9038")
	if err != nil {
		zlog.Fatalf("failed to listen: %v", err)
	}
	s := getGrpcServer()

	pb.RegisterChartManagerServer(s, &rpc.Server{
		Handler: handler,
	})
	zlog.Infof("grpc listening at %v", lis.Addr())
	go func() {
		if err := s.Serve(lis); err != nil {
			zlog.Fatalf("failed to serve: %v", err)
		}
	}()
}

func getGrpcServer() *grpc.Server {
	var enableTLS bool
	enableTLS, err := httputil.IsHttpsEnabled(constant.TLSCertPath, constant.TLSKeyPath, constant.CAPath)
	if err != nil {
		enableTLS = false
	}
	tlsCfg, err := httputil.GetHttpConfig(constant.TLSCertPath, constant.TLSKeyPath, constant.CAPath, enableTLS)
	if err != nil {
		enableTLS = false
	}

	if !enableTLS {
		s := grpc.NewServer()
		return s
	}
	zlog.Info("use grpc tls successfully")
	creds := credentials.NewTLS(tlsCfg)
	s := grpc.NewServer(grpc.Creds(creds))
	return s
}

func (s *CServer) registerAPI() {
	marketplaceServiceWebService := runtime.GetMarketplaceWebService()
	handler := helmv1.BindMarketPlaceRoute(marketplaceServiceWebService, s.KubernetesClient.Config())
	initRPCServer(handler)
	s.container.Add(marketplaceServiceWebService)
}
