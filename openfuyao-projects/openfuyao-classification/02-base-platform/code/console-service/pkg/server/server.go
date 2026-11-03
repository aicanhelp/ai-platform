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
	"net"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/emicklei/go-restful/v3"
	urlruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"

	"console-service/cmd/config"
	"console-service/pkg/api/utility/v1beta1"
	"console-service/pkg/auth"
	"console-service/pkg/client/k8s"
	"console-service/pkg/server/filters"
	"console-service/pkg/server/request"
	"console-service/pkg/server/runtime"
	"console-service/pkg/zlog"
)

// CServer including http server config, go-restful container and kubernetes client for connection
type CServer struct {
	// server
	Server *http.Server

	// Container 表示一个 Web Server（服务器），由多个 WebServices 组成，此外还包含了若干个 Filters（过滤器）、
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

	// 初始化 Container
	server.container = restful.NewContainer()
	server.container.Router(restful.CurlyRouter{})
	server.container.Filter(filters.RecordAccessLogs)

	// 初始化client和informers
	kubernetesClient, err := k8s.NewKubernetesClient(cfg.KubernetesCfg)
	if err != nil {
		return nil, err
	}
	server.KubernetesClient = kubernetesClient

	return server, nil
}

func initServer(cfg *config.RunConfig) (*http.Server, error) {
	// 初始化 cServer；JoinHostPort 正确处理 IPv6 字面量（如 [::]:9037）
	httpServer := &http.Server{
		Addr:              net.JoinHostPort(cfg.Server.BindAddress, strconv.Itoa(cfg.Server.InsecurePort)),
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      60 * time.Second,
		IdleTimeout:       60 * time.Second,
		ReadHeaderTimeout: 120 * time.Second,
	}
	// https 证书配置
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
			ClientAuth:   tls.RequestClientCert,
			MinVersion:   tls.VersionTLS12,
			ClientCAs:    caCertPool,
		}
		httpServer.Addr = net.JoinHostPort(cfg.Server.BindAddress, strconv.Itoa(cfg.Server.SecurePort))
	}
	return httpServer, nil
}

// Run init console-service server, bind route, set tls config, etc.
func (s *CServer) Run(ctx context.Context) error {
	var err error = nil
	// 向 container 注册 api
	s.registerAPI()
	// apiServer.cServer.handler 绑定了一个 container
	s.Server.Handler = s.container
	// 添加各个调用链的拦截器, 用于验证和路由分发
	s.buildHandlerChain()
	// 安全相关响应头
	s.Server.Handler = addSecurityHeader(s.Server.Handler)

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
	consoleWebService := runtime.GetConsoleWebService()
	v1beta1.BindUtilityWebService(consoleWebService)
	s.container.Add(consoleWebService)

	urlruntime.Must(auth.AddToContainer(s.container, s.KubernetesClient.Config()))
}

// 验证和路由分发，handler是先注册的后调用
// handlers are FIRST declared, LAST called
func (s *CServer) buildHandlerChain() {
	serverHandler := s.Server.Handler

	// proxy static resource requests console-website
	staticPageHandler := filters.ProxyStatic(serverHandler, s.KubernetesClient.Config())

	// proxy console plugin requests (backend & frontend)
	pluginHandler := filters.ProxyConsolePlugin(staticPageHandler, s.KubernetesClient.Config())

	// proxy request to other fuyao components
	componentAPIHandler := filters.ProxyComponentRequest(serverHandler, pluginHandler, s.KubernetesClient)

	// 代理到kube-apiServer
	kubeAPIHandler := filters.ProxyAPIServer(componentAPIHandler, s.KubernetesClient)

	// 定义API前缀和资源组信息，在filter中会过滤校验相关前缀和分组路由。目前版本没有实现分组，仅仅是
	requestInfoResolver := &request.RequestInfoFactory{
		RequestInfoFactory: &k8srequest.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
	}
	// 到这一步 /clusters/{cluster}/api/kubernetes 或者 /clusters/{cluster}/rest 两种前缀
	buildRequestInfoHandler := filters.BuildRequestInfo(kubeAPIHandler, requestInfoResolver, s.KubernetesClient.Config())

	// 这个filter只为了time-offset接口
	consoleAPIHandler := filters.HandleConsoleRequests(serverHandler, buildRequestInfoHandler)

	authReqCheckHandler := filters.CheckRequestAuth(serverHandler, consoleAPIHandler, s.KubernetesClient.Config())

	localeHandler := filters.WithLocale(authReqCheckHandler)
	s.Server.Handler = localeHandler
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
