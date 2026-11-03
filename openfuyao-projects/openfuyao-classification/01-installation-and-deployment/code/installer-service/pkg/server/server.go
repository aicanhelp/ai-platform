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
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"k8s.io/apimachinery/pkg/runtime/schema"
	urlruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"

	"installer-service/cmd/config"
	"installer-service/pkg/api/clustermanage"
	"installer-service/pkg/client/k8s"
	"installer-service/pkg/constant"
	"installer-service/pkg/server/filters"
	"installer-service/pkg/server/request"
)

// CServer apiserver的组件
type CServer struct {
	// server
	Server *http.Server

	// Container 表示一个 Web Server（服务器），由多个 WebServices 组成，此外还包含了若干个 Filters（过滤器）、
	container *restful.Container

	// k8s client
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
		certificate, err := tls.LoadX509KeyPair(cfg.Server.CertFile, cfg.Server.PrivateKey)
		if err != nil {
			return nil, err
		}

		httpServer.TLSConfig = &tls.Config{
			Certificates: []tls.Certificate{certificate},
		}
		httpServer.Addr = fmt.Sprintf(":%d", cfg.Server.SecurePort)
	}
	return httpServer, nil
}

// Run 注册api,开始监听端口
func (s *CServer) Run(ctx context.Context) error {
	// 向 container 注册 api
	s.registerAPI()
	s.registerAPIWs()
	// apiServer.cServer.handler 绑定了一个 container
	s.Server.Handler = s.container

	s.buildHandlerChain()

	shutdownCtx, cancel := context.WithCancel(context.Background())
	defer cancel()

	go func() {
		<-ctx.Done()
		err := s.Server.Shutdown(shutdownCtx)
		if err != nil {
			return
		}
	}()

	if s.Server.TLSConfig != nil {
		return s.Server.ListenAndServeTLS("", "")
	} else {
		return s.Server.ListenAndServe()
	}
}

func (s *CServer) registerAPI() {
	urlruntime.Must(clustermanage.ConfigInstaller(s.container, s.KubernetesClient.Config()))
}

func (s *CServer) registerAPIWs() {
	urlruntime.Must(clustermanage.ConfigInstallerWs(s.container, s.KubernetesClient.Config()))
}

const (
	groupName = "resources.fuyao.io"
	version   = "v1alpha1"
)

var groupVersion = schema.GroupVersion{Group: groupName, Version: version}

func resource(resource string) schema.GroupResource {
	return groupVersion.WithResource(resource).GroupResource()
}

// 验证和路由分发，handler是先注册的后调用
func (s *CServer) buildHandlerChain() {
	handler := s.Server.Handler

	// 代理到kube-apiServer
	handler = filters.ProxyAPIServer(handler, s.KubernetesClient.Config())

	// 多集群处理，此处仅仅是去掉前缀，后续待处理多集群能力
	handler = filters.DispatchCluster(handler)

	// 定义API前缀和资源组信息，在fliter中会过滤校验相关前缀和分组路由。目前版本没有实现分组，仅仅是
	requestInfoResolver := &request.RequestInfoFactory{
		RequestInfoFactory: &k8srequest.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
		GlobalResources: []schema.GroupResource{
			resource(constant.ResourcesPluralCluster),
		},
	}

	handler = filters.BuildRequestInfo(handler, requestInfoResolver)

	s.Server.Handler = handler
}
