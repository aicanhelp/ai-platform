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

package service

import (
	"context"
	"fmt"
	"net/http"
	"os"

	"github.com/emicklei/go-restful"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog/v2"

	colocationclientset "openfuyao.com/colocation-service/pkg/client/clientset/versioned"
)

// Server colocation-service服务器的结构体
type Server struct {
	// 关闭Server的通道
	StopEverything <-chan struct{}
	Handle         http.Handler
	container      *restful.Container
	ws             *restful.WebService

	client             clientset.Interface
	colocationclient   colocationclientset.Interface
	prometheusEndpoint string

	// logger 必须首先初始化，否则会panic
	logger klog.Logger
}

type serverOptions struct {
	kubeConfig *restclient.Config
}

// Option 配置Server的选项
type Option func(*serverOptions)

// New 创建一个Server实例
func New(ctx context.Context, client clientset.Interface, colocationclient colocationclientset.Interface, opts ...Option) (*Server, error) {
	logger := klog.FromContext(ctx)
	stopEverything := ctx.Done()

	server := &Server{
		client:           client,
		colocationclient: colocationclient,
		StopEverything:   stopEverything,
		logger:           logger,
	}

	server.init()

	return server, nil
}

// init 初始化Server
func (server *Server) init() {
	container := restful.NewContainer()
	ws := new(restful.WebService)

	ws.Path("/rest/colocation/v1").Consumes(restful.MIME_JSON).Produces(restful.MIME_JSON)
	container.Add(ws)

	server.Handle = container
	server.container = container
	server.ws = ws
	server.prometheusEndpoint = getPrometheusEndpoint()

	server.initColocationNodeRouteNew()
	server.initColocationStatusRoute()
}

// Run 启动Server
func (server *Server) Run(ctx context.Context) {
	// 启动informer
	factory := informers.NewSharedInformerFactoryWithOptions(
		server.client, 0, informers.WithNamespace(""),
	)
	cmInformer := factory.Core().V1().ConfigMaps().Informer()

	// 创建 EventHandler 实例
	eventHandler := NewConfigMapEvenHandler(server.client)

	// 注册事件处理器
	cmInformer.AddEventHandler(eventHandler)

	// 启动Informer并同步缓存
	stopCh := make(chan struct{})
	defer close(stopCh)
	go factory.Start(stopCh)

	// 等待缓存同步完成
	if !cache.WaitForCacheSync(stopCh, cmInformer.HasSynced) {
		fmt.Errorf("timed out waiting configmap for caches to sync")
		return
	}

	<-ctx.Done()
}

func getPrometheusEndpoint() string {
	// 使用默认的普罗服务
	prometheusService := os.Getenv("PROMETHEUS_SERVICE")
	if prometheusService == "" {
		klog.Error("PROMETHEUS_SERVICE is not provided")
		return prometheusService
	}

	prometheusURL := fmt.Sprintf("%s%s", prometheusService, "/api/v1/query_range")

	return prometheusURL
}
