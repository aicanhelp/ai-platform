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

	"github.com/emicklei/go-restful/v3"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/client/k8s"
	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/runtime"
	"openfuyao.com/web-terminal-service/pkg/zlog"
)

const (
	// KeyOpenApiTags is a API tag
	KeyOpenApiTags = "openapi.tags"
	// TagTerminal is a tag
	TagTerminal = "Web Terminal"
)

// NewClientandConfig reads the kubeconfig file and returns a rest.Config and a kubernetes.Clientset.
func NewClientandConfig() (*rest.Config, kubernetes.Interface) {
	config := k8s.GetKubeConfig()
	clientset, err := kubernetes.NewForConfig(config)
	if err != nil {
		zlog.LogWarn("LogError creating client")
	}
	return config, clientset
}

// AddToContainer initializes and adds routes to a RESTful container for mcs API service.
func AddToContainer(container *restful.Container, client client.Client) error {
	ws := runtime.NewWebService()
	k8sconfig, k8sclient := NewClientandConfig()
	handler := NewHandler(k8sclient, k8sconfig, client)
	handler.ApiClient = NewAPIClient()
	handler.MgrClient = client

	// 调用接口注册
	sayHello(ws, handler)

	terminalPod(ws, handler)
	terminalCluster(ws, handler)

	container.Add(ws)
	return nil
}

func sayHello(ws *restful.WebService, h *Handler) {
	ws.Route(ws.GET("/hello").To(h.sayHello))
}

// 创建容器命令行的交互接口
func terminalPod(ws *restful.WebService, h *Handler) {
	ws.Route(ws.GET("/namespace/{namespace}/pod/{pod}/container/{container}/terminal").
		To(h.HandlePodTerminal).
		Doc("Create Pod Terminal").
		Metadata(KeyOpenApiTags, []string{TagTerminal}).
		Operation("create-pod-exec").
		Param(ws.PathParameter("namespace", "Namespace")).
		Param(ws.PathParameter("pod", "pod")))
}

// 创建集群命令行的交互接口
func terminalCluster(ws *restful.WebService, h *Handler) {
	ws.Route(ws.GET("user/{user}/terminal").
		To(func(req *restful.Request, resp *restful.Response) {
			username := req.PathParameter("user")
			ctx := context.WithValue(req.Request.Context(), "username", username)
			ctx = context.WithValue(ctx, "path", req.Request.URL.Path)
			h.HandleClusterTerminal(req, resp, ctx)
		}).
		Doc("Create Web Terminal Template").
		Metadata(KeyOpenApiTags, []string{TagTerminal}).
		Param(ws.PathParameter("user", "username")).
		Operation("create-web-terminal-template"))
}
