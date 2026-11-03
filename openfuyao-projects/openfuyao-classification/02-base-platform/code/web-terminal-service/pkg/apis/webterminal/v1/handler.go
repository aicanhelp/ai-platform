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

// Package v1 provides the necessary tools and utilities to interact with wts api service.
// It is responsible for creating API clients, API servers, registering endpoints and handling
// requests and reponses efficiently.
package v1

import (
	"context"
	"fmt"
	"net/http"

	"github.com/emicklei/go-restful/v3"
	"github.com/gorilla/websocket"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/responsehandlers"
	"openfuyao.com/web-terminal-service/pkg/webterminal"
	"openfuyao.com/web-terminal-service/pkg/zlog"
)

var upgrade = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

// Handler centralizes the clients used to interface with different kubernetes APIs.
type Handler struct {
	// http.client
	client    kubernetes.Interface
	config    *rest.Config
	ApiClient *APIClient
	// MgrClient is the controller client to interact with CRD, not specific to any kubernetes version.
	MgrClient client.Client
	terminal  webterminal.HandleInterface
}

// NewHandler defines a new handler structure.
func NewHandler(client kubernetes.Interface, config *rest.Config, mgrclient client.Client) *Handler {
	return &Handler{
		client:   client,
		config:   config,
		terminal: webterminal.NewTerminal(client, config, mgrclient),
	}
}

func (h *Handler) sayHello(req *restful.Request, resp *restful.Response) {
	responsehandlers.SendStatusOk(resp, "Hello, The WTS API Server Working Successfully !")
}

// HandlePodTerminal 是Pod terminal 的 handler 的方法
func (h *Handler) HandlePodTerminal(req *restful.Request, resp *restful.Response) {
	namespace := req.PathParameter("namespace")
	podName := req.PathParameter("pod")
	containerName := req.PathParameter("container")

	// 更新 websocket
	ctx := req.Request.Context()
	ctx = context.WithValue(ctx, "path", req.Request.URL.Path)

	conn, err := upgrade.Upgrade(resp.ResponseWriter, req.Request, nil)
	if err != nil {
		zlog.LogWarn(err)
		return
	}
	// 用户权限校验
	permission, err := checkUserAccess(h.client, ctx, req, conn)
	if !permission {
		fmt.Println("User has no access:", err)
		return
	}
	// 调用 HandleTerminal 方法
	h.terminal.HandleTerminal(ctx, namespace, podName, containerName, conn)

}

// HandleClusterTerminal 是 user pod 的方法
func (h *Handler) HandleClusterTerminal(req *restful.Request, resp *restful.Response, ctx context.Context) {
	username, status := ctx.Value("username").(string)
	if !status {
		resp.WriteErrorString(http.StatusUnauthorized, "username not found in context")
		return
	}

	conn, Err := upgrade.Upgrade(resp.ResponseWriter, req.Request, nil)
	if Err != nil {
		zlog.LogWarnf("Failed to upgrade WebSocket: %v", Err)
		return
	}
	permission, err := checkUserAccess(h.client, ctx, req, conn)
	if !permission {
		fmt.Println("User has no access:", err)
		return
	}
	h.terminal.HandleCusterTerminal(ctx, username, conn)

}

func checkUserAccess(c kubernetes.Interface, ctx context.Context, req *restful.Request, conn *websocket.Conn) (bool, error) {
	// 从上下文中获取用户名
	username, ok := req.Request.Context().Value("user").(string)
	if !ok {
		zlog.LogErrorf("LogError retrieving user information")
		sendWebSocketError(conn, "LogError retrieving user information")
		return false, fmt.Errorf("error retrieving user information")
	}
	zlog.LogInfof("Retrieving user -- %s -- info", username)

	pAdminName := fmt.Sprintf("%s-%s", username, "platform-admin")
	cAdminName := fmt.Sprintf("%s-%s", username, "cluster-admin")

	clusterroleList, err := c.RbacV1().ClusterRoleBindings().List(ctx, metav1.ListOptions{})
	if err != nil {
		zlog.LogErrorf("LogError retrieving clusterrolebinding: %v", err)
		sendWebSocketError(conn, "LogError retrieving cluster role bindings")
		return false, err
	}

	for _, item := range clusterroleList.Items {
		if item.Name == pAdminName || item.Name == cAdminName {
			zlog.LogInfof("User %s has access via role %s", username, item.Name)
			sendWebSocketMessage(conn, "User has access")
			return true, nil
		}
	}

	zlog.LogInfof("User %s has no access", username)
	sendWebSocketError(conn, "User has no access")
	return false, nil
}

func sendWebSocketError(conn *websocket.Conn, errorMessage string) {
	if conn != nil {
		err := conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("LogError: %s", errorMessage)))
		if err != nil {
			zlog.LogErrorf("Failed to send WebSocket error message: %v", err)
		}
	}
}

func sendWebSocketMessage(conn *websocket.Conn, message string) {
	if conn != nil {
		err := conn.WriteMessage(websocket.TextMessage, []byte(message))
		if err != nil {
			zlog.LogErrorf("Failed to send WebSocket message: %v", err)
		}
	}
}
