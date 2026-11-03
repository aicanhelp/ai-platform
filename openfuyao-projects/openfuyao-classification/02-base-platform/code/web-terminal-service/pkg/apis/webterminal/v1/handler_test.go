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
package v1

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/gorilla/websocket"
	"github.com/stretchr/testify/assert"
	k8sv1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	fake2 "k8s.io/client-go/kubernetes/typed/rbac/v1/fake"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	faker "sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/web-terminal-service/api/v1beta1"
	"openfuyao.com/web-terminal-service/pkg/webterminal"
)

func FakeClient() client.Client {
	Scheme := runtime.NewScheme()
	v1beta1.AddToScheme(Scheme)
	k8sv1.AddToScheme(Scheme)
	fakeMgrClient := faker.NewClientBuilder().WithScheme(Scheme).Build()
	return fakeMgrClient
}

func TestCheckUserAccess(t *testing.T) {
	fakeClient := fake.NewSimpleClientset()

	// 使用 monkey.Patch 替换 ClusterRoleBindings().List
	patch := gomonkey.ApplyMethod(
		reflect.TypeOf(fakeClient.RbacV1().ClusterRoleBindings()),
		"List",
		func(_ *fake2.FakeClusterRoleBindings, ctx context.Context,
			opts v1.ListOptions) (*rbacv1.ClusterRoleBindingList, error) {
			return &rbacv1.ClusterRoleBindingList{
				Items: []rbacv1.ClusterRoleBinding{
					{
						ObjectMeta: v1.ObjectMeta{Name: "test-user-platform-admin"},
					},
					{
						ObjectMeta: v1.ObjectMeta{Name: "non-matching-role"},
					},
				},
			}, nil
		},
	)
	defer patch.Reset()

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upgrader := websocket.Upgrader{}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			fmt.Println(err)
		}
		req := &restful.Request{
			Request: r.WithContext(context.WithValue(r.Context(), "user", "test-user")),
		}
		// 调用被测试函数
		allowed, err := checkUserAccess(fakeClient, context.TODO(), req, conn)
		if allowed != true {
			fmt.Println("not ok")
		}
	}))
	// 创建 WebSocket 客户端连接
	dialer := websocket.DefaultDialer
	_, _, err := dialer.Dial(fmt.Sprintf("ws://%s", server.Listener.Addr().String()), nil)
	fmt.Println(err)
	defer server.Close()
}

func TestNewHandler(t *testing.T) {
	type args struct {
		client    kubernetes.Interface
		config    *rest.Config
		mgrclient client.Client
	}
	client := fake.NewSimpleClientset()
	config := &rest.Config{}
	mgrclient := FakeClient()

	tests := []struct {
		name string
		args args
		want *Handler
	}{
		{
			name: "ok",
			args: args{
				client:    client,
				config:    config,
				mgrclient: mgrclient,
			},
			want: &Handler{
				client:   client,
				config:   config,
				terminal: webterminal.NewTerminal(client, config, mgrclient),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewHandler(tt.args.client, tt.args.config, tt.args.mgrclient); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewHandler() = %v, want %v", got, tt.want)
			}
		})
	}
}

func setupWebSocketTest(handlerFunc func(req *restful.Request,
	resp *restful.Response)) (*httptest.Server, string) {
	upgrader := websocket.Upgrader{}

	// 创建模拟的 HTTP 服务器
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// WebSocket 升级
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			http.Error(w, "Failed to upgrade WebSocket", http.StatusBadRequest)
			return
		}
		defer conn.Close()

		// 模拟 RESTful 请求
		req := &restful.Request{
			Request: r.WithContext(context.WithValue(r.Context(), "user", "test-user")),
		}
		resp := &restful.Response{
			ResponseWriter: w,
		}

		// 调用传入的 handlerFunc
		handlerFunc(req, resp)
	}))

	url := fmt.Sprintf("ws://%s/namespaces/default/pods/test-pod/containers/test-container/terminal",
		server.Listener.Addr().String())
	return server, url
}

func TestHandlePodTerminalSuccess(t *testing.T) {
	client := fake.NewSimpleClientset()
	config := &rest.Config{}
	mgrclient := FakeClient()

	// 设置测试环境
	server, url := setupWebSocketTest(func(req *restful.Request, resp *restful.Response) {
		h := &Handler{
			client:   client,
			terminal: webterminal.NewTerminal(client, config, mgrclient),
		}
		h.HandlePodTerminal(req, resp)
	})
	defer server.Close()

	dialer := websocket.DefaultDialer
	conn, _, err := dialer.Dial(url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, conn)
	defer conn.Close()
}

func TestHandlerHandleClusterTerminal(t *testing.T) {
	client := fake.NewSimpleClientset()
	config := &rest.Config{}
	mgrclient := FakeClient()

	// 设置测试环境
	server, url := setupWebSocketTest(func(req *restful.Request, resp *restful.Response) {
		h := &Handler{
			client:   client,
			terminal: webterminal.NewTerminal(client, config, mgrclient),
		}
		ctx := req.Request.Context()
		ctx = context.WithValue(ctx, "path", req.Request.URL.Path)
		h.HandleClusterTerminal(req, resp, ctx)
	})
	defer server.Close()
	// 模拟 WebSocket 客户端连接
	dialer := websocket.DefaultDialer
	conn, _, err := dialer.Dial(url, nil)
	if err != nil {
		t.Fatalf("Failed to connect to WebSocket server: %v", err)
	}
	defer conn.Close()

	// 验证连接成功
	assert.NotNil(t, conn, "WebSocket connection should not be nil")
}
func TestSendWebSocketError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upgrader := websocket.Upgrader{}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			http.Error(w, "Failed to upgrade WebSocket", http.StatusBadRequest)
			return
		}
		defer conn.Close()

		sendWebSocketError(conn, "Test error message")
	}))
	defer server.Close()

	// 模拟 WebSocket 客户端连接
	conn, _, err := MockWeb(t, server)
	defer conn.Close()

	// 读取错误消息
	_, message, err := conn.ReadMessage()
	assert.NoError(t, err)
	assert.Contains(t, string(message), "Error: Test error message")
}

func MockWeb(t *testing.T, server *httptest.Server) (*websocket.Conn, *http.Response, error) {
	dialer := websocket.DefaultDialer
	url := fmt.Sprintf("ws://%s", server.Listener.Addr().String())
	conn, _, err := dialer.Dial(url, nil)
	assert.NoError(t, err)
	assert.NotNil(t, conn)
	return conn, nil, err
}

func TestSendWebSocketMessage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upgrader := websocket.Upgrader{}
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			http.Error(w, "Failed to upgrade WebSocket", http.StatusBadRequest)
			return
		}
		defer conn.Close()

		sendWebSocketMessage(conn, "Test  message")
	}))
	defer server.Close()

	// 模拟 WebSocket 客户端连接
	conn, _, err := MockWeb(t, server)
	defer conn.Close()

	// 读取错误消息
	_, message, err := conn.ReadMessage()
	assert.NoError(t, err)
	assert.Contains(t, string(message), "Test  message")
}
