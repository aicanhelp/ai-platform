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
	"github.com/stretchr/testify/assert"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/client/k8s"
)

func TestNewClientandConfig(t *testing.T) {
	tests := []struct {
		name  string
		want  *rest.Config
		want1 *kubernetes.Clientset
	}{
		{
			name:  "ok",
			want:  &rest.Config{},
			want1: &kubernetes.Clientset{},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {

			patch1 := gomonkey.ApplyFunc(k8s.GetKubeConfig, func() *rest.Config {
				return &rest.Config{}
			})
			patch2 := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(config *rest.Config) (*kubernetes.Clientset, error) {
				return &kubernetes.Clientset{}, nil
			})
			defer patch1.Reset()
			defer patch2.Reset()
			got, got1 := NewClientandConfig()
			assert.Equalf(t, tt.want, got, "NewClientandConfig()")
			assert.Equalf(t, tt.want1, got1, "NewClientandConfig()")
		})
	}
}
func TestTerminalPod(t *testing.T) {
	// 创建一个 RESTful WebService 和 Handler
	ws := new(restful.WebService)
	h := &Handler{}

	patch := gomonkey.ApplyMethod(reflect.TypeOf(&Handler{}), "HandlePodTerminal",
		func(_ *Handler, req *restful.Request, resp *restful.Response) {
			fmt.Println(http.StatusOK)
		})
	defer patch.Reset()
	// 调用 terminalPod 方法注册路由
	terminalPod(ws, h)

	// 验证路由是否被正确注册
	assert.Len(t, ws.Routes(), 1, "Expected one route to be registered")

	route := ws.Routes()[0]
	assert.Equal(t, "GET", route.Method, "Expected HTTP method to be GET")
	assert.Equal(t, "/namespace/{namespace}/pod/{pod}/container/{container}/terminal",
		route.Path, "Expected route path to match")
	assert.Equal(t, "Create Pod Terminal", route.Doc, "Expected route documentation to match")
	assert.Equal(t, "create-pod-exec", route.Operation, "Expected route operation to match")

	// 验证路由处理函数是否正确
	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET",
		"/namespace/default/pod/test-pod/container/test-container/terminal", nil)
	req = req.WithContext(req.Context())

	container := restful.NewContainer()
	ws.Consumes(restful.MIME_JSON).Produces(restful.MIME_JSON)
	container.Add(ws)
	container.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code, "ok")
}

func TestSayHello(t *testing.T) {
	type args struct {
		ws *restful.WebService
		h  *Handler
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: " ok",
			args: args{
				ws: new(restful.WebService),
				h:  &Handler{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sayHello(tt.args.ws, tt.args.h)
		})
	}
}

func TestTerminalCluster(t *testing.T) {
	// 创建一个 RESTful WebService 和 Handler
	ws := new(restful.WebService)
	h := &Handler{}

	patch := gomonkey.ApplyMethod(reflect.TypeOf(&Handler{}), "HandleClusterTerminal",
		func(_ *Handler, req *restful.Request, resp *restful.Response, ctx context.Context) {
			fmt.Println(http.StatusOK)
		})
	defer patch.Reset()

	terminalCluster(ws, h)

	// 验证路由是否被正确注册
	assert.Len(t, ws.Routes(), 1, "Expected one route to be registered")

	route := ws.Routes()[0]
	assert.Equal(t, "GET", route.Method, "Expected HTTP method to be GET")
	assert.Equal(t, "/user/{user}/terminal", route.Path, "Expected route path to match")
	assert.Equal(t, "Create Web Terminal Template", route.Doc, "Expected route documentation to match")
	assert.Equal(t, "create-web-terminal-template", route.Operation, "Expected route operation to match")

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/user/jack/terminal", nil)
	req = req.WithContext(req.Context())

	container := restful.NewContainer()
	container.Add(ws)
	container.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code, "pass")
}

func TestNewClientandConfig_Error(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(k8s.GetKubeConfig, func() *rest.Config {
		return &rest.Config{}
	})
	patch2 := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(config *rest.Config) (*kubernetes.Clientset, error) {
		return nil, fmt.Errorf("boom")
	})
	defer patch1.Reset()
	defer patch2.Reset()

	cfg, clientset := NewClientandConfig()
	assert.NotNil(t, cfg)
	assert.Nil(t, clientset)
}

func TestSayHello_Response(t *testing.T) {
	ws := new(restful.WebService)
	ws.Consumes(restful.MIME_JSON).Produces(restful.MIME_JSON)
	h := &Handler{}
	sayHello(ws, h)

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/hello", nil)
	req.Header.Set("Accept", restful.MIME_JSON)

	container := restful.NewContainer()
	container.Add(ws)
	container.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "WTS API Server Working Successfully")
}

func TestTerminalCluster_ContextValues(t *testing.T) {
	ws := new(restful.WebService)
	h := &Handler{}

	var gotUser string
	var gotPath string
	patch := gomonkey.ApplyMethod(reflect.TypeOf(&Handler{}), "HandleClusterTerminal",
		func(_ *Handler, req *restful.Request, resp *restful.Response, ctx context.Context) {
			user, _ := ctx.Value("username").(string)
			path, _ := ctx.Value("path").(string)
			gotUser = user
			gotPath = path
			resp.WriteHeader(http.StatusOK)
		})
	defer patch.Reset()

	terminalCluster(ws, h)

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/user/jack/terminal", nil)

	container := restful.NewContainer()
	container.Add(ws)
	container.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Equal(t, "jack", gotUser)
	assert.Equal(t, "/user/jack/terminal", gotPath)
}

func TestAddToContainer_HelloRoute(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(k8s.GetKubeConfig, func() *rest.Config {
		return &rest.Config{}
	})
	patch2 := gomonkey.ApplyFunc(kubernetes.NewForConfig, func(config *rest.Config) (*kubernetes.Clientset, error) {
		return &kubernetes.Clientset{}, nil
	})
	defer patch1.Reset()
	defer patch2.Reset()

	container := restful.NewContainer()
	err := AddToContainer(container, nil)
	assert.NoError(t, err)

	recorder := httptest.NewRecorder()
	req := httptest.NewRequest("GET", "/rest/webterminal/v1/hello", nil)
	req.Header.Set("Accept", restful.MIME_JSON)
	container.ServeHTTP(recorder, req)

	assert.Equal(t, http.StatusOK, recorder.Code)
	assert.Contains(t, recorder.Body.String(), "WTS API Server Working Successfully")
}
