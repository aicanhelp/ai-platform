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
package filters

import (
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/assert"

	"openfuyao.com/web-terminal-service/pkg/apis/webterminal/v1/responsehandlers"
)

func TestExactSubjectAccess(t *testing.T) {
	type args struct {
		req   *restful.Request
		resp  *restful.Response
		chain *restful.FilterChain
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "success",
			args: args{
				req:   restful.NewRequest(httptest.NewRequest("GET", "/Access", nil)),
				resp:  restful.NewResponse(httptest.NewRecorder()),
				chain: &restful.FilterChain{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch3 := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.resp), "StatusCode", func(_ *restful.Response) int {
				return http.StatusOK
			})

			// 模拟 restful.FilterChain.ProcessFilter
			patch4 := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.chain), "ProcessFilter",
				func(_ *restful.FilterChain, req *restful.Request, resp *restful.Response) {
					resp.WriteHeader(http.StatusOK)
				})

			defer patch3.Reset()
			defer patch4.Reset()
			ExactSubjectAccess(tt.args.req, tt.args.resp, tt.args.chain)
		})
	}
}

func TestExactSubjectAccess_NoHeaderCallsChain(t *testing.T) {
	req := restful.NewRequest(httptest.NewRequest("GET", "/Access", nil))
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	var called bool
	patch := gomonkey.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter",
		func(_ *restful.FilterChain, r *restful.Request, w *restful.Response) {
			called = true
			_, ok := r.Request.Context().Value("user").(string)
			assert.False(t, ok)
		})
	defer patch.Reset()

	ExactSubjectAccess(req, resp, chain)
	assert.True(t, called)
}

func TestExactSubjectAccess_DefaultHeaderSetsUser(t *testing.T) {
	req := httptest.NewRequest("GET", "/Access", nil)
	req.Header.Set(defaultAuthHeader, "Bearer "+validJWTToken())
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	var subject string
	patch := gomonkey.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter",
		func(_ *restful.FilterChain, r *restful.Request, w *restful.Response) {
			subject, _ = r.Request.Context().Value("user").(string)
		})
	defer patch.Reset()

	ExactSubjectAccess(restful.NewRequest(req), resp, chain)
	assert.Equal(t, "myuser", subject)
}

func TestExactSubjectAccess_OpenFuyaoHeaderSetsUser(t *testing.T) {
	req := httptest.NewRequest("GET", "/Access", nil)
	req.Header.Set(openFuyaoAuthHeader, "Bearer "+validJWTToken())
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	var subject string
	patch := gomonkey.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter",
		func(_ *restful.FilterChain, r *restful.Request, w *restful.Response) {
			subject, _ = r.Request.Context().Value("user").(string)
		})
	defer patch.Reset()

	ExactSubjectAccess(restful.NewRequest(req), resp, chain)
	assert.Equal(t, "myuser", subject)
}

func TestExactSubjectAccess_InvalidTokenStopsChain(t *testing.T) {
	req := httptest.NewRequest("GET", "/Access", nil)
	req.Header.Set(defaultAuthHeader, "Bearer bad-token")
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	var chainCalled bool
	var errHandled bool
	patch1 := gomonkey.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter",
		func(_ *restful.FilterChain, r *restful.Request, w *restful.Response) {
			chainCalled = true
		})
	patch2 := gomonkey.ApplyFunc(responsehandlers.SendStatusServerError,
		func(_ *restful.Response, message string, err error) {
			errHandled = true
		})
	defer patch1.Reset()
	defer patch2.Reset()

	ExactSubjectAccess(restful.NewRequest(req), resp, chain)
	assert.True(t, errHandled)
	assert.False(t, chainCalled)
}

func TestGetSubject(t *testing.T) {
	type args struct {
		token string
		resp  *restful.Response
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "yes",
			args: args{
				token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." +
					"eyJzdWIiOiJteXVzZXIifQ.sflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", // 伪造但合法的 JWT 令牌
				resp: restful.NewResponse(httptest.NewRecorder()),
			},
			want:    "myuser", // 示例期望的解码结果
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			patch := gomonkey.ApplyFunc(responsehandlers.SendStatusServerError,
				func(_ *restful.Response, message string, err error) {
					fmt.Println(http.StatusForbidden)
				})
			defer patch.Reset()
			got, err := getSubject(tt.args.resp, tt.args.token)
			assert.Equal(t, tt.wantErr, err != nil)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestGetSubject_InvalidToken(t *testing.T) {
	resp := restful.NewResponse(httptest.NewRecorder())
	var errHandled bool
	patch := gomonkey.ApplyFunc(responsehandlers.SendStatusServerError,
		func(_ *restful.Response, message string, err error) {
			errHandled = true
		})
	defer patch.Reset()

	got, err := getSubject(resp, "bad-token")
	assert.True(t, errHandled)
	assert.Error(t, err)
	assert.Equal(t, "", got)
}

func validJWTToken() string {
	return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." +
		"eyJzdWIiOiJteXVzZXIifQ.sflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
}
