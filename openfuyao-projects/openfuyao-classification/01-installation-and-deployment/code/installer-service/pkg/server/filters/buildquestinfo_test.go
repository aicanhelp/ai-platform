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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"

	"installer-service/pkg/server/request"
)

func TestRequestInfoBuilderServeHTTP(t *testing.T) {
	tests := []struct {
		name                  string
		setup                 func(b *requestInfoBuilder) *gomonkey.Patches
		expectInternalError   bool
		expectNextHandlerBody string
	}{
		{
			name: "创建RequestInfo失败",
			setup: func(b *requestInfoBuilder) *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyMethod(b.resolver, "NewRequestInfo", func(_ *request.RequestInfoFactory, _ *http.Request) (*request.RequestInfo, error) {
					return nil, fmt.Errorf("mock error")
				})
				p.ApplyFunc(responsewriters.InternalError, func(w http.ResponseWriter, r *http.Request, err error) {
					// ensure the internal error path is invoked
					w.WriteHeader(http.StatusInternalServerError)
				})
				return p
			},
			expectInternalError: true,
		},
		{
			name: "创建RequestInfo成功",
			setup: func(b *requestInfoBuilder) *gomonkey.Patches {
				p := gomonkey.NewPatches()
				p.ApplyMethod(b.resolver, "NewRequestInfo", func(_ *request.RequestInfoFactory, _ *http.Request) (*request.RequestInfo, error) {
					return &request.RequestInfo{}, nil
				})
				return p
			},
			expectInternalError:   false,
			expectNextHandlerBody: "ok",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 创建测试对象
			builder := &requestInfoBuilder{
				resolver: &request.RequestInfoFactory{},
				nextHandler: http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					// next handler will check for request info in context
					if info, ok := request.RequestInfoFrom(r.Context()); ok && info != nil {
						w.Write([]byte("ok"))
					} else {
						w.Write([]byte("no-info"))
					}
				}),
			}

			// 创建请求和响应
			req := httptest.NewRequest("GET", "/test", nil)
			w := httptest.NewRecorder()

			patches := tt.setup(builder)
			if patches != nil {
				defer patches.Reset()
			}

			// 执行测试
			builder.ServeHTTP(w, req)

			// 验证结果
			if tt.expectInternalError {
				// internal error should have written a 500
				assert.Equal(t, http.StatusInternalServerError, w.Result().StatusCode)
			} else {
				assert.Equal(t, 200, w.Result().StatusCode)
				body := w.Body.String()
				assert.Equal(t, tt.expectNextHandlerBody, body)
			}
		})
	}
}

func TestBuildRequestInfo(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {})
	resolver := &request.RequestInfoFactory{}
	handler := BuildRequestInfo(inner, resolver)
	assert.NotNil(t, handler)

	builder, ok := handler.(*requestInfoBuilder)
	assert.True(t, ok)
	assert.NotNil(t, builder.nextHandler)
	assert.Equal(t, resolver, builder.resolver)
}
