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
	"fmt"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
)

type args struct {
	c         *APIClient
	req       *restful.Request
	body      interface{}
	cluster   string
	operation string
}

func TestSync2UserMgr(t *testing.T) {
	tests := createTestCases()

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.name == "successful" {
				patch := gomonkey.ApplyMethod(reflect.TypeOf(&APIClient{}), "Post",
					func(_ *APIClient, path string, payload interface{}, token string) ([]byte, error) {
						if tt.mockPostErr != nil {
							return nil, tt.mockPostErr
						}
						return []byte("Success"), nil
					})
				defer patch.Reset()
			}
			if tt.authHeader != "" {
				tt.args.req.Request.Header.Set("Authorization", tt.authHeader)
			}
			if got := sync2UserMgr(tt.args.c, tt.args.req, tt.args.body, tt.args.cluster,
				tt.args.operation); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("sync2UserMgr() = %v, want %v", got, tt.want)
			}
		})
	}
}

// createTestCases 创建测试用例集合
func createTestCases() []struct {
	name           string
	args           args
	want           []byte
	authHeader     string
	mockPostReturn []byte
	mockPostErr    error
	mockPost       []byte
} {
	return []struct {
		name           string
		args           args
		want           []byte
		authHeader     string
		mockPostReturn []byte
		mockPostErr    error
		mockPost       []byte
	}{
		{
			name: "successful",
			args: args{
				c:         NewAPIClient(),
				req:       restful.NewRequest(httptest.NewRequest("GET", "/hello", nil)),
				body:      "test",
				cluster:   "cluster1",
				operation: "create",
			},
			authHeader:     "Bearer myToken",
			mockPostReturn: []byte("Success"),
			mockPostErr:    nil,
			want:           []byte("Success"),
			mockPost:       []byte("Success"),
		},
		{
			name: "missing token",
			args: args{
				c:         NewAPIClient(),
				req:       restful.NewRequest(httptest.NewRequest("GET", "/hello", nil)),
				body:      "test",
				cluster:   "cluster1",
				operation: "create",
			},
			authHeader:     "",
			mockPostReturn: nil,
			mockPostErr:    nil,
			want:           nil,
		},
		{
			name: "post request error",
			args: args{
				c:         NewAPIClient(),
				req:       restful.NewRequest(httptest.NewRequest("GET", "/hello", nil)),
				body:      "test",
				cluster:   "cluster1",
				operation: "create",
			},
			authHeader:     "Bearer myToken",
			mockPostReturn: nil,
			mockPostErr:    fmt.Errorf("failed to post data"),
			want:           nil,
		},
	}
}
