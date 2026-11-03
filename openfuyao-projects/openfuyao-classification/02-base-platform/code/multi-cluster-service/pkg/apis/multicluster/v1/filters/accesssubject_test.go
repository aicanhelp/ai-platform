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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"
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
				req:   restful.NewRequest(httptest.NewRequest("GET", "/test", nil)),
				resp:  restful.NewResponse(httptest.NewRecorder()),
				chain: &restful.FilterChain{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			statusPatch := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.resp),
				"StatusCode", func(_ *restful.Response) int {
					return http.StatusOK
				})
			chainPatch := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.chain),
				"ProcessFilter", func(_ *restful.FilterChain,
					req *restful.Request, resp *restful.Response) {
					resp.WriteHeader(http.StatusOK)
				})
			defer statusPatch.Reset()
			defer chainPatch.Reset()
			ExactSubjectAccess(tt.args.req, tt.args.resp, tt.args.chain)
		})
	}
}

func TestGetSubject(t *testing.T) {
	type args struct {
		token string
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
				token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJteXVzZXIifQ.sf" +
					"lKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c", // 伪造但合法的 JWT 令牌
			},
			want:    "myuser", // 示例期望的解码结果
			wantErr: false,
		},
		{
			name: "invalid_token_format",
			args: args{
				token: "not.a.valid.token",
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := getSubject(tt.args.token)
			if (err != nil) != tt.wantErr {
				t.Errorf("getSubject() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("getSubject() got = %v, want %v", got, tt.want)
			}
		})
	}
}
