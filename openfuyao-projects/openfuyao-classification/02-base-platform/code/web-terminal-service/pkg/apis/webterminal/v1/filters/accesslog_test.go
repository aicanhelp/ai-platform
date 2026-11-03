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
	"github.com/stretchr/testify/assert"
)

func TestRecordAccessLogs(t *testing.T) {
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
			name: "restful success",
			args: args{
				req:   restful.NewRequest(httptest.NewRequest("GET", "/test", nil)),
				resp:  restful.NewResponse(httptest.NewRecorder()),
				chain: &restful.FilterChain{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 模拟 restful.Response.StatusCode
			patch1 := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.resp), "StatusCode", func(_ *restful.Response) int {
				return http.StatusOK
			})

			// 模拟 restful.FilterChain.ProcessFilter
			patch2 := gomonkey.ApplyMethod(reflect.TypeOf(tt.args.chain), "ProcessFilter",
				func(_ *restful.FilterChain, req *restful.Request, resp *restful.Response) {
					resp.WriteHeader(http.StatusOK)
				})

			defer patch1.Reset()
			defer patch2.Reset()

			RecordAccessLogs(tt.args.req, tt.args.resp, tt.args.chain)

		})
	}
}

func TestRecordAccessLogs_ErrorStatus(t *testing.T) {
	req := restful.NewRequest(httptest.NewRequest("GET", "/test", nil))
	resp := restful.NewResponse(httptest.NewRecorder())
	chain := &restful.FilterChain{}

	var called bool
	patch1 := gomonkey.ApplyMethod(reflect.TypeOf(resp), "StatusCode", func(_ *restful.Response) int {
		return http.StatusInternalServerError
	})
	patch2 := gomonkey.ApplyMethod(reflect.TypeOf(chain), "ProcessFilter",
		func(_ *restful.FilterChain, req *restful.Request, resp *restful.Response) {
			called = true
			resp.WriteHeader(http.StatusInternalServerError)
		})
	defer patch1.Reset()
	defer patch2.Reset()

	RecordAccessLogs(req, resp, chain)
	assert.True(t, called)
}
