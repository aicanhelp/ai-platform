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
package responsehandlers

import (
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/emicklei/go-restful/v3"
)

func TestLogErrorInfo(t *testing.T) {
	type args struct {
		err error
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "error",
			args: args{
				err: errors.New("error"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logErrorInfo(tt.args.err)
		})
	}
}

func TestSendStatusBadRequest(t *testing.T) {
	type args struct {
		resp    *restful.Response
		message string
		err     error
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "ok",
			args: args{
				resp:    restful.NewResponse(httptest.NewRecorder()),
				message: "statues",
				err:     nil,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SendStatusBadRequest(tt.args.resp, tt.args.message, tt.args.err)
		})
	}
}

func TestSendStatusForbidden(t *testing.T) {
	type args struct {
		resp    *restful.Response
		message string
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: " forbbiden",
			args: args{
				resp:    restful.NewResponse(httptest.NewRecorder()),
				message: "no",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SendStatusForbidden(tt.args.resp, tt.args.message)
		})
	}
}

func TestSendStatusOk(t *testing.T) {
	type args struct {
		resp    *restful.Response
		message string
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: " send ok",
			args: args{
				resp:    restful.NewResponse(httptest.NewRecorder()),
				message: "",
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SendStatusOk(tt.args.resp, tt.args.message)
		})
	}
}

func TestSendStatusServerError(t *testing.T) {
	type args struct {
		resp    *restful.Response
		message string
		err     error
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "send status ",
			args: args{
				resp:    restful.NewResponse(httptest.NewRecorder()),
				message: "status ok ",
				err:     errors.New("test"),
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			SendStatusServerError(tt.args.resp, tt.args.message, tt.args.err)
		})
	}
}
