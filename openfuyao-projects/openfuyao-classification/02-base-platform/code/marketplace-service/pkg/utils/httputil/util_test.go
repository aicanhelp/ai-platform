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

package httputil

import (
	"reflect"
	"testing"

	"marketplace-service/pkg/constant"
)

func TestGetResponseJson(t *testing.T) {
	type args struct {
		code int32
		msg  string
		data any
	}
	var response = &ResponseJson{
		Code: 200,
		Msg:  "ok",
		Data: "TestGetResponseJson",
	}
	tests := []struct {
		name string
		args args
		want *ResponseJson
	}{
		{
			name: "TestGetResponseJson",
			args: args{
				code: 200,
				msg:  "ok",
				data: "TestGetResponseJson",
			},
			want: response,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetResponseJson(tt.args.code, tt.args.msg, tt.args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetDefaultClientFailureResponseJson(t *testing.T) {
	type args struct {
		code int32
		msg  string
		data any
	}
	var response = &ResponseJson{
		Code: constant.ClientError,
		Msg:  "bad request",
		Data: nil,
	}
	tests := []struct {
		name string
		args args
		want *ResponseJson
	}{
		{
			name: "TestGetDefaultClientFailureResponseJson",
			args: args{},
			want: response,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDefaultClientFailureResponseJson(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetDefaultClientFailureResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetDefaultServerFailureResponseJson(t *testing.T) {
	type args struct {
		code int32
		msg  string
		data any
	}
	var response = &ResponseJson{
		Code: constant.ServerError,
		Msg:  "remote server busy",
		Data: nil,
	}
	tests := []struct {
		name string
		args args
		want *ResponseJson
	}{
		{
			name: "TestGetDefaultServerFailureResponseJson",
			args: args{},
			want: response,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDefaultServerFailureResponseJson(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetDefaultServerFailureResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetDefaultSuccessResponseJson(t *testing.T) {
	type args struct {
		code int32
		msg  string
		data any
	}
	var response = &ResponseJson{
		Code: constant.Success,
		Msg:  "success",
		Data: nil,
	}
	tests := []struct {
		name string
		args args
		want *ResponseJson
	}{
		{
			name: "TestGetDefaultSuccessResponseJson",
			args: args{},
			want: response,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetDefaultSuccessResponseJson(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetDefaultSuccessResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestGetParamsEmptyErrorResponseJson(t *testing.T) {
	type args struct {
		code int32
		msg  string
		data any
	}
	var response = &ResponseJson{
		Code: constant.ClientError,
		Msg:  "parameters not found",
		Data: nil,
	}
	tests := []struct {
		name string
		args args
		want *ResponseJson
	}{
		{
			name: "TestGetParamsEmptyErrorResponseJson",
			args: args{},
			want: response,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := GetParamsEmptyErrorResponseJson(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetParamsEmptyErrorResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}
