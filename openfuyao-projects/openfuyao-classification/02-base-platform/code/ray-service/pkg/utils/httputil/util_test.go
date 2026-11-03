/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
	"net/http"
	"reflect"
	"testing"
)

type responseTestCase struct {
	name string
	want *ResponseJson
}

type responseWithArgsTestCase struct {
	name string
	args any
	want *ResponseJson
}

type receiveFileTestCase struct {
	name string
	args struct {
		r   *http.Request
		key string
	}
	want    string
	wantErr bool
}

func newTestResponseJson(code int32, msg string) *ResponseJson {
	return &ResponseJson{
		Code: code,
		Msg:  msg,
		Data: nil,
	}
}

const (
	HTTPStatusOK          = 200
	HTTPStatusError       = 400
	HTTPStatusServerError = 500
)

func TestGetResponseJson(t *testing.T) {
	tests := []responseWithArgsTestCase{
		{
			name: "success response",
			args: struct {
				code int32
				msg  string
				data any
			}{
				code: HTTPStatusOK,
				msg:  "success",
				data: nil,
			},
			want: newTestResponseJson(HTTPStatusOK, "success"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// 需要进行类型断言来获取具体的参数
			args, ok := tt.args.(struct {
				code int32
				msg  string
				data any
			})
			if !ok {
				t.Fatalf("Failed to assert args type for test case: %s", tt.name)
			}
			if got := GetResponseJson(args.code, args.msg, args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetResponseJson() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestGetDefaultSuccessResponseJson(t *testing.T) {
	tests := []responseTestCase{
		{
			name: "default success",
			want: newTestResponseJson(HTTPStatusOK, "success"),
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
func TestGetDefaultClientFailureResponseJson(t *testing.T) {
	tests := []responseTestCase{
		{
			name: "default client failure",
			want: newTestResponseJson(HTTPStatusError, "bad request"),
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
func TestGetClientFailureResponseJsonWithData(t *testing.T) {
	tests := []responseWithArgsTestCase{
		{
			name: "client failure with nil data",
			args: struct{ data any }{
				data: nil,
			},
			want: newTestResponseJson(HTTPStatusError, "bad request"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, ok := tt.args.(struct{ data any })
			if !ok {
				t.Fatalf("Failed to assert args type for test case: %s", tt.name)
			}
			if got := GetClientFailureResponseJsonWithData(args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetClientFailureResponseJsonWithData() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestGetDefaultServerFailureResponseJson(t *testing.T) {
	tests := []responseTestCase{
		{
			name: "default server failure",
			want: newTestResponseJson(HTTPStatusServerError, "remote server busy"),
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
func TestGetServerFailureResponseJsonWithData(t *testing.T) {
	tests := []responseWithArgsTestCase{
		{
			name: "server failure with nil data",
			args: struct{ data any }{
				data: nil,
			},
			want: newTestResponseJson(HTTPStatusServerError, "operation failed"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			args, ok := tt.args.(struct{ data any })
			if !ok {
				t.Fatalf("Failed to assert args type for test case: %s", tt.name)
			}
			if got := GetServerFailureResponseJsonWithData(args.data); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetServerFailureResponseJsonWithData() = %v, want %v", got, tt.want)
			}
		})
	}
}
func TestGetParamsEmptyErrorResponseJson(t *testing.T) {
	tests := []responseTestCase{
		{
			name: "params empty error",
			want: newTestResponseJson(HTTPStatusError, "parameters not found"),
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
func TestReceiveFile(t *testing.T) {
	tests := []receiveFileTestCase{
		{
			name: "nil request",
			args: struct {
				r   *http.Request
				key string
			}{
				r:   nil,
				key: "",
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ReceiveFile(tt.args.r, tt.args.key)
			if (err != nil) != tt.wantErr {
				t.Errorf("ReceiveFile() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ReceiveFile() = %v, want %v", got, tt.want)
			}
		})
	}
}
