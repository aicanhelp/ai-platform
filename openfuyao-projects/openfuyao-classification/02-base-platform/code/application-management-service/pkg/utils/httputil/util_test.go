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

// Package httputil
package httputil

import (
	"crypto/tls"
	"crypto/x509"
	"net/http"
	"net/http/httptest"
	"os"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/emicklei/go-restful/v3"

	"application-management-service/pkg/constant"
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

func TestGetHttpConfigTLSFailed(t *testing.T) {
	tests := []struct {
		name      string
		enableTLS bool
		want      *tls.Config
		wantErr   bool
	}{
		{
			name:      "TestDisableTLS",
			enableTLS: false,
			want:      &tls.Config{InsecureSkipVerify: true},
			wantErr:   false,
		},
		{
			name:      "TestLoadCertFailed",
			enableTLS: true,
			want:      nil,
			wantErr:   true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetHttpConfig(tt.enableTLS)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetHttpConfig() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetHttpConfig() got = %v, want %v", got, tt.want)
			}
		})
	}

	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(certFile, keyFile string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	defer patch1.Reset()

	t.Run("TestLoadCAFailed", func(t *testing.T) {
		got, err := GetHttpConfig(true)
		if err == nil {
			t.Errorf("GetHttpConfig() error = %v, wantErr true", err)
		}
		if got != nil {
			t.Errorf("GetHttpConfig() got = %v, want nil", got)
		}
	})
}

func TestGetHttpConfigSuccess(t *testing.T) {
	patch1 := gomonkey.ApplyFunc(tls.LoadX509KeyPair, func(certFile, keyFile string) (tls.Certificate, error) {
		return tls.Certificate{}, nil
	})
	defer patch1.Reset()

	fakeCABytes := []byte("-----BEGIN CERTIFICATE-----\nxxxxxxxx\n-----END CERTIFICATE-----")
	fakeCAPool := x509.NewCertPool()
	fakeCAPool.AppendCertsFromPEM(fakeCABytes)

	patch2 := gomonkey.ApplyFunc(os.ReadFile, func(filename string) ([]byte, error) {
		return fakeCABytes, nil
	})
	defer patch2.Reset()

	got, err := GetHttpConfig(true)
	want := &tls.Config{
		Certificates: []tls.Certificate{{}},
		MinVersion:   tls.VersionTLS12,
		MaxVersion:   tls.VersionTLS13,
		RootCAs:      fakeCAPool,
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("TLS Config is incorrect: %v, want %v", got, want)
		return
	}
	if err != nil {
		t.Errorf("error = %v, wantErr false", err)
		return
	}
}

func TestGetHttpTransport(t *testing.T) {
	type args struct {
		enableTLS bool
	}
	tests := []struct {
		name    string
		args    args
		want    *http.Transport
		wantErr bool
	}{
		{
			name: "GetHTTPConfigFailed",
			args: args{
				enableTLS: true,
			},
			want:    nil,
			wantErr: true,
		},
		{
			name: "GetHTTPConfigSucceeded",
			args: args{
				enableTLS: false,
			},
			want: &http.Transport{
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: true,
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetHttpTransport(tt.args.enableTLS)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetHttpTransport() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("GetHttpTransport() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestIsHttpsEnabled(t *testing.T) {
	patch := gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, os.ErrNotExist
	})
	defer patch.Reset()
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err != nil {
			t.Errorf("IsHttpsEnabled error = %v, want nil", err)
		}
		if got {
			t.Error("IsHttpsEnabled got = true, want false")
		}
	})

	patch = gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, os.ErrPermission
	})
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err == nil {
			t.Error("IsHttpsEnabled error = nil, want not nil")
		}
		if got {
			t.Error("IsHttpsEnabled got = true, want false")
		}
	})

	patch = gomonkey.ApplyFunc(os.Stat, func() (os.FileInfo, error) {
		return nil, nil
	})
	t.Run("PathNotExist", func(t *testing.T) {
		got, err := IsHttpsEnabled()
		if err != nil {
			t.Errorf("IsHttpsEnabled error = %v, want nil", err)
		}
		if !got {
			t.Errorf("IsHttpsEnabled got = %v, want true", got)
		}
	})
}

func TestHandleNotAuthorized(t *testing.T) {
	recorder := httptest.NewRecorder()
	response := &restful.Response{ResponseWriter: recorder}
	message := "未授权访问"
	testErr := os.ErrPermission

	HandleNotAuthorized(response, message, testErr)

	if recorder.Code != http.StatusForbidden {
		t.Errorf("期望状态码 %d，实际为 %d", http.StatusForbidden, recorder.Code)
	}
}
