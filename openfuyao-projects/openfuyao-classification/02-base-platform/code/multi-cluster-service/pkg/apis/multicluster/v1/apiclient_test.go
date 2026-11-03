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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestNewAPIClient(t *testing.T) {
	tests := []struct {
		name string
		want *APIClient
	}{
		{
			name: "API Client",
			want: &APIClient{
				Client: &http.Client{},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := NewAPIClient(); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("NewAPIClient() = %v, want %v", *got.Client, tt.want)
			}
		})
	}
}

func TestAPIClientGet(t *testing.T) {
	type fields struct {
		Client *http.Client
	}
	type args struct {
		path string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []byte
		wantErr bool
	}{
		{
			name: "successful GET request",
			fields: fields{
				Client: &http.Client{},
			},
			args: args{
				path: "/test",
			},
			want:    []byte(`{"message": "success"}`),
			wantErr: false,
		},
		{
			name: "404 Not Found error",
			fields: fields{
				Client: &http.Client{},
			},
			args: args{
				path: "/notfound",
			},
			want:    nil,
			wantErr: true,
		},
	}
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/test":
			w.WriteHeader(http.StatusOK)
			w.Write([]byte(`{"message": "success"}`))
		case "/notfound":
			w.WriteHeader(http.StatusNotFound)
		default:
			w.WriteHeader(http.StatusBadRequest)
		}
	}))
	defer mockServer.Close()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &APIClient{
				Client: tt.fields.Client,
			}
			tt.args.path = mockServer.URL + tt.args.path
			got, err := c.Get(tt.args.path)
			if (err != nil) != tt.wantErr {
				t.Errorf("Get() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Get() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestAPIClientPost(t *testing.T) {
	type fields struct {
		Client *http.Client
	}
	type args struct {
		path    string
		payload interface{}
		token   string
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    []byte
		wantErr bool
	}{
		{
			name: "successful POST request with token",
			fields: fields{
				Client: &http.Client{},
			},
			args: args{
				path:    "/test",
				payload: map[string]string{"name": "test"},
				token:   "test-token",
			},
			want:    []byte(`{"message":"success"}`),
			wantErr: false,
		},
		{
			name: "POST request without token",
			fields: fields{
				Client: &http.Client{},
			},
			args: args{
				path:    "/test",
				payload: map[string]string{"name": "no-token"},
				token:   "",
			},
			want:    []byte(`{"message":"no auth"}`),
			wantErr: false,
		},
		{
			name: "POST request with error response",
			fields: fields{
				Client: &http.Client{},
			},
			args: args{
				path:    "/error",
				payload: map[string]string{"name": "error-case"},
				token:   "",
			},
			want:    nil,
			wantErr: true,
		},
	}
	mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			t.Errorf("Expected 'POST' request, got '%s'", r.Method)
		}
		if r.URL.Path == "/test" && r.Header.Get("Authorization") == "Bearer test-token" {
			w.WriteHeader(http.StatusOK)
			_, err := w.Write([]byte(`{"message":"success"}`))
			if err != nil {
				fmt.Printf("%v", err)
			}
			return
		}
		if r.URL.Path == "/test" && r.Header.Get("Authorization") == "" {
			w.WriteHeader(http.StatusOK)
			_, err := w.Write([]byte(`{"message":"no auth"}`))
			if err != nil {
				fmt.Printf("%v", err)
			}
			return
		}
		if r.URL.Path == "/error" {
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
	}))
	defer mockServer.Close()
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &APIClient{
				Client: tt.fields.Client,
			}
			tt.args.path = mockServer.URL + tt.args.path
			got, err := c.Post(tt.args.path, tt.args.payload, tt.args.token)
			if (err != nil) != tt.wantErr {
				t.Errorf("Post() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Post() got = %s, want %s", string(got), string(tt.want))
			}
		})
	}
}
