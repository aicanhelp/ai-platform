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

// Package request build request info
package request

import (
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
)

func TestRequestInfoFactoryNewRequestInfo(t *testing.T) {
	type fields struct {
		RequestInfoFactory *k8srequest.RequestInfoFactory
		GlobalResources    []schema.GroupResource
	}
	type args struct {
		req *http.Request
	}
	tests := []struct {
		name    string
		fields  fields
		args    args
		want    *RequestInfo
		wantErr bool
	}{
		{
			name: "valid request",
			fields: fields{
				RequestInfoFactory: &k8srequest.RequestInfoFactory{
					APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
					GrouplessAPIPrefixes: sets.NewString("api"),
				},
				GlobalResources: []schema.GroupResource{{Group: "apps", Resource: "deployments"}},
			},
			args: args{
				req: &http.Request{
					Method: "GET",
					URL: &url.URL{
						Path:    "http://example.com/api/v1/namespaces/default/pods",
						RawPath: "/api/v1/namespaces/default/pods",
					},
				},
			},
			want: &RequestInfo{
				RequestInfo:   &k8srequest.RequestInfo{},
				IsK8sRequest:  true,
				Cluster:       "",
				ResourceScope: "namespace",
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &RequestInfoFactory{
				RequestInfoFactory: tt.fields.RequestInfoFactory,
				GlobalResources:    tt.fields.GlobalResources,
			}
			_, err := r.NewRequestInfo(tt.args.req)
			if (err != nil) != tt.wantErr {
				t.Errorf("RequestInfoFactory.NewRequestInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
		})
	}
}

func TestSetIsK8sRequest(t *testing.T) {
	type args struct {
		requestInfo *RequestInfo
	}
	tests := []struct {
		name string
		args args
	}{
		{
			name: "set IsK8sRequest to true",
			args: args{
				requestInfo: &RequestInfo{
					RequestInfo: &k8srequest.RequestInfo{
						Verb: "GET",
						Path: "/api/v1/namespaces/default/pods",
					},
					IsK8sRequest:  false,
					Cluster:       "",
					ResourceScope: "",
				},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			setIsK8sRequest(tt.args.requestInfo)
		})
	}
}

func TestSplitPath(t *testing.T) {
	type args struct {
		path string
	}
	tests := []struct {
		name string
		args args
		want []string
	}{
		{
			name: "empty path",
			args: args{path: ""},
			want: []string{},
		},
		{
			name: "not empty path",
			args: args{path: "/api/v1/namespaces/default/pods"},
			want: []string{"api", "v1", "namespaces", "default", "pods"},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := splitPath(tt.args.path); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("splitPath() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestRequestInfoFactoryExtractCluster(t *testing.T) {
	type fields struct {
		RequestInfoFactory *k8srequest.RequestInfoFactory
		GlobalResources    []schema.GroupResource
	}
	type args struct {
		factory  k8srequest.RequestInfoFactory
		pathname string
	}
	tests := []struct {
		name   string
		fields fields
		args   args
		want   string
		want1  string
		want2  bool
	}{
		{
			name: "extract cluster from valid path",
			fields: fields{
				RequestInfoFactory: &k8srequest.RequestInfoFactory{
					APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
					GrouplessAPIPrefixes: sets.NewString("api"),
				},
				GlobalResources: []schema.GroupResource{{Group: "apps", Resource: "deployments"}},
			},
			args: args{
				factory: k8srequest.RequestInfoFactory{
					APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
					GrouplessAPIPrefixes: sets.NewString("api"),
				},
				pathname: "/api/clusters/my-cluster/namespaces/default/pods",
			},
			want:  "my-cluster",
			want1: "api/namespaces/default/pods",
			want2: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := &RequestInfoFactory{
				RequestInfoFactory: tt.fields.RequestInfoFactory,
				GlobalResources:    tt.fields.GlobalResources,
			}
			got, got1, got2 := r.extractCluster(tt.args.factory, tt.args.pathname)
			if got != tt.want {
				t.Errorf("RequestInfoFactory.extractCluster() got = %v, want %v", got, tt.want)
			}
			if got1 != tt.want1 {
				t.Errorf("RequestInfoFactory.extractCluster() got1 = %v, want %v", got1, tt.want1)
			}
			if got2 != tt.want2 {
				t.Errorf("RequestInfoFactory.extractCluster() got2 = %v, want %v", got2, tt.want2)
			}
		})
	}
}
