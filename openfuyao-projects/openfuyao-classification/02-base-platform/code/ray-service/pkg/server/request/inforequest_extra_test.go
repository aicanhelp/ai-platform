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

package request

import (
	"context"
	"net/http"
	"net/url"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
)

func newTestRequestInfoFactory() *RequestInfoFactory {
	return &RequestInfoFactory{
		RequestInfoFactory: &k8srequest.RequestInfoFactory{
			APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
			GrouplessAPIPrefixes: sets.NewString("api"),
		},
		GlobalResources: []schema.GroupResource{
			{Group: "resources.fuyao.io", Resource: "clusters"},
		},
	}
}

func TestNewRequestInfoWithClusterPath(t *testing.T) {
	r := newTestRequestInfoFactory()
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: "/api/clusters/guest/v1/namespaces/default/pods"},
	}

	info, err := r.NewRequestInfo(req)
	if err != nil {
		t.Fatalf("NewRequestInfo() error = %v, want nil", err)
	}
	if info.Cluster != "guest" {
		t.Errorf("NewRequestInfo() Cluster = %q, want %q", info.Cluster, "guest")
	}
	if !info.IsK8sRequest {
		t.Errorf("NewRequestInfo() IsK8sRequest = false, want true")
	}
	if info.Namespace != "default" || info.Resource != "pods" {
		t.Errorf("NewRequestInfo() namespace/resource = %q/%q, want default/pods", info.Namespace, info.Resource)
	}
	if info.ResourceScope != namespaceScopeVal {
		t.Errorf("NewRequestInfo() ResourceScope = %q, want %q", info.ResourceScope, namespaceScopeVal)
	}
	// the original path should be recovered after the call
	if req.URL.Path != "/api/clusters/guest/v1/namespaces/default/pods" {
		t.Errorf("NewRequestInfo() should recover original path, got %q", req.URL.Path)
	}
}

func TestNewRequestInfoNonResourcePath(t *testing.T) {
	r := newTestRequestInfoFactory()
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: "/rest/v1/overview/all"},
	}

	info, err := r.NewRequestInfo(req)
	if err != nil {
		t.Fatalf("NewRequestInfo() error = %v, want nil", err)
	}
	if info.IsK8sRequest {
		t.Errorf("NewRequestInfo() IsK8sRequest = true, want false for non k8s prefix")
	}
	if info.ResourceScope != clusterScopeVal {
		t.Errorf("NewRequestInfo() ResourceScope = %q, want %q", info.ResourceScope, clusterScopeVal)
	}
}

func TestNewRequestInfoGlobalResource(t *testing.T) {
	r := newTestRequestInfoFactory()
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: "/apis/resources.fuyao.io/v1alpha1/clusters"},
	}

	info, err := r.NewRequestInfo(req)
	if err != nil {
		t.Fatalf("NewRequestInfo() error = %v, want nil", err)
	}
	if info.APIGroup != "resources.fuyao.io" || info.Resource != "clusters" {
		t.Errorf("NewRequestInfo() apiGroup/resource = %q/%q, want resources.fuyao.io/clusters",
			info.APIGroup, info.Resource)
	}
	if info.ResourceScope != globalScopeVal {
		t.Errorf("NewRequestInfo() ResourceScope = %q, want %q", info.ResourceScope, globalScopeVal)
	}
}

func TestNewRequestInfoWithoutClusterSegment(t *testing.T) {
	r := newTestRequestInfoFactory()
	req := &http.Request{
		Method: http.MethodGet,
		URL:    &url.URL{Path: "/apis/apps/v1/namespaces/default/deployments"},
	}

	info, err := r.NewRequestInfo(req)
	if err != nil {
		t.Fatalf("NewRequestInfo() error = %v, want nil", err)
	}
	if info.Cluster != "" {
		t.Errorf("NewRequestInfo() Cluster = %q, want empty", info.Cluster)
	}
	if !info.IsK8sRequest {
		t.Errorf("NewRequestInfo() IsK8sRequest = false, want true")
	}
	if info.ResourceScope != namespaceScopeVal {
		t.Errorf("NewRequestInfo() ResourceScope = %q, want %q", info.ResourceScope, namespaceScopeVal)
	}
}

func TestExtractClusterShortPath(t *testing.T) {
	r := newTestRequestInfoFactory()
	factory := k8srequest.RequestInfoFactory{
		APIPrefixes:          sets.NewString("api", "apis", "fuyao"),
		GrouplessAPIPrefixes: sets.NewString("api"),
	}

	// path shorter than resourcePathMinLength is non-resource
	clusterName, pathname, ok := r.extractCluster(factory, "/api/v1")
	if ok {
		t.Errorf("extractCluster() ok = true, want false for short path")
	}
	if clusterName != "" || pathname != "/api/v1" {
		t.Errorf("extractCluster() = %q, %q, want empty cluster and original path", clusterName, pathname)
	}

	// path not starting with a known api prefix
	clusterName, pathname, ok = r.extractCluster(factory, "/clusters/guest/pods")
	if ok {
		t.Errorf("extractCluster() ok = true, want false for unknown api prefix")
	}
	if clusterName != "" || pathname != "/clusters/guest/pods" {
		t.Errorf("extractCluster() = %q, %q, want empty cluster and original path", clusterName, pathname)
	}

	// resource path without clusters segment
	clusterName, pathname, ok = r.extractCluster(factory, "/api/v1/pods")
	if ok {
		t.Errorf("extractCluster() ok = true, want false for path without clusters segment")
	}
	if clusterName != "" || pathname != "/api/v1/pods" {
		t.Errorf("extractCluster() = %q, %q, want empty cluster and original path", clusterName, pathname)
	}
}

func TestWithRequestInfoAndRequestInfoFrom(t *testing.T) {
	info := &RequestInfo{
		RequestInfo: &k8srequest.RequestInfo{Path: "/api/v1/pods", Verb: "GET"},
		Cluster:     "guest",
	}

	ctx := WithRequestInfo(context.Background(), info)
	got, ok := RequestInfoFrom(ctx)
	if !ok {
		t.Fatalf("RequestInfoFrom() ok = false, want true")
	}
	if got != info {
		t.Errorf("RequestInfoFrom() = %v, want %v", got, info)
	}

	if _, ok := RequestInfoFrom(context.Background()); ok {
		t.Errorf("RequestInfoFrom() ok = true, want false for empty context")
	}
}

func TestSetIsK8sRequestWithApiPrefix(t *testing.T) {
	info := &RequestInfo{
		RequestInfo: &k8srequest.RequestInfo{APIPrefix: "api"},
	}
	setIsK8sRequest(info)
	if !info.IsK8sRequest {
		t.Errorf("setIsK8sRequest() IsK8sRequest = false, want true for api prefix")
	}

	info = &RequestInfo{
		RequestInfo: &k8srequest.RequestInfo{APIPrefix: "rest"},
	}
	setIsK8sRequest(info)
	if info.IsK8sRequest {
		t.Errorf("setIsK8sRequest() IsK8sRequest = true, want false for rest prefix")
	}
}

func TestResolveResourceScope(t *testing.T) {
	r := newTestRequestInfoFactory()

	tests := []struct {
		name string
		info RequestInfo
		want string
	}{
		{
			name: "global resource",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					APIGroup: "resources.fuyao.io",
					Resource: "clusters",
				},
			},
			want: globalScopeVal,
		},
		{
			name: "namespace scoped resource",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					APIGroup:  "",
					Resource:  "pods",
					Namespace: "default",
				},
			},
			want: namespaceScopeVal,
		},
		{
			name: "cluster scoped resource",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					APIGroup: "",
					Resource: "nodes",
				},
			},
			want: clusterScopeVal,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := r.resolveResourceScope(tt.info); got != tt.want {
				t.Errorf("resolveResourceScope() = %q, want %q", got, tt.want)
			}
		})
	}
}
