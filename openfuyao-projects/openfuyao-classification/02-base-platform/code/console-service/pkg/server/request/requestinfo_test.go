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

package request

import (
	"context"
	"net/http"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/kubernetes/fake"

	"console-service/pkg/constant"
)

var testRequests = []struct {
	reqAdr              string
	reqID               string
	reqMtd              string
	reqEptClt           string
	reqEptError         error
	reqEptResource      string
	reqEptVerb          string
	reqEptK8SReq        bool
	reqEptNamespace     string
	reqEptIsResourceReq bool
}{
	{
		reqAdr:              "/api/v1/nodes",
		reqID:               "list nodes",
		reqMtd:              http.MethodGet,
		reqEptVerb:          "GET",
		reqEptError:         nil,
		reqEptClt:           "",
		reqEptResource:      "",
		reqEptK8SReq:        true,
		reqEptIsResourceReq: false,
	},
	{
		reqAdr:              "/apis/rbac.authorization.k8s.io/v1/namespaces/namespace1/roles",
		reqID:               "list roles",
		reqEptError:         nil,
		reqMtd:              http.MethodGet,
		reqEptClt:           "",
		reqEptNamespace:     "",
		reqEptVerb:          "GET",
		reqEptResource:      "",
		reqEptK8SReq:        true,
		reqEptIsResourceReq: false,
	},
	{
		reqAdr:              "/foo/bar",
		reqID:               "random query",
		reqEptError:         nil,
		reqMtd:              http.MethodGet,
		reqEptResource:      "",
		reqEptVerb:          "GET",
		reqEptClt:           "",
		reqEptK8SReq:        false,
		reqEptIsResourceReq: false,
	},
}

func newTestRequestInfoResolver() *RequestInfoFactory {
	var groupVersions = schema.GroupVersion{Group: "resources.fuyao.io", Version: "v1alpha1"}
	requestInfoResolver := &RequestInfoFactory{
		GlobalResources: []schema.GroupResource{
			groupVersions.WithResource(constant.ResourcesPluralCluster).GroupResource(),
		},
		RequestInfoFactory: &k8srequest.RequestInfoFactory{
			GrouplessAPIPrefixes: sets.NewString("api", "kapi"),
			APIPrefixes:          sets.NewString("api", "apis", "kapis", "kapi"),
		},
	}

	return requestInfoResolver
}

func TestRequestInfoFactoryNewRequestInfo(arg *testing.T) {
	requestInfoResolver := newTestRequestInfoResolver()

	for _, testReq := range testRequests {
		arg.Run(testReq.reqAdr, func(arg *testing.T) {
			genReq, genErr := http.NewRequest(testReq.reqMtd, testReq.reqAdr, nil)
			if genErr != nil {
				arg.Fatal(genErr)
			}
			clientMake := fake.NewSimpleClientset()
			genReqInfo, genErr := requestInfoResolver.NewRequestInfo(genReq, clientMake)

			if genErr == nil {
				if genReqInfo.IsResourceRequest != testReq.reqEptIsResourceReq {
					arg.Errorf("%s: expected is resource request %v, actual %+v", testReq.reqID,
						testReq.reqEptIsResourceReq, genReqInfo.IsResourceRequest)
				}
				if genReqInfo.IsK8sRequest != testReq.reqEptK8SReq {
					arg.Errorf("%s: expected kubernetes request %v, actual %+v", testReq.reqID,
						testReq.reqEptK8SReq, genReqInfo.IsK8sRequest)
				}
				if genReqInfo.ClusterName != testReq.reqEptClt {
					arg.Errorf("%s: expected cluster %v, actual %+v", testReq.reqID,
						testReq.reqEptClt, genReqInfo.ClusterName)
				}
				if testReq.reqEptVerb != genReqInfo.Verb {
					arg.Errorf("%s: expected verb %v, actual %+v", testReq.reqID,
						testReq.reqEptVerb, genReqInfo.RequestInfo.Verb)
				}
				if genReqInfo.Namespace != testReq.reqEptNamespace {
					arg.Errorf("%s: expected namespace %v, actual %+v", testReq.reqID,
						testReq.reqEptNamespace, genReqInfo.RequestInfo.Namespace)
				}
				if genReqInfo.Resource != testReq.reqEptResource {
					arg.Errorf("%s: expected resource %v, actual %+v", testReq.reqID,
						testReq.reqEptResource, genReqInfo.RequestInfo.Resource)
				}

			} else {
				if testReq.reqEptError != genErr {
					arg.Errorf("%s: expected error %v, actual %v", testReq.reqID,
						testReq.reqEptError, genErr)
				}
			}
		})

	}
}

func TestExtractCluster(t *testing.T) {
	resolver := newTestRequestInfoResolver()

	tests := []struct {
		name        string
		path        string
		wantCluster string
		wantPath    string
		wantOK      bool
	}{
		{
			name:        "cluster_request",
			path:        "/clusters/my-cluster/api/kubernetes/api/v1/pods",
			wantCluster: "my-cluster",
			wantPath:    "/api/kubernetes/api/v1/pods",
			wantOK:      true,
		},
		{
			name:        "non_cluster_path",
			path:        "/api/v1/pods",
			wantCluster: "",
			wantPath:    "/api/v1/pods",
			wantOK:      false,
		},
		{
			name:        "short_path",
			path:        "/clusters",
			wantCluster: "",
			wantPath:    "/clusters",
			wantOK:      false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cluster, path, ok := resolver.extractCluster(tt.path)
			if ok != tt.wantOK {
				t.Errorf("extractCluster() ok = %v, want %v", ok, tt.wantOK)
			}
			if cluster != tt.wantCluster {
				t.Errorf("extractCluster() cluster = %q, want %q", cluster, tt.wantCluster)
			}
			if path != tt.wantPath {
				t.Errorf("extractCluster() path = %q, want %q", path, tt.wantPath)
			}
		})
	}
}

func TestSetClusterProxyURL(t *testing.T) {
	resolver := newTestRequestInfoResolver()

	t.Run("nil_client", func(t *testing.T) {
		info := &RequestInfo{ClusterName: "cluster-a"}
		resolver.setClusterProxyURL(info, nil)
		if info.ClusterProxyURL != "" || info.ClusterProxyHost != "" {
			t.Error("setClusterProxyURL() should not set proxy fields when client is nil")
		}
	})

	t.Run("karmada_service_exists", func(t *testing.T) {
		client := fake.NewSimpleClientset(&corev1.Service{
			ObjectMeta: metav1.ObjectMeta{
				Name:      constant.KaramdaAPIServer,
				Namespace: constant.KaramdaNamespace,
			},
		})
		info := &RequestInfo{ClusterName: "cluster-a"}
		resolver.setClusterProxyURL(info, client)
		if info.ClusterProxyHost != constant.MultiClusterProxyHost {
			t.Errorf("ClusterProxyHost = %q, want %q", info.ClusterProxyHost, constant.MultiClusterProxyHost)
		}
		if info.ClusterProxyScheme != constant.MultiClusterProxyScheme {
			t.Errorf("ClusterProxyScheme = %q, want %q", info.ClusterProxyScheme, constant.MultiClusterProxyScheme)
		}
		if info.ClusterProxyURL == "" {
			t.Error("ClusterProxyURL should be set when karmada service exists")
		}
	})
}

func TestResolveResourceScope(t *testing.T) {
	resolver := newTestRequestInfoResolver()

	tests := []struct {
		name string
		info RequestInfo
		want string
	}{
		{
			name: "global_resource",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					APIGroup: "resources.fuyao.io",
					Resource: constant.ResourcesPluralCluster,
				},
			},
			want: globalScope,
		},
		{
			name: "namespace_scope",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					Namespace: "default",
					Resource:  "pods",
				},
			},
			want: namespaceScope,
		},
		{
			name: "cluster_scope",
			info: RequestInfo{
				RequestInfo: &k8srequest.RequestInfo{
					Resource: "nodes",
				},
			},
			want: clusterScope,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := resolver.resolveResourceScope(tt.info); got != tt.want {
				t.Errorf("resolveResourceScope() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestCheckK8sRequest(t *testing.T) {
	resolver := newTestRequestInfoResolver()

	req, err := http.NewRequest(http.MethodGet, "/api/kubernetes/api/v1/pods", nil)
	if err != nil {
		t.Fatal(err)
	}
	if !resolver.checkK8sRequest(req) {
		t.Error("checkK8sRequest() should return true for kubernetes API path")
	}
	if req.URL.Path != "/api/v1/pods" {
		t.Errorf("checkK8sRequest() path = %q, want /api/v1/pods", req.URL.Path)
	}

	nonK8sReq, err := http.NewRequest(http.MethodGet, "/rest/console/v1beta1/pods", nil)
	if err != nil {
		t.Fatal(err)
	}
	if resolver.checkK8sRequest(nonK8sReq) {
		t.Error("checkK8sRequest() should return false for non-kubernetes path")
	}
}

func TestRequestInfoContextHelpers(t *testing.T) {
	info := &RequestInfo{ClusterName: "test-cluster"}
	ctx := WithRequestInfo(context.Background(), info)

	got, ok := RequestInfoFrom(ctx)
	if !ok {
		t.Fatal("RequestInfoFrom() should find request info in context")
	}
	if got.ClusterName != "test-cluster" {
		t.Errorf("RequestInfoFrom() cluster = %q, want test-cluster", got.ClusterName)
	}

	got.SetSingleCluster()
	if got.ClusterName != "" || got.ClusterProxyURL != "" {
		t.Error("SetSingleCluster() should clear cluster proxy fields")
	}
}

func TestNewRequestInfoWithCluster(t *testing.T) {
	resolver := newTestRequestInfoResolver()
	client := fake.NewSimpleClientset()

	req, err := http.NewRequest(http.MethodGet, "/clusters/test-cluster/api/kubernetes/api/v1/pods", nil)
	if err != nil {
		t.Fatal(err)
	}
	info, err := resolver.NewRequestInfo(req, client)
	if err != nil {
		t.Fatalf("NewRequestInfo() error = %v", err)
	}
	if info.ClusterName != "test-cluster" {
		t.Errorf("ClusterName = %q, want test-cluster", info.ClusterName)
	}
	if !info.IsK8sRequest {
		t.Error("IsK8sRequest should be true for kubernetes API under cluster path")
	}
	if info.ResourceScope != clusterScope {
		t.Errorf("ResourceScope = %q, want %q", info.ResourceScope, clusterScope)
	}
}
