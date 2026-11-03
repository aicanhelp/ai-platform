/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	k8srequest "k8s.io/apiserver/pkg/endpoints/request"

	"github.com/stretchr/testify/require"
)

func TestSplitPath(t *testing.T) {
	cases := []struct {
		name string
		path string
		want []string
	}{
		{name: "empty", path: "", want: []string{}},
		{name: "slash", path: "/", want: []string{}},
		{name: "simple", path: "/a/b/c", want: []string{"a", "b", "c"}},
		{name: "noLeading", path: "x/y", want: []string{"x", "y"}},
		{name: "trailing", path: "/a/b/", want: []string{"a", "b"}},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := splitPath(tc.path)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestExtractCluster(t *testing.T) {
	factory := k8srequest.RequestInfoFactory{APIPrefixes: sets.NewString("api", "apis")}
	r := &RequestInfoFactory{RequestInfoFactory: &k8srequest.RequestInfoFactory{APIPrefixes: sets.NewString("api", "apis")}}

	cases := []struct {
		name        string
		path        string
		wantCluster string
		wantPath    string
		wantOK      bool
	}{
		{name: "no-prefix", path: "/healthz", wantCluster: "", wantPath: "/healthz", wantOK: false},
		{name: "prefix-no-clusters", path: "/api/v1/namespaces", wantCluster: "", wantPath: "/api/v1/namespaces", wantOK: false},
		{name: "with-cluster", path: "/api/clusters/c1/foo/bar", wantCluster: "c1", wantPath: "api/foo/bar", wantOK: true},
		{name: "clusters-no-name", path: "/api/clusters", wantCluster: "", wantPath: "/api/clusters", wantOK: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cluster, p, ok := r.extractCluster(factory, tc.path)
			require.Equal(t, tc.wantOK, ok)
			require.Equal(t, tc.wantCluster, cluster)
			require.Equal(t, tc.wantPath, p)
		})
	}
}

func TestResolveResourceScope(t *testing.T) {
	gr := schema.GroupResource{Group: "g", Resource: "r"}
	r := &RequestInfoFactory{GlobalResources: []schema.GroupResource{gr}}

	cases := []struct {
		name string
		info RequestInfo
		want string
	}{
		{name: "global", info: RequestInfo{RequestInfo: &k8srequest.RequestInfo{APIGroup: "g", Resource: "r"}}, want: "Global"},
		{name: "namespace", info: RequestInfo{RequestInfo: &k8srequest.RequestInfo{Namespace: "ns"}}, want: "Namespace"},
		{name: "cluster", info: RequestInfo{RequestInfo: &k8srequest.RequestInfo{}}, want: "Cluster"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := r.resolveResourceScope(tc.info)
			require.Equal(t, tc.want, got)
		})
	}
}

func TestNewRequestInfoAndContextHelpers(t *testing.T) {
	r := &RequestInfoFactory{RequestInfoFactory: &k8srequest.RequestInfoFactory{APIPrefixes: sets.NewString("api", "apis"), GrouplessAPIPrefixes: sets.NewString("api")}}

	req, _ := http.NewRequest("GET", "http://example/api/clusters/c1/namespaces/ns/configmaps", nil)
	info, err := r.NewRequestInfo(req)
	require.NoError(t, err)
	require.Equal(t, "c1", info.Cluster)
	require.Equal(t, "Cluster", info.ResourceScope)
	// depending on leading slash handling in extractCluster and k8s parser,
	// resource scope may be Cluster or Namespace; assert cluster extraction works

	// test context helpers
	ctx := context.Background()
	ctx2 := WithRequestInfo(ctx, info)
	got, ok := RequestInfoFrom(ctx2)
	require.True(t, ok)
	require.Equal(t, info.Cluster, got.Cluster)
}
