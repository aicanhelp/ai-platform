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

package utils

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
)

var testCtx = context.Background()

// newTestMgrClient returns a fake controller-runtime client for v1beta1 objects.
func newTestMgrClient(t *testing.T, clusters ...*v1beta1.Cluster) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	builder := fake.NewClientBuilder().WithScheme(scheme)
	for _, c := range clusters {
		builder = builder.WithObjects(c)
	}
	return builder.Build()
}

// newTestMgrCluster returns a named v1beta1.Cluster object.
func newTestMgrCluster(name string) *v1beta1.Cluster {
	c := &v1beta1.Cluster{}
	c.Name = name
	return c
}

// fakeK8sAPI is a minimal API server stub serving the endpoints touched by
// kubeconfig-based clients created in these tests.
type fakeK8sAPI struct {
	mu         sync.Mutex
	srv        *httptest.Server
	namespaces map[string]*corev1.Namespace
}

func newFakeK8sAPI(namespaces map[string]*corev1.Namespace) *fakeK8sAPI {
	if namespaces == nil {
		namespaces = map[string]*corev1.Namespace{}
	}
	return &fakeK8sAPI{namespaces: namespaces}
}

func (f *fakeK8sAPI) start(t *testing.T) {
	t.Helper()
	if f.srv != nil {
		return
	}
	srv := httptest.NewServer(f)
	f.srv = srv
	t.Cleanup(srv.Close)
}

// newClient returns a real kubernetes clientset pointing at the stub server.
func (f *fakeK8sAPI) newClient(t *testing.T) *kubernetes.Clientset {
	t.Helper()
	f.start(t)
	cs, err := kubernetes.NewForConfig(&rest.Config{Host: f.srv.URL})
	if err != nil {
		t.Fatalf("new kubernetes client: %v", err)
	}
	return cs
}

func (f *fakeK8sAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	switch {
	case r.URL.Path == "/api/v1/namespaces/kube-system":
		ns, ok := f.namespaces["kube-system"]
		if !ok {
			writeStatus(w, http.StatusNotFound, "NotFound", "namespace not found")
			return
		}
		writeJSON(w, http.StatusOK, ns)
	default:
		writeStatus(w, http.StatusNotFound, "NotFound", "unknown path "+r.URL.Path)
	}
}

// kubeconfigMapForm returns a kubeconfig in the api.Config (map-based) shape,
// as expected when stored raw inside v1beta1.Cluster.Spec.KubeConfig.
func (f *fakeK8sAPI) kubeconfigMapForm(t *testing.T) []byte {
	t.Helper()
	f.start(t)
	return fmt.Appendf(nil, `{
		"kind": "Config",
		"apiVersion": "v1",
		"current-context": "test-context",
		"clusters": {"test-cluster": {"server": %q}},
		"contexts": {"test-context": {"cluster": "test-cluster", "user": "test-user"}},
		"users": {"test-user": {}}
	}`, f.srv.URL)
}
