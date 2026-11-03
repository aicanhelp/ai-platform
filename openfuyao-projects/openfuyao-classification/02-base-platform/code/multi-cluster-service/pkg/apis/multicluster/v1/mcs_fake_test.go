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
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
)

// fakeMcsKarmada is a lightweight karmada cluster.karmada.io API stub so real
// versioned.Clientset call paths are exercised from handler tests.
type fakeMcsKarmada struct {
	mu       sync.Mutex
	clusters map[string]*v1alpha1.Cluster
	failGet  map[string]int
	failList int
}

func newFakeMcsKarmada(clusters ...*v1alpha1.Cluster) *fakeMcsKarmada {
	f := &fakeMcsKarmada{clusters: make(map[string]*v1alpha1.Cluster), failGet: make(map[string]int)}
	for _, c := range clusters {
		f.clusters[c.Name] = c
	}
	return f
}

func (f *fakeMcsKarmada) newClient(t *testing.T) *versioned.Clientset {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	cs, err := versioned.NewForConfig(&rest.Config{Host: srv.URL})
	if err != nil {
		t.Fatalf("new karmada client: %v", err)
	}
	return cs
}

func writeMcsJSON(w http.ResponseWriter, code int, obj interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(obj)
}

func writeMcsStatus(w http.ResponseWriter, code int, reason, message string) {
	writeMcsJSON(w, code, map[string]interface{}{
		"kind": "Status", "apiVersion": "v1", "status": "Failure",
		"reason": reason, "message": message, "code": code,
	})
}

func (f *fakeMcsKarmada) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	const prefix = "/apis/cluster.karmada.io/v1alpha1/clusters"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		writeMcsStatus(w, http.StatusNotFound, "NotFound", "unknown path")
		return
	}
	name := strings.Trim(strings.TrimPrefix(r.URL.Path, prefix), "/")

	f.mu.Lock()
	defer f.mu.Unlock()

	switch r.Method {
	case http.MethodGet:
		if name == "" {
			if f.failList != 0 {
				writeMcsStatus(w, f.failList, "InternalError", "list failed")
				return
			}
			items := make([]*v1alpha1.Cluster, 0, len(f.clusters))
			for _, c := range f.clusters {
				items = append(items, c)
			}
			writeMcsJSON(w, http.StatusOK, map[string]interface{}{
				"kind": "ClusterList", "apiVersion": "cluster.karmada.io/v1alpha1",
				"metadata": map[string]string{"resourceVersion": "1"}, "items": items,
			})
			return
		}
		if code, ok := f.failGet[name]; ok {
			writeMcsStatus(w, code, "InternalError", "get failed")
			return
		}
		c, ok := f.clusters[name]
		if !ok {
			writeMcsStatus(w, http.StatusNotFound, "NotFound", "cluster not found")
			return
		}
		writeMcsJSON(w, http.StatusOK, c)
	case http.MethodPost:
		body, _ := io.ReadAll(r.Body)
		var c v1alpha1.Cluster
		if err := json.Unmarshal(body, &c); err != nil {
			writeMcsStatus(w, http.StatusBadRequest, "BadRequest", err.Error())
			return
		}
		f.clusters[c.Name] = &c
		writeMcsJSON(w, http.StatusCreated, &c)
	default:
		writeMcsStatus(w, http.StatusMethodNotAllowed, "MethodNotAllowed", "unsupported method")
	}
}

func newMcsKarmadaCluster(name string, withSummary bool) *v1alpha1.Cluster {
	c := &v1alpha1.Cluster{}
	c.Name = name
	if withSummary {
		c.Status.ResourceSummary = &v1alpha1.ResourceSummary{
			Allocatable: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("4"),
				corev1.ResourceMemory: resource.MustParse("8Gi"),
			},
			Allocated: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("1"),
				corev1.ResourceMemory: resource.MustParse("2Gi"),
			},
		}
	}
	return c
}

// fakeMcsK8s is a minimal kubernetes API stub for kubeconfig-based clients.
type fakeMcsK8s struct {
	mu         sync.Mutex
	srv        *httptest.Server
	namespaces map[string]*corev1.Namespace
	crbNames   []string
}

func newFakeMcsK8s(kubeSystemUID string, crbNames []string) *fakeMcsK8s {
	f := &fakeMcsK8s{namespaces: map[string]*corev1.Namespace{}, crbNames: crbNames}
	if kubeSystemUID != "" {
		ns := &corev1.Namespace{}
		ns.Name = "kube-system"
		ns.UID = "test-uid"
		if kubeSystemUID != "-" {
			ns.UID = types.UID(kubeSystemUID)
		}
		f.namespaces["kube-system"] = ns
	}
	return f
}

func (f *fakeMcsK8s) start(t *testing.T) {
	t.Helper()
	if f.srv != nil {
		return
	}
	srv := httptest.NewServer(f)
	f.srv = srv
	t.Cleanup(srv.Close)
}

// listKubeconfig returns a kubeconfig in file (list) shape, used as Join input.
func (f *fakeMcsK8s) listKubeconfig(t *testing.T) []byte {
	t.Helper()
	f.start(t)
	return fmt.Appendf(nil, `{
		"apiVersion": "v1", "kind": "Config", "current-context": "test-context",
		"clusters": [{"name": "test-cluster", "cluster": {"server": %q}}],
		"contexts": [{"name": "test-context", "context": {"cluster": "test-cluster", "user": "test-user"}}],
		"users": [{"name": "test-user", "user": {}}]
	}`, f.srv.URL)
}

// mapKubeconfig returns a kubeconfig in api.Config (map) shape, as stored in
// v1beta1.Cluster.Spec.KubeConfig.
func (f *fakeMcsK8s) mapKubeconfig(t *testing.T) []byte {
	t.Helper()
	f.start(t)
	return fmt.Appendf(nil, `{
		"kind": "Config", "apiVersion": "v1", "current-context": "test-context",
		"clusters": {"test-cluster": {"server": %q}},
		"contexts": {"test-context": {"cluster": "test-cluster", "user": "test-user"}},
		"users": {"test-user": {}}
	}`, f.srv.URL)
}

func (f *fakeMcsK8s) newClient(t *testing.T) *kubernetes.Clientset {
	t.Helper()
	f.start(t)
	cs, err := kubernetes.NewForConfig(&rest.Config{Host: f.srv.URL})
	if err != nil {
		t.Fatalf("new kubernetes client: %v", err)
	}
	return cs
}

func (f *fakeMcsK8s) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	switch {
	case r.URL.Path == "/api/v1/namespaces/kube-system":
		ns, ok := f.namespaces["kube-system"]
		if !ok {
			writeMcsStatus(w, http.StatusNotFound, "NotFound", "namespace not found")
			return
		}
		writeMcsJSON(w, http.StatusOK, ns)
	case r.URL.Path == "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings":
		items := make([]rbacv1.ClusterRoleBinding, 0, len(f.crbNames))
		for _, name := range f.crbNames {
			item := rbacv1.ClusterRoleBinding{}
			item.Name = name
			items = append(items, item)
		}
		writeMcsJSON(w, http.StatusOK, map[string]interface{}{
			"kind": "ClusterRoleBindingList", "apiVersion": "rbac.authorization.k8s.io/v1",
			"metadata": map[string]string{"resourceVersion": "1"}, "items": items,
		})
	default:
		writeMcsStatus(w, http.StatusNotFound, "NotFound", "unknown path "+r.URL.Path)
	}
}

// newMcsMgrClient returns a fake controller-runtime client for v1beta1 objects.
func newMcsMgrClient(t *testing.T, clusters ...*v1beta1.Cluster) client.Client {
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

func newMcsMgrCluster(name string, kubeconfigRaw, outConfigRaw []byte) *v1beta1.Cluster {
	c := &v1beta1.Cluster{}
	c.Name = name
	if kubeconfigRaw != nil {
		c.Spec.KubeConfig.Raw = kubeconfigRaw
	}
	if outConfigRaw != nil {
		c.Spec.OutClusterConfig.Raw = outConfigRaw
	}
	c.Status.ResourceList.AllocatableCPU = "4"
	c.Status.ResourceList.AllocatableMemory = "8Gi"
	c.Status.ResourceList.AllocatedCPU = "1"
	c.Status.ResourceList.AllocatedMemory = "2Gi"
	return c
}

// newV1beta1Cluster returns an empty named v1beta1.Cluster.
func newV1beta1Cluster(name string) *v1beta1.Cluster {
	c := &v1beta1.Cluster{}
	c.Name = name
	return c
}

// v1beta1ObjectKey builds an object key for v1beta1 clusters.
func v1beta1ObjectKey(name string) client.ObjectKey {
	return client.ObjectKey{Name: name}
}

// clustersSnapshot returns a defensive copy of the named cluster.
func (f *fakeMcsKarmada) clustersSnapshot(name string) (*v1alpha1.Cluster, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	c, ok := f.clusters[name]
	if !ok {
		return nil, nil
	}
	return c.DeepCopy(), nil
}
