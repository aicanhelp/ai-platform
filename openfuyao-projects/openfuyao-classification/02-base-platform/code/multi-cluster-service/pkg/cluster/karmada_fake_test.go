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

package cluster

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"

	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
)

var (
	debugStubRequests = os.Getenv("STUB_DEBUG") != ""
	tLog              = log.New(os.Stderr, "", log.LstdFlags)
)

// fakeKarmadaAPI is a lightweight in-memory karmada cluster.karmada.io API used to
// exercise real versioned.Clientset call paths without patching product code.
type fakeKarmadaAPI struct {
	mu         sync.Mutex
	clusters   map[string]*v1alpha1.Cluster
	failGet    map[string]int // cluster name -> status code for GET
	failList   int
	failPut    int
	failDelete int
}

func newFakeKarmadaAPI(clusters ...*v1alpha1.Cluster) *fakeKarmadaAPI {
	f := &fakeKarmadaAPI{
		clusters: make(map[string]*v1alpha1.Cluster),
		failGet:  make(map[string]int),
	}
	for _, c := range clusters {
		f.clusters[c.Name] = c
	}
	return f
}

func (f *fakeKarmadaAPI) newClient(t *testing.T) *versioned.Clientset {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	cs, err := versioned.NewForConfig(&rest.Config{Host: srv.URL})
	if err != nil {
		t.Fatalf("new karmada client: %v", err)
	}
	return cs
}

func writeJSON(w http.ResponseWriter, code int, obj interface{}) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(obj)
}

func writeStatus(w http.ResponseWriter, code int, reason, message string) {
	writeJSON(w, code, map[string]interface{}{
		"kind":       "Status",
		"apiVersion": "v1",
		"status":     "Failure",
		"reason":     reason,
		"message":    message,
		"code":       code,
	})
}

func (f *fakeKarmadaAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	const prefix = "/apis/cluster.karmada.io/v1alpha1/clusters"
	if !strings.HasPrefix(r.URL.Path, prefix) {
		writeStatus(w, http.StatusNotFound, "NotFound", "unknown path")
		return
	}
	name := strings.Trim(strings.TrimPrefix(r.URL.Path, prefix), "/")

	f.mu.Lock()
	defer f.mu.Unlock()

	switch r.Method {
	case http.MethodGet:
		if name == "" {
			if f.failList != 0 {
				writeStatus(w, f.failList, "InternalError", "list failed")
				return
			}
			items := make([]*v1alpha1.Cluster, 0, len(f.clusters))
			for _, c := range f.clusters {
				items = append(items, c)
			}
			writeJSON(w, http.StatusOK, map[string]interface{}{
				"kind":       "ClusterList",
				"apiVersion": "cluster.karmada.io/v1alpha1",
				"metadata":   map[string]string{"resourceVersion": "1"},
				"items":      items,
			})
			return
		}
		if code, ok := f.failGet[name]; ok {
			writeStatus(w, code, "InternalError", "get failed")
			return
		}
		c, ok := f.clusters[name]
		if !ok {
			writeStatus(w, http.StatusNotFound, "NotFound", "cluster not found")
			return
		}
		writeJSON(w, http.StatusOK, c)
	case http.MethodPost:
		body, _ := io.ReadAll(r.Body)
		var c v1alpha1.Cluster
		if err := json.Unmarshal(body, &c); err != nil {
			writeStatus(w, http.StatusBadRequest, "BadRequest", err.Error())
			return
		}
		f.clusters[c.Name] = &c
		writeJSON(w, http.StatusCreated, &c)
	case http.MethodPut:
		if f.failPut != 0 {
			writeStatus(w, f.failPut, "InternalError", "put failed")
			return
		}
		body, _ := io.ReadAll(r.Body)
		var c v1alpha1.Cluster
		if err := json.Unmarshal(body, &c); err != nil {
			writeStatus(w, http.StatusBadRequest, "BadRequest", err.Error())
			return
		}
		f.clusters[c.Name] = &c
		writeJSON(w, http.StatusOK, &c)
	case http.MethodDelete:
		if f.failDelete != 0 {
			writeStatus(w, f.failDelete, "InternalError", "delete failed")
			return
		}
		delete(f.clusters, name)
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"kind": "Status", "apiVersion": "v1", "status": "Success",
		})
	default:
		writeStatus(w, http.StatusMethodNotAllowed, "MethodNotAllowed", "unsupported method")
	}
}

// newTestKarmadaCluster builds a karmada cluster object with resource summary.
func newTestKarmadaCluster(name string, labels map[string]string, withSummary bool) *v1alpha1.Cluster {
	c := &v1alpha1.Cluster{}
	c.Name = name
	c.Labels = labels
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

// fakeK8sAPI is a minimal kubernetes API stub serving the endpoints touched by
// kubeconfig-based clients created in these tests.
type fakeK8sAPI struct {
	mu          sync.Mutex
	srv         *httptest.Server
	namespaces  map[string]*corev1.Namespace
	deployments map[string]bool // "namespace/name" -> exists
	crbNames    []string
	failCRB     int
}

func newFakeK8sAPI(namespaces map[string]*corev1.Namespace, crbNames []string) *fakeK8sAPI {
	if namespaces == nil {
		namespaces = map[string]*corev1.Namespace{}
	}
	return &fakeK8sAPI{namespaces: namespaces, deployments: map[string]bool{}, crbNames: crbNames}
}

// start launches the stub server; it is closed automatically with the test.
func (f *fakeK8sAPI) start(t *testing.T) {
	t.Helper()
	if f.srv != nil {
		return
	}
	srv := httptest.NewServer(f)
	f.srv = srv
	t.Cleanup(srv.Close)
}

// kubeconfig returns a kubeconfig pointing at the stub server.
func (f *fakeK8sAPI) kubeconfig(t *testing.T) []byte {
	t.Helper()
	f.start(t)
	return fmt.Appendf(nil, `{
		"apiVersion": "v1",
		"kind": "Config",
		"current-context": "test-context",
		"clusters": [{"name": "test-cluster", "cluster": {"server": %q}}],
		"contexts": [{"name": "test-context", "context": {"cluster": "test-cluster", "user": "test-user"}}],
		"users": [{"name": "test-user", "user": {}}]
	}`, f.srv.URL)
}

func (f *fakeK8sAPI) newClient(t *testing.T) *kubernetes.Clientset {
	t.Helper()
	f.start(t)
	cs, err := kubernetes.NewForConfig(&rest.Config{Host: f.srv.URL})
	if err != nil {
		t.Fatalf("new kubernetes client: %v", err)
	}
	return cs
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

func (f *fakeK8sAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if debugStubRequests {
		tLog.Printf("k8sAPI %s %s", r.Method, r.URL.Path)
	}

	switch {
	case r.URL.Path == "/api/v1/namespaces/kube-system":
		ns, ok := f.namespaces["kube-system"]
		if !ok {
			writeStatus(w, http.StatusNotFound, "NotFound", "namespace not found")
			return
		}
		writeJSON(w, http.StatusOK, ns)
	case r.URL.Path == "/api/v1/namespaces/karmada-system":
		ns, ok := f.namespaces["karmada-system"]
		if !ok {
			writeStatus(w, http.StatusNotFound, "NotFound", "namespace not found")
			return
		}
		writeJSON(w, http.StatusOK, ns)
	case r.URL.Path == "/api/v1/namespaces/karmada-cluster":
		ns, ok := f.namespaces["karmada-cluster"]
		if !ok {
			writeStatus(w, http.StatusNotFound, "NotFound", "namespace not found")
			return
		}
		writeJSON(w, http.StatusOK, ns)
	case r.Method == http.MethodPost && r.URL.Path == "/api/v1/namespaces":
		body, _ := io.ReadAll(r.Body)
		var created corev1.Namespace
		if err := json.Unmarshal(body, &created); err != nil || created.Name == "" {
			writeStatus(w, http.StatusBadRequest, "BadRequest", "invalid namespace")
			return
		}
		created.Status.Phase = corev1.NamespaceActive
		f.namespaces[created.Name] = &created
		writeJSON(w, http.StatusCreated, &created)
	case strings.HasPrefix(r.URL.Path, "/apis/apps/v1/namespaces/") && strings.Contains(r.URL.Path, "/deployments/"):
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/apis/apps/v1/namespaces/"), "/deployments/")
		if len(parts) == 2 && f.deployments[parts[0]+"/"+parts[1]] {
			deploy := map[string]interface{}{
				"kind": "Deployment", "apiVersion": "apps/v1",
				"metadata": map[string]interface{}{"name": parts[1], "namespace": parts[0]},
			}
			writeJSON(w, http.StatusOK, deploy)
			return
		}
		writeStatus(w, http.StatusNotFound, "NotFound", "deployment not found")
	case r.URL.Path == "/apis/rbac.authorization.k8s.io/v1/clusterrolebindings":
		if f.failCRB != 0 {
			writeStatus(w, f.failCRB, "InternalError", "list crb failed")
			return
		}
		items := make([]rbacv1.ClusterRoleBinding, 0, len(f.crbNames))
		for _, name := range f.crbNames {
			item := rbacv1.ClusterRoleBinding{}
			item.Name = name
			items = append(items, item)
		}
		writeJSON(w, http.StatusOK, map[string]interface{}{
			"kind":       "ClusterRoleBindingList",
			"apiVersion": "rbac.authorization.k8s.io/v1",
			"metadata":   map[string]string{"resourceVersion": "1"},
			"items":      items,
		})
	default:
		writeStatus(w, http.StatusNotFound, "NotFound", "unknown path "+r.URL.Path)
	}
}

// testKubeSystemNS returns a kube-system namespace object with a fixed UID.
func testKubeSystemNS() *corev1.Namespace {
	ns := &corev1.Namespace{}
	ns.Name = "kube-system"
	ns.UID = "test-cluster-uid-0001"
	ns.Status.Phase = corev1.NamespaceActive
	return ns
}
