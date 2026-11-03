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
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/karmada-io/karmada/pkg/apis/cluster/v1alpha1"
	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/client-go/rest"
)

// fakeKarmadaAPI is a lightweight in-memory karmada cluster.karmada.io API used to
// exercise real versioned.Clientset call paths without patching product code.
type fakeKarmadaAPI struct {
	mu         sync.Mutex
	clusters   map[string]*v1alpha1.Cluster
	failList   int
	failDelete int
}

func newFakeKarmadaAPI(clusters ...*v1alpha1.Cluster) *fakeKarmadaAPI {
	f := &fakeKarmadaAPI{clusters: make(map[string]*v1alpha1.Cluster)}
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
