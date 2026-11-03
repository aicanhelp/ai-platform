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

package controller

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/karmada-io/karmada/pkg/generated/clientset/versioned"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/manager"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
)

// fakeKarmada is a lightweight karmada cluster.karmada.io API stub so Reconcile
// exercises the real RetrieveClusterStatus path.
type fakeKarmada struct {
	mu       sync.Mutex
	clusters map[string]*v1alpha1Cluster
}

type v1alpha1Cluster struct {
	Name            string `json:"name"`
	ResourceSummary *struct {
		Allocatable map[string]string `json:"allocatable,omitempty"`
		Allocated   map[string]string `json:"allocated,omitempty"`
	} `json:"resourceSummary,omitempty"`
}

func newFakeKarmada(names ...string) *fakeKarmada {
	f := &fakeKarmada{clusters: map[string]*v1alpha1Cluster{}}
	for _, name := range names {
		c := &v1alpha1Cluster{Name: name}
		c.ResourceSummary = &struct {
			Allocatable map[string]string `json:"allocatable,omitempty"`
			Allocated   map[string]string `json:"allocated,omitempty"`
		}{
			Allocatable: map[string]string{"cpu": "4", "memory": "8Gi"},
			Allocated:   map[string]string{"cpu": "1", "memory": "2Gi"},
		}
		f.clusters[name] = c
	}
	return f
}

func (f *fakeKarmada) newClient(t *testing.T) *versioned.Clientset {
	t.Helper()
	srv := httptest.NewServer(f)
	t.Cleanup(srv.Close)
	cs, err := versioned.NewForConfig(&rest.Config{Host: srv.URL})
	if err != nil {
		t.Fatalf("new karmada client: %v", err)
	}
	return cs
}

func (f *fakeKarmada) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	const prefix = "/apis/cluster.karmada.io/v1alpha1/clusters/"
	name := strings.TrimPrefix(r.URL.Path, prefix)

	f.mu.Lock()
	defer f.mu.Unlock()

	if r.Method != http.MethodGet || name == "" {
		writeKarmadaStatus(w, http.StatusMethodNotAllowed, "unsupported")
		return
	}
	c, ok := f.clusters[name]
	if !ok {
		writeKarmadaStatus(w, http.StatusNotFound, "cluster not found")
		return
	}
	// serve the karmada Cluster shape: metadata + status nesting
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"kind":       "Cluster",
		"apiVersion": "cluster.karmada.io/v1alpha1",
		"metadata":   map[string]interface{}{"name": c.Name},
		"status": map[string]interface{}{
			"resourceSummary": c.ResourceSummary,
		},
	})
}

func writeKarmadaStatus(w http.ResponseWriter, code int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"kind": "Status", "apiVersion": "v1", "status": "Failure",
		"reason": "NotFound", "message": message, "code": code,
	})
}

func newReconcileClient(t *testing.T, clusters ...*v1beta1.Cluster) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	builder := fake.NewClientBuilder().WithScheme(scheme).WithStatusSubresource(&v1beta1.Cluster{})
	for _, c := range clusters {
		builder = builder.WithObjects(c)
	}
	return builder.Build()
}

func newTestCluster(name string) *v1beta1.Cluster {
	c := &v1beta1.Cluster{}
	c.Name = name
	return c
}

func TestReconcile(t *testing.T) {
	req := ctrl.Request{NamespacedName: types.NamespacedName{Name: "member1"}}

	t.Run("cluster object not found", func(t *testing.T) {
		r := &ClusterReconciler{Client: newReconcileClient(t), Scheme: runtime.NewScheme()}
		result, err := r.Reconcile(context.TODO(), req)
		if err != nil {
			t.Errorf("Reconcile() error = %v, want nil", err)
		}
		if result.RequeueAfter != 0 {
			t.Errorf("RequeueAfter = %v, want 0", result.RequeueAfter)
		}
	})

	t.Run("cluster object not ready syncs error", func(t *testing.T) {
		karmada := newFakeKarmada() // no clusters -> not ready
		r := &ClusterReconciler{
			Client:        newReconcileClient(t, newTestCluster("member1")),
			Scheme:        runtime.NewScheme(),
			KarmadaClient: karmada.newClient(t),
		}
		if _, err := r.Reconcile(context.TODO(), req); err == nil {
			t.Error("Reconcile() expected error when karmada cluster is not ready")
		}
	})

	t.Run("sync cluster status success", func(t *testing.T) {
		karmada := newFakeKarmada("member1")
		mgrClient := newReconcileClient(t, newTestCluster("member1"))
		r := &ClusterReconciler{
			Client:        mgrClient,
			Scheme:        runtime.NewScheme(),
			KarmadaClient: karmada.newClient(t),
		}
		result, err := r.Reconcile(context.TODO(), req)
		if err != nil {
			t.Fatalf("Reconcile() error = %v", err)
		}
		if result.RequeueAfter == 0 {
			t.Error("RequeueAfter = 0, want periodic requeue")
		}

		updated := &v1beta1.Cluster{}
		if err := mgrClient.Get(context.TODO(), client.ObjectKey{Name: "member1"}, updated); err != nil {
			t.Fatalf("get cluster: %v", err)
		}
		if updated.Status.ResourceList.AllocatableCPU != "4.0" {
			t.Errorf("AllocatableCPU = %q, want 4.0", updated.Status.ResourceList.AllocatableCPU)
		}
	})
}

func TestSetupWithManager(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	mgr, err := manager.New(&rest.Config{Host: "127.0.0.1:1"}, manager.Options{Scheme: scheme})
	if err != nil {
		t.Fatalf("manager.New() error = %v", err)
	}
	r := &ClusterReconciler{Client: mgr.GetClient(), Scheme: scheme}
	if err := r.SetupWithManager(mgr); err != nil {
		// With -count>1 the test re-runs in the same process and the manager
		// rejects re-registering the same controller name. Tolerate exactly
		// that case; every other error must still fail the test.
		if strings.Contains(err.Error(), "already exists") {
			return
		}
		t.Errorf("SetupWithManager() error = %v", err)
	}
}
