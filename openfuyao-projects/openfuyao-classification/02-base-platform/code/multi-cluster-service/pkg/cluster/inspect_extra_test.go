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
	"context"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
)

func newTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	return scheme
}

// newTestMgrClient returns a fake mgr client seeded with the given clusters.
func newTestMgrClient(t *testing.T, clusters ...*v1beta1.Cluster) client.Client {
	t.Helper()
	builder := fake.NewClientBuilder().WithScheme(newTestScheme(t))
	for _, c := range clusters {
		builder = builder.WithObjects(c)
	}
	return builder.Build()
}

func newTestMgrCluster(name string, allocatableCPU string, outConfig []byte) *v1beta1.Cluster {
	c := &v1beta1.Cluster{}
	c.Name = name
	c.Spec.OutClusterConfig = runtime.RawExtension{Raw: outConfig}
	if allocatableCPU != "" {
		c.Status.ResourceList.AllocatableCPU = allocatableCPU
		c.Status.ResourceList.AllocatableMemory = "8Gi"
		c.Status.ResourceList.AllocatedCPU = "1"
		c.Status.ResourceList.AllocatedMemory = "2Gi"
	}
	return c
}

func TestInspectPrivate(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", map[string]string{"env": "prod"}, true))
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", []byte(`{"server":"https://1.2.3.4"}`)))

		info, err := inspect("member1", api.newClient(t), mgr)
		if err != nil {
			t.Fatalf("inspect() error = %v", err)
		}
		if info.ClusterName != "member1" {
			t.Errorf("ClusterName = %q", info.ClusterName)
		}
		if len(info.ClusterLabels) != 1 || info.ClusterLabels[0] != "env:prod" {
			t.Errorf("ClusterLabels = %v", info.ClusterLabels)
		}
		if info.ClusterResourcesUsage == nil || info.ClusterResourcesUsage.AllocatableCPU != "4" {
			t.Errorf("ClusterResourcesUsage = %+v", info.ClusterResourcesUsage)
		}
		if string(info.ClusterConfig) != `{"server":"https://1.2.3.4"}` {
			t.Errorf("ClusterConfig = %s", info.ClusterConfig)
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		mgr := newTestMgrClient(t)

		_, err := inspect("member1", api.newClient(t), mgr)
		if err == nil || err.Error() != "cluster ID no exists" {
			t.Errorf("error = %v, want cluster ID no exists", err)
		}
	})

	t.Run("karmada cluster get failed", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", nil))

		_, err := inspect("member1", api.newClient(t), mgr)
		if err == nil {
			t.Fatal("inspect() expected error when karmada cluster missing")
		}
	})

	t.Run("mgr cluster not ready yet", func(t *testing.T) {
		// mgr cluster exists but resource list is empty -> poll times out (10s) and
		// inspect degrades to partial information with "cluster object is not ready".
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "", nil))

		info, err := inspect("member1", api.newClient(t), mgr)
		if err == nil || err.Error() != "cluster object is not ready" {
			t.Errorf("error = %v, want cluster object is not ready", err)
		}
		if info == nil || info.ClusterName != "member1" {
			t.Errorf("info = %+v, want partial cluster information", info)
		}
		if info != nil && info.ClusterResourcesUsage == nil {
			t.Error("partial info should carry empty ResourceUsage")
		}
	})
}

func TestOperationClientInspect(t *testing.T) {
	ctx := context.TODO()

	t.Run("list all clusters", func(t *testing.T) {
		api := newFakeKarmadaAPI(
			newTestKarmadaCluster("host", nil, true),
			newTestKarmadaCluster("member1", map[string]string{"a": "b"}, true),
		)
		mgr := newTestMgrClient(t,
			newTestMgrCluster("host", "4", nil),
			newTestMgrCluster("member1", "4", nil),
		)
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		list, err := c.Inspect(ctx, InspectOptions{})
		if err != nil {
			t.Fatalf("Inspect() error = %v", err)
		}
		if _, ok := list.Info["member1"]; !ok {
			t.Errorf("Info = %v, want member1 entry", list.Info)
		}
	})

	t.Run("single cluster", func(t *testing.T) {
		api := newFakeKarmadaAPI(
			newTestKarmadaCluster("host", nil, true),
			newTestKarmadaCluster("member1", nil, true),
		)
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", nil))
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		list, err := c.Inspect(ctx, InspectOptions{ClusterName: "member1"})
		if err != nil {
			t.Fatalf("Inspect() error = %v", err)
		}
		if len(list.Info) != 1 {
			t.Errorf("Info = %v, want only member1", list.Info)
		}
	})

	t.Run("single cluster not found", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("host", nil, true))
		mgr := newTestMgrClient(t)
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		_, err := c.Inspect(ctx, InspectOptions{ClusterName: "ghost"})
		if err == nil || err.Error() != "cluster ID no exists" {
			t.Errorf("error = %v, want cluster ID no exists", err)
		}
	})

	t.Run("host cluster is being onboarded", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		mgr := newTestMgrClient(t)
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		_, err := c.Inspect(ctx, InspectOptions{})
		if err == nil || err.Error() != "host Cluster is being onboarded" {
			t.Errorf("error = %v, want host Cluster is being onboarded", err)
		}
	})

	t.Run("karmada service is not ready", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		api.failList = 500
		mgr := newTestMgrClient(t)
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		_, err := c.Inspect(ctx, InspectOptions{})
		if err == nil || err.Error() != "karmada service is not ready" {
			t.Errorf("error = %v, want karmada service is not ready", err)
		}
	})
}

func TestOperationClientInspection(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t)}

		usage, err := c.Inspection("member1")
		if err != nil {
			t.Fatalf("Inspection() error = %v", err)
		}
		if usage == nil || usage.Allocatable == nil {
			t.Errorf("usage = %+v", usage)
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t)}

		_, err := c.Inspection("ghost")
		if err == nil || err.Error() != "cluster ID no exists" {
			t.Errorf("error = %v, want cluster ID no exists", err)
		}
	})
}
