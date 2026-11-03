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
	"errors"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	"openfuyao.com/multi-cluster-service/api/v1beta1"
)

func TestOperationClientEdit(t *testing.T) {
	ctx := context.TODO()
	opt := EditOptions{ClusterName: "member1", ClusterLabels: map[string]string{"env": "test"}}

	t.Run("success", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		kVersioned := api.newClient(t)
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", nil))
		c := &OperationClient{KarmadaVersionedClient: kVersioned, MgrClient: mgr}

		if err := c.Edit(ctx, opt); err != nil {
			t.Fatalf("Edit() error = %v", err)
		}

		mgrCluster := &v1beta1.Cluster{}
		if err := mgr.Get(ctx, client.ObjectKey{Name: "member1"}, mgrCluster); err != nil {
			t.Fatalf("get mgr cluster: %v", err)
		}
		if mgrCluster.Labels["env"] != "test" {
			t.Errorf("mgr labels = %v, want env=test", mgrCluster.Labels)
		}

		got, err := kVersioned.ClusterV1alpha1().Clusters().Get(ctx, "member1", metav1.GetOptions{})
		if err != nil {
			t.Fatalf("get karmada cluster: %v", err)
		}
		if got.Labels["env"] != "test" {
			t.Errorf("karmada labels = %v, want env=test", got.Labels)
		}
	})

	t.Run("mgr cluster get failed", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		mgr := newTestMgrClient(t)
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		if err := c.Edit(ctx, opt); err == nil {
			t.Error("Edit() expected error when mgr cluster missing")
		}
	})

	t.Run("cluster ID no exists", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", nil))
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		err := c.Edit(ctx, opt)
		if err == nil || err.Error() != "cluster ID no exists" {
			t.Errorf("error = %v, want cluster ID no exists", err)
		}
	})

	t.Run("mgr update failed", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		mgr := fake.NewClientBuilder().WithScheme(newTestScheme(t)).
			WithObjects(newTestMgrCluster("member1", "4", nil)).
			WithInterceptorFuncs(interceptor.Funcs{
				Update: func(context.Context, client.WithWatch, client.Object, ...client.UpdateOption) error {
					return errors.New("update boom")
				},
			}).Build()
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		if err := c.Edit(ctx, opt); err == nil {
			t.Error("Edit() expected error when mgr update fails")
		}
	})

	t.Run("karmada update failed", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		api.failPut = 500
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", nil))
		c := &OperationClient{KarmadaVersionedClient: api.newClient(t), MgrClient: mgr}

		if err := c.Edit(ctx, opt); err == nil {
			t.Error("Edit() expected error when karmada update fails")
		}
	})
}

func TestUnregsiterHostCluster(t *testing.T) {
	t.Run("success", func(t *testing.T) {
		api := newFakeKarmadaAPI(newTestKarmadaCluster("member1", nil, true))
		if err := unregsiterHostCluster(api.newClient(t), "member1"); err != nil {
			t.Errorf("unregsiterHostCluster() error = %v", err)
		}
	})

	t.Run("delete failed", func(t *testing.T) {
		api := newFakeKarmadaAPI()
		api.failDelete = 500
		if err := unregsiterHostCluster(api.newClient(t), "member1"); err == nil {
			t.Error("unregsiterHostCluster() expected error on delete failure")
		}
	})
}
