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
	"errors"
	"testing"

	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/runtime"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
)

func TestCheckUserKubeconfigAccess(t *testing.T) {
	t.Run("access pass returns kubeconfig", func(t *testing.T) {
		k8sAPI := newFakeK8sAPI(nil, []string{"alice-platform-admin"})
		kubeconfig := k8sAPI.kubeconfigMapForm(t)

		mgrCluster := newTestMgrCluster("member1", "4", []byte(`{"out":"config"}`))
		mgrCluster.Spec.KubeConfig = runtime.RawExtension{Raw: kubeconfig}
		mgr := newTestMgrClient(t, mgrCluster)

		c := &OperationClient{MgrClient: mgr}
		pass, config, err := c.CheckUserKubeconfigAccess("alice", "member1")
		if err != nil {
			t.Fatalf("CheckUserKubeconfigAccess() error = %v", err)
		}
		if !pass {
			t.Error("pass = false, want true")
		}
		if string(config) != `{"out":"config"}` {
			t.Errorf("config = %s", config)
		}
	})

	t.Run("access denied", func(t *testing.T) {
		k8sAPI := newFakeK8sAPI(nil, nil)
		kubeconfig := k8sAPI.kubeconfigMapForm(t)

		mgrCluster := newTestMgrCluster("member1", "4", nil)
		mgrCluster.Spec.KubeConfig = runtime.RawExtension{Raw: kubeconfig}
		mgr := newTestMgrClient(t, mgrCluster)

		c := &OperationClient{MgrClient: mgr}
		pass, _, err := c.CheckUserKubeconfigAccess("bob", "member1")
		if err != nil {
			t.Fatalf("CheckUserKubeconfigAccess() error = %v", err)
		}
		if pass {
			t.Error("pass = true, want false")
		}
	})

	t.Run("cluster not found", func(t *testing.T) {
		mgr := newTestMgrClient(t)
		c := &OperationClient{MgrClient: mgr}

		if _, _, err := c.CheckUserKubeconfigAccess("bob", "ghost"); err == nil {
			t.Error("expected error for missing cluster object")
		}
	})

	t.Run("invalid kubeconfig data", func(t *testing.T) {
		mgr := newTestMgrClient(t, newTestMgrCluster("member1", "4", []byte("{invalid")))
		c := &OperationClient{MgrClient: mgr}

		if _, _, err := c.CheckUserKubeconfigAccess("bob", "member1"); err == nil {
			t.Error("expected error for invalid kubeconfig data")
		}
	})
}

func TestCheckClusterrolebindingForKubeconfigAccess(t *testing.T) {
	newBinding := func(name string) *rbacv1.ClusterRoleBinding {
		b := &rbacv1.ClusterRoleBinding{}
		b.Name = name
		return b
	}

	t.Run("platform admin binding found", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newBinding("alice-platform-admin"))
		pass, err := checkClusterrolebinding(cs, "alice")
		if err != nil || !pass {
			t.Errorf("pass = %v, err = %v, want true/nil", pass, err)
		}
	})

	t.Run("cluster admin binding found", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newBinding("bob-cluster-admin"))
		pass, err := checkClusterrolebinding(cs, "bob")
		if err != nil || !pass {
			t.Errorf("pass = %v, err = %v, want true/nil", pass, err)
		}
	})

	t.Run("no binding", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		pass, err := checkClusterrolebinding(cs, "bob")
		if err != nil || pass {
			t.Errorf("pass = %v, err = %v, want false/nil", pass, err)
		}
	})

	t.Run("list failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("list", "clusterrolebindings",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if _, err := checkClusterrolebinding(cs, "bob"); err == nil {
			t.Error("expected error when listing clusterrolebindings fails")
		}
	})
}
