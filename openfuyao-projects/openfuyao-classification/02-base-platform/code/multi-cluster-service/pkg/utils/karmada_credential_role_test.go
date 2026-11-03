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
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sfake "k8s.io/client-go/kubernetes/fake"
)

// newProxyClusterRole builds a user clusterrole granting clusters/proxy access.
func newProxyClusterRole(username string, resourceNames []string) *rbacv1.ClusterRole {
	obj := &rbacv1.ClusterRole{
		Rules: []rbacv1.PolicyRule{
			{
				APIGroups:     []string{DefaultKarmdaClusterAPIGroup},
				Resources:     []string{"clusters/proxy"},
				ResourceNames: resourceNames,
				Verbs:         []string{"*"},
			},
		},
	}
	obj.Name = DefaultUserClusterRolePrefix + username
	obj.Labels = map[string]string{
		CreatedByopenFuyaoLabel: CreatedByopenFuyaoLabelValue,
		DefaultopenFuyaoLabel:   DefaultMemCusterRoleLabel,
	}
	return obj
}

func testKubeSystemNS() *corev1.Namespace {
	ns := &corev1.Namespace{}
	ns.Name = "kube-system"
	ns.UID = "test-cluster-uid-0001"
	ns.Status.Phase = corev1.NamespaceActive
	return ns
}

func TestUpdatePlatformRoleExtra(t *testing.T) {
	t.Run("stale role removed and new created", func(t *testing.T) {
		stale := &rbacv1.ClusterRole{}
		stale.Name = DefaultUserPlatformRolePrefix + "olduser"
		cs := k8sfake.NewSimpleClientset(stale)

		crList := &rbacv1.ClusterRoleList{Items: []rbacv1.ClusterRole{*stale}}
		if err := UpdatePlatformRole(cs, crList, []string{"newuser"}); err != nil {
			t.Fatalf("UpdatePlatformRole() error = %v", err)
		}

		if _, err := cs.RbacV1().ClusterRoles().Get(testCtx, DefaultUserPlatformRolePrefix+"olduser", metav1.GetOptions{}); !apierrors.IsNotFound(err) {
			t.Errorf("stale platform role still exists: %v", err)
		}
		if _, err := cs.RbacV1().ClusterRoles().Get(testCtx, DefaultUserPlatformRolePrefix+"newuser", metav1.GetOptions{}); err != nil {
			t.Errorf("new platform role not created: %v", err)
		}
	})

	t.Run("kept user keeps role", func(t *testing.T) {
		kept := &rbacv1.ClusterRole{}
		kept.Name = DefaultUserPlatformRolePrefix + "alice"
		cs := k8sfake.NewSimpleClientset(kept)

		crList := &rbacv1.ClusterRoleList{Items: []rbacv1.ClusterRole{*kept}}
		if err := UpdatePlatformRole(cs, crList, []string{"alice"}); err != nil {
			t.Fatalf("UpdatePlatformRole() error = %v", err)
		}
		if _, err := cs.RbacV1().ClusterRoles().Get(testCtx, DefaultUserPlatformRolePrefix+"alice", metav1.GetOptions{}); err != nil {
			t.Errorf("kept platform role was removed: %v", err)
		}
	})
}

func TestUpdateKarmadaClusterroleExtra(t *testing.T) {
	t.Run("no matching roles", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		if err := updateKarmadaClusterrole(cs, "member1"); err != nil {
			t.Errorf("updateKarmadaClusterrole() error = %v", err)
		}
	})

	t.Run("role with member tag removed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newProxyClusterRole("alice", []string{"member1"}))
		if err := updateKarmadaClusterrole(cs, "member1"); err != nil {
			t.Fatalf("updateKarmadaClusterrole() error = %v", err)
		}
		if _, err := cs.RbacV1().ClusterRoles().Get(testCtx, DefaultUserClusterRolePrefix+"alice", metav1.GetOptions{}); !apierrors.IsNotFound(err) {
			t.Errorf("clusterrole should be deleted when last member removed: %v", err)
		}
	})

	t.Run("role without member tag unchanged", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newProxyClusterRole("bob", []string{"member2"}))
		if err := updateKarmadaClusterrole(cs, "member1"); err != nil {
			t.Fatalf("updateKarmadaClusterrole() error = %v", err)
		}
		if _, err := cs.RbacV1().ClusterRoles().Get(testCtx, DefaultUserClusterRolePrefix+"bob", metav1.GetOptions{}); err != nil {
			t.Errorf("clusterrole for other members should be kept: %v", err)
		}
	})

	t.Run("role without proxy rule errors", func(t *testing.T) {
		role := newProxyClusterRole("carol", nil)
		role.Rules = nil
		cs := k8sfake.NewSimpleClientset(role)
		if err := updateKarmadaClusterrole(cs, ""); err == nil {
			t.Error("updateKarmadaClusterrole() expected error for role without clusters/proxy rule")
		}
	})
}

func TestIsClusterUIDUniqueExtra(t *testing.T) {
	t.Run("uid unique", func(t *testing.T) {
		k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{"kube-system": testKubeSystemNS()})
		kVersioned := newFakeKarmadaAPI().newClient(t)

		uid, err := IsClusterUIDUnique(k8sAPI.newClient(t), kVersioned)
		if err != nil {
			t.Fatalf("IsClusterUIDUnique() error = %v", err)
		}
		if uid != "test-cluster-uid-0001" {
			t.Errorf("uid = %q, want test-cluster-uid-0001", uid)
		}
	})

	t.Run("uid already registered", func(t *testing.T) {
		k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{"kube-system": testKubeSystemNS()})
		registered := newTestKarmadaCluster("member1", nil, false)
		registered.Spec.ID = "test-cluster-uid-0001"
		kVersioned := newFakeKarmadaAPI(registered).newClient(t)

		uid, err := IsClusterUIDUnique(k8sAPI.newClient(t), kVersioned)
		if err != nil {
			t.Fatalf("IsClusterUIDUnique() error = %v", err)
		}
		if uid != "" {
			t.Errorf("uid = %q, want empty for duplicate uid", uid)
		}
	})

	t.Run("karmada list failed", func(t *testing.T) {
		k8sAPI := newFakeK8sAPI(map[string]*corev1.Namespace{"kube-system": testKubeSystemNS()})
		api := newFakeKarmadaAPI()
		api.failList = 500

		if _, err := IsClusterUIDUnique(k8sAPI.newClient(t), api.newClient(t)); err == nil {
			t.Error("IsClusterUIDUnique() expected error when karmada list fails")
		}
	})
}
