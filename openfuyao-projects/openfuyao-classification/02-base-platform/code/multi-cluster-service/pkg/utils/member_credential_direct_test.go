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
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	k8sfake "k8s.io/client-go/kubernetes/fake"
)

func TestGenerateClusterSecretDirect(t *testing.T) {
	// pre-create SA token secrets (in a real cluster the token controller
	// populates them; checkSecret polls for their presence)
	newTokenSecret := func(name string) *corev1.Secret {
		s := &corev1.Secret{}
		s.Name = name
		s.Namespace = DefaultKarmadaNamespace
		s.Data = map[string][]byte{"token": []byte("token-" + name)}
		s.Type = corev1.SecretTypeServiceAccountToken
		return s
	}

	t.Run("success provisions member cluster resources", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(
			newTokenSecret(DefaultSAPrefix+"-member1"),
			newTokenSecret(DefaultSAPrefix+"-"+DefaultImpersonatorPrefix),
		)

		clusterSecret, impersonatorSecret, err := GenerateClusterSecret(cs, "member1")
		if err != nil {
			t.Fatalf("GenerateClusterSecret() error = %v", err)
		}
		if clusterSecret == nil || impersonatorSecret == nil {
			t.Fatal("GenerateClusterSecret() returned nil secrets")
		}
		if len(clusterSecret.Data[TLSTokenData]) == 0 {
			t.Error("cluster secret carries no token data")
		}
		if len(impersonatorSecret.Data[TLSTokenData]) == 0 {
			t.Error("impersonator secret carries no token data")
		}

		// service accounts created in the member cluster namespace
		saList, err := cs.CoreV1().ServiceAccounts(DefaultKarmadaNamespace).List(testCtx, metav1.ListOptions{})
		if err != nil || len(saList.Items) < 2 {
			t.Errorf("service accounts not created: %v (%d items)", err, len(saList.Items))
		}

		// cluster roles and bindings created
		roleList, err := cs.RbacV1().ClusterRoles().List(testCtx, metav1.ListOptions{})
		if err != nil || len(roleList.Items) == 0 {
			t.Errorf("cluster roles not created: %v", err)
		}
		bindingList, err := cs.RbacV1().ClusterRoleBindings().List(testCtx, metav1.ListOptions{})
		if err != nil || len(bindingList.Items) == 0 {
			t.Errorf("cluster role bindings not created: %v", err)
		}
	})

	t.Run("idempotent on rerun", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(
			newTokenSecret(DefaultSAPrefix+"-member1"),
			newTokenSecret(DefaultSAPrefix+"-"+DefaultImpersonatorPrefix),
		)
		if _, _, err := GenerateClusterSecret(cs, "member1"); err != nil {
			t.Fatalf("first run error = %v", err)
		}
		if _, _, err := GenerateClusterSecret(cs, "member1"); err != nil {
			t.Errorf("second run error = %v; want idempotent success", err)
		}
	})
}

func TestUpdateClusterRoleExtra(t *testing.T) {
	t.Run("member list replaced", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newProxyClusterRole("alice", []string{"old1", "old2"}))
		idClusterList := map[string][]string{"alice": {"member1", "member2"}}

		if err := UpdateClusterRole(cs, &rbacv1.ClusterRoleList{}, idClusterList); err != nil {
			t.Fatalf("UpdateClusterRole() with empty role list error = %v", err)
		}

		// real update path: role from list with proxy rule replaced by member list
		crObj := newProxyClusterRole("alice", []string{"old1"})
		crList := &rbacv1.ClusterRoleList{Items: []rbacv1.ClusterRole{*crObj}}
		if err := UpdateClusterRole(cs, crList, idClusterList); err != nil {
			t.Fatalf("UpdateClusterRole() error = %v", err)
		}

		updated, err := cs.RbacV1().ClusterRoles().Get(context.TODO(), crObj.Name, metav1.GetOptions{})
		if err != nil {
			t.Fatalf("get role: %v", err)
		}
		if len(updated.Rules) == 0 || !hasString(updated.Rules[0].ResourceNames, "member1") {
			t.Errorf("member list not updated: %+v", updated.Rules)
		}
	})
}
