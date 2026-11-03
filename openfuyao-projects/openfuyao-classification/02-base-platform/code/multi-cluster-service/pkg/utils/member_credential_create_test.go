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
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	k8sfake "k8s.io/client-go/kubernetes/fake"
	testcore "k8s.io/client-go/testing"
)

func namedClusterRoleBinding(name string) *rbacv1.ClusterRoleBinding {
	obj := &rbacv1.ClusterRoleBinding{}
	obj.Name = name
	return obj
}

func namedClusterRole(name string) *rbacv1.ClusterRole {
	obj := &rbacv1.ClusterRole{}
	obj.Name = name
	return obj
}

func TestCreareClusterRoleBindingExtra(t *testing.T) {
	t.Run("create success", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		obj, err := creareClusterRoleBinding(cs, namedClusterRoleBinding("binding-1"))
		if err != nil || obj == nil {
			t.Errorf("creareClusterRoleBinding() = %v, %v", obj, err)
		}
	})

	t.Run("already exists", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(namedClusterRoleBinding("binding-1"))
		obj, err := creareClusterRoleBinding(cs, namedClusterRoleBinding("binding-1"))
		if err != nil || obj == nil {
			t.Errorf("creareClusterRoleBinding() = %v, %v; want existing object accepted", obj, err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "clusterrolebindings",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if _, err := creareClusterRoleBinding(cs, namedClusterRoleBinding("binding-1")); err == nil {
			t.Error("creareClusterRoleBinding() expected error")
		}
	})
}

func TestCreateClusterRoleExtra(t *testing.T) {
	t.Run("create success", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		obj, err := createClusterRole(cs, namedClusterRole("role-1"))
		if err != nil || obj == nil {
			t.Errorf("createClusterRole() = %v, %v", obj, err)
		}
	})

	t.Run("already exists", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(namedClusterRole("role-1"))
		obj, err := createClusterRole(cs, namedClusterRole("role-1"))
		if err != nil || obj == nil {
			t.Errorf("createClusterRole() = %v, %v; want existing object accepted", obj, err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "clusterroles",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if _, err := createClusterRole(cs, namedClusterRole("role-1")); err == nil {
			t.Error("createClusterRole() expected error")
		}
	})
}

func TestCreateServiceAccountExtra(t *testing.T) {
	newSA := func(name string) *corev1.ServiceAccount {
		obj := &corev1.ServiceAccount{}
		obj.Name = name
		obj.Namespace = DefaultKarmadaNamespace
		return obj
	}

	t.Run("create success", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		obj, err := createServiceAccount(cs, newSA("sa-1"))
		if err != nil || obj == nil {
			t.Errorf("createServiceAccount() = %v, %v", obj, err)
		}
	})

	t.Run("already exists", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newSA("sa-1"))
		obj, err := createServiceAccount(cs, newSA("sa-1"))
		if err != nil || obj == nil {
			t.Errorf("createServiceAccount() = %v, %v; want existing object accepted", obj, err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "serviceaccounts",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if _, err := createServiceAccount(cs, newSA("sa-1")); err == nil {
			t.Error("createServiceAccount() expected error")
		}
	})
}

func TestCreateSecretExtra(t *testing.T) {
	t.Run("create success", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		obj, err := createSecret(cs, "sa-1", DefaultKarmadaNamespace)
		if err != nil || obj == nil {
			t.Errorf("createSecret() = %v, %v", obj, err)
		}
	})

	t.Run("already exists returns existing", func(t *testing.T) {
		existing := &corev1.Secret{
			ObjectMeta: metav1.ObjectMeta{Name: "sa-1", Namespace: DefaultKarmadaNamespace},
		}
		cs := k8sfake.NewSimpleClientset(existing)
		obj, err := createSecret(cs, "sa-1", DefaultKarmadaNamespace)
		if err != nil || obj == nil || obj.Name != "sa-1" {
			t.Errorf("createSecret() = %v, %v; want existing secret", obj, err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "secrets",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if _, err := createSecret(cs, "sa-1", DefaultKarmadaNamespace); err == nil {
			t.Error("createSecret() expected error")
		}
	})
}

func TestCreateNamespaceExtra(t *testing.T) {
	newNS := func(name string) *corev1.Namespace {
		obj := &corev1.Namespace{}
		obj.Name = name
		return obj
	}

	t.Run("create success", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		if err := createNamespace(cs, newNS("ns-1")); err != nil {
			t.Errorf("createNamespace() error = %v", err)
		}
	})

	t.Run("already exists", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset(newNS("ns-1"))
		if err := createNamespace(cs, newNS("ns-1")); err != nil {
			t.Errorf("createNamespace() error = %v; want accepted", err)
		}
	})

	t.Run("create failed", func(t *testing.T) {
		cs := k8sfake.NewSimpleClientset()
		cs.PrependReactor("create", "namespaces",
			func(action testcore.Action) (bool, runtime.Object, error) {
				return true, nil, apierrors.NewInternalError(errors.New("boom"))
			})
		if err := createNamespace(cs, newNS("ns-1")); err == nil {
			t.Error("createNamespace() expected error")
		}
	})
}
