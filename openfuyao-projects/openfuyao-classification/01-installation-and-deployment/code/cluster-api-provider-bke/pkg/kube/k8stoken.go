/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package kube

import (
	"context"
	"time"

	"github.com/pkg/errors"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils"
)

type Token struct {
	C                      *kubernetes.Clientset
	ServiceAccountName     string
	ClusterRoleBindingRole string
	Namespace              string
	Ctx                    context.Context
}

func NewTokenGenerator(ctx context.Context, c *kubernetes.Clientset) *Token {
	return &Token{
		C:                      c,
		ServiceAccountName:     "management-admin",
		ClusterRoleBindingRole: "cluster-admin",
		Namespace:              "kube-system",
		Ctx:                    ctx,
	}
}

func (c *Client) NewK8sToken() (string, error) {
	return NewTokenGenerator(c.Ctx, c.ClientSet).LookupOrCreateToken()
}

func (t *Token) LookupOrCreateToken() (string, error) {
	token, err := t.GetToken()
	if err != nil {
		return "", err
	}
	if token != "" {
		return token, nil
	}
	return t.NewToken()
}

func (t *Token) GetToken() (string, error) {
	secret, err := t.getTokenSecret()
	if err != nil {
		return "", client.IgnoreNotFound(err)
	}
	token := b2s(secret.Data["token"])

	return token, nil
}

func (t *Token) NewToken() (string, error) {
	if err := t.createServiceAccount(); err != nil {
		return "", err
	}
	time.Sleep(1 * time.Second)
	if err := t.createServiceSecret(); err != nil {
		return "", err
	}
	time.Sleep(1 * time.Second)

	if err := t.createClusterRoleBinding(); err != nil {
		return "", err
	}
	time.Sleep(1 * time.Second)
	secret, err := t.getTokenSecret()
	if err != nil {
		return "", err
	}
	token := b2s(secret.Data["token"])

	return token, nil
}

func b2s(b []byte) string {
	return string(b)
}

// createServiceAccount creates a service account with the given name in the given namespace.
func (t *Token) createServiceAccount() error {
	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:      t.ServiceAccountName,
			Namespace: t.Namespace,
		},
	}

	if _, err := t.C.CoreV1().ServiceAccounts(t.Namespace).Create(t.Ctx, sa, metav1.CreateOptions{}); err != nil {
		return err
	}

	return nil
}

func (t *Token) createServiceSecret() error {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      t.ServiceAccountName,
			Namespace: t.Namespace,
			Annotations: map[string]string{
				corev1.ServiceAccountNameKey: t.ServiceAccountName,
			},
		},
		Type: corev1.SecretTypeServiceAccountToken,
	}

	if _, err := t.C.CoreV1().Secrets(t.Namespace).Create(t.Ctx, secret, metav1.CreateOptions{}); err != nil {
		return err
	}

	return nil
}

func (t *Token) createClusterRoleBinding() error {
	crb := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: t.ServiceAccountName,
		},

		Subjects: []rbacv1.Subject{
			{
				Kind:      "ServiceAccount",
				Name:      t.ServiceAccountName,
				Namespace: t.Namespace,
			},
		},
		RoleRef: rbacv1.RoleRef{
			Kind: "ClusterRole",
			Name: t.ClusterRoleBindingRole,
		},
	}

	if _, err := t.C.RbacV1().ClusterRoleBindings().Create(t.Ctx, crb, metav1.CreateOptions{}); err != nil {
		return err
	}

	return nil
}

func (t *Token) getTokenSecret() (*corev1.Secret, error) {
	secret, err := t.C.CoreV1().Secrets(t.Namespace).Get(t.Ctx, t.ServiceAccountName, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	if secret.Data == nil {
		return nil, errors.Errorf("no sa secret data found in secret  %s", utils.ClientObjNS(secret))
	}
	if v, ok := secret.Data["token"]; !ok || v == nil {
		return nil, errors.Errorf("no token data found in secret  %s", utils.ClientObjNS(secret))
	}

	return secret, nil
}
