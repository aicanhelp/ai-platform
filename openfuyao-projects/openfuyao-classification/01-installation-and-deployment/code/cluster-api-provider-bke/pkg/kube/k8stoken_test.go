/*
 *
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 *
 */

package kube

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/kubernetes"
)

func TestB2s(t *testing.T) {
	assert.Equal(t, "", b2s([]byte{}))
	assert.Equal(t, "test", b2s([]byte("test")))
	assert.Equal(t, "token-123", b2s([]byte("token-123")))
	assert.Equal(t, "", b2s(nil))
}

func TestNewTokenGenerator(t *testing.T) {
	ctx := context.Background()
	var clientset *kubernetes.Clientset
	token := NewTokenGenerator(ctx, clientset)
	assert.NotNil(t, token)
	assert.Equal(t, "management-admin", token.ServiceAccountName)
	assert.Equal(t, "cluster-admin", token.ClusterRoleBindingRole)
	assert.Equal(t, "kube-system", token.Namespace)
	assert.Equal(t, ctx, token.Ctx)
	assert.Nil(t, token.C)
}

func TestNewTokenGenerator_WithClientset(t *testing.T) {
	ctx := context.Background()
	clientset := &kubernetes.Clientset{}
	token := NewTokenGenerator(ctx, clientset)
	assert.NotNil(t, token)
	assert.Equal(t, clientset, token.C)
	assert.Equal(t, "management-admin", token.ServiceAccountName)
	assert.Equal(t, "cluster-admin", token.ClusterRoleBindingRole)
	assert.Equal(t, "kube-system", token.Namespace)
}

func TestToken_StructFields(t *testing.T) {
	ctx := context.Background()
	clientset := &kubernetes.Clientset{}
	token := &Token{
		C:                      clientset,
		ServiceAccountName:     "test-sa",
		ClusterRoleBindingRole: "admin",
		Namespace:              "default",
		Ctx:                    ctx,
	}
	assert.Equal(t, clientset, token.C)
	assert.Equal(t, "test-sa", token.ServiceAccountName)
	assert.Equal(t, "admin", token.ClusterRoleBindingRole)
	assert.Equal(t, "default", token.Namespace)
	assert.Equal(t, ctx, token.Ctx)
}

func TestToken_LookupOrCreateToken_ExistingToken(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).GetToken, func(*Token) (string, error) {
		return "existing-token", nil
	})
	defer patches.Reset()
	result, err := token.LookupOrCreateToken()
	assert.NoError(t, err)
	assert.Equal(t, "existing-token", result)
}

func TestToken_LookupOrCreateToken_CreateNew(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).GetToken, func(*Token) (string, error) {
		return "", nil
	})
	patches.ApplyFunc((*Token).NewToken, func(*Token) (string, error) {
		return "new-token", nil
	})
	defer patches.Reset()
	result, err := token.LookupOrCreateToken()
	assert.NoError(t, err)
	assert.Equal(t, "new-token", result)
}

func TestToken_LookupOrCreateToken_GetTokenError(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).GetToken, func(*Token) (string, error) {
		return "", errors.New("get error")
	})
	defer patches.Reset()
	result, err := token.LookupOrCreateToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_GetToken_Success(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	secret := &corev1.Secret{Data: map[string][]byte{"token": []byte("test-token")}}
	patches := gomonkey.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return secret, nil
	})
	defer patches.Reset()
	result, err := token.GetToken()
	assert.NoError(t, err)
	assert.Equal(t, "test-token", result)
}

func TestToken_GetToken_NotFound(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return nil, errors.New("not found")
	})
	defer patches.Reset()
	result, err := token.GetToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_NewToken_Success(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createServiceSecret, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createClusterRoleBinding, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return &corev1.Secret{Data: map[string][]byte{"token": []byte("created-token")}}, nil
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.NoError(t, err)
	assert.Equal(t, "created-token", result)
}

func TestToken_NewToken_ServiceAccountError(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error {
		return errors.New("sa error")
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_NewToken_SecretError(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createServiceSecret, func(*Token) error {
		return errors.New("secret error")
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_NewToken_ClusterRoleBindingError(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createServiceSecret, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createClusterRoleBinding, func(*Token) error {
		return errors.New("crb error")
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_NewToken_GetSecretError(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createServiceSecret, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createClusterRoleBinding, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return nil, errors.New("get secret error")
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestClient_NewK8sToken_Success(t *testing.T) {
	client := &Client{Ctx: context.Background(), ClientSet: &kubernetes.Clientset{}}
	patches := gomonkey.ApplyFunc(NewTokenGenerator, func(ctx context.Context, c *kubernetes.Clientset) *Token {
		return &Token{C: c, Ctx: ctx}
	})
	patches.ApplyFunc((*Token).LookupOrCreateToken, func(*Token) (string, error) {
		return "client-token", nil
	})
	defer patches.Reset()
	result, err := client.NewK8sToken()
	assert.NoError(t, err)
	assert.Equal(t, "client-token", result)
}

func TestClient_NewK8sToken_Error(t *testing.T) {
	client := &Client{Ctx: context.Background(), ClientSet: &kubernetes.Clientset{}}
	patches := gomonkey.ApplyFunc(NewTokenGenerator, func(ctx context.Context, c *kubernetes.Clientset) *Token {
		return &Token{C: c, Ctx: ctx}
	})
	patches.ApplyFunc((*Token).LookupOrCreateToken, func(*Token) (string, error) {
		return "", errors.New("lookup error")
	})
	defer patches.Reset()
	result, err := client.NewK8sToken()
	assert.Error(t, err)
	assert.Equal(t, "", result)
}

func TestToken_GetToken_EmptyToken(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	secret := &corev1.Secret{Data: map[string][]byte{"token": []byte("")}}
	patches := gomonkey.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return secret, nil
	})
	defer patches.Reset()
	result, err := token.GetToken()
	assert.NoError(t, err)
	assert.Equal(t, "", result)
}

func TestToken_NewToken_EmptyTokenData(t *testing.T) {
	token := &Token{C: &kubernetes.Clientset{}, Ctx: context.Background()}
	patches := gomonkey.ApplyFunc((*Token).createServiceAccount, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createServiceSecret, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).createClusterRoleBinding, func(*Token) error { return nil })
	patches.ApplyFunc((*Token).getTokenSecret, func(*Token) (*corev1.Secret, error) {
		return &corev1.Secret{Data: map[string][]byte{"token": []byte("")}}, nil
	})
	defer patches.Reset()
	result, err := token.NewToken()
	assert.NoError(t, err)
	assert.Equal(t, "", result)
}
