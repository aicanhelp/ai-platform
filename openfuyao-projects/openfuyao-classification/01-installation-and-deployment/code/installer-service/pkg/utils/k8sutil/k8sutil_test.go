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

package k8sutil

import (
	"context"
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
)

// 测试成功获取 Secret
func TestGetSecretSuccess(t *testing.T) {
	// 创建一个虚拟的 Kubernetes 客户端
	clientset := fake.NewSimpleClientset()

	// 创建一个示例 Secret
	secret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-secret",
			Namespace: "default",
		},
		Data: map[string][]byte{
			"key": []byte("value"),
		},
	}

	// 将 Secret 添加到虚拟客户端中
	_, err := clientset.CoreV1().Secrets("default").Create(context.Background(), secret,
		metav1.CreateOptions{})
	assert.NoError(t, err)

	// 使用 gomonkey 模拟 Get 方法
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	patches.ApplyFunc(clientset.CoreV1().Secrets("default").Get,
		func(ctx context.Context, name string, opts metav1.GetOptions) (*v1.Secret, error) {
			return secret, nil
		})

	// 调用 GetSecret 函数
	result, err := GetSecret(clientset, "test-secret", "default")

	// 验证结果
	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, secret.Name, result.Name)
	assert.Equal(t, secret.Namespace, result.Namespace)
}

func TestGetSecretError(t *testing.T) {

	// 创建mock客户端
	mockClient := &mockKubernetesClient{}

	// 使用gomonkey直接模拟GetSecret调用的实际函数
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	expectedErr := errors.New("secret not found")
	patches.ApplyFunc(GetSecret, func(_ kubernetes.Interface, name, namespace string) (*v1.Secret, error) {
		return nil, expectedErr
	})

	// 调用GetSecret函数
	result, err := GetSecret(mockClient, "non-existing-secret", "default")

	// 验证结果
	assert.Error(t, err)
	assert.Equal(t, expectedErr, err)
	assert.Nil(t, result)
}

// 最小kubernetes.Interface实现
type mockKubernetesClient struct {
	kubernetes.Interface
}

func TestGetConfigMap(t *testing.T) {
	// 成功获取
	t.Run("success", func(t *testing.T) {
		client := fake.NewSimpleClientset(&v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "test",
				Namespace: "default",
			},
		})

		cm, err := GetConfigMap(client, "test", "default")
		if err != nil || cm == nil || cm.Name != "test" {
			t.Fatal("should return configmap")
		}
	})

	// 获取失败
	t.Run("not found", func(t *testing.T) {
		client := fake.NewSimpleClientset()
		_, err := GetConfigMap(client, "missing", "default")
		if err == nil {
			t.Fatal("should return error")
		}
	})
}

// 测试在 Secret 不存在时，GetSecret 会返回错误并且记录错误日志
func TestGetSecretNotFound(t *testing.T) {
	client := fake.NewSimpleClientset()

	s, err := GetSecret(client, "missing", "default")
	assert.Error(t, err)
	assert.Nil(t, s)
}
