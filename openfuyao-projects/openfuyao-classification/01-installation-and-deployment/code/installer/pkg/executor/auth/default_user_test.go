/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *           http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */

package auth

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	rbacv1 "k8s.io/api/rbac/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes/fake"

	"installer/pkg/fuyao/v1beta1/base"
	"installer/pkg/fuyao/v1beta1/cluster"
)

// MockDynamicClient 用于模拟dynamic client
type MockDynamicClient struct {
	mock.Mock
}

func (m *MockDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	args := m.Called(resource)
	return args.Get(0).(dynamic.NamespaceableResourceInterface)
}

func TestGeneratePassword(t *testing.T) {
	tests := []struct {
		name   string
		length int
	}{
		{"Length 8", 8},
		{"Length 12", 12},
		{"Length 16", 16},
		{"Length 20", 20},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			password := generatePassword(tt.length)
			assert.Equal(t, tt.length, len(password))

			// 验证密码包含字母、数字和符号
			hasAlpha := false
			hasDigit := false
			hasSymbol := false

			for _, char := range password {
				if (char >= 'a' && char <= 'z') || (char >= 'A' && char <= 'Z') {
					hasAlpha = true
				} else if char >= '0' && char <= '9' {
					hasDigit = true
				} else {
					hasSymbol = true
				}
			}

			assert.True(t, hasAlpha, "密码应包含字母")
			assert.True(t, hasDigit, "密码应包含数字")
			assert.True(t, hasSymbol, "密码应包含符号")
		})
	}
}

func TestGenerateRandomIndex(t *testing.T) {
	tests := []struct {
		name   string
		length int
	}{
		{"Length 10", 10},
		{"Length 100", 100},
		{"Length 1000", 1000},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			index := generateRandomIndex(tt.length)
			assert.GreaterOrEqual(t, index, 0)
			assert.Less(t, index, tt.length)
		})
	}
}

func TestEncryptPassword(t *testing.T) {
	cfg := UserInfoConfig{
		PasswdLength:  12,
		SaltLength:    16,
		Iterations:    100000,
		KeyLength:     64,
		EncryptMethod: sha256.New,
	}

	tests := []struct {
		name        string
		rawPassword []byte
	}{
		{"Simple password", []byte("password123")},
		{"Complex password", []byte("P@ssw0rd!@#")},
		{"Empty password", []byte("")},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encrypted, err := encryptPassword(tt.rawPassword, cfg)
			assert.NoError(t, err)
			assert.NotEmpty(t, encrypted)

			// 验证base64编码
			_, err = base64.StdEncoding.DecodeString(encrypted)
			assert.NoError(t, err)

			// 验证每次加密结果不同（因为盐值随机）
			encrypted2, err := encryptPassword(tt.rawPassword, cfg)
			assert.NoError(t, err)
			assert.NotEqual(t, encrypted, encrypted2)
		})
	}
}

func TestPrepareUserInstance(t *testing.T) {
	username := "testuser"
	encryptedPassword := "encrypted_password_base64"

	user := prepareUserInstance(username, encryptedPassword)

	assert.Equal(t, "User", user.Kind)
	assert.Equal(t, "users.openfuyao.com/v1alpha1", user.APIVersion)
	assert.Equal(t, username, user.Name)
	assert.Equal(t, username, user.Spec.Username)
	assert.Equal(t, []byte(encryptedPassword), user.Spec.EncryptedPassword)
	assert.Equal(t, "内置默认平台管理员", user.Spec.Description)
	assert.Equal(t, "platform-admin", user.Spec.PlatformRole)
	assert.True(t, user.Spec.FirstLogin)
}

func TestCreateAdminClusterRoleBinding(t *testing.T) {
	// 创建fake kubernetes client
	k8sClient := fake.NewSimpleClientset()

	// 测试创建新的ClusterRoleBinding
	err := createAdminClusterRoleBinding(k8sClient)
	assert.NoError(t, err)

	// 验证ClusterRoleBinding是否被创建
	crb, err := k8sClient.RbacV1().ClusterRoleBindings().Get(context.Background(),
		"admin-platform-admin", metav1.GetOptions{})
	assert.NoError(t, err)
	assert.Equal(t, "admin-platform-admin", crb.Name)
	assert.Equal(t, "openfuyao-platform-admin", crb.RoleRef.Name)
	assert.Equal(t, "admin", crb.Subjects[0].Name)
}

func TestCreateAdminClusterRoleBindingUpdate(t *testing.T) {
	k8sClient := fake.NewSimpleClientset()

	// 预先创建一个ClusterRoleBinding
	existingCRB := &rbacv1.ClusterRoleBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "admin-cluster-admin",
			ResourceVersion: "1",
		},
		RoleRef: rbacv1.RoleRef{
			APIGroup: "rbac.authorization.k8s.io",
			Kind:     "ClusterRole",
			Name:     "old-role",
		},
		Subjects: []rbacv1.Subject{
			{
				APIGroup: "rbac.authorization.k8s.io",
				Kind:     "User",
				Name:     "olduser",
			},
		},
	}

	_, err := k8sClient.RbacV1().ClusterRoleBindings().Create(context.Background(),
		existingCRB, metav1.CreateOptions{})
	assert.NoError(t, err)

	// 测试更新
	createAdminClusterRoleBinding(k8sClient)
}

const (
	TestVIPAds = "192" + ".168." + "1.100"
)

// 测试用参数名称常量
const (
	VIPParamName = "vip"
	HAGroupName  = "ha"
)

func TestGetVIP(t *testing.T) {
	tests := []struct {
		name       string
		components []cluster.Component
		expected   string
	}{
		{
			name: "找到VIP",
			components: []cluster.Component{
				{
					Group: HAGroupName,
					Param: []base.Param{
						{Name: VIPParamName, Value: TestVIPAds},
					},
				},
			},
			expected: TestVIPAds,
		},
		{
			name: "未找到VIP",
			components: []cluster.Component{
				{
					Group: "other",
					Param: []base.Param{
						{Name: "other_param", Value: "value"},
					},
				},
			},
			expected: "",
		},
		{
			name:       "空组件列表",
			components: []cluster.Component{},
			expected:   "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getVIP(tt.components)
			assert.Equal(t, tt.expected, result)
		})
	}
}

func newUserInfoCfg() UserInfoConfig {
	passWordLen, saltLen, iterations, keyLen := 12, 16, 100000, 64
	return UserInfoConfig{
		PasswdLength:  passWordLen,
		SaltLength:    saltLen,
		Iterations:    iterations,
		KeyLength:     keyLen,
		EncryptMethod: sha256.New,
	}
}

func TestUserInfoConfig(t *testing.T) {
	passWordLen, saltLen, iterations, keyLen := 12, 16, 100000, 64
	cfg := newUserInfoCfg()

	assert.Equal(t, passWordLen, cfg.PasswdLength)
	assert.Equal(t, saltLen, cfg.SaltLength)
	assert.Equal(t, iterations, cfg.Iterations)
	assert.Equal(t, keyLen, cfg.KeyLength)
	assert.NotNil(t, cfg.EncryptMethod)
}

// 测试密码生成的边界情况
func TestGeneratePasswordEdgeCases(t *testing.T) {
	const passwdLen = 3
	// 测试最小长度
	password := generatePassword(passwdLen)
	assert.Equal(t, passwdLen, len(password))
}

// 测试加密配置的边界情况
func TestEncryptPasswordEdgeCases(t *testing.T) {
	// 最小配置
	cfg := UserInfoConfig{
		SaltLength:    1,
		Iterations:    1,
		KeyLength:     1,
		EncryptMethod: sha256.New,
	}

	encrypted, err := encryptPassword([]byte("test"), cfg)
	assert.NoError(t, err)
	assert.NotEmpty(t, encrypted)
}

// 集成测试 - 测试完整的用户创建流程
func TestGenerateDefaultUserInfoIntegration(t *testing.T) {
	fake.NewSimpleClientset()

	// 这里需要mock dynamic client，实际测试中可能需要更复杂的mock设置
	// 由于dynamic client较难mock，这里主要测试配置和基本流程
	passWordLen, saltLen := 12, 16

	cfg := newUserInfoCfg()

	// 测试配置的有效性
	assert.Equal(t, passWordLen, cfg.PasswdLength)
	assert.Equal(t, saltLen, cfg.SaltLength)
	assert.NotNil(t, cfg.EncryptMethod)

	// 测试密码生成
	rawPasswd := generatePassword(cfg.PasswdLength)
	assert.Equal(t, cfg.PasswdLength, len(rawPasswd))

	// 测试密码加密
	encrypted, err := encryptPassword(rawPasswd, cfg)
	assert.NoError(t, err)
	assert.NotEmpty(t, encrypted)
}
