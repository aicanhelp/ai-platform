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

package auth

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/mock"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

// MockDynamicInterface 模拟dynamic.Interface
type MockDynamicInterface struct {
	mock.Mock
}

func (m *MockDynamicInterface) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	args := m.Called(resource)
	return args.Get(0).(dynamic.NamespaceableResourceInterface)
}

// MockResourceInterface 模拟NamespaceableResourceInterface
type MockResourceInterface struct {
	mock.Mock
}

func (m *MockResourceInterface) Namespace(string) dynamic.ResourceInterface {
	return nil
}

func (m *MockResourceInterface) Get(ctx context.Context, name string, options metav1.GetOptions, subresources ...string) (*unstructured.Unstructured, error) {
	args := m.Called(ctx, name, options, subresources)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*unstructured.Unstructured), args.Error(1)
}

func (m *MockResourceInterface) Apply(context context.Context, namespace string, unstructured *unstructured.Unstructured, applyOptions metav1.ApplyOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *MockResourceInterface) Create(ctx context.Context, obj *unstructured.Unstructured, options metav1.CreateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	args := m.Called(ctx, obj, options, subresources)
	if args.Get(0) == nil {
		return nil, args.Error(1)
	}
	return args.Get(0).(*unstructured.Unstructured), args.Error(1)
}

func (m *MockResourceInterface) Update(ctx context.Context, obj *unstructured.Unstructured, options metav1.UpdateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *MockResourceInterface) UpdateStatus(ctx context.Context, obj *unstructured.Unstructured, options metav1.UpdateOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *MockResourceInterface) Delete(ctx context.Context, name string, options metav1.DeleteOptions, subresources ...string) error {
	return nil
}

func (m *MockResourceInterface) DeleteCollection(ctx context.Context, options metav1.DeleteOptions, listOptions metav1.ListOptions) error {
	return nil
}

func (m *MockResourceInterface) List(ctx context.Context, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	return nil, nil
}

func (m *MockResourceInterface) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	return nil, nil
}

func (m *MockResourceInterface) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, options metav1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *MockResourceInterface) ApplyStatus(context context.Context, namespace string, unstructured *unstructured.Unstructured, applyOptions metav1.ApplyOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}

func TestGetUserInfo_Success(t *testing.T) {
	// 准备测试数据
	mockDynamic := &MockDynamicInterface{}
	mockResource := &MockResourceInterface{}

	// 创建预期的用户数据
	userObject := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "users.openfuyao.com/v1alpha1",
			"kind":       "User",
			"metadata": map[string]interface{}{
				"name": "testuser",
			},
			"spec": map[string]interface{}{
				"username":          "testuser",
				"encryptedPassword": "encrypted_password",
				"description":       "测试用户",
				"platformRole":      "user",
				"firstLogin":        true,
			},
		},
	}

	// 设置mock期望
	mockDynamic.On("Resource", userMgmtGVR).Return(mockResource)
	mockResource.On("Get", mock.Anything, "testuser", mock.Anything, mock.Anything).Return(userObject, nil)

	// 执行测试
	user, err := GetUserInfo(mockDynamic, "testuser")

	// 验证结果
	assert.NoError(t, err)
	assert.NotNil(t, user)
	assert.Equal(t, "testuser", user.Name)

	// 验证mock调用
	mockDynamic.AssertExpectations(t)
	mockResource.AssertExpectations(t)
}

func TestGetUserInfo_NotFound(t *testing.T) {
	mockDynamic := &MockDynamicInterface{}
	mockResource := &MockResourceInterface{}

	expectedError := errors.New("user not found")

	mockDynamic.On("Resource", userMgmtGVR).Return(mockResource)
	mockResource.On("Get", mock.Anything, "nonexistent", mock.Anything, mock.Anything).Return(nil, expectedError)

	user, err := GetUserInfo(mockDynamic, "nonexistent")

	assert.Error(t, err)
	assert.Nil(t, user)
	assert.Equal(t, expectedError, err)

	mockDynamic.AssertExpectations(t)
	mockResource.AssertExpectations(t)
}

func TestGetUserInfo_ConversionError(t *testing.T) {
	mockDynamic := &MockDynamicInterface{}
	mockResource := &MockResourceInterface{}

	// 创建无效的unstructured对象（缺少必要字段）
	invalidObject := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"invalid": "data",
		},
	}

	mockDynamic.On("Resource", userMgmtGVR).Return(mockResource)
	mockResource.On("Get", mock.Anything, "testuser", mock.Anything, mock.Anything).Return(invalidObject, nil)

	_, err := GetUserInfo(mockDynamic, "testuser")

	assert.NoError(t, err)

	mockDynamic.AssertExpectations(t)
	mockResource.AssertExpectations(t)
}

func TestCreateUserInfo_Success(t *testing.T) {
	mockDynamic := &MockDynamicInterface{}
	mockResource := &MockResourceInterface{}

	// 创建测试用户
	testUser := createTestUser()

	// 预期返回的对象
	createdObject := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "users.openfuyao.com/v1alpha1",
			"kind":       "User",
			"metadata": map[string]interface{}{
				"name": "testuser",
			},
		},
	}

	mockDynamic.On("Resource", userMgmtGVR).Return(mockResource)
	mockResource.On("Create", mock.Anything, mock.AnythingOfType("*unstructured.Unstructured"), mock.Anything, mock.Anything).Return(createdObject, nil)

	err := CreateUserInfo(mockDynamic, testUser)

	assert.NoError(t, err)

	mockDynamic.AssertExpectations(t)
	mockResource.AssertExpectations(t)
}

func TestCreateUserInfo_CreateError(t *testing.T) {
	mockDynamic := &MockDynamicInterface{}
	mockResource := &MockResourceInterface{}

	testUser := createTestUser(
		withDescription(""),
		withPlatformRole(""),
		withFirstLogin(false),
	)

	expectedError := errors.New("creation failed")

	mockDynamic.On("Resource", userMgmtGVR).Return(mockResource)
	mockResource.On("Create", mock.Anything, mock.AnythingOfType("*unstructured.Unstructured"), mock.Anything, mock.Anything).Return(nil, expectedError)

	err := CreateUserInfo(mockDynamic, testUser)

	assert.Error(t, err)
	assert.Equal(t, expectedError, err)

	mockDynamic.AssertExpectations(t)
	mockResource.AssertExpectations(t)
}

func TestStructToUnstructured_WithUser(t *testing.T) {
	testUser := createTestUser(withPassword([]byte("password")))

	result, err := StructToUnstructured(testUser)

	assert.NoError(t, err)
	assert.NotNil(t, result)
	assert.Equal(t, "User", result.Object["kind"])
	assert.Equal(t, "users.openfuyao.com/v1alpha1", result.Object["apiVersion"])

	metadata, ok := result.Object["metadata"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "testuser", metadata["name"])

	_, ok = result.Object["spec"].(map[string]interface{})
	assert.True(t, ok)
}

func TestStructToUnstructured_EmptyStruct(t *testing.T) {
	testStruct := struct{}{}

	_, err := StructToUnstructured(testStruct)

	assert.Error(t, err)
}

// 测试UserMgmtGVR常量
func TestUserMgmtGVR(t *testing.T) {
	assert.Equal(t, "users.openfuyao.com", userMgmtGVR.Group)
	assert.Equal(t, "v1alpha1", userMgmtGVR.Version)
	assert.Equal(t, "users", userMgmtGVR.Resource)
}

// 集成测试 - 测试完整的用户管理流程
func TestUserManagementIntegration(t *testing.T) {
	// 测试结构转换
	testUser := createTestUser(
		withName("integrationuser"),
		withDescription("集成测试用户"),
		withPlatformRole("admin"),
		withFirstLogin(false),
	)

	// 测试转换为unstructured
	unstructuredUser, err := StructToUnstructured(testUser)
	assert.NoError(t, err)
	assert.NotNil(t, unstructuredUser)

	// 验证转换结果
	assert.Equal(t, "User", unstructuredUser.Object["kind"])
	assert.Equal(t, "users.openfuyao.com/v1alpha1", unstructuredUser.Object["apiVersion"])

	metadata, ok := unstructuredUser.Object["metadata"].(map[string]interface{})
	assert.True(t, ok)
	assert.Equal(t, "integrationuser", metadata["name"])
}

// createTestUser 创建测试用户对象，可通过options自定义字段
func createTestUser(options ...func(*User)) *User {
	user := &User{
		TypeMeta: metav1.TypeMeta{
			Kind:       "User",
			APIVersion: "users.openfuyao.com/v1alpha1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: "testuser",
		},
		Spec: UserSpec{
			Username:          "testuser",
			EncryptedPassword: []byte("encrypted_password"),
			Description:       "测试用户",
			PlatformRole:      "user",
			FirstLogin:        true,
		},
	}

	// 应用自定义选项
	for _, option := range options {
		option(user)
	}

	return user
}

// 定义常用的配置选项
func withName(name string) func(*User) {
	return func(u *User) {
		u.Name = name
		u.Spec.Username = name
	}
}

func withDescription(desc string) func(*User) {
	return func(u *User) {
		u.Spec.Description = desc
	}
}

func withPlatformRole(role string) func(*User) {
	return func(u *User) {
		u.Spec.PlatformRole = role
	}
}

func withFirstLogin(firstLogin bool) func(*User) {
	return func(u *User) {
		u.Spec.FirstLogin = firstLogin
	}
}

func withPassword(password []byte) func(*User) {
	return func(u *User) {
		u.Spec.EncryptedPassword = password
	}
}
