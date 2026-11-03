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

package fuyaouser

import (
	"context"
	"errors"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
)

// 测试 GetUserInfo 成功场景
func TestGetUserInfoSuccess(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 Resource 方法
	patches.ApplyMethod(
		reflect.TypeOf(mockClient),
		"Resource",
		func(_ *mockDynamicClient, gvr schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
			assert.Equal(t, userMgmtGVR, gvr)
			return &mockResourceInterface{}
		},
	)

	// 模拟 Get 方法
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Get",
		func(_ *mockResourceInterface, ctx context.Context, name string, opts v1.GetOptions,
			subResources ...string) (*unstructured.Unstructured, error) {
			assert.Equal(t, "test-user", name)
			user := &User{
				ObjectMeta: v1.ObjectMeta{Name: "test-user"},
				Spec:       UserSpec{Username: "testuser"},
			}
			unstr, _ := runtime.DefaultUnstructuredConverter.ToUnstructured(user)
			return &unstructured.Unstructured{Object: unstr}, nil
		},
	)

	// 调用函数
	user, err := GetUserInfo(mockClient, "test-user")

	// 验证结果
	assert.NoError(t, err)
	assert.Equal(t, "test-user", user.Name)
	assert.Equal(t, "testuser", user.Spec.Username)
}

// 测试 GetUserInfo 获取用户失败
func TestGetUserInfoGetFailure(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 Get 方法返回错误
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Get",
		func(_ *mockResourceInterface, ctx context.Context, name string, opts v1.GetOptions,
			subResources ...string) (*unstructured.Unstructured, error) {
			return nil, errors.New("user not found")
		},
	)

	// 调用函数
	user, err := GetUserInfo(mockClient, "test-user")

	// 验证结果
	assert.Error(t, err)
	assert.Nil(t, user)
	assert.Contains(t, err.Error(), "user not found")
}

// 测试 PatchUserInfo 成功
func TestPatchUserInfoSuccess(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 Patch 方法
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Patch",
		func(_ *mockResourceInterface, ctx context.Context, name string, pt types.PatchType, data []byte,
			opts v1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
			assert.Equal(t, "test-user", name)
			assert.Equal(t, types.MergePatchType, pt)
			assert.Equal(t, []byte(`{"patch":"data"}`), data)
			return nil, nil
		},
	)

	// 调用函数
	err := PatchUserInfo(mockClient, "test-user", []byte(`{"patch":"data"}`))

	// 验证结果
	assert.NoError(t, err)
}

// 测试 PatchUserInfo 失败
func TestPatchUserInfoFailure(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟 Patch 方法返回错误
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Patch",
		func(_ *mockResourceInterface, ctx context.Context, name string, pt types.PatchType, data []byte,
			opts v1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
			return nil, errors.New("patch failed")
		},
	)

	// 调用函数
	err := PatchUserInfo(mockClient, "test-user", []byte(`{}`))

	// 验证结果
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "patch failed")
}

// 测试 UpdateUserInfo 成功
func TestUpdateUserInfoSuccess(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 准备测试数据
	user := &User{
		ObjectMeta: v1.ObjectMeta{Name: "test-user"},
	}

	// 模拟转换函数
	patches.ApplyFunc(
		StructToUnstructured,
		func(v interface{}) (*unstructured.Unstructured, error) {
			assert.Equal(t, user, v)
			unstr, _ := runtime.DefaultUnstructuredConverter.ToUnstructured(v)
			return &unstructured.Unstructured{Object: unstr}, nil
		},
	)

	// 模拟更新方法
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Update",
		func(_ *mockResourceInterface, ctx context.Context, obj *unstructured.Unstructured, opts v1.UpdateOptions,
			subResources ...string) (*unstructured.Unstructured, error) {
			assert.Equal(t, "test-user", obj.GetName())
			return obj, nil
		},
	)

	// 调用函数
	err := UpdateUserInfo(mockClient, user)

	// 验证结果
	assert.NoError(t, err)
}

// 测试 UpdateUserInfo 更新失败
func TestUpdateUserInfoUpdateFailure(t *testing.T) {
	// 创建模拟动态客户端
	mockClient := &mockDynamicClient{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 模拟转换成功
	patches.ApplyFunc(
		StructToUnstructured,
		func(v interface{}) (*unstructured.Unstructured, error) {
			return &unstructured.Unstructured{}, nil
		},
	)

	// 模拟更新失败
	patches.ApplyMethod(
		reflect.TypeOf(&mockResourceInterface{}),
		"Update",
		func(_ *mockResourceInterface, ctx context.Context, obj *unstructured.Unstructured, opts v1.UpdateOptions,
			subResources ...string) (*unstructured.Unstructured, error) {
			return nil, errors.New("update failed")
		},
	)

	// 调用函数
	err := UpdateUserInfo(mockClient, &User{})

	// 验证结果
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "update failed")
}

// 测试 StructToUnstructured 成功
func TestStructToUnstructuredSuccess(t *testing.T) {
	// 准备测试数据
	testUser := &User{
		ObjectMeta: v1.ObjectMeta{Name: "test-user"},
		Spec:       UserSpec{Username: "testuser"},
	}

	// 调用函数
	unstr, err := StructToUnstructured(testUser)

	// 验证结果
	assert.NoError(t, err)
	assert.Equal(t, "test-user", unstr.GetName())

	username, found, err := unstructured.NestedString(
		unstr.Object, "spec", "Username")
	assert.NoError(t, err)
	assert.True(t, found)
	assert.Equal(t, "testuser", username)
}

// 测试 StructToUnstructured 失败
func TestStructToUnstructuredFailure(t *testing.T) {
	// 准备无效数据（函数不能转换）
	invalidData := make(chan int)

	// 调用函数
	unstr, err := StructToUnstructured(invalidData)

	// 验证结果
	assert.Error(t, err)
	assert.Nil(t, unstr)
	assert.Contains(t, err.Error(), "got chan int")
}

// mockDynamicClient 模拟动态客户端
type mockDynamicClient struct{}

func (m *mockDynamicClient) Resource(gvr schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &mockResourceInterface{}
}

// mockResourceInterface 模拟资源接口
type mockResourceInterface struct{}

func (m *mockResourceInterface) Namespace(string) dynamic.ResourceInterface {
	return m
}

func (m *mockResourceInterface) Create(ctx context.Context, obj *unstructured.Unstructured, options v1.CreateOptions,
	subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) Update(ctx context.Context, obj *unstructured.Unstructured, options v1.UpdateOptions,
	subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) UpdateStatus(ctx context.Context, obj *unstructured.Unstructured,
	options v1.UpdateOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) Delete(ctx context.Context, name string, options v1.DeleteOptions,
	subresources ...string) error {
	return nil
}

func (m *mockResourceInterface) DeleteCollection(ctx context.Context, options v1.DeleteOptions,
	listOptions v1.ListOptions) error {
	return nil
}

func (m *mockResourceInterface) Get(ctx context.Context, name string, options v1.GetOptions,
	subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) List(ctx context.Context, opts v1.ListOptions) (*unstructured.UnstructuredList, error) {
	return nil, nil
}

func (m *mockResourceInterface) Watch(ctx context.Context, opts v1.ListOptions) (watch.Interface, error) {
	return nil, nil
}

func (m *mockResourceInterface) Patch(ctx context.Context, name string, pt types.PatchType, data []byte,
	options v1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) Apply(ctx context.Context, name string, obj *unstructured.Unstructured,
	options v1.ApplyOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}

func (m *mockResourceInterface) ApplyStatus(ctx context.Context, name string, obj *unstructured.Unstructured,
	options v1.ApplyOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}
