/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * openFuyao is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 */
package marketplace

import (
	"context"
	"fmt"
	"sync"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/dynamic/fake"
)

// 可配置的 Mock dynamic client，用于模拟错误场景
type MockDynamicClient struct {
	dynamic.Interface
	mu               sync.RWMutex
	listShouldFail   bool
	getShouldFail    bool
	createShouldFail bool
	updateShouldFail bool
	deleteShouldFail bool
}

// 创建一个新的 Mock dynamic client
func NewMockDynamicClient(scheme *runtime.Scheme, listKinds map[schema.GroupVersionResource]string) *MockDynamicClient {
	baseClient := fake.NewSimpleDynamicClientWithCustomListKinds(scheme, listKinds)
	return &MockDynamicClient{
		Interface: baseClient,
	}
}

func (m *MockDynamicClient) SetListFail(shouldFail bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.listShouldFail = shouldFail
}

func (m *MockDynamicClient) SetGetFail(shouldFail bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.getShouldFail = shouldFail
}

func (m *MockDynamicClient) SetCreateFail(shouldFail bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.createShouldFail = shouldFail
}

func (m *MockDynamicClient) SetUpdateFail(shouldFail bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.updateShouldFail = shouldFail
}

func (m *MockDynamicClient) SetDeleteFail(shouldFail bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.deleteShouldFail = shouldFail
}

func (m *MockDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &MockNamespaceableResource{
		NamespaceableResourceInterface: m.Interface.Resource(resource),
		mockClient:                     m,
	}
}

type MockNamespaceableResource struct {
	dynamic.NamespaceableResourceInterface
	mockClient *MockDynamicClient
}

func (m *MockNamespaceableResource) Namespace(namespace string) dynamic.ResourceInterface {
	return &MockResource{
		ResourceInterface: m.NamespaceableResourceInterface.Namespace(namespace),
		mockClient:        m.mockClient,
	}
}

func (m *MockNamespaceableResource) List(ctx context.Context, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.listShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return nil, errors.NewInternalError(fmt.Errorf("simulated list error for cluster-scoped resource"))
	}
	return m.NamespaceableResourceInterface.List(ctx, opts)
}

type MockResource struct {
	dynamic.ResourceInterface
	mockClient *MockDynamicClient
}

func (m *MockResource) List(ctx context.Context, opts metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.listShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return nil, errors.NewInternalError(fmt.Errorf("simulated list error"))
	}
	return m.ResourceInterface.List(ctx, opts)
}

func (m *MockResource) Get(ctx context.Context, name string, opts metav1.GetOptions, subresources ...string) (*unstructured.Unstructured, error) {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.getShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return nil, errors.NewInternalError(fmt.Errorf("simulated get error"))
	}
	return m.ResourceInterface.Get(ctx, name, opts, subresources...)
}

func (m *MockResource) Create(ctx context.Context, obj *unstructured.Unstructured, opts metav1.CreateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.createShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return nil, errors.NewInternalError(fmt.Errorf("simulated create error"))
	}
	return m.ResourceInterface.Create(ctx, obj, opts, subresources...)
}

func (m *MockResource) Update(ctx context.Context, obj *unstructured.Unstructured, opts metav1.UpdateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.updateShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return nil, errors.NewInternalError(fmt.Errorf("simulated update error"))
	}
	return m.ResourceInterface.Update(ctx, obj, opts, subresources...)
}

func (m *MockResource) Delete(ctx context.Context, name string, opts metav1.DeleteOptions, subresources ...string) error {
	m.mockClient.mu.RLock()
	shouldFail := m.mockClient.deleteShouldFail
	m.mockClient.mu.RUnlock()

	if shouldFail {
		return errors.NewInternalError(fmt.Errorf("simulated delete error"))
	}
	return m.ResourceInterface.Delete(ctx, name, opts, subresources...)
}

func (m *MockResource) DeleteCollection(ctx context.Context, opts metav1.DeleteOptions, listOpts metav1.ListOptions) error {
	return m.ResourceInterface.DeleteCollection(ctx, opts, listOpts)
}

func (m *MockResource) Watch(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
	return m.ResourceInterface.Watch(ctx, opts)
}

func (m *MockResource) Patch(ctx context.Context, name string, pt types.PatchType, data []byte, opts metav1.PatchOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return m.ResourceInterface.Patch(ctx, name, pt, data, opts, subresources...)
}

func (m *MockResource) Apply(ctx context.Context, name string, obj *unstructured.Unstructured, opts metav1.ApplyOptions, subresources ...string) (*unstructured.Unstructured, error) {
	return m.ResourceInterface.Apply(ctx, name, obj, opts, subresources...)
}

func (m *MockResource) ApplyStatus(ctx context.Context, name string, obj *unstructured.Unstructured, opts metav1.ApplyOptions) (*unstructured.Unstructured, error) {
	return m.ResourceInterface.ApplyStatus(ctx, name, obj, opts)
}
