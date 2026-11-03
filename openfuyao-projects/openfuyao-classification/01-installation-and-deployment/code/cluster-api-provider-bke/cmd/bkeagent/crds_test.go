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

package main

import (
	"context"
	"errors"
	"io"
	"strings"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/restmapper"
	ctrl "sigs.k8s.io/controller-runtime"

	crdembed "gopkg.openfuyao.cn/cluster-api-provider-bke/config"
)

func TestEnableCrdHasInstalledClientSetError(t *testing.T) {
	patches := gomonkey.ApplyFuncReturn(ctrl.GetConfigOrDie, nil)
	defer patches.Reset()
	patches.ApplyFuncReturn(kubernetes.NewForConfig, nil, errors.New("clientset error"))

	err := enableCrdHasInstalled()
	assert.Error(t, err)
}

func TestInstallCrdOpenFileError(t *testing.T) {
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return nil, errors.New("open error")
	})
	defer patches.Reset()

	err := installCrd(&kubernetes.Clientset{}, nil)
	assert.Error(t, err)
}

func TestInstallCrdGetAPIGroupError(t *testing.T) {
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("")), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return nil, errors.New("api error")
	})

	err := installCrd(&kubernetes.Clientset{}, nil)
	assert.Error(t, err)
}

func TestInstallCrdDecodeEOF(t *testing.T) {
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("")), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return []*restmapper.APIGroupResources{}, nil
	})

	err := installCrd(&kubernetes.Clientset{}, nil)
	assert.NoError(t, err)
}

func TestInstallCrdDecodeError(t *testing.T) {
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader("invalid yaml content")), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return []*restmapper.APIGroupResources{}, nil
	})

	err := installCrd(&kubernetes.Clientset{}, nil)
	assert.Error(t, err)
}

func TestInstallCrdUnstructuredDecodeError(t *testing.T) {
	yamlContent := "apiVersion: v1\nkind: Pod"
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(yamlContent)), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return []*restmapper.APIGroupResources{}, nil
	})

	patches.ApplyFunc(unstructured.UnstructuredJSONScheme.Decode, func(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
		return nil, nil, errors.New("decode error")
	})

	err := installCrd(&kubernetes.Clientset{}, nil)
	assert.Error(t, err)
}

func TestInstallCrdToUnstructuredError(t *testing.T) {
	yamlContent := "apiVersion: v1\nkind: Pod"
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(yamlContent)), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return []*restmapper.APIGroupResources{}, nil
	})

	patches.ApplyFunc(runtime.DefaultUnstructuredConverter.ToUnstructured, func(obj interface{}) (map[string]interface{}, error) {
		return nil, errors.New("convert error")
	})

	err := installCrd(&kubernetes.Clientset{}, &mockDynamicClient{})
	assert.Error(t, err)
}

func TestInstallCrdCreateError(t *testing.T) {
	yamlContent := "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test"
	patches := gomonkey.ApplyMethod(crdembed.CRDs, "Open", func(_ interface{}, name string) (io.ReadCloser, error) {
		return io.NopCloser(strings.NewReader(yamlContent)), nil
	})
	defer patches.Reset()

	patches.ApplyFunc(restmapper.GetAPIGroupResources, func(d interface{}) ([]*restmapper.APIGroupResources, error) {
		return []*restmapper.APIGroupResources{}, nil
	})

	mockDynamic := &mockDynamicClient{createErr: errors.New("create error")}
	err := installCrd(&kubernetes.Clientset{}, mockDynamic)
	assert.Error(t, err)
}

type mockDynamicClient struct {
	createErr error
}

func (m *mockDynamicClient) Resource(resource schema.GroupVersionResource) dynamic.NamespaceableResourceInterface {
	return &mockResourceInterface{createErr: m.createErr}
}

type mockResourceInterface struct {
	createErr error
}

func (m *mockResourceInterface) Namespace(string) dynamic.ResourceInterface { return m }
func (m *mockResourceInterface) Create(ctx context.Context, obj *unstructured.Unstructured, opts metav1.CreateOptions, subresources ...string) (*unstructured.Unstructured, error) {
	if m.createErr != nil {
		return nil, m.createErr
	}
	return &unstructured.Unstructured{}, nil
}
func (m *mockResourceInterface) Update(context.Context, *unstructured.Unstructured, metav1.UpdateOptions, ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}
func (m *mockResourceInterface) UpdateStatus(context.Context, *unstructured.Unstructured, metav1.UpdateOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}
func (m *mockResourceInterface) Delete(context.Context, string, metav1.DeleteOptions, ...string) error {
	return nil
}
func (m *mockResourceInterface) DeleteCollection(context.Context, metav1.DeleteOptions, metav1.ListOptions) error {
	return nil
}
func (m *mockResourceInterface) Get(context.Context, string, metav1.GetOptions, ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}
func (m *mockResourceInterface) List(context.Context, metav1.ListOptions) (*unstructured.UnstructuredList, error) {
	return nil, nil
}
func (m *mockResourceInterface) Watch(context.Context, metav1.ListOptions) (watch.Interface, error) {
	return nil, nil
}
func (m *mockResourceInterface) Patch(context.Context, string, types.PatchType, []byte, metav1.PatchOptions, ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}
func (m *mockResourceInterface) Apply(context.Context, string, *unstructured.Unstructured, metav1.ApplyOptions, ...string) (*unstructured.Unstructured, error) {
	return nil, nil
}
func (m *mockResourceInterface) ApplyStatus(context.Context, string, *unstructured.Unstructured, metav1.ApplyOptions) (*unstructured.Unstructured, error) {
	return nil, nil
}
