/*
* Copyright (c) 2024 China Unicom Digital Technology Co., Ltd.
* openFuyao is licensed under Mulan PSL v2.
* You can use this software according to the terms and conditions of the Mulan PSL v2.
* You may obtain a copy of Mulan PSL v2 at:
*          http://license.coscl.org.cn/MulanPSL2
* THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
* EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
* MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
* See the Mulan PSL v2 for more details.
* Author: YuXiang Guo
* Date: 2025-09-05
 */

package apps

import (
	"reflect"
	"testing"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"

	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
)

// TestSchemeRegistered 测试 Scheme 是否被正确初始化并且注册了所需的类型
func TestSchemeRegistered(t *testing.T) {
	allKnownTypes := GetScheme().AllKnownTypes()

	expectedGVKs := []schema.GroupVersionKind{
		{
			Group:   "",
			Version: "v1",
			Kind:    "Pod",
		},
		{
			Group:   colocationv1.GroupVersion.Group,
			Version: colocationv1.GroupVersion.Version,
			Kind:    "ContainerCheckpoint",
		},
		{
			Group:   v1.SchemeGroupVersion.Group,
			Version: v1.SchemeGroupVersion.Version,
			Kind:    "UpdateOptions",
		},
		{
			Group:   v1.SchemeGroupVersion.Group,
			Version: v1.SchemeGroupVersion.Version,
			Kind:    "DeleteOptions",
		},
		{
			Group:   v1.SchemeGroupVersion.Group,
			Version: v1.SchemeGroupVersion.Version,
			Kind:    "CreateOptions",
		},
	}

	for _, expectedGVK := range expectedGVKs {
		if _, exists := allKnownTypes[expectedGVK]; !exists {
			t.Errorf("Expected GVK %s to be registered in the scheme, but it was not found", expectedGVK.String())
		}
	}

	testObject, err := GetScheme().New(schema.GroupVersionKind{
		Group:   colocationv1.GroupVersion.Group,
		Version: colocationv1.GroupVersion.Version,
		Kind:    "ContainerCheckpoint",
	})
	if err != nil {
		t.Errorf("Failed to create new object of custom kind: %v", err)
	} else {
		expectedType := reflect.TypeOf(&colocationv1.ContainerCheckpoint{}).Elem()
		actualType := reflect.TypeOf(testObject).Elem()
		if actualType != expectedType {
			t.Errorf("Expected type %v, but got %v", expectedType, actualType)
		}
	}
}

// TestInitFunction 验证 init 函数执行后的 Scheme 状态
func TestInitFunction(t *testing.T) {
	if GetScheme() == nil {
		t.Fatal("Scheme should not be nil after init()")
	}

	ContainerCheckpoint := schema.GroupVersionKind{
		Group:   colocationv1.GroupVersion.Group,
		Version: colocationv1.GroupVersion.Version,
		Kind:    "ContainerCheckpoint", // 请替换为实际的种类名
	}
	_, err := GetScheme().New(ContainerCheckpoint)
	if err != nil {
		t.Errorf("Scheme should know about YourCRDKind type, failed to create new instance: %v", err)
	}

	updateOptionsGVK := schema.GroupVersionKind{Group: "", Version: "v1", Kind: "UpdateOptions"}
	_, err = GetScheme().New(updateOptionsGVK)
	if err != nil {
		t.Errorf("Scheme should know about UpdateOptions type, failed to create new instance: %v", err)
	}
}
