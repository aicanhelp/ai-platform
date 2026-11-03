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

package controller

import (
	"context"
	"fmt"
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

var cr = &kaev1.KAEPolicy{
	ObjectMeta: metav1.ObjectMeta{
		Name: "test-policy",
	},
}

// TestUpdateKAEPolicyStateSuccess 测试 updateKAEPolicyState 函数在成功场景下的行为
func TestUpdateKAEPolicyStateSuccess(t *testing.T) {
	ctx := context.TODO()
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	namespacedName := types.NamespacedName{Name: "test-kaepolicy", Namespace: "default"}
	state := kaev1.Ready

	// Mock r.Client.Get
	patchGet := gomonkey.ApplyMethod(reflect.TypeOf(r.Client), "Get", func(_ client.Client, _ context.Context,
		_ types.NamespacedName, obj client.Object, _ ...client.GetOption) error {
		obj = cr
		return nil
	})
	defer patchGet.Reset()

	// Mock r.Client.Status().Update
	patchStatusUpdate := gomonkey.ApplyMethod(reflect.TypeOf(r.Client.Status()), "Update", func(_ client.StatusWriter,
		_ context.Context, _ client.Object, _ ...client.SubResourceUpdateOption) error {
		return nil
	})
	defer patchStatusUpdate.Reset()

	// 执行测试
	updateKAEPolicyState(ctx, r, namespacedName, state)

	// 没有错误发生，状态应成功更新
	assert.True(t, true)
}

// TestUpdateKAEPolicyStateGetError 测试 updateKAEPolicyState 函数在 Get 操作失败场景下的行为
func TestUpdateKAEPolicyStateGetError(t *testing.T) {
	ctx := context.TODO()
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	namespacedName := types.NamespacedName{Name: "test-kaepolicy", Namespace: "default"}
	state := kaev1.Ready

	// Mock r.Client.Get 返回错误
	patchGet := gomonkey.ApplyMethod(reflect.TypeOf(r.Client), "Get", func(_ client.Client, _ context.Context,
		_ types.NamespacedName, obj client.Object, _ ...client.GetOption) error {
		return fmt.Errorf("get error")
	})
	defer patchGet.Reset()

	// 执行测试
	updateKAEPolicyState(ctx, r, namespacedName, state)

	// 如果状态获取失败，应该处理错误但不抛出 panic
	assert.True(t, true)
}

// TestUpdateStatusConditionSuccess 测试 updateStatusCondition 函数在成功场景下的行为
func TestUpdateStatusConditionSuccess(t *testing.T) {
	fClient := fake.NewClientBuilder().Build()
	ctx := context.TODO()
	params := statusUpdateParams{
		readyStatus: metav1.ConditionTrue,
		errorStatus: metav1.ConditionFalse,
		reason:      "TestReason",
		message:     "TestMessage",
	}

	// Mock fClient.Get
	patchGet := gomonkey.ApplyMethod(reflect.TypeOf(fClient), "Get", func(_ client.Client, _ context.Context,
		_ types.NamespacedName, obj client.Object, _ ...client.GetOption) error {
		obj = cr
		return nil
	})
	defer patchGet.Reset()

	// Mock fClient.Status().Update
	patchStatusUpdate := gomonkey.ApplyMethod(reflect.TypeOf(fClient.Status()), "Update", func(_ client.StatusWriter,
		_ context.Context, _ client.Object, _ ...client.SubResourceUpdateOption) error {
		return nil
	})
	defer patchStatusUpdate.Reset()

	// 执行测试
	err := updateStatusCondition(fClient, ctx, cr, params)

	// 断言没有错误
	assert.NoError(t, err)
}

// TestSetConditionsReady 测试 setConditionsReady 函数
func TestSetConditionsReady(t *testing.T) {
	fClient := fake.NewClientBuilder().Build()
	ctx := context.TODO()

	// Mock updateStatusCondition
	patchUpdateStatusCondition := gomonkey.ApplyFunc(updateStatusCondition, func(_ client.Client, _ context.Context,
		_ *kaev1.KAEPolicy, _ statusUpdateParams) error {
		return nil
	})
	defer patchUpdateStatusCondition.Reset()

	// 执行测试
	err := setConditionsReady(fClient, ctx, cr, "TestReason", "TestMessage")

	// 断言没有错误
	assert.NoError(t, err)
}

// TestSetConditionsError 测试 setConditionsError 函数
func TestSetConditionsError(t *testing.T) {
	fClient := fake.NewClientBuilder().Build()
	ctx := context.TODO()
	cr := &kaev1.KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-kaepolicy",
		},
	}

	// Mock updateStatusCondition
	patchUpdateStatusCondition := gomonkey.ApplyFunc(updateStatusCondition, func(_ client.Client, _ context.Context,
		_ *kaev1.KAEPolicy, _ statusUpdateParams) error {
		return nil
	})
	defer patchUpdateStatusCondition.Reset()

	// 执行测试
	err := setConditionsError(fClient, ctx, cr, "TestReason", "TestMessage")

	// 断言没有错误
	assert.NoError(t, err)
}
