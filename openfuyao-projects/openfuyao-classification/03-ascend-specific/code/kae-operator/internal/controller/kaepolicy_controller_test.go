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
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	kaev1 "gopkg.openfuyao.cn/kae-operator/api/v1"
)

var fakeScheme = runtime.NewScheme()

func init() {
	_ = kaev1.AddToScheme(fakeScheme)
	_ = appsv1.AddToScheme(fakeScheme)
	_ = corev1.AddToScheme(fakeScheme)
}

// TestReconcileGetInstanceError 获取实例失败
func TestReconcileGetInstanceError(t *testing.T) {
	ctx := context.TODO()
	// 清理环境
	kaePolicyManager = KAEPolicyManager{}
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "test-kae-policy",
		},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethodReturn(r.Client, "Get",
		errors.NewNotFound(schema.GroupResource{Group: "test", Resource: "test"}, "test"))

	result, err := r.Reconcile(ctx, req)
	assert.Nil(t, err)
	assert.Equal(t, reconcile.Result{}, result)

	patches.Reset()
	patches.ApplyMethodReturn(r.Client, "Get", fmt.Errorf("test err"))
	result, err = r.Reconcile(ctx, req)
	assert.NotNil(t, err)
	assert.Equal(t, reconcile.Result{}, result)
}

// TestReconcileInstanceAlreadyExists 已存在实例，不需要创建新实例
func TestReconcileInstanceAlreadyExists(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	// 清理环境
	kaePolicyManager = KAEPolicyManager{}
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	patches.ApplyMethodReturn(r.Client, "Get", nil)
	patches.ApplyMethodReturn(r.Client.Status(), "Update", nil)

	// Mock kaePolicyManager state
	kaePolicyManager.kaeInstance = &kaev1.KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "existing-instance",
		},
	}
	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "test-kae-exist",
		},
	}
	result, err := r.Reconcile(context.TODO(), req)
	assert.NoError(t, err)
	assert.Equal(t, reconcile.Result{}, result)

}

// TestReconcileInitError 初始化异常
func TestReconcileInitError(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	// 环境清理
	kaePolicyManager = KAEPolicyManager{}
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyMethodReturn(r.Client, "Get", nil)
	patches.ApplyPrivateMethod(&kaePolicyManager, "init", func(_ *KAEPolicyManager, _ context.Context,
		_ *KAEPolicyReconciler, _ *kaev1.KAEPolicy) error {
		return fmt.Errorf("test err")
	})

	// Mock setConditionsError
	patches.ApplyFuncReturn(setConditionsError, nil)
	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "test-init-err",
		},
	}

	result, err := r.Reconcile(context.TODO(), req)
	assert.Error(t, err)
	assert.Equal(t, reconcile.Result{}, result)

}

// TestReconcileSuccess 协调成功
func TestReconcileSuccess(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	// 清理环境
	kaePolicyManager = KAEPolicyManager{}

	patches.ApplyMethodReturn(r.Client, "Get", nil)
	patches.ApplyPrivateMethod(&kaePolicyManager, "init", func(_ *KAEPolicyManager, _ context.Context,
		_ *KAEPolicyReconciler, _ *kaev1.KAEPolicy) error {
		return nil
	})
	patches.ApplyPrivateMethod(r, "deployAllComponents", func(_ *KAEPolicyReconciler, _ context.Context,
		_ ctrl.Request, _ *kaev1.KAEPolicy) (kaev1.State, []string, error) {
		return kaev1.Ready, []string{}, nil
	})
	patches.ApplyPrivateMethod(r, "updateStateAndConditions", func(_ *KAEPolicyReconciler, _ context.Context,
		_ ctrl.Request, _ kaev1.State, _ []string, _ *kaev1.KAEPolicy) (ctrl.Result, error) {
		return ctrl.Result{}, nil
	})

	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "test-success",
		},
	}
	result, err := r.Reconcile(context.TODO(), req)
	assert.NoError(t, err)
	assert.Equal(t, reconcile.Result{}, result)
}

// TestUpdateStateAndConditions 状态更新
func TestUpdateStateAndConditions(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}
	req := reconcile.Request{
		NamespacedName: client.ObjectKey{
			Namespace: "default",
			Name:      "AndConditionsNotReady",
		},
	}
	result, err := r.updateStateAndConditions(context.TODO(), req, kaev1.NotReady, []string{"driver"},
		&kaev1.KAEPolicy{})
	assert.Nil(t, err)
	assert.Equal(t, ctrl.Result{RequeueAfter: requeueAfterTime}, result)

	kaePolicyManager = KAEPolicyManager{}
	kaePolicyManager.hasKAENodes = true
	result, err = r.updateStateAndConditions(context.TODO(), req, kaev1.Ready, []string{"driver"},
		&kaev1.KAEPolicy{})
	assert.NotNil(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

// TestDeployAllComponentsErr 部署错误
func TestDeployAllComponentsErr(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}

	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Namespace: "default",
			Name:      "DeployAllComponentsErr",
		},
	}
	instance := &kaev1.KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "DeployAllComponentsErr",
			Namespace: "default",
		},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(&kaePolicyManager, "init", func(_ *KAEPolicyManager, _ context.Context,
		_ *KAEPolicyReconciler, _ *kaev1.KAEPolicy) error {
		return nil
	})

	kaePolicyManager.index = 1
	status, _, err := r.deployAllComponents(context.TODO(), req, instance)
	assert.NotNil(t, err)
	assert.Equal(t, kaev1.NotReady, status)
}

func TestDeployAllComponentsNotReady(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().Build(),
	}

	req := ctrl.Request{
		NamespacedName: types.NamespacedName{
			Namespace: "default",
			Name:      "DeployNotReady",
		},
	}
	instance := &kaev1.KAEPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "DeployNotReady",
			Namespace: "default",
		},
	}

	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyPrivateMethod(&kaePolicyManager, "init", func(_ *KAEPolicyManager, _ context.Context,
		_ *KAEPolicyReconciler, _ *kaev1.KAEPolicy) error {
		return nil
	})
	patches.ApplyPrivateMethod(&kaePolicyManager, "deployComponent", func(_ *KAEPolicyManager) (kaev1.State, error) {
		return kaev1.NotReady, nil
	})

	kaePolicyManager = KAEPolicyManager{}
	status, _, err := r.deployAllComponents(context.TODO(), req, instance)
	assert.Nil(t, err)
	assert.Equal(t, kaev1.NotReady, status)
}

// TestGetComponentName 组件名称获取
func TestGetComponentName(t *testing.T) {
	kaePolicyManager.componentNames = []string{"component1"}
	tests := []struct {
		name     string
		index    int
		expected string
	}{
		{
			name:     "Valid Index",
			index:    0,
			expected: "component1",
		},
		{
			name:     "Invalid Index",
			index:    2,
			expected: "outbounds",
		},
		{
			name:     "Negative Index",
			index:    -1,
			expected: "outbounds",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getComponentName(tt.index)
			assert.Equal(t, tt.expected, result)
		})
	}
}

// TestPredicateCreate 测试 predicateCreate 函数
func TestPredicateCreate(t *testing.T) {
	// Test with KAE HPRE labels
	e1 := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-1",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
					"feature.node.kubernetes.io/system-os_release.ID":        "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
	result1 := predicateCreate(e1)
	assert.True(t, result1)

	// Test without KAE labels
	e2 := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-2",
				Labels: map[string]string{"test": "value"},
			},
		},
	}
	result2 := predicateCreate(e2)
	assert.False(t, result2)

	// Test with nil labels
	e3 := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-3",
				Labels: nil,
			},
		},
	}
	result3 := predicateCreate(e3)
	assert.False(t, result3)

	// Test with KAE SEC labels
	e4 := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-4",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a255.present": "true",
					"feature.node.kubernetes.io/system-os_release.ID":        "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
	result4 := predicateCreate(e4)
	assert.True(t, result4)

	// Test with KAE ZIP labels
	e5 := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-5",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1200_19e5_a250.present": "true",
					"feature.node.kubernetes.io/system-os_release.ID":        "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
	result5 := predicateCreate(e5)
	assert.True(t, result5)
}

// TestPredicateUpdate 测试 predicateUpdate 函数
func TestPredicateUpdate(t *testing.T) {
	// Test with watch label change
	e1 := event.UpdateEvent{
		ObjectOld: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-1",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "false",
				},
			},
		},
		ObjectNew: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-1",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				},
			},
		},
	}
	result1 := predicateUpdate(e1)
	assert.True(t, result1)

	// Test without label change
	e2 := event.UpdateEvent{
		ObjectOld: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-2",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				},
			},
		},
		ObjectNew: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-2",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				},
			},
		},
	}
	result2 := predicateUpdate(e2)
	assert.False(t, result2)

	// Test with nil old labels
	e3 := event.UpdateEvent{
		ObjectOld: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-3",
				Labels: nil,
			},
		},
		ObjectNew: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-3",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				},
			},
		},
	}
	result3 := predicateUpdate(e3)
	assert.True(t, result3)

	// Test with label removed
	e4 := event.UpdateEvent{
		ObjectOld: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-4",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
				},
			},
		},
		ObjectNew: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-4",
				Labels: map[string]string{},
			},
		},
	}
	result4 := predicateUpdate(e4)
	assert.True(t, result4)
}

// TestPredicateDelete 测试 predicateDelete 函数
func TestPredicateDelete(t *testing.T) {
	// Test with KAE HPRE labels
	e1 := event.DeleteEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-1",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a258.present": "true",
					"feature.node.kubernetes.io/system-os_release.ID":        "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
	result1 := predicateDelete(e1)
	assert.True(t, result1)

	// Test without KAE labels
	e2 := event.DeleteEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-2",
				Labels: map[string]string{"test": "value"},
			},
		},
	}
	result2 := predicateDelete(e2)
	assert.False(t, result2)

	// Test with nil labels
	e3 := event.DeleteEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name:   "test-node-3",
				Labels: nil,
			},
		},
	}
	result3 := predicateDelete(e3)
	assert.False(t, result3)

	// Test with KAE SEC labels
	e4 := event.DeleteEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Name: "test-node-4",
				Labels: map[string]string{
					"feature.node.kubernetes.io/pci-1000_19e5_a255.present": "true",
					"feature.node.kubernetes.io/system-os_release.ID":        "openEuler",
					"feature.node.kubernetes.io/system-os_release.VERSION_ID": "22.03",
				},
			},
		},
	}
	result4 := predicateDelete(e4)
	assert.True(t, result4)
}

// TestMapToKAEPolicyRequests 测试 mapToKAEPolicyRequests 函数
func TestMapToKAEPolicyRequests(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().WithScheme(fakeScheme).WithObjects(
			&kaev1.KAEPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-kae-1",
					Namespace: "default",
				},
			},
			&kaev1.KAEPolicy{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test-kae-2",
					Namespace: "default",
				},
			},
		).Build(),
	}

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	}

	requests := r.mapToKAEPolicyRequests(context.TODO(), node)
	assert.Len(t, requests, 2)
	assert.Equal(t, "test-kae-1", requests[0].Name)
	assert.Equal(t, "default", requests[0].Namespace)
	assert.Equal(t, "test-kae-2", requests[1].Name)
	assert.Equal(t, "default", requests[1].Namespace)
}

// TestMapToKAEPolicyRequestsEmpty 测试 mapToKAEPolicyRequests 空列表
func TestMapToKAEPolicyRequestsEmpty(t *testing.T) {
	r := &KAEPolicyReconciler{
		Client: fake.NewClientBuilder().WithScheme(fakeScheme).Build(),
	}

	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-node",
		},
	}

	requests := r.mapToKAEPolicyRequests(context.TODO(), node)
	assert.Len(t, requests, 0)
}

// TestDaemonSetIndexerExtractValueFunc 测试 DaemonSet indexer 的提取函数逻辑
func TestDaemonSetIndexerExtractValueFunc(t *testing.T) {
	// Test extractValueFunc with valid DaemonSet
	ds := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-ds",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: kaev1.GroupVersion.String(),
					Kind:       "KAEPolicy",
					Name:       "test-kae-policy",
					Controller: func() *bool { b := true; return &b }(),
				},
			},
		},
	}

	// Test with DaemonSet (should return owner name)
	extractValueFunc := func(rawObj client.Object) []string {
		daemonSet, ok := rawObj.(*appsv1.DaemonSet)
		if !ok {
			return nil
		}
		owner := metav1.GetControllerOf(daemonSet)
		if owner == nil {
			return nil
		}
		if owner.APIVersion != kaev1.GroupVersion.String() || owner.Kind != "KAEPolicy" {
			return nil
		}
		return []string{owner.Name}
	}

	result := extractValueFunc(ds)
	assert.NotNil(t, result)
	assert.Len(t, result, 1)
	assert.Equal(t, "test-kae-policy", result[0])

	// Test with nil object
	result = extractValueFunc(nil)
	assert.Nil(t, result)

	// Test with non-DaemonSet object
	node := &corev1.Node{}
	result = extractValueFunc(node)
	assert.Nil(t, result)

	// Test with DaemonSet without owner
	dsNoOwner := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-ds-no-owner",
			Namespace: "default",
		},
	}
	result = extractValueFunc(dsNoOwner)
	assert.Nil(t, result)

	// Test with DaemonSet with non-KAEPolicy owner
	dsOtherOwner := &appsv1.DaemonSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-ds-other-owner",
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: "apps/v1",
					Kind:       "Deployment",
					Name:       "test-deployment",
					Controller: func() *bool { b := true; return &b }(),
				},
			},
		},
	}
	result = extractValueFunc(dsOtherOwner)
	assert.Nil(t, result)
}
