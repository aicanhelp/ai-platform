// Copyright (c) 2024 Huawei Technologies Co., Ltd.
// openFuyao is licensed under Mulan PSL v2.
// You can use this software according to the terms and conditions of the Mulan PSL v2.
// You may obtain a copy of Mulan PSL v2 at:
//          http://license.coscl.org.cn/MulanPSL2
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
// EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
// MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See the Mulan PSL v2 for more details.

package controller

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	v1 "openfuyao.com/npu-operator/api/v1"
)

type listErrClient struct {
	client.Client
	err error
}

func (c listErrClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	return c.err
}

func TestNPUClusterPolicyReconcilerGetReferrerToNode(t *testing.T) {
	instance1 := getInstance()
	instance1.Name = "cluster-a"
	instance2 := getInstance()
	instance2.Name = "cluster-b"

	scheme := mockScheme()
	cl := fakeClientBuilder(scheme, instance1, instance2)
	r := &NPUClusterPolicyReconciler{Client: cl}

	reqs := r.getReferrerToNode(context.Background(), &corev1.Node{})
	require.Len(t, reqs, 2)
	assert.ElementsMatch(t, []string{"cluster-a", "cluster-b"}, []string{
		reqs[0].NamespacedName.Name, reqs[1].NamespacedName.Name,
	})
}

func TestNPUClusterPolicyReconcilerGetReferrerToNode_ListError(t *testing.T) {
	r := &NPUClusterPolicyReconciler{
		Client: listErrClient{
			Client: fakeClient(),
			err:    errors.New("list failed"),
		},
	}
	reqs := r.getReferrerToNode(context.Background(), &corev1.Node{})
	assert.Empty(t, reqs)
}

func TestNodeEventFilter(t *testing.T) {
	createEvent := event.CreateEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					npuDeviceLabelPrefix + "d100.present": npuDeviceLabelValue,
				},
			},
		},
	}
	assert.True(t, nodeEventFilter.Create(createEvent))

	updateEvent := event.UpdateEvent{
		ObjectNew: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					npuPresentLabelKey: "true",
				},
			},
		},
	}
	assert.True(t, nodeEventFilter.Update(updateEvent))

	deleteEvent := event.DeleteEvent{
		Object: &corev1.Node{
			ObjectMeta: metav1.ObjectMeta{
				Labels: map[string]string{
					npuDeviceLabelPrefix + "d500.present": npuDeviceLabelValue,
				},
			},
		},
	}
	assert.True(t, nodeEventFilter.Delete(deleteEvent))
}

func TestNPUClusterPolicyReconcilerReconcile(t *testing.T) {
	req := reconcile.Request{NamespacedName: types.NamespacedName{Name: "cluster"}}
	ctx := context.Background()

	t.Run("ignore not found", func(t *testing.T) {
		r := &NPUClusterPolicyReconciler{}
		patch := gomonkey.ApplyPrivateMethod(r, "getInstance", func(*NPUClusterPolicyReconciler, context.Context, string) error {
			return apierrors.NewNotFound(schema.GroupResource{Group: "npu.openfuyao.com", Resource: "npuclusterpolicies"}, "cluster")
		})
		defer patch.Reset()

		result, err := r.Reconcile(ctx, req)
		require.NoError(t, err)
		assert.Equal(t, ctrl.Result{}, result)
	})

	t.Run("load components failed", func(t *testing.T) {
		r := &NPUClusterPolicyReconciler{}
		p1 := gomonkey.ApplyPrivateMethod(r, "getInstance", func(*NPUClusterPolicyReconciler, context.Context, string) error {
			return nil
		})
		p2 := gomonkey.ApplyPrivateMethod(r, "loadComponents", func(*NPUClusterPolicyReconciler, context.Context) error {
			return errors.New("load failed")
		})
		p3 := gomonkey.ApplyPrivateMethod(r, "setConditionsError", func(*NPUClusterPolicyReconciler, context.Context, string, string) error {
			return nil
		})
		defer p1.Reset()
		defer p2.Reset()
		defer p3.Reset()

		_, err := r.Reconcile(ctx, req)
		require.Error(t, err)
		assert.ErrorContains(t, err, "load components")
	})

	t.Run("label npu nodes failed", func(t *testing.T) {
		r := &NPUClusterPolicyReconciler{}
		p1 := gomonkey.ApplyPrivateMethod(r, "getInstance", func(*NPUClusterPolicyReconciler, context.Context, string) error {
			return nil
		})
		p2 := gomonkey.ApplyPrivateMethod(r, "loadComponents", func(*NPUClusterPolicyReconciler, context.Context) error {
			return nil
		})
		p3 := gomonkey.ApplyPrivateMethod(r, "labelNPUNodes", func(*NPUClusterPolicyReconciler, context.Context) error {
			return errors.New("label failed")
		})
		p4 := gomonkey.ApplyPrivateMethod(r, "setConditionsError", func(*NPUClusterPolicyReconciler, context.Context, string, string) error {
			return nil
		})
		defer p1.Reset()
		defer p2.Reset()
		defer p3.Reset()
		defer p4.Reset()

		_, err := r.Reconcile(ctx, req)
		require.Error(t, err)
		assert.ErrorContains(t, err, "label NPU nodes")
	})

	t.Run("success and requeue result", func(t *testing.T) {
		r := &NPUClusterPolicyReconciler{}
		expected := ctrl.Result{RequeueAfter: 2 * time.Second}
		p1 := gomonkey.ApplyPrivateMethod(r, "getInstance", func(*NPUClusterPolicyReconciler, context.Context, string) error {
			return nil
		})
		p2 := gomonkey.ApplyPrivateMethod(r, "loadComponents", func(*NPUClusterPolicyReconciler, context.Context) error {
			return nil
		})
		p3 := gomonkey.ApplyPrivateMethod(r, "labelNPUNodes", func(rr *NPUClusterPolicyReconciler, _ context.Context) error {
			rr.hasNFDLabels = false
			rr.hasNPUNodes = false
			return nil
		})
		p4 := gomonkey.ApplyPrivateMethod(r, "reconcileComponents", func(*NPUClusterPolicyReconciler, context.Context) (ctrl.Result, error) {
			return expected, nil
		})
		defer p1.Reset()
		defer p2.Reset()
		defer p3.Reset()
		defer p4.Reset()

		result, err := r.Reconcile(ctx, req)
		require.NoError(t, err)
		assert.Equal(t, expected, result)
	})
}

func TestSetConditionsFunctions(t *testing.T) {
	r := &NPUClusterPolicyReconciler{}

	t.Run("setConditionsReady and callback updates status", func(t *testing.T) {
		var callback updateStatusCallback
		patch := gomonkey.ApplyPrivateMethod(r, "updateStatus", func(*NPUClusterPolicyReconciler, context.Context, updateStatusCallback) error {
			callback = func(s *v1.NPUClusterPolicyStatus) bool {
				return s.Phase != v1.PolicyReady
			}
			return nil
		})
		defer patch.Reset()
		require.NoError(t, r.setConditionsReady(context.Background(), reconciledReason, "ok"))
		require.NotNil(t, callback)
	})

	t.Run("setConditionsError returns updateStatus error", func(t *testing.T) {
		patch := gomonkey.ApplyPrivateMethod(r, "updateStatus", func(*NPUClusterPolicyReconciler, context.Context, updateStatusCallback) error {
			return errors.New("status update failed")
		})
		defer patch.Reset()
		err := r.setConditionsError(context.Background(), reconcileFailedReason, "failed")
		require.Error(t, err)
		assert.ErrorContains(t, err, "status update failed")
	})
}

func fakeClientBuilder(s *runtime.Scheme, objs ...client.Object) client.Client {
	return fake.NewClientBuilder().WithScheme(s).WithObjects(objs...).Build()
}
