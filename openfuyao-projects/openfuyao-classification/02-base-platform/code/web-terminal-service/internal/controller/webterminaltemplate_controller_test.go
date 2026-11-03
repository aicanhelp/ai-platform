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
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	"openfuyao.com/web-terminal-service/api/v1beta1"
)

type reconcileTestCase struct {
	name         string
	existingObjs []client.Object
	req          ctrl.Request
	expectErr    bool
	verify       func(t *testing.T, c client.Client, result ctrl.Result)
}

type updateStatusTestCase struct {
	name           string
	existingObj    *v1beta1.WebterminalTemplate
	targetObj      *v1beta1.WebterminalTemplate
	newStatus      v1beta1.WebterminalTemplateStatus
	mockUpdateFail bool
	expectPhase    v1beta1.WebTerminalTemplatePhase
	expectRequeue  time.Duration
}

type updateErrClient struct {
	client.Client
	err error
}

func (u *updateErrClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	return u.err
}

type deleteErrClient struct {
	client.Client
	err error
}

func (d *deleteErrClient) Delete(ctx context.Context, obj client.Object, opts ...client.DeleteOption) error {
	return d.err
}

type getErrClient struct {
	client.Client
	err error
}

func (g *getErrClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return g.err
}

func setupScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	utilruntime.Must(clientgoscheme.AddToScheme(s))
	utilruntime.Must(v1beta1.AddToScheme(s))
	return s
}

func newReadyPod(name, namespace string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Status: corev1.PodStatus{
			Conditions: []corev1.PodCondition{
				{
					Type:   corev1.PodReady,
					Status: corev1.ConditionTrue,
				},
			},
		},
	}
}

func TestWebterminalTemplateReconcilerReconcile(t *testing.T) {
	scheme := setupScheme()
	testCases := getReconcileTestCases()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeReconcileTest(t, tc, scheme)
		})
	}
}

const (
	bufferSize = 10
	updateTime = -30 * time.Minute
)

func executeReconcileTest(t *testing.T, tc reconcileTestCase, scheme *runtime.Scheme) {
	clientBuilder := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(tc.existingObjs...).
		WithStatusSubresource(tc.existingObjs...)

	fakeClient := clientBuilder.Build()
	recorder := record.NewFakeRecorder(bufferSize)

	r := &WebterminalTemplateReconciler{
		Client:   fakeClient,
		Scheme:   scheme,
		Recorder: recorder,
	}

	result, err := r.Reconcile(context.Background(), tc.req)

	if tc.expectErr {
		assert.Error(t, err)
	} else {
		assert.NoError(t, err)
	}

	if tc.verify != nil {
		tc.verify(t, fakeClient, result)
	}
}

func getReconcileTestCases() []reconcileTestCase {
	now := metav1.Now()
	return []reconcileTestCase{
		{
			name:         "Scenario 1: Resource Not Found",
			existingObjs: []client.Object{},
			req: ctrl.Request{
				NamespacedName: types.NamespacedName{Name: "test-term", Namespace: "default"},
			},
			expectErr: false,
			verify: func(t *testing.T, c client.Client, result ctrl.Result) {
				assert.Equal(t, ctrl.Result{}, result)
			},
		},
		{
			name: "Scenario 2: Add Finalizer",
			existingObjs: []client.Object{
				&v1beta1.WebterminalTemplate{
					ObjectMeta: metav1.ObjectMeta{Name: "test-term", Namespace: "default"},
					Spec: v1beta1.WebterminalTemplateSpec{
						PodTemplate: v1beta1.PodTemplate{
							ObjectMeta: v1beta1.PodTemplateObjectMeta{Name: "test-term", Namespace: "default"},
							Spec:       v1beta1.PodTemplateSpec{Containers: []corev1.Container{{Name: "c", Image: "i"}}},
						},
					},
				},
				newReadyPod("test-term", "default"),
			},
			req:       ctrl.Request{NamespacedName: types.NamespacedName{Name: "test-term", Namespace: "default"}},
			expectErr: false,
			verify: func(t *testing.T, c client.Client, result ctrl.Result) {
				fetched := &v1beta1.WebterminalTemplate{}
				_ = c.Get(context.TODO(), types.NamespacedName{Name: "test-term", Namespace: "default"}, fetched)
				assert.Contains(t, fetched.Finalizers, finalizer)
			},
		},
		{
			name: "Scenario 3: TTL Expiration",
			existingObjs: []client.Object{
				&v1beta1.WebterminalTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "expired-term",
						Namespace:  "default",
						Finalizers: []string{finalizer},
					},
					Spec: v1beta1.WebterminalTemplateSpec{
						RenewTime:  metav1.NewTime(time.Now().Add(updateTime)),
						ExistsTime: metav1.NewTime(time.Now().Add(updateTime)),
					},
					Status: v1beta1.WebterminalTemplateStatus{Phase: v1beta1.WebTerminalTemplateRunning},
				},
			},
			req:       ctrl.Request{NamespacedName: types.NamespacedName{Name: "expired-term", Namespace: "default"}},
			expectErr: false,
			verify: func(t *testing.T, c client.Client, result ctrl.Result) {
				fetched := &v1beta1.WebterminalTemplate{}
				err := c.Get(context.TODO(), types.NamespacedName{Name: "expired-term", Namespace: "default"}, fetched)
				assert.NoError(t, err)
				assert.False(t, fetched.DeletionTimestamp.IsZero(), "Object should be marked for deletion")
			},
		},
		{
			name: "Scenario 4: Logic after Pod Creation",
			existingObjs: []client.Object{
				&v1beta1.WebterminalTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Name:       "new-term",
						Namespace:  "default",
						Finalizers: []string{finalizer},
					},
					Spec: v1beta1.WebterminalTemplateSpec{
						PodTemplate: v1beta1.PodTemplate{
							ObjectMeta: v1beta1.PodTemplateObjectMeta{Name: "new-term", Namespace: "default"},
							Spec: v1beta1.PodTemplateSpec{
								Containers: []corev1.Container{{Name: "shell", Image: "busybox"}},
							},
						},
					},
				},
				newReadyPod("new-term", "default"),
			},
			req:       ctrl.Request{NamespacedName: types.NamespacedName{Name: "new-term", Namespace: "default"}},
			expectErr: false,
			verify: func(t *testing.T, c client.Client, result ctrl.Result) {
				fetched := &v1beta1.WebterminalTemplate{}
				err := c.Get(context.TODO(), types.NamespacedName{Name: "new-term", Namespace: "default"}, fetched)
				assert.NoError(t, err)
				assert.False(t, fetched.Spec.ExistsTime.IsZero())
				assert.Equal(t, v1beta1.WebTerminalTemplateRunning, fetched.Status.Phase)
			},
		},
		{
			name: "Scenario 5: Finalizer Cleanup",
			existingObjs: []client.Object{
				&v1beta1.WebterminalTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Name:              "deleting-term",
						Namespace:         "default",
						Finalizers:        []string{finalizer},
						DeletionTimestamp: &now,
					},
				},
				newReadyPod("deleting-term", "default"),
			},
			req:       ctrl.Request{NamespacedName: types.NamespacedName{Name: "deleting-term", Namespace: "default"}},
			expectErr: false,
			verify: func(t *testing.T, c client.Client, result ctrl.Result) {
				pod := &corev1.Pod{}
				err := c.Get(context.TODO(), types.NamespacedName{Name: "deleting-term", Namespace: "default"}, pod)
				assert.True(t, apierrors.IsNotFound(err), "Pod should be deleted")

				fetched := &v1beta1.WebterminalTemplate{}
				err = c.Get(context.TODO(), types.NamespacedName{Name: "deleting-term", Namespace: "default"}, fetched)
				assert.True(t, apierrors.IsNotFound(err), "Template should be deleted")
			},
		},
	}
}

func TestWebterminalTemplateReconcilerUpdateStatus(t *testing.T) {
	scheme := setupScheme()
	testCases := getUpdateStatusTestCases()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			executeUpdateStatusTest(t, tc, scheme)
		})
	}
}

func executeUpdateStatusTest(t *testing.T, tc updateStatusTestCase, scheme *runtime.Scheme) {
	builder := fake.NewClientBuilder().WithScheme(scheme)
	if tc.existingObj != nil {
		builder.WithObjects(tc.existingObj).WithStatusSubresource(tc.existingObj)
	}

	if tc.mockUpdateFail {
		builder.WithInterceptorFuncs(interceptor.Funcs{
			SubResourceUpdate: func(ctx context.Context, client client.Client, subResourceName string,
				obj client.Object, opts ...client.SubResourceUpdateOption) error {
				return errors.New("mock update error")
			},
		})
	}

	fakeClient := builder.Build()

	r := &WebterminalTemplateReconciler{
		Client: fakeClient,
		Scheme: scheme,
	}

	result, err := r.updateStatus(context.Background(), tc.targetObj, tc.newStatus)

	assert.NoError(t, err)
	assert.Equal(t, tc.expectRequeue, result.RequeueAfter)

	if tc.existingObj != nil && !tc.mockUpdateFail {
		updatedObj := &v1beta1.WebterminalTemplate{}
		err := fakeClient.Get(context.Background(),
			types.NamespacedName{Name: tc.targetObj.Name, Namespace: tc.targetObj.Namespace}, updatedObj)
		assert.NoError(t, err)
		assert.Equal(t, tc.expectPhase, updatedObj.Status.Phase)

		if len(tc.newStatus.Conditions) > 0 {
			assert.Equal(t, tc.newStatus.Conditions[0].Reason, updatedObj.Status.Conditions[0].Reason)
		}
	}
}

func getUpdateStatusTestCases() []updateStatusTestCase {
	return []updateStatusTestCase{
		{
			name: "Success: Status should be updated",
			existingObj: &v1beta1.WebterminalTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "term-1", Namespace: "default"},
				Status:     v1beta1.WebterminalTemplateStatus{Phase: v1beta1.WebTerminalTemplateStarting},
			},
			targetObj: &v1beta1.WebterminalTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "term-1", Namespace: "default"},
			},
			newStatus: v1beta1.WebterminalTemplateStatus{
				Phase: v1beta1.WebTerminalTemplateRunning,
				Conditions: []v1beta1.WebTerminalTemplateCondition{
					{Reason: "Test", Status: "True"},
				},
			},
			expectPhase:   v1beta1.WebTerminalTemplateRunning,
			expectRequeue: defaultUpdateFrequency,
		},
		{
			name:        "Failure: Object not found",
			existingObj: nil,
			targetObj: &v1beta1.WebterminalTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "term-missing", Namespace: "default"},
			},
			newStatus:     v1beta1.WebterminalTemplateStatus{Phase: v1beta1.WebTerminalTemplateRunning},
			expectRequeue: 0,
		},
		{
			name: "Failure: Update fails",
			existingObj: &v1beta1.WebterminalTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "term-fail", Namespace: "default"},
			},
			targetObj: &v1beta1.WebterminalTemplate{
				ObjectMeta: metav1.ObjectMeta{Name: "term-fail", Namespace: "default"},
			},
			newStatus:      v1beta1.WebterminalTemplateStatus{Phase: v1beta1.WebTerminalTemplateRunning},
			mockUpdateFail: true,
			expectRequeue:  0,
		},
	}
}

func TestHandleFinalizer_AddFinalizerFail(t *testing.T) {
	r := &WebterminalTemplateReconciler{}
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"},
	}
	patch := gomonkey.ApplyFunc(controllerutil.AddFinalizer, func(obj client.Object, finalizer string) bool {
		return false
	})
	defer patch.Reset()

	shouldReturn, result, err := r.handleFinalizer(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Equal(t, ctrl.Result{}, result)
	assert.NoError(t, err)
}

func TestHandleFinalizer_UpdateError(t *testing.T) {
	r := &WebterminalTemplateReconciler{
		Client: &updateErrClient{
			Client: fake.NewClientBuilder().Build(),
			err:    errors.New("update error"),
		},
	}
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"},
	}
	patch1 := gomonkey.ApplyFunc(controllerutil.AddFinalizer, func(obj client.Object, finalizer string) bool {
		return true
	})
	defer patch1.Reset()

	shouldReturn, _, err := r.handleFinalizer(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestHandleFinalizer_DeletePathDeleteError(t *testing.T) {
	r := &WebterminalTemplateReconciler{}
	now := metav1.Now()
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "t1",
			Namespace:         "default",
			Finalizers:        []string{finalizer},
			DeletionTimestamp: &now,
		},
	}
	patch1 := gomonkey.ApplyFunc(controllerutil.ContainsFinalizer, func(obj client.Object, f string) bool { return true })
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(r), "deletePodTemplate",
		func(_ *WebterminalTemplateReconciler, ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
			return errors.New("delete error")
		})
	defer patch1.Reset()
	defer patch2.Reset()

	shouldReturn, _, err := r.handleFinalizer(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestHandleFinalizer_RemoveFinalizerFail(t *testing.T) {
	r := &WebterminalTemplateReconciler{}
	now := metav1.Now()
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:              "t1",
			Namespace:         "default",
			Finalizers:        []string{finalizer},
			DeletionTimestamp: &now,
		},
	}
	patch1 := gomonkey.ApplyFunc(controllerutil.ContainsFinalizer, func(obj client.Object, f string) bool { return true })
	patch2 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(r), "deletePodTemplate",
		func(_ *WebterminalTemplateReconciler, ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
			return nil
		})
	patch3 := gomonkey.ApplyFunc(controllerutil.RemoveFinalizer, func(obj client.Object, f string) bool { return false })
	defer patch1.Reset()
	defer patch2.Reset()
	defer patch3.Reset()

	shouldReturn, _, err := r.handleFinalizer(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.NoError(t, err)
}

func TestEnsurePodRunning_CreateError(t *testing.T) {
	r := &WebterminalTemplateReconciler{}
	wt := &v1beta1.WebterminalTemplate{
		Spec: v1beta1.WebterminalTemplateSpec{},
	}
	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(r), "createPodTemplate",
		func(_ *WebterminalTemplateReconciler, ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
			return errors.New("create error")
		})
	defer patch.Reset()

	shouldReturn, _, err := r.ensurePodRunning(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestEnsurePodRunning_UpdateError(t *testing.T) {
	r := &WebterminalTemplateReconciler{
		Client: &updateErrClient{
			Client: fake.NewClientBuilder().Build(),
			err:    errors.New("update error"),
		},
	}
	wt := &v1beta1.WebterminalTemplate{
		Spec: v1beta1.WebterminalTemplateSpec{},
	}
	patch1 := gomonkey.ApplyPrivateMethod(reflect.TypeOf(r), "createPodTemplate",
		func(_ *WebterminalTemplateReconciler, ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
			return nil
		})
	defer patch1.Reset()

	shouldReturn, _, err := r.ensurePodRunning(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestEnsurePodRunning_StatusUpdateError(t *testing.T) {
	scheme := setupScheme()
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"},
		Spec:       v1beta1.WebterminalTemplateSpec{},
	}
	builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(wt).WithStatusSubresource(wt)
	builder.WithInterceptorFuncs(interceptor.Funcs{
		SubResourceUpdate: func(ctx context.Context, c client.Client, subResourceName string,
			obj client.Object, opts ...client.SubResourceUpdateOption) error {
			return errors.New("status error")
		},
	})
	fakeClient := builder.Build()
	r := &WebterminalTemplateReconciler{Client: fakeClient, Scheme: scheme}

	patch := gomonkey.ApplyPrivateMethod(reflect.TypeOf(r), "createPodTemplate",
		func(_ *WebterminalTemplateReconciler, ctx context.Context, obj *v1beta1.WebterminalTemplate) error {
			return nil
		})
	defer patch.Reset()

	shouldReturn, _, err := r.ensurePodRunning(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestCheckTTL_DeleteError(t *testing.T) {
	r := &WebterminalTemplateReconciler{
		Client: &deleteErrClient{
			Client: fake.NewClientBuilder().Build(),
			err:    errors.New("delete error"),
		},
	}
	wt := &v1beta1.WebterminalTemplate{
		Spec: v1beta1.WebterminalTemplateSpec{
			RenewTime: metav1.NewTime(time.Now().Add(-checkTimePeriod * 2)),
		},
	}

	shouldReturn, _, err := r.checkTTL(context.Background(), wt)
	assert.True(t, shouldReturn)
	assert.Error(t, err)
}

func TestSyncStatus_PodGetError(t *testing.T) {
	r := &WebterminalTemplateReconciler{
		Client: &getErrClient{
			Client: fake.NewClientBuilder().Build(),
			err:    errors.New("get error"),
		},
	}
	wt := &v1beta1.WebterminalTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: "t1", Namespace: "default"},
		Status:     v1beta1.WebterminalTemplateStatus{Phase: v1beta1.WebTerminalTemplateRunning},
	}
	_, err := r.syncStatus(context.Background(), wt)
	assert.Error(t, err)
}
