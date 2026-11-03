/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	colocationv1 "openfuyao.com/colocation-management/pkg/apis/v1"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
)

func newCheckpointScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	require.NoError(t, colocationv1.AddToScheme(scheme))
	return scheme
}

func newCheckpointReconciler(t *testing.T, objs ...client.Object) (*ContainerCheckpointReconciler, *aggregate.ClusterState) {
	t.Helper()
	scheme := newCheckpointScheme(t)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		Build()

	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})
	r := &ContainerCheckpointReconciler{
		ctx:          context.Background(),
		Client:       fakeClient,
		Scheme:       scheme,
		Recorder:     record.NewFakeRecorder(16),
		clusterState: cs,
	}
	return r, cs
}

func sampleCheckpoint(namespace, name, podName, containerName string) *colocationv1.ContainerCheckpoint {
	return &colocationv1.ContainerCheckpoint{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: colocationv1.ContainerCheckpointSpec{
			Namespace:     namespace,
			PodName:       podName,
			ContainerName: containerName,
		},
	}
}

func TestContainerCheckpointReconciler_Reconcile_NotFound_DeletesFromState(t *testing.T) {
	r, cs := newCheckpointReconciler(t)

	// Seed a checkpoint in the state, then reconcile the (missing) object.
	nn := types.NamespacedName{Namespace: "default", Name: "ckpt-1"}
	cs.AddOrUpdateCheckpoint(nn, sampleCheckpoint("default", "ckpt-1", "pod-1", "c1"))
	aggregateKey := aggregate.MakeAggregateStateKey(
		aggregate.PodID{Namespace: "default", PodName: "pod-1"}, "c1")
	require.NotNil(t, cs.GetCheckpoint(aggregateKey))

	res, err := r.Reconcile(context.Background(), controllerruntime.Request{NamespacedName: nn})
	assert.NoError(t, err)
	assert.False(t, res.Requeue)

	// Should have been removed from state.
	assert.Nil(t, cs.GetCheckpoint(aggregateKey))
}

func TestContainerCheckpointReconciler_Reconcile_NotFound_NoOpWhenAbsent(t *testing.T) {
	r, _ := newCheckpointReconciler(t)
	nn := types.NamespacedName{Namespace: "default", Name: "never-existed"}
	res, err := r.Reconcile(context.Background(), controllerruntime.Request{NamespacedName: nn})
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
}

func TestContainerCheckpointReconciler_Reconcile_Present_AddsToState(t *testing.T) {
	ckpt := sampleCheckpoint("default", "ckpt-1", "pod-1", "c1")
	r, cs := newCheckpointReconciler(t, ckpt)

	nn := types.NamespacedName{Namespace: "default", Name: "ckpt-1"}
	res, err := r.Reconcile(context.Background(), controllerruntime.Request{NamespacedName: nn})
	assert.NoError(t, err)
	assert.False(t, res.Requeue)

	aggregateKey := aggregate.MakeAggregateStateKey(
		aggregate.PodID{Namespace: "default", PodName: "pod-1"}, "c1")
	retrieved := cs.GetCheckpoint(aggregateKey)
	require.NotNil(t, retrieved)
	assert.Equal(t, "pod-1", retrieved.Spec.PodName)
	assert.Equal(t, "c1", retrieved.Spec.ContainerName)

	// GetAllCheckpoint should reflect the single stored object.
	all := cs.GetAllCheckpoint()
	assert.Len(t, all, 1)
}

func TestContainerCheckpointReconciler_SetupWithManager(t *testing.T) {
	scheme := newCheckpointScheme(t)
	mgr := newTestManager(t, scheme)
	r := &ContainerCheckpointReconciler{
		ctx:          context.Background(),
		mgr:          mgr,
		Client:       mgr.GetClient(),
		Scheme:       scheme,
		Recorder:     record.NewFakeRecorder(16),
		clusterState: aggregate.NewClusterState(context.Background(), &apps.Configuration{}),
	}
	assert.NoError(t, r.SetupWithManager())
}

func TestNewContainerCheckpointReconciler(t *testing.T) {
	scheme := newCheckpointScheme(t)
	mgr := newTestManager(t, scheme)
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})

	r := NewContainerCheckpointReconciler(context.Background(), mgr, cs)
	require.NotNil(t, r)
	assert.Equal(t, cs, r.clusterState)
	assert.NotNil(t, r.Client)
	assert.NotNil(t, r.Scheme)
	assert.NotNil(t, r.Recorder)
}
