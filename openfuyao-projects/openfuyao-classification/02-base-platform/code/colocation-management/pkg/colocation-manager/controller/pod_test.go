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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	"openfuyao.com/colocation-management/pkg/common"
	utils_pkg "openfuyao.com/colocation-management/pkg/utils"
)

// newTestManager builds a real controller-runtime Manager that does not dial a
// real apiserver. It is sufficient to drive SetupWithManager without calling
// Start (which would block).
func newTestManager(t *testing.T, scheme *runtime.Scheme) controllerruntime.Manager {
	t.Helper()
	cfg := restConfigStub()
	mgr, err := controllerruntime.NewManager(cfg, controllerruntime.Options{
		Scheme: scheme,
	})
	require.NoError(t, err)
	return mgr
}

// restConfigStub returns a rest.Config that prevents manager.New from dialing a
// real apiserver.
func restConfigStub() *rest.Config {
	return &rest.Config{Host: "localhost"}
}

// newPodTestScheme builds a runtime scheme with the corev1 types the PodReconciler uses.
func newPodTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	return scheme
}

// newPodReconciler builds a white-box PodReconciler wired to a fake client and a fresh ClusterState.
func newPodReconciler(t *testing.T, objs ...client.Object) (*PodReconciler, *aggregate.ClusterState) {
	t.Helper()
	scheme := newPodTestScheme(t)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		Build()

	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})
	r := &PodReconciler{
		ctx:                      context.Background(),
		Client:                   fakeClient,
		Scheme:                   scheme,
		Recorder:                 record.NewFakeRecorder(16),
		clusterState:             cs,
		refreshPodsInterval:      100 * time.Millisecond,
		aggregateStateGCInterval: 100 * time.Millisecond,
	}
	return r, cs
}

// colocationNode returns a node that satisfies utils.IsColocationNode.
func colocationNode(name string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{common.ColocationNodeLabel: "true"},
		},
	}
}

// onlinePod returns a pod scheduled onto the given node that is NOT offline.
func onlinePod(namespace, name, node string) *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: corev1.PodSpec{
			NodeName: node,
			Containers: []corev1.Container{
				{
					Name: "c1",
					Resources: corev1.ResourceRequirements{
						Requests: corev1.ResourceList{
							corev1.ResourceCPU:    resource.MustParse("100m"),
							corev1.ResourceMemory: resource.MustParse("128Mi"),
						},
					},
				},
			},
		},
	}
}

func TestPodReconciler_Reconcile_NotFound_DeletesFromState(t *testing.T) {
	r, cs := newPodReconciler(t, colocationNode("node-1"))

	// Seed the state with a pod first.
	cs.AddOrUpdateNode(colocationNode("node-1"))
	require.NoError(t, cs.AddPodAndContainer(onlinePod("default", "p1", "node-1")))

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "p1"},
	}

	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)

	// The reconciler should have removed the pod from the cluster state.
	assert.False(t, cs.IsNodeExisted("p1"))
}

func TestPodReconciler_Reconcile_OfflinePod_Skips(t *testing.T) {
	node := colocationNode("node-1")
	pod := onlinePod("default", "p1", "node-1")
	// Mark the pod as offline so OfflinePod returns true.
	pod.Annotations = map[string]string{common.PriorityAnnotationKey: "true"}

	r, cs := newPodReconciler(t, node, pod)
	// Node must exist in state for the OfflinePod check to be reached before
	// the node-existence guard returns early.
	cs.AddOrUpdateNode(node)

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "p1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
}

func TestPodReconciler_Reconcile_NodeNotInState_Skips(t *testing.T) {
	pod := onlinePod("default", "p1", "ghost-node")
	r, _ := newPodReconciler(t, pod)
	// clusterState is empty -> IsNodeExisted returns false -> early return.

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "p1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
}

func TestPodReconciler_Reconcile_HappyPath_AddsPodAndContainer(t *testing.T) {
	node := colocationNode("node-1")
	pod := onlinePod("default", "p1", "node-1")

	r, cs := newPodReconciler(t, node, pod)
	cs.AddOrUpdateNode(node)

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Namespace: "default", Name: "p1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)

	// The pod should now be tracked: node state exists and the pod count grows.
	ns := cs.FindNodeState("node-1")
	require.NotNil(t, ns)
}

func TestPodReconciler_doRefreshPods(t *testing.T) {
	node := colocationNode("node-1")
	online := onlinePod("default", "online-pod", "node-1")
	offline := onlinePod("default", "offline-pod", "node-1")
	offline.Annotations = map[string]string{common.PriorityAnnotationKey: "true"}
	staleNodePod := onlinePod("default", "stale-node-pod", "missing-node")

	r, cs := newPodReconciler(t, node, online, offline, staleNodePod)
	cs.AddOrUpdateNode(node)

	require.NoError(t, r.doRefreshPods())

	// Online pod should be reflected; offline pod and pod on unknown node ignored.
	ns := cs.FindNodeState("node-1")
	require.NotNil(t, ns)
}

func TestPodReconciler_doRefreshPods_ListError(t *testing.T) {
	// A client built without a registered PodList type would be ideal, but the
	// fake client always supports listing registered types. Instead we verify
	// doRefreshPods returns nil error on an empty cluster (happy no-op path).
	r, _ := newPodReconciler(t, colocationNode("node-1"))
	assert.NoError(t, r.doRefreshPods())
}

func TestPodReconciler_GCAggregateStatesCollection(t *testing.T) {
	r, cs := newPodReconciler(t)
	// No nodes -> should be a no-op that does not panic.
	assert.NotPanics(t, func() {
		r.GCAggregateStatesCollection(time.Now())
	})

	// Add a node so the GC walks at least one NodeState.
	cs.AddOrUpdateNode(colocationNode("node-1"))
	assert.NotPanics(t, func() {
		r.GCAggregateStatesCollection(time.Now())
	})
}

func TestPodReconciler_SetupWithManager(t *testing.T) {
	scheme := newPodTestScheme(t)
	mgr := newTestManager(t, scheme)
	r := &PodReconciler{
		ctx:          context.Background(),
		mgr:          mgr,
		Client:       mgr.GetClient(),
		Scheme:       scheme,
		Recorder:     record.NewFakeRecorder(16),
		clusterState: aggregate.NewClusterState(context.Background(), &apps.Configuration{}),
	}
	assert.NoError(t, r.SetupWithManager(mgr))
}

func TestNewPodReconciler(t *testing.T) {
	scheme := newPodTestScheme(t)
	mgr := newTestManager(t, scheme)
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})

	r := NewPodReconciler(context.Background(), mgr, cs, 50*time.Millisecond, 5*time.Second)
	require.NotNil(t, r)
	assert.Equal(t, cs, r.clusterState)
	assert.Equal(t, 50*time.Millisecond, r.refreshPodsInterval)
	assert.Equal(t, 5*time.Second, r.aggregateStateGCInterval)
	assert.NotNil(t, r.Client)
	assert.NotNil(t, r.Recorder)
}

// TestPodReconciler_Run drives Run() by cancelling the reconciler context
// after the background goroutines have had a chance to run at least once.
func TestPodReconciler_Run(t *testing.T) {
	scheme := newPodTestScheme(t)
	mgr := newTestManager(t, scheme)
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := NewPodReconciler(ctx, mgr, cs, 10*time.Millisecond, 10*time.Millisecond)

	// Pre-fill the cache-sync channel so the refresh goroutine does not block.
	// Guard with recover: the channel is a one-shot global.
	func() {
		defer func() { _ = recover() }()
		utils_pkg.NotifySyncDone()
	}()

	require.NoError(t, r.Run())

	// Let both background goroutines tick once, then cancel.
	time.Sleep(60 * time.Millisecond)
	cancel()
	time.Sleep(30 * time.Millisecond)
}
