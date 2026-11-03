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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"openfuyao.com/colocation-management/cmd/colocation-manager/apps"
	"openfuyao.com/colocation-management/pkg/colocation-manager/aggregate"
	utils_pkg "openfuyao.com/colocation-management/pkg/utils"
)

// plainNode returns a node that is NOT a colocation node (no label).
func plainNode(name string) *corev1.Node {
	return &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func newNodeTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, clientgoscheme.AddToScheme(scheme))
	return scheme
}

func newNodeReconciler(t *testing.T, objs ...client.Object) (*NodeReconciler, *aggregate.ClusterState) {
	t.Helper()
	scheme := newNodeTestScheme(t)
	fakeClient := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		Build()

	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})
	r := &NodeReconciler{
		ctx:                  context.Background(),
		Client:               fakeClient,
		Scheme:               scheme,
		Recorder:             record.NewFakeRecorder(16),
		clusterState:         cs,
		refreshNodesInterval: 100 * time.Millisecond,
	}
	return r, cs
}

func TestNodeReconciler_Reconcile_NotFound_DeletesFromState(t *testing.T) {
	r, cs := newNodeReconciler(t)
	// Seed state with a node, then reconcile a (now-missing) node.
	cs.AddOrUpdateNode(colocationNode("node-1"))
	require.True(t, cs.IsNodeExisted("node-1"))

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Name: "node-1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
	assert.False(t, cs.IsNodeExisted("node-1"))
}

func TestNodeReconciler_Reconcile_NonColocationNode_DeletesFromState(t *testing.T) {
	node := plainNode("node-1")
	r, cs := newNodeReconciler(t, node)
	// Pre-seed the state so we can observe the deletion path.
	cs.AddOrUpdateNode(colocationNode("node-1"))
	require.True(t, cs.IsNodeExisted("node-1"))

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Name: "node-1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
	// Not a colocation node -> removed from state.
	assert.False(t, cs.IsNodeExisted("node-1"))
}

func TestNodeReconciler_Reconcile_ColocationNode_AddsToState(t *testing.T) {
	node := colocationNode("node-1")
	r, cs := newNodeReconciler(t, node)

	req := controllerruntime.Request{
		NamespacedName: types.NamespacedName{Name: "node-1"},
	}
	res, err := r.Reconcile(context.Background(), req)
	assert.NoError(t, err)
	assert.False(t, res.Requeue)
	assert.True(t, cs.IsNodeExisted("node-1"))
	assert.NotNil(t, cs.FindNodeState("node-1"))
}

func TestNodeReconciler_doRefreshNode(t *testing.T) {
	colNode := colocationNode("col-node")
	plainNodeObj := plainNode("plain-node")

	r, cs := newNodeReconciler(t, colNode, plainNodeObj)

	// Pre-seed a node that will NOT be in the list so DeleteNotExistedNodes removes it.
	cs.AddOrUpdateNode(colocationNode("ghost-node"))
	require.True(t, cs.IsNodeExisted("ghost-node"))

	require.NoError(t, r.doRefreshNode())

	// Colocation node stays, plain node never added, ghost node removed.
	assert.True(t, cs.IsNodeExisted("col-node"))
	assert.False(t, cs.IsNodeExisted("plain-node"))
	assert.False(t, cs.IsNodeExisted("ghost-node"))
}

func TestNodeReconciler_SetupWithManager(t *testing.T) {
	scheme := newNodeTestScheme(t)
	mgr := newTestManager(t, scheme)
	r := &NodeReconciler{
		ctx:          context.Background(),
		mgr:          mgr,
		Client:       mgr.GetClient(),
		Scheme:       scheme,
		Recorder:     record.NewFakeRecorder(16),
		clusterState: aggregate.NewClusterState(context.Background(), &apps.Configuration{}),
	}
	assert.NoError(t, r.SetupWithManager(mgr))
}

func TestNodeReconciler_doRefreshNode_RemovesNonColocationFromState(t *testing.T) {
	colNode := colocationNode("col-node")
	r, cs := newNodeReconciler(t, colNode)

	// Seed a node that is in the list but later loses its colocation status.
	// Simulate by pre-adding a plain node into the state, then refreshing with
	// only the colocation node present -> plain node is not in curNodes map but
	// also was never added via refresh. The DeleteNotExistedNodes path is the
	// important one here.
	cs.AddOrUpdateNode(colocationNode("will-vanish"))
	require.NoError(t, r.doRefreshNode())
	assert.False(t, cs.IsNodeExisted("will-vanish"))
	assert.True(t, cs.IsNodeExisted("col-node"))
}

func TestNewNodeReconciler(t *testing.T) {
	scheme := newNodeTestScheme(t)
	mgr := newTestManager(t, scheme)
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})

	r := NewNodeReconciler(context.Background(), mgr, cs, 50*time.Millisecond)
	require.NotNil(t, r)
	assert.Equal(t, cs, r.clusterState)
	assert.Equal(t, 50*time.Millisecond, r.refreshNodesInterval)
	assert.NotNil(t, r.Client)
	assert.NotNil(t, r.Scheme)
	assert.NotNil(t, r.Recorder)
}

// TestNodeReconciler_Run drives Run() to completion by cancelling the
// reconciler's context. NotifySyncDone is guarded by recover because the
// underlying cache-sync channel is a one-shot global.
func TestNodeReconciler_Run(t *testing.T) {
	scheme := newNodeTestScheme(t)
	mgr := newTestManager(t, scheme)
	cs := aggregate.NewClusterState(context.Background(), &apps.Configuration{})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	r := NewNodeReconciler(ctx, mgr, cs, 10*time.Millisecond)

	// Pre-fill the cache-sync channel so the refresh goroutine does not block
	// on WaitCacheSync. Guard with recover: the channel is a one-shot global
	// that may already be closed if another test ran first.
	func() {
		defer func() { _ = recover() }()
		utils_pkg.NotifySyncDone()
	}()

	require.NoError(t, r.Run())

	// Let the refresh goroutine run doRefreshNode at least once, then cancel
	// so it exits via ctx.Done().
	time.Sleep(50 * time.Millisecond)
	cancel()
	time.Sleep(30 * time.Millisecond)
}
