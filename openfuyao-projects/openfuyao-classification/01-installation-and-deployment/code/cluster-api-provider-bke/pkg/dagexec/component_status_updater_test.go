/******************************************************************
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * installer is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package dagexec

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
)

func newUpdaterTestCluster(name string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: "ns"},
	}
}

func newUpdaterFakeClient(t *testing.T, bc *bkev1beta1.BKECluster) client.Client {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	return fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(bc).
		WithStatusSubresource(bc).
		Build()
}

func markRef(bc *bkev1beta1.BKECluster, name string, typ StatusComponentType, nodeIP string) ComponentMarkRef {
	return ComponentMarkRef{Cluster: bc, Name: name, ComponentType: typ, NodeIP: nodeIP}
}

func TestBKEComponentStatusUpdater_MarkInstalledIdempotent(t *testing.T) {
	bc := newUpdaterTestCluster("c1")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)
	ctx := context.Background()

	require.NoError(t, updater.MarkInstalled(ctx, markRef(bc, "etcd", StatusComponentTypeCluster, ""), "v1.2.0"))
	first := bc.Status.ClusterComponentStatuses["etcd"]
	require.Equal(t, confv1beta1.LifecyclePhaseInstalled, first.Phase)
	require.Equal(t, "v1.2.0", first.CurrentVersion)
	require.NotNil(t, first.LastTransitionTime)

	require.NoError(t, updater.MarkInstalled(ctx, markRef(bc, "etcd", StatusComponentTypeCluster, ""), "v1.2.0"))
	second := bc.Status.ClusterComponentStatuses["etcd"]
	assert.Equal(t, confv1beta1.LifecyclePhaseInstalled, second.Phase)
	assert.Equal(t, "v1.2.0", second.CurrentVersion)
	require.NotNil(t, second.LastTransitionTime)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, cli.Get(ctx, client.ObjectKeyFromObject(bc), got))
	assert.Equal(t, confv1beta1.LifecyclePhaseInstalled, got.Status.ClusterComponentStatuses["etcd"].Phase)
	assert.Equal(t, "v1.2.0", got.Status.ClusterComponentStatuses["etcd"].CurrentVersion)
}

func TestBKEComponentStatusUpdater_MarkFailedWritesMessageAndTransition(t *testing.T) {
	bc := newUpdaterTestCluster("c2")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)
	ctx := context.Background()

	require.NoError(t, updater.MarkPending(ctx, markRef(bc, "kubelet", StatusComponentTypeNode, "10.0.0.1")))
	pending := bc.Status.ClusterComponentStatuses["kubelet"]
	require.Equal(t, confv1beta1.LifecyclePhasePending, pending.Phase)
	require.NotNil(t, pending.LastTransitionTime)

	require.NoError(t, updater.MarkFailed(ctx, markRef(bc, "kubelet", StatusComponentTypeNode, "10.0.0.1"), errors.New("boom")))
	failed := bc.Status.ClusterComponentStatuses["kubelet"]
	assert.Equal(t, confv1beta1.LifecyclePhaseFailed, failed.Phase)
	assert.Equal(t, "boom", failed.Message)
	assert.Equal(t, confv1beta1.LifecycleComponentTypeNode, failed.ComponentType)
	assert.Equal(t, "10.0.0.1", failed.NodeIP)
	require.NotNil(t, failed.LastTransitionTime)
}

func TestBKEComponentStatusUpdater_MarkRemovedClearsVersion(t *testing.T) {
	bc := newUpdaterTestCluster("c3")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)
	ctx := context.Background()

	require.NoError(t, updater.MarkInstalled(ctx, markRef(bc, "cni", StatusComponentTypeCluster, ""), "v3"))
	require.NoError(t, updater.MarkUninstalling(ctx, markRef(bc, "cni", StatusComponentTypeCluster, "")))
	require.NoError(t, updater.MarkRemoved(ctx, markRef(bc, "cni", StatusComponentTypeCluster, "")))

	st := bc.Status.ClusterComponentStatuses["cni"]
	assert.Equal(t, confv1beta1.LifecyclePhaseRemoved, st.Phase)
	assert.Empty(t, st.CurrentVersion)
}

func TestBKEComponentStatusUpdater_ClearComponentStatus(t *testing.T) {
	bc := newUpdaterTestCluster("c-clear")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)
	ctx := context.Background()

	require.NoError(t, updater.MarkPending(ctx, markRef(bc, "provider", StatusComponentTypeCluster, "")))
	require.NoError(t, updater.ClearComponentStatus(ctx, markRef(bc, "provider", StatusComponentTypeCluster, "")))

	if bc.Status.ClusterComponentStatuses != nil {
		if _, ok := bc.Status.ClusterComponentStatuses["provider"]; ok {
			t.Fatal("expected provider entry removed")
		}
	}

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, cli.Get(ctx, client.ObjectKeyFromObject(bc), got))
	if got.Status.ClusterComponentStatuses != nil {
		if _, ok := got.Status.ClusterComponentStatuses["provider"]; ok {
			t.Fatal("expected provider entry removed from API object")
		}
	}
}

func TestBKEComponentStatusUpdater_MarkRollingBack(t *testing.T) {
	bc := newUpdaterTestCluster("c4")
	cli := newUpdaterFakeClient(t, bc)
	updater := NewBKEComponentStatusUpdater(cli)
	ctx := context.Background()

	require.NoError(t, updater.MarkRollingBack(ctx, markRef(bc, "api", StatusComponentTypeCluster, "")))
	assert.Equal(t, confv1beta1.LifecyclePhaseRollingBack, bc.Status.ClusterComponentStatuses["api"].Phase)
}
