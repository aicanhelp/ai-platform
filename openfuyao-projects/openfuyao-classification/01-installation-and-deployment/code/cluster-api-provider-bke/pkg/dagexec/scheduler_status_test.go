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
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/topology"
)

func TestRequeueAwareError(t *testing.T) {
	res, requeue := RequeueAwareError(nil)
	assert.False(t, requeue)
	assert.Equal(t, ctrl.Result{}, res)

	res, requeue = RequeueAwareError(errors.New("retry"))
	assert.True(t, requeue)
	assert.Equal(t, ctrl.Result{}, res)
}

func TestSchedulerMarkComponentCompleted(t *testing.T) {
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "kubelet", Version: "v1.32.0"}
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{TargetVersion: "v2"},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster: bc,
		Client:  cli,
	})

	require.NoError(t, sched.markComponentCompleted(context.Background(), execCtx, node))
	assert.True(t, bc.Status.DeclarativeUpgrade.IsCompleted("kubelet", "v1.32.0"))

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, cli.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c"}, got))
	require.NotNil(t, got.Status.DeclarativeUpgrade)
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("kubelet", "v1.32.0"))

	assert.NoError(t, sched.markComponentCompleted(context.Background(), nil, node))
}

func TestSchedulerMarkComponentFailed(t *testing.T) {
	sched := NewScheduler(Config{})
	node := &topology.ComponentNode{Name: "etcd", Version: "3.5.0"}
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{TargetVersion: "v2"},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()
	execCtx := NewExecutionContext(NewExecutionContextOptions{
		Cluster: bc,
		Client:  cli,
	})

	require.NoError(t, sched.markComponentFailed(context.Background(), execCtx, node, errors.New("upgrade failed")))
	require.NotNil(t, bc.Status.DeclarativeUpgrade.LastFailure)
	assert.Contains(t, bc.Status.DeclarativeUpgrade.LastError, "upgrade failed")
	assert.Equal(t, "etcd", bc.Status.DeclarativeUpgrade.LastFailure.Name)

	assert.NoError(t, sched.markComponentFailed(context.Background(), execCtx, node, nil))
	assert.NoError(t, sched.markComponentFailed(context.Background(), nil, node, errors.New("ignored")))
}

func TestSchedulerShouldSkipComponent(t *testing.T) {
	sched := NewScheduler(Config{})
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{
				TargetVersion: "v2",
				Completed: []confv1beta1.DeclarativeUpgradeComponentRecord{{
					Name:        "kubelet",
					Version:     "v1.32.0",
					CompletedAt: metav1.NewTime(time.Now()),
				}},
			},
		},
	}
	execCtx := NewExecutionContext(NewExecutionContextOptions{Cluster: bc})
	node := &topology.ComponentNode{Name: "kubelet", Version: "v1.32.0"}
	assert.True(t, sched.shouldSkipComponent(execCtx, node))
	assert.False(t, sched.shouldSkipComponent(execCtx, &topology.ComponentNode{Name: "etcd", Version: "3.5.0"}))
	assert.False(t, sched.shouldSkipComponent(nil, node))
}

func TestPersistBatchSuccesses_OnePatchForAllCompleted(t *testing.T) {
	sched := NewScheduler(Config{})
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
		Status: confv1beta1.BKEClusterStatus{
			DeclarativeUpgrade: &confv1beta1.DeclarativeUpgradeStatus{TargetVersion: "v2"},
		},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()
	execCtx := NewExecutionContext(NewExecutionContextOptions{Cluster: bc, Client: cli})

	calico := &topology.ComponentNode{Name: "calico", Version: "v3.28.0"}
	coredns := &topology.ComponentNode{Name: "coredns", Version: "v1.12.3"}
	kubeproxy := &topology.ComponentNode{Name: "kubeproxy", Version: "v1.34.4"}

	errs, failFast := sched.persistBatchResults(context.Background(), execCtx, []componentResult{
		{name: calico.Name, node: calico},
		{name: coredns.Name, node: coredns},
		{name: kubeproxy.Name, node: kubeproxy},
	}, nil)
	require.Empty(t, errs)
	assert.False(t, failFast)

	got := &bkev1beta1.BKECluster{}
	require.NoError(t, cli.Get(context.Background(), types.NamespacedName{Namespace: "ns", Name: "c"}, got))
	require.NotNil(t, got.Status.DeclarativeUpgrade)
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("calico", "v3.28.0"))
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("coredns", "v1.12.3"))
	assert.True(t, got.Status.DeclarativeUpgrade.IsCompleted("kubeproxy", "v1.34.4"))
	assert.Len(t, got.Status.DeclarativeUpgrade.Completed, 3)
}

func TestPatchDeclarativeUpgrade_NoopWhenUnset(t *testing.T) {
	bc := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c", Namespace: "ns"},
	}
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	cli := fake.NewClientBuilder().WithScheme(scheme).WithObjects(bc).WithStatusSubresource(bc).Build()

	err := patchDeclarativeUpgrade(context.Background(), cli, bc, func(st *confv1beta1.DeclarativeUpgradeStatus) {
		st.MarkCompleted("calico", "v1", metav1.Now())
	})
	require.NoError(t, err)
	assert.Nil(t, bc.Status.DeclarativeUpgrade)
}
