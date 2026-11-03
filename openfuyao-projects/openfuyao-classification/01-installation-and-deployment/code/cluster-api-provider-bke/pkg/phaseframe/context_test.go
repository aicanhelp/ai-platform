/******************************************************************
 * Copyright (c) 2025 Bocloud Technologies Co., Ltd.
 * cluster-api-provider-bke is licensed under Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *          http://license.coscl.org.cn/MulanPSL2
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 * See the Mulan PSL v2 for more details.
 ******************************************************************/

package phaseframe

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	"sigs.k8s.io/controller-runtime/pkg/client"

	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
)

type fakeClient struct {
	client.Client
}

func (f *fakeClient) Get(ctx context.Context, key client.ObjectKey, obj client.Object, opts ...client.GetOption) error {
	return nil
}

func (f *fakeClient) List(ctx context.Context, list client.ObjectList, opts ...client.ListOption) error {
	return nil
}

func (f *fakeClient) Update(ctx context.Context, obj client.Object, opts ...client.UpdateOption) error {
	return nil
}

func TestNewReconcilePhaseCtx(t *testing.T) {
	ctx := context.Background()
	pc := NewReconcilePhaseCtx(ctx)
	if pc == nil {
		t.Fatal("NewReconcilePhaseCtx returned nil")
	}
	if pc.Context == nil {
		t.Error("Context is nil")
	}
}

func TestPhaseContext_SetBKECluster(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	bkeCluster := &bkev1beta1.BKECluster{}
	result := pc.SetBKECluster(bkeCluster)
	if result.BKECluster != bkeCluster {
		t.Error("BKECluster not set correctly")
	}
}

func TestPhaseContext_SetCluster(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	cluster := &clusterv1.Cluster{}
	result := pc.SetCluster(cluster)
	if result.Cluster != cluster {
		t.Error("Cluster not set correctly")
	}
}

func TestPhaseContext_SetLogger(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	log := &bkev1beta1.BKELogger{}
	result := pc.SetLogger(log)
	if result.Log != log {
		t.Error("Logger not set correctly")
	}
}

func TestPhaseContext_SetScheme(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	scheme := &runtime.Scheme{}
	result := pc.SetScheme(scheme)
	if result.Scheme != scheme {
		t.Error("Scheme not set correctly")
	}
}

func TestPhaseContext_SetRestConfig(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	config := &rest.Config{}
	result := pc.SetRestConfig(config)
	if result.RestConfig != config {
		t.Error("RestConfig not set correctly")
	}
}

func TestPhaseContext_Untie(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	bkeCluster := &bkev1beta1.BKECluster{}
	scheme := &runtime.Scheme{}
	log := &bkev1beta1.BKELogger{}

	pc.SetBKECluster(bkeCluster).SetScheme(scheme).SetLogger(log)

	phaseContextFields116 := pc.Untie()
	ctx, _, bke, sch, l := phaseContextFields116.Context,
		phaseContextFields116.Client,
		phaseContextFields116.BKECluster,
		phaseContextFields116.Scheme,
		phaseContextFields116.Log
	if ctx == nil {
		t.Error("Context is nil")
	}
	if bke != bkeCluster {
		t.Error("BKECluster mismatch")
	}
	if sch != scheme {
		t.Error("Scheme mismatch")
	}
	if l != log {
		t.Error("Logger mismatch")
	}
}

func TestPhaseContext_Cancel(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	pc.Cancel()
	select {
	case <-pc.Done():
	default:
		t.Error("Context not cancelled")
	}
}

func TestPhaseContext_NodeFetcher(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	fetcher := pc.NodeFetcher()
	if fetcher == nil {
		t.Error("NodeFetcher is nil")
	}
	fetcher2 := pc.NodeFetcher()
	if fetcher != fetcher2 {
		t.Error("NodeFetcher should return same instance")
	}
}

func TestPhaseContext_SetClient(t *testing.T) {
	pc := NewReconcilePhaseCtx(context.Background())
	result := pc.SetClient(nil)
	if result != pc {
		t.Error("SetClient should return self")
	}
}

func TestPhaseContext_GetNewestBKECluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace}}, nil
	})

	result, err := pc.GetNewestBKECluster()
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestPhaseContext_GetNewestBKECluster_WithCustomCtx(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	customCtx := context.Background()
	result, err := pc.GetNewestBKECluster(customCtx)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestPhaseContext_GetNewestBKECluster_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("get error")
	})

	result, err := pc.GetNewestBKECluster()
	assert.Error(t, err)
	assert.Nil(t, result)
}

func TestPhaseContext_RefreshCtxBKECluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "new"}}, nil
	})

	err := pc.RefreshCtxBKECluster()
	assert.NoError(t, err)
	assert.Equal(t, "new", pc.BKECluster.Name)
}

func TestPhaseContext_RefreshCtxBKECluster_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("refresh error")
	})

	err := pc.RefreshCtxBKECluster()
	assert.Error(t, err)
}

func TestPhaseContext_RefreshCtxCluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	patches.ApplyFunc(util.GetOwnerCluster, func(ctx context.Context, c client.Client, obj metav1.ObjectMeta) (*clusterv1.Cluster, error) {
		return &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "owner"}}, nil
	})

	err := pc.RefreshCtxCluster()
	assert.NoError(t, err)
	assert.NotNil(t, pc.Cluster)
}

func TestPhaseContext_RefreshCtxCluster_RefreshError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("refresh error")
	})

	err := pc.RefreshCtxCluster()
	assert.Error(t, err)
}

func TestPhaseContext_RefreshCtxCluster_OwnerError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	patches.ApplyFunc(util.GetOwnerCluster, func(ctx context.Context, c client.Client, obj metav1.ObjectMeta) (*clusterv1.Cluster, error) {
		return nil, errors.New("owner error")
	})

	err := pc.RefreshCtxCluster()
	assert.Error(t, err)
}

func TestPhaseContext_RefreshCtxCluster_NilOwner(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	patches.ApplyFunc(util.GetOwnerCluster, func(ctx context.Context, c client.Client, obj metav1.ObjectMeta) (*clusterv1.Cluster, error) {
		return nil, nil
	})

	err := pc.RefreshCtxCluster()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "owner cluster is nil")
}

func TestPhaseContext_GetNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).GetNodes, func(_ *PhaseContext) (bkenode.Nodes, error) {
		return bkenode.Nodes{{IP: "192.168.1.1"}}, nil
	})

	nodes, err := pc.GetNodes()
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestPhaseContext_GetBKENodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).GetBKENodes, func(_ *PhaseContext) (bkev1beta1.BKENodes, error) {
		return bkev1beta1.BKENodes{{ObjectMeta: metav1.ObjectMeta{Name: "node1"}}}, nil
	})

	nodes, err := pc.GetBKENodes()
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestPhaseContext_HasNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).HasNodes, func(_ *PhaseContext) (bool, error) {
		return true, nil
	})

	hasNodes, err := pc.HasNodes()
	assert.NoError(t, err)
	assert.True(t, hasNodes)
}

func TestPhaseContext_HasNodes_Zero(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).HasNodes, func(_ *PhaseContext) (bool, error) {
		return false, nil
	})

	hasNodes, err := pc.HasNodes()
	assert.NoError(t, err)
	assert.False(t, hasNodes)
}

func TestPhaseContext_HasNodes_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).HasNodes, func(_ *PhaseContext) (bool, error) {
		return false, errors.New("count error")
	})

	hasNodes, err := pc.HasNodes()
	assert.Error(t, err)
	assert.False(t, hasNodes)
}

func TestPhaseContext_GetNodeStateFlag(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).GetNodeStateFlag, func(_ *PhaseContext, ip string, flag int) (bool, error) {
		return true, nil
	})

	result, err := pc.GetNodeStateFlag("192.168.1.1", 1)
	assert.NoError(t, err)
	assert.True(t, result)
}

func TestPhaseContext_SetNodeStateWithMessage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*PhaseContext).SetNodeStateWithMessage, func(_ *PhaseContext, ip string, state confv1beta1.NodeState, message string) error {
		return nil
	})

	err := pc.SetNodeStateWithMessage("192.168.1.1", confv1beta1.NodeState("Ready"), "ready")
	assert.NoError(t, err)
}

func TestPhaseContext_SetNodeStateMessage(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "test"}},
	}

	patches.ApplyFunc((*PhaseContext).SetNodeStateMessage, func(_ *PhaseContext, ip string, message string) error {
		return nil
	})

	err := pc.SetNodeStateMessage("192.168.1.1", "test message")
	assert.NoError(t, err)
}

func TestPhaseContext_SetNodeStateMessage_Error(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "test"}},
	}

	patches.ApplyFunc((*PhaseContext).SetNodeStateMessage, func(_ *PhaseContext, ip string, message string) error {
		return errors.New("get error")
	})

	err := pc.SetNodeStateMessage("192.168.1.1", "test message")
	assert.Error(t, err)
}

func TestPhaseContext_RefreshCtxBKECluster_WithCustomCtx(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	customCtx := context.Background()
	err := pc.RefreshCtxBKECluster(customCtx)
	assert.NoError(t, err)
}

func TestPhaseContext_RefreshCtxCluster_WithCustomCtx(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context: context.Background(),
		Client:  &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
		},
	}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(ctx context.Context, c client.Client, namespace, name string) (*bkev1beta1.BKECluster, error) {
		return &bkev1beta1.BKECluster{}, nil
	})

	patches.ApplyFunc(util.GetOwnerCluster, func(ctx context.Context, c client.Client, obj metav1.ObjectMeta) (*clusterv1.Cluster, error) {
		return &clusterv1.Cluster{}, nil
	})

	customCtx := context.Background()
	err := pc.RefreshCtxCluster(customCtx)
	assert.NoError(t, err)
}

func TestPhaseContext_GetNodes_Real(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}

	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster, func(_ *nodeutil.NodeFetcher, ctx context.Context, bkeCluster *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
		return bkenode.Nodes{{IP: "192.168.1.1"}}, nil
	})

	nodes, err := pc.GetNodes()
	assert.NoError(t, err)
	assert.Len(t, nodes, 1)
}

func TestPhaseContext_GetBKENodes_Call(t *testing.T) {
	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}
	_, _ = pc.GetBKENodes()
}

func TestPhaseContext_HasNodes_Call(t *testing.T) {
	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}
	_, _ = pc.HasNodes()
}

func TestPhaseContext_GetNodeStateFlag_Call(t *testing.T) {
	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}
	_, _ = pc.GetNodeStateFlag("192.168.1.1", 1)
}

func TestPhaseContext_SetNodeStateWithMessage_Call(t *testing.T) {
	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{},
	}
	_ = pc.SetNodeStateWithMessage("192.168.1.1", confv1beta1.NodeState("Ready"), "ready")
}

func TestPhaseContext_SetNodeStateMessage_Call(t *testing.T) {
	pc := &PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Namespace: "default", Name: "test"}},
	}
	_ = pc.SetNodeStateMessage("192.168.1.1", "test message")
}
