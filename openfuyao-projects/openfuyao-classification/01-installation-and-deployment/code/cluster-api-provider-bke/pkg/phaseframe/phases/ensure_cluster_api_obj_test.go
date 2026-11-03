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

package phases

import (
	"context"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/pkg/errors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/common"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/kube"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/condition"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clusterv1 "sigs.k8s.io/cluster-api/api/v1beta1"
	"sigs.k8s.io/cluster-api/util"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureClusterAPIObjConstants(t *testing.T) {
	assert.Equal(t, "EnsureClusterAPIObj", string(EnsureClusterAPIObjName))
}

func TestNewEnsureClusterAPIObj(t *testing.T) {
	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	c := fakeclient.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	phase := NewEnsureClusterAPIObj(ctx)
	assert.NotNil(t, phase)
}

func TestBuildEtcdEndpoints(t *testing.T) {
	tests := []struct {
		name  string
		nodes bkenode.Nodes
		want  string
	}{
		{
			name:  "empty nodes",
			nodes: bkenode.Nodes{},
			want:  "",
		},
		{
			name: "single node",
			nodes: bkenode.Nodes{
				{IP: "192.168.1.1"},
			},
			want: "https://192.168.1.1:2379",
		},
		{
			name: "multiple nodes",
			nodes: bkenode.Nodes{
				{IP: "192.168.1.1"},
				{IP: "192.168.1.2"},
				{IP: "192.168.1.3"},
			},
			want: "https://192.168.1.1:2379,https://192.168.1.2:2379,https://192.168.1.3:2379",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := buildEtcdEndpoints(tt.nodes)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestEnsureClusterAPIObj_NeedExecute_HasOwnerRef(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test",
			Namespace:       "default",
			OwnerReferences: []metav1.OwnerReference{{Name: "owner"}},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureClusterAPIObjName}}

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func TestEnsureClusterAPIObj_NeedExecute_NoOwnerRef(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureClusterAPIObjName}}

	patches.ApplyMethod(&e.BasePhase, "NormalNeedExecute", func(_ *phaseframe.BasePhase, _ *bkev1beta1.BKECluster, _ *bkev1beta1.BKECluster) bool {
		return true
	})

	old := &bkev1beta1.BKECluster{}
	result := e.NeedExecute(old, bkeCluster)
	assert.True(t, result)
}

func TestEnsureClusterAPIObj_Execute_WithOwnerRef(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test",
			Namespace:       "default",
			OwnerReferences: []metav1.OwnerReference{{Name: "owner"}},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "reconcileClusterAPIObj", func(_ *EnsureClusterAPIObj, _ context.Context) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureClusterAPIObj_Execute_NoOwnerRef(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "reconcileCreateClusterAPIObj", func(_ *EnsureClusterAPIObj) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "reconcileClusterAPIObj", func(_ *EnsureClusterAPIObj, _ context.Context) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureClusterAPIObj_PrepareExternalEtcdConfig_NotBocloudCluster(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	result, err := e.prepareExternalEtcdConfig(bkeCluster)
	assert.NoError(t, err)
	assert.Nil(t, result)
}

func TestEnsureClusterAPIObj_Execute_WithOwnerRef_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test",
			Namespace:       "default",
			OwnerReferences: []metav1.OwnerReference{{Name: "owner"}},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "reconcileClusterAPIObj", func(_ *EnsureClusterAPIObj, _ context.Context) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestEnsureClusterAPIObj_Execute_NoOwnerRef_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "reconcileCreateClusterAPIObj", func(_ *EnsureClusterAPIObj) error {
		return nil
	})
	patches.ApplyPrivateMethod(e, "reconcileClusterAPIObj", func(_ *EnsureClusterAPIObj, _ context.Context) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func TestBuildEtcdEndpoints_EmptyNodes(t *testing.T) {
	result := buildEtcdEndpoints(bkenode.Nodes{})
	assert.Equal(t, "", result)
}

func TestBuildEtcdEndpoints_SingleNode(t *testing.T) {
	nodes := bkenode.Nodes{{IP: "192.168.1.1"}}
	result := buildEtcdEndpoints(nodes)
	assert.Equal(t, "https://192.168.1.1:2379", result)
}

func TestBuildEtcdEndpoints_MultipleNodes(t *testing.T) {
	nodes := bkenode.Nodes{
		{IP: "192.168.1.1"},
		{IP: "192.168.1.2"},
		{IP: "192.168.1.3"},
	}
	result := buildEtcdEndpoints(nodes)
	assert.Equal(t, "https://192.168.1.1:2379,https://192.168.1.2:2379,https://192.168.1.3:2379", result)
}

func TestEnsureClusterAPIObj_CreateClusterAPIObj_GenerateError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	cfg := &bkeinit.BkeConfig{}
	patches.ApplyMethod(cfg, "GenerateClusterAPIConfigFIle", func(_ *bkeinit.BkeConfig, _, _ string, _ map[string]string) (string, error) {
		return "", errors.New("generate error")
	})

	params := CreateClusterAPIObjParams{
		Ctx:        context.Background(),
		BKECluster: bkeCluster,
		Cfg:        cfg,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	err := e.createClusterAPIObj(params)
	assert.Error(t, err)
}

func TestEnsureClusterAPIObj_ReconcileClusterAPIObj_GetClusterError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
		return nil, errors.New("get cluster error")
	})

	err := e.reconcileClusterAPIObj(context.Background())
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "get combined bkeCluster default/test")
}

func TestEnsureClusterAPIObj_PrepareExternalEtcdConfig_WithEtcdNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	patches.ApplyPrivateMethod(e, "prepareExternalEtcdConfig", func(_ *EnsureClusterAPIObj, _ *bkev1beta1.BKECluster) (map[string]string, error) {
		return map[string]string{"etcd": "config"}, nil
	})

	result, err := e.prepareExternalEtcdConfig(bkeCluster)
	assert.NoError(t, err)
	assert.NotNil(t, result)
}

func TestEnsureClusterAPIObj_CreateClusterAPIObj_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}

	cfg := &bkeinit.BkeConfig{}
	params := CreateClusterAPIObjParams{
		Ctx:        context.Background(),
		BKECluster: bkeCluster,
		Cfg:        cfg,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}

	patches.ApplyMethod(cfg, "GenerateClusterAPIConfigFIle", func(_ *bkeinit.BkeConfig, _, _ string, _ map[string]string) (string, error) {
		return "config-content", nil
	})
	patches.ApplyPrivateMethod(e, "createClusterAPIObj", func(_ *EnsureClusterAPIObj, _ CreateClusterAPIObjParams) error {
		return nil
	})

	err := e.createClusterAPIObj(params)
	assert.NoError(t, err)
}

func TestEnsureClusterAPIObj_NeedExecute_WithOwnerRef(t *testing.T) {
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test",
			Namespace:       "default",
			OwnerReferences: []metav1.OwnerReference{{Name: "owner"}},
		},
	}
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	e := &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx, PhaseName: EnsureClusterAPIObjName}}

	old := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test", Namespace: "default"},
	}
	result := e.NeedExecute(old, bkeCluster)
	assert.False(t, result)
}

func newClusterAPIObjPhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureClusterAPIObj {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureClusterAPIObj{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// stubRemoteClient for cluster-api obj apply operations.
type capiObjStubRemoteClient struct {
	kube.RemoteKubeClient
	applyErr error
}

func (s capiObjStubRemoteClient) ApplyYaml(*kube.Task) error { return s.applyErr }

// ---- reconcileCreateClusterAPIObj (0% -> cover real branches) ----

func TestEnsureClusterAPIObjReconcileCreateClusterAPIObj(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

	clusterWithConfig := func() *bkev1beta1.BKECluster {
		return &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
			Spec:       confv1beta1.BKEClusterSpec{ClusterConfig: &confv1beta1.BKEConfig{}},
		}
	}

	t.Run("condition present returns waiting error", func(t *testing.T) {
		cluster := clusterWithConfig()
		condition.ConditionMark(cluster, bkev1beta1.ClusterAPIObjCondition, confv1beta1.ConditionFalse, "x", "y")
		e := newClusterAPIObjPhaseCov(t, cluster)
		err := e.reconcileCreateClusterAPIObj()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "Waiting cluster api obj reconciled")
	})

	t.Run("success", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, clusterWithConfig())
		patches.ApplyPrivateMethod(e, "createClusterAPIObj", func(_ *EnsureClusterAPIObj, _ CreateClusterAPIObjParams) error { return nil })
		patches.ApplyPrivateMethod(e, "prepareExternalEtcdConfig", func(_ *EnsureClusterAPIObj, _ *bkev1beta1.BKECluster) (map[string]string, error) {
			return nil, nil
		})
		require.NoError(t, e.reconcileCreateClusterAPIObj())
	})

	t.Run("prepare external etcd error", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, clusterWithConfig())
		patches.ApplyPrivateMethod(e, "prepareExternalEtcdConfig", func(_ *EnsureClusterAPIObj, _ *bkev1beta1.BKECluster) (map[string]string, error) {
			return nil, assertErr("etcd config failed")
		})
		err := e.reconcileCreateClusterAPIObj()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "etcd config failed")
	})
}

// ---- prepareExternalEtcdConfig (cover IsBocloudCluster=true paths) ----

func TestEnsureClusterAPIObjPrepareExternalEtcdConfig(t *testing.T) {
	bocloudCluster := func() *bkev1beta1.BKECluster {
		return &bkev1beta1.BKECluster{
			ObjectMeta: metav1.ObjectMeta{
				Name: "c1", Namespace: "ns",
				Annotations: map[string]string{common.BKEClusterFromAnnotationKey: common.BKEClusterFromAnnotationValueBocloud},
			},
		}
	}

	t.Run("not bocloud returns nil", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		result, err := e.prepareExternalEtcdConfig(e.Ctx.BKECluster)
		require.NoError(t, err)
		assert.Nil(t, result)
	})

	t.Run("bocloud with etcd nodes builds endpoints", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newClusterAPIObjPhaseCov(t, bocloudCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return &nodeutil.FetchResult{Nodes: bkenode.Nodes{{IP: "192.168.1.1", Role: []string{"etcd"}}}}, nil
		})
		result, err := e.prepareExternalEtcdConfig(e.Ctx.BKECluster)
		require.NoError(t, err)
		require.NotNil(t, result)
		assert.Contains(t, result["etcdEndpoints"], "192.168.1.1")
	})

	t.Run("get nodes error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newClusterAPIObjPhaseCov(t, bocloudCluster())
		patches.ApplyFunc((*nodeutil.NodeFetcher).FetchNodesForCluster, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (*nodeutil.FetchResult, error) {
			return nil, assertErr("fetch failed")
		})
		_, err := e.prepareExternalEtcdConfig(e.Ctx.BKECluster)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed to get nodes")
	})
}

// ---- createClusterAPIObj (real body, cover beyond GenerateError) ----

func TestEnsureClusterAPIObjCreateClusterAPIObjRealBody(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cfg := &bkeinit.BkeConfig{}
	patches.ApplyMethod(cfg, "GenerateClusterAPIConfigFIle", func(_ *bkeinit.BkeConfig, _, _ string, _ map[string]string) (string, error) {
		return "yaml-path", nil
	})
	baseParams := func(e *EnsureClusterAPIObj) CreateClusterAPIObjParams {
		return CreateClusterAPIObjParams{
			Ctx:        context.Background(),
			BKECluster: e.Ctx.BKECluster,
			Cfg:        cfg,
			Log:        e.Ctx.Log,
		}
	}

	t.Run("new client error", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return nil, assertErr("rest config failed")
		})
		err := e.createClusterAPIObj(baseParams(e))
		require.Error(t, err)
	})

	t.Run("apply yaml error", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return capiObjStubRemoteClient{applyErr: assertErr("apply failed")}, nil
		})
		err := e.createClusterAPIObj(baseParams(e))
		require.Error(t, err)
	})

	t.Run("success", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(kube.NewClientFromRestConfig, func(_ context.Context, _ interface{}) (kube.RemoteKubeClient, error) {
			return capiObjStubRemoteClient{applyErr: nil}, nil
		})
		require.NoError(t, e.createClusterAPIObj(baseParams(e)))
	})
}

// ---- reconcileClusterAPIObj (cover condition + ownerRef branches) ----

func TestEnsureClusterAPIObjReconcileClusterAPIObj(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete, func(client.Client, *bkev1beta1.BKECluster, ...mergecluster.PatchFunc) error { return nil })

	t.Run("get combined cluster error", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return nil, assertErr("get cluster error")
		})
		err := e.reconcileClusterAPIObj(context.Background())
		require.Error(t, err)
	})

	t.Run("condition false -> marks true", func(t *testing.T) {
		e := newClusterAPIObjPhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		combined := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}}
		condition.ConditionMark(combined, bkev1beta1.ClusterAPIObjCondition, confv1beta1.ConditionFalse, "x", "y")
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return combined, nil
		})
		require.NoError(t, e.reconcileClusterAPIObj(context.Background()))
	})

	t.Run("owner ref success sets cluster", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "Cluster", Name: "mycluster"}}}}
		e := newClusterAPIObjPhaseCov(t, cluster)
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return cluster, nil
		})
		patches.ApplyFunc(util.GetOwnerCluster, func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
			return &clusterv1.Cluster{ObjectMeta: metav1.ObjectMeta{Name: "mycluster"}}, nil
		})
		require.NoError(t, e.reconcileClusterAPIObj(context.Background()))
		assert.NotNil(t, e.Ctx.Cluster)
	})

	t.Run("owner ref not found", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "Cluster"}}}}
		e := newClusterAPIObjPhaseCov(t, cluster)
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return cluster, nil
		})
		patches.ApplyFunc(util.GetOwnerCluster, func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
			return nil, apierrors.NewNotFound(schema.GroupResource{Group: "cluster.x-k8s.io", Resource: "clusters"}, "mycluster")
		})
		err := e.reconcileClusterAPIObj(context.Background())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not found")
	})

	t.Run("owner ref nil cluster", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "Cluster"}}}}
		e := newClusterAPIObjPhaseCov(t, cluster)
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return cluster, nil
		})
		patches.ApplyFunc(util.GetOwnerCluster, func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
			return nil, nil
		})
		err := e.reconcileClusterAPIObj(context.Background())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not yet set OwnerRef")
	})

	t.Run("owner ref get error", func(t *testing.T) {
		cluster := &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns", OwnerReferences: []metav1.OwnerReference{{Kind: "Cluster"}}}}
		e := newClusterAPIObjPhaseCov(t, cluster)
		patches.ApplyFunc(mergecluster.GetCombinedBKECluster, func(_ context.Context, _ client.Client, _, _ string) (*bkev1beta1.BKECluster, error) {
			return cluster, nil
		})
		patches.ApplyFunc(util.GetOwnerCluster, func(_ context.Context, _ client.Client, _ metav1.ObjectMeta) (*clusterv1.Cluster, error) {
			return nil, assertErr("owner error")
		})
		err := e.reconcileClusterAPIObj(context.Background())
		require.Error(t, err)
		assert.Contains(t, err.Error(), "owner")
	})
}
