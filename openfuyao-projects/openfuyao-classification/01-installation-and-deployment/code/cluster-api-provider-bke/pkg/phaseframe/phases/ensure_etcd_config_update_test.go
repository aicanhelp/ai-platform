/******************************************************************
 * Copyright (c) 2026 ICBC Co., Ltd.
 * installer is licensed under Mulan PSL v2.
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
	"errors"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/statusmanage"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/annotation"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureEtcdConfigUpdateConstants(t *testing.T) {
	assert.Equal(t, "EnsureEtcdConfigUpdate", string(EnsureEtcdConfigUpdateName))
}

func TestNewEnsureEtcdConfigUpdate(t *testing.T) {
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
	phase := NewEnsureEtcdConfigUpdate(ctx)
	assert.NotNil(t, phase)
	assert.Equal(t, EnsureEtcdConfigUpdateName, phase.Name())
}

// ---- helpers ----

// newEtcdConfigUpdatePhaseCov builds an EnsureEtcdConfigUpdate wired to a fake client.
func newEtcdConfigUpdatePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureEtcdConfigUpdate {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	c := ctrlfake.NewClientBuilder().WithScheme(scheme).WithObjects(bkeCluster).Build()
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Client:     c,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureEtcdConfigUpdate{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// etcdConfigUpdateCluster builds a BKECluster with matching spec/status versions and
// optional etcd mutable config (ExtraArgs) so that the desired etcd hash is non-trivial.
func etcdConfigUpdateCluster(etcdExtraArgs map[string]string) *bkev1beta1.BKECluster {
	return &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					EtcdVersion:       "v3.5.6",
					ControlPlane: confv1beta1.ControlPlane{
						Etcd: &confv1beta1.Etcd{
							ControlPlaneComponent: confv1beta1.ControlPlaneComponent{
								ExtraArgs: etcdExtraArgs,
							},
						},
					},
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			KubernetesVersion: "v1.28.0",
			EtcdVersion:       "v3.5.6",
		},
	}
}

// etcdConfigUpdateBKENodes converts legacy nodes into BKENodes for fetch mocks.
func etcdConfigUpdateBKENodes(nodes ...confv1beta1.Node) bkev1beta1.BKENodes {
	var result bkev1beta1.BKENodes
	for _, n := range nodes {
		result = append(result, confv1beta1.BKENode{Spec: confv1beta1.FromNode(n)})
	}
	return result
}

// patchFetchBKENodesForConfigUpdate mocks fetchBKENodesIfCPInitialized used by NeedExecute.
func patchFetchBKENodesForConfigUpdate(patches *gomonkey.Patches, nodes bkev1beta1.BKENodes, ok bool) {
	patches.ApplyFunc(fetchBKENodesIfCPInitialized, func(_ *phaseframe.PhaseContext, _ *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, bool) {
		return nodes, ok
	})
}

// patchNodeFetcherWrapper mocks NodeFetcher.GetBKENodesWrapperForCluster used by reconcile.
func patchNodeFetcherWrapper(patches *gomonkey.Patches, nodes bkev1beta1.BKENodes, err error) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapperForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkev1beta1.BKENodes, error) {
			return nodes, err
		})
}

// patchUpgradeCmdSuccess mocks command.Upgrade New/Wait to succeed. The New mock also
// initializes Command so that LogCommandFailed(*upgrade.Command, ...) never panics.
func patchUpgradeCmdSuccess(patches *gomonkey.Patches, captured *[]*command.Upgrade) {
	patches.ApplyMethod(&command.Upgrade{}, "New", func(u *command.Upgrade) error {
		if captured != nil {
			cp := *u
			*captured = append(*captured, &cp)
		}
		u.Command = &agentv1beta1.Command{}
		return nil
	})
	patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
		return nil, []string{"10.0.0.1"}, nil
	})
}

// patchSyncStatusApplyPatches mocks mergecluster.SyncStatusUntilComplete so patch funcs
// are applied to the in-memory cluster object, mirroring real behavior.
func patchSyncStatusApplyPatches(patches *gomonkey.Patches) {
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, bc *bkev1beta1.BKECluster, patchs ...mergecluster.PatchFunc) error {
			for _, p := range patchs {
				p(bc)
			}
			return nil
		})
}

// ---- UpgradeEtcdConfigNodesParams structure ----

func TestUpgradeEtcdConfigNodesParams_Structure(t *testing.T) {
	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           &fakeClient{},
		BKECluster:       &bkev1beta1.BKECluster{},
		NeedUpgradeNodes: bkenode.Nodes{{IP: "192.168.1.1"}},
		Log:              createTestLogger(),
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
	assert.Len(t, params.NeedUpgradeNodes, 1)
}

// ---- Execute ----

// TestEnsureEtcdConfigUpdate_Reconcile_SyncStatusError 覆盖全部 etcd 节点成功应用
// 新 manifest 后、回写成功基线 hash（Status.EtcdConfigHash）失败的错误分支。
// 旧代码曾在 Execute 中 patch deployAction 注解并在该步失败即返回，annotation 机制已移除，
// 故改为在 reconcile 的 SyncStatusUntilComplete 失败路径上覆盖。
func TestEnsureEtcdConfigUpdate_Reconcile_SyncStatusError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(map[string]string{"listen-metrics-urls": "http://0.0.0.0:2381"})
	cluster.Status.EtcdConfigHash = "stale-old-hash"
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), nil)

	patchUpgradeCmdSuccess(patches, nil)

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return errors.New("sync boom")
		})

	_, err := e.reconcileEtcdConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sync boom")
}

// ---- reconcileEtcdConfigUpdate ----

func TestEnsureEtcdConfigUpdate_Reconcile_GetBKENodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, nil, errors.New("fetch boom"))

	result, err := e.reconcileEtcdConfigUpdate()
	require.Error(t, err)
	assert.True(t, result.Requeue)
	assert.Contains(t, err.Error(), "failed to get BKENodes")
}

func TestEnsureEtcdConfigUpdate_Reconcile_NoEtcdNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, bkev1beta1.BKENodes{}, nil)

	result, err := e.reconcileEtcdConfigUpdate()
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	// hash must not be written back when nothing was updated
	assert.Empty(t, cluster.Status.EtcdConfigHash)
}

func TestEnsureEtcdConfigUpdate_Reconcile_HappyPath(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(map[string]string{"listen-metrics-urls": "http://0.0.0.0:2381"})
	cluster.Status.EtcdConfigHash = "stale-old-hash"
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1"), etcdNode("10.0.0.2", "node2")), nil)

	var captured []*command.Upgrade
	patchUpgradeCmdSuccess(patches, &captured)
	patchSyncStatusApplyPatches(patches)

	result, err := e.reconcileEtcdConfigUpdate()
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	// one config-update command per etcd node with the right phase
	require.Len(t, captured, 2)
	assert.Equal(t, bkev1beta1.UpgradeEtcdConfig, captured[0].Phase)
	assert.Equal(t, bkev1beta1.UpgradeEtcdConfig, captured[1].Phase)
	assert.NotNil(t, captured[0].Node)
	assert.Equal(t, "10.0.0.1", captured[0].Node.IP)

	// success baseline hash written back
	desired, hashErr := phaseutil.CalculateEtcdConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, hashErr)
	assert.NotEmpty(t, desired)
	assert.Equal(t, desired, cluster.Status.EtcdConfigHash)
}

func TestEnsureEtcdConfigUpdate_Reconcile_UpdateConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(map[string]string{"listen-metrics-urls": "http://0.0.0.0:2381"})
	cluster.Status.EtcdConfigHash = "old-good"
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), nil)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})
	patchSyncStatusApplyPatches(patches)

	_, err := e.reconcileEtcdConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create config update command failed")
	assert.Equal(t, "old-good", cluster.Status.EtcdConfigHash)
	assert.Equal(t, "1", cluster.Annotations[annotation.EtcdConfigUpdateFailCountAnnotationKey])
}

func TestEnsureEtcdConfigUpdate_Reconcile_UpdateConfigErrorThorough(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	cluster := etcdConfigUpdateCluster(map[string]string{"listen-metrics-urls": "http://0.0.0.0:2381"})
	cluster.Status.EtcdConfigHash = "old-good"
	cluster.Annotations = map[string]string{
		annotation.EtcdConfigUpdateFailCountAnnotationKey: "2",
	}
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), nil)
	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})
	patchSyncStatusApplyPatches(patches)

	_, err := e.reconcileEtcdConfigUpdate()
	require.Error(t, err)

	desired, hashErr := phaseutil.CalculateEtcdConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, hashErr)
	assert.Equal(t, desired, cluster.Status.EtcdConfigHash)
	assert.Equal(t, "3", cluster.Annotations[annotation.EtcdConfigUpdateFailCountAnnotationKey])
}

func TestEnsureEtcdConfigUpdate_Reconcile_PersistHashError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), nil)
	patchUpgradeCmdSuccess(patches, nil)
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return errors.New("persist boom")
		})

	_, err := e.reconcileEtcdConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "persist boom")
}

// ---- updateEtcdConfig ----

func TestEnsureEtcdConfigUpdate_UpdateEtcdConfig_EmptyNodes(t *testing.T) {
	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{},
		Log:              e.Ctx.Log,
	}
	require.NoError(t, e.updateEtcdConfig(params))
}

func TestEnsureEtcdConfigUpdate_UpdateEtcdConfig_NewError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})

	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{etcdNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateEtcdConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create config update command failed")
	assert.Contains(t, err.Error(), "new boom")
}

func TestEnsureEtcdConfigUpdate_UpdateEtcdConfig_WaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(u *command.Upgrade) error {
		u.Command = &agentv1beta1.Command{}
		return nil
	})
	patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
		return errors.New("wait boom"), nil, nil
	})

	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{etcdNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateEtcdConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "wait config update command failed")
	assert.Contains(t, err.Error(), "wait boom")
}

func TestEnsureEtcdConfigUpdate_UpdateEtcdConfig_FailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(u *command.Upgrade) error {
		u.Command = &agentv1beta1.Command{}
		return nil
	})
	patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
		return nil, []string{}, []string{"10.0.0.1"}
	})
	patches.ApplyFunc(phaseutil.LogCommandFailed,
		func(_ agentv1beta1.Command, _ []string, _ *bkev1beta1.BKELogger, _ string) (map[string][]string, error) {
			return map[string][]string{"10.0.0.1": {"boom"}}, nil
		})
	patches.ApplyFunc(phaseutil.MarkNodeStatusByCommandErrs,
		func(_ context.Context, _ client.Client, _ *bkev1beta1.BKECluster, _ map[string][]string) {})

	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{etcdNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateEtcdConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "config update node")
}

func TestEnsureEtcdConfigUpdate_UpdateEtcdConfig_CommandWiring(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	var captured []*command.Upgrade
	patchUpgradeCmdSuccess(patches, &captured)

	params := UpgradeEtcdConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{etcdNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	require.NoError(t, e.updateEtcdConfig(params))

	require.Len(t, captured, 1)
	cmd := captured[0]
	assert.Equal(t, bkev1beta1.UpgradeEtcdConfig, cmd.Phase)
	assert.Equal(t, "ns", cmd.NameSpace)
	assert.Equal(t, "c1", cmd.ClusterName)
	assert.Equal(t, "c1", cmd.BKEConfig)
	require.NotNil(t, cmd.Node)
	assert.Equal(t, "10.0.0.1", cmd.Node.IP)
	assert.False(t, cmd.BackUpEtcd)
}

// ---- NeedExecute ----

func TestEnsureEtcdConfigUpdate_NeedExecute_DefaultConditionsBlock(t *testing.T) {
	t.Run("paused cluster returns false", func(t *testing.T) {
		cluster := etcdConfigUpdateCluster(nil)
		cluster.Spec.Pause = true
		e := newEtcdConfigUpdatePhaseCov(t, cluster)
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, cluster))
	})
	t.Run("dry run cluster returns false", func(t *testing.T) {
		cluster := etcdConfigUpdateCluster(nil)
		cluster.Spec.DryRun = true
		e := newEtcdConfigUpdatePhaseCov(t, cluster)
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, cluster))
	})
}

func TestEnsureEtcdConfigUpdate_NeedExecute_FetchBKENodesNotOk(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches, nil, false)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureEtcdConfigUpdate_NeedExecute_EtcdVersionNotSynced(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := etcdConfigUpdateCluster(nil)
	newCluster := etcdConfigUpdateCluster(nil)
	newCluster.Spec.ClusterConfig.Cluster.EtcdVersion = "v3.5.12" // spec 与 status 未对齐
	e := newEtcdConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

func TestEnsureEtcdConfigUpdate_NeedExecute_KubernetesVersionNotSynced(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := etcdConfigUpdateCluster(nil)
	newCluster := etcdConfigUpdateCluster(nil)
	newCluster.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.29.0" // spec 与 status 未对齐
	e := newEtcdConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

// TestEnsureEtcdConfigUpdate_NeedExecute_EtcdVersionInProgress 是旧逻辑并发 bug 的
// 回归用例：Etcd 升级进行中（Spec 已切到新版本、但 Status 仍为旧版本）时，
// 即便 old/new 的 Spec 完全一致，也不应放行纯配置更新。
func TestEnsureEtcdConfigUpdate_NeedExecute_EtcdVersionInProgress(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := etcdConfigUpdateCluster(nil)
	old.Spec.ClusterConfig.Cluster.EtcdVersion = "v3.5.12"
	newCluster := etcdConfigUpdateCluster(nil)
	newCluster.Spec.ClusterConfig.Cluster.EtcdVersion = "v3.5.12"
	// Status.EtcdVersion 仍为 v3.5.6（升级未完成）
	e := newEtcdConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

func TestEnsureEtcdConfigUpdate_NeedExecute_NoEtcdNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(nil)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	// only worker nodes -> no etcd node to update
	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(confv1beta1.Node{IP: "10.0.0.9", Hostname: "worker1", Role: []string{"worker"}}), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureEtcdConfigUpdate_NeedExecute_HashUnchanged(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(map[string]string{"extra": "arg"})
	desired, err := phaseutil.CalculateEtcdConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, err)
	cluster.Status.EtcdConfigHash = desired // already applied
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureEtcdConfigUpdate_NeedExecute_EmptyBaselineSkips(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 创建/纳管阶段：Status.EtcdConfigHash 尚未建立成功基线，
	// 即使 Spec 配置了 etcd 可变字段也不触发（避免创建阶段误刷新静态 Pod）。
	cluster := etcdConfigUpdateCluster(map[string]string{"extra": "arg"})
	assert.Empty(t, cluster.Status.EtcdConfigHash)
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestNeedExecute_EtcdConfigUpdate_HashChangedTriggers(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := etcdConfigUpdateCluster(map[string]string{"extra": "arg"})
	cluster.Status.EtcdConfigHash = "old-successful-baseline"
	e := newEtcdConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(etcdNode("10.0.0.1", "node1")), true)

	assert.True(t, e.NeedExecute(nil, cluster))
	assert.Equal(t, bkev1beta1.PhaseWaiting, e.GetStatus())
}
