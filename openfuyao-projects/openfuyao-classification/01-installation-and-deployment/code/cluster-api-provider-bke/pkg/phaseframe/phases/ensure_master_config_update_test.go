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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	ctrlfake "sigs.k8s.io/controller-runtime/pkg/client/fake"
	fakeclient "sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestEnsureMasterConfigUpdateConstants(t *testing.T) {
	assert.Equal(t, "EnsureMasterConfigUpdate", string(EnsureMasterConfigUpdateName))
}

func TestNewEnsureMasterConfigUpdate(t *testing.T) {
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
	phase := NewEnsureMasterConfigUpdate(ctx)
	assert.NotNil(t, phase)
	assert.Equal(t, EnsureMasterConfigUpdateName, phase.Name())
}

// ---- helpers ----

// configUpdateMasterNode builds a legacy master node.
func configUpdateMasterNode(ip, host string) confv1beta1.Node {
	return confv1beta1.Node{IP: ip, Hostname: host, Role: []string{"master"}}
}

// masterConfigUpdateCluster builds a BKECluster with matching spec/status versions.
// When withComponents is true, APIServer / ControllerManager / Scheduler mutable
// configs are populated so the desired control-plane hash is non-empty.
func masterConfigUpdateCluster(withComponents bool) *bkev1beta1.BKECluster {
	cluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					KubernetesVersion: "v1.28.0",
					EtcdVersion:       "v3.5.6",
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			KubernetesVersion: "v1.28.0",
			EtcdVersion:       "v3.5.6",
		},
	}
	if withComponents {
		cluster.Spec.ClusterConfig.Cluster.APIServer = &confv1beta1.APIServer{
			CertSANs: []string{"api.example.com"},
		}
		cluster.Spec.ClusterConfig.Cluster.ControllerManager = &confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{"leader-elect": "true"},
		}
		cluster.Spec.ClusterConfig.Cluster.Scheduler = &confv1beta1.ControlPlaneComponent{
			ExtraArgs: map[string]string{"leader-elect": "true"},
		}
	}
	return cluster
}

// newMasterConfigUpdatePhaseCov builds an EnsureMasterConfigUpdate wired to a fake client.
func newMasterConfigUpdatePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureMasterConfigUpdate {
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
	return &EnsureMasterConfigUpdate{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

// ---- pure version-drift helpers ----

func TestKubernetesVersionNotSynced(t *testing.T) {
	t.Run("nil cluster returns true", func(t *testing.T) {
		assert.True(t, kubernetesVersionNotSynced(nil))
	})

	t.Run("nil clusterconfig returns true", func(t *testing.T) {
		assert.True(t, kubernetesVersionNotSynced(&bkev1beta1.BKECluster{}))
	})

	t.Run("spec equals status returns false", func(t *testing.T) {
		c := masterConfigUpdateCluster(false)
		assert.False(t, kubernetesVersionNotSynced(c))
	})

	t.Run("spec differs status returns true", func(t *testing.T) {
		c := masterConfigUpdateCluster(false)
		c.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.29.0"
		assert.True(t, kubernetesVersionNotSynced(c))
	})
}

func TestEtcdVersionNotSynced(t *testing.T) {
	t.Run("nil cluster returns true", func(t *testing.T) {
		assert.True(t, etcdVersionNotSynced(nil))
	})

	t.Run("nil clusterconfig returns true", func(t *testing.T) {
		assert.True(t, etcdVersionNotSynced(&bkev1beta1.BKECluster{}))
	})

	t.Run("spec equals status returns false", func(t *testing.T) {
		c := masterConfigUpdateCluster(false)
		assert.False(t, etcdVersionNotSynced(c))
	})

	t.Run("spec differs status returns true", func(t *testing.T) {
		c := masterConfigUpdateCluster(false)
		c.Spec.ClusterConfig.Cluster.EtcdVersion = "v3.5.12"
		assert.True(t, etcdVersionNotSynced(c))
	})
}

// ---- UpgradeMasterConfigNodesParams structure ----

func TestUpgradeMasterConfigNodesParams_Structure(t *testing.T) {
	params := UpgradeMasterConfigNodesParams{
		Ctx:                       context.Background(),
		Client:                    &fakeClient{},
		BKECluster:                &bkev1beta1.BKECluster{},
		NeedUpgradeNodes:          bkenode.Nodes{{IP: "192.168.1.1"}},
		Log:                       createTestLogger(),
		ControlPlaneConfigOldHash: confv1beta1.ComponentConfigHash{APIServer: "old"},
		ControlPlaneConfigNewHash: confv1beta1.ComponentConfigHash{APIServer: "new"},
	}
	assert.NotNil(t, params.Ctx)
	assert.NotNil(t, params.Client)
	assert.NotNil(t, params.BKECluster)
	assert.NotNil(t, params.Log)
	assert.Equal(t, "old", params.ControlPlaneConfigOldHash.APIServer)
	assert.Equal(t, "new", params.ControlPlaneConfigNewHash.APIServer)
}

// ---- Execute ----

// TestEnsureMasterConfigUpdate_Reconcile_SyncStatusError 覆盖全部 Master 节点成功应用
// 新 manifest 后、回写成功基线 hash（Status.ControlPlaneConfigHash）失败的错误分支。
// 旧代码曾在 Execute 中 patch deployAction 注解并在该步失败即返回，annotation 机制已移除，
// 故改为在 reconcile 的 SyncStatusUntilComplete 失败路径上覆盖。
func TestEnsureMasterConfigUpdate_Reconcile_SyncStatusError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	cluster.Status.ControlPlaneConfigHash = confv1beta1.ComponentConfigHash{APIServer: "stale"}
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), nil)

	patches.ApplyPrivateMethod(e, "updateMasterConfig",
		func(_ *EnsureMasterConfigUpdate, _ UpgradeMasterConfigNodesParams) error {
			return nil
		})

	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return errors.New("sync boom")
		})

	_, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "sync boom")
}

// ---- reconcileMasterConfigUpdate ----

func TestEnsureMasterConfigUpdate_Reconcile_GetBKENodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, nil, errors.New("fetch boom"))

	result, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)
	assert.True(t, result.Requeue)
	assert.Contains(t, err.Error(), "failed to get BKENodes")
}

func TestEnsureMasterConfigUpdate_Reconcile_NoMasterNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches, bkev1beta1.BKENodes{}, nil)

	result, err := e.reconcileMasterConfigUpdate()
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
	// hash must not be written back when nothing was updated
	assert.Equal(t, confv1beta1.ComponentConfigHash{}, cluster.Status.ControlPlaneConfigHash)
}

func TestEnsureMasterConfigUpdate_Reconcile_HappyPath(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	oldBaseline := confv1beta1.ComponentConfigHash{APIServer: "stale"}
	cluster.Status.ControlPlaneConfigHash = oldBaseline
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1"), configUpdateMasterNode("10.0.0.2", "node2")), nil)

	var capturedParams UpgradeMasterConfigNodesParams
	patches.ApplyPrivateMethod(e, "updateMasterConfig",
		func(_ *EnsureMasterConfigUpdate, p UpgradeMasterConfigNodesParams) error {
			capturedParams = p
			return nil
		})
	patchSyncStatusApplyPatches(patches)

	result, err := e.reconcileMasterConfigUpdate()
	require.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)

	desired, hashErr := phaseutil.CalculateComponentConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, hashErr)

	// old/new hashes passed to updateMasterConfig
	assert.Equal(t, oldBaseline, capturedParams.ControlPlaneConfigOldHash)
	assert.Equal(t, desired, capturedParams.ControlPlaneConfigNewHash)
	require.Len(t, capturedParams.NeedUpgradeNodes, 2)

	// success baseline hash written back
	assert.Equal(t, desired, cluster.Status.ControlPlaneConfigHash)
}

func TestEnsureMasterConfigUpdate_Reconcile_CalculateHashError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), nil)
	patches.ApplyFunc(phaseutil.CalculateComponentConfigHash,
		func(_ *confv1beta1.Cluster) (confv1beta1.ComponentConfigHash, error) {
			return confv1beta1.ComponentConfigHash{}, errors.New("hash boom")
		})

	_, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "hash boom")
}

func TestEnsureMasterConfigUpdate_Reconcile_UpdateConfigError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	oldBaseline := confv1beta1.ComponentConfigHash{APIServer: "old-good"}
	cluster := masterConfigUpdateCluster(true)
	cluster.Status.ControlPlaneConfigHash = oldBaseline
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), nil)
	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})
	patchSyncStatusApplyPatches(patches)

	_, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create config update command failed")
	// first failures keep success baseline so same Spec can auto-retry
	assert.Equal(t, oldBaseline, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "1", cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey])
}

func TestEnsureMasterConfigUpdate_Reconcile_UpdateConfigErrorThorough(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	origAllowed := statusmanage.ReconcileAllowedFailedCount
	t.Cleanup(func() { statusmanage.ReconcileAllowedFailedCount = origAllowed })
	statusmanage.ReconcileAllowedFailedCount = 3

	cluster := masterConfigUpdateCluster(true)
	cluster.Status.ControlPlaneConfigHash = confv1beta1.ComponentConfigHash{APIServer: "old-good"}
	cluster.Annotations = map[string]string{
		annotation.ControlPlaneConfigUpdateFailCountAnnotationKey: "2",
	}
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), nil)
	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})
	patchSyncStatusApplyPatches(patches)

	_, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)

	desired, hashErr := phaseutil.CalculateComponentConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, hashErr)
	assert.Equal(t, desired, cluster.Status.ControlPlaneConfigHash)
	assert.Equal(t, "3", cluster.Annotations[annotation.ControlPlaneConfigUpdateFailCountAnnotationKey])
}

func TestEnsureMasterConfigUpdate_Reconcile_PersistHashError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchNodeFetcherWrapper(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), nil)
	patchUpgradeCmdSuccess(patches, nil)
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return errors.New("persist boom")
		})

	_, err := e.reconcileMasterConfigUpdate()
	require.Error(t, err)
	assert.Contains(t, err.Error(), "persist boom")
}

// ---- updateMasterConfig ----

func TestEnsureMasterConfigUpdate_UpdateMasterConfig_EmptyNodes(t *testing.T) {
	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	params := UpgradeMasterConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{},
		Log:              e.Ctx.Log,
	}
	require.NoError(t, e.updateMasterConfig(params))
}

func TestEnsureMasterConfigUpdate_UpdateMasterConfig_NewError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(_ *command.Upgrade) error {
		return errors.New("new boom")
	})

	params := UpgradeMasterConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{configUpdateMasterNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateMasterConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "create config update command failed")
	assert.Contains(t, err.Error(), "new boom")
}

func TestEnsureMasterConfigUpdate_UpdateMasterConfig_WaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patches.ApplyMethod(&command.Upgrade{}, "New", func(u *command.Upgrade) error {
		u.Command = &agentv1beta1.Command{}
		return nil
	})
	patches.ApplyMethod(&command.Upgrade{}, "Wait", func(_ *command.Upgrade) (error, []string, []string) {
		return errors.New("wait boom"), nil, nil
	})

	params := UpgradeMasterConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{configUpdateMasterNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateMasterConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "wait config update command failed")
	assert.Contains(t, err.Error(), "wait boom")
}

func TestEnsureMasterConfigUpdate_UpdateMasterConfig_FailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

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

	params := UpgradeMasterConfigNodesParams{
		Ctx:              context.Background(),
		Client:           e.Ctx.Client,
		BKECluster:       cluster,
		NeedUpgradeNodes: bkenode.Nodes{configUpdateMasterNode("10.0.0.1", "node1")},
		Log:              e.Ctx.Log,
	}
	err := e.updateMasterConfig(params)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "config update node")
}

func TestEnsureMasterConfigUpdate_UpdateMasterConfig_CommandWiring(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	oldHash := confv1beta1.ComponentConfigHash{APIServer: "old-hash"}
	desired, err := phaseutil.CalculateComponentConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, err)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	var captured []*command.Upgrade
	patchUpgradeCmdSuccess(patches, &captured)

	params := UpgradeMasterConfigNodesParams{
		Ctx:                       context.Background(),
		Client:                    e.Ctx.Client,
		BKECluster:                cluster,
		NeedUpgradeNodes:          bkenode.Nodes{configUpdateMasterNode("10.0.0.1", "node1")},
		Log:                       e.Ctx.Log,
		ControlPlaneConfigOldHash: oldHash,
		ControlPlaneConfigNewHash: desired,
	}
	require.NoError(t, e.updateMasterConfig(params))

	require.Len(t, captured, 1)
	cmd := captured[0]
	assert.Equal(t, bkev1beta1.UpgradeControlPlaneConfig, cmd.Phase)
	assert.Equal(t, "ns", cmd.NameSpace)
	assert.Equal(t, "c1", cmd.ClusterName)
	assert.Equal(t, "c1", cmd.BKEConfig)
	assert.Equal(t, oldHash, cmd.ControlPlaneConfigOldHash)
	assert.Equal(t, desired, cmd.ControlPlaneConfigNewHash)
	require.NotNil(t, cmd.Node)
	assert.Equal(t, "10.0.0.1", cmd.Node.IP)
}

// ---- NeedExecute ----

func TestEnsureMasterConfigUpdate_NeedExecute_DefaultConditionsBlock(t *testing.T) {
	t.Run("paused cluster returns false", func(t *testing.T) {
		cluster := masterConfigUpdateCluster(true)
		cluster.Spec.Pause = true
		e := newMasterConfigUpdatePhaseCov(t, cluster)
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, cluster))
	})

	t.Run("dry run cluster returns false", func(t *testing.T) {
		cluster := masterConfigUpdateCluster(true)
		cluster.Spec.DryRun = true
		e := newMasterConfigUpdatePhaseCov(t, cluster)
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, cluster))
	})
}

func TestEnsureMasterConfigUpdate_NeedExecute_FetchBKENodesNotOk(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches, nil, false)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_EtcdVersionNotSynced(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := masterConfigUpdateCluster(true)
	newCluster := masterConfigUpdateCluster(true)
	newCluster.Spec.ClusterConfig.Cluster.EtcdVersion = "v3.5.12" // spec 与 status 未对齐
	e := newMasterConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_KubernetesVersionNotSynced(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := masterConfigUpdateCluster(true)
	newCluster := masterConfigUpdateCluster(true)
	newCluster.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.29.0" // spec 与 status 未对齐
	e := newMasterConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

// TestEnsureMasterConfigUpdate_NeedExecute_KubernetesVersionInProgress 是旧逻辑并发
// bug 的回归用例：版本升级进行中（Spec 已切到新版本、但 Status 仍为旧版本）时，
// 即便 old/new 的 Spec 完全一致，也不应放行纯配置更新，避免与升级阶段并发。
func TestEnsureMasterConfigUpdate_NeedExecute_KubernetesVersionInProgress(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	old := masterConfigUpdateCluster(true)
	old.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.29.0"
	newCluster := masterConfigUpdateCluster(true)
	newCluster.Spec.ClusterConfig.Cluster.KubernetesVersion = "v1.29.0"
	// Status.KubernetesVersion 仍为 v1.28.0（升级未完成）
	e := newMasterConfigUpdatePhaseCov(t, newCluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(old, newCluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_NoMasterNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	// only worker nodes -> no master node to update
	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(confv1beta1.Node{IP: "10.0.0.9", Hostname: "worker1", Role: []string{"worker"}}), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_DesiredHashEmpty(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// no component configs -> desired hash struct is empty -> skip to avoid pointless restarts
	cluster := masterConfigUpdateCluster(false)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_HashUnchanged(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	desired, err := phaseutil.CalculateComponentConfigHash(&cluster.Spec.ClusterConfig.Cluster)
	require.NoError(t, err)
	cluster.Status.ControlPlaneConfigHash = desired // already applied
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_EmptyBaselineSkips(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	// 创建/纳管阶段：Status.ControlPlaneConfigHash 尚未建立成功基线，
	// 即使 Spec 配置了控制面组件可变字段也不触发（避免创建阶段误刷新静态 Pod）。
	cluster := masterConfigUpdateCluster(true)
	assert.Equal(t, confv1beta1.ComponentConfigHash{}, cluster.Status.ControlPlaneConfigHash)
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.False(t, e.NeedExecute(nil, cluster))
}

func TestEnsureMasterConfigUpdate_NeedExecute_HashChangedTriggers(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	cluster := masterConfigUpdateCluster(true)
	cluster.Status.ControlPlaneConfigHash = confv1beta1.ComponentConfigHash{APIServer: "old-baseline"}
	e := newMasterConfigUpdatePhaseCov(t, cluster)

	patchFetchBKENodesForConfigUpdate(patches,
		etcdConfigUpdateBKENodes(configUpdateMasterNode("10.0.0.1", "node1")), true)

	assert.True(t, e.NeedExecute(nil, cluster))
	assert.Equal(t, bkev1beta1.PhaseWaiting, e.GetStatus())
}
