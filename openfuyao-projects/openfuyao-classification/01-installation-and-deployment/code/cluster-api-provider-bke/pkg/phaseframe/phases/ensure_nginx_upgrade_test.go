/******************************************************************
 * Copyright (c) 2026 China Mobile (Suzhou) Software Technology Co., Ltd
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
	"reflect"
	"testing"

	"github.com/agiledragon/gomonkey/v2"
	"github.com/stretchr/testify/assert"
	agentv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkeagent/v1beta1"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/command"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/mergecluster"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func createTestEnsureNginxUpgrade() *EnsureNginxUpgrade {
	logger := createTestLogger()
	bkeCluster := &bkev1beta1.BKECluster{
		ObjectMeta: metav1.ObjectMeta{Name: "test-cluster", Namespace: "default"},
		Spec: confv1beta1.BKEClusterSpec{
			ClusterConfig: &confv1beta1.BKEConfig{
				Cluster: confv1beta1.Cluster{
					LoadBalanceType:   "nginx",
					NginxProxyVersion: "1.28.0",
				},
			},
		},
		Status: confv1beta1.BKEClusterStatus{
			NginxProxyVersion: "1.27.0",
		},
	}

	scheme := runtime.NewScheme()
	_ = bkev1beta1.AddToScheme(scheme)

	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		Client:     &fakeClient{},
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        logger,
	}

	return &EnsureNginxUpgrade{BasePhase: phaseframe.NewBasePhase(ctx, EnsureNginxUpgradeName)}
}

func TestEnsureNginxUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureNginxUpgrade", string(EnsureNginxUpgradeName))
}

func TestNewEnsureNginxUpgrade(t *testing.T) {
	base := createTestEnsureNginxUpgrade()
	phase := NewEnsureNginxUpgrade(base.Ctx)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureNginxUpgrade{}, phase)
}

func TestEnsureNginxUpgrade_resolveNginxUpgradeVersion(t *testing.T) {
	e := createTestEnsureNginxUpgrade()

	// VersionContext target takes precedence over spec.
	e.Ctx.VersionContext = upgrade.NewVersionContext()
	e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.29.0")
	assert.Equal(t, "1.29.0", e.resolveNginxUpgradeVersion())

	// VersionContext present but no nginx target: fall back to spec.
	e.Ctx.VersionContext = upgrade.NewVersionContext()
	assert.Equal(t, "1.28.0", e.resolveNginxUpgradeVersion())

	// No VersionContext: fall back to spec.
	e.Ctx.VersionContext = nil
	assert.Equal(t, "1.28.0", e.resolveNginxUpgradeVersion())

	// No VersionContext and nil ClusterConfig.
	e.Ctx.BKECluster.Spec.ClusterConfig = nil
	assert.Empty(t, e.resolveNginxUpgradeVersion())

	// nil BKECluster.
	e.Ctx.BKECluster = nil
	assert.Empty(t, e.resolveNginxUpgradeVersion())
}

func TestEnsureNginxUpgrade_NeedExecute(t *testing.T) {
	t.Run("version differs triggers upgrade", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = upgrade.NewVersionContext()
		e.Ctx.VersionContext.SetCurrent(upgrade.ComponentNginxProxy, "1.27.0")
		e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.29.0")
		assert.True(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})

	t.Run("same version skips upgrade", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = upgrade.NewVersionContext()
		e.Ctx.VersionContext.SetCurrent(upgrade.ComponentNginxProxy, "1.28.0")
		e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.28.0")
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})

	t.Run("no version context returns false", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = nil
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})

	t.Run("ha cluster never upgrades even with pending target", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = upgrade.NewVersionContext()
		e.Ctx.VersionContext.SetCurrent(upgrade.ComponentNginxProxy, "1.27.0")
		e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.29.0")
		e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.LoadBalanceType = "ha"
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})

	t.Run("empty loadBalanceType defaults to ha and skips", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = upgrade.NewVersionContext()
		e.Ctx.VersionContext.SetCurrent(upgrade.ComponentNginxProxy, "1.27.0")
		e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.29.0")
		e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.LoadBalanceType = ""
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})

	t.Run("nil cluster config skips", func(t *testing.T) {
		e := createTestEnsureNginxUpgrade()
		e.Ctx.VersionContext = upgrade.NewVersionContext()
		e.Ctx.VersionContext.SetCurrent(upgrade.ComponentNginxProxy, "1.27.0")
		e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, "1.29.0")
		e.Ctx.BKECluster.Spec.ClusterConfig = nil
		assert.False(t, e.NeedExecute(&bkev1beta1.BKECluster{}, e.Ctx.BKECluster))
	})
}

func withNginxTarget(e *EnsureNginxUpgrade, version string) {
	e.Ctx.VersionContext = upgrade.NewVersionContext()
	e.Ctx.VersionContext.SetTarget(upgrade.ComponentNginxProxy, version)
}

// patchNginxStatusSync neutralizes status persistence (fake client would loop forever).
func patchNginxStatusSync(patches *gomonkey.Patches) {
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error { return nil })
}

// patchReadyWorkerFetcher makes NodeFetcher return a single ready, non-deleting worker.
func patchReadyWorkerFetcher(patches *gomonkey.Patches) {
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
			return bkenode.Nodes{{IP: "10.0.0.5", Hostname: "worker-1", Role: []string{bkenode.WorkerNodeRole}}}, nil
		})
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlagForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, flag int) (bool, error) {
			if flag == bkev1beta1.NodeDeletingFlag {
				return false, nil
			}
			return true, nil
		})
}

// patchCustomCommand intercepts the BaseCommand internals that Custom.New/Wait delegate to.
// Patching these (rather than the small, inlined Custom.New/Wait wrappers) keeps the stubs
// reliable across Go versions and avoids touching a real client.
func patchCustomCommand(patches *gomonkey.Patches, newErr, waitErr error, successNodes, failedNodes []string) {
	patches.ApplyPrivateMethod(reflect.TypeOf(&command.BaseCommand{}), "newCommand",
		func(_ *command.BaseCommand, _ string, _ string, _ *agentv1beta1.CommandSpec, _ ...string) error {
			return newErr
		})
	patches.ApplyPrivateMethod(reflect.TypeOf(&command.BaseCommand{}), "waitCommandComplete",
		func(_ *command.BaseCommand) (error, bool, command.CommandNodes) {
			return waitErr, true, command.CommandNodes{SuccessNodes: successNodes, FailedNodes: failedNodes}
		})
}

func TestEnsureNginxUpgrade_Execute_NoVersion(t *testing.T) {
	e := createTestEnsureNginxUpgrade()
	e.Ctx.VersionContext = upgrade.NewVersionContext() // present but no nginx target
	e.Ctx.BKECluster.Spec.ClusterConfig.Cluster.NginxProxyVersion = ""

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nginx proxy upgrade version not specified")
}

// All workers are agent-not-ready: the upgrade must be blocked instead of
// silently marked complete, otherwise NeedExecute would never fire again and
// the workers would stay on the old image forever.
func TestEnsureNginxUpgrade_Execute_AllWorkersNotReady(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchNginxStatusSync(patches)
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
			return bkenode.Nodes{{IP: "10.0.0.5", Role: []string{bkenode.WorkerNodeRole}}}, nil
		})
	// worker not deleting (eligible) but agent not ready
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlagForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, _ string, _ int) (bool, error) {
			return false, nil
		})

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nginx upgrade blocked")
	assert.Equal(t, "1.27.0", e.Ctx.BKECluster.Status.NginxProxyVersion,
		"status must not advance while any eligible worker is not upgraded")
}

// A cluster without worker nodes has no nginx-proxy to upgrade: the version
// is recorded directly.
func TestEnsureNginxUpgrade_Execute_NoWorkers(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchNginxStatusSync(patches)
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
			return bkenode.Nodes{{IP: "10.0.0.1", Role: []string{bkenode.MasterNodeRole}}}, nil
		})

	_, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, "1.28.0", e.Ctx.BKECluster.Status.NginxProxyVersion)
}

// Ready workers are upgraded while a not-ready worker is skipped this round:
// the dispatch succeeds but the phase must keep failing (so it retries) and
// the status must stay on the old version until the skipped worker catches up.
func TestEnsureNginxUpgrade_Execute_PartialWorkersNotReady(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchNginxStatusSync(patches)
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
			return bkenode.Nodes{
				{IP: "10.0.0.5", Role: []string{bkenode.WorkerNodeRole}},
				{IP: "10.0.0.6", Role: []string{bkenode.WorkerNodeRole}},
			}, nil
		})
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodeStateFlagForCluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster, ip string, flag int) (bool, error) {
			if flag == bkev1beta1.NodeDeletingFlag {
				return false, nil
			}
			return ip == "10.0.0.5", nil // only worker-1 agent ready
		})
	patchCustomCommand(patches, nil, nil, []string{"10.0.0.5"}, nil)

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nginx upgrade incomplete")
	assert.Equal(t, "1.27.0", e.Ctx.BKECluster.Status.NginxProxyVersion,
		"status must not advance while any eligible worker is not upgraded")
}

func TestEnsureNginxUpgrade_Execute_Success(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchNginxStatusSync(patches)
	patchReadyWorkerFetcher(patches)
	patchCustomCommand(patches, nil, nil, []string{"10.0.0.5"}, nil)

	_, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, "1.28.0", e.Ctx.BKECluster.Status.NginxProxyVersion)
}

func TestEnsureNginxUpgrade_Execute_GetNodesError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patches.ApplyFunc((*nodeutil.NodeFetcher).GetNodesForBKECluster,
		func(_ *nodeutil.NodeFetcher, _ context.Context, _ *bkev1beta1.BKECluster) (bkenode.Nodes, error) {
			return nil, assert.AnError
		})

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to get nodes for cluster")
}

func TestEnsureNginxUpgrade_Execute_CommandNewError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchReadyWorkerFetcher(patches)
	patchCustomCommand(patches, assert.AnError, nil, nil, nil) // newCommand fails

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to create nginx upgrade command")
}

func TestEnsureNginxUpgrade_Execute_WaitError(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchReadyWorkerFetcher(patches)
	patchCustomCommand(patches, nil, assert.AnError, nil, nil) // newCommand ok, wait fails

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "failed to upgrade nginx")
}

func TestEnsureNginxUpgrade_Execute_FailedNodes(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchReadyWorkerFetcher(patches)
	patchCustomCommand(patches, nil, nil, nil, []string{"10.0.0.5"}) // wait reports failed node

	_, err := e.Execute()
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "nginx upgrade failed on nodes")
}

func TestEnsureNginxUpgrade_Execute_SyncStatusError_NoReady(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	// real NodeFetcher returns no nodes against the fake client -> no ready workers
	// -> status sync path, which fails.
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return assert.AnError
		})

	_, err := e.Execute()
	assert.Error(t, err)
}

func TestEnsureNginxUpgrade_Execute_SyncStatusError_AfterUpgrade(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	e := createTestEnsureNginxUpgrade()
	withNginxTarget(e, "1.28.0")
	patchReadyWorkerFetcher(patches)
	patchCustomCommand(patches, nil, nil, []string{"10.0.0.5"}, nil)
	patches.ApplyFunc(mergecluster.SyncStatusUntilComplete,
		func(_ client.Client, _ *bkev1beta1.BKECluster, _ ...mergecluster.PatchFunc) error {
			return assert.AnError
		})

	_, err := e.Execute()
	assert.Error(t, err)
}
