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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	confv1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/bkecommon/v1beta1"
	bkev1beta1 "gopkg.openfuyao.cn/cluster-api-provider-bke/api/capbke/v1beta1"
	bkeinit "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/initialize"
	bkenode "gopkg.openfuyao.cn/cluster-api-provider-bke/common/cluster/node"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/phaseframe/phaseutil/agentssh"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/pkg/upgrade"
	"gopkg.openfuyao.cn/cluster-api-provider-bke/utils/capbke/nodeutil"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
)

func TestEnsureAgentUpgradeConstants(t *testing.T) {
	assert.Equal(t, "EnsureAgentUpgrade", string(EnsureAgentUpgradeName))
	assert.Equal(t, "bkeagent", bkeagentAddonName)
}

func TestNewEnsureAgentUpgrade(t *testing.T) {
	InitinitPhaseContextFun()
	phase := NewEnsureAgentUpgrade(initPhaseContext)
	assert.NotNil(t, phase)
	assert.IsType(t, &EnsureAgentUpgrade{}, phase)
}

func TestEnsureAgentUpgrade_Version(t *testing.T) {
	InitinitPhaseContextFun()
	e := NewEnsureAgentUpgrade(initPhaseContext).(*EnsureAgentUpgrade)

	initPhaseContext.BKECluster = &bkev1beta1.BKECluster{
		Status: confv1beta1.BKEClusterStatus{
			AddonStatus: []confv1beta1.Product{{Name: bkeagentAddonName, Version: "v1.2.3"}},
		},
	}
	assert.Equal(t, "v1.2.3", e.Version())

	initPhaseContext.BKECluster = &bkev1beta1.BKECluster{}
	assert.Equal(t, "", e.Version())
}

func TestEnsureAgentUpgrade_NeedExecute(t *testing.T) {
	InitinitPhaseContextFun()
	e := NewEnsureAgentUpgrade(initPhaseContext).(*EnsureAgentUpgrade)

	vc := upgrade.NewVersionContext()
	vc.SetCurrent(upgrade.ComponentBKEAgent, "v1.0.0")
	vc.SetTarget(upgrade.ComponentBKEAgent, "v1.0.1")
	initPhaseContext.SetVersionContext(vc)

	oldCluster := &bkev1beta1.BKECluster{}
	newCluster := oldCluster.DeepCopy()
	assert.True(t, e.NeedExecute(oldCluster, newCluster))

	vc.SetCurrent(upgrade.ComponentBKEAgent, "v1.0.1")
	assert.False(t, e.NeedExecute(oldCluster, newCluster))
}

func TestBKEAgentArtifactName(t *testing.T) {
	cfg := bkeinit.BkeConfig{}
	assert.Equal(t, agentssh.DefaultBKEAgentArtifact, agentssh.BinaryArtifactName(cfg, ""))
	assert.Equal(t, "bkeagent-2.1.0-linux-{.arch}", agentssh.BinaryArtifactName(cfg, "v2.1.0"))

	cfg.CustomExtra = map[string]string{"bkeagent": "custom-bkeagent-{.arch}"}
	assert.Equal(t, "custom-bkeagent-{.arch}", agentssh.BinaryArtifactName(cfg, "v9.9.9"))
}

func TestEnsureAgentUpgrade_AgentTargetVersion(t *testing.T) {
	InitinitPhaseContextFun()
	e := NewEnsureAgentUpgrade(initPhaseContext).(*EnsureAgentUpgrade)

	vc := upgrade.NewVersionContext()
	vc.SetTarget(upgrade.ComponentBKEAgent, "v2.1.0")
	initPhaseContext.SetVersionContext(vc)
	assert.Equal(t, "v2.1.0", e.agentTargetVersion())

	vc = upgrade.NewVersionContext()
	vc.SetTarget(legacyReleaseBKEAgentComponent, "v2.2.0")
	initPhaseContext.SetVersionContext(vc)
	assert.Equal(t, "v2.2.0", e.agentTargetVersion())
}

func TestAgentSSHParamsFromCluster(t *testing.T) {
	InitinitPhaseContextFun()
	cluster := initNewBkeCluster.DeepCopy()
	cluster.Spec.ClusterConfig.Cluster.HTTPRepo = confv1beta1.Repo{
		Domain: "repo.example.com", Port: "8080", Prefix: "files",
	}

	params := agentssh.ParamsFromCluster(cluster, "")
	assert.Equal(t, "http://repo.example.com:8080/files", params.BaseURL)
	assert.Equal(t, agentssh.DefaultBKEAgentArtifact, params.BinaryArtifact)

	params = agentssh.ParamsFromCluster(cluster, "v2.1.0")
	assert.Equal(t, "bkeagent-2.1.0-linux-{.arch}", params.BinaryArtifact)
	assert.Equal(t, "http://repo.example.com:8080/files/bkeagent-2.1.0-linux-amd64",
		agentssh.BinaryURLForArch(params, "amd64"))

	cluster.Spec.ClusterConfig.CustomExtra = map[string]string{"bkeagent": "custom-bkeagent-{.arch}"}
	params = agentssh.ParamsFromCluster(cluster, "v9.9.9")
	assert.Equal(t, "custom-bkeagent-{.arch}", params.BinaryArtifact)
}

func TestEnsureAgentUpgrade_Execute_SSHUpgrade(t *testing.T) {
	patches := gomonkey.NewPatches()
	defer patches.Reset()

	InitinitPhaseContextFun()
	e := NewEnsureAgentUpgrade(initPhaseContext).(*EnsureAgentUpgrade)
	patches.ApplyPrivateMethod(e, "upgradeBKEAgentViaSSH", func(_ *EnsureAgentUpgrade) error {
		return nil
	})

	result, err := e.Execute()
	assert.NoError(t, err)
	assert.Equal(t, ctrl.Result{}, result)
}

func newAgentUpgradePhaseCov(t *testing.T, bkeCluster *bkev1beta1.BKECluster) *EnsureAgentUpgrade {
	t.Helper()
	scheme := runtime.NewScheme()
	require.NoError(t, bkev1beta1.AddToScheme(scheme))
	ctx := &phaseframe.PhaseContext{
		Context:    context.Background(),
		BKECluster: bkeCluster,
		Scheme:     scheme,
		Log:        bkev1beta1.NewBKELogger(nil, &fakeRecorder{}, bkeCluster),
	}
	return &EnsureAgentUpgrade{BasePhase: phaseframe.BasePhase{Ctx: ctx}}
}

func bkeNodesFor(ip, role string) bkev1beta1.BKENodes {
	return bkev1beta1.BKENodes{{Spec: confv1beta1.BKENodeSpec{IP: ip, Role: []string{role}}}}
}

// ---- upgradeFailure (pure) ----

func TestEnsureAgentUpgradeUpgradeFailure(t *testing.T) {
	e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})

	t.Run("failure on master node", func(t *testing.T) {
		nodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{"master"}}}
		failures := map[string]error{"192.168.1.1": assertErr("ssh fail")}
		err := e.upgradeFailure(nodes, failures, 0, 1)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "failed on master node(s)")
	})

	t.Run("failure on worker node", func(t *testing.T) {
		nodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{"master"}}, {IP: "192.168.1.2", Role: []string{"worker"}}}
		failures := map[string]error{"192.168.1.2": assertErr("ssh fail")}
		err := e.upgradeFailure(nodes, failures, 1, 2)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "bkeagent ssh upgrade failed")
		assert.NotContains(t, err.Error(), "master node(s)")
	})

	t.Run("empty failures", func(t *testing.T) {
		nodes := bkenode.Nodes{{IP: "192.168.1.1", Role: []string{"master"}}}
		err := e.upgradeFailure(nodes, map[string]error{}, 1, 1)
		require.Error(t, err)
	})
}

// ---- upgradeBKEAgentViaSSH (real body via seams) ----

func TestEnsureAgentUpgradeUpgradeBKEAgentViaSSH(t *testing.T) {
	t.Run("get bke nodes error returns nil", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return nil, assertErr("fetch failed")
		})
		require.NoError(t, e.upgradeBKEAgentViaSSH())
	})

	t.Run("no nodes returns error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkev1beta1.BKENodes{}, nil
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "no cluster nodes")
	})

	t.Run("discover archs error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "worker"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return nil, nil, assertErr("discover failed")
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "discover failed")
	})

	t.Run("discover errs triggers upgrade failure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "master"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{}, map[string]error{"192.168.1.1": assertErr("arch fail")}, nil
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master node(s)")
	})

	t.Run("prepare staging error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "worker"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{"amd64": {}}, nil, nil
		})
		patches.ApplyFunc(agentssh.PrepareStaging, func(_ *bkev1beta1.BKECluster, _ agentssh.ArtifactParams, _ []string) (*agentssh.Staging, error) {
			return nil, assertErr("staging failed")
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "prepare upgrade artifacts")
	})

	t.Run("ssh upgrade error", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "worker"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{"amd64": {}}, nil, nil
		})
		patches.ApplyFunc(agentssh.PrepareStaging, func(_ *bkev1beta1.BKECluster, _ agentssh.ArtifactParams, _ []string) (*agentssh.Staging, error) {
			return &agentssh.Staging{}, nil
		})
		patches.ApplyFunc(agentssh.SSHUpgrade, func(_ context.Context, _ bkenode.Nodes, _ *agentssh.Staging, _ interface{}) (map[string]error, error) {
			return nil, assertErr("ssh failed")
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "ssh failed")
	})

	t.Run("ssh push errors triggers upgrade failure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "master"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{"amd64": {}}, nil, nil
		})
		patches.ApplyFunc(agentssh.PrepareStaging, func(_ *bkev1beta1.BKECluster, _ agentssh.ArtifactParams, _ []string) (*agentssh.Staging, error) {
			return &agentssh.Staging{}, nil
		})
		patches.ApplyFunc(agentssh.SSHUpgrade, func(_ context.Context, _ bkenode.Nodes, _ *agentssh.Staging, _ interface{}) (map[string]error, error) {
			return map[string]error{"192.168.1.1": assertErr("push fail")}, nil
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "master node(s)")
	})

	t.Run("ping failure", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "worker"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{"amd64": {}}, nil, nil
		})
		patches.ApplyFunc(agentssh.PrepareStaging, func(_ *bkev1beta1.BKECluster, _ agentssh.ArtifactParams, _ []string) (*agentssh.Staging, error) {
			return &agentssh.Staging{}, nil
		})
		patches.ApplyFunc(agentssh.SSHUpgrade, func(_ context.Context, _ bkenode.Nodes, _ *agentssh.Staging, _ interface{}) (map[string]error, error) {
			return nil, nil
		})
		patches.ApplyFunc(phaseutil.PingBKEAgentOnNodes, func(_ context.Context, _ interface{}, _ *runtime.Scheme, _ *bkev1beta1.BKECluster, _ bkenode.Nodes) (error, []string, []string) {
			return assertErr("ping failed"), nil, []string{"192.168.1.1"}
		})
		err := e.upgradeBKEAgentViaSSH()
		require.Error(t, err)
		assert.Contains(t, err.Error(), "ping after upgrade failed")
	})

	t.Run("success", func(t *testing.T) {
		patches := gomonkey.NewPatches()
		defer patches.Reset()
		e := newAgentUpgradePhaseCov(t, &bkev1beta1.BKECluster{ObjectMeta: metav1.ObjectMeta{Name: "c1", Namespace: "ns"}})
		patches.ApplyFunc((*nodeutil.NodeFetcher).GetBKENodesWrapper, func(_ *nodeutil.NodeFetcher, _ context.Context, _, _ string) (bkev1beta1.BKENodes, error) {
			return bkeNodesFor("192.168.1.1", "worker"), nil
		})
		patches.ApplyFunc(agentssh.DiscoverArchs, func(_ context.Context, _ bkenode.Nodes, _ interface{}) (map[string]struct{}, map[string]error, error) {
			return map[string]struct{}{"amd64": {}}, nil, nil
		})
		patches.ApplyFunc(agentssh.PrepareStaging, func(_ *bkev1beta1.BKECluster, _ agentssh.ArtifactParams, _ []string) (*agentssh.Staging, error) {
			return &agentssh.Staging{}, nil
		})
		patches.ApplyFunc(agentssh.SSHUpgrade, func(_ context.Context, _ bkenode.Nodes, _ *agentssh.Staging, _ interface{}) (map[string]error, error) {
			return nil, nil
		})
		patches.ApplyFunc(phaseutil.PingBKEAgentOnNodes, func(_ context.Context, _ interface{}, _ *runtime.Scheme, _ *bkev1beta1.BKECluster, _ bkenode.Nodes) (error, []string, []string) {
			return nil, nil, nil
		})
		require.NoError(t, e.upgradeBKEAgentViaSSH())
	})
}
